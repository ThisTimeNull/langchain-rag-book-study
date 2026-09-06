"""
Q0012 · LLM 추출이 이득인 경우는 언제인가?

Q0011에서 규칙 적재가 LLM 추출을 이겼다. 하지만 그 데이터는 문장 틀이 정해진 정형이었다.
"- A는 B에 의존한다." 같은 문장만 나오니 정규식 다섯 줄이면 충분했다.

이번에는 조건을 뒤집는다. 사람이 자유롭게 쓴 글로 같은 일을 시킨다.

  data/prose_sample/
    weekly_meeting.md   주간 회의 기록
    incident_review.md  장애 회고
    handover.md         인수인계 메모

같은 사실이 문서마다 다른 말로 적혀 있다.

  "결제 API가 인증 서비스를 매번 두드리는데"        (회의록)
  "인증을 기다리던 결제 API가 줄줄이 밀렸다"        (회고)
  "로그인 확인 때문에 인증 서비스에 얹혀 있고"       (인수인계)

셋 다 "결제 API는 인증 서비스에 의존한다"는 같은 사실이다. 정규식으로는 잡을 수 없다.

실행: python code/Q0012_prose_extraction.py
      python code/Q0012_prose_extraction.py --rebuild
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import LLMGraphTransformer, GraphCypherQAChain

from config.settings import get_azure_llm, get_azure_embeddings, get_neo4j_graph, DATA_DIR

DOC_DIR = DATA_DIR / "prose_sample"
WIPE = "MATCH (n) WHERE n:__Entity__ OR n:Document"

# Q0011에서 썼던 규칙 그대로. 정형 문서에서는 24개를 100% 잡아냈다.
RULES = [
    (r"- (.+?)은 (.+?)의 팀장이다\.", "팀장"),
    (r"- (.+?)은 (.+?) 소속이다\.", "소속"),
    (r"- (.+?)이 (.+?)를 운영한다\.", "운영"),
    (r"- (.+?)는 (.+?)에 의존한다\.", "의존"),
    (r"- (.+?)은 (.+?)의 온콜을 맡는다\.", "온콜"),
]

# 사람이 문서를 읽고 정리한 정답. 이걸 기준으로 추출 성적을 매긴다.
GOLD = {
    ("결제 API", "의존", "인증 서비스"),
    ("인증 서비스", "의존", "사용자 DB"),
    ("추천 엔진", "의존", "결제 API"),
    ("알림 서비스", "의존", "결제 API"),
    ("인프라팀", "운영", "인증 서비스"),
    ("인프라팀", "운영", "알림 서비스"),
    ("정하늘", "팀장", "인프라팀"),
    ("이민수", "팀장", "결제팀"),
    ("김철수", "소속", "인프라팀"),
    ("한서준", "소속", "인프라팀"),
    ("박영희", "소속", "결제팀"),
}

QUESTIONS = [
    ("결제 API가 의존하는 서비스는 무엇인가요?", "인증 서비스"),
    ("인증 서비스를 운영하는 팀은 어디인가요?", "인프라팀"),
    ("결제 API가 멈추면 함께 영향을 받는 서비스는 무엇인가요?", "추천 엔진, 알림 서비스"),
]


def section(n, title):
    print(f"\n{'=' * 68}\n[{n}] {title}\n{'=' * 68}")


def load_docs():
    return DirectoryLoader(
        str(DOC_DIR), glob="*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    ).load()


# --- 1) 규칙으로 해보기 -------------------------------------------------------
def try_rules(docs):
    section(1, "먼저 규칙(정규식)으로 뽑아본다")

    text = "\n".join(d.page_content for d in docs)
    total = 0
    for pattern, rel in RULES:
        found = re.findall(pattern, text)
        print(f"  {rel:4s} 규칙 -> {len(found)}개")
        total += len(found)

    print(f"\n  합계 {total}개. 정답은 {len(GOLD)}개다.")
    print("\n  왜 하나도 못 잡나. 같은 사실이 문서마다 다른 말로 적혀 있기 때문이다.")
    print("    '결제 API가 인증 서비스를 매번 두드리는데'")
    print("    '인증을 기다리던 결제 API가 줄줄이 밀렸다'")
    print("    '로그인 확인 때문에 인증 서비스에 얹혀 있고'")
    print("  셋 다 같은 의존 관계다. 표현을 미리 다 적어둘 방법이 없다.")
    return total


# --- 2) LLM으로 뽑기 ---------------------------------------------------------
def build_with_llm(llm, rebuild):
    section(2, "LLM으로 뽑아본다")

    graph = get_neo4j_graph()
    n = graph.query(f"{WIPE} RETURN count(n) AS c")[0]["c"]
    if n and not rebuild:
        print(f"  기존 그래프 재사용: 노드 {n}개")
        graph.refresh_schema()
        return graph
    if n:
        graph.query(f"{WIPE} DETACH DELETE n")

    docs = load_docs()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=350, chunk_overlap=80, separators=["\n## ", "\n\n", "\n", " ", ""]
    ).split_documents(docs)
    print(f"  문서 {len(docs)}개, 조각 {len(chunks)}개")

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["사람", "팀", "서비스"],
        allowed_relationships=["소속", "팀장", "운영", "의존", "온콜"],
    )
    graph_docs = transformer.convert_to_graph_documents(chunks)
    graph.add_graph_documents(graph_docs, include_source=True, baseEntityLabel=True)

    # Q0011에서 배운 것. 이름을 원문 표기로 되돌리지 않으면 질의가 통째로 실패한다.
    align_ids_to_source(graph, docs)
    graph.refresh_schema()
    return graph


def align_ids_to_source(graph, docs):
    text = "\n".join(d.page_content for d in docs)
    fixed = []
    for row in graph.query("MATCH (n:__Entity__) RETURN n.id AS id"):
        nid = row["id"]
        if not nid or nid in text:
            continue
        m = re.search(re.escape(nid), text, re.IGNORECASE)
        if m and m.group(0) != nid:
            graph.query("MATCH (n:__Entity__ {id:$old}) SET n.id = $new",
                        {"old": nid, "new": m.group(0)})
            fixed.append((nid, m.group(0)))
    if fixed:
        print("  이름 정합: " + ", ".join(f"{a}->{b}" for a, b in fixed[:5]))
    return fixed


def score(graph):
    section(3, "채점: 사람이 읽고 정리한 정답과 맞춰본다")

    got = {(r["s"], r["t"], r["o"]) for r in graph.query("""
        MATCH (a:__Entity__)-[r]->(b:__Entity__)
        RETURN a.id AS s, type(r) AS t, b.id AS o
    """)}
    hit = GOLD & got
    print(f"  정답 {len(GOLD)}개 중 {len(hit)}개를 맞혔다. (재현율 {len(hit)/len(GOLD):.0%})")

    print("\n  맞힌 것:")
    for s, t, o in sorted(hit):
        print(f"    ({s}) -[{t}]-> ({o})")
    miss = GOLD - got
    if miss:
        print("\n  놓친 것:")
        for s, t, o in sorted(miss):
            print(f"    ({s}) -[{t}]-> ({o})")

    extra = got - GOLD
    print(f"\n  정답에 없는데 뽑은 것 {len(extra)}개 (전부 오류는 아니다. 문서에 있는 다른 사실일 수 있다):")
    for s, t, o in sorted(extra)[:6]:
        print(f"    ({s}) -[{t}]-> ({o})")
    return len(hit)


def compare(llm, graph):
    section(4, "질문 비교")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=60, separators=["\n## ", "\n\n", "\n", " ", ""]
    ).split_documents(load_docs())
    k = 3
    retriever = FAISS.from_documents(chunks, get_azure_embeddings()).as_retriever(
        search_kwargs={"k": k})
    print(f"  (벡터는 조각 {len(chunks)}개 중 {k}개를 봅니다. 코퍼스의 {k/len(chunks):.0%})")

    prompt = ChatPromptTemplate.from_template(
        "아래 근거만 써서 한국어로 짧게 답하세요. 근거에 없으면 '문서에서 찾을 수 없습니다'라고 답하세요.\n"
        "[근거]\n{context}\n\n[질문] {question}\n[답변]"
    )
    chain = GraphCypherQAChain.from_llm(
        llm, graph=graph, verbose=False,
        allow_dangerous_requests=True, return_intermediate_steps=True)

    for i, (q, gold) in enumerate(QUESTIONS, 1):
        print(f"\n{'-' * 68}\n질문 {i}: {q}\n  정답: {gold}\n{'-' * 68}")
        hits = retriever.invoke(q)
        ctx = "\n\n".join(f"[{Path(d.metadata['source']).name}] {d.page_content}" for d in hits)
        v = (prompt | llm | StrOutputParser()).invoke({"context": ctx, "question": q})
        print(f"  [Vector] {v.strip()[:150]}")
        try:
            out = chain.invoke({"query": q})
            print(f"  [Graph ] {str(out.get('result', '')).strip()[:150]}")
        except Exception as e:
            print(f"  [Graph ] 실패 {type(e).__name__}")


def main():
    rebuild = "--rebuild" in sys.argv
    print("Q0012 · 사람이 자유롭게 쓴 글에서 그래프를 만든다")

    docs = load_docs()
    rule_hits = try_rules(docs)

    llm = get_azure_llm()
    graph = build_with_llm(llm, rebuild)
    llm_hits = score(graph)
    compare(llm, graph)

    print(f"\n{'=' * 68}")
    print("정리")
    print(f"  같은 문서, 같은 정답 {len(GOLD)}개 기준")
    print(f"    규칙(정규식) : {rule_hits}개")
    print(f"    LLM 추출     : {llm_hits}개")
    print("  Q0011의 정형 문서에서는 정반대였다. 규칙이 100%, LLM이 67%였다.")
    print("  문장 틀이 있으면 규칙을 쓰고, 없으면 LLM을 쓴다. 그게 갈림길이다.")


if __name__ == "__main__":
    main()
