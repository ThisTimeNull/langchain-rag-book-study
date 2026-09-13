"""
Q0013 · 하이브리드 검색은 어떻게 하나?

벡터와 그래프는 각자 못 하는 게 있다.

  벡터: 긴 사슬을 따라가지 못한다. 조각 k개로는 사슬 전체를 못 모은다.
  그래프: 그래프에 없는 정보는 답하지 못한다. 서술형 설명은 노드로 안 들어간다.

하이브리드는 둘을 잇는다.

  질문 -> 벡터 검색으로 관련 조각을 찾는다        (진입점 찾기)
       -> 그 조각에 등장하는 엔티티를 그래프에서 찾는다  (연결)
       -> 그 엔티티 주변을 그래프로 넓힌다          (확장)
       -> 텍스트 근거와 그래프 근거를 함께 LLM에 준다

코퍼스: data/hybrid_sample/
  구조화된 문서 4개 (팀, 서비스, 의존, 온콜)  + 서술형 메모 1개 (notes.md)
  notes.md의 내용은 그래프에 안 들어간다. 문장 틀이 없어서 규칙이 못 읽기 때문이다.

실행: python code/Q0013_hybrid_search.py
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
from langchain_neo4j import GraphCypherQAChain

from config.settings import get_azure_llm, get_azure_embeddings, get_neo4j_graph, DATA_DIR

DOC_DIR = DATA_DIR / "hybrid_sample"
WIPE = "MATCH (n) WHERE n:__Entity__ OR n:Document"

# Q0011에서 확인한 대로, 문장 틀이 있는 문서는 규칙으로 읽는 게 정확하다.
RULES = [
    (r"- (.+?)은 (.+?)의 팀장이다\.", "사람", "팀", "팀장"),
    (r"- (.+?)은 (.+?) 소속이다\.", "사람", "팀", "소속"),
    (r"- (.+?)이 (.+?)를 운영한다\.", "팀", "서비스", "운영"),
    (r"- (.+?)는 (.+?)에 의존한다\.", "서비스", "서비스", "의존"),
    (r"- (.+?)은 (.+?)의 온콜을 맡는다\.", "사람", "팀", "온콜"),
]

QUESTIONS = [
    ("인증 서비스의 세션 만료 시간은 얼마인가요?", "30분",
     "서술형 메모에만 있음. 그래프에 없는 정보"),
    ("감사 로그가 직접 또는 간접으로 의존하는 서비스는 모두 몇 개인가요?", "17개",
     "긴 사슬. 조각 몇 개로는 못 셈"),
    ("결제 API가 의존하는 서비스를 운영하는 팀의 온콜 담당자는 누구인가요?", "김철수",
     "3다리. 문서 세 개를 넘나듦"),
]


def section(n, title):
    print(f"\n{'=' * 70}\n[{n}] {title}\n{'=' * 70}")


def load_docs():
    return DirectoryLoader(
        str(DOC_DIR), glob="*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    ).load()


def build_graph():
    graph = get_neo4j_graph()
    graph.query(f"{WIPE} DETACH DELETE n")
    text = "\n".join(d.page_content for d in load_docs())
    total = 0
    for pattern, la, lb, rel in RULES:
        for a, b in re.findall(pattern, text):
            graph.query(
                f"MERGE (a:__Entity__:{la} {{id:$a}}) MERGE (b:__Entity__:{lb} {{id:$b}}) "
                f"MERGE (a)-[:{rel}]->(b)",
                {"a": a.strip(), "b": b.strip()},
            )
            total += 1
    graph.refresh_schema()
    return graph, total


def build_retriever(k):
    # 조각을 작게 잘라 코퍼스를 넉넉히 만든다.
    # 조각이 몇 개 안 되면 벡터가 사실상 전체를 다 보게 되어 비교가 무의미해진다.
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=120, chunk_overlap=20, separators=["\n- ", "\n\n", "\n", " ", ""]
    ).split_documents(load_docs())
    store = FAISS.from_documents(chunks, get_azure_embeddings())
    return store.as_retriever(search_kwargs={"k": k}), len(chunks)


# --- 하이브리드의 핵심 두 단계 -------------------------------------------------
def link_entities(graph, question, texts, limit=4):
    """벡터가 찾아온 텍스트에서 그래프 노드 이름을 찾는다. 이게 '진입점'이다.

    중요한 건 '좁히는 것'이다. 처음에는 검색 결과에 나온 이름을 전부 진입점으로 삼았더니
    19개가 잡혔고, 그래프 사실 145개가 쏟아져 LLM이 노이즈에 묻혔다. 답이 틀렸다.

    그래서 순서를 준다.
      1순위: 질문에 직접 등장한 이름 (사용자가 콕 집은 대상)
      2순위: 검색 상위 조각에 나온 이름
    그리고 상위 몇 개만 쓴다. 이 limit이 하이브리드의 핵심 손잡이다.
    """
    names = [r["id"] for r in graph.query("MATCH (n:__Entity__) RETURN n.id AS id")]
    in_question = [n for n in names if n in question]
    top_text = "\n".join(texts[:2])
    in_text = [n for n in names if n in top_text and n not in in_question]
    return (in_question + in_text)[:limit]


def expand(graph, entries):
    """진입점 주변을 그래프로 넓힌다.

    어디까지 넓힐지는 설계 결정이다. 무작정 넓히면 관계없는 사실이 쏟아진다.
    여기서는 두 가지를 쓴다.
      1. 진입점의 1홉 이웃 (누가 어느 팀인지, 어느 팀이 뭘 운영하는지)
      2. 의존 관계만 사슬 전체 (긴 사슬 질문에 답하려면 1홉으로는 부족하다)
    """
    facts = []
    for e in entries:
        # 1홉이 아니라 2홉을 가져온다. 1홉만 주면 LLM이 사실 조각을 직접 이어 붙여야 하는데
        # 그걸 잘 못한다. 경로를 통째로 보여주면 훨씬 안정적으로 답한다.
        for r in graph.query(
            "MATCH p = (a:__Entity__ {id:$id})-[*1..2]-(b:__Entity__) "
            "WITH p LIMIT 40 "
            "RETURN [n IN nodes(p) | n.id] AS ids, [x IN relationships(p) | type(x)] AS rels",
            {"id": e},
        ):
            ids, rels = r["ids"], r["rels"]
            parts = [ids[0]]
            for rel, nxt in zip(rels, ids[1:]):
                parts.append(f" -[{rel}]- {nxt}")
            facts.append("".join(parts))
        for r in graph.query(
            "MATCH (a:서비스 {id:$id})-[:의존*1..]->(b:서비스) RETURN DISTINCT b.id AS o",
            {"id": e},
        ):
            facts.append(f"({e}) 는 ({r['o']}) 에 직접 또는 간접으로 의존")
    return sorted(set(facts))


ROUTE_PROMPT = ChatPromptTemplate.from_template(
    "질문을 읽고 어느 저장소로 보낼지 한 단어로만 답하세요.\n"
    "graph: 관계를 따라가거나(누가 무엇에 의존/소속/운영), 개수를 세거나, 경로를 묻는 질문\n"
    "vector: 설명, 정책, 수치, 배경처럼 문장으로 서술된 내용을 묻는 질문\n\n"
    "질문: {question}\n답(graph 또는 vector):"
)


def route(llm, question):
    """질문을 보고 어디로 보낼지 정한다.

    근거를 합치는 방식은 정보가 많아질수록 오히려 나빠졌다. 그래서 실무 하이브리드는
    합치기보다 '나누기'를 쓴다. 질문 유형을 먼저 판별해 맞는 저장소 하나로 보낸다.
    집계나 경로는 Cypher가 DB 안에서 정확히 계산하고, LLM은 그 결과만 읽으면 된다.
    """
    r = (ROUTE_PROMPT | llm | StrOutputParser()).invoke({"question": question})
    return "graph" if "graph" in r.strip().lower() else "vector"


def main():
    print("Q0013 · 벡터로 진입하고 그래프로 넓힌다")

    llm = get_azure_llm()
    graph, n_rel = build_graph()
    k = 4
    retriever, n_chunks = build_retriever(k)

    section(1, "준비")
    print(f"  그래프: 규칙으로 관계 {n_rel}개 적재 (정확하고 매번 같음)")
    print(f"  벡터  : 조각 {n_chunks}개 중 {k}개를 검색 (코퍼스의 {k/n_chunks:.0%})")
    print(f"  코퍼스: 구조화 문서 4개 + 서술형 메모 1개")

    answer_prompt = ChatPromptTemplate.from_template(
        "아래 근거만 써서 한국어로 짧게 답하세요. 근거에 없으면 '문서에서 찾을 수 없습니다'라고 답하세요.\n"
        "[근거]\n{context}\n\n[질문] {question}\n[답변]"
    )
    answer = answer_prompt | llm | StrOutputParser()
    cypher_chain = GraphCypherQAChain.from_llm(
        llm, graph=graph, verbose=False, allow_dangerous_requests=True)

    section(2, "세 방식 비교")
    for i, (q, gold, why) in enumerate(QUESTIONS, 1):
        print(f"\n{'-' * 70}\n질문 {i}: {q}\n  정답: {gold}   ({why})\n{'-' * 70}")

        hits = retriever.invoke(q)
        texts = [d.page_content for d in hits]

        # 1) 벡터만
        v = answer.invoke({"context": "\n\n".join(texts), "question": q})
        print(f"  [벡터만  ] {v.strip()[:130]}")

        # 2) 그래프만
        try:
            g = cypher_chain.invoke({"query": q}).get("result", "")
            print(f"  [그래프만] {str(g).strip()[:130]}")
        except Exception as e:
            print(f"  [그래프만] 실패 {type(e).__name__}")

        # 3) 하이브리드
        entries = link_entities(graph, q, texts)
        facts = expand(graph, entries)
        ctx = ("[문서에서 찾은 내용]\n" + "\n\n".join(texts)
               + "\n\n[그래프에서 넓힌 사실]\n" + "\n".join(facts))
        h = answer.invoke({"context": ctx, "question": q})
        print(f"  [합치기  ] {h.strip()[:130]}")
        print(f"             진입점 {len(entries)}개, 그래프 사실 {len(facts)}개 (많을수록 오히려 헷갈린다)")

        # 4) 라우팅. 합치지 않고 질문 유형에 따라 한쪽으로 보낸다
        dest = route(llm, q)
        if dest == "graph":
            r = str(cypher_chain.invoke({"query": q}).get("result", ""))
        else:
            r = answer.invoke({"context": "\n\n".join(texts), "question": q})
        print(f"  [라우팅  ] {r.strip()[:130]}   (-> {dest})")

    print(f"\n{'=' * 70}")
    print("정리")
    print("  벡터만  : 서술형 정보에 강하고, 긴 사슬에 약하다")
    print("  그래프만: 사슬에 강하고, 그래프에 없는 정보는 아예 모른다")
    print("  합치기  : 근거를 다 모아 LLM에 넘긴다. 정보가 많아질수록 오히려 나빠졌다")
    print("  라우팅  : 질문 유형을 판별해 맞는 쪽으로 보낸다. 가장 안정적이었다")
    print("\n  교훈: 하이브리드는 '섞기'가 아니라 '나누기'에 가깝다.")
    print("        집계와 경로는 DB가 계산하게 두고, LLM에게는 결과만 읽게 한다.")


if __name__ == "__main__":
    main()
