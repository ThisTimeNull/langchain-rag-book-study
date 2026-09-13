"""
Q0016 · 상호참조 해결은 어떻게 하나?

Q0012에서 회의록으로 그래프를 만들었더니 이런 게 나왔다.

  (그쪽) -[의존]-> (우리 쪽)
  (알림 서비스) -[소속]-> (그쪽)

"그쪽", "우리 쪽", "우리 팀", "팀장님", "저"는 사람이 읽으면 누구인지 안다.
앞뒤 문맥이 있으니까. 하지만 조각 단위로 읽는 LLM은 모른다. 그래서 대명사가
그대로 노드가 된다. 이걸 푸는 게 상호참조 해결(coreference resolution)이다.

방법은 단순하다. 추출하기 전에 문서를 한 번 다시 쓴다.

  "인증 서비스는 우리 팀 소관이라"  ->  "인증 서비스는 인프라팀 소관이라"
  "팀장님은 이민수 님이고"          ->  "결제팀 팀장은 이민수이고"

문서 전체를 주고 바꾸게 하므로 제목과 참석자 같은 문맥을 쓸 수 있다.

실행: python code/Q0016_coreference.py
      python code/Q0016_coreference.py --runs 3
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import LLMGraphTransformer

from config.settings import get_azure_llm, get_neo4j_graph, DATA_DIR

DOC_DIR = DATA_DIR / "prose_sample"
WIPE = "MATCH (n) WHERE n:__Entity__ OR n:Document"

# Q0012와 같은 정답
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

# 이런 게 노드로 나오면 상호참조 실패다
PRONOUNS = ["그쪽", "우리 쪽", "우리 팀", "우리", "저", "팀장님", "후임자", "그날", "이쪽", "본인"]

REWRITE = ChatPromptTemplate.from_template(
    "아래 문서를 다시 쓰세요. 규칙:\n"
    "1. 대명사와 지시어(우리 팀, 우리 쪽, 그쪽, 팀장님, 저, 후임자 등)를 문서 안의 정보로\n"
    "   구체적인 이름(사람 이름, 팀 이름, 서비스 이름)으로 바꾸세요.\n"
    "2. 제목, 참석자, 서명 같은 곳에서 누구인지 알 수 있으면 그 이름을 쓰세요.\n"
    "3. 문서에 없는 정보를 지어내지 마세요. 알 수 없으면 그대로 두세요.\n"
    "4. 내용, 순서, 문장 수를 바꾸지 마세요. 대명사만 바꾸세요.\n\n"
    "[문서]\n{text}\n\n[다시 쓴 문서]"
)


def section(n, title):
    print(f"\n{'=' * 70}\n[{n}] {title}\n{'=' * 70}")


def load_docs():
    return sorted(DirectoryLoader(
        str(DOC_DIR), glob="*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    ).load(), key=lambda d: d.metadata["source"])


def resolve(llm, docs):
    """문서 단위로 대명사를 실명으로 바꾼다. 조각이 아니라 문서 전체를 준다."""
    chain = REWRITE | llm | StrOutputParser()
    out = []
    for d in docs:
        text = chain.invoke({"text": d.page_content}).strip()
        out.append(Document(page_content=text, metadata=d.metadata))
    return out


def extract(llm, graph, docs):
    graph.query(f"{WIPE} DETACH DELETE n")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=350, chunk_overlap=80, separators=["\n## ", "\n\n", "\n", " ", ""]
    ).split_documents(docs)
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["사람", "팀", "서비스"],
        allowed_relationships=["소속", "팀장", "운영", "의존", "온콜"],
    )
    graph.add_graph_documents(transformer.convert_to_graph_documents(chunks),
                              include_source=True, baseEntityLabel=True)

    # 이름 정합 (Q0011에서 배운 것)
    text = "\n".join(d.page_content for d in docs)
    for row in graph.query("MATCH (n:__Entity__) RETURN n.id AS id"):
        nid = row["id"]
        if nid and nid not in text:
            m = re.search(re.escape(nid), text, re.IGNORECASE)
            if m and m.group(0) != nid:
                graph.query("MATCH (n:__Entity__ {id:$o}) SET n.id=$n", {"o": nid, "n": m.group(0)})


def score(graph):
    got = {(r["s"], r["t"], r["o"]) for r in graph.query(
        "MATCH (a:__Entity__)-[r]->(b:__Entity__) RETURN a.id AS s, type(r) AS t, b.id AS o")}
    names = [r["id"] for r in graph.query("MATCH (n:__Entity__) RETURN n.id AS id")]
    junk = [n for n in names if any(p == n or n.startswith(p + " ") or n.endswith(" " + p) for p in PRONOUNS)]
    return len(GOLD & got), len(got - GOLD), junk


def main():
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 2
    print("Q0016 · 대명사를 실명으로 바꾸고 나서 추출하면 얼마나 달라지나")

    llm = get_azure_llm()
    graph = get_neo4j_graph()
    docs = load_docs()

    section(1, "다시 쓰기 전과 후")
    resolved = resolve(llm, docs)
    for before, after in zip(docs, resolved):
        name = Path(before.metadata["source"]).name
        b_lines = before.page_content.splitlines()
        a_lines = after.page_content.splitlines()
        changed = [(x, y) for x, y in zip(b_lines, a_lines) if x != y and x.strip()]
        print(f"\n  {name}: {len(changed)}줄 바뀜")
        for x, y in changed[:3]:
            print(f"    전: {x.strip()[:60]}")
            print(f"    후: {y.strip()[:60]}")

    section(2, f"추출 결과 비교 (각 {runs}회)")
    results = {"원문 그대로": [], "상호참조 해결 후": []}
    for label, source in (("원문 그대로", docs), ("상호참조 해결 후", resolved)):
        for i in range(runs):
            extract(llm, graph, source)
            hit, extra, junk = score(graph)
            results[label].append((hit, extra, junk))
            print(f"  [{label}] {i+1}회: 정답 {hit}/{len(GOLD)}, 정답 외 {extra}개, "
                  f"대명사 노드 {len(junk)}개 {junk[:4]}")

    section(3, "정리")
    for label, rs in results.items():
        avg_hit = sum(r[0] for r in rs) / len(rs)
        avg_junk = sum(len(r[2]) for r in rs) / len(rs)
        print(f"  {label:12s} 재현율 평균 {avg_hit/len(GOLD):.0%}, 대명사 노드 평균 {avg_junk:.1f}개")
    print("\n  상호참조 해결은 추출 '전'에 한다. 추출 후에 대명사 노드를 지우는 건 늦다.")
    print("  이미 관계가 엉뚱한 노드에 붙은 뒤라, 지우면 관계까지 같이 사라진다.")


if __name__ == "__main__":
    main()
