"""
Q0015 · 전역 요약 질문은 왜 그래프가 이기나?

지금까지 벡터가 진 질문은 "긴 사슬" 하나였다. 여기 하나가 더 있다. 전역 질문이다.

  "장애 회고 24건에서 가장 자주 나온 원인은 무엇인가요?"

이 질문에 답하려면 24건을 전부 읽고 세어야 한다. 벡터 검색은 질문과 비슷한 조각 k개만
가져온다. k가 4면 24건 중 4건만 본다. 나머지 20건은 존재하지 않는 것과 같다.

그래프는 다르다. 회고마다 (장애)-[:원인]->(원인) 을 만들어 두면 집계 한 줄이면 끝난다.

  MATCH (:장애)-[:원인]->(c:원인) RETURN c.id, count(*) ORDER BY count(*) DESC

실험은 세 방식을 비교한다.
  벡터 k=4     : 보통 쓰는 설정
  벡터 k=전체  : 조각을 전부 넣어 준다. 그러면 되는가? 비용은?
  그래프       : LLM이 추출한 그래프에 집계 Cypher

실행: python code/Q0015_global_summary.py
      python code/Q0015_global_summary.py --rebuild
"""

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import LLMGraphTransformer, GraphCypherQAChain

from config.settings import get_azure_llm, get_azure_embeddings, get_neo4j_graph, DATA_DIR

DOC_DIR = DATA_DIR / "incident_sample"
WIPE = "MATCH (n) WHERE n:__Entity__ OR n:Document"

# 생성 스크립트가 정해둔 분포. 이게 정답이다.
GOLD_CAUSE = {"배포 실수": 8, "커넥션 고갈": 6, "디스크 부족": 4, "외부 API 지연": 3, "설정 오류": 3}

QUESTIONS = [
    ("장애 원인 중 가장 자주 나온 것은 무엇이고 몇 건인가요?", "배포 실수, 8건", "전역"),
    ("커넥션 고갈로 분류된 장애는 모두 몇 건인가요?", "6건", "전역"),
    ("장애 회고 03에서 영향을 받은 서비스는 무엇인가요?", None, "국소"),
]


def section(n, title):
    print(f"\n{'=' * 70}\n[{n}] {title}\n{'=' * 70}")


def load_docs():
    docs = DirectoryLoader(
        str(DOC_DIR), glob="*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    ).load()
    return sorted(docs, key=lambda d: d.metadata["source"])


CAUSE_CATEGORIES = tuple(GOLD_CAUSE.keys())


class Incident(BaseModel):
    """회고 한 건에서 뽑을 것. 원인은 정해진 다섯 가지 중 하나만 고르게 한다."""
    incident_id: str = Field(description="문서 제목의 '장애 회고 NN' 그대로")
    cause: Literal[CAUSE_CATEGORIES] = Field(description="'원인은 X로 분류했다'의 X")
    service: str = Field(description="영향을 받은 서비스 이름, 원문 표기 그대로")
    team: str = Field(description="대응한 팀 이름")


def build_graph_free(llm, graph, docs):
    """자유 추출. Q0010부터 써온 LLMGraphTransformer 그대로."""
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["장애", "원인", "서비스", "팀"],
        allowed_relationships=["원인", "영향", "대응"],
    )
    graph_docs = transformer.convert_to_graph_documents(docs)
    graph.add_graph_documents(graph_docs, include_source=True, baseEntityLabel=True)
    n_causes = graph.query("MATCH (c:원인) RETURN count(c) AS c")[0]["c"]
    return n_causes


def build_graph_typed(llm, graph, docs):
    """고정 스키마 추출. 문서마다 Incident 하나를 받아 그대로 적재한다.

    자유 추출은 '원인'을 서술 그대로 노드로 만들어 이름이 흩어진다. 집계가 안 된다.
    전역 질문에 답하려면 세는 단위가 통일돼야 한다. 그래서 LLM에게 '다섯 가지 중 하나를
    고르라'고 강제한다. 이것도 LLM 추출이다. 다만 그래프 모양을 LLM이 아니라 사람이 정한다.
    """
    extractor = llm.with_structured_output(Incident)
    n_ok = 0
    for d in docs:
        try:
            inc = extractor.invoke(
                "아래 장애 회고에서 정보를 뽑으세요.\n\n" + d.page_content)
        except Exception as e:
            print(f"    추출 실패: {Path(d.metadata['source']).name} ({type(e).__name__})")
            continue
        graph.query(
            "MERGE (i:__Entity__:장애 {id:$iid}) "
            "MERGE (c:__Entity__:원인 {id:$cause}) "
            "MERGE (s:__Entity__:서비스 {id:$svc}) "
            "MERGE (t:__Entity__:팀 {id:$team}) "
            "MERGE (i)-[:원인]->(c) MERGE (i)-[:영향]->(s) MERGE (i)-[:대응]->(t)",
            {"iid": inc.incident_id, "cause": inc.cause, "svc": inc.service, "team": inc.team},
        )
        n_ok += 1
    return n_ok


def build_graph(llm, rebuild):
    section(1, "그래프 만들기: 자유 추출 vs 고정 스키마 추출")
    graph = get_neo4j_graph()
    n = graph.query(f"{WIPE} RETURN count(n) AS c")[0]["c"]
    if n and not rebuild:
        print(f"  기존 그래프 재사용: 노드 {n}개")
        graph.refresh_schema()
        return graph
    if n:
        graph.query(f"{WIPE} DETACH DELETE n")

    docs = load_docs()
    print(f"  회고 {len(docs)}건")

    print("\n  (a) 자유 추출 (LLMGraphTransformer)")
    n_causes = build_graph_free(llm, graph, docs)
    print(f"      원인 노드가 {n_causes}개 생겼다. 정답 분류는 5개다.")
    sample = [r["id"] for r in graph.query("MATCH (c:원인) RETURN c.id AS id LIMIT 6")]
    print(f"      예: {sample}")
    print("      -> 서술을 그대로 노드로 만든다. 이름이 흩어져 셀 수가 없다.")
    graph.query(f"{WIPE} DETACH DELETE n")

    print("\n  (b) 고정 스키마 추출 (with_structured_output, 원인은 5개 중 택일)")
    n_ok = build_graph_typed(llm, graph, docs)
    print(f"      {n_ok}/{len(docs)}건 적재")
    graph.refresh_schema()
    return graph


def check_extraction(graph):
    section(2, "추출이 정답 분포와 맞는지 먼저 본다")
    rows = graph.query("""
        MATCH (i:장애)-[:원인]->(c:원인)
        RETURN c.id AS cause, count(DISTINCT i) AS n ORDER BY n DESC
    """)
    got = {r["cause"]: r["n"] for r in rows}
    print(f"  {'원인':14s} {'정답':>4s} {'그래프':>6s}")
    for cause, gold in GOLD_CAUSE.items():
        print(f"  {cause:14s} {gold:>4d} {got.get(cause, 0):>6d}")
    extra = [c for c in got if c not in GOLD_CAUSE]
    if extra:
        print(f"  정답에 없는 원인 이름: {extra}")
    print("\n  💡 전역 질문의 정확도는 추출 정확도에 그대로 묶인다.")
    print("     한 건을 놓치거나 이름을 다르게 뽑으면 집계가 어긋난다.")


def vector_runner(llm, k):
    docs = load_docs()
    store = FAISS.from_documents(docs, get_azure_embeddings())
    retriever = store.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(
        "아래 근거만 써서 한국어로 짧게 답하세요. 근거에 없으면 '문서에서 찾을 수 없습니다'라고 답하세요.\n"
        "[근거]\n{context}\n\n[질문] {question}\n[답변]"
    )
    chain = prompt | llm | StrOutputParser()

    def run(q):
        hits = retriever.invoke(q)
        ctx = "\n\n".join(d.page_content for d in hits)
        return chain.invoke({"context": ctx, "question": q}), len(ctx)

    return run


def main():
    rebuild = "--rebuild" in sys.argv
    print("Q0015 · 문서 전체를 세어야 하는 질문")

    llm = get_azure_llm()
    graph = build_graph(llm, rebuild)
    check_extraction(graph)

    n_docs = len(load_docs())
    vec4 = vector_runner(llm, 4)
    vec_all = vector_runner(llm, n_docs)
    cypher = GraphCypherQAChain.from_llm(
        llm, graph=graph, verbose=False, allow_dangerous_requests=True)

    section(3, "질문 비교")
    for i, (q, gold, kind) in enumerate(QUESTIONS, 1):
        print(f"\n{'-' * 70}\n질문 {i} ({kind}): {q}")
        if gold:
            print(f"  정답: {gold}")
        print(f"{'-' * 70}")

        a, chars = vec4(q)
        print(f"  [벡터 k=4     ] {a.strip()[:90]}   (근거 {chars:,}자)")
        a, chars = vec_all(q)
        print(f"  [벡터 k={n_docs:<2d}    ] {a.strip()[:90]}   (근거 {chars:,}자)")
        for attempt in range(2):
            try:
                r = cypher.invoke({"query": q}).get("result", "")
                print(f"  [그래프       ] {str(r).strip()[:90]}")
                break
            except Exception as e:
                if attempt == 1:
                    print(f"  [그래프       ] 실패 {type(e).__name__}")

    section(4, "정리")
    print("  벡터 k=4   : 24건 중 4건만 본다. 전역 질문에는 구조적으로 답할 수 없다.")
    print(f"  벡터 k=전체: 24건이면 다 넣을 수 있다. 하지만 근거가 {n_docs}배 커지고,")
    print("               회고가 2,400건이면 넣을 수도 없다. 그리고 LLM이 세는 건 믿을 수 없다.")
    print("  그래프     : 건수와 무관하게 집계 한 줄. 단, 세는 단위가 통일돼야 한다.")
    print("               자유 추출은 원인 이름을 흩어놓아 집계가 불가능했다.")
    print("               고정 스키마로 '5개 중 택일'을 강제하니 집계가 됐다.")


if __name__ == "__main__":
    main()
