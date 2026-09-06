"""
Q0010 · GraphRAG는 VectorRAG와 무엇이 다른가?

Q0007에서 VectorRAG 6단계를 만들었다. 이번엔 같은 문서로 GraphRAG를 만들고,
**같은 질문을 둘 다에게 던져** 어디서 갈리는지 본다.

  VectorRAG:  로드 → 청킹 → 임베딩 → FAISS 저장 → 거리 검색 → 생성
  GraphRAG :  로드 → **LLM이 엔티티, 관계 추출** → Neo4j 저장 → **LLM이 Cypher 생성** → 실행 → 생성
                     ↑ ①                                    ↑ ②

LLM이 쓰이는 지점이 1곳(생성)에서 3곳으로 늘어나는 게 핵심 차이다.

실행: python code/Q0010_graphrag_pipeline.py
      python code/Q0010_graphrag_pipeline.py --rebuild    # 그래프를 새로 만든다
필요: Neo4j 실행 + .env의 Azure 설정 (LLM, 임베딩)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import LLMGraphTransformer, Neo4jGraph, GraphCypherQAChain

from config.settings import get_azure_llm, get_azure_embeddings, get_neo4j_graph, DATA_DIR

# 정리용 조건. 공통 라벨을 따로 붙이지 않는 이유는, 그 라벨이 스키마에 노출되면
# LLM이 Cypher를 짤 때 어느 라벨로 매칭할지 헷갈려 정확도가 떨어지기 때문이다.
DEMO_MATCH = "MATCH (n) WHERE n:__Entity__ OR n:Document"

QUESTIONS = [
    # ① 답이 한 문단 안에 통째로 들어 있는 질문 → VectorRAG의 홈그라운드
    "연차는 사용 며칠 전까지 신청해야 하나요?",
    # ② '무엇이 무엇에 속하는가'를 묻는 구조 질문 → 그래프가 그대로 갖고 있는 정보
    "'한빛소프트 개발 가이드' 문서에는 어떤 항목들이 포함되어 있나요?",
    # ③ 상위-하위 관계를 모두 모아야 하는 질문
    "휴가에 포함되는 제도를 모두 알려주세요.",
]


def section(n, title):
    print(f"\n{'=' * 66}\n[{n}] {title}\n{'=' * 66}")


def load_docs():
    return DirectoryLoader(
        str(DATA_DIR / "sample"), glob="*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    ).load()


# --- ① LLM이 문서에서 그래프를 뽑아낸다 --------------------------------------
def build_graph(llm, rebuild: bool):
    section(1, "GraphRAG 준비, LLM이 문서에서 엔티티, 관계를 추출한다")

    graph = get_neo4j_graph()
    existing = graph.query(f"{DEMO_MATCH} RETURN count(n) AS c")[0]["c"]

    if existing and not rebuild:
        print(f"  기존 그래프 재사용: 노드 {existing}개 (--rebuild 로 다시 만들 수 있습니다)")
        graph.refresh_schema()
        return graph

    if existing:
        graph.query(f"{DEMO_MATCH} DETACH DELETE n")
        print(f"  기존 노드 {existing}개 삭제 후 재구축")

    docs = load_docs()
    # 문서가 길면 LLM이 놓치므로 적당히 잘라서 넣는다 (청킹은 VectorRAG만의 것이 아니다)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n## ", "\n- ", "\n", " ", ""]
    ).split_documents(docs)
    print(f"  문서 {len(docs)}개 → 조각 {len(chunks)}개를 LLM에 넣어 추출 중 …")

    # allowed_nodes/relationships를 주면 LLM이 제멋대로 라벨을 만드는 걸 억제할 수 있다.
    # (아무 제약 없이 두면 '규정'/'정책'/'규칙'처럼 같은 뜻의 라벨이 난립한다)
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["문서", "제도", "규칙", "부서", "기간"],
        allowed_relationships=["포함", "적용대상", "기한"],
        node_properties=["설명"],
    )
    graph_docs = transformer.convert_to_graph_documents(chunks)

    n_nodes = sum(len(d.nodes) for d in graph_docs)
    n_rels = sum(len(d.relationships) for d in graph_docs)
    print(f"  추출 결과: 노드 {n_nodes}개, 관계 {n_rels}개")

    # 데모 노드를 구분하기 위해 공통 라벨을 하나 더 붙인다
    for d in graph_docs:
        for node in d.nodes:
            node.type = node.type or "엔티티"
    graph.add_graph_documents(graph_docs, include_source=True, baseEntityLabel=True)

    graph.refresh_schema()
    print("\n  💡 이 단계가 GraphRAG의 진짜 비용이다. 문서마다 LLM을 호출해야 하고,")
    print("     같은 문서를 다시 돌리면 다른 그래프가 나올 수도 있다 (비결정적).")
    print("     VectorRAG의 임베딩은 같은 입력이면 항상 같은 벡터가 나온다, 이게 결정적 차이.")
    return graph


def show_graph(graph):
    section(2, "만들어진 그래프 들여다보기")

    labels = graph.query("""
        MATCH (n) WHERE n:__Entity__
        UNWIND labels(n) AS l
        WITH l WHERE l <> '__Entity__'
        RETURN l AS 라벨, count(*) AS 개수 ORDER BY 개수 DESC LIMIT 8
    """)
    print("\n  추출된 노드 종류:")
    for r in labels:
        print(f"    · {r['라벨']:12s} {r['개수']}개")

    rels = graph.query("""
        MATCH (a)-[r]->(b) WHERE a:__Entity__ AND b:__Entity__
        RETURN type(r) AS 관계, count(*) AS 개수 ORDER BY 개수 DESC LIMIT 8
    """)
    print("\n  추출된 관계 종류:")
    for r in rels:
        print(f"    · {r['관계']:12s} {r['개수']}개")

    sample = graph.query("""
        MATCH (a)-[r]->(b) WHERE a:__Entity__ AND b:__Entity__
        RETURN a.id AS 출발, type(r) AS 관계, b.id AS 도착 LIMIT 6
    """)
    print("\n  실제로 뽑힌 관계 몇 개:")
    for r in sample:
        print(f"    ({r['출발']}) -[{r['관계']}]-> ({r['도착']})")

    print("\n  💡 여기서 GraphRAG의 어려움이 보인다: 같은 대상이 다른 이름으로 갈라지거나")
    print("     (엔티티 해소 문제), 관계 이름이 기대와 다르게 나올 수 있다.")


# --- VectorRAG 쪽 ------------------------------------------------------------
def build_vector_chain(llm):
    docs = load_docs()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=60, separators=["\n## ", "\n- ", "\n", " ", ""]
    ).split_documents(docs)
    store = FAISS.from_documents(chunks, get_azure_embeddings())
    retriever = store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "아래 근거만 사용해 한국어로 간결히 답하세요. 근거에 없으면 '문서에서 찾을 수 없습니다'라고 답하세요.\n"
        "[근거]\n{context}\n\n[질문] {question}\n[답변]"
    )

    def run(q):
        hits = retriever.invoke(q)
        ctx = "\n\n".join(f"[{Path(d.metadata['source']).name}] {d.page_content}" for d in hits)
        answer = (prompt | llm | StrOutputParser()).invoke({"context": ctx, "question": q})
        srcs = sorted({Path(d.metadata["source"]).name for d in hits})
        return answer, srcs

    return run


# --- ② LLM이 질문을 Cypher로 바꾼다 -------------------------------------------
def build_graph_chain(llm, graph):
    section(3, "GraphRAG 질의, LLM이 자연어 질문을 Cypher로 번역한다")
    print("\n  LLM에게 넘겨지는 그래프 스키마 (이걸 보고 Cypher를 짠다):")
    for line in graph.schema.splitlines()[:8]:
        print(f"    {line[:90]}")

    chain = GraphCypherQAChain.from_llm(
        llm, graph=graph, verbose=False,
        allow_dangerous_requests=True,   # LLM이 만든 Cypher를 실행하므로 명시적 동의가 필요
        return_intermediate_steps=True,
    )
    print("\n  ⚠️ allow_dangerous_requests=True 가 필요하다.")
    print("     LLM이 생성한 쿼리를 그대로 DB에 실행하기 때문이다 (읽기 전용 계정을 쓰는 게 안전).")
    return chain


def compare(vector_run, graph_chain):
    section(4, "같은 질문을 둘 다에게 던져보기")

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'─' * 66}\n질문 {i}: {q}\n{'─' * 66}")

        answer, srcs = vector_run(q)
        print(f"  [VectorRAG] {answer.strip()[:200]}")
        print(f"              근거 출처: {', '.join(srcs)}")

        try:
            out = graph_chain.invoke({"query": q})
            cypher = ""
            for step in out.get("intermediate_steps", []):
                if isinstance(step, dict) and "query" in step:
                    cypher = " ".join(str(step["query"]).split())
            print(f"  [GraphRAG ] {str(out.get('result', '')).strip()[:200]}")
            if cypher:
                print(f"              생성된 Cypher: {cypher[:150]}")
        except Exception as e:
            print(f"  [GraphRAG ] ❌ {type(e).__name__}: {str(e).splitlines()[0][:120]}")


def main():
    rebuild = "--rebuild" in sys.argv
    print("Q0010 · GraphRAG vs VectorRAG, 같은 문서, 같은 질문")

    llm = get_azure_llm()
    graph = build_graph(llm, rebuild)
    show_graph(graph)
    graph_chain = build_graph_chain(llm, graph)
    vector_run = build_vector_chain(llm)
    compare(vector_run, graph_chain)

    print(f"\n{'=' * 66}")
    print("정리")
    print("  · VectorRAG: LLM 1곳(생성). 준비가 기계적이고 결정적. '문단 하나에 답이 있는' 질문에 강함")
    print("  · GraphRAG : LLM 3곳(추출, 번역, 생성). 준비가 비싸고 비결정적. '사실을 이어야 하는' 질문에 강함")
    print("  · 실무는 둘을 섞는다, 벡터로 진입점을 찾고 그래프로 확장")


if __name__ == "__main__":
    main()
