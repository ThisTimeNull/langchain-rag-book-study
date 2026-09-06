"""
Q0011 · GraphRAG가 이기는 데이터는 어떤 모양인가?

Q0010에서는 GraphRAG가 VectorRAG에 졌다. 문서가 평평한 규칙 나열이었기 때문이다.
이번에는 조건을 바꾼다. 사람, 팀, 서비스가 서로 얽힌 문서를 준비했다.

  data/graph_sample/
    team_roster.md   누가 어느 팀인지, 팀장이 누구인지
    service_map.md   어느 팀이 어느 서비스를 맡는지, 서비스끼리 뭐에 의존하는지
    oncall.md        팀별 온콜 담당자가 누구인지

핵심은 이것이다. 답이 한 문단에 없다. 문서 세 개에 흩어진 사실을 이어야 답이 나온다.

  "결제 API가 의존하는 서비스의 온콜 담당자는?"
    결제 API -의존-> 인증 서비스 -담당- 인프라팀 -온콜-> 한서준
    (service_map.md)          (service_map.md)   (oncall.md)

실행: python code/Q0011_graphrag_wins.py
      python code/Q0011_graphrag_wins.py --rebuild

주의: Neo4j Community는 데이터베이스가 하나다. 이 코드는 그래프를 새로 만들 때
      기존 실습 그래프(Q0010 포함)를 지운다. 한 번에 하나씩 돌리면 된다.
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

DOC_DIR = DATA_DIR / "graph_sample"
WIPE = "MATCH (n) WHERE n:__Entity__ OR n:Document"

QUESTIONS = [
    # (질문, 난이도 설명, 정답)
    ("인프라팀 팀장은 누구인가요?", "1다리, 한 문장에 답이 있음", "정하늘"),
    ("결제 API가 의존하는 서비스를 운영하는 팀은 어디인가요?", "2다리, 문서 2개", "인프라팀"),
    ("결제 API가 의존하는 서비스를 운영하는 팀의 온콜 담당자는 누구인가요?", "3다리, 문서 3개", "김철수"),
    ("추천 엔진이 직접 또는 간접으로 의존하는 서비스를 모두 알려주세요.", "짧은 사슬", "결제 API, 인증 서비스"),
    # 사슬이 길면 벡터는 조각 k개로 절대 다 못 모은다. 그래프는 -[:의존*]-> 한 줄이면 끝난다.
    ("감사 로그가 직접 또는 간접으로 의존하는 서비스는 모두 몇 개인가요?", "17다리 사슬", "17개"),
]


def section(n, title):
    print(f"\n{'=' * 68}\n[{n}] {title}\n{'=' * 68}")


def load_docs():
    return DirectoryLoader(
        str(DOC_DIR), glob="*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    ).load()


def align_ids_to_source(graph, docs):
    """노드 이름을 원문 표기로 되돌린다.

    LLM은 '결제 API'를 '결제 Api'처럼 제목 형태로 바꿔 뽑는다. 그러면 나중에 LLM이
    Cypher에서 {id:'결제 API'}로 찾을 때 하나도 안 걸린다. GraphRAG에서 말하는
    엔티티 정합(entity resolution) 문제가 가장 흔하게 나타나는 모습이다.

    여기서는 원문 텍스트에 같은 문자열이 대소문자만 다르게 있으면 원문 쪽으로 맞춘다.
    실무에서는 별칭 사전이나 임베딩 유사도로 더 정교하게 푼다.
    """
    text = "\n".join(d.page_content for d in docs)
    fixed = []
    for row in graph.query("MATCH (n:__Entity__) RETURN n.id AS id"):
        nid = row["id"]
        if not nid or nid in text:
            continue
        m = re.search(re.escape(nid), text, re.IGNORECASE)
        if m and m.group(0) != nid:
            graph.query(
                "MATCH (n:__Entity__ {id:$old}) SET n.id = $new",
                {"old": nid, "new": m.group(0)},
            )
            fixed.append((nid, m.group(0)))
    return fixed


def build_graph_directly(graph):
    """LLM 없이 규칙으로 그래프를 적재한다.

    우리 문서는 이미 정형이다. "A는 B에 의존한다" 같은 문장 틀이 정해져 있다.
    이런 데이터는 정규식으로 읽어 그대로 넣으면 100% 정확하고 매번 같은 결과가 나온다.
    LLM 추출은 문장 틀이 없는 비정형 문서에나 필요하다.
    """
    graph.query(f"{WIPE} DETACH DELETE n")
    text = {f.name: f.read_text(encoding="utf-8") for f in DOC_DIR.glob("*.md")}

    rules = [
        (r"- (.+?)은 (.+?)의 팀장이다\.", "사람", "팀", "팀장"),
        (r"- (.+?)은 (.+?) 소속이다\.", "사람", "팀", "소속"),
        (r"- (.+?)이 (.+?)를 운영한다\.", "팀", "서비스", "운영"),
        (r"- (.+?)는 (.+?)에 의존한다\.", "서비스", "서비스", "의존"),
        (r"- (.+?)은 (.+?)의 온콜을 맡는다\.", "사람", "팀", "온콜"),
    ]
    total = 0
    whole = "\n".join(text.values())
    for pattern, la, lb, rel in rules:
        pairs = re.findall(pattern, whole)
        for a, b in pairs:
            graph.query(
                f"MERGE (a:__Entity__:{la} {{id:$a}}) MERGE (b:__Entity__:{lb} {{id:$b}}) "
                f"MERGE (a)-[:{rel}]->(b)",
                {"a": a.strip(), "b": b.strip()},
            )
        print(f"  {rel:4s} {len(pairs)}개")
        total += len(pairs)
    graph.refresh_schema()
    print(f"  합계 {total}개 관계를 넣었습니다. 누락 없음, 매번 같은 결과.")
    return graph


def build_graph(llm, rebuild):
    section(1, "그래프 만들기")

    graph = get_neo4j_graph()
    n = graph.query(f"{WIPE} RETURN count(n) AS c")[0]["c"]

    if n and not rebuild:
        print(f"  기존 그래프 재사용: 노드 {n}개")
        fixed = align_ids_to_source(graph, load_docs())
        if fixed:
            print(f"  이름 정합: {len(fixed)}개 수정 " + ", ".join(f"{a}->{b}" for a, b in fixed[:4]))
        graph.refresh_schema()
        return graph

    if n:
        graph.query(f"{WIPE} DETACH DELETE n")
        print(f"  기존 노드 {n}개를 지우고 새로 만듭니다.")

    docs = load_docs()
    # 추출용 청크는 작게 자른다. 400자로 넣었더니 원문 의존 관계 24개 중 8개를 놓쳤다.
    # 한 번에 많이 주면 LLM이 뒷부분을 흘린다. 청킹은 검색만의 문제가 아니다.
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=150, chunk_overlap=30, separators=["\n- ", "\n", " ", ""]
    ).split_documents(docs)
    print(f"  문서 {len(docs)}개, 조각 {len(chunks)}개를 LLM에 넣습니다.")

    transformer = LLMGraphTransformer(
        llm=llm,
        # 관계 이름이 겹치면 안 된다. 처음에 '담당'을 사람과 팀, 팀과 서비스 양쪽에 썼더니
        # LLM이 둘을 섞어 뽑아서 다중 홉 경로가 끊겼다. 역할마다 다른 이름을 준다.
        allowed_nodes=["사람", "팀", "서비스"],
        allowed_relationships=["소속", "팀장", "운영", "의존", "온콜"],
    )
    graph_docs = transformer.convert_to_graph_documents(chunks)
    graph.add_graph_documents(graph_docs, include_source=True, baseEntityLabel=True)
    graph.refresh_schema()

    print(f"  추출 결과: 노드 {sum(len(d.nodes) for d in graph_docs)}개, "
          f"관계 {sum(len(d.relationships) for d in graph_docs)}개")

    fixed = align_ids_to_source(graph, docs)
    if fixed:
        print(f"  이름 정합: {len(fixed)}개 수정 " + ", ".join(f"{a} -> {b}" for a, b in fixed[:5]))
        print("  💡 이걸 안 하면 LLM이 짠 Cypher가 원문 이름으로 찾다가 0건을 반환한다.")
    graph.refresh_schema()
    return graph


def show_graph(graph):
    section(2, "만들어진 그래프")

    stats = graph.query("""
        MATCH (a:__Entity__)-[r]->(b:__Entity__)
        RETURN type(r) AS 관계, count(*) AS 개수 ORDER BY 개수 DESC
    """)
    print("  관계 종류별 개수:")
    for r in stats:
        print(f"    {r['관계']:6s} {r['개수']}개")
    print("\n  샘플:")
    for r in graph.query("""
        MATCH (a:__Entity__)-[r]->(b:__Entity__)
        RETURN a.id AS s, type(r) AS t, b.id AS o ORDER BY t, s LIMIT 6
    """):
        print(f"    ({r['s']}) -[{r['t']}]-> ({r['o']})")

    print("\n  3다리 경로가 실제로 이어지는지 직접 확인합니다.")
    path = graph.query("""
        MATCH p = (a:__Entity__ {id:'결제 API'})-[*1..4]-(b:__Entity__ {id:'김철수'})
        RETURN [x IN nodes(p) | x.id] AS 경로, length(p) AS 다리
        ORDER BY 다리 LIMIT 3
    """)
    for r in path:
        print(f"    {' -> '.join(r['경로'])}  ({r['다리']}다리)")
    if not path:
        print("    (경로 없음. 추출이 기대대로 안 됐다는 뜻입니다)")


def build_vector_run(llm):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=60, separators=["\n## ", "\n- ", "\n", " ", ""]
    ).split_documents(load_docs())
    k = 4
    retriever = FAISS.from_documents(chunks, get_azure_embeddings()).as_retriever(
        search_kwargs={"k": k}
    )
    print(f"\n  [VectorRAG 준비] 조각 {len(chunks)}개 중 {k}개를 검색합니다 "
          f"(코퍼스의 {k / len(chunks):.0%}).")
    print("  💡 코퍼스가 작으면 벡터가 사실상 전체를 다 넣게 되어 비교가 무의미해진다.")
    prompt = ChatPromptTemplate.from_template(
        "아래 근거만 써서 한국어로 짧게 답하세요. 근거에 없으면 '문서에서 찾을 수 없습니다'라고 답하세요.\n"
        "[근거]\n{context}\n\n[질문] {question}\n[답변]"
    )

    def run(q):
        hits = retriever.invoke(q)
        ctx = "\n\n".join(f"[{Path(d.metadata['source']).name}] {d.page_content}" for d in hits)
        return (prompt | llm | StrOutputParser()).invoke({"context": ctx, "question": q})

    return run


def compare(vector_run, graph_chain):
    section(3, "같은 질문을 둘 다에게 던지기")

    for i, (q, hop, gold) in enumerate(QUESTIONS, 1):
        print(f"\n{'-' * 68}\n질문 {i} ({hop})\n  {q}\n  정답: {gold}\n{'-' * 68}")
        print(f"  [Vector] {vector_run(q).strip()[:180]}")
        try:
            out = graph_chain.invoke({"query": q})
            cypher = ""
            for step in out.get("intermediate_steps", []):
                if isinstance(step, dict) and "query" in step:
                    cypher = " ".join(str(step["query"]).split())
            print(f"  [Graph ] {str(out.get('result', '')).strip()[:180]}")
            if cypher:
                print(f"           Cypher: {cypher[:140]}")
        except Exception as e:
            print(f"  [Graph ] 실패 {type(e).__name__}: {str(e).splitlines()[0][:110]}")


def main():
    rebuild = "--rebuild" in sys.argv
    direct = "--direct" in sys.argv
    print("Q0011 · 관계가 얽힌 문서로 GraphRAG와 VectorRAG를 다시 붙입니다.")

    llm = get_azure_llm()
    if direct:
        section(1, "그래프 만들기 (LLM 없이 규칙으로 적재)")
        graph = build_graph_directly(get_neo4j_graph())
    else:
        graph = build_graph(llm, rebuild)
    show_graph(graph)

    chain = GraphCypherQAChain.from_llm(
        llm, graph=graph, verbose=False,
        allow_dangerous_requests=True, return_intermediate_steps=True,
    )
    compare(build_vector_run(llm), chain)

    print(f"\n{'=' * 68}")
    print("정리")
    print("  1. 답이 한 문단에 있으면 벡터로 충분합니다.")
    print("  2. 긴 사슬을 따라가야 하는 질문은 벡터가 못 풉니다. 조각 k개로는 사슬을 다 못 모읍니다.")
    print("  3. 그래프가 이기려면 그래프가 정확해야 합니다.")
    print("     LLM 추출은 원문 의존 관계 24개 중 8개를 놓쳤고, 돌릴 때마다 놓치는 게 달라집니다.")
    print("     --direct 로 규칙 적재하면 누락이 없고 5문제를 다 맞힙니다.")
    print("\n  실행 비교: python code/Q0011_graphrag_wins.py --rebuild   (LLM 추출)")
    print("             python code/Q0011_graphrag_wins.py --direct    (규칙 적재)")


if __name__ == "__main__":
    main()
