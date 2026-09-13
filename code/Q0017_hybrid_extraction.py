"""
Q0017 · 규칙과 LLM을 섞어 추출하면?

지금까지 확인한 것.
  Q0011  문장 틀이 있으면 규칙이 100%, LLM이 67%
  Q0012  문장 틀이 없으면 규칙이 0%, LLM이 82%
  Q0016  LLM은 "인증 서비스는 인프라팀 소관"을 거꾸로 읽는다 (방향 오류)

실무 문서는 둘이 섞여 있다. 정형 목록도 있고 회의록도 있다.
그래서 이렇게 한다.

  1. 규칙으로 먼저 긁는다.          정확하고, 방향이 틀릴 일이 없고, 공짜다
  2. 규칙이 하나도 못 읽은 조각만    남은 것만 LLM에 보낸다. 호출 수가 줄어든다
     LLM에 넘긴다
  3. 합친다                         같은 사실이 양쪽에서 나오면 규칙 쪽을 믿는다

코퍼스: data/mixed_sample/
  roster.md                          정형 목록 (규칙이 읽는다)
  weekly_meeting.md 등 산문 3개       회의록, 회고, 인수인계 (LLM이 읽는다)

실행: python code/Q0017_hybrid_extraction.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j import LLMGraphTransformer

from config.settings import get_azure_llm, get_neo4j_graph, DATA_DIR

DOC_DIR = DATA_DIR / "mixed_sample"
WIPE = "MATCH (n) WHERE n:__Entity__ OR n:Document"

RULES = [
    (r"- (.+?)은 (.+?)의 팀장이다\.", "사람", "팀", "팀장"),
    (r"- (.+?)은 (.+?) 소속이다\.", "사람", "팀", "소속"),
    (r"- (.+?)이 (.+?)를 운영한다\.", "팀", "서비스", "운영"),
    (r"- (.+?)는 (.+?)에 의존한다\.", "서비스", "서비스", "의존"),
    (r"- (.+?)은 (.+?)의 온콜을 맡는다\.", "사람", "팀", "온콜"),
]

# 정답. roster.md 에 7개, 산문에만 있는 것 4개 (의존 관계), 양쪽에 다 있는 것도 있다.
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
DIRECTION_TRAP = {("인증 서비스", "운영", "인프라팀"), ("알림 서비스", "운영", "인프라팀")}


def section(n, title):
    print(f"\n{'=' * 70}\n[{n}] {title}\n{'=' * 70}")


def load_chunks():
    docs = DirectoryLoader(str(DOC_DIR), glob="*.md",
                           loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}).load()
    return RecursiveCharacterTextSplitter(
        chunk_size=350, chunk_overlap=60, separators=["\n## ", "\n\n", "\n- ", "\n", " ", ""]
    ).split_documents(docs)


def rule_extract(text):
    """규칙으로 읽는다. 매칭된 (주어, 관계, 목적어)와 라벨을 돌려준다."""
    found = []
    for pattern, la, lb, rel in RULES:
        for a, b in re.findall(pattern, text):
            found.append((a.strip(), la, rel, b.strip(), lb))
    return found


def load_triples(graph, triples, source):
    for a, la, rel, b, lb in triples:
        graph.query(
            f"MERGE (a:__Entity__:{la} {{id:$a}}) MERGE (b:__Entity__:{lb} {{id:$b}}) "
            f"MERGE (a)-[r:{rel}]->(b) SET r.source = $src",
            {"a": a, "b": b, "src": source})


def llm_extract(llm, graph, chunks):
    if not chunks:
        return
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["사람", "팀", "서비스"],
        allowed_relationships=["소속", "팀장", "운영", "의존", "온콜"],
    )
    graph.add_graph_documents(transformer.convert_to_graph_documents(chunks),
                              include_source=True, baseEntityLabel=True)
    text = "\n".join(c.page_content for c in chunks)
    for row in graph.query("MATCH (n:__Entity__) RETURN n.id AS id"):
        nid = row["id"]
        if nid and nid not in text:
            m = re.search(re.escape(nid), text, re.IGNORECASE)
            if m and m.group(0) != nid:
                merge_into(graph, nid, m.group(0))


def merge_into(graph, dup_id, canon_id):
    """중복 노드를 정식 노드로 합친다.

    하이브리드에서는 규칙이 '결제 API'를 먼저 만들고 LLM이 '결제 Api'를 또 만든다.
    이름만 바꾸면 유일성 제약에 걸린다. 그래서 관계를 옮기고 중복을 지운다.
    정식 노드가 없으면 그냥 이름을 바꾼다.
    """
    exists = graph.query("MATCH (n:__Entity__ {id:$c}) RETURN count(n) AS c", {"c": canon_id})[0]["c"]
    if not exists:
        graph.query("MATCH (n:__Entity__ {id:$o}) SET n.id=$n", {"o": dup_id, "n": canon_id})
        return
    graph.query(
        "MATCH (canon:__Entity__ {id:$c}), (dup:__Entity__ {id:$d}) "
        "CALL apoc.refactor.mergeNodes([canon, dup], {properties:'discard', mergeRels:true}) "
        "YIELD node RETURN count(node)",
        {"c": canon_id, "d": dup_id},
    )


def score(graph):
    got = {(r["s"], r["t"], r["o"]) for r in graph.query(
        "MATCH (a:__Entity__)-[r]->(b:__Entity__) RETURN a.id AS s, type(r) AS t, b.id AS o")}
    return len(GOLD & got), len(got - GOLD), len(DIRECTION_TRAP & got)


def run_rules_only(graph, chunks):
    graph.query(f"{WIPE} DETACH DELETE n")
    for c in chunks:
        load_triples(graph, rule_extract(c.page_content), "rule")
    return 0  # LLM 호출 수


def run_llm_only(llm, graph, chunks):
    graph.query(f"{WIPE} DETACH DELETE n")
    llm_extract(llm, graph, chunks)
    return len(chunks)


def run_hybrid(llm, graph, chunks):
    """규칙이 하나라도 읽은 조각은 규칙 결과만 쓴다. 못 읽은 조각만 LLM에 보낸다."""
    graph.query(f"{WIPE} DETACH DELETE n")
    to_llm = []
    for c in chunks:
        triples = rule_extract(c.page_content)
        if triples:
            load_triples(graph, triples, "rule")
        else:
            to_llm.append(c)
    llm_extract(llm, graph, to_llm)
    resolved = resolve_conflicts(graph)
    if resolved:
        print(f"      (충돌 해소: LLM이 거꾸로 뽑은 관계 {resolved}개를 규칙 쪽으로 정리)")
    return len(to_llm)


def resolve_conflicts(graph):
    """같은 사실이 양쪽에서 나오면 규칙을 믿는다.

    규칙이 (인프라팀)-[운영]->(인증 서비스)를 만들었는데 LLM이 산문에서
    (인증 서비스)-[운영]->(인프라팀)을 또 만든다. 규칙 관계에는 source='rule'이 찍혀 있으니
    그 역방향으로 LLM이 만든 관계가 있으면 지운다. 처음 실행 때 이걸 빼먹어서
    하이브리드의 방향 오류가 LLM만일 때보다 많이 나왔다.
    """
    rows = graph.query(
        "MATCH (a)-[r]->(b) WHERE r.source = 'rule' "
        "MATCH (b)-[x]->(a) WHERE type(x) = type(r) AND x.source IS NULL "
        "DELETE x RETURN count(*) AS n")
    return rows[0]["n"] if rows else 0


def main():
    print("Q0017 · 규칙으로 긁고 남은 것만 LLM에")
    llm = get_azure_llm()
    graph = get_neo4j_graph()
    chunks = load_chunks()

    section(1, "코퍼스")
    by_file = {}
    for c in chunks:
        by_file.setdefault(Path(c.metadata["source"]).name, []).append(c)
    for name, cs in sorted(by_file.items()):
        readable = sum(1 for c in cs if rule_extract(c.page_content))
        print(f"  {name:22s} 조각 {len(cs)}개, 규칙이 읽는 조각 {readable}개")
    print(f"  전체 조각 {len(chunks)}개, 정답 관계 {len(GOLD)}개")

    section(2, "세 방식 비교")
    print(f"  {'방식':10s} {'정답':>6s} {'정답 외':>7s} {'방향 오류':>8s} {'LLM 호출':>8s}")
    for label, fn in (("규칙만", lambda: run_rules_only(graph, chunks)),
                      ("LLM만", lambda: run_llm_only(llm, graph, chunks)),
                      ("하이브리드", lambda: run_hybrid(llm, graph, chunks))):
        calls = fn()
        hit, extra, wrong_dir = score(graph)
        print(f"  {label:10s} {hit:>3d}/{len(GOLD)} {extra:>7d} {wrong_dir:>8d} {calls:>8d}")

    section(3, "정리")
    print("  규칙만    : 정형 목록은 다 읽는다. 산문에만 있는 의존 관계 4개는 못 읽는다.")
    print("  LLM만     : 다 읽지만 방향을 거꾸로 뽑고, 정형 목록까지 LLM에 보내 호출이 많다.")
    print("  하이브리드: 정형은 규칙이, 산문은 LLM이. 겹치는 사실은 규칙을 믿는다.")
    print("\n  '방향 오류' 열은 (인증 서비스)-[운영]->(인프라팀) 처럼 거꾸로 뽑힌 개수다.")
    print("  규칙이 읽은 사실은 방향이 틀릴 수가 없다. 패턴에 방향이 박혀 있으니까.")
    print("  단, 산문에 같은 사실이 또 있으면 LLM이 거꾸로 뽑는다. 그래서 충돌 해소가 필요하다.")


if __name__ == "__main__":
    main()
