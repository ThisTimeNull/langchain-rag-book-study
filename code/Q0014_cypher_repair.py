"""
Q0014 · LLM이 짠 Cypher가 틀렸을 때 어떻게 고쳐서 다시 실행하나?

Q0013에서 같은 질문에 17개, 19개, 20개가 번갈아 나왔다. 그래프는 정확한데
LLM이 매번 다른 Cypher를 쓰기 때문이다. 자주 보인 실수는 이런 것들이다.

  1. 그래프에 없는 라벨을 쓴다.        MATCH (d:Document)-[:MENTIONS]->...   (그런 노드 없음)
  2. 시작점을 넓게 잡는다.             MATCH (:서비스)-[:의존*]->  (특정 서비스가 아니라 전부)
  3. 방향을 거꾸로 탄다.               (a)<-[:의존]-(b) 를 (a)-[:의존]->(b) 로

이 파일은 같은 질문을 여러 번 돌려 "얼마나 자주 틀리는지"를 먼저 잰다.
그 다음 검증과 재시도를 넣고 다시 잰다.

  검증 1. 쿼리에 쓰인 라벨과 관계가 실제 스키마에 있는가   (없으면 즉시 재생성)
  검증 2. EXPLAIN 으로 문법이 맞는가                        (틀리면 오류를 보여주고 재생성)
  검증 3. 실행 결과가 비어 있지 않은가                       (비면 "0건이었다"고 알려주고 재생성)

실행: python code/Q0014_cypher_repair.py            (질문당 5회)
      python code/Q0014_cypher_repair.py --runs 10
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import get_azure_llm
from Q0013_hybrid_search import build_graph   # Q0013과 같은 그래프 (규칙 적재, 정확함)

QUESTIONS = [
    ("감사 로그가 직접 또는 간접으로 의존하는 서비스는 모두 몇 개인가요?", "17"),
    ("결제 API가 의존하는 서비스를 운영하는 팀의 온콜 담당자는 누구인가요?", "김철수"),
    ("인프라팀이 운영하는 서비스에 의존하는 서비스는 무엇인가요?", "결제 API"),
]

GEN_PROMPT = ChatPromptTemplate.from_template(
    "아래 스키마를 가진 Neo4j 그래프에 대해 질문에 답하는 Cypher 쿼리를 하나만 작성하세요.\n"
    "스키마에 있는 라벨과 관계 이름만 쓰세요. 설명 없이 쿼리만 출력하세요.\n\n"
    "[스키마]\n{schema}\n\n{examples}{feedback}[질문] {question}\n[Cypher]"
)

# 검증으로 못 잡는 '의미 오류'는 예시로 막는다.
# 실제로 틀린 패턴을 그대로 보여주고 왜 틀렸는지 적는다.
FEW_SHOT = """[예시]
질문: X가 직접 또는 간접으로 의존하는 서비스는 몇 개인가요?
잘못된 쿼리: MATCH (s:서비스 {id:'X'})-[:의존*0..]->(d) RETURN count(DISTINCT d)
  -> *0.. 은 시작점 자신을 포함하므로 하나가 더 세어진다.
올바른 쿼리: MATCH (s:서비스 {id:'X'})-[:의존*1..]->(d:서비스) RETURN count(DISTINCT d) AS n

질문: X를 운영하는 팀은 어디인가요?
올바른 쿼리: MATCH (t:팀)-[:운영]->(s:서비스 {id:'X'}) RETURN t.id
  -> 운영 관계는 팀에서 서비스로 향한다. 방향을 지켜라.

"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "질문과 DB 조회 결과를 보고 한국어로 짧게 답하세요.\n"
    "[질문] {question}\n[조회 결과] {rows}\n[답변]"
)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def clean(cypher):
    cypher = re.sub(r"```(?:cypher)?", "", cypher).strip()
    return cypher.strip("` \n")


# --- 검증 세 가지 ------------------------------------------------------------
def check_schema(cypher, labels, rel_types):
    """쿼리에 쓰인 라벨과 관계 이름이 실제로 있는지 본다."""
    used_labels = set(re.findall(r"\(\s*\w*\s*:\s*([\w가-힣]+)", cypher))
    used_rels = set(re.findall(r"\[\s*\w*\s*:\s*([\w가-힣]+)", cypher))
    bad = [f"라벨 :{l}" for l in used_labels - labels] + [f"관계 :{r}" for r in used_rels - rel_types]
    return bad


def check_syntax(graph, cypher):
    try:
        graph.query("EXPLAIN " + cypher)
        return None
    except Exception as e:
        return str(e).splitlines()[0][:200]


def run_with_repair(llm, graph, question, max_tries=3, examples=""):
    labels = {r["label"] for r in graph.query("CALL db.labels() YIELD label")}
    rel_types = {r["relationshipType"] for r in graph.query("CALL db.relationshipTypes() YIELD relationshipType")}
    labels -= {"__Entity__"}

    feedback = ""
    log = []
    for attempt in range(1, max_tries + 1):
        cypher = clean((GEN_PROMPT | llm | StrOutputParser()).invoke(
            {"schema": graph.schema, "question": question, "feedback": feedback, "examples": examples}))

        bad = check_schema(cypher, labels, rel_types)
        if bad:
            feedback = f"[이전 시도의 문제] 스키마에 없는 이름을 썼습니다: {', '.join(bad)}. 스키마에 있는 것만 쓰세요.\n"
            log.append(f"시도 {attempt}: 스키마 위반 {bad}")
            continue

        err = check_syntax(graph, cypher)
        if err:
            feedback = f"[이전 시도의 문제] 문법 오류: {err}\n"
            log.append(f"시도 {attempt}: 문법 오류")
            continue

        try:
            rows = graph.query(cypher)
        except Exception as e:
            feedback = f"[이전 시도의 문제] 실행 오류: {str(e)[:150]}\n"
            log.append(f"시도 {attempt}: 실행 오류")
            continue

        if not rows:
            feedback = (f"[이전 시도의 문제] 이 쿼리는 0건을 반환했습니다: {cypher}\n"
                        f"시작 노드의 id 값이 정확한지, 관계 방향이 맞는지 다시 보세요.\n")
            log.append(f"시도 {attempt}: 0건")
            continue

        return cypher, rows, log

    return cypher, [], log


def run_plain(llm, graph, question, examples=""):
    """검증 없이 한 번 만들고 그대로 실행한다."""
    cypher = clean((GEN_PROMPT | llm | StrOutputParser()).invoke(
        {"schema": graph.schema, "question": question, "feedback": "", "examples": examples}))
    try:
        return cypher, graph.query(cypher)
    except Exception:
        return cypher, []


def answer(llm, question, rows):
    return (ANSWER_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question, "rows": str(rows)[:800]}).strip()


def main():
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 5
    print(f"Q0014 · 같은 질문을 {runs}번씩 돌려 Cypher 생성의 안정성을 잰다")

    llm = get_azure_llm()
    graph, n_rel = build_graph()
    print(f"  그래프: 관계 {n_rel}개 (규칙 적재, 정확함)")

    modes = [
        ("검증 없음",           lambda q: run_plain(llm, graph, q)[1]),
        ("검증 + 재시도",       lambda q: run_with_repair(llm, graph, q)[1]),
        ("예시 + 검증 + 재시도", lambda q: run_with_repair(llm, graph, q, examples=FEW_SHOT)[1]),
    ]
    totals = {name: 0 for name, _ in modes}
    for q, gold in QUESTIONS:
        section(f"질문: {q}\n정답: {gold}")
        for name, fn in modes:
            ok = 0
            answers = []
            for i in range(runs):
                rows = fn(q)
                a = answer(llm, q, rows) if rows else "(0건)"
                hit = gold in a
                ok += hit
                answers.append(("✅" if hit else "❌") + a[:36])
            print(f"\n  [{name}] {ok}/{runs}")
            for a in answers:
                print(f"    {a}")
            totals[name] += ok

    n = runs * len(QUESTIONS)
    section("정리")
    for name, ok in totals.items():
        print(f"  {name:20s}: {ok}/{n} ({ok/n:.0%})")
    print("\n  검증은 '틀린 쿼리를 잡는' 게 아니라 '틀린 줄 알 수 있는 쿼리를 잡는' 것이다.")
    print("  스키마 위반, 문법 오류, 0건은 잡힌다. 실행은 되는데 답이 틀린 쿼리는 못 잡는다.")


if __name__ == "__main__":
    main()
