"""
Q0009 · Cypher란 무엇이고 APOC은 뭔가?

  Cypher = 그래프 DB의 질의 언어. 관계형 DB의 SQL에 해당한다.
  APOC   = Neo4j 서버 안에 끼워 넣는 **플러그인 라이브러리**(jar 파일 하나).
           별도 서버가 아니다. Cypher가 기본 제공하지 않는 기능을 프로시저/함수로 더해준다.

이 코드가 보여주는 것
  1) Cypher 기본, 화살표로 관계를 '그리는' 언어
  2) SQL과의 결정적 차이, 몇 다리 건널지 모를 때 (가변 길이 경로)
  3) 프로시저 vs 함수, CALL로 부르는 것과 RETURN 안에서 쓰는 것
  4) APOC이 '서버'가 아니라 '라이브러리'라는 증거

실행: python code/Q0009_cypher_apoc.py
(Neo4j가 떠 있어야 한다: brew services start neo4j)
이 코드가 만드는 노드는 :Q0009 라벨을 달고, 끝나면 스스로 지운다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_neo4j_graph


def section(n, title):
    print(f"\n{'=' * 64}\n[{n}] {title}\n{'=' * 64}")


def cleanup(g):
    g.query("MATCH (n:Q0009) DETACH DELETE n")


# --- 1) Cypher 기본 ---------------------------------------------------------
def demo_basics(g):
    section(1, "Cypher 기본, 관계를 '그림처럼' 쓰는 언어")

    cleanup(g)
    g.query("""
        CREATE (철수:Q0009 {name:'김철수', dept:'개발팀'}),
               (영희:Q0009 {name:'박영희', dept:'개발팀'}),
               (민수:Q0009 {name:'이민수', dept:'데이터팀'}),
               (지민:Q0009 {name:'최지민', dept:'데이터팀'}),
               (철수)-[:협업]->(영희),
               (영희)-[:협업]->(민수),
               (민수)-[:협업]->(지민)
    """)
    print("  사원 4명과 협업 관계를 만들었다:")
    print("    김철수 → 박영희 → 이민수 → 최지민")

    print("\n  Cypher 문법의 핵심은 이 화살표다:")
    print("    (a)-[:협업]->(b)     a가 b와 협업했다")
    print("    (  )  노드          [  ]  관계          ->  방향")
    print("  실제로 관계를 그린 모양 그대로 쓴다. 이게 SQL과 가장 다른 점이다.")

    rows = g.query("""
        MATCH (p:Q0009)-[:협업]->(other)
        WHERE p.dept = '개발팀'
        RETURN p.name AS 사람, other.name AS 협업상대
    """)
    print("\n  질의: 개발팀 사람이 협업한 상대는?")
    print("    MATCH (p:Q0009)-[:협업]->(other) WHERE p.dept = '개발팀' RETURN ...")
    for r in rows:
        print(f"      · {r['사람']} → {r['협업상대']}")

    print("\n  💡 SQL과 대응시키면:")
    print("     MATCH  ≈ FROM + JOIN      WHERE ≈ WHERE      RETURN ≈ SELECT")


# --- 2) SQL이 못 하는 것 ------------------------------------------------------
def demo_variable_hops(g):
    section(2, "SQL과의 결정적 차이, '몇 다리 건널지 모를 때'")

    print("\n  질문: 김철수와 최지민은 어떻게 연결돼 있나? (몇 다리인지 모른다)")
    rows = g.query("""
        MATCH path = (a:Q0009 {name:'김철수'})-[:협업*1..5]-(b:Q0009 {name:'최지민'})
        RETURN [n IN nodes(path) | n.name] AS 경로, length(path) AS 다리수
        ORDER BY 다리수 LIMIT 1
    """)
    for r in rows:
        print(f"    → {' → '.join(r['경로'])}  ({r['다리수']}다리)")

    print("\n  핵심은 `*1..5` 다. '1~5다리 사이 어디든' 이라는 뜻이다.")
    print("""
  SQL로 같은 걸 하려면:
    SELECT ... FROM emp e1 JOIN collab c1 ... JOIN emp e2 ON ...   -- 1다리
    UNION SELECT ... JOIN ... JOIN ... JOIN ...                    -- 2다리
    UNION SELECT ... (3다리) ... (4다리) ... (5다리)
    → 몇 다리인지 모르니 JOIN을 몇 번 걸지 미리 못 정한다.""")
    print("  이게 그래프 DB를 쓰는 이유다 (→ Q0004에서 networkx로 봤던 그 문제).")


# --- 3) 프로시저 vs 함수 -----------------------------------------------------
def demo_procedure_vs_function(g):
    section(3, "프로시저(CALL) vs 함수(RETURN 안에서 사용)")

    version = g.query("RETURN apoc.version() AS v")[0]["v"]
    print(f"\n  함수  : RETURN apoc.version()        → {version}")
    print("          값 하나를 돌려준다. 계산식 안에 끼워 쓴다. (SQL의 UPPER(), NOW() 같은 것)")

    meta = g.query("CALL apoc.meta.data() YIELD label, property RETURN label, property LIMIT 4")
    print(f"\n  프로시저: CALL apoc.meta.data() YIELD ...  → {len(meta)}행 (여러 줄을 쏟아낸다)")
    for m in meta:
        print(f"            label={m['label']}  property={m['property']}")
    print("\n  💡 차이: 함수는 '값 하나', 프로시저는 '행들의 스트림'.")
    print("     그래서 프로시저는 CALL로 부르고 YIELD로 컬럼을 받아야 한다.")
    print("     LangChain의 Neo4jGraph가 스키마를 파악할 때 쓰는 게 이 apoc.meta.data 다.")


# --- 4) APOC은 서버가 아니라 라이브러리 ---------------------------------------
def demo_apoc_is_a_library(g):
    section(4, "APOC은 '서버'가 아니라 Neo4j 안에 끼워 넣은 라이브러리")

    n_apoc = g.query("SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' RETURN count(*) AS c")[0]["c"]
    n_all = g.query("SHOW PROCEDURES YIELD name RETURN count(*) AS c")[0]["c"]
    n_func = g.query("SHOW FUNCTIONS YIELD name WHERE name STARTS WITH 'apoc' RETURN count(*) AS c")[0]["c"]

    print(f"\n  이 서버에 등록된 프로시저 {n_all}개 중 APOC이 {n_apoc}개, APOC 함수는 {n_func}개")
    print("  → 별도 서버에 접속해서 쓰는 게 아니라, Neo4j 프로세스 '안'에 등록돼 있다.")

    plugins = Path("/opt/homebrew/opt/neo4j/libexec/plugins")
    if plugins.exists():
        jars = [p.name for p in plugins.iterdir() if p.suffix == ".jar"]
        print(f"\n  실체는 jar 파일 하나다: plugins/{jars[0] if jars else '(없음)'}")
    print("  설치 = 그 jar를 plugins/ 에 두고 Neo4j를 재시작하는 것. 그게 전부다.")
    print("\n  💡 DB 프로세스 안에서 자바 코드가 도는 셈이라, Neo4j는 기본적으로 이걸")
    print("     샌드박스에 가둔다. 그래서 neo4j.conf 에 아래를 넣어 신뢰를 표시해야 했다:")
    print("       dbms.security.procedures.unrestricted=apoc.*")
    print("       dbms.security.procedures.allowlist=apoc.*")


def main():
    print("Q0009 · Cypher(질의 언어)와 APOC(플러그인 라이브러리)")
    try:
        g = get_neo4j_graph()
    except Exception as e:
        print(f"\n❌ Neo4j 연결 실패: {e}")
        print("   brew services start neo4j 로 서버를 먼저 띄우세요.")
        return

    try:
        demo_basics(g)
        demo_variable_hops(g)
        demo_procedure_vs_function(g)
        demo_apoc_is_a_library(g)
    finally:
        cleanup(g)   # 이 데모가 만든 :Q0009 노드만 정리
        print(f"\n{'=' * 64}")
        print("정리")
        print("  · Cypher = 그래프용 질의 언어 (SQL의 자리). 화살표로 관계를 그린다.")
        print("  · APOC   = Neo4j에 끼워 넣는 jar 라이브러리. 별도 서버가 아니다.")
        print("  · 프로시저(CALL)는 행 스트림, 함수(RETURN 안)는 값 하나.")
        print("  · APOC이 없으면 LangChain의 Neo4jGraph가 스키마를 못 읽는다.")
        print("  (데모 노드는 정리했습니다)")


if __name__ == "__main__":
    main()
