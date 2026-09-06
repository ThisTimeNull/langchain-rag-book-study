"""
Q0004 · 데이터베이스는 종류별로 언제 써야 하나?

핵심 시연: "좋은 DB / 나쁜 DB"는 없다. **질문의 모양**이 DB를 고른다.
같은 회사 데이터(사원 6명)를 4가지 방식으로 저장해두고,
각 DB가 '쉽게 답하는 질문'과 '거의 못 답하는 질문'을 나란히 보여준다.

  1) 정형 DB (SQLite)   → 집계, 정확한 조건  : "개발팀 평균 연차는?"
  2) 문서 DB (JSON)     → 제각각인 스키마    : "사원마다 다른 부가정보"
  3) 그래프 DB (networkx) → 관계를 따라가기  : "철수와 지민을 잇는 협업 경로는?"
  4) 벡터 DB (FAISS)    → 의미로 찾기        : "클라우드 인프라 잘 아는 사람?"

실행: python code/Q0004_db_selection.py
(API 키 불필요. 4)만 로컬 임베딩 모델을 쓰며, 불러오지 못하면 자동으로 건너뛴다)
"""

import sys
import json
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- 원본 데이터 (이 하나를 4가지 DB에 나눠 담는다) -------------------------
EMPLOYEES = [
    # (이름, 부서, 연차, 자기소개 문장)
    ("김철수", "개발팀", 7, "사내 서버 배포와 컨테이너 운영을 담당합니다. 장애 대응 경험이 많습니다."),
    ("박영희", "개발팀", 3, "웹 프론트엔드를 만들고 화면 성능을 개선합니다."),
    ("이민수", "데이터팀", 5, "추천 모델을 학습시키고 지표를 분석합니다."),
    ("최지민", "데이터팀", 2, "쿠버네티스 위에서 학습 파이프라인을 돌립니다."),
    ("정하늘", "인사팀", 9, "채용과 사내 교육 프로그램을 운영합니다."),
    ("한서준", "개발팀", 1, "결제 API 서버를 개발하고 테스트를 작성합니다."),
]

# 같이 프로젝트를 한 적 있는 사이 (그래프의 '간선')
COLLABORATIONS = [
    ("김철수", "박영희"),
    ("박영희", "이민수"),
    ("이민수", "최지민"),
    ("정하늘", "한서준"),
    ("한서준", "김철수"),
]


def section(n, title, why):
    print(f"\n{'=' * 62}\n[{n}] {title}\n     쓰는 이유: {why}\n{'=' * 62}")


# --- 1) 정형 DB : 행과 열, 집계 --------------------------------------------
def demo_relational():
    section(1, "정형 DB (RDB / SQLite)", "숫자를 세고, 합치고, 정확히 거르는 질문")

    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE emp (name TEXT, dept TEXT, years INT, bio TEXT)")
    con.executemany("INSERT INTO emp VALUES (?,?,?,?)", EMPLOYEES)

    print("\n  질문: '부서별 인원수와 평균 연차는?'")
    rows = con.execute(
        "SELECT dept, COUNT(*), ROUND(AVG(years),1) FROM emp GROUP BY dept ORDER BY 2 DESC"
    ).fetchall()
    for dept, cnt, avg in rows:
        print(f"    • {dept:6s} {cnt}명, 평균 {avg}년")

    print("\n  ✅ 잘하는 것 : 집계(COUNT/AVG/SUM), 정확한 조건, 트랜잭션(돈 계산)")
    print("  ❌ 못하는 것 : '클라우드 잘 아는 사람' 같은 의미 기반 검색")
    print("               → 아래 4)에서 LIKE 검색이 어떻게 실패하는지 확인")
    con.close()


# --- 2) 문서 DB : 스키마가 제각각일 때 --------------------------------------
def demo_document():
    section(2, "문서 DB (MongoDB류 / 여기선 JSON)", "레코드마다 필드가 달라지는 데이터")

    # 사원마다 가진 부가정보가 완전히 다르다, 표로 만들면 빈칸(NULL) 투성이가 된다.
    docs = [
        {"name": "김철수", "certs": ["AWS SA", "CKA"], "on_call": True},
        {"name": "박영희", "portfolio": {"url": "http://ex.com", "works": 12}},
        {"name": "정하늘", "languages": ["영어", "일본어"], "mentees": 4},
    ]
    print("\n  같은 '사원'인데 가진 필드가 전부 다르다:")
    for d in docs:
        print(f"    • {json.dumps(d, ensure_ascii=False)}")

    print("\n  질문: '자격증을 가진 사원은?' → 필드가 있는 문서만 골라내면 끝")
    print(f"    → {[d['name'] for d in docs if 'certs' in d]}")

    print("\n  ✅ 잘하는 것 : 스키마 변경이 잦은 데이터, 중첩 구조를 통째로 저장")
    print("  ❌ 못하는 것 : 여러 컬렉션을 넘나드는 복잡한 JOIN, 집계 (정형 DB가 유리)")


# --- 3) 그래프 DB : 관계를 몇 다리 건너 따라가기 -----------------------------
def demo_graph():
    section(3, "그래프 DB (Neo4j류 / 여기선 networkx)", "'관계를 몇 다리 건너' 따라가는 질문")

    import networkx as nx

    g = nx.Graph()
    g.add_edges_from(COLLABORATIONS)

    print("\n  질문: '김철수와 최지민은 어떻게 연결돼 있나?' (협업 인맥 경로)")
    path = nx.shortest_path(g, "김철수", "최지민")
    print(f"    → {' → '.join(path)}  ({len(path) - 1}다리)")

    print("\n  질문: '김철수와 함께 일한 적은 없지만, 한 다리 건너 아는 사람은?' (추천)")
    direct = set(g.neighbors("김철수"))
    second = {p for d in direct for p in g.neighbors(d)} - direct - {"김철수"}
    print(f"    → {sorted(second)}")

    print("\n  ✅ 잘하는 것 : 경로 탐색, N다리 건너 추천, 사기 탐지처럼 '연결'이 핵심인 질문")
    print("  ❌ 못하는 것 : SQL이면 한 줄인 단순 집계 (굳이 그래프로 갈 이유 없음)")
    print("  💡 SQL로 하면 '몇 다리'를 모르니 셀프 JOIN을 몇 번 걸지 미리 못 정한다 → 이게 그래프 DB의 존재 이유")


# --- 4) 벡터 DB : 단어가 안 겹쳐도 '의미'로 찾기 ------------------------------
def demo_vector():
    section(4, "벡터 DB (FAISS / Chroma / Neo4j 벡터인덱스)", "단어가 안 겹쳐도 '뜻'으로 찾는 질문")

    query = "클라우드 인프라를 잘 다루는 사람?"
    print(f"\n  질문: '{query}'")

    # (a) 정형 DB의 키워드 검색으로 시도 → 실패
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE emp (name TEXT, bio TEXT)")
    con.executemany("INSERT INTO emp VALUES (?,?)", [(e[0], e[3]) for e in EMPLOYEES])
    like = con.execute("SELECT name FROM emp WHERE bio LIKE '%클라우드%'").fetchall()
    con.close()
    print(f"    [정형 DB] LIKE '%클라우드%' 검색 결과 → {like or '없음 (0건)'}")
    print("      ↑ 아무도 '클라우드'라는 단어를 쓰지 않았다. 키워드 검색은 여기서 끝난다.")

    # (b) 벡터 검색 → 의미가 가까운 문장을 찾아낸다
    try:
        from langchain_community.vectorstores import FAISS
        from config.settings import get_local_embeddings

        print("\n    [벡터 DB] 문장을 숫자 벡터로 바꿔 '의미가 가까운' 순으로 검색 …")
        # 한국어 문장이므로 다국어 임베딩 모델을 쓴다.
        # (영어 전용 all-MiniLM-L6-v2로는 한국어 의미를 제대로 못 잡아 엉뚱한 사람이 뽑힌다, 직접 바꿔보면 체감된다)
        embeddings = get_local_embeddings(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        store = FAISS.from_texts(
            [e[3] for e in EMPLOYEES],
            embeddings,
            metadatas=[{"name": e[0]} for e in EMPLOYEES],
        )
        # score = 거리 → 작을수록 의미가 가깝다
        for d, score in store.similarity_search_with_score(query, k=2):
            print(f"      • [거리 {score:5.1f}] {d.metadata['name']}: {d.page_content}")
        print("      ↑ '서버 배포, 컨테이너', '쿠버네티스' = 클라우드 인프라라는 걸 '의미'로 잡아냈다.")
        print("        (질문에 없던 단어인데도 찾아낸다, 이게 RAG의 Retrieve 단계다)")
    except Exception as e:  # 모델 다운로드 실패 등
        print(f"\n    [벡터 DB] 로컬 임베딩을 불러오지 못해 생략: {type(e).__name__}")

    print("\n  ✅ 잘하는 것 : 의미 검색, RAG의 문서 검색(Retrieve)")
    print("  ❌ 못하는 것 : 정확한 집계, 조건 필터, '항상 정답 1개'가 필요한 질문")


def main():
    print("Q0004 · 같은 데이터, 4가지 DB, 질문의 모양이 DB를 고른다")
    demo_relational()
    demo_document()
    demo_graph()
    demo_vector()

    print(f"\n{'=' * 62}")
    print("결론: 데이터의 '모양'이 아니라 **던질 질문의 모양**으로 고른다.")
    print("  세고 합친다        → 정형 DB")
    print("  필드가 제각각      → 문서 DB")
    print("  관계를 따라간다    → 그래프 DB")
    print("  뜻으로 찾는다      → 벡터 DB")
    print("  ※ 실무는 하나만 쓰지 않는다. RAG도 보통 벡터 DB + 정형/그래프 DB 조합.")


if __name__ == "__main__":
    main()
