"""
Q0005 · FAISS란 무엇인가?

한 줄 정의: FAISS = **"벡터 수백만 개 중에서 질문 벡터와 가까운 k개를 빨리 찾아주는 라이브러리"**
(Facebook AI Similarity Search, Meta가 만듦)

이 코드가 보여주는 것
  1) FAISS가 하는 계산은 사실 numpy 몇 줄과 '똑같다' → 결과가 일치하는지 확인
  2) 그런데 규모가 커지면 속도가 갈린다 → 10만 개 벡터로 완전탐색 vs FAISS 비교
  3) 더 빠른 대신 정확도를 조금 포기하는 '근사 검색(IVF)' → 속도, 정확도 트레이드오프
  4) FAISS는 '서버'가 아니라 '라이브러리' → 파일로 저장하고, 원문 매핑은 직접 관리

⚠️ macOS 주의: 이 파일에서 임베딩 모델(torch)까지 같이 부르면 OpenMP 충돌로 죽는다.
   (faiss와 torch가 각자 libomp.dylib를 갖고 있어, faiss가 스레드를 쓴 뒤 torch가 로드되면 crash)
   그래서 여기서는 FAISS 자체 동작만 다루고, 임베딩과 함께 쓰는 예시는
   code/Q0001_why_rag.py · code/Q0004_db_selection.py 에 있다.

실행: python code/Q0005_faiss.py   (API 키 불필요)
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss

SEED = 42


def section(n, title):
    print(f"\n{'=' * 62}\n[{n}] {title}\n{'=' * 62}")


# --- 1) FAISS가 하는 일 = numpy 몇 줄과 같다 --------------------------------
def demo_same_as_numpy():
    section(1, "FAISS가 하는 계산은 numpy와 '똑같다'")

    # 문장 5개를 이미 임베딩했다고 치고, 4차원 벡터로 단순화
    docs = ["서버 배포", "쿠버네티스 운영", "프론트엔드 개발", "채용 면접", "결제 API"]
    vecs = np.array(
        [
            [0.9, 0.8, 0.1, 0.0],  # 서버 배포
            [0.8, 0.9, 0.0, 0.1],  # 쿠버네티스   ← 위와 의미가 가까움
            [0.1, 0.0, 0.9, 0.2],  # 프론트엔드
            [0.0, 0.1, 0.2, 0.9],  # 채용
            [0.3, 0.1, 0.7, 0.3],  # 결제 API
        ],
        dtype="float32",
    )
    # 주의: 질문 벡터가 두 문서의 정확히 중간이면 거리가 동점이라 순서가 갈린다.
    # (처음 이 코드는 [0.85, 0.85, ...]였고 거리가 둘 다 0.01로 같아 순서가 뒤집혔다)
    query = np.array([[0.88, 0.78, 0.05, 0.05]], dtype="float32")  # "서버 인프라 관련"

    # (a) numpy로 직접: 모든 벡터와 거리를 재고 정렬한다
    dists = ((vecs - query) ** 2).sum(axis=1)
    numpy_top2 = np.argsort(dists)[:2]

    # (b) FAISS로: 인덱스에 넣고 search
    index = faiss.IndexFlatL2(vecs.shape[1])  # L2 = 유클리드 거리, Flat = 전부 다 비교
    index.add(vecs)
    faiss_d, faiss_i = index.search(query, 2)

    print(f"\n  numpy 완전탐색 → {[docs[i] for i in numpy_top2]}")
    print(f"  FAISS  search  → {[docs[i] for i in faiss_i[0]]}  (거리 {faiss_d[0].round(3)})")
    print(f"  결과 일치: {list(numpy_top2) == list(faiss_i[0])}")
    print("\n  💡 즉 FAISS는 마법이 아니다. '거리 재서 가까운 순 정렬'을 대신 해줄 뿐.")
    print("     차이는 아래 2)처럼 데이터가 커졌을 때 드러난다.")


# --- 2) 규모가 커지면 속도가 갈린다 ------------------------------------------
def demo_scale():
    section(2, "10만 개 벡터에서 속도 비교, 그리고 '속도↔정확도' 손잡이")

    rng = np.random.default_rng(SEED)
    n, dim, k, nq = 100_000, 128, 5, 100

    # 실제 임베딩처럼 '의미가 비슷한 것끼리 뭉쳐 있는' 데이터를 만든다.
    # (완전 무작위로 만들면 고차원에서 모든 거리가 고만고만해져, '차원의 저주',
    #  근사 검색이 제 실력을 못 낸다)
    centers = rng.random((100, dim), dtype="float32")
    data = (centers[rng.integers(0, 100, n)] + rng.normal(0, 0.3, (n, dim))).astype("float32")
    # 질문 1개는 운에 좌우되므로 100개를 던져 '평균'으로 비교한다
    queries = (centers[rng.integers(0, 100, nq)] + rng.normal(0, 0.3, (nq, dim))).astype("float32")

    print(f"\n  데이터 {n:,}개 × {dim}차원 · 질문 {nq}개 평균 · 상위 {k}개 검색")

    # (a) numpy 완전탐색, 모든 벡터와 거리를 잰다
    t = time.perf_counter()
    for q in queries:
        np.argsort(((data - q) ** 2).sum(axis=1))[:k]
    numpy_ms = (time.perf_counter() - t) * 1000 / nq

    # (b) FAISS Flat, 역시 전부 비교하지만 SIMD로 최적화돼 있다 (정확도 100%)
    flat = faiss.IndexFlatL2(dim)
    flat.add(data)
    t = time.perf_counter()
    _, truth = flat.search(queries, k)   # 이 결과를 '정답'으로 삼는다
    flat_ms = (time.perf_counter() - t) * 1000 / nq

    # (c) FAISS IVF, 벡터를 미리 nlist개 구역으로 나눠두고, 가까운 몇 구역만 뒤진다 (근사)
    nlist = 1024   # 구역 개수. 경험칙: 대략 sqrt(전체 개수)
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist)
    ivf.train(data)   # 구역 나누기 = '학습' 단계가 필요하다 (Flat에는 없던 단계)
    ivf.add(data)

    def run_ivf(nprobe):
        ivf.nprobe = nprobe   # 몇 개 구역을 뒤질지 = 속도↔정확도 손잡이
        t = time.perf_counter()
        _, found = ivf.search(queries, k)
        ms = (time.perf_counter() - t) * 1000 / nq
        recall = np.mean([len(set(a) & set(b)) / k for a, b in zip(truth, found)])
        return ms, recall

    print(f"\n  {'방식':24s} {'질문당 시간':>12s}   {'정답을 몇 % 찾았나':>16s}")
    print(f"  {'numpy 완전탐색':22s} {numpy_ms:9.3f}ms   {'100% (기준)':>14s}")
    print(f"  {'FAISS Flat (정확)':21s} {flat_ms:9.3f}ms   {'100%':>14s}")
    for nprobe in (1, 5, 20):
        ms, recall = run_ivf(nprobe)
        print(f"  {f'FAISS IVF nprobe={nprobe}':21s} {ms:9.3f}ms   {recall:>13.0%}")

    print(f"\n  💡 Flat은 '전부 비교'라 항상 정확하다. 하지만 데이터가 늘면 그만큼 느려진다.")
    print(f"     IVF는 {nlist}개 구역 중 nprobe개만 뒤진다 → 훨씬 빠른 대신 놓치는 게 생긴다.")
    print("     nprobe를 올릴수록 정확도가 올라가고 속도는 떨어진다, 이 손잡이를 돌리는 게 튜닝이다.")
    print("     💡 데이터가 몇만 건 수준이면 그냥 Flat을 써라. 근사 검색은 그 이상에서 의미가 있다.")


# --- 3) FAISS는 '벡터만' 저장한다 → 원문 매핑은 내가 관리 -------------------
def demo_persist():
    section(3, "저장하고 다시 불러오기, 그리고 FAISS가 '저장하지 않는 것'")

    docs = [
        "환불은 구매 후 7일 이내에 가능합니다.",
        "재택근무는 주 2회까지 신청할 수 있습니다.",
        "사내 코드네임은 프로젝트 아틀라스입니다.",
    ]
    rng = np.random.default_rng(SEED)
    vecs = rng.random((len(docs), 8), dtype="float32")  # 임베딩했다고 가정

    index = faiss.IndexFlatL2(8)
    index.add(vecs)

    # FAISS는 파이썬 라이브러리다 → 서버가 아니라 '파일'로 저장한다
    out = Path(__file__).resolve().parent.parent / "data" / "q0005_demo.index"
    faiss.write_index(index, str(out))
    reloaded = faiss.read_index(str(out))

    print(f"\n  faiss.write_index() → {out.name} ({out.stat().st_size:,} bytes)")
    print(f"  faiss.read_index()  → 벡터 {reloaded.ntotal}개 복원")

    # ⚠️ 핵심: 검색 결과로 돌아오는 건 '문서'가 아니라 '몇 번째 벡터인지'(정수 id)뿐이다
    _, idx = reloaded.search(vecs[:1], 2)
    print(f"\n  search 결과 = {idx[0]}  ← 문장이 아니라 '번호'가 나온다")
    print(f"  번호 → 원문 매핑은 내가 따로 갖고 있어야 한다: docs[{idx[0][0]}] = '{docs[idx[0][0]]}'")

    print("\n  💡 FAISS는 원문도, 메타데이터도 저장하지 않는다. 오직 벡터와 번호뿐이다.")
    print("     그래서 LangChain의 FAISS 래퍼는 원문, 메타데이터를 담은 docstore를")
    print("     .pkl 파일로 '따로' 저장한다 (save_local 하면 .faiss + .pkl 두 개가 생기는 이유).")
    print("""
  # 실제 RAG에서는 이 래퍼를 쓴다 (원문 매핑을 대신 관리해준다)
  store = FAISS.from_texts(texts, embeddings)
  store.save_local("data/my_index")          # → index.faiss + index.pkl
  store = FAISS.load_local("data/my_index", embeddings,
                           allow_dangerous_deserialization=True)
  store.similarity_search("질문", k=3)        # → 문서 객체가 그대로 나온다

  → 임베딩까지 붙여 돌려보는 코드: code/Q0001_why_rag.py, code/Q0004_db_selection.py""")

    out.unlink()  # 데모 파일 정리


def main():
    print("Q0005 · FAISS = 벡터 수백만 개 중 '가까운 k개'를 빨리 찾아주는 라이브러리")
    demo_same_as_numpy()
    demo_scale()
    demo_persist()

    print(f"\n{'=' * 62}")
    print("정리")
    print("  · FAISS는 DB 서버가 아니라 **파이썬 라이브러리**다 (설치만 하면 끝, 서버 없음)")
    print("  · 하는 일은 단 하나: 벡터 N개 중 질문과 가까운 k개 찾기")
    print("  · 그래서 빠르고 가볍지만, 필터링, 동시 쓰기, 운영 기능은 약하다")
    print("  · 학습/프로토타입 = FAISS, 서비스 운영 = Chroma, Qdrant, pgvector 등을 고려")


if __name__ == "__main__":
    main()
