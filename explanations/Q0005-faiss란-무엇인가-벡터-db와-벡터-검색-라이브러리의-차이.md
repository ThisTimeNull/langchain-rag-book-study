# Q0005 · FAISS란 무엇인가? (벡터 DB와 벡터 검색 라이브러리의 차이)

- **태그**: vectorrag
- **날짜**: 2026-08-18
- **코드**: [`code/Q0005_faiss.py`](../code/Q0005_faiss.py)

## ❓ 질문
FAISS가 뭐더라? 벡터 DB라고 하던데, Chroma, Pinecone 같은 것들과는 뭐가 다른가?

## 💡 한 줄 답
> **FAISS = "벡터 수백만 개 중에서 질문 벡터와 가까운 k개를 빨리 찾아주는 라이브러리"**
> (Facebook AI Similarity Search, Meta가 만든 오픈소스)
> **DB 서버가 아니다.** `pip install faiss-cpu` 하면 끝나는 파이썬 라이브러리다.

## 📖 설명

### FAISS가 하는 일은 딱 하나다

```
입력:  질문 벡터 1개  +  저장된 벡터 100만 개
출력:  가까운 순서대로 k개의 '번호'와 '거리'
```

그게 전부다. 임베딩을 만들어주지도 않고, 원문을 저장하지도 않고, 답변을 생성하지도 않는다.
**"가까운 벡터 찾기"라는 한 가지 작업만** 극도로 빠르게 한다.

사실 이 계산은 numpy 두 줄로도 된다.

```python
dists = ((vecs - query) ** 2).sum(axis=1)   # 모든 벡터와의 거리
top_k = np.argsort(dists)[:k]               # 가까운 순 정렬
```

코드 1번 파트에서 실제로 FAISS와 numpy 결과가 **똑같이** 나오는 걸 확인한다.
FAISS는 마법이 아니다. 차이는 **규모**에서 벌어진다.

### 실측, 10만 개 벡터, 질문 100개 평균 (코드 2번 파트)

| 방식 | 질문당 시간 | 정답을 몇 % 찾았나 |
|---|---:|---:|
| numpy 완전탐색 | 12.945 ms | 100% (기준) |
| **FAISS Flat** (정확) | 0.453 ms | 100% |
| FAISS IVF `nprobe=1` | 0.005 ms | 27% |
| FAISS IVF `nprobe=5` | 0.011 ms | 78% |
| FAISS IVF `nprobe=20` | 0.021 ms | 100% |

두 가지를 읽어야 한다.

1. **같은 완전탐색인데도 FAISS Flat이 numpy보다 ~28배 빠르다.** C++ + SIMD로 최적화돼 있다.
2. **IVF는 정확도를 조금 내주고 속도를 크게 얻는다.** 미리 벡터를 1024개 구역으로 나눠두고
   질문과 가까운 `nprobe`개 구역만 뒤진다. 구역 1개만 보면 0.005ms에 27%,
   20개를 보면 0.021ms에 100%. **이 `nprobe` 손잡이를 돌리는 게 벡터 검색 튜닝이다.**

> 💡 단, 데이터가 몇만 건 수준이면 그냥 **Flat**을 써라. 위 표에서도 Flat이 0.45ms다.
> 근사 검색(IVF/HNSW)은 수십만~수백만 건부터 의미가 있다.

### 인덱스 종류, 뭘 고르나

| 인덱스 | 방식 | 언제 |
|---|---|---|
| **`IndexFlatL2`** | 전부 비교 (완전탐색) | 기본값. 수만 건 이하면 그냥 이것 |
| **`IndexIVFFlat`** | 구역으로 나눠 일부만 탐색 | 수십만 건 이상, 속도가 필요할 때 |
| **`IndexHNSWFlat`** | 그래프를 타고 이웃 탐색 | 대용량 + 높은 정확도 (메모리를 많이 씀) |
| **`IndexIVFPQ`** | 벡터를 압축해 저장 | 메모리가 부족할 때 (정확도 손해) |

`Flat`이 붙으면 "벡터를 압축 없이 그대로 저장한다"는 뜻이다.
IVF/HNSW 계열은 쓰기 전에 **`train()`** 단계가 필요하다 (구역을 나누려면 데이터를 먼저 봐야 하니까).

### ⭐ 핵심, FAISS는 '원문'을 저장하지 않는다

이게 라이브러리와 DB를 가르는 지점이다.

```python
_, idx = index.search(query, 2)
print(idx)   # → [[0 2]]   ← 문장이 아니라 '번호'가 나온다
```

번호 → 원문 매핑은 **내가 직접 갖고 있어야 한다.** 메타데이터도, 문서 ID도 저장 안 된다.
그래서 LangChain의 FAISS 래퍼는 원문, 메타데이터를 담은 `docstore`를 **따로** 관리하고,
`save_local()` 하면 파일이 **두 개** 생긴다.

```
data/my_index/
├── index.faiss   ← 벡터 (FAISS가 저장)
└── index.pkl     ← 원문 + 메타데이터 (LangChain이 저장)
```

`save_local`이 왜 폴더를 만드는지, `load_local`에 왜 `allow_dangerous_deserialization=True`가
필요한지(= pkl을 언피클링하므로 신뢰할 수 있는 파일만 열라는 뜻)가 여기서 설명된다.

### 그래서, 벡터 DB와 뭐가 다른가

| | **FAISS** (라이브러리) | **벡터 DB** (Chroma, Qdrant, pgvector, Pinecone) |
|---|---|---|
| 형태 | `pip install` 후 import | 서버 프로세스(또는 클라우드 서비스) |
| 저장 | 내가 파일로 저장/로드 | 알아서 영속화 |
| 원문, 메타데이터 | **저장 안 함** (직접 관리) | 문서와 함께 저장 |
| 조건 필터 | 거의 없음 (`부서='개발팀'`인 것만 검색 ✗) | 지원 |
| 추가, 삭제, 수정 | 삭제, 갱신이 번거로움 | 일반 DB처럼 CRUD |
| 여러 프로세스 동시 접근 | ✗ | ✓ |
| 속도 | **가장 빠름** | 충분히 빠름 (기능값을 치름) |

**정리하면 FAISS는 벡터 DB의 "검색 엔진 부분"만 떼어낸 것**이다.
실제로 여러 벡터 DB가 내부적으로 FAISS나 비슷한 알고리즘을 쓴다.

**선택 기준:**
- **학습, 프로토타입, 단일 프로세스 앱** → FAISS. 서버 띄울 필요 없이 가장 간단하다.
- **문서가 계속 바뀌거나, 조건 필터가 필요하거나, 여러 서버가 붙는다** → 벡터 DB로 간다.
- 이미 PostgreSQL을 쓰고 있다면 **pgvector**가 현실적인 첫 선택이다
  (→ [[Q0004-데이터베이스는-종류별로-언제-써야-하나-정형db-문서db-vectordb]]의 "하나만 쓰지 않는다" 얘기).

### ⚠️ macOS 실전 함정, OpenMP 충돌

이 코드를 처음 짤 때 실제로 겪은 문제다.

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

`faiss`와 `torch`(임베딩 모델이 쓴다)가 **각자 자기 `libomp.dylib`를 들고 있어서**,
FAISS가 병렬 연산을 실제로 수행한 뒤 torch가 로드되면 프로세스가 죽는다.
(단순 `import`만으로는 안 죽는다, FAISS가 OMP 스레드를 쓴 다음이 문제다.)

```
.venv/lib/python3.13/site-packages/torch/lib/libomp.dylib
.venv/lib/python3.13/site-packages/faiss/.dylibs/libomp.dylib
.venv/lib/python3.13/site-packages/sklearn/.dylibs/libomp.dylib   ← 3개나 있다
```

**대처**: 이 파일에서는 FAISS 자체 동작만 다루고 임베딩은 섞지 않았다.
임베딩 + FAISS를 함께 쓰는 예시는 [[Q0001-rag는-왜-필요한가]]·
[[Q0004-데이터베이스는-종류별로-언제-써야-하나-정형db-문서db-vectordb]]에 있고, 그쪽은 정상 동작한다.
(`KMP_DUPLICATE_LIB_OK=TRUE`로 넘길 수도 있지만 공식 문서가 "unsafe"라고 못 박은 우회책이다.)

## 🔗 코드와 연결

[`code/Q0005_faiss.py`](../code/Q0005_faiss.py), API 키 없이 돈다.

| 파트 | 보여주는 것 |
|---|---|
| 1 | FAISS 결과 == numpy 완전탐색 결과 → "마법이 아니다" |
| 2 | 10만 벡터에서 numpy vs Flat vs IVF(nprobe 1/5/20) 속도, 정확도 |
| 3 | `write_index`/`read_index`로 파일 저장, 그리고 검색 결과가 '번호'뿐이라는 사실 |

> 1번 파트에는 함정이 하나 심어져 있다. 처음엔 질문 벡터가 두 문서의 **정확히 중간**이라
> 거리가 동점이 나왔고, numpy와 FAISS의 정렬 순서가 갈렸다 (`결과 일치: False`).
> 동점일 때 순서는 보장되지 않는다, 검색 결과를 비교할 때 기억해둘 것.

## 📚 더 알아보기
- 선행: [[Q0004-데이터베이스는-종류별로-언제-써야-하나-정형db-문서db-vectordb]], 벡터 DB를 언제 쓰나
- 관련: [[Q0001-rag는-왜-필요한가]], FAISS를 실제 RAG 검색에 쓰는 코드
- 다음 질문 후보:
  - "임베딩은 어떻게 문장을 숫자로 바꾸나?" (FAISS 성능을 좌우하는 건 사실 임베딩이다)
  - "L2 거리 vs 코사인 유사도, 뭘 써야 하나?"
  - "문서가 계속 바뀌면 인덱스를 어떻게 갱신하나?"
