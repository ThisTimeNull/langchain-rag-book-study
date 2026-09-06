"""
Q0008 · RAG 품질은 어떻게 튜닝하나?

Q0007에서 파이프라인은 돌았지만 문제가 보였다.
"금요일에 배포해도 되나요?" → 1위는 정확했는데 2, 3위는 완전히 무관한 조각이었다.

이 파일은 **손잡이를 하나씩 돌려가며 실제로 측정한다.** 감이 아니라 숫자로.

  실험 1. chunk_size, 조각을 얼마나 크게 자를까
  실험 2. k         , 몇 조각을 가져올까
  실험 3. 임베딩 모델, 영어 전용 vs 다국어
  실험 4. 무관한 근거 걸러내기, score threshold

평가 방법: 정답이 어느 문서 어느 문장에 있는지 아는 질문 7개를 미리 만들어두고,
검색 결과에 그게 들어있는지 센다. (RAG 개발의 첫걸음은 이런 '평가셋'을 만드는 것이다)

실행: python code/Q0008_rag_tuning.py
(API 키 불필요, 검색 단계만 측정한다. 생성은 검색이 좋아야 좋아진다)
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config.settings import get_local_embeddings, DATA_DIR

MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENGLISH_ONLY = "sentence-transformers/all-MiniLM-L6-v2"

# --- 평가셋: 질문 / 정답이 있는 파일 / 정답 조각에 반드시 들어있는 문구 -------
EVAL = [
    ("연차를 쓰려면 며칠 전에 신청해야 하나요?", "hr_guide.md", "3일 전까지 신청"),
    ("금요일에 배포해도 되나요?", "dev_handbook.md", "금요일 배포는 원칙적으로 금지"),
    ("점심 식대는 얼마까지 지원되나요?", "company_policy.md", "12,000원"),
    ("코드를 병합하려면 승인이 몇 명 필요한가요?", "dev_handbook.md", "최소 1명의 승인"),
    ("자기계발비는 연간 얼마인가요?", "hr_guide.md", "연간 100만원"),
    ("원격 근무는 얼마나 쓸 수 있나요?", "company_policy.md", "분기당 최대 30일"),
    ("장애가 나면 몇 분 안에 공유해야 하나요?", "dev_handbook.md", "15분 이내"),
]

_docs_cache = None
_embed_cache = {}


def load_docs():
    global _docs_cache
    if _docs_cache is None:
        _docs_cache = DirectoryLoader(
            str(DATA_DIR / "sample"), glob="*.md",
            loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
        ).load()
    return _docs_cache


def get_embeddings(model):
    if model not in _embed_cache:
        # 모델 로딩 경고가 표 중간에 끼어들지 않도록 잠시 막는다
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _embed_cache[model] = get_local_embeddings(model)
    return _embed_cache[model]


def build_store(chunk_size, overlap, model=MULTILINGUAL):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n## ", "\n- ", "\n", " ", ""],
    ).split_documents(load_docs())
    return FAISS.from_documents(chunks, get_embeddings(model)), len(chunks)


def evaluate(store, k):
    """평가셋을 돌려 세 가지를 잰다.

    hit    : 정답 문구가 검색된 조각 안에 있었는가 (몇 %의 질문에서)
    rank   : 정답이 몇 등으로 나왔나 (1등이 최고, 못 찾으면 제외)
    noise  : 가져온 k개 중 정답 파일이 아닌 조각이 몇 개였나 (낮을수록 좋다)
    chars  : LLM에 들어갈 근거의 총 길이 (= 비용)
    """
    hits, ranks, noises, chars = 0, [], [], []
    for question, src_file, answer_text in EVAL:
        docs = store.similarity_search(question, k=k)
        found = None
        for i, d in enumerate(docs, 1):
            if answer_text in d.page_content:
                found = i
                break
        if found:
            hits += 1
            ranks.append(found)
        noises.append(sum(1 for d in docs if Path(d.metadata["source"]).name != src_file))
        chars.append(sum(len(d.page_content) for d in docs))
    n = len(EVAL)
    return {
        "hit": hits / n,
        "rank": sum(ranks) / len(ranks) if ranks else float("nan"),
        "noise": sum(noises) / n,
        "chars": sum(chars) / n,
    }


def row(label, r, k):
    rank = "  –  " if r["rank"] != r["rank"] else f"{r['rank']:.2f}"
    return (f"  {label:22s} {r['hit']:>6.0%}   {rank:>6s}   "
            f"{r['noise']:>5.1f}/{k}   {r['chars']:>6.0f}자")


def header(_=None):
    print(f"\n  {'설정':20s} {'정답 찾음':>8s} {'평균등수':>8s} {'무관조각':>9s} {'근거길이':>8s}")
    print(f"  {'─' * 58}")


def section(n, title):
    print(f"\n{'=' * 66}\n[{n}] {title}\n{'=' * 66}")


# --- 실험 1: chunk_size ------------------------------------------------------
def exp_chunk_size():
    section(1, "chunk_size, 조각을 얼마나 크게 자를까 (k=3 고정)")
    k = 3
    header(k)
    for size in (80, 150, 300, 600, 1200):
        store, n_chunks = build_store(size, int(size * 0.2))
        r = evaluate(store, k)
        print(row(f"chunk={size:<5d}({n_chunks:2d}조각)", r, k))
    print("\n  💡 작으면: 문맥이 끊겨 정답 문장만 덩그러니 → 답에 필요한 주변 정보가 빠진다")
    print("     크면  : 조각 하나에 여러 주제가 섞여 검색이 뭉개지고, 근거 길이(비용)가 커진다")
    print("     문서 성격에 따라 다르므로 '정답이 있는 평가셋'으로 직접 재보는 수밖에 없다.")


# --- 실험 2: k ---------------------------------------------------------------
def exp_k():
    section(2, "k, 몇 조각을 가져올까 (chunk=300 고정)")
    store, n_chunks = build_store(300, 60)
    print(f"\n  전체 조각 수: {n_chunks}개")
    header(None)
    for k in (1, 2, 3, 5):
        print(row(f"k={k}", evaluate(store, k), k))
    print("\n  💡 k를 올리면 정답을 놓칠 확률은 줄지만, 무관한 조각과 비용이 같이 늘어난다.")
    print("     전체 조각이 적을 때 k를 크게 잡으면 '억지로 채워진' 쓰레기 근거가 들어간다.")
    print("     → 정답을 다 찾는 선에서 가장 작은 k가 좋다.")


# --- 실험 3: 임베딩 모델 -----------------------------------------------------
def exp_embedding_model():
    section(3, "임베딩 모델, 영어 전용 vs 다국어 (chunk=300, k=3)")
    k = 3
    header(k)
    for label, model in (("다국어 MiniLM-L12", MULTILINGUAL), ("영어전용 MiniLM-L6", ENGLISH_ONLY)):
        store, _ = build_store(300, 60, model)
        print(row(label, evaluate(store, k), k))
    print("\n  💡 같은 파이프라인, 같은 설정인데 모델 하나로 결과가 갈린다.")
    print("     한국어 문서에 영어 전용 모델을 쓰면 '의미'를 제대로 못 잡는다.")
    print("     → 벡터 DB보다 **임베딩 모델 선택**이 RAG 품질에 더 크게 작용한다.")


# --- 실험 4: 무관한 근거 걸러내기 ---------------------------------------------
def exp_threshold():
    section(4, "무관한 근거 걸러내기, 거리 임계값(score threshold)")
    store, _ = build_store(300, 60)
    question = "금요일에 배포해도 되나요?"

    print(f"\n  질문: {question}")
    print(f"\n  k=3으로 그냥 가져오면 (거리: 작을수록 가까움)")
    scored = store.similarity_search_with_score(question, k=3)
    for i, (d, dist) in enumerate(scored, 1):
        src = Path(d.metadata["source"]).name
        snippet = " ".join(d.page_content.split())[:38]
        print(f"    {i}. 거리 {dist:5.2f}  [{src:18s}] {snippet}…")

    print("\n  → 1위와 나머지의 '거리'가 확 벌어진다. 이 간격을 이용해 잘라낼 수 있다.")
    # 임계값은 '절대 숫자'로 정할 수 없다, 임베딩 모델, 거리 척도마다 스케일이 다르다.
    # (여기서는 거리가 8~13인데, 코사인 유사도를 쓰면 0~1 범위가 된다)
    # 그래서 실측한 1위 거리를 기준으로 잡는 게 현실적이다.
    cut = scored[0][1] * 1.2
    kept = [(d, sc) for d, sc in scored if sc < cut]
    print(f"\n  1위 거리의 1.2배({cut:.2f})보다 가까운 것만 남기면 → {len(kept)}개")
    for d, sc in kept:
        snippet = " ".join(d.page_content.split())[:44]
        print(f"    · 거리 {sc:5.2f}  {snippet}…")

    print("\n  💡 LangChain에서는 retriever 단계에서 바로 걸 수 있다:")
    print("       store.as_retriever(search_type='similarity_score_threshold',")
    print("                          search_kwargs={'score_threshold': 0.5, 'k': 5})")
    print("     (주의: score_threshold는 '유사도 0~1' 기준이라 거리와 방향이 반대다)")
    print("     다른 방법: search_type='mmr', 비슷한 조각끼리 중복되지 않게 다양성을 섞어 뽑는다.")


def main():
    print("Q0008 · 손잡이를 돌려가며 RAG 검색 품질을 '측정'한다")
    print(f"평가셋: 정답을 아는 질문 {len(EVAL)}개 / 문서 3개")

    exp_chunk_size()
    exp_k()
    exp_embedding_model()
    exp_threshold()

    print(f"\n{'=' * 66}")
    print("정리, 튜닝 순서")
    print("  1. 평가셋부터 만든다 (질문 + 정답 위치). 이게 없으면 튜닝이 아니라 느낌이다.")
    print("  2. 임베딩 모델을 먼저 고른다, 영향이 가장 크다 (실험 3)")
    print("  3. chunk_size를 문서 성격에 맞춘다 (실험 1)")
    print("  4. 정답을 다 찾는 선에서 k를 최소로 (실험 2)")
    print("  5. 그래도 노이즈가 섞이면 threshold, MMR, rerank로 거른다 (실험 4)")


if __name__ == "__main__":
    main()
