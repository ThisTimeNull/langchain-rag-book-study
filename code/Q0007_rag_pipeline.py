"""
Q0007 · RAG 파이프라인을 처음부터 끝까지 직접 만들어보자

지금까지는 부품을 하나씩 봤다 (왜 RAG인가 → 랭체인 → DB 선택 → FAISS → LCEL).
이 파일에서 그 부품들을 **하나의 파이프라인으로 잇는다.**

    [1] 로드    문서 파일을 읽는다            (data/sample/*.md)
    [2] 청킹    긴 문서를 조각낸다            (RecursiveCharacterTextSplitter)
    [3] 임베딩  조각을 숫자 벡터로 바꾼다     (로컬 다국어 모델)
    [4] 저장    벡터를 인덱스에 넣는다        (FAISS)
    [5] 검색    질문과 가까운 조각을 찾는다   (retriever)
    [6] 생성    근거를 붙여 LLM에게 묻는다    (LCEL 체인)

여기까지가 RAG의 전부다. 이후 질문들은 전부 이 6단계 중 하나를 '튜닝'하는 이야기다.

실행: python code/Q0007_rag_pipeline.py
      python code/Q0007_rag_pipeline.py "직접 물어볼 질문"
(API 키 없으면 6단계에서 'LLM에 실제로 들어갈 프롬프트'를 대신 보여준다)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from config.settings import get_local_embeddings, DATA_DIR

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE, CHUNK_OVERLAP = 200, 40

# 인덱스 폴더 이름에 설정을 박아둔다 → 설정을 바꾸면 자동으로 새 인덱스를 만든다.
# (안 그러면 chunk_size를 바꿔도 예전 인덱스를 그대로 읽어 '왜 결과가 그대로지?' 하게 된다)
INDEX_DIR = DATA_DIR / f"index_q0007_c{CHUNK_SIZE}_o{CHUNK_OVERLAP}"

DEFAULT_QUESTION = "연차를 쓰려면 며칠 전에 신청해야 하나요?"


def step(n, title):
    print(f"\n{'─' * 62}\n[{n}] {title}\n{'─' * 62}")


# --- [1] 로드 ---------------------------------------------------------------
def load_documents():
    step(1, "로드, 문서 파일을 읽는다")

    docs = DirectoryLoader(
        str(DATA_DIR / "sample"),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()

    for d in docs:
        name = Path(d.metadata["source"]).name
        print(f"  · {name:22s} {len(d.page_content):>5,}자")
    print(f"\n  💡 loader는 파일을 Document(page_content + metadata)로 바꿔준다.")
    print("     metadata의 source가 나중에 '출처 표시'의 근거가 된다.")
    print("     PDF, CSV, 웹페이지, Notion 등 로더만 갈아끼우면 같은 파이프라인이 돈다.")
    return docs


# --- [2] 청킹 ---------------------------------------------------------------
def split_documents(docs):
    step(2, "청킹, 긴 문서를 조각낸다")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # 조각 하나의 최대 길이(글자)
        chunk_overlap=CHUNK_OVERLAP,  # 앞 조각의 끝 40자를 다음 조각이 겹쳐 갖는다
        separators=["\n## ", "\n- ", "\n", " ", ""],  # 위에서부터 시도: 의미 단위로 자르기
    )
    chunks = splitter.split_documents(docs)

    print(f"  문서 {len(docs)}개 → 조각 {len(chunks)}개")
    print(f"\n  첫 조각 미리보기:")
    print(f"    ┌{'─' * 56}")
    for line in chunks[0].page_content.splitlines()[:4]:
        print(f"    │ {line}")
    print(f"    └{'─' * 56}")

    print("\n  💡 왜 자르나?")
    print("     ① 문서 전체를 LLM에 넣으면 비싸고, 길면 넣지도 못한다")
    print("     ② 검색 단위 = 조각이라, 너무 크면 관련 없는 내용이 섞이고")
    print("        너무 작으면 문맥이 끊긴다 → chunk_size가 RAG 품질의 첫 번째 손잡이")
    print("     overlap은 문장이 조각 경계에서 잘려 의미를 잃는 걸 막아준다.")
    return chunks


# --- [3][4] 임베딩 + 저장 ----------------------------------------------------
def build_index(chunks):
    step("3+4", "임베딩 & 저장, 조각을 벡터로 바꿔 FAISS에 넣는다")

    embeddings = get_local_embeddings(EMBED_MODEL)

    if INDEX_DIR.exists():
        store = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        print(f"  기존 인덱스 재사용: {INDEX_DIR.name}/ (벡터 {store.index.ntotal}개)")
        print("  💡 임베딩 계산은 느리고(=비용) 문서가 안 바뀌면 결과가 같다 → 저장해두고 재사용한다.")
    else:
        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(str(INDEX_DIR))
        files = ", ".join(p.name for p in sorted(INDEX_DIR.iterdir()))
        print(f"  조각 {len(chunks)}개 → 벡터 {store.index.ntotal}개 생성 후 저장")
        print(f"  저장 위치: {INDEX_DIR.name}/ ({files})")
        print("  💡 .faiss = 벡터, .pkl = 원문+메타데이터 (→ Q0005: FAISS는 원문을 저장하지 않는다)")

    sample_vec = embeddings.embed_query("연차")
    print(f"\n  벡터 한 개의 모습: {len(sample_vec)}차원, 앞 5개 = "
          f"[{', '.join(f'{v:+.3f}' for v in sample_vec[:5])}, ...]")
    return store


# --- [5] 검색 ---------------------------------------------------------------
def make_retriever(store, k=3):
    step(5, f"검색, 질문과 가까운 조각 {k}개를 찾는다")

    retriever = store.as_retriever(search_kwargs={"k": k})
    print(f"  retriever = store.as_retriever(search_kwargs={{'k': {k}}})")
    print("\n  💡 k = 몇 조각을 가져올지. 작으면 근거가 부족하고,")
    print("     크면 관련 없는 조각이 섞여 LLM이 헷갈린다 → 두 번째 손잡이")
    return retriever


# --- [6] 생성 ---------------------------------------------------------------
def build_chain(retriever):
    step(6, "생성, 근거를 붙여 LLM에게 묻는다 (LCEL 체인)")

    prompt = ChatPromptTemplate.from_template(
        "당신은 사내 규정 안내 담당자입니다.\n"
        "아래 [근거]에 있는 내용만 사용해 한국어로 간결하게 답하세요.\n"
        "근거에 없는 내용은 절대 지어내지 말고 '문서에서 찾을 수 없습니다'라고 답하세요.\n\n"
        "[근거]\n{context}\n\n"
        "[질문] {question}\n"
        "[답변]"
    )

    def format_docs(docs):
        # 조각들을 하나의 문자열로. 출처를 같이 넣어야 LLM이 출처를 인용할 수 있다.
        return "\n\n".join(
            f"({i}) [{Path(d.metadata['source']).name}]\n{d.page_content}"
            for i, d in enumerate(docs, 1)
        )

    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),  # 질문 → 검색 → 문자열
            question=RunnableLambda(lambda q: q),             # 질문은 그대로 통과
        )
        | prompt
        | RunnableLambda(lambda p: p.to_string())   # 아래에서 프롬프트를 눈으로 보려고 잠깐 끊는다
    )

    print("  chain = {context: retriever | format, question: 통과} | prompt | model | parser")
    print("\n  💡 프롬프트에 '근거에 없으면 지어내지 마라'를 넣는 게 중요하다.")
    print("     RAG의 목적은 환각(hallucination)을 줄이는 것이므로 (→ Q0001)")
    return chain, prompt


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    print("Q0007 · RAG 파이프라인 6단계를 처음부터 끝까지")

    docs = load_documents()
    chunks = split_documents(docs)
    store = build_index(chunks)
    retriever = make_retriever(store, k=3)
    prompt_chain, prompt = build_chain(retriever)

    # ---- 실제로 돌려보기 ----
    print(f"\n{'=' * 62}\n실행: \"{question}\"\n{'=' * 62}")

    hits = retriever.invoke(question)
    print(f"\n검색된 근거 {len(hits)}개:")
    for i, d in enumerate(hits, 1):
        src = Path(d.metadata["source"]).name
        head = d.page_content.replace("\n", " ")[:70]
        print(f"  {i}. [{src}] {head}…")

    try:
        from config.settings import get_openai_llm

        chain = prompt_chain | get_openai_llm() | StrOutputParser()
        print(f"\n답변: {chain.invoke(question)}")
        print(f"출처: {', '.join(sorted({Path(d.metadata['source']).name for d in hits}))}")
    except Exception:
        print("\n(OPENAI_API_KEY 없음 → LLM 호출 대신, 실제로 LLM에 들어갈 프롬프트를 보여준다)")
        print(f"\n{'┄' * 62}")
        print(prompt_chain.invoke(question))
        print(f"{'┄' * 62}")
        print("\n💡 RAG란 결국 '이 프롬프트를 자동으로 조립해주는 것'이다.")
        print("   질문만 던졌는데 관련 근거가 알아서 붙어 들어간다, 그게 전부다.")

    print(f"\n{'=' * 62}")
    print("정리, 이 6단계가 RAG의 전부다")
    print("  로드 → 청킹 → 임베딩 → 저장 → 검색 → 생성")
    print("  이후의 모든 개선은 이 중 한 단계를 손보는 일이다:")
    print("    · 청킹: chunk_size / overlap / 자르는 기준")
    print("    · 검색: k / 임베딩 모델 / 필터 / 재순위(rerank)")
    print("    · 생성: 프롬프트 / 모델")
    print(f"\n  다른 질문으로 실험: python code/Q0007_rag_pipeline.py \"금요일에 배포해도 되나요?\"")


if __name__ == "__main__":
    main()
