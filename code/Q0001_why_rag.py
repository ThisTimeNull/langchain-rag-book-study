"""
Q0001 · RAG는 왜 필요한가?

핵심 시연: LLM은 '학습하지 않은 사내/최신 정보'를 모른다.
RAG는 답하기 전에 관련 문서를 검색해 근거로 제공함으로써 이 문제를 해결한다.

이 코드는 가상의 사내 규정 문서(data/sample/company_policy.md)에서
질문과 관련된 문장을 '검색'해 보여준다. → RAG의 Retrieve 단계가 왜 필요한지 체감.

실행: python code/Q0001_why_rag.py
(API 키 없이 로컬 임베딩으로 동작. OPENAI_API_KEY가 있으면 LLM 답변까지 시연)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config.settings import get_local_embeddings, get_embeddings, DATA_DIR


QUESTION = "한빛소프트의 사내 코드네임은 뭐야?"


def build_retriever():
    doc = DATA_DIR / "sample" / "company_policy.md"
    docs = TextLoader(str(doc), encoding="utf-8").load()
    # 한 줄(사실 1개)을 한 청크로 → 검색이 또렷해진다.
    # (청크가 너무 크면 여러 사실이 섞여 엉뚱한 청크가 뽑힐 수 있다, 학습 포인트!)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=50, chunk_overlap=0, separators=["\n", " ", ""]
    ).split_documents(docs)

    # 임베딩: OpenAI 키 있으면 OpenAI, 없으면 무료 로컬 임베딩으로 폴백
    try:
        embeddings = get_embeddings()
        print("  임베딩: OpenAI")
    except RuntimeError:
        embeddings = get_local_embeddings()
        print("  임베딩: 로컬(all-MiniLM-L6-v2)")

    store = FAISS.from_documents(chunks, embeddings)
    return store.as_retriever(search_kwargs={"k": 2})


def main():
    print(f"질문: {QUESTION}\n")

    print("[A] RAG 없이, LLM은 이 사내 정보를 학습한 적이 없어 알 수 없다.")
    print("    → 모델은 모르거나, 그럴듯하게 지어낼 위험(환각)이 있다.\n")

    print("[B] RAG로, 먼저 사내 문서에서 관련 문장을 '검색'한다.")
    retriever = build_retriever()
    hits = retriever.invoke(QUESTION)
    print(f"\n  검색된 근거 {len(hits)}개:")
    for d in hits:
        print(f"    • {d.page_content.strip()}")

    # OPENAI_API_KEY가 설정돼 있으면 검색 근거로 실제 답변까지 생성
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from config.settings import get_openai_llm

        context = "\n".join(d.page_content for d in hits)
        prompt = ChatPromptTemplate.from_template(
            "아래 근거만 사용해 한국어로 한 문장으로 답하세요.\n근거:\n{context}\n\n질문: {q}"
        )
        chain = prompt | get_openai_llm() | StrOutputParser()
        print(f"\n  RAG 답변: {chain.invoke({'context': context, 'q': QUESTION})}")
    except RuntimeError:
        print("\n  (OPENAI_API_KEY 없음 → LLM 답변 생략. 검색 단계만 시연)")

    print("\n결론: RAG가 없으면 모델은 사내, 최신 정보를 모른다.")
    print("      검색으로 근거를 먼저 제공하면 정확한 답과 출처를 얻는다.")


if __name__ == "__main__":
    main()
