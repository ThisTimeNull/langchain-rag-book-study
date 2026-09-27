"""
Q0019 · 진짜 PDF로 대화형 RAG를 만들면 어디서 깨지나?

지금까지 쓴 문서는 전부 직접 만든 짧은 마크다운이었다. 이번엔 공개 논문 PDF 세 편이다.

  data/pdf/attention.pdf   Attention Is All You Need (2017)     Q0018의 트랜스포머 원조
  data/pdf/rag.pdf         Retrieval-Augmented Generation (2020)  RAG라는 이름의 원조
  data/pdf/graphrag.pdf    From Local to Global (2024)           Q0015 전역 요약의 원조

그리고 처음으로 '대화'를 한다. 앞 질문을 기억하고 이어받는다.

  사용자: GraphRAG 논문은 뭘 주장해?
  사용자: 그걸 실험으로 어떻게 증명했어?        <- '그걸'이 뭔지 알아야 한다

두 가지 방식으로 쓸 수 있다.
  python code/Q0019_pdf_chat.py                 직접 입력 (빈 줄이나 quit 로 종료)
  python code/Q0019_pdf_chat.py --script        미리 적어둔 질문 목록을 순서대로
  python code/Q0019_pdf_chat.py --rebuild       PDF를 다시 읽어 인덱스 재생성
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import get_azure_llm, get_azure_embeddings, DATA_DIR

PDF_DIR = DATA_DIR / "pdf"
INDEX_DIR = DATA_DIR / "index_q0019_pdf"

# --script 모드에서 순서대로 던질 질문. 뒤 질문은 앞 질문 없이는 뜻이 통하지 않게 짰다.
SCRIPT = [
    "GraphRAG 논문의 핵심 주장이 뭐야?",
    "그걸 실험으로 어떻게 증명했어?",
    "그 실험에서 비교 대상이 된 방식은 뭐였어?",
    "트랜스포머 논문에서 어텐션을 설명하는 수식이 뭐야?",
    "그 논문 저자는 몇 명이야?",
    "RAG 논문은 몇 년도에 나왔고 누가 썼어?",
    "세 논문 중에 가장 먼저 나온 건?",
]

REWRITE = ChatPromptTemplate.from_template(
    "대화 기록을 보고, 마지막 질문을 혼자서도 뜻이 통하는 완전한 질문으로 바꾸세요.\n"
    "'그것', '그 논문', '거기서' 같은 대명사만 실제 대상으로 바꾸세요.\n"
    "질문에 이미 적힌 고유명사(RAG, GraphRAG, 트랜스포머 등)는 절대 다른 것으로 바꾸지 마세요.\n"
    "'세 논문', '둘 다'처럼 개수를 말한 부분은 그대로 두세요. 질문만 출력하세요.\n\n"
    "[대화 기록]\n{history}\n\n[마지막 질문] {question}\n[완전한 질문]"
)

ANSWER = ChatPromptTemplate.from_template(
    "아래 근거만 써서 한국어로 답하세요. 근거에 없으면 '문서에서 찾을 수 없습니다'라고 답하세요.\n"
    "답 끝에 어느 논문 몇 쪽을 참고했는지 [파일명 p.쪽] 형식으로 적으세요.\n\n"
    "[근거]\n{context}\n\n[질문] {question}\n[답변]"
)


def build_index(rebuild):
    emb = get_azure_embeddings()
    if INDEX_DIR.exists() and not rebuild:
        return FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)

    docs = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        pages = PyPDFLoader(str(pdf)).load()
        for p in pages:
            p.metadata["file"] = pdf.stem
            p.metadata["page"] = p.metadata.get("page", 0) + 1   # 0부터 세는 걸 1부터로
        docs.extend(pages)
        print(f"  {pdf.name:16s} {len(pages)}쪽")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(docs)
    print(f"  전체 {len(docs)}쪽 -> 조각 {len(chunks)}개")

    store = FAISS.from_documents(chunks, emb)
    store.save_local(str(INDEX_DIR))
    return store


class Chat:
    """대화 기록을 들고 있는 RAG. 질문마다 기록을 보고 질문을 다시 쓴 뒤 검색한다."""

    def __init__(self, store, llm, k=5):
        self.retriever = store.as_retriever(search_kwargs={"k": k})
        self.rewrite = REWRITE | llm | StrOutputParser()
        self.answer = ANSWER | llm | StrOutputParser()
        self.history = []   # (질문, 답) 쌍

    def ask(self, question):
        # 1) 앞 대화가 있으면 질문을 완전한 문장으로 다시 쓴다
        if self.history:
            hist = "\n".join(f"사용자: {q}\n답변: {a[:200]}" for q, a in self.history[-3:])
            full_q = self.rewrite.invoke({"history": hist, "question": question}).strip()
        else:
            full_q = question

        # 2) 다시 쓴 질문으로 검색한다 (원래 질문이 아니라)
        hits = self.retriever.invoke(full_q)
        ctx = "\n\n".join(
            f"[{d.metadata['file']} p.{d.metadata['page']}]\n{d.page_content}" for d in hits)

        # 3) 답한다
        ans = self.answer.invoke({"context": ctx, "question": full_q}).strip()
        self.history.append((question, ans))
        srcs = sorted({f"{d.metadata['file']} p.{d.metadata['page']}" for d in hits})
        return full_q, ans, srcs


def main():
    rebuild = "--rebuild" in sys.argv
    scripted = "--script" in sys.argv
    print("Q0019 · 논문 PDF 세 편과 대화하기\n")

    store = build_index(rebuild)
    chat = Chat(store, get_azure_llm().bind(reasoning_effort="low"))

    def turn(q):
        full_q, ans, srcs = chat.ask(q)
        if full_q != q:
            print(f"   (다시 쓴 질문: {full_q})")
        print(f"답: {ans}")
        print(f"   출처 후보: {', '.join(srcs)}\n")

    if scripted:
        for q in SCRIPT:
            print(f"질문: {q}")
            turn(q)
        return

    print("질문을 입력하세요. 빈 줄이나 quit 로 종료합니다.\n")
    while True:
        try:
            q = input("질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        # 한글 입력기가 조합 중이던 글자가 섞여 들어오면 API 전송 시 UnicodeEncodeError가 난다.
        # (실제로 겪음: "GraphQ RAG의 핵심 주장응ㄴ=   ㅇ  ㅇ   이 뭐야?")
        q = q.encode("utf-8", "ignore").decode("utf-8")
        turn(q)


if __name__ == "__main__":
    main()
