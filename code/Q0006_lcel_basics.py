"""
Q0006 · 랭체인은 실제로 어떻게 사용하나? (LCEL 파이프 | 로 조립하기)

랭체인 사용법의 90%는 이 한 줄이다:

    chain = 프롬프트 | 모델 | 출력파서
    chain.invoke({"q": "질문"})

`|`(파이프)로 부품을 이어 붙이는 문법을 **LCEL**(LangChain Expression Language)이라 한다.
쉘의 `cat file | grep 검색어`와 완전히 같은 발상, 앞의 출력이 뒤의 입력이 된다.

이 코드가 보여주는 것
  0) 랭체인 없이 짜면 어떤가 (비교 기준)
  1) 가장 기본 체인: 프롬프트 | 모델 | 파서
  2) 부품 갈아끼우기: 모델만 바꿔도 나머지 코드는 그대로
  3) 함수도 부품이 된다 (RunnableLambda) → RAG 체인의 뼈대
  4) 병렬 실행, 스트리밍 등 공짜로 따라오는 것들

실행: python code/Q0006_lcel_basics.py
(API 키 없으면 '가짜 모델(FakeListChatModel)'로 동작. 키가 있으면 진짜 LLM으로 실행)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel


def section(n, title):
    print(f"\n{'=' * 62}\n[{n}] {title}\n{'=' * 62}")


def get_model():
    """OPENAI_API_KEY가 있으면 진짜 LLM, 없으면 가짜 모델로 폴백.

    핵심: 어느 쪽이든 아래 체인 코드는 **한 글자도 바뀌지 않는다**.
    """
    try:
        from config.settings import get_openai_llm

        return get_openai_llm(), "OpenAI gpt-4o-mini"
    except Exception:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        fake = FakeListChatModel(
            responses=[
                "RAG는 답하기 전에 관련 문서를 찾아 근거로 삼는 방식입니다.",
                "임베딩은 문장을 숫자 벡터로 바꾸는 것입니다.",
            ]
        )
        return fake, "가짜 모델(FakeListChatModel), API 키 없이 흐름만 확인"


# --- 0) 랭체인 없이 짜면? ----------------------------------------------------
def demo_without_langchain():
    section(0, "랭체인 없이 짜면, 매번 이런 코드를 직접 쓴다")
    print("""
  # OpenAI SDK 직접 사용
  prompt = f"다음 질문에 한국어로 한 문장 답: {question}"        # 문자열 조립
  res = client.chat.completions.create(                          # 모델마다 호출법이 다름
      model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
  answer = res.choices[0].message.content                        # 응답 구조도 제각각

  → 모델을 Claude/DeepSeek로 바꾸면 이 코드를 전부 고쳐야 한다.
    검색, 메모리를 붙이면 if문이 계속 늘어난다.""")


# --- 1) 가장 기본 체인 -------------------------------------------------------
def demo_basic_chain(model, model_name):
    section(1, "기본 체인, 프롬프트 | 모델 | 파서")

    prompt = ChatPromptTemplate.from_template(
        "다음 질문에 한국어로 한 문장으로 답하세요.\n질문: {question}"
    )
    chain = prompt | model | StrOutputParser()
    #        ①        ②      ③
    # ① 질문을 프롬프트 문장으로 만들고
    # ② LLM에 넘기고
    # ③ 응답 객체에서 문자열만 꺼낸다

    print(f"\n  사용 모델: {model_name}")
    print(f"  chain = prompt | model | StrOutputParser()")
    print(f"\n  invoke → {chain.invoke({'question': 'RAG가 뭐야?'})}")

    print("\n  💡 각 부품은 모두 'Runnable'이라는 같은 규격을 따른다.")
    print("     그래서 규격만 맞으면 무엇이든 | 로 이어 붙일 수 있다.")


# --- 2) 부품 갈아끼우기 ------------------------------------------------------
def demo_swap(model):
    section(2, "부품 갈아끼우기, 모델을 바꿔도 체인 코드는 그대로")

    prompt = ChatPromptTemplate.from_template("{question}")
    print("""
  chain = prompt | model | StrOutputParser()

  model = ChatOpenAI(...)      # OpenAI
  model = ChatDeepSeek(...)    # DeepSeek   ← 이 줄만 바꾸면 끝
  model = ChatOllama(...)      # 로컬 모델

  → 이게 랭체인을 쓰는 가장 큰 이유다: **부품 교체가 한 줄**.""")
    print(f"\n  실제로 지금 이 코드도 키 유무에 따라 모델이 바뀌었지만,")
    print(f"  체인 코드는 그대로였다 → {(prompt | model | StrOutputParser()).invoke({'question': '임베딩이 뭐야?'})}")


# --- 3) 함수도 부품이 된다 → RAG 체인의 뼈대 ---------------------------------
def demo_rag_shape(model):
    section(3, "함수도 부품이 된다 (RunnableLambda), RAG 체인의 뼈대")

    # 진짜 벡터 검색 대신, 이해를 위해 '검색 함수'를 직접 만든다.
    # (실제 임베딩 검색은 code/Q0001_why_rag.py, code/Q0004_db_selection.py 참고)
    DOCS = [
        "환불은 구매 후 7일 이내에 가능합니다.",
        "재택근무는 주 2회까지 신청할 수 있습니다.",
        "사내 코드네임은 프로젝트 아틀라스입니다.",
    ]

    def retrieve(question: str) -> str:
        hits = [d for d in DOCS if any(w in d for w in question.split())]
        return "\n".join(hits or DOCS[:1])

    prompt = ChatPromptTemplate.from_template(
        "아래 근거만 사용해 한국어로 답하세요.\n근거:\n{context}\n\n질문: {question}"
    )

    # 질문 하나가 들어오면 → context(검색 결과)와 question(원본)을 동시에 만든다
    rag_chain = (
        RunnableParallel(
            context=RunnableLambda(retrieve),   # 질문 → 검색 결과
            question=RunnableLambda(lambda q: q),  # 질문 → 그대로 통과
        )
        | prompt
        | model
        | StrOutputParser()
    )

    q = "환불은 며칠 이내인가요?"
    print(f"\n  질문: {q}")
    print(f"  검색된 근거: {retrieve(q)}")
    print(f"  체인 결과: {rag_chain.invoke(q)}")
    print("  (가짜 모델은 근거와 무관하게 정해진 문장을 뱉는다,\n   진짜 LLM이면 위 근거를 읽고 '7일 이내'라고 답한다. 여기서 볼 것은 '흐름'이다)")

    print("\n  💡 이 모양이 RAG 체인의 표준 형태다.")
    print("     retrieve 자리에 진짜 벡터 검색기(retriever)를 꽂으면 그대로 실전 RAG가 된다.")


# --- 4) 공짜로 따라오는 것들 -------------------------------------------------
def demo_extras(model):
    section(4, "체인으로 만들면 공짜로 따라오는 것들")

    prompt = ChatPromptTemplate.from_template("{question}")
    chain = prompt | model | StrOutputParser()

    print("\n  · invoke()  : 한 번 실행")
    print("  · batch()   : 여러 입력을 한꺼번에 (내부적으로 병렬 처리)")
    results = chain.batch([{"question": "RAG?"}, {"question": "임베딩?"}])
    for r in results:
        print(f"      → {r}")

    print("\n  · stream()  : 토큰이 생성되는 대로 흘려보내기 (타이핑 효과)")
    print("      → ", end="")
    for chunk in chain.stream({"question": "RAG?"}):
        print(chunk, end="", flush=True)
    print()

    print("\n  · ainvoke() : 비동기 실행 (async/await)")
    print("\n  💡 함수로 직접 짰다면 병렬, 스트리밍, 비동기를 전부 직접 구현해야 한다.")
    print("     Runnable 규격을 따르기만 하면 이 기능들이 전부 딸려온다, 이게 조립의 대가다.")


def main():
    print("Q0006 · 랭체인 사용법 = 부품을 | 로 이어 붙이기 (LCEL)")
    model, model_name = get_model()

    demo_without_langchain()
    demo_basic_chain(model, model_name)
    demo_swap(model)
    demo_rag_shape(model)
    demo_extras(model)

    print(f"\n{'=' * 62}")
    print("정리, 랭체인 사용법 3단계")
    print("  1. 부품을 고른다   : 프롬프트 · 모델 · 파서 (+ 검색기 · 메모리 · 도구)")
    print("  2. | 로 잇는다     : chain = prompt | model | parser")
    print("  3. invoke() 한다   : chain.invoke({...})")
    print("  ※ 부품을 바꾸고 싶으면 그 한 줄만 바꾼다. 나머지 체인은 그대로다.")


if __name__ == "__main__":
    main()
