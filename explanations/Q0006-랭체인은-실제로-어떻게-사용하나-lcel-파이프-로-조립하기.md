# Q0006 · 랭체인은 실제로 어떻게 사용하나? (LCEL 파이프 `|` 로 조립하기)

- **태그**: concept
- **날짜**: 2026-08-18
- **코드**: [`code/Q0006_lcel_basics.py`](../code/Q0006_lcel_basics.py)

## ❓ 질문
랭체인이란 무엇이며 어떻게 사용하는 것인가?

> "무엇인가"는 [[Q0002-랭체인-langchain-이란-무엇인가]]에서 이미 정리했다 (한 줄: LLM 앱을
> 레고처럼 조립하게 해주는 오픈소스 프레임워크). **이 문서는 "어떻게 쓰는가"를 다룬다.**

## 💡 한 줄 답
> **부품을 `|`(파이프)로 이어 붙이고 `invoke()` 하면 끝.**
> ```python
> chain = prompt | model | parser
> chain.invoke({"question": "RAG가 뭐야?"})
> ```
> 이 파이프 문법을 **LCEL**(LangChain Expression Language)이라고 부른다.

## 📖 설명

### 왜 파이프인가, 쉘과 똑같다

```bash
cat 파일 | grep 검색어 | head -3      # 앞의 출력이 뒤의 입력이 된다
```
```python
prompt | model | parser               # 완전히 같은 발상
```

랭체인의 모든 부품은 **`Runnable`이라는 하나의 규격**을 따른다. 규격이 같으니 순서만 맞으면
무엇이든 이어 붙일 수 있다. 이것이 랭체인 사용법의 거의 전부다.

### 사용 3단계

| 단계 | 하는 일 | 코드 |
|---|---|---|
| **1. 부품 고르기** | 프롬프트 · 모델 · 파서 (+ 검색기 · 메모리 · 도구) | `prompt = ChatPromptTemplate.from_template(...)` |
| **2. `\|` 로 잇기** | 데이터가 흐를 순서를 정한다 | `chain = prompt \| model \| parser` |
| **3. 실행** | 입력을 넣고 돌린다 | `chain.invoke({"question": "..."})` |

각 부품의 역할:

```
{"question": "RAG가 뭐야?"}
        ↓  prompt  , 질문을 완성된 프롬프트 문장으로 만든다
"다음 질문에 한국어로 답하세요. 질문: RAG가 뭐야?"
        ↓  model   , LLM에 보내고 응답 객체를 받는다
AIMessage(content="RAG는 …", response_metadata={...})
        ↓  parser  , 응답 객체에서 필요한 것만 꺼낸다
"RAG는 …"           ← 최종 결과 (문자열)
```

### 랭체인을 쓰는 진짜 이유 두 가지

**① 부품 교체가 한 줄이다.**

```python
model = ChatOpenAI(...)      # OpenAI
model = ChatDeepSeek(...)    # DeepSeek   ← 이 줄만 바꾸면 끝
model = ChatOllama(...)      # 로컬 모델
```
`chain = prompt | model | parser`는 **한 글자도 바뀌지 않는다.** SDK를 직접 쓰면 호출 방식도
응답 구조도 제각각이라 코드를 전부 고쳐야 한다.

**② 조립하면 부가 기능이 공짜로 따라온다.**

| 메서드 | 하는 일 |
|---|---|
| `invoke()` | 한 번 실행 |
| `batch()` | 여러 입력을 한꺼번에 (내부 병렬 처리) |
| `stream()` | 토큰이 생성되는 대로 흘려보내기 (ChatGPT식 타이핑 효과) |
| `ainvoke()` | 비동기 실행 |

직접 함수로 짰다면 병렬, 스트리밍, 비동기를 전부 손으로 구현해야 한다.
`Runnable` 규격만 지키면 이 기능들이 전부 딸려온다, **이게 조립의 대가**다.

### RAG 체인의 표준 형태

RAG도 결국 같은 파이프다. 앞에 **검색 단계**가 하나 붙을 뿐이다.

```python
chain = (
    RunnableParallel(
        context=retriever,          # 질문 → 관련 문서를 찾아온다
        question=RunnablePassthrough(),  # 질문은 그대로 통과시킨다
    )
    | prompt      # 근거 + 질문을 하나의 프롬프트로
    | model
    | StrOutputParser()
)
chain.invoke("환불은 며칠 이내인가요?")
```

- **`RunnableParallel`**: 여러 갈래를 동시에 만들어 dict로 합친다 (여기선 `context`와 `question`)
- **`RunnablePassthrough`**: 입력을 손대지 않고 그대로 흘려보낸다
- **`RunnableLambda`**: **아무 파이썬 함수나 부품으로 만든다**, 랭체인이 안 해주는 건 직접 함수로 끼워 넣으면 된다

### LCEL과 LangGraph

흐름이 **직선**이면 LCEL 파이프로 충분하다. 반복, 분기, 조건이 많은 Agent라면
[[Q0003-랭체인은-agent-개발용-프레임워크인가-chain-vs-agent]]에서 말한 **LangGraph**를 쓴다.
지금 공부하는 RAG는 대부분 직선이라 LCEL로 해결된다.

## 🔗 코드와 연결

[`code/Q0006_lcel_basics.py`](../code/Q0006_lcel_basics.py)는 **API 키 없이** 위 내용을 전부 실행해 본다.
(키가 없으면 `FakeListChatModel`이라는 가짜 모델로 폴백, 정해진 문장을 뱉기만 하지만
**체인 코드는 진짜 모델일 때와 완전히 동일**하다. 그 자체가 ①번 논점의 증명이다.)

| 파트 | 보여주는 것 |
|---|---|
| 0 | 랭체인 없이 OpenAI SDK로 짤 때의 코드, 비교 기준 |
| 1 | `prompt \| model \| parser` 기본 체인 |
| 2 | 모델을 바꿔도 체인 코드는 그대로 |
| 3 | `RunnableLambda`로 검색 함수를 부품화 → RAG 체인 뼈대 |
| 4 | `batch()` · `stream()` 실제 동작 |

3번의 `retrieve` 자리에 진짜 벡터 검색기를 꽂으면 그대로 실전 RAG가 된다.
그 진짜 검색기를 만드는 코드가 [[Q0001-rag는-왜-필요한가]]의 `FAISS.as_retriever()`다.

## 📚 더 알아보기
- 선행: [[Q0002-랭체인-langchain-이란-무엇인가]], 랭체인이 무엇인지
- 관련: [[Q0003-랭체인은-agent-개발용-프레임워크인가-chain-vs-agent]], 직선(Chain) vs 유동(Agent)
- 관련: [[Q0004-데이터베이스는-종류별로-언제-써야-하나-정형db-문서db-vectordb]], 체인에 꽂을 검색기가 사는 곳
- 다음 질문 후보:
  - "retriever는 어떻게 만들고 튜닝하나? (k값, 청크 크기)"
  - "프롬프트 템플릿을 잘 쓰는 법은?"
  - "LangGraph는 언제 LCEL 대신 쓰나?"
