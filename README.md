# 질문 주도형 RAG 학습 프로젝트

책 **「랭체인으로 RAG 개발하기: VectorRAG & GraphRAG」**(길벗)를 따라가되,
**내가 주도적으로 질문 → 코드 작성·실행 → 설명 정리**하는 흐름으로 학습하는 워크스페이스입니다.

- **LLM**: OpenAI + DeepSeek · **프레임워크**: LangChain · **GraphRAG**: Neo4j

---

## 핵심 워크플로우

```
 ① 질문한다            ② 코드로 검증한다        ③ 설명으로 남긴다
 ┌──────────────┐      ┌──────────────┐         ┌──────────────┐
 │ python ask.py│ ───▶ │ code/QNNNN.py │  ───▶  │explanations/  │
 │  "질문..."   │      │ 작성 후 실행   │         │ QNNNN-….md    │
 └──────┬───────┘      └──────────────┘         └──────────────┘
        │
        ▼  자동 기록
   logs/questions.md  (모든 질문이 번호·날짜·링크와 함께 한 표에)
```

질문 하나 = **번호(QNNNN) 하나** = 로그 1줄 + 설명 1개 + 코드 1개. 셋이 같은 번호로 연결됩니다.

---

## 시작하기

### 1) 새 질문 등록

```bash
python ask.py "RAG는 왜 필요한가?"                      # 기본
python ask.py "임베딩은 어떻게 동작해?" --slug embedding --tag concept
python ask.py "그래프 질의 개념만" --tag graphrag --no-code   # 코드 없이 설명만
```

실행하면 자동으로:
- `logs/questions.md` 표에 새 줄 추가
- `explanations/QNNNN-질문.md` 설명 스텁 생성
- `code/QNNNN_slug.py` 코드 스텁 생성 (`--no-code`면 생략)

### 2) 코드 작성하고 실행

생성된 `code/QNNNN_*.py`를 채운 뒤:

```bash
python code/QNNNN_slug.py
```

`config/settings.py`의 헬퍼(`get_openai_llm`, `get_deepseek_llm`, `get_embeddings`, `get_neo4j_graph`)를 import 해서 씁니다.

### 3) 설명 정리

`explanations/QNNNN-*.md`의 빈 섹션(한 줄 답 / 설명 / 코드와 연결)을 채웁니다.
→ 이 과정은 보통 Claude에게 "이 질문 설명 채워줘"라고 요청하면 됩니다.

---

## 폴더 구조

```
.
├── ask.py                # 질문 등록 헬퍼 (로그+설명+코드 자동 생성)
├── config/settings.py    # OpenAI/DeepSeek/Neo4j 공통 헬퍼
├── logs/
│   └── questions.md      # 전체 질문 로그 (자동 갱신)
├── explanations/         # 질문별 설명 md (QNNNN-….md)
├── code/                 # 질문별 실행 코드 (QNNNN_….py)
├── data/sample/          # 실습용 샘플 문서
├── requirements.txt
└── env.sample            # API 키 템플릿 (cp env.sample .env)
```

---

## 환경

`.venv`와 `.env`는 이미 구성되어 있습니다. 새로 세팅한다면:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.sample .env      # OPENAI_API_KEY, DEEPSEEK_API_KEY 등 입력
```

### Neo4j (GraphRAG 질문용)

```bash
docker run --name neo4j-rag -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 -d neo4j:5
```

→ http://localhost:7474 · `.env`의 `NEO4J_*`를 맞춰주세요.
