# 질문 로그 (Question Log)

이 워크스페이스에서 내가 던진 모든 질문의 기록입니다. `ask.py`가 자동으로 추가합니다.

| 번호 | 날짜 | 질문 | 태그 | 설명 | 코드 |
|------|------|------|------|------|------|
| Q0001 | 2026-06-27 | RAG는 왜 필요한가? | concept | [설명](../explanations/Q0001-rag는-왜-필요한가.md) | [코드](../code/Q0001_why_rag.py) |
| Q0002 | 2026-06-27 | 랭체인(LangChain)이란 무엇인가? | concept | [설명](../explanations/Q0002-랭체인-langchain-이란-무엇인가.md) | — |
| Q0003 | 2026-06-27 | 랭체인은 Agent 개발용 프레임워크인가? (Chain vs Agent) | concept | [설명](../explanations/Q0003-랭체인은-agent-개발용-프레임워크인가-chain-vs-agent.md) | — |
| Q0004 | 2026-08-18 | 데이터베이스는 종류별로 언제 써야 하나? (정형DB, 문서DB, VectorDB, GraphDB 선택 기준) | concept | [설명](../explanations/Q0004-데이터베이스는-종류별로-언제-써야-하나-정형db-문서db-vectordb.md) | [코드](../code/Q0004_db_selection.py) |
| Q0005 | 2026-08-18 | FAISS란 무엇인가? (벡터 DB와 벡터 검색 라이브러리의 차이) | vectorrag | [설명](../explanations/Q0005-faiss란-무엇인가-벡터-db와-벡터-검색-라이브러리의-차이.md) | [코드](../code/Q0005_faiss.py) |
| Q0006 | 2026-08-18 | 랭체인은 실제로 어떻게 사용하나? (LCEL 파이프 \| 로 조립하기) | concept | [설명](../explanations/Q0006-랭체인은-실제로-어떻게-사용하나-lcel-파이프-로-조립하기.md) | [코드](../code/Q0006_lcel_basics.py) |
| Q0007 | 2026-08-27 | RAG 파이프라인을 처음부터 끝까지 직접 만들어보자 (로드→청킹→임베딩→저장→검색→답변) | vectorrag | [설명](../explanations/Q0007-rag-파이프라인을-처음부터-끝까지-직접-만들어보자-로드-청킹-임베딩-저.md) | [코드](../code/Q0007_rag_pipeline.py) |
| Q0008 | 2026-08-27 | RAG 품질은 어떻게 튜닝하나? (chunk_size, k, 임베딩 모델을 바꿔가며 측정하기) | vectorrag | [설명](../explanations/Q0008-rag-품질은-어떻게-튜닝하나-chunk_size-k-임베딩-모델을-바꿔.md) | [코드](../code/Q0008_rag_tuning.py) |
| Q0009 | 2026-09-05 | Cypher란 무엇이고 APOC은 뭔가? (그래프 질의 언어와 플러그인 라이브러리) | graphrag | [설명](../explanations/Q0009-cypher란-무엇이고-apoc은-뭔가-그래프-질의-언어와-플러그인-라이.md) | [코드](../code/Q0009_cypher_apoc.py) |
| Q0010 | 2026-09-05 | GraphRAG는 VectorRAG와 무엇이 다른가? (LLM으로 문서에서 그래프를 만들고 질의하기) | graphrag | [설명](../explanations/Q0010-graphrag는-vectorrag와-무엇이-다른가-llm으로-문서에서-.md) | [코드](../code/Q0010_graphrag_pipeline.py) |
| Q0011 | 2026-09-06 | GraphRAG가 이기는 데이터는 어떤 모양인가? (관계가 얽힌 문서로 다시 붙여보기) | graphrag | [설명](../explanations/Q0011-graphrag가-이기는-데이터는-어떤-모양인가-관계가-얽힌-문서로-다시.md) | [코드](../code/Q0011_graphrag_wins.py) |
| Q0012 | 2026-09-06 | LLM 추출이 이득인 경우는 언제인가? (비정형 문서에서 그래프 만들기) | graphrag | [설명](../explanations/Q0012-llm-추출이-이득인-경우는-언제인가-비정형-문서에서-그래프-만들기.md) | [코드](../code/Q0012_prose_extraction.py) |
| Q0013 | 2026-09-09 | 하이브리드 검색은 어떻게 하나? (벡터로 진입점을 찾고 그래프로 확장하기) | graphrag | [설명](../explanations/Q0013-하이브리드-검색은-어떻게-하나-벡터로-진입점을-찾고-그래프로-확장하기.md) | [코드](../code/Q0013_hybrid_search.py) |
| Q0014 | 2026-09-12 | LLM이 짠 Cypher가 틀렸을 때 어떻게 고쳐서 다시 실행하나? (검증과 재시도) | graphrag | [설명](../explanations/Q0014-llm이-짠-cypher가-틀렸을-때-어떻게-고쳐서-다시-실행하나-검증과.md) | [코드](../code/Q0014_cypher_repair.py) |
| Q0015 | 2026-09-12 | 전역 요약 질문은 왜 그래프가 이기나? (문서 전체를 세어야 답이 나오는 질문) | graphrag | [설명](../explanations/Q0015-전역-요약-질문은-왜-그래프가-이기나-문서-전체를-세어야-답이-나오는-질.md) | [코드](../code/Q0015_global_summary.py) |
