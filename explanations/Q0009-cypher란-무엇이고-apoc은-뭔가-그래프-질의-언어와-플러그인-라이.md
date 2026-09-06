# Q0009 · Cypher란 무엇이고 APOC은 뭔가? (그래프 질의 언어와 플러그인 라이브러리)

- **태그**: graphrag
- **날짜**: 2026-09-05
- **코드**: [`code/Q0009_cypher_apoc.py`](../code/Q0009_cypher_apoc.py)

## ❓ 질문
Cypher가 뭐야? APOC은 Cypher의 확장 버전인가? 프로시저가 함수라면, APOC은 라이브러리인가
아니면 별도 서버를 띄워서 통신하는 건가?

## 💡 한 줄 답
> **Cypher = 그래프 DB의 질의 언어** (관계형 DB의 SQL 자리).
> **APOC = Neo4j 서버 안에 끼워 넣는 jar 파일 하나짜리 라이브러리.** 별도 서버가 아니다.
> Cypher의 *문법*을 바꾸지 않고, 부를 수 있는 *함수와 프로시저를 추가*한다.

## 📖 설명

### Cypher, 관계를 그림처럼 쓰는 언어

Neo4j의 질의 언어다. SQL이 표(행과 열)를 다루듯, Cypher는 노드와 관계를 다룬다.
문법의 핵심은 **화살표**다.

```cypher
(a)-[:협업]->(b)
```
- `( )` 노드, `[ ]` 관계, `->` 방향

관계를 그린 모양을 **그대로** 쓴다. 이게 SQL과 가장 다른 점이다.

```cypher
MATCH (p:Q0009)-[:협업]->(other)
WHERE p.dept = '개발팀'
RETURN p.name, other.name
```

SQL과 대응시키면 이렇다.

| Cypher | SQL |
|---|---|
| `MATCH` | `FROM` + `JOIN` |
| `WHERE` | `WHERE` |
| `RETURN` | `SELECT` |
| `CREATE` | `INSERT` |

> 참고: Cypher는 이제 ISO 표준 그래프 질의 언어 **GQL**의 바탕이 됐다. Neo4j 전용 언어에서
> 업계 표준으로 자리를 옮기는 중이다.

### SQL이 못 하는 것, 가변 길이 경로

Cypher를 쓰는 진짜 이유는 이것이다.

```cypher
MATCH path = (a {name:'김철수'})-[:협업*1..5]-(b {name:'최지민'})
RETURN [n IN nodes(path) | n.name], length(path)
```
→ `김철수 → 박영희 → 이민수 → 최지민 (3다리)`

핵심은 **`*1..5`**, "1~5다리 사이 어디든"이라는 뜻이다. SQL로 같은 질문을 하려면
1다리, 2다리, 3다리…를 각각 JOIN으로 짜서 `UNION`해야 하는데, **몇 다리인지 모르면
JOIN을 몇 번 걸지 미리 정할 수가 없다.**
([[Q0004-데이터베이스는-종류별로-언제-써야-하나-정형db-문서db-vectordb]]에서 networkx로 봤던
그 문제를, Cypher는 문법 차원에서 해결한다.)

### 프로시저 vs 함수, 둘 다 "함수"지만 다르다

질문에서 "프로시저라는 건 함수잖아"라고 하셨는데, Cypher에서는 이 둘을 **구분**한다.

| | **함수** (function) | **프로시저** (procedure) |
|---|---|---|
| 돌려주는 것 | **값 하나** | **행들의 스트림** (여러 줄) |
| 쓰는 법 | 표현식 안에 끼워 씀 | `CALL` 로 부르고 `YIELD` 로 컬럼을 받음 |
| 예 | `RETURN apoc.version()` | `CALL apoc.meta.data() YIELD label, property` |
| SQL 비유 | `UPPER()`, `NOW()` | 저장 프로시저 |

실제 실행 결과:

```
함수    : RETURN apoc.version()             → 2026.07.1        (한 값)
프로시저: CALL apoc.meta.data() YIELD ...   → 4행              (여러 줄)
            label=Q0009  property=name
            label=Q0009  property=dept
            ...
```

`apoc.meta.data`는 "이 DB에 어떤 라벨과 속성이 있나"를 훑는 프로시저인데, 결과가 여러 줄이라
함수가 될 수 없다. **LangChain의 `Neo4jGraph`가 스키마를 파악할 때 쓰는 게 정확히 이것**이다.

### ⭐ APOC은 서버가 아니라 라이브러리다

질문의 핵심에 답하면: **별도 서버를 띄워 통신하는 게 아니다.**

APOC(**A**wesome **P**rocedures **O**n **C**ypher)의 실체는 **jar 파일 하나**다.
설치는 그 파일을 Neo4j의 `plugins/` 폴더에 두고 재시작하는 것, 그게 전부다.

```
/opt/homebrew/opt/neo4j/libexec/
├── labs/apoc-2026.07.1-core.jar    ← 배포판에 동봉돼 있다
└── plugins/apoc-2026.07.1-core.jar ← 여기로 복사하면 설치 끝
```

Neo4j가 뜰 때 이 jar를 읽어 프로시저, 함수를 **자기 프로세스 안에** 등록한다. 실측하면:

```
등록된 프로시저 225개 중 APOC이 174개, APOC 함수는 242개
```

전체 프로시저의 대부분이 APOC인 셈이다. 그래서 "Neo4j를 쓴다"는 건 사실상 "APOC과 함께 쓴다"에
가깝다.

> 📌 "Cypher의 확장 버전"이라는 표현은 절반만 맞다. APOC은 **Cypher 문법을 바꾸지 않는다.**
> `CALL`과 `RETURN`은 원래 Cypher 문법이고, APOC은 거기서 **부를 수 있는 목록을 늘릴 뿐**이다.
> 파이썬에 비유하면 문법을 바꾸는 게 아니라 `pip install`로 쓸 수 있는 함수가 늘어나는 것과 같다.

### 그래서 샌드박스 문제가 생긴다

APOC이 **DB 프로세스 안에서 도는 자바 코드**라는 사실이 보안 문제를 만든다. Neo4j 입장에선
외부 코드가 자기 내부에서 실행되는 것이므로, 기본적으로 **샌드박스**에 가둔다.

```
apoc.meta.data is unavailable because it is sandboxed
and has dependencies outside of the sandbox
```

`apoc.meta.data`는 스키마를 훑으려고 커널 내부에 접근해야 해서 샌드박스 밖 의존성을 갖는다.
그래서 `neo4j.conf`에 "이 라이브러리는 신뢰한다"고 명시해야 한다.

```
dbms.security.procedures.unrestricted=apoc.*   # 샌드박스 제한 해제
dbms.security.procedures.allowlist=apoc.*      # 호출 허용 목록
```

두 설정은 층위가 다르다, `allowlist`는 "불러도 되는가", `unrestricted`는 "샌드박스 밖에
나가도 되는가". 보통 같이 쓴다.

> ⚠️ 운영 환경이라면 `apoc.*` 전체를 여는 대신 실제로 쓰는 것만 좁히는 게 맞다.
> 예: `apoc.meta.*,apoc.merge.*`

### APOC Core와 Extended

- **Core**: Neo4j 배포판에 동봉. 공식 지원. 우리가 쓰는 것
- **Extended**: 별도 다운로드. 외부 시스템 연동(MongoDB, Elasticsearch 등) 등 더 넓은 기능

## 🔗 코드와 연결

```bash
.venv/bin/python code/Q0009_cypher_apoc.py
```

Neo4j에 사원 4명과 협업 관계를 만들어 위 내용을 전부 실행해 본다.
데모 노드는 `:Q0009` 라벨을 달고 **끝나면 스스로 지운다** (기존 데이터를 건드리지 않는다).

| 파트 | 보여주는 것 |
|---|---|
| 1 | `CREATE`/`MATCH`/`WHERE`/`RETURN`, 화살표 문법 |
| 2 | `*1..5` 가변 길이 경로로 3다리 경로 찾기 |
| 3 | `RETURN apoc.version()` vs `CALL apoc.meta.data() YIELD ...` |
| 4 | 등록된 프로시저 개수와 jar 파일 위치, "서버가 아니다"의 증거 |

## 🔍 이어서 나온 질문 셋

### ① "샌드박스는 DB 내부 접근을 전부 막는 것인가?"

거의 맞다. 다만 **전부 차단**은 아니고 층이 나뉘어 있다.

Neo4j는 플러그인이 쓰는 API를 두 종류로 본다.
- **공개 API**: 플러그인이 써도 되는 안정된 인터페이스 → 샌드박스 안에서 그냥 동작한다
- **내부 API**: 커널 구조에 직접 손대는 것 → 기본 차단, `unrestricted`에 올려야 열린다

그래서 APOC 프로시저 대부분은 설정 없이도 돌아간다. `apoc.meta.data`처럼 **스키마를 훑으려고
커널 내부를 봐야 하는 것들만** 막혔던 것이다. 그리고 이 허용은 자동 판단이 아니라
**관리자가 설정 파일에 직접 적어 "신뢰한다"고 선언**하는 방식이다.

### ② "Neo4j는 파이썬 아닌가? APOC이 자바인 게 어떻게 가능하지?"

**Neo4j 서버 자체가 자바로 만들어졌다.** 파이썬은 그 서버에 말을 거는 *클라이언트*일 뿐이다.

```
[파이썬 프로세스]                        [자바 프로세스 = Neo4j 서버]
 langchain-neo4j                          Neo4j 커널 (JVM)
 neo4j 드라이버 6.2.0   ── Bolt(TCP) ──>   ├── lib/*.jar        (235개)
 (내 코드)                 7687번 포트      └── plugins/apoc.jar  ← APOC
```

실제 프로세스를 보면 이렇다.

```
$ pgrep -fl neo4j
57010 /opt/homebrew/opt/openjdk@21/.../bin/java -cp .../plugins/*:...
```

`java` 명령으로 떠 있고, `-cp`(클래스패스)에 `plugins/*`가 들어 있다. **APOC이 자바인 건
당연한 일**이다, 자바 프로그램 안에 끼워 넣는 부품이니까.

파이썬 쪽이 하는 일은 **Bolt 프로토콜로 Cypher 문자열을 보내고 결과를 받는 것**뿐이다.
그래서 드라이버는 파이썬 말고도 자바스크립트, Go, 자바 등 언어별로 따로 있다.
**언어가 다른 게 아니라 층이 다르다**, MySQL 서버가 C++인데 파이썬으로 붙는 것과 똑같다.

### ③ "행들의 스트림이란? 프로시저가 '더 처리할 명령어'를 반환하나?"

**아니다. 명령어가 아니라 데이터(행)를 반환한다.** SQL의 결과 테이블과 같다.

"스트림"은 **결과를 한꺼번에 만들어 넘기지 않고, 한 행씩 흘려보낸다**는 뜻이다.
그래서 프로시저 뒤에 Cypher를 계속 이어 붙일 수 있다, `MATCH`가 찾아낸 행을 이어받듯이.

```cypher
CALL apoc.meta.data() YIELD label, property   -- 여기서 행들이 흘러나오고
WHERE label = 'Q0009x'                        -- 그 행들을 거르고
RETURN label, collect(property), count(*)     -- 집계까지 한다
```
```
→ {'label': 'Q0009x', '속성들': ['name', 'dept'], '개수': 2}
```

심지어 `MATCH`와도 이어진다.

```cypher
CALL apoc.meta.data() YIELD label
WHERE label = 'Q0009x'
MATCH (n:Q0009x)
RETURN DISTINCT n.name
```

즉 **프로시저는 쿼리 파이프라인의 한 단계**로 끼어든다. 함수는 값 하나라 표현식 안에서 끝나지만,
프로시저는 행을 뿜어내므로 그 뒤에 필터, 집계, 조인이 계속될 수 있다. `YIELD`는 "그 행에서
어떤 컬럼을 받을지" 고르는 문법이다.


## 📚 더 알아보기
- 선행: [[Q0004-데이터베이스는-종류별로-언제-써야-하나-정형db-문서db-vectordb]], 그래프 DB를 언제 쓰나
- 다음: **GraphRAG**, 이 Cypher를 LLM이 대신 써주게 만드는 것
- 다음 질문 후보:
  - "GraphRAG는 언제 VectorRAG보다 나은가?"
  - "LLM이 자연어 질문을 Cypher로 바꿔주는 원리는? (GraphCypherQAChain)"
  - "문서에서 엔티티, 관계를 자동 추출해 그래프를 만드는 법 (LLMGraphTransformer)"
