#!/usr/bin/env python3
"""
ask.py — 질문 등록 헬퍼
========================
새 질문을 등록하면 다음 3가지를 자동으로 만들어 줍니다.

  1) logs/questions.md          전체 질문 로그(표)에 한 줄 추가
  2) explanations/QNNNN-<제목>.md  설명 문서 스텁 생성
  3) code/QNNNN_<slug>.py       실행 코드 스텁 생성

번호(QNNNN)는 자동으로 다음 번호가 매겨지고, 세 파일이 같은 번호로 1:1 연결됩니다.

사용법
------
  python ask.py "RAG란 무엇인가?"
  python ask.py "임베딩은 어떻게 동작해?" --slug embedding --tag concept
  python ask.py "그래프 RAG 질의" --tag graphrag --no-code   # 코드 없이 개념만

옵션
----
  --slug   코드 파일명에 쓸 영문 슬러그 (예: embedding → code/Q0003_embedding.py)
  --tag    분류 태그 (concept / vectorrag / graphrag / openai / deepseek ...)
  --no-code  코드 파일을 만들지 않음 (개념 질문용)
"""

import argparse
import datetime as dt
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs", "questions.md")
EXPL_DIR = os.path.join(ROOT, "explanations")
CODE_DIR = os.path.join(ROOT, "code")

LOG_HEADER = """# 질문 로그 (Question Log)

이 워크스페이스에서 내가 던진 모든 질문의 기록입니다. `ask.py`가 자동으로 추가합니다.

| 번호 | 날짜 | 질문 | 태그 | 설명 | 코드 |
|------|------|------|------|------|------|
"""


def next_id() -> str:
    """code/ 와 explanations/ 를 스캔해 다음 질문 번호를 정합니다."""
    nums = []
    for d in (CODE_DIR, EXPL_DIR):
        for p in glob.glob(os.path.join(d, "Q[0-9]*")):
            m = re.search(r"Q(\d+)", os.path.basename(p))
            if m:
                nums.append(int(m.group(1)))
    return f"Q{(max(nums) + 1) if nums else 1:04d}"


def title_slug(text: str) -> str:
    """설명 파일명용 슬러그. 한글을 그대로 살리고 공백/기호만 정리."""
    s = text.strip().lower()
    s = re.sub(r"[^\w가-힣]+", "-", s, flags=re.UNICODE)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:40] or "question"


def code_slug(slug: str | None, fallback: str) -> str:
    """코드 파일명용 영문 슬러그."""
    base = slug or fallback
    s = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    return s or "main"


def ensure_log():
    os.makedirs(os.path.dirname(LOGS), exist_ok=True)
    if not os.path.exists(LOGS):
        with open(LOGS, "w", encoding="utf-8") as f:
            f.write(LOG_HEADER)


def append_log(qid, date, question, tag, expl_name, code_name):
    code_cell = f"[코드](../code/{code_name})" if code_name else "—"
    question = question.replace("|", "\\|")  # 표가 깨지지 않도록 이스케이프
    row = f"| {qid} | {date} | {question} | {tag} | [설명](../explanations/{expl_name}) | {code_cell} |\n"
    with open(LOGS, "a", encoding="utf-8") as f:
        f.write(row)


def write_explanation(path, qid, date, question, tag, code_name):
    code_line = (
        f"- **코드**: [`code/{code_name}`](../code/{code_name})\n" if code_name else ""
    )
    content = f"""# {qid} · {question}

- **태그**: {tag}
- **날짜**: {date}
{code_line}
## ❓ 질문
{question}

## 💡 한 줄 답
> (작성 예정)

## 📖 설명
(작성 예정 — 개념을 쉽게 풀어서 설명)

## 🔗 코드와 연결
(작성 예정 — 위 코드가 이 질문에 어떻게 답하는지)

## 📚 더 알아보기
- (관련 개념·다음 질문 링크)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_code(path, qid, question, code_name):
    content = f'''"""
{qid} · {question}

이 질문에 답하기 위한 실행 코드.
실행: python code/{code_name}
"""

import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가 (config 패키지 사용)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 필요한 헬퍼만 import 하세요.
# from config.settings import get_openai_llm, get_deepseek_llm, get_embeddings, get_local_embeddings


def main():
    # TODO: 이 질문을 검증하는 코드를 작성하세요.
    print("{qid}: {question}")


if __name__ == "__main__":
    main()
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="새 학습 질문을 등록하고 로그·설명·코드 스텁을 생성합니다."
    )
    parser.add_argument("question", help="질문 내용 (따옴표로 감싸기)")
    parser.add_argument("--slug", help="코드 파일명용 영문 슬러그")
    parser.add_argument("--tag", default="general", help="분류 태그")
    parser.add_argument("--no-code", action="store_true", help="코드 파일 생략")
    args = parser.parse_args()

    os.makedirs(EXPL_DIR, exist_ok=True)
    os.makedirs(CODE_DIR, exist_ok=True)
    ensure_log()

    qid = next_id()
    date = dt.date.today().isoformat()
    tslug = title_slug(args.question)

    expl_name = f"{qid}-{tslug}.md"
    expl_path = os.path.join(EXPL_DIR, expl_name)

    code_name = None
    code_path = None
    if not args.no_code:
        cslug = code_slug(args.slug, tslug if tslug.isascii() else "main")
        code_name = f"{qid}_{cslug}.py"
        code_path = os.path.join(CODE_DIR, code_name)

    write_explanation(expl_path, qid, date, args.question, args.tag, code_name)
    if code_path:
        write_code(code_path, qid, args.question, code_name)
    append_log(qid, date, args.question, args.tag, expl_name, code_name)

    print(f"✅ {qid} 등록 완료")
    print(f"   로그   : logs/questions.md")
    print(f"   설명   : explanations/{expl_name}")
    if code_name:
        print(f"   코드   : code/{code_name}")
    else:
        print(f"   코드   : (생략)")


if __name__ == "__main__":
    main()
