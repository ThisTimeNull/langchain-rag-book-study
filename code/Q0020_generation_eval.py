"""
Q0020 · 생성 단계는 어떻게 평가하나?

Q0008에서 검색은 숫자로 쟀다. "정답 문구가 검색된 조각에 있나."
생성은 지금까지 눈으로 봤다. 그래서 Q0019에서 같은 코드를 세 번 돌려 세 번 다른 결과를
얻고도 "4패"라고 잘못 적었다. 사람이 채점하면 실행마다 흔들리는 걸 잡을 수 없다.

여기서는 세 가지를 만든다.

  1. 정답셋       질문마다 "이게 맞는 답이다"를 한 줄로 적어둔다
  2. LLM 채점기   다른 LLM 호출에게 답을 보여주고 세 가지를 0~2점으로 매기게 한다
       정확성   정답과 같은 내용인가
       충실성   근거에 없는 말을 지어냈나
       거부 판단 모른다고 해야 할 때 모른다고 했나 (또는 그 반대)
  3. 반복 실행    질문마다 N번 돌려 "N번 중 몇 번"으로 낸다

채점기도 LLM이다. 채점기가 틀릴 수 있다. 그래서 채점 결과를 사람이 다시 볼 수 있게
채점 이유를 함께 남긴다.

실행: python code/Q0020_generation_eval.py            (질문당 3회)
      python code/Q0020_generation_eval.py --runs 5
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Literal

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_azure_llm
from Q0019_pdf_chat import build_index, Chat   # 평가 대상은 Q0019 그대로

# --- 1. 정답셋 -------------------------------------------------------------------
# expect: 정답 요지. "모름" 이면 문서에 없어서 "찾을 수 없습니다"가 정답이라는 뜻.
# 되쓰기 시험용으로 앞 질문에 기대는 순서를 그대로 둔다.
GOLD = [
    ("GraphRAG 논문의 핵심 주장이 뭐야?",
     "LLM으로 지식 그래프를 만들고 커뮤니티 요약을 계층적으로 생성해 코퍼스 전체에 대한 전역 질문에 답한다"),
    ("그걸 실험으로 어떻게 증명했어?",
     "정답 없는 전역 질문을 LLM으로 생성하고 다른 LLM이 답변을 비교 평가했다. vector RAG보다 포괄성과 다양성에서 우세"),
    ("그 실험에서 비교 대상이 된 방식은 뭐였어?",
     "vector RAG (naive RAG)"),
    ("트랜스포머 논문에서 어텐션을 설명하는 수식이 뭐야?",
     "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V"),
    ("그 논문 저자는 몇 명이야?",
     "8명"),
    ("RAG 논문은 몇 년도에 나왔고 누가 썼어?",
     "2020년(arXiv 2005.11401), Patrick Lewis 외 (Facebook AI Research 등)"),
    ("세 논문 중에 가장 먼저 나온 건?",
     "트랜스포머 논문 (2017)"),
]


# --- 2. LLM 채점기 -------------------------------------------------------------
class Grade(BaseModel):
    correctness: Literal[0, 1, 2] = Field(description="[정답]과 비교. 2=같은 내용, 1=일부만, 0=틀리거나 답을 안 함")
    faithfulness: Literal[0, 1, 2] = Field(description="[근거]와 비교. 2=근거에 있는 말만 씀, 1=일부 추가, 0=근거에 없는 사실을 지어냄")
    reason: str = Field(description="한 문장. 사람이 다시 볼 수 있게")


JUDGE = ChatPromptTemplate.from_template(
    "당신은 채점자입니다. 두 기준을 따로 매기세요. 둘을 섞지 마세요.\n"
    "정확성: [답변]이 [정답]과 같은 내용인가. [근거]는 보지 마세요.\n"
    "  '문서에서 찾을 수 없습니다'라고 답했는데 [정답]이 있으면 정확성은 0입니다. 근거에 있든 없든 상관없습니다.\n"
    "충실성: [답변]이 [근거]에 있는 말만 썼는가. [정답]은 보지 마세요.\n"
    "  '문서에서 찾을 수 없습니다'라고 답한 것은 아무것도 지어내지 않았으므로 충실성 2입니다.\n\n"
    "[질문] {question}\n[정답] {gold}\n[근거]\n{context}\n\n[답변] {answer}"
)

REFUSAL_MARK = "찾을 수 없습니다"


def refusal_flag(answer, gold_answerable=True):
    """거부 판단은 LLM에게 맡기지 않는다. 코드로 한다.

    처음엔 채점기 LLM에게 시켰더니 21번 전부 '적절'이라고 했다. 채점기는 검색된 근거만 보니
    '검색이 놓쳐서 근거에 없는 것'과 '문서 자체에 없는 것'을 구분할 수 없다.
    정답이 있는 질문에서 모른다고 하면 그건 무조건 부당한 거부다. 이건 문자열 검사로 충분하다.
    """
    refused = REFUSAL_MARK in answer
    if refused and gold_answerable:
        return "R"      # 답이 있는데 모른다고 함
    if not refused and not gold_answerable:
        return "H"      # 없는데 답을 함
    return "o"


def grade(judge, question, gold, context, answer):
    return judge.invoke({"question": question, "gold": gold, "context": context, "answer": answer})


# --- 3. 반복 실행 ---------------------------------------------------------------
def run_once(store, llm, judge):
    """대본 7개를 처음부터 끝까지 한 번 돌리고 질문마다 채점한다. 대화 기록은 매 회 새로 시작."""
    chat = Chat(store, llm)
    out = []
    for q, gold in GOLD:
        for attempt in range(4):
            try:
                full_q, ans, srcs = chat.ask(q)
                # Chat.ask 는 근거를 돌려주지 않으니 같은 검색을 다시 해서 채점용 근거를 만든다
                hits = chat.retriever.invoke(full_q)
                ctx = "\n\n".join(f"[{d.metadata['file']} p.{d.metadata['page']}] {d.page_content}" for d in hits)
                g = grade(judge, q, gold, ctx, ans)
                break
            except Exception as e:   # 네트워크가 잠깐 끊기면 한 질문 때문에 전체를 버리지 않는다
                if attempt == 3:
                    raise
                print(f"    (재시도 {attempt+1}: {type(e).__name__})")
                time.sleep(10 * (attempt + 1))
        out.append((q, full_q, ans, g))
    return out


def main():
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 3
    print(f"Q0020 · 생성 답변을 기계가 채점한다 (질문 7개 × {runs}회)\n")

    store = build_index(rebuild=False)
    llm = get_azure_llm().bind(reasoning_effort="low")
    judge = JUDGE | get_azure_llm().with_structured_output(Grade)   # 채점은 기본 추론 강도로

    results = [run_once(store, llm, judge) for _ in range(runs)]

    print(f"{'=' * 72}")
    print(f"{'#':>2s} {'정확성':>6s} {'충실성':>6s} {'거부판단':>10s}  질문")
    print(f"{'=' * 72}")
    for i, (q, gold) in enumerate(GOLD):
        grades = [r[i][3] for r in results]
        corr = sum(g.correctness for g in grades)
        faith = sum(g.faithfulness for g in grades)
        refusal = "".join(refusal_flag(r[i][2]) for r in results)
        print(f"{i+1:>2d} {corr:>4d}/{runs*2:<2d} {faith:>4d}/{runs*2:<2d} {refusal:>10s}  {q[:34]}")
    print(f"{'=' * 72}")
    print("  거부판단: o=적절  R=답이 있는데 모른다고 함  H=없는데 지어냄")

    print(f"\n{'-' * 72}\n채점 이유 (사람이 다시 볼 것)\n{'-' * 72}")
    for i, (q, gold) in enumerate(GOLD):
        print(f"\n{i+1}. {q}")
        for r_idx, r in enumerate(results, 1):
            _, full_q, ans, g = r[i]
            flag = "" if g.correctness == 2 else " <-"
            print(f"   {r_idx}회 [정확{g.correctness} 충실{g.faithfulness} {refusal_flag(ans)}] {g.reason[:70]}{flag}")

    print(f"\n{'=' * 72}")
    total = runs * len(GOLD)
    all_g = [r[i][3] for r in results for i in range(len(GOLD))]
    print(f"정확성 만점 비율   : {sum(g.correctness == 2 for g in all_g)}/{total}")
    all_ans = [r[i][2] for r in results for i in range(len(GOLD))]
    print(f"지어낸 답(H) 횟수  : {sum(refusal_flag(a) == 'H' for a in all_ans)}")
    print(f"부당한 거부(R) 횟수: {sum(refusal_flag(a) == 'R' for a in all_ans)}")
    print("\n  채점기도 LLM이다. 이유가 이상하면 채점기를 의심해야 한다.")


if __name__ == "__main__":
    main()
