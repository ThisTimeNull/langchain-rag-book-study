"""
Q0018 · OpenAI와 DeepSeek 모델은 어떻게 다른가?

책 2장과 3장은 개념이다. 여기서는 우리 모델(gpt-5.4-mini)로 직접 확인할 수 있는 것만 돌린다.

  (a) 추론 모델은 답하기 전에 '생각'을 한다. 그 생각도 토큰이고 돈이다.
  (b) reasoning_effort 로 생각의 양을 조절할 수 있다.
  (c) temperature 는 이 모델에서 실제로 동작한다. (처음엔 안 될 줄 알았다)
  (d) 같은 질문을 추론 없이 답하게 하면 어떻게 되나.

실행: python code/Q0018_model_internals.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_azure_llm


def section(n, title):
    print(f"\n{'=' * 66}\n[{n}] {title}\n{'=' * 66}")


def usage(r):
    u = r.usage_metadata
    reasoning = u.get("output_token_details", {}).get("reasoning", 0)
    return u["input_tokens"], u["output_tokens"], reasoning


def main():
    print("Q0018 · 추론 모델을 만져본다")

    section("a", "추론 토큰: 보이지 않는 출력")
    r = get_azure_llm().invoke("RAG가 뭔지 한 문장으로.")
    i, o, reasoning = usage(r)
    print(f"  답: {r.content[:70]}")
    print(f"  입력 {i}토큰, 출력 {o}토큰. 출력 중 추론 {reasoning}토큰, 실제 답 {o - reasoning}토큰")
    print("  💡 추론 토큰은 화면에 안 보이지만 출력 토큰으로 과금된다.")

    section("b", "reasoning_effort: 생각의 양을 조절한다")
    q = ("A는 B에 의존하고 B는 C에, C는 D에 의존한다. D가 죽으면 영향받는 서비스는? "
         "이유를 한 줄로.")
    for effort in ("low", "medium", "high"):
        llm = get_azure_llm().bind(reasoning_effort=effort)
        t = time.perf_counter()
        r = llm.invoke(q)
        dt = time.perf_counter() - t
        _, o, reasoning = usage(r)
        print(f"  {effort:6s} {dt:4.1f}초  추론 {reasoning:3d}토큰  답: {r.content.strip()[:50]}")
    print("  💡 쉬운 질문은 low 로도 맞는다. 어려운 질문에서 high 가 값을 한다.")
    print("     RAG 답변 생성처럼 근거가 이미 있는 일에는 low 가 싸고 빠르다.")

    section("c", "temperature: 실제로 동작하나")
    # 답이 하나로 쏠리는 질문("한국 도시 하나")은 temperature 를 올려도 잘 안 흩어진다.
    # 후보가 많은 질문이어야 차이가 보인다.
    q = "1부터 1000 사이 숫자 하나를 무작위로. 숫자만."
    for label, llm in (("0", get_azure_llm(temperature=0)),
                       ("1.5", get_azure_llm(temperature=1.5))):
        outs = [llm.invoke(q).content.strip()[:6] for _ in range(6)]
        print(f"  temperature={label:4s} -> {outs}  ({len(set(outs))}종)")
    print("  💡 초기 o1, o3 는 temperature 를 거부했다. gpt-5.4-mini 는 받는다.")
    print("     '추론 모델이라 안 된다'고 외우지 말고 쓰는 모델에서 직접 확인한다.")

    section("d", "추론이 정확도에 미치는 영향")
    q = ("다음 중 소수가 아닌 것은? 91, 97, 101, 103. 답만.")
    for effort in ("low", "high"):
        llm = get_azure_llm().bind(reasoning_effort=effort)
        outs = [llm.invoke(q).content.strip()[:10] for _ in range(3)]
        print(f"  {effort:5s} -> {outs}")
    print("  💡 91 = 7 x 13 이다. 언뜻 보면 소수 같아서 추론 없이는 틀리기 쉽다.")


if __name__ == "__main__":
    main()
