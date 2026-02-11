"""
OpenAI vs Anthropic SDK 비교

같은 질문을 두 provider에 동시에 보내서 비교:
- SDK 사용법이 얼마나 비슷한지
- 응답 구조의 차이
- 속도 / 토큰 사용량 차이

두 SDK 모두 내부적으로 httpx를 사용함 — 우리가 직접 했던 것을 감싸놓은 것
"""

import asyncio
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

load_dotenv()

# SDK 클라이언트 생성 — httpx.AsyncClient()와 같은 역할
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


async def call_openai(prompt: str) -> dict:
    """OpenAI API 호출 (SDK)"""

    start = time.time()

    # httpx로 직접 할 때 5줄이던 것이 이 한 줄로
    response = await openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )

    elapsed = round(time.time() - start, 2)

    return {
        "provider": "OpenAI",
        "model": response.model,
        "answer": response.choices[0].message.content,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "time": elapsed,
    }


async def call_anthropic(prompt: str) -> dict:
    """Anthropic API 호출 (SDK)"""

    start = time.time()

    # OpenAI와 구조가 거의 같지만 미묘한 차이가 있음 — 찾아보세요
    response = await anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )

    elapsed = round(time.time() - start, 2)

    return {
        "provider": "Anthropic",
        "model": response.model,
        "answer": response.content[0].text,       # OpenAI와 다른 부분
        "prompt_tokens": response.usage.input_tokens,     # 필드명이 다름
        "completion_tokens": response.usage.output_tokens, # 필드명이 다름
        "time": elapsed,
    }


async def main():
    prompt = "Python에서 async/await의 핵심을 2문장으로 설명해줘."

    print(f"프롬프트: {prompt}\n")
    print("두 provider에 동시 호출 중...\n")

    # gather로 동시 호출 — async 실습에서 배운 패턴 그대로
    results = await asyncio.gather(
        call_openai(prompt),
        call_anthropic(prompt),
    )

    for r in results:
        print(f"=== {r['provider']} ({r['model']}) | {r['time']}초 ===")
        print(f"응답: {r['answer']}")
        print(f"토큰: 입력 {r['prompt_tokens']} + 출력 {r['completion_tokens']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())


# 프롬프트: Python에서 async/await의 핵심을 2문장으로 설명해줘.

# 두 provider에 동시 호출 중...

# === OpenAI (gpt-4.1-nano-2025-04-14) | 1.67초 ===
# 응답: Python의 async/await는 비동기 프로그래밍을 가능하게 하는 문법으로, 이벤트 루프를 통해 비동기 함수를 실행하여 I/O 작업 등 기다리는 시간을 효율적으로 처리할 수 있게 합니
# 다. 이를 통해 블로킹 없이 여러 작업을 동시에 수행할 수 있으며, 코드의 가독성과 유지보수성을 높여줍니다.
# 토큰: 입력 25 + 출력 78

# === Anthropic (claude-sonnet-4-20250514) | 3.33초 ===
# 응답: async/await는 Python에서 비동기 프로그래밍을 위한 문법으로, I/O 작업 등에서 대기 시간 동안 다른 코드를 실행할 수 있게 해줍니다. async로 정의한 함수는 코루틴이 되며, awa
# it 키워드로 비동기 작업의 완료를 기다리면서도 전체 프로그램이 블로킹되지 않습니다.
# 토큰: 입력 33 + 출력 141