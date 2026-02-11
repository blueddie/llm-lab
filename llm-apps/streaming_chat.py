"""
스트리밍 응답 — LLM이 토큰을 생성하는 즉시 출력

기본 모드: LLM이 전체 응답 완성 → 한꺼번에 반환
스트리밍:  LLM이 토큰 하나 생성 → 즉시 전송 → 반복

stream=True 하나만 추가하면 되지만, 응답 처리 방식이 달라짐:
- 기본: response.choices[0].message.content (완성된 텍스트)
- 스트리밍: async for chunk in stream (조각이 하나씩 날아옴)
"""

import asyncio
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def chat_no_stream(prompt: str):
    """기본 모드 — 전체 응답을 기다림"""

    print("=== 기본 모드 ===")
    start = time.time()

    response = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )

    elapsed = round(time.time() - start, 2)
    print(response.choices[0].message.content)
    print(f"\n(전체 응답까지 {elapsed}초 대기)\n")


async def chat_stream(prompt: str):
    """스트리밍 모드 — 토큰이 생성되는 즉시 출력"""

    print("=== 스트리밍 모드 ===")
    start = time.time()
    first_token_time = None

    # stream=True — 이것만 추가하면 스트리밍 활성화
    stream = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        stream=True,
    )

    # async for — 비동기 반복. chunk가 하나씩 날아옴
    async for chunk in stream:
        # chunk.choices[0].delta.content — 이번에 새로 생성된 토큰
        # delta = "변화분". 전체가 아니라 새로 추가된 부분만 옴
        content = chunk.choices[0].delta.content

        if content is not None:
            if first_token_time is None:
                first_token_time = round(time.time() - start, 2)
            print(content, end="", flush=True)  # 줄바꿈 없이 이어서 출력

    elapsed = round(time.time() - start, 2)
    print(f"\n\n(첫 토큰까지 {first_token_time}초, 전체 완료 {elapsed}초)\n")


async def main():
    prompt = "Python의 async/await가 실무에서 쓰이는 대표적인 사례 3가지를 설명해줘."

    print(f"프롬프트: {prompt}\n")

    # 기본 모드 먼저
    await chat_no_stream(prompt)

    # 스트리밍 모드
    await chat_stream(prompt)


if __name__ == "__main__":
    asyncio.run(main())

"""
프롬프트: Python의 async/await가 실무에서 쓰이는 대표적인 사례 3가지를 설명해줘.

=== 기본 모드 ===
Python의 async/await는 비동기 프로그래밍을 쉽게 구현하기 위해 도입된 기능으로, 실무에서 다양한 상황에 활용되고 있습니다. 대표적인 사례 3가지는 다음과 같습니다:

1. **네트워크 I/O 작업 최적화 (웹 크롤러/HTTP 요청 처리)**
   - 웹 크롤러나 API 클라이언트 등은 여러 서버로부터 데이터를 요청하는 과정에서 I/O 대기 시간이 큽니다. async/await를 사용하면 많은 요청을 병렬로 보내고, 응답을 기다리면서
다른 작업을 수행할 수 있어 전체 처리 시간이 대폭 줄어듭니다. 예를 들어, `aiohttp`와 같은 라이브러리와 함께 활용됩니다.

2. **다중 데이터베이스 처리 및 서버 응답 최적화**
   - 웹 서버 개발 시, 데이터베이스 또는 외부 서비스와의 비동기 요청을 처리하는 부분에 async/await를 도입하여 요청 응답 속도를 높입니다. 예를 들어, FastAPI 같은 프레임워크는
 비동기 함수를 지원하여, 동시에 여러 요청을 처리할 때 서버의 처리 효율성을 향상시킵니다.

3. **실시간 데이터 스트리밍 및 처리 시스템**
   - 실시간 채팅, 알림 서비스, 금융 데이터 스트리밍 등에서는 지속적으로 데이터를 받고 처리하는 작업이 많습니다. async/

(전체 응답까지 3.29초 대기)

=== 스트리밍 모드 ===
Python의 async/await는 비동기 프로그래밍을 가능하게 하여, I/O 바운드 작업의 효율성을 극대화하는 데 매우 유용합니다. 실무에서 흔히 사용되는 대표적인 사례 세 가지는 다음과 같
습니다.

1. **웹 서버 및 API 서버 개발**
   - **설명:** 비동기 프레임워크(예: FastAPI, Sanic, Aiohttp)를 사용하여 여러 클라이언트 요청을 병렬로 처리할 수 있습니다.
   - **이유:** 요청마다 데이터베이스 조회, 외부 API 호출, 파일 읽기 등이 I/O 작업이 많은데, async/await를 통해 이러한 작업을 효율적으로 병렬 처리하면서 성능을 높일 수 있습
니다.

2. **데이터 크롤러 및 웹 스크래퍼**
   - **설명:** 여러 웹 페이지를 동시다발적으로 요청하여 데이터를 수집하는 작업에 활용됩니다.
   - **이유:** 네트워크 I/O가 많은 작업이기 때문에, asyncio와 aiohttp 등 비동기 네트워크 라이브러리를 이용하면 크롤링 속도를 크게 향상시킬 수 있습니다.

3. **실시간 데이터 처리 및 메시지 큐 연동**
   - **설명:** Kafka, RabbitMQ 등의 메시지 브로커와 연동하거나, 실시간 데이터 스트리밍 시스템을 구축할

(첫 토큰까지 0.29초, 전체 완료 2.85초)
"""