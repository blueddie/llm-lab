"""
여러 LLM 모델에 같은 프롬프트를 동시에 보내서 응답 비교하기 (A/B 테스트)

배우는 것:
- asyncio.gather()의 실전 활용
- httpx.AsyncClient 하나를 여러 호출에서 공유 (연결 풀 재사용)
- 모델별 응답 속도/품질 비교
"""

import asyncio
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


async def call_model(client: httpx.AsyncClient, model: str, prompt: str) -> dict:
    """
    특정 모델에 프롬프트를 보내고 결과를 dict로 반환

    client를 파라미터로 받는 이유:
    → 하나의 연결 풀을 여러 호출이 공유해서 효율적
    """

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
    }

    start = time.time()
    response = await client.post(OPENAI_URL, headers=headers, json=body, timeout=30.0)
    elapsed = time.time() - start

    if response.status_code != 200:
        return {"model": model, "error": response.text, "time": elapsed}

    data = response.json()
    return {
        "model": model,
        "answer": data["choices"][0]["message"]["content"],
        "tokens": data["usage"],
        "time": round(elapsed, 2),
    }


async def main():
    prompt = "Python에서 GIL(Global Interpreter Lock)이 뭔지 초보자에게 설명해줘. 2문장으로."

    # 비교할 모델 목록
    models = ["gpt-4.1-nano", "gpt-4.1-mini"]

    print(f"프롬프트: {prompt}")
    print(f"비교 모델: {models}")
    print("동시 호출 중...\n")

    total_start = time.time()

    # 핵심: client를 한 번만 만들고 모든 호출에서 공유
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[call_model(client, model, prompt) for model in models]
        )

    total_elapsed = round(time.time() - total_start, 2)

    # 결과 비교 출력
    for r in results:
        print(f"--- {r['model']} ({r['time']}초) ---")
        if "error" in r:
            print(f"  에러: {r['error']}")
        else:
            print(f"  응답: {r['answer']}")
            print(f"  토큰: 입력 {r['tokens']['prompt_tokens']} + 출력 {r['tokens']['completion_tokens']}")
        print()

    print(f"전체 소요 시간: {total_elapsed}초 (동시 호출이므로 가장 느린 모델 기준)")


if __name__ == "__main__":
    asyncio.run(main())
