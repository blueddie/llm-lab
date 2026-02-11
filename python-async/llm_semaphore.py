"""
Semaphore로 동시 요청 수 제한하기

문제: gather로 요청 20개를 한꺼번에 보내면 Rate Limit(429)에 걸림
해결: Semaphore로 "동시에 최대 N개만" 실행되도록 제한

배우는 것:
- asyncio.Semaphore의 동작 원리
- async with의 또 다른 활용 (자원 획득/반납)
- 실무에서 Rate Limit 대응하는 기본 패턴
"""

import asyncio
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# 동시에 최대 3개까지만 API 호출 허용
MAX_CONCURRENT = 1
sem = asyncio.Semaphore(MAX_CONCURRENT)


async def call_llm(client: httpx.AsyncClient, task_id: int, prompt: str) -> dict:
    """Semaphore로 보호된 LLM 호출"""

    # sem 획득 대기 → 자리가 나면 진입 → 블록 끝나면 자동 반납
    async with sem:
        print(f"  [작업 {task_id:2d}] 호출 시작 (현재 시각: {time.strftime('%H:%M:%S')})")

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }

        body = {
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30,  # 짧게 — 속도 확인이 목적
        }

        start = time.time()
        response = await client.post(OPENAI_URL, headers=headers, json=body, timeout=30.0)
        elapsed = round(time.time() - start, 2)

        if response.status_code != 200:
            print(f"  [작업 {task_id:2d}] 에러! status={response.status_code}")
            return {"task_id": task_id, "error": response.status_code, "time": elapsed}

        answer = response.json()["choices"][0]["message"]["content"]
        print(f"  [작업 {task_id:2d}] 완료 ({elapsed}초)")
        return {"task_id": task_id, "answer": answer, "time": elapsed}


async def main():
    # 10개의 서로 다른 질문
    prompts = [
        "Python을 만든 사람은?",
        "HTTP 상태코드 404의 의미는?",
        "JSON이 뭔지 한 줄로",
        "REST API란?",
        "Git이 뭔지 한 줄로",
        "Docker를 쓰는 이유는?",
        "SQL과 NoSQL의 차이는?",
        "CI/CD가 뭔지 한 줄로",
        "환경변수를 쓰는 이유는?",
        "비동기 프로그래밍이란?",
    ]

    total = len(prompts)
    print(f"총 {total}개 요청, 동시 최대 {MAX_CONCURRENT}개 제한\n")

    total_start = time.time()

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[call_llm(client, i + 1, p) for i, p in enumerate(prompts)]
        )

    total_elapsed = round(time.time() - total_start, 2)

    # 결과 요약
    print(f"\n=== 결과 요약 ===")
    success = [r for r in results if "answer" in r]
    errors = [r for r in results if "error" in r]
    times = [r["time"] for r in results]

    print(f"성공: {len(success)}개 / 실패: {len(errors)}개")
    print(f"개별 호출 시간: 최소 {min(times)}초, 최대 {max(times)}초")
    print(f"전체 소요 시간: {total_elapsed}초")
    print()

    # Semaphore 없이 전부 동시에 보냈다면?
    print(f"만약 제한 없이 {total}개를 동시에 보냈다면:")
    print(f"  → Rate Limit(429) 에러 발생 가능")
    print(f"만약 순차 실행했다면:")
    print(f"  → 약 {round(sum(times), 1)}초 소요 (각 호출 시간의 합)")
    print(f"Semaphore({MAX_CONCURRENT})로 제한한 결과:")
    print(f"  → {total_elapsed}초 (안전하면서도 빠르게)")


if __name__ == "__main__":
    asyncio.run(main())
