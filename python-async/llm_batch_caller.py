"""
프로덕션급 LLM 배치 호출기

시나리오: 사용자 질문 목록을 받아 LLM으로 한꺼번에 처리
실무에서는 이런 상황이 흔함:
- CS 문의 자동 분류
- 대량 텍스트 요약
- 배치 번역

조합하는 패턴:
- httpx.AsyncClient — 연결 풀 공유
- Semaphore — 동시 요청 수 제한 (Rate Limit 대응)
- Retry + Exponential Backoff — 일시적 실패 자동 복구
- return_exceptions=True — 하나 실패해도 전체가 죽지 않음
"""

import asyncio
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# 설정값 — 실무에서는 환경변수나 config로 관리
MAX_CONCURRENT = 3   # 동시 최대 요청 수
MAX_RETRIES = 3      # 최대 재시도 횟수
MODEL = "gpt-4.1-nano"


async def call_llm_with_retry(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    task_id: int,
    prompt: str,
) -> dict:
    """
    Semaphore + Retry가 적용된 LLM 호출

    흐름:
    1. Semaphore 자리 대기
    2. API 호출 시도
    3. 실패하면 exponential backoff 후 재시도
    4. 최대 재시도 초과 시 에러 결과 반환 (예외를 던지지 않음)
    """

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
    }

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                start = time.time()
                response = await client.post(
                    OPENAI_URL, headers=headers, json=body, timeout=30.0
                )
                elapsed = round(time.time() - start, 2)

                # 429(Rate Limit) or 5xx(서버 에러) → 재시도 대상
                if response.status_code == 429 or response.status_code >= 500:
                    wait = 2 ** attempt
                    print(f"  [작업 {task_id:2d}] {response.status_code} 에러, {wait}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(wait)
                    continue

                # 그 외 에러 (400, 401 등) → 재시도 의미 없음
                if response.status_code != 200:
                    return {
                        "task_id": task_id,
                        "status": "error",
                        "error": f"HTTP {response.status_code}",
                        "time": elapsed,
                    }

                # 성공
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                tokens = data["usage"]["total_tokens"]
                print(f"  [작업 {task_id:2d}] 완료 ({elapsed}초, {tokens}토큰)")

                return {
                    "task_id": task_id,
                    "status": "success",
                    "answer": answer,
                    "tokens": tokens,
                    "time": elapsed,
                }

            except httpx.TimeoutException:
                wait = 2 ** attempt
                print(f"  [작업 {task_id:2d}] 타임아웃, {wait}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)

            except httpx.ConnectError as e:
                wait = 2 ** attempt
                print(f"  [작업 {task_id:2d}] 연결 실패, {wait}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)

        # 모든 재시도 실패
        print(f"  [작업 {task_id:2d}] 최대 재시도 초과!")
        return {
            "task_id": task_id,
            "status": "failed",
            "error": "최대 재시도 초과",
            "time": 0,
        }


async def batch_call(prompts: list[str]) -> list[dict]:
    """
    프롬프트 목록을 받아 배치로 처리하고 결과 반환

    이 함수가 배치 호출기의 진입점:
    1. Client, Semaphore 생성
    2. gather로 전체 호출을 스케줄링
    3. 결과 수집 후 반환
    """

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[
                call_llm_with_retry(client, sem, i + 1, prompt)
                for i, prompt in enumerate(prompts)
            ],
            return_exceptions=True,  # 예상 못한 예외도 결과로 받음
        )

    return results


def print_report(results: list, total_time: float):
    """결과 리포트 출력"""

    print("\n" + "=" * 50)
    print("배치 처리 결과 리포트")
    print("=" * 50)

    success = [r for r in results if isinstance(r, dict) and r["status"] == "success"]
    errors = [r for r in results if isinstance(r, dict) and r["status"] in ("error", "failed")]
    exceptions = [r for r in results if isinstance(r, Exception)]

    print(f"\n총 요청: {len(results)}개")
    print(f"성공: {len(success)}개 | 실패: {len(errors)}개 | 예외: {len(exceptions)}개")

    if success:
        total_tokens = sum(r["tokens"] for r in success)
        avg_time = round(sum(r["time"] for r in success) / len(success), 2)
        print(f"총 토큰: {total_tokens} | 평균 응답 시간: {avg_time}초")

    print(f"전체 소요 시간: {total_time}초")

    # 개별 결과
    print(f"\n--- 개별 결과 ---")
    for r in results:
        if isinstance(r, Exception):
            print(f"  [예외] {r}")
        elif r["status"] == "success":
            # 응답 첫 50자만 출력
            short = r["answer"][:50].replace("\n", " ")
            print(f"  [작업 {r['task_id']:2d}] {short}...")
        else:
            print(f"  [작업 {r['task_id']:2d}] {r['status']}: {r['error']}")


async def main():
    # 배치 처리할 프롬프트 목록
    prompts = [
        "Python의 GIL을 한 문장으로 설명해줘.",
        "REST API와 GraphQL의 차이를 한 문장으로.",
        "Docker 컨테이너와 VM의 차이는?",
        "JWT 토큰이 뭔지 한 문장으로.",
        "SQL에서 인덱스가 왜 필요한지 한 문장으로.",
        "Git에서 rebase와 merge의 차이는?",
        "마이크로서비스의 장단점을 한 문장으로.",
        "캐싱이 성능을 높이는 원리를 한 문장으로.",
    ]

    print(f"배치 호출 시작: {len(prompts)}개 프롬프트")
    print(f"설정: 동시 {MAX_CONCURRENT}개, 최대 재시도 {MAX_RETRIES}회, 모델 {MODEL}\n")

    start = time.time()
    results = await batch_call(prompts)
    total_time = round(time.time() - start, 2)

    print_report(results, total_time)


if __name__ == "__main__":
    asyncio.run(main())
