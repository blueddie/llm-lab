"""
실패한 요청만 재시도(retry)하는 패턴
실무에서 LLM API 호출 시 거의 필수
"""
import asyncio
import random


call_count = {}  # 각 유저별 호출 횟수 추적


async def call_llm(user_id: int) -> str:
    call_count[user_id] = call_count.get(user_id, 0) + 1
    attempt = call_count[user_id]
    print(f"[user {user_id}] attempt {attempt}...")
    await asyncio.sleep(1)

    # 첫 시도는 50% 실패, 두 번째부터는 성공
    if attempt == 1 and random.random() < 0.5:
        raise Exception(f"user {user_id}: API error!")

    return f"user {user_id} result"


async def call_llm_with_retry(user_id: int, max_retries: int = 3) -> str:
    """개별 함수에 retry 로직을 감싸는 패턴"""
    for attempt in range(max_retries):
        try:
            return await call_llm(user_id)
        except Exception as e:
            print(f"  -> FAILED: {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1초, 2초, 4초... (exponential backoff)
                print(f"  -> retry in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  -> max retries reached!")
                raise  # 최종 실패는 위로 던짐


async def main():
    random.seed(42)

    results = await asyncio.gather(
        call_llm_with_retry(1),
        call_llm_with_retry(2),
        call_llm_with_retry(3),
        return_exceptions=True,
    )

    print("\n=== final results ===")
    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"  user {i}: FAILED")
        else:
            print(f"  user {i}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
