"""
비동기 에러 처리 패턴
LLM API 호출은 실패할 수 있다 — 타임아웃, rate limit, 서버 에러 등
"""
import asyncio
import time
import random


async def call_llm(user_id: int) -> str:
    """랜덤하게 실패하는 LLM API 호출"""
    print(f"[user {user_id}] request...")
    await asyncio.sleep(1)

    # 50% 확률로 실패
    if random.random() < 0.5:
        raise Exception(f"user {user_id}: API error!")

    print(f"[user {user_id}] response!")
    return f"user {user_id} result"


# ============================================================
# 문제: gather에서 하나가 실패하면 전부 날아감
# ============================================================
async def dangerous():
    print("=== gather without error handling ===")
    try:
        results = await asyncio.gather(
            call_llm(1),
            call_llm(2),
            call_llm(3),
        )
        print(f"results: {list(results)}")
    except Exception as e:
        # 하나라도 실패하면 여기로 — 성공한 결과도 못 받음!
        print(f"FAILED: {e}")
        print("success results? LOST!\n")


# ============================================================
# 해결: return_exceptions=True — 에러도 결과로 받기
# ============================================================
async def safe():
    print("=== gather with return_exceptions=True ===")
    results = await asyncio.gather(
        call_llm(1),
        call_llm(2),
        call_llm(3),
        return_exceptions=True,  # 에러가 나도 멈추지 않고 결과에 포함
    )

    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"  user {i}: FAILED - {result}")
        else:
            print(f"  user {i}: OK - {result}")


async def main():
    random.seed(42)  # 결과 고정

    await dangerous()
    print()
    await safe()


if __name__ == "__main__":
    asyncio.run(main())
