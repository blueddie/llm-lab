"""
동시 실행 패턴 비교: asyncio.gather() vs asyncio.create_task()
둘 다 여러 작업을 동시에 실행하지만, 사용 방식이 다르다
"""
import asyncio
import time


async def call_llm(user_id: int) -> str:
    print(f"[user {user_id}] request...")
    await asyncio.sleep(2)
    print(f"[user {user_id}] response!")
    return f"user {user_id} result"


# ============================================================
# 방법 1: asyncio.gather() — 여러 작업을 한 번에 묶어서 실행
# ============================================================
async def with_gather():
    print("=== gather ===")
    start = time.time()

    # 한 줄에 다 넣고, 전부 끝날 때까지 기다림
    results = await asyncio.gather(
        call_llm(1),
        call_llm(2),
        call_llm(3),
    )

    print(f"time: {time.time() - start:.1f}s")
    print(f"results: {list(results)}\n")


# ============================================================
# 방법 2: asyncio.create_task() — 작업을 하나씩 예약하고, 나중에 결과 수집
# ============================================================
async def with_create_task():
    print("=== create_task ===")
    start = time.time()

    # 1) 작업을 예약 — 이 시점에 바로 실행이 시작됨
    task1 = asyncio.create_task(call_llm(1))
    task2 = asyncio.create_task(call_llm(2))
    task3 = asyncio.create_task(call_llm(3))

    # 2) 각 task의 결과를 await로 받음
    result1 = await task1
    result2 = await task2
    result3 = await task3

    print(f"time: {time.time() - start:.1f}s")
    print(f"results: {[result1, result2, result3]}\n")


async def main():
    await with_gather()
    await with_create_task()


if __name__ == "__main__":
    asyncio.run(main())
