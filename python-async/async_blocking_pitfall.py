"""
잘못된 비동기 — async def 안에서 time.sleep 사용
await로 양보하지 않으면 동기 코드와 똑같아진다
"""
import asyncio
import time


async def call_llm(user_id: int) -> str:
    print(f"[user {user_id}] API request...")
    time.sleep(2)  # await 없이 그냥 멈춤 -> 양보 안 함!
    print(f"[user {user_id}] response!")
    return f"user {user_id} response"


async def main():
    start = time.time()

    results = await asyncio.gather(
        call_llm(1),
        call_llm(2),
        call_llm(3),
    )

    elapsed = time.time() - start
    print(f"\ntotal: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
