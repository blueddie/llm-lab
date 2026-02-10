"""
비동기(Async) 방식 — 대기 시간에 다른 작업 처리
동일한 LLM API 호출을 asyncio로 처리
"""
import asyncio
import time


async def call_llm(user_id: int) -> str:
    """LLM API 호출을 흉내내는 비동기 함수 (2초 걸린다고 가정)"""
    print(f"[유저 {user_id}] API 요청 보냄...")
    await asyncio.sleep(2)  # time.sleep 대신 await asyncio.sleep
    print(f"[유저 {user_id}] 응답 받음!")
    return f"유저 {user_id}의 응답"


async def main():
    start = time.time()

    # 3명의 유저 요청을 동시에 보냄
    results = await asyncio.gather(
        call_llm(1),
        call_llm(2),
        call_llm(3),
    )

    elapsed = time.time() - start
    print(f"\n총 소요 시간: {elapsed:.1f}초")
    print(f"결과: {list(results)}")


if __name__ == "__main__":
    asyncio.run(main())
