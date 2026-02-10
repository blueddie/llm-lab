"""
동기(Sync) 방식 — 순차적으로 하나씩 처리
LLM API 호출을 time.sleep()으로 흉내냄
"""
import time


def call_llm(user_id: int) -> str:
    """LLM API 호출을 흉내내는 함수 (2초 걸린다고 가정)"""
    print(f"[유저 {user_id}] API 요청 보냄...")
    time.sleep(2)  # 네트워크 대기 시간
    print(f"[유저 {user_id}] 응답 받음!")
    return f"유저 {user_id}의 응답"


def main():
    start = time.time()

    # 3명의 유저가 동시에 요청했다고 가정
    results = []
    for user_id in range(1, 4):
        result = call_llm(user_id)
        results.append(result)

    elapsed = time.time() - start
    print(f"\n총 소요 시간: {elapsed:.1f}초")
    print(f"결과: {results}")


if __name__ == "__main__":
    main()
