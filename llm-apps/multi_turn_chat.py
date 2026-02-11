"""
멀티턴 대화 — LLM이 이전 대화를 "기억"하는 원리

LLM은 상태가 없다 (stateless):
- 매 요청이 독립적
- "기억"하려면 매번 전체 대화 기록을 보내야 함

messages 배열의 role:
- system: LLM의 성격/규칙 (한 번만 설정)
- user: 사용자 메시지
- assistant: LLM의 이전 응답 (다음 요청에 포함시켜야 맥락 유지)
"""

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def chat(messages: list[dict], user_input: str) -> str:
    """
    대화 기록에 새 메시지를 추가하고 LLM 응답을 받는 함수

    핵심: messages 리스트가 점점 길어짐
    → 매 요청마다 전체 대화를 보내는 것
    → 이게 "기억"의 정체
    """

    # 사용자 메시지 추가
    messages.append({"role": "user", "content": user_input})

    response = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,  # 전체 대화 기록을 매번 전송
        max_tokens=300,
    )

    assistant_message = response.choices[0].message.content

    # LLM 응답도 기록에 추가 — 다음 턴에서 맥락으로 사용됨
    messages.append({"role": "assistant", "content": assistant_message})

    # 토큰 사용량 출력 — 대화가 길어질수록 증가하는 걸 관찰
    print(f"  [토큰] 입력: {response.usage.prompt_tokens}, 출력: {response.usage.completion_tokens}")

    return assistant_message


async def main():
    # system 메시지 — LLM의 성격/규칙을 여기서 설정
    messages = [
        {
            "role": "system",
            "content": "너는 Python 전문가야. 간결하게 답변해. 한국어로 답변해.",
        }
    ]

    print("=== 멀티턴 대화 시작 (종료: quit) ===\n")

    while True:
        user_input = input("나: ")
        if user_input.strip().lower() == "quit":
            break

        response = await chat(messages, user_input)
        print(f"AI: {response}\n")

    # 대화 끝난 후 — messages가 어떻게 쌓였는지 확인
    print(f"\n=== 대화 기록 (총 {len(messages)}개 메시지) ===")
    for msg in messages:
        role = msg["role"]
        content = msg["content"][:50]  # 앞 50자만
        print(f"  [{role:9s}] {content}...")


if __name__ == "__main__":
    asyncio.run(main())

# 나: 오늘은 볶음밥, 내일은 짬뽕
#   [토큰] 입력: 47, 출력: 33
# AI: 좋아, 오늘은 볶음밥, 내일은 짬뽕! 혹시 요리 레시피가 필요하면 알려줘.

# 나: 내가 내일 뭐 먹는다고 했지
#   [토큰] 입력: 98, 출력: 12
# AI: 내일은 짬뽕이라고 했어.

# 나: 아니야 내일은 냉면이야
#   [토큰] 입력: 129, 출력: 20
# AI: 아, 내일은 냉면이군요! 알려줘서 고마워요.

# 나: 오늘 뭐 먹지?
#   [토큰] 입력: 162, 출력: 17
# AI: 오늘은 볶음밥이군요! 맛있게 드세요!

# 나: 내일은?
#   [토큰] 입력: 191, 출력: 15
# AI: 내일은 냉면이네요! 맛있게 드세요!