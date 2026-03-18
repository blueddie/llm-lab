"""
tool_choice 실험
- auto vs required vs 특정 함수 강제 지정
- 같은 질문에 tool_choice만 바꿔서 LLM의 행동 비교
"""

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def get_weather(city: str) -> dict:
    fake_data = {
        "서울": {"temp": 15, "condition": "맑음"},
        "도쿄": {"temp": 12, "condition": "흐림"},
    }
    return fake_data.get(city, {"temp": 0, "condition": "알 수 없음"})


def get_exchange_rate(base: str, target: str) -> dict:
    fake_rates = {("KRW", "JPY"): 0.11, ("KRW", "USD"): 0.00075}
    return {"rate": fake_rates.get((base, target), 0.0)}


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_exchange_rate": get_exchange_rate,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨(기온, 날씨 상태)를 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "도시 이름"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "두 통화 간의 환율을 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "기준 통화 코드"},
                    "target": {"type": "string", "description": "대상 통화 코드"},
                },
                "required": ["base", "target"],
            },
        },
    },
]


def test(label: str, user_message: str, tool_choice):
    print(f"\n{'='*60}")
    print(f"[{label}] 사용자: {user_message}")
    print(f"  tool_choice = {tool_choice}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  -> 도구 호출: {tc.function.name}({args})")
    else:
        print(f"  -> 도구 호출 없음")
        print(f"  -> 응답: {msg.content[:100]}")


if __name__ == "__main__":
    # ----- 테스트 1: 도구가 필요 없는 질문 -----
    print("\n" + "#" * 60)
    print("# 테스트 1: '안녕! 오늘 기분 어때?'")
    print("#" * 60)

    test("auto", "안녕! 오늘 기분 어때?", "auto")
    test("required", "안녕! 오늘 기분 어때?", "required")

    # ----- 테스트 2: 도구가 필요한 질문 -----
    print("\n" + "#" * 60)
    print("# 테스트 2: '도쿄 여행 준비할 거 알려줘'")
    print("#" * 60)

    test("auto", "도쿄 여행 준비할 거 알려줘", "auto")
    test("required", "도쿄 여행 준비할 거 알려줘", "required")

    # ----- 테스트 3: 특정 함수 강제 지정 -----
    print("\n" + "#" * 60)
    print("# 테스트 3: 특정 함수 강제 - 환율 질문에 날씨 도구 강제")
    print("#" * 60)

    test(
        "강제: get_weather",
        "원화를 엔화로 바꾸면 환율이 얼마야?",
        {"type": "function", "function": {"name": "get_weather"}},
    )
