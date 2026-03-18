"""
실험: 도구 description이 LLM의 판단에 미치는 영향
- 같은 함수, 같은 질문인데 description만 바꿔서 비교
- 좋은 description vs 모호한 description
"""

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


# 함수는 동일
def get_weather(city: str) -> dict:
    return {"temp": 15, "condition": "맑음"}


def search_flights(origin: str, destination: str, date: str) -> dict:
    return {"flights": [{"airline": "대한항공", "price": 350000}]}


def get_exchange_rate(base: str, target: str) -> dict:
    return {"rate": 0.11}


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "search_flights": search_flights,
    "get_exchange_rate": get_exchange_rate,
}

# =============================================================================
# A: 명확한 description
# =============================================================================

tools_good = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨(기온, 날씨 상태)를 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "도시 이름 (예: 서울, 도쿄)"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "출발지에서 목적지까지의 항공편을 검색한다. 날짜별 항공편 가격과 시간을 조회.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "출발 도시"},
                    "destination": {"type": "string", "description": "도착 도시"},
                    "date": {"type": "string", "description": "출발 날짜 (YYYY-MM-DD)"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "두 통화 간의 환율을 조회한다. 여행 경비 계산에 사용.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "기준 통화 코드 (예: KRW, USD)"},
                    "target": {"type": "string", "description": "대상 통화 코드 (예: JPY, THB)"},
                },
                "required": ["base", "target"],
            },
        },
    },
]

# =============================================================================
# B: 모호한 description
# =============================================================================

tools_vague = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "정보를 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "값"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "검색한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "값1"},
                    "destination": {"type": "string", "description": "값2"},
                    "date": {"type": "string", "description": "값3"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "데이터를 가져온다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "입력1"},
                    "target": {"type": "string", "description": "입력2"},
                },
                "required": ["base", "target"],
            },
        },
    },
]


# =============================================================================
# 실행 함수
# =============================================================================


def test_with_tools(label: str, tools: list, user_message: str):
    print(f"\n{'='*60}")
    print(f"[{label}] 사용자: {user_message}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
    )

    assistant_message = response.choices[0].message

    if assistant_message.tool_calls:
        for tc in assistant_message.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  호출: {tc.function.name}({args})")
    else:
        print(f"  도구 호출 없음 - 바로 응답")
        print(f"  응답: {assistant_message.content[:100]}...")


# =============================================================================
# 비교 실험
# =============================================================================

if __name__ == "__main__":
    test_questions = [
        "서울 날씨 알려줘",
        "서울에서 도쿄 가는 비행기 찾아줘",
        "원화를 엔화로 바꾸면 환율이 얼마야?",
    ]

    for question in test_questions:
        test_with_tools("좋은 description", tools_good, question)
        test_with_tools("모호한 description", tools_vague, question)
        print()  # 비교하기 쉽게 빈 줄
