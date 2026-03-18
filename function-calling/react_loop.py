"""
ReAct 패턴: 다중 턴 도구 호출 (체이닝)
- LLM이 도구 결과를 보고 → 다음 도구를 호출할지 판단하는 루프
- 이것이 Agent의 핵심 동작 원리
- 핵심: while 루프 안에서 LLM이 "더 호출할 도구가 있는지" 스스로 결정
"""

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# =============================================================================
# 도구 함수들 - 여행 플래너에 필요한 기능
# =============================================================================


def get_weather(city: str) -> dict:
    """도시의 날씨를 조회한다."""
    fake_data = {
        "서울": {"temp": 15, "condition": "맑음"},
        "도쿄": {"temp": 12, "condition": "비"},
        "방콕": {"temp": 33, "condition": "맑음"},
        "오사카": {"temp": 14, "condition": "맑음"},
    }
    return fake_data.get(city, {"temp": 0, "condition": "알 수 없음"})


def search_attractions(city: str, category: str) -> dict:
    """도시의 관광지를 카테고리별로 검색한다."""
    fake_data = {
        ("도쿄", "실내"): [
            {"name": "도쿄 국립박물관", "rating": 4.5},
            {"name": "teamLab Borderless", "rating": 4.8},
            {"name": "아키하바라 전자상가", "rating": 4.2},
        ],
        ("도쿄", "야외"): [
            {"name": "메이지 신궁", "rating": 4.6},
            {"name": "우에노 공원", "rating": 4.3},
            {"name": "오다이바 해변공원", "rating": 4.1},
        ],
        ("방콕", "야외"): [
            {"name": "왓 아룬", "rating": 4.7},
            {"name": "짜뚜짝 시장", "rating": 4.4},
        ],
    }
    return {
        "city": city,
        "category": category,
        "attractions": fake_data.get((city, category), [{"name": "정보 없음"}]),
    }


def get_exchange_rate(base: str, target: str) -> dict:
    """환율을 조회한다."""
    fake_rates = {
        ("KRW", "JPY"): 0.11,
        ("KRW", "THB"): 0.026,
        ("KRW", "USD"): 0.00075,
    }
    return {"base": base, "target": target, "rate": fake_rates.get((base, target), 0.0)}


def search_restaurants(city: str, cuisine: str) -> dict:
    """도시의 맛집을 검색한다."""
    fake_data = {
        ("도쿄", "라멘"): [
            {"name": "이치란 라멘", "rating": 4.6, "price": "보통"},
            {"name": "후쿠멘", "rating": 4.3, "price": "저렴"},
        ],
        ("도쿄", "스시"): [
            {"name": "스시 다이", "rating": 4.8, "price": "비쌈"},
            {"name": "츠키지 스시", "rating": 4.4, "price": "보통"},
        ],
    }
    return {
        "city": city,
        "cuisine": cuisine,
        "restaurants": fake_data.get((city, cuisine), [{"name": "정보 없음"}]),
    }


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "search_attractions": search_attractions,
    "get_exchange_rate": get_exchange_rate,
    "search_restaurants": search_restaurants,
}

# =============================================================================
# 도구 스키마
# =============================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨(기온, 날씨 상태)를 조회한다. 여행 전 날씨를 확인해서 일정을 계획하는 데 사용.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "도시 이름"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description": "도시의 관광지를 카테고리별로 검색한다. 날씨에 따라 실내/야외를 선택할 수 있다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "도시 이름"},
                    "category": {
                        "type": "string",
                        "description": "관광지 카테고리",
                        "enum": ["실내", "야외", "전체"],
                    },
                },
                "required": ["city", "category"],
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
                    "base": {"type": "string", "description": "기준 통화 코드 (예: KRW)"},
                    "target": {"type": "string", "description": "대상 통화 코드 (예: JPY)"},
                },
                "required": ["base", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "도시의 맛집을 요리 종류별로 검색한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "도시 이름"},
                    "cuisine": {"type": "string", "description": "요리 종류 (예: 라멘, 스시, 태국음식)"},
                },
                "required": ["city", "cuisine"],
            },
        },
    },
]


# =============================================================================
# ReAct 루프 - Agent의 핵심!
# - while 루프 안에서 LLM이 도구를 호출할지 말지 매번 판단
# - 도구 호출이 없으면 = LLM이 "충분하다"고 판단한 것 = 루프 종료
# =============================================================================

MAX_ITERATIONS = 10  # 무한 루프 방지용 안전장치


def agent_loop(user_message: str):
    print(f"\n{'='*60}")
    print(f"사용자: {user_message}")
    print(f"{'='*60}")

    messages = [
        {
            "role": "system",
            "content": (
                "너는 여행 플래너 AI야. 사용자의 여행 계획을 도와줘.\n"
                "반드시 도구를 활용해서 정보를 단계적으로 수집해. 사용자에게 되묻지 말고 바로 행동해.\n"
                "모르는 정보는 합리적으로 가정하고 진행해.\n"
                "다음 순서로 도구를 사용해:\n"
                "1. 먼저 날씨를 확인하고\n"
                "2. 날씨에 따라 적절한 관광지를 추천하고 (비 오면 실내, 맑으면 야외)\n"
                "3. 환율 정보를 조회하고\n"
                "4. 맛집도 추천해줘\n"
                "모든 정보를 충분히 수집한 후에 종합적인 여행 계획을 제시해."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n--- 반복 {iteration} ---")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
        )

        assistant_message = response.choices[0].message

        # 도구 호출이 없으면 = LLM이 "충분하다"고 판단 = 루프 종료
        if not assistant_message.tool_calls:
            print(f"[루프 종료] LLM이 최종 응답 생성 (총 {iteration}번 반복)")
            print(f"\n최종 응답:\n{assistant_message.content}")
            messages.append(assistant_message)
            return messages

        # 도구 호출이 있으면 = 아직 정보가 더 필요하다는 뜻
        print(f"[도구 호출] {len(assistant_message.tool_calls)}개")
        messages.append(assistant_message)

        for tc in assistant_message.tool_calls:
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)
            print(f"  -> {func_name}({func_args})")

            func = TOOL_FUNCTIONS.get(func_name)
            result = func(**func_args) if func else {"error": "알 수 없는 함수"}
            print(f"  <- {json.dumps(result, ensure_ascii=False)[:150]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    print(f"[경고] 최대 반복 횟수({MAX_ITERATIONS}) 도달!")
    return messages


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    # LLM이 날씨를 보고 → 실내/야외를 판단 → 관광지 검색 → 환율 → 맛집 순서로
    # 단계적으로 도구를 호출하는지 관찰!
    agent_loop("도쿄 여행 계획 짜줘! 2박 3일이야.")

'''

============================================================
사용자: 도쿄 여행 계획 짜줘! 2박 3일이야.
============================================================

--- 반복 1 ---
[도구 호출] 1개
  -> get_weather({'city': '도쿄'})
  <- {"temp": 12, "condition": "비"}

--- 반복 2 ---
[도구 호출] 3개
  -> search_attractions({'city': '도쿄', 'category': '실내'})
  <- {"city": "도쿄", "category": "실내", "attractions": [{"name": "도쿄 국립박물관", "rating": 4.5}, {"name": "teamLab Borderless", "rating": 4.8}, {"name": "아키하바라 전
  -> get_exchange_rate({'base': 'KRW', 'target': 'JPY'})
  <- {"base": "KRW", "target": "JPY", "rate": 0.11}
  -> search_restaurants({'city': '도쿄', 'cuisine': '일본식'})
  <- {"city": "도쿄", "cuisine": "일본식", "restaurants": [{"name": "정보 없음"}]}

--- 반복 3 ---
[루프 종료] LLM이 최종 응답 생성 (총 3번 반복)

최종 응답:
도쿄에서의 2박 3일 여행 계획을 아래와 같이 제안합니다. 현재 도쿄의 날씨는 비가 오는 상태입니다. 따라서 실내 관광지를 중심으로 계획하겠습니다.

### 여행 계획

**1일차: 도착 및 실내 관광**
- **오후**
  - **도쿄 국립박물관**: 일본의 역사와 문화를 배울 수 있는 곳. 관람 시간: 약 2~3시간.
    - 평점: 4.5
  - **teamLab Borderless**: 디지털 아트를 즐길 수 있는 독특한 공간. 관람 시간: 약 2시간.
    - 평점: 4.8
- **저녁**
  - **식사**: 일본식 음식 맛집 (추천: 현지인 추천 맛집) 유익한 정보를 기반으로 맛집을 찾기 어려울 경우, 인근의 일반적인 일본식 레스토랑을 이용하세요.
  
**2일차: 실내 관광 및 쇼핑**
- **오전**
  - **아키하바라 전자상가**: 전자제품, 애니메이션 관련 상품 등을 쇼핑할 수 있는 장소. 관람 시간: 2~3시간.
    - 평점: 4.2
- **오후**
  - 추가적인 실내 관광 장소나 쇼핑을 계획할 수 있습니다.

**3일차: 귀국 준비 및 마지막 관광**
- **오전**
  - 자유롭게 실내 카페나 쇼핑센터를 방문.

### 환율 정보
- **1 KRW = 0.11 JPY**

### 추가 사항
더 많은 정보나 특정 맛집에 대한 정보를 원하시면, 현지에서 위생과 안전을 고려하여 선택하시길 바랍니다.

즐거운 도쿄 여행 되세요!
'''