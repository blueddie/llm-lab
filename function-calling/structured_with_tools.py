"""
Structured Output + Function Calling 조합
- 도구 호출은 기존과 동일하게 처리
- 최종 응답을 정해진 JSON 구조로 반환받음
- 실무 시나리오: 여행 정보를 조회하고, 프론트엔드에서 쓸 수 있는 구조화된 응답 반환
"""

import json
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


# =============================================================================
# 1단계: 도구 함수 (기존과 동일)
# =============================================================================


def get_weather(city: str) -> dict:
    fake_data = {
        "서울": {"temp": 15, "condition": "맑음"},
        "도쿄": {"temp": 12, "condition": "흐림"},
        "방콕": {"temp": 33, "condition": "맑음"},
    }
    return fake_data.get(city, {"temp": 0, "condition": "알 수 없음"})


def get_exchange_rate(base: str, target: str) -> dict:
    fake_rates = {
        ("KRW", "JPY"): 0.11,
        ("KRW", "THB"): 0.026,
        ("KRW", "USD"): 0.00075,
    }
    rate = fake_rates.get((base, target), 0.0)
    return {"base": base, "target": target, "rate": rate}


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_exchange_rate": get_exchange_rate,
}

# =============================================================================
# 2단계: 도구 스키마 (기존과 동일)
# =============================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨(기온, 날씨 상태)를 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "도시 이름 (예: 서울, 도쿄, 방콕)"}
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
                    "base": {"type": "string", "description": "기준 통화 코드 (예: KRW)"},
                    "target": {"type": "string", "description": "대상 통화 코드 (예: JPY)"},
                },
                "required": ["base", "target"],
            },
        },
    },
]


# =============================================================================
# 3단계: 최종 응답의 구조 정의 (Pydantic 모델)
# - 이게 이번 예제의 핵심!
# - LLM이 도구 결과를 종합한 후, 이 구조에 맞춰서 응답하게 됨
# =============================================================================


class TravelInfo(BaseModel):
    """여행 정보 응답 구조"""
    city: str                  # 조회한 도시
    summary: str               # 한 줄 요약
    weather_description: str   # 날씨 설명
    packing_suggestion: str    # 짐 싸기 제안
    tools_used: list[str]      # 사용된 도구 목록
    exchange_info: str | None  # 환율 정보 (없으면 null)


# =============================================================================
# 4단계: 도구 호출 → Structured Output 응답
# - 1차 호출: 도구 호출 (일반 모드)
# - 2차 호출: 도구 결과 포함해서 Structured Output 모드로 최종 응답
# =============================================================================


def chat(user_message: str):
    print(f"\n{'='*60}")
    print(f"사용자: {user_message}")
    print(f"{'='*60}")

    messages = [
        {
            "role": "system",
            "content": (
                "너는 여행 도우미 AI야. 도구를 활용해서 정보를 조회하고, "
                "결과를 정해진 형식에 맞춰 응답해."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    # --- 1차 호출: 도구 호출 판단 (일반 모드) ---
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
    )

    assistant_message = response.choices[0].message

    if assistant_message.tool_calls:
        print(f"\n  [1차] 도구 {len(assistant_message.tool_calls)}개 호출")
        messages.append(assistant_message)

        for tc in assistant_message.tool_calls:
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)
            print(f"    -> {func_name}({func_args})")

            func = TOOL_FUNCTIONS.get(func_name)
            result = func(**func_args) if func else {"error": "알 수 없는 함수"}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # --- 2차 호출: Structured Output으로 최종 응답 ---
        # response_format에 Pydantic 모델을 넘기면 그 구조대로 응답이 나옴
        print(f"  [2차] Structured Output으로 최종 응답 생성")
        final_response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=TravelInfo,  # <-- 핵심: 응답 구조 강제
        )

        parsed = final_response.choices[0].message.parsed
        print(f"\n  [결과] 구조화된 응답:")
        print(f"    city: {parsed.city}")
        print(f"    summary: {parsed.summary}")
        print(f"    weather_description: {parsed.weather_description}")
        print(f"    packing_suggestion: {parsed.packing_suggestion}")
        print(f"    tools_used: {parsed.tools_used}")
        print(f"    exchange_info: {parsed.exchange_info}")

        # JSON으로도 출력 (프론트엔드에서 이걸 받아서 쓰는 상황)
        print(f"\n  [JSON 출력]")
        print(f"  {parsed.model_dump_json(ensure_ascii=False, indent=2)}")

        return parsed
    else:
        print(f"\n  도구 호출 없이 응답: {assistant_message.content}")
        return None


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    # 날씨만 조회
    chat("도쿄 여행 가려는데 준비할 거 알려줘")

    # 날씨 + 환율 함께 조회
    chat("방콕 여행 준비 중이야. 날씨랑 환율 정보 줘")

'''
============================================================
사용자: 도쿄 여행 가려는데 준비할 거 알려줘
============================================================

  도구 호출 없이 응답: 도쿄 여행을 준비하면서 알아두면 좋은 팁과 필수 준비물들을 안내해 드리겠습니다.

### 여행 준비 목록

1. **여권 및 비자**
   - 일본 여행을 위해 유효한 여권을 준비하세요. 대부분의 국가에서는 비자 없이 90일 이하 체류가 가능합니다.

2. **항공권**
   - 항공권을 미리 예약해 두세요. 항공권 비교 사이트를 활용하면 저렴한 가격을 찾을 수 있습니다.

3. **숙박 예약**
   - 도쿄에는 다양한 숙박 옵션이 있으니 예산과 취향에 맞춰 미리 예약하세요 (호텔, 게스트하우스, 롯지 등).

4. **환전 및 예산**
   - 일본 엔(JPY)으로 환전하세요. 신용카드 사용도 가능하지만 현금을 준비하는 것이 좋습니다.

5. **교통카드**
   - 교통 혼잡을 줄이기 위해 '스이카(Suica)' 또는 '파스모(Pasmo)' 교통카드를 구입하면 편리합니다.

6. **관광지 정보**
   - 방문할 관광지를 미리 정리하세요. 도쿄의 주요 명소로는 센소지, 도쿄타워, 시부야 스크램블 교차로, 아키하바라 등이 있습니다.

7. **의류 및 개인 용품**
   - 여행 기간과 계절에 맞는 옷과 개인 용품을 챙기세요. 특히 일본은 계절에 따라 날씨 변화가 크니 확인이 필요합니다.

8. **모바일 데이터 및 통신**
   - 현지 SIM카드 또는 포켓 와이파이 렌탈을 준비하여 인터넷을 사용할 수 있게 하세요.

9. **필수 앱 다운로드**
   - 지도 앱, 번역 앱, 교통 앱 등을 미리 다운로드해 두세요.

### 추가 팁
- 일본의 문화와 예절을 미리 알아두면 여행이 더욱 풍성해질 수 있습니다.
- 음식도 다양하니 일본의 다양한 요리를 경험해 보세요. 스시, 라멘, 우동 등 꼭 맛보시길 추천합니다.

도쿄에서 즐거운 여행 되시길 바랍니다! 다른 정보가 필요하시면 언제든지 말씀해 주세요.

============================================================
사용자: 방콕 여행 준비 중이야. 날씨랑 환율 정보 줘
============================================================

  [1차] 도구 2개 호출
    -> get_weather({'city': '방콕'})
    -> get_exchange_rate({'base': 'KRW', 'target': 'THB'})
  [2차] Structured Output으로 최종 응답 생성

  [결과] 구조화된 응답:
    city: 방콕
    summary: 방콕은 태국의 수도로, 다양한 문화와 맛있는 음식, 그리고 이국적인 경치를 제공합니다.
    weather_description: 현재 방콕의 기온은 33도이며, 날씨는 맑습니다.
    packing_suggestion: 가벼운 여름 옷과 썬크림, 모자 등을 준비하세요.
    tools_used: ['functions.get_weather', 'functions.get_exchange_rate']
    exchange_info: 1 KRW는 약 0.026 THB입니다.

  [JSON 출력]
  {
  "city": "방콕",
  "summary": "방콕은 태국의 수도로, 다양한 문화와 맛있는 음식, 그리고 이국적인 경치를 제공합니다.",
  "weather_description": "현재 방콕의 기온은 33도이며, 날씨는 맑습니다.",
  "packing_suggestion": "가벼운 여름 옷과 썬크림, 모자 등을 준비하세요.",
  "tools_used": [
    "functions.get_weather",
    "functions.get_exchange_rate"
  ],
  "exchange_info": "1 KRW는 약 0.026 THB입니다."
}
'''