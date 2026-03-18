"""
다중 도구 Function Calling 예제
- 도구가 여러 개일 때 LLM이 어떻게 선택하는지 관찰
- 핵심 포인트:
  1. LLM은 도구의 description을 보고 적절한 도구를 선택한다
  2. 한 번에 여러 도구를 동시에 호출할 수도 있다 (parallel tool calls)
  3. 어떤 도구도 필요 없으면 그냥 답변한다
"""

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# =============================================================================
# 1단계: 함수 3개 정의 - 여행 도우미에 필요한 기능들
# =============================================================================


def get_weather(city: str) -> dict:
    """도시의 날씨를 조회한다."""
    fake_data = {
        "서울": {"temp": 15, "condition": "맑음"},
        "도쿄": {"temp": 12, "condition": "흐림"},
        "방콕": {"temp": 33, "condition": "맑음"},
    }
    return fake_data.get(city, {"temp": 0, "condition": "알 수 없음"})


def search_flights(origin: str, destination: str, date: str) -> dict:
    """항공편을 검색한다."""
    # 실제로는 스카이스캐너 API 등을 호출
    fake_flights = [
        {"airline": "대한항공", "price": 350000, "departure": "08:00", "arrival": "10:30"},
        {"airline": "아시아나", "price": 320000, "departure": "14:00", "arrival": "16:30"},
    ]
    return {"origin": origin, "destination": destination, "date": date, "flights": fake_flights}


def get_exchange_rate(base: str, target: str) -> dict:
    """환율을 조회한다."""
    fake_rates = {
        ("KRW", "JPY"): 0.11,
        ("KRW", "THB"): 0.026,
        ("KRW", "USD"): 0.00075,
    }
    rate = fake_rates.get((base, target), 0.0)
    return {"base": base, "target": target, "rate": rate}


# =============================================================================
# 2단계: 도구 스키마 3개 정의
# - description이 곧 LLM의 "판단 근거"
# - 모호하게 쓰면 LLM이 엉뚱한 도구를 선택할 수 있음
# =============================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨(기온, 날씨 상태)를 조회한다. 여행 전 날씨 확인에 사용.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "도시 이름 (예: 서울, 도쿄, 방콕)",
                    }
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
                    "origin": {
                        "type": "string",
                        "description": "출발 도시 (예: 서울)",
                    },
                    "destination": {
                        "type": "string",
                        "description": "도착 도시 (예: 도쿄)",
                    },
                    "date": {
                        "type": "string",
                        "description": "출발 날짜 (YYYY-MM-DD 형식)",
                    },
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
                    "base": {
                        "type": "string",
                        "description": "기준 통화 코드 (예: KRW, USD, JPY)",
                    },
                    "target": {
                        "type": "string",
                        "description": "대상 통화 코드 (예: JPY, THB, USD)",
                    },
                },
                "required": ["base", "target"],
            },
        },
    },
]

# 함수 이름 -> 실제 함수 매핑 (if-else 대신 딕셔너리로 관리)
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "search_flights": search_flights,
    "get_exchange_rate": get_exchange_rate,
}

# =============================================================================
# 3단계: 대화 루프 (여러 도구 호출 대응)
# =============================================================================


def chat(user_message: str):
    print(f"\n{'='*60}")
    print(f"사용자: {user_message}")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": "너는 여행 도우미 AI야. 도구를 활용해서 정확한 정보를 제공해."},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
    )

    assistant_message = response.choices[0].message

    if assistant_message.tool_calls:
        # 핵심: tool_calls가 여러 개일 수 있음! (parallel tool calls)
        print(f"\n[LLM 판단] 도구 {len(assistant_message.tool_calls)}개 호출!")

        # LLM의 응답을 먼저 대화에 추가
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"  [{func_name}] 인자: {func_args}")

            # 딕셔너리에서 함수 찾아서 실행
            func = TOOL_FUNCTIONS.get(func_name)
            if func:
                result = func(**func_args)
            else:
                result = {"error": f"알 수 없는 함수: {func_name}"}

            print(f"  [{func_name}] 결과: {result}")

            # 각 도구 호출의 결과를 대화에 추가
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # 모든 도구 결과를 포함해서 최종 응답 생성
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        final_answer = final_response.choices[0].message.content
    else:
        print(f"\n[LLM 판단] 도구 호출 불필요")
        final_answer = assistant_message.content

    print(f"\n최종 응답: {final_answer}")
    return final_answer


# =============================================================================
# 테스트: LLM의 도구 선택 판단력 관찰
# =============================================================================

if __name__ == "__main__":
    # 테스트 1: 도구 1개만 필요한 질문
    chat("도쿄 날씨 어때?")

    # 테스트 2: 도구 여러 개가 동시에 필요한 질문 (parallel tool calls 관찰!)
    chat("다음 주 금요일에 서울에서 도쿄 가려는데, 항공편이랑 환율 알려줘")

    # 테스트 3: 도구가 필요 없는 질문
    chat("여행 갈 때 짐 싸는 팁 알려줘")

'''
============================================================
사용자: 도쿄 날씨 어때?
============================================================

[LLM 판단] 도구 1개 호출!
  [get_weather] 인자: {'city': '도쿄'}
  [get_weather] 결과: {'temp': 12, 'condition': '흐림'}

최종 응답: 오늘 도쿄의 날씨는 흐리고, 기온은 약 12도입니다. 외출하실 때 따뜻한 옷을 챙기는 것이 좋겠습니다!

============================================================
사용자: 다음 주 금요일에 서울에서 도쿄 가려는데, 항공편이랑 환율 알려줘
============================================================

[LLM 판단] 도구 2개 호출!
  [search_flights] 인자: {'origin': '서울', 'destination': '도쿄', 'date': '2023-11-03'}
  [search_flights] 결과: {'origin': '서울', 'destination': '도쿄', 'date': '2023-11-03', 'flights': [{'airline': '대한항공', 'price': 350000, 'departure': '08:00', 'arrival': '10:30'}, {'airline': '아시아나', 'price': 320000, 'departure': '14:00', 'arrival': '16:30'}]}
  [get_exchange_rate] 인자: {'base': 'KRW', 'target': 'JPY'}
  [get_exchange_rate] 결과: {'base': 'KRW', 'target': 'JPY', 'rate': 0.11}

최종 응답: 다음 주 금요일(2023년 11월 3일)에 서울에서 도쿄로 가는 항공편 정보는 다음과 같습니다:

1. **대한항공**
   - 출발: 08:00
   - 도착: 10:30
   - 가격: 350,000 KRW

2. **아시아나**
   - 출발: 14:00
   - 도착: 16:30
   - 가격: 320,000 KRW

환율 정보는 현재 1 KRW = 0.11 JPY입니다.

추가로 궁금한 점이 있으면 말씀해 주세요!

============================================================
사용자: 여행 갈 때 짐 싸는 팁 알려줘
============================================================

[LLM 판단] 도구 호출 불필요

최종 응답: 여행 갈 때 짐 싸는 팁은 다음과 같습니다:

1. **목록 작성**: 필요한 물품을 미리 목록으로 작성하여 빠뜨리는 것이 없도록 합니다.

2. **다용도 아이템 활용**: 여러 용도로 사용할 수 있는 옷이나 액세서리를 선택하세요. 예를 들어, 스카프는 패션 아이템이자 보온용 담요로 사용할 수 있습니다.

3. **옷은 롤로 싸기**: 옷을 말아서 싸면 공간을 절약하고 주름도 최소화 할 수 있습니다.

4. **신발 포장**: 신발은 내부에 양말이나 작고 부피가 작은 물건을 넣어서 공간을 활용하세요. 또한 신발은 여행 가방 바닥에 넣어 무게의 중심을 낮춥니다.

5. **세면도구와 의약품**: 여행용 사이즈의 용기에 필요한 세면도구를 미리 담고, 필요한 의약품은 잊지 말고 챙기세요.

6. **전자기기와 충전기**: 카메라, 휴대폰, 노트북 등 필요한 전자기기와 충전기를 함께 챙기세요. 만약 여러 개의 기기를 가져간다면 멀티탭을 이용하면 편리합니다.

7. **비상용 간식**: 비행기나 기차에서 간단히 먹을 수 있는 간식을 챙기세요.

8. **가벼운 짐**: 필요하지 않은 물건은 최대한 줄이고, 여행이 끝난 후 돌아오면서 장을 볼 아이템도 고려해 가벼운 짐으로 설정합니다.

이 팁들을 활용하면 더 효율적으로 짐을 싸고, 여행을 더욱 즐길 수 있을 거예요!
'''