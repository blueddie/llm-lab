"""
Function Calling 기본 예제
- LLM이 사용자 질문을 보고, 적절한 함수를 선택해서 호출하는 흐름
- 핵심: LLM은 함수를 "직접 실행"하지 않고, "호출 요청(JSON)"만 생성한다
"""

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# =============================================================================
# 1단계: 실제 함수 정의
# - 이건 우리가 만드는 평범한 Python 함수
# - 실제로는 API 호출, DB 조회 등 뭐든 될 수 있음
# =============================================================================


def get_weather(city: str) -> dict:
    """도시의 날씨를 조회하는 함수 (여기선 가짜 데이터 반환)"""
    # 실제로는 기상청 API 등을 호출하겠지만, 학습용이므로 하드코딩
    fake_data = {
        "서울": {"temp": 15, "condition": "맑음", "humidity": 45},
        "부산": {"temp": 18, "condition": "흐림", "humidity": 65},
        "제주": {"temp": 20, "condition": "비", "humidity": 80},
    }
    return fake_data.get(city, {"temp": 0, "condition": "알 수 없음", "humidity": 0})


# =============================================================================
# 2단계: 도구 스키마 정의 (tools)
# - LLM에게 "이런 함수를 쓸 수 있어"라고 알려주는 명세서
# - LLM은 이 스키마만 보고 언제, 어떤 인자로 호출할지 판단함
# - 그래서 description을 명확하게 쓰는 게 매우 중요!
# =============================================================================

tools = [
    {
        "type": "function",
        "function": {
            # 함수 이름 - LLM이 호출할 때 이 이름을 사용
            "name": "get_weather",
            # 함수 설명 - LLM이 "이 함수를 써야 하나?" 판단하는 핵심 근거
            "description": "특정 도시의 현재 날씨 정보(기온, 날씨 상태, 습도)를 조회한다.",
            # 파라미터 스키마 - JSON Schema 형식
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "날씨를 조회할 도시 이름 (예: 서울, 부산, 제주)",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

# =============================================================================
# 3단계: 대화 루프
# - 사용자 메시지 → LLM → (도구 호출 판단) → 함수 실행 → 결과 전달 → 최종 응답
# =============================================================================


def chat(user_message: str):
    print(f"\n{'='*50}")
    print(f"사용자: {user_message}")
    print(f"{'='*50}")

    messages = [{"role": "user", "content": user_message}]

    # 1) LLM에게 메시지 + 도구 목록을 보냄
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,  # <-- 여기서 도구를 알려줌
    )

    assistant_message = response.choices[0].message

    # 2) LLM의 판단 확인: 도구를 호출하려 하는가?
    if assistant_message.tool_calls:
        print(f"\n[LLM 판단] 도구 호출 필요!")

        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"  - 호출할 함수: {func_name}")
            print(f"  - 인자: {func_args}")

            # 3) 실제 함수 실행
            if func_name == "get_weather":
                result = get_weather(**func_args)
            else:
                result = {"error": f"알 수 없는 함수: {func_name}"}

            print(f"  - 실행 결과: {result}")

            # 4) 함수 실행 결과를 대화에 추가
            messages.append(assistant_message)  # LLM의 도구 호출 메시지
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,  # 어떤 호출에 대한 응답인지 매칭
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        # 5) 함수 결과를 포함해서 LLM에게 다시 요청 → 최종 자연어 응답 생성
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        final_answer = final_response.choices[0].message.content

    else:
        # 도구 호출 없이 바로 응답 (도구가 필요 없는 질문)
        print(f"\n[LLM 판단] 도구 호출 불필요 - 바로 응답")
        final_answer = assistant_message.content

    print(f"\n최종 응답: {final_answer}")
    return final_answer


# =============================================================================
# 실행: 다양한 질문으로 테스트
# =============================================================================

if __name__ == "__main__":
    # 도구가 필요한 질문
    chat("서울 날씨 어때?")

    # 다른 도시
    chat("제주도 여행 가려는데 날씨 좀 알려줘")

    # 도구가 필요 없는 질문 - LLM이 도구를 안 쓰고 바로 답할까?
    chat("안녕! 넌 뭐하는 AI야?")

'''

==================================================
사용자: 서울 날씨 어때?
==================================================

[LLM 판단] 도구 호출 필요!
  - 호출할 함수: get_weather
  - 인자: {'city': '서울'}
  - 실행 결과: {'temp': 15, 'condition': '맑음', 'humidity': 45}

최종 응답: 현재 서울의 날씨는 맑고, 기온은 15도입니다. 습도는 45%입니다. 외출하기 좋은 날씨네요!

==================================================
사용자: 제주도 여행 가려는데 날씨 좀 알려줘
==================================================

[LLM 판단] 도구 호출 필요!
  - 호출할 함수: get_weather
  - 인자: {'city': '제주'}
  - 실행 결과: {'temp': 20, 'condition': '비', 'humidity': 80}

최종 응답: 현재 제주도의 날씨는 비가 오고 있으며, 기온은 20도, 습도는 80%입니다. 여행 계획을 세우실 때 우산이나 방수 옷을 준비하시는 것이 좋겠습니다. 안전한 여행 되세 요!

==================================================
사용자: 안녕! 넌 뭐하는 AI야?
==================================================

[LLM 판단] 도구 호출 불필요 - 바로 응답

최종 응답: 안녕하세요! 저는 다양한 정보와 도움을 제공하는 AI입니다. 날씨 정보, 일반 상식, 언어 번역, 데이터 분석 등 여러 주제에 대해 질문하실 수 있습니다. 어떻게 도와 드릴까요?
'''