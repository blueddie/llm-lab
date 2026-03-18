"""
실용 예제: 리서치 어시스턴트
- 진짜 API를 호출하는 도구들로 구성
- Wikipedia 검색, 계산기, 번역 3개 도구
- 대화형 루프로 여러 번 질문 가능
"""

import json
import math
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


# =============================================================================
# 도구 1: Wikipedia 검색 (실제 API 호출)
# - 위키피디아 REST API는 무료, 키 불필요
# - 검색어로 문서를 찾고 요약을 반환
# =============================================================================


def search_wikipedia(query: str, lang: str = "ko") -> dict:
    """위키피디아에서 검색어에 대한 요약 정보를 가져온다."""
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{query}"
    headers = {"User-Agent": "LLMLabResearchAssistant/1.0"}
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 200:
        data = response.json()
        return {
            "title": data.get("title", ""),
            "summary": data.get("extract", "정보를 찾을 수 없습니다."),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    else:
        # 제목이 정확하지 않으면 검색 API로 후보를 찾아본다
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "format": "json",
        }
        search_resp = requests.get(search_url, params=params, headers=headers, timeout=10)
        if search_resp.status_code == 200:
            results = search_resp.json().get("query", {}).get("search", [])
            if results:
                titles = [r["title"] for r in results]
                return {"error": "정확한 문서를 찾지 못했습니다.", "suggestions": titles}
        return {"error": f"검색 실패 (HTTP {response.status_code})"}


# =============================================================================
# 도구 2: 계산기 (수학 연산)
# - 사칙연산, 거듭제곱, 삼각함수 등 지원
# - eval 대신 안전한 방식으로 계산
# =============================================================================

# eval에 허용할 이름만 화이트리스트로 제한 (보안)
SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "pow": pow, "sqrt": math.sqrt, "log": math.log,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
}


def calculate(expression: str) -> dict:
    """수학 표현식을 계산한다."""
    try:
        # __builtins__를 빈 딕셔너리로 → 위험한 내장함수 차단
        result = eval(expression, {"__builtins__": {}}, SAFE_MATH)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


# =============================================================================
# 도구 3: 번역 (OpenAI API 활용)
# - LLM이 또 다른 LLM을 도구로 사용하는 구조
# - 실무에서도 흔한 패턴: 메인 LLM이 서브 LLM을 호출
# =============================================================================


def translate(text: str, target_language: str) -> dict:
    """텍스트를 지정된 언어로 번역한다."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Translate the following text to {target_language}. Return only the translation."},
            {"role": "user", "content": text},
        ],
    )
    translated = response.choices[0].message.content
    return {"original": text, "translated": translated, "target_language": target_language}


# =============================================================================
# 도구 스키마 정의
# =============================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "위키피디아에서 특정 주제에 대한 요약 정보를 검색한다. 인물, 사건, 개념, 장소 등의 정보를 조회할 때 사용.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 주제 (위키피디아 문서 제목, 예: 인공지능, 파이썬)",
                    },
                    "lang": {
                        "type": "string",
                        "description": "검색 언어 코드 (기본값: ko, 영어: en)",
                        "enum": ["ko", "en", "ja"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "수학 표현식을 계산한다. 사칙연산, 거듭제곱, 제곱근, 삼각함수, 로그 등 수학 계산이 필요할 때 사용.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "계산할 수학 표현식 (Python 문법, 예: 2**10, sqrt(144), log(100))",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "텍스트를 다른 언어로 번역한다. 외국어 자료를 이해하거나 번역이 필요할 때 사용.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "번역할 텍스트",
                    },
                    "target_language": {
                        "type": "string",
                        "description": "번역 대상 언어 (예: 한국어, English, 日本語)",
                    },
                },
                "required": ["text", "target_language"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_wikipedia": search_wikipedia,
    "calculate": calculate,
    "translate": translate,
}

# =============================================================================
# 대화형 루프
# =============================================================================


def chat(user_message: str, messages: list) -> list:
    """한 턴의 대화를 처리하고 업데이트된 messages를 반환한다."""
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
    )

    assistant_message = response.choices[0].message

    if assistant_message.tool_calls:
        print(f"\n  [도구 호출]")
        messages.append(assistant_message)

        for tc in assistant_message.tool_calls:
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)
            print(f"  -> {func_name}({func_args})")

            func = TOOL_FUNCTIONS.get(func_name)
            result = func(**func_args) if func else {"error": "알 수 없는 함수"}
            print(f"  <- 결과: {json.dumps(result, ensure_ascii=False)[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # 도구 결과를 바탕으로 최종 응답 생성
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        final_answer = final_response.choices[0].message.content
    else:
        final_answer = assistant_message.content

    messages.append({"role": "assistant", "content": final_answer})
    print(f"\n  AI: {final_answer}")
    return messages


def main():
    print("=" * 60)
    print("  리서치 어시스턴트 (종료: quit)")
    print("  도구: Wikipedia 검색 / 계산기 / 번역")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "너는 리서치 어시스턴트야. "
                "사용자의 질문에 도구를 활용해서 정확한 정보를 제공해. "
                "필요하면 여러 도구를 조합해서 사용해."
            ),
        }
    ]

    while True:
        user_input = input("\n사용자: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("종료합니다.")
            break
        if not user_input:
            continue
        messages = chat(user_input, messages)


if __name__ == "__main__":
    main()
