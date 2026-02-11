"""
httpx로 OpenAI API 직접 호출하기

SDK 없이 HTTP 요청을 직접 보내면:
- API가 실제로 어떤 형태의 데이터를 주고받는지 눈으로 확인
- async with (async context manager) 패턴 학습
- 어떤 API든 호출할 수 있는 범용 스킬 습득
"""

import asyncio
import os

import httpx
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드 (프로젝트 루트의 .env를 찾아 올라감)
load_dotenv()

# OpenAI API의 Chat Completions 엔드포인트
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


async def call_llm(prompt: str) -> str:
    """
    httpx로 OpenAI API를 직접 호출하는 함수

    SDK가 내부에서 하는 일을 직접 해보는 것:
    1. HTTP 헤더 설정 (인증, Content-Type)
    2. JSON body 구성 (모델, 메시지, 파라미터)
    3. POST 요청 전송
    4. 응답 JSON 파싱
    """

    # HTTP 헤더 — Bearer 토큰 방식 인증
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    # 요청 body — OpenAI API 스펙에 맞춰 구성
    body = {
        "model": "gpt-4.1-nano",           
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,                   # 응답 길이 제한 (비용 절약)
    }

    # async with — 연결 풀 생성 → 요청 → 연결 정리까지 자동 관리
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENAI_URL,
            headers=headers,
            json=body,         # dict → JSON 자동 변환
            timeout=30.0,      # 30초 타임아웃 (LLM은 응답이 느릴 수 있음)
        )

    # 상태 코드 확인 — 200이 아니면 에러 내용 출력
    if response.status_code != 200:
        print(f"[에러] status={response.status_code}")
        print(f"[에러] body={response.text}")
        return ""

    # 응답 JSON 파싱
    data = response.json()

    # 실제 응답 구조를 눈으로 확인 (처음이니까 전체 구조를 봐야 함)
    print("=== 원본 응답 구조 ===")
    print(f"모델: {data['model']}")
    print(f"토큰 사용량: {data['usage']}")
    print(f"선택지 수: {len(data['choices'])}")
    print()

    # LLM 응답 텍스트 추출
    answer = data["choices"][0]["message"]["content"]
    return answer


async def main():
    prompt = "Python의 async/await를 한 문장으로 설명해줘."

    print(f"프롬프트: {prompt}")
    print("호출 중...\n")

    answer = await call_llm(prompt)
    print(f"=== LLM 응답 ===\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
