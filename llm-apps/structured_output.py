"""
Structured Output — LLM 응답을 Pydantic 모델로 강제하기

자유 텍스트 → 파싱 불안정, 형태가 달라지면 코드가 깨짐
구조화 응답 → JSON으로 고정, 코드에서 바로 사용 가능

FastAPI에서 배운 Pydantic이 여기서도 쓰임:
- FastAPI: "클라이언트야, 이 형태로 요청해"
- Structured Output: "LLM아, 이 형태로 응답해"
"""

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- 예제 1: 감정 분석 ---

class SentimentResult(BaseModel):
    """LLM 응답이 반드시 이 형태여야 함"""
    sentiment: str       # "positive", "negative", "neutral"
    score: int           # 1~5
    keywords: list[str]  # 핵심 키워드
    summary: str         # 한 줄 요약


async def analyze_sentiment(review: str) -> SentimentResult:
    """리뷰 텍스트를 분석해서 구조화된 결과 반환"""

    # beta.chat.completions.parse — Pydantic 모델을 직접 전달
    response = await client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": "리뷰를 분석해서 감정, 점수(1~5), 키워드, 요약을 추출해.",
            },
            {"role": "user", "content": review},
        ],
        response_format=SentimentResult,  # ← Pydantic 모델을 그대로 전달
    )

    # .parsed — JSON이 아니라 Pydantic 객체로 바로 받음
    return response.choices[0].message.parsed


# --- 예제 2: 정보 추출 ---

class ExtractedInfo(BaseModel):
    """문장에서 구조화된 정보 추출"""
    person_name: str
    action: str
    location: str
    time: str


async def extract_info(text: str) -> ExtractedInfo:
    """비정형 텍스트에서 구조화된 정보 추출"""

    response = await client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": "텍스트에서 인물, 행동, 장소, 시간을 추출해. 없으면 'unknown'으로.",
            },
            {"role": "user", "content": text},
        ],
        response_format=ExtractedInfo,
    )

    return response.choices[0].message.parsed


async def main():
    # 예제 1: 감정 분석
    print("=== 감정 분석 ===\n")

    reviews = [
        "배송도 빠르고 품질도 좋아요! 다음에도 구매할 의향 있습니다.",
        "제품이 설명과 전혀 다릅니다. 환불 요청했는데 답도 없네요.",
        "보통이에요. 가격 대비 나쁘진 않은데 특별히 좋지도 않아요.",
    ]

    # gather로 동시 분석
    results = await asyncio.gather(*[analyze_sentiment(r) for r in reviews])

    for review, result in zip(reviews, results):
        print(f"리뷰: {review[:30]}...")
        print(f"  감정: {result.sentiment} (점수: {result.score}/5)")
        print(f"  키워드: {result.keywords}")
        print(f"  요약: {result.summary}")
        print()

    # 예제 2: 정보 추출
    print("=== 정보 추출 ===\n")

    texts = [
        "철수가 어제 강남역 카페에서 코딩 공부를 했다.",
        "내일 오후에 영희가 회사에서 발표할 예정이다.",
    ]

    results = await asyncio.gather(*[extract_info(t) for t in texts])

    for text, result in zip(texts, results):
        print(f"원문: {text}")
        print(f"  인물: {result.person_name}")
        print(f"  행동: {result.action}")
        print(f"  장소: {result.location}")
        print(f"  시간: {result.time}")
        print()


if __name__ == "__main__":
    asyncio.run(main())

'''
=== 감정 분석 ===

리뷰: 배송도 빠르고 품질도 좋아요! 다음에도 구매할 의향 있...
  감정: 긍정 (점수: 5/5)
  키워드: ['배송', '빠름', '품질', '좋음', '구매', '의향']
  요약: 배송이 빠르고 품질이 좋아서 만족스럽고, 앞으로도 재구매를 고려하고 있는 긍정적인 리뷰입니다.

리뷰: 제품이 설명과 전혀 다릅니다. 환불 요청했는데 답도 없...
  감정: 부정적 (점수: 2/5)
  키워드: ['제품', '설명과 다른', '환불', '답 없음']
  요약: 제품이 설명과 달라서 환불 요청했으나 아무 답변도 받지 못했습니다.

리뷰: 보통이에요. 가격 대비 나쁘진 않은데 특별히 좋지도 않...
  감정: 중립 (점수: 3/5)
  키워드: ['가격', '별로임']
  요약: 가격 대비 보통 수준이며 특별히 좋거나 나쁘지 않음.

=== 정보 추출 ===

원문: 철수가 어제 강남역 카페에서 코딩 공부를 했다.
  인물: 철수
  행동: 코딩 공부
  장소: 강남역 카페
  시간: 어제

원문: 내일 오후에 영희가 회사에서 발표할 예정이다.
  인물: 영희
  행동: 발표할 예정
  장소: 회사
  시간: 내일 오후
'''