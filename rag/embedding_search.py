"""
임베딩 & 유사도 검색 — RAG의 "R"이 작동하는 원리

텍스트를 벡터(숫자 배열)로 변환하고,
코사인 유사도로 "의미적으로 가장 가까운 텍스트"를 찾는다.

키워드 검색: 단어가 일치해야 찾음 ("휴가 정책" → "연차 사용 규정" 못 찾음)
임베딩 검색: 의미가 비슷하면 찾음 ("휴가 정책" → "연차 사용 규정" 찾음)
"""

import asyncio
import os

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- 1. 임베딩 생성 ---

async def get_embedding(text: str) -> list[float]:
    """
    텍스트 → 벡터(1536차원 숫자 배열)로 변환

    LLM의 chat.completions.create()가 텍스트를 반환한다면,
    embeddings.create()는 숫자 배열을 반환한다.
    """
    response = await client.embeddings.create(
        model="text-embedding-3-small",  # OpenAI 임베딩 모델
        input=text,
    )
    return response.data[0].embedding


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    여러 텍스트를 한 번에 임베딩 — API 호출 1번으로 처리

    하나씩 보내면 N번 호출, 배치로 보내면 1번 호출.
    실무에서는 항상 배치를 쓴다.
    """
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,  # 리스트를 그대로 전달
    )
    return [item.embedding for item in response.data]


# --- 2. 코사인 유사도 ---

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    두 벡터의 방향이 얼마나 같은지 측정

    1에 가까울수록 의미가 비슷, 0이면 관계없음, -1이면 반대
    공식: cos(θ) = (A·B) / (|A| × |B|)
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- 3. 유사도 검색 ---

async def search(query: str, documents: list[str], top_k: int = 3) -> list[dict]:
    """
    query와 의미적으로 가장 가까운 문서 top_k개를 찾는다.

    이게 RAG의 "R" (Retrieve)의 핵심이다:
    1. 문서들을 벡터로 변환 (보통 미리 해둠)
    2. 질문도 벡터로 변환
    3. 코사인 유사도로 가장 가까운 것을 찾음
    """
    # 문서 임베딩 (실제로는 미리 계산해서 DB에 저장해둠)
    doc_embeddings = await get_embeddings_batch(documents)

    # 질문 임베딩
    query_embedding = await get_embedding(query)

    # 각 문서와의 유사도 계산
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_emb)
        similarities.append({
            "document": documents[i],
            "score": round(score, 4),
        })

    # 유사도 높은 순으로 정렬
    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities[:top_k]


async def main():
    # --- 예제 1: 임베딩이 뭔지 직접 확인 ---
    print("=== 임베딩 확인 ===\n")

    text = "고양이는 귀엽다"
    embedding = await get_embedding(text)

    print(f"텍스트: {text}")
    print(f"벡터 차원: {len(embedding)}")
    print(f"앞 5개 값: {embedding[:5]}")
    print()

    # --- 예제 2: 의미적 유사도 비교 ---
    print("=== 유사도 비교 ===\n")

    pairs = [
        ("고양이는 귀엽다", "강아지는 사랑스럽다"),     # 의미 비슷
        ("고양이는 귀엽다", "Python은 프로그래밍 언어다"), # 의미 다름
        ("오늘 날씨가 좋다", "햇살이 따뜻하다"),         # 의미 비슷
        ("오늘 날씨가 좋다", "주식이 폭락했다"),         # 의미 다름
    ]

    # 모든 텍스트를 한 번에 임베딩
    all_texts = list(set(t for pair in pairs for t in pair))
    all_embeddings = await get_embeddings_batch(all_texts)
    emb_map = dict(zip(all_texts, all_embeddings))

    for text_a, text_b in pairs:
        score = cosine_similarity(emb_map[text_a], emb_map[text_b])
        print(f"  {score:.4f}  |  \"{text_a}\" ↔ \"{text_b}\"")
    print()

    # --- 예제 3: 유사도 검색 (RAG의 R) ---
    print("=== 유사도 검색 ===\n")

    # 회사 내부 문서라고 가정
    documents = [
        "연차는 입사일 기준으로 매년 15일이 부여됩니다.",
        "점심시간은 12시부터 1시까지이며, 자율적으로 조정 가능합니다.",
        "원격 근무는 주 2회까지 가능하며, 팀장 승인이 필요합니다.",
        "경조사 휴가는 결혼 5일, 출산 10일, 사망 3일이 지급됩니다.",
        "야근 식대는 1만원까지 법인카드로 결제 가능합니다.",
        "성과 평가는 반기별로 진행되며, OKR 기반으로 측정합니다.",
        "신규 입사자는 첫 주에 온보딩 프로그램에 참여해야 합니다.",
        "퇴직금은 근속 1년 이상 시 지급되며, 평균 임금 기준으로 산정됩니다.",
    ]

    queries = [
        "휴가 며칠 쓸 수 있어?",      # "연차" 키워드 없이도 찾을 수 있을까?
        "재택근무 가능한가요?",         # "원격 근무"와 매칭되는지?
        "야근하면 밥값 나와?",          # 구어체도 찾을 수 있을까?
    ]

    for query in queries:
        print(f"질문: {query}")
        results = await search(query, documents, top_k=3)
        for rank, r in enumerate(results, 1):
            print(f"  {rank}. [{r['score']}] {r['document']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
