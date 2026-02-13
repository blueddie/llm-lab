"""
차원 축소 — 1536차원을 줄이면 검색 품질은 얼마나 떨어질까?

text-embedding-3 모델은 dimensions 파라미터를 지원한다.
1536차원 → 256차원으로 줄이면 저장 공간이 6배 줄어드는데,
검색 품질은 얼마나 희생되는지 직접 측정해본다.

실무에서 벡터 DB에 수십만 건을 저장할 때,
차원 수는 저장 비용과 검색 속도에 직접 영향을 준다.
"품질을 크게 잃지 않으면서 차원을 줄일 수 있는가?"가 핵심 질문이다.
"""

import asyncio
import os

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 테스트할 차원들
DIMENSIONS = [256, 512, 1024, 1536]


async def get_embeddings_with_dim(
    texts: list[str], dimensions: int | None = None
) -> list[list[float]]:
    """
    지정한 차원으로 임베딩 생성

    dimensions 파라미터를 넘기면 모델이 해당 차원으로 잘라서 반환한다.
    단순히 뒤를 자르는 게 아니라, Matryoshka Representation Learning이라는
    기법으로 학습되어 있어서 앞쪽 차원에 중요한 정보가 집중되어 있다.
    """
    params = {
        "model": "text-embedding-3-small",
        "input": texts,
    }
    if dimensions is not None:
        params["dimensions"] = dimensions

    response = await client.embeddings.create(**params)
    return [item.embedding for item in response.data]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def search_with_dim(
    query_emb: list[float],
    doc_embeddings: list[list[float]],
    documents: list[str],
    top_k: int = 3,
) -> list[dict]:
    """미리 계산된 임베딩으로 검색"""
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_emb, doc_emb)
        similarities.append({
            "document": documents[i],
            "score": round(score, 4),
        })
    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities[:top_k]


async def main():
    # 회사 내부 문서
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
        "휴가 며칠 쓸 수 있어?",
        "재택근무 가능한가요?",
        "야근하면 밥값 나와?",
    ]

    # --- 차원별 검색 결과 비교 ---
    print("=== 차원별 검색 결과 비교 ===\n")

    for dim in DIMENSIONS:
        print(f"--- {dim}차원 (저장 공간: {dim * 4}bytes/문서) ---\n")

        # 해당 차원으로 문서 + 질문 임베딩
        all_texts = documents + queries
        all_embeddings = await get_embeddings_with_dim(all_texts, dimensions=dim)

        doc_embeddings = all_embeddings[: len(documents)]
        query_embeddings = all_embeddings[len(documents) :]

        for query, query_emb in zip(queries, query_embeddings):
            results = await search_with_dim(query_emb, doc_embeddings, documents, top_k=3)
            print(f"  질문: {query}")
            for rank, r in enumerate(results, 1):
                print(f"    {rank}. [{r['score']}] {r['document'][:30]}...")
            print()

    # --- 차원별 유사도 점수 변화 요약 ---
    print("=== 차원별 1위 유사도 점수 비교 ===\n")
    print(f"  {'질문':<20s}", end="")
    for dim in DIMENSIONS:
        print(f"  {dim}차원", end="")
    print()
    print("  " + "─" * 60)

    # 각 차원별 1위 점수를 모아서 비교
    for qi, query in enumerate(queries):
        short_query = query[:18] + ".." if len(query) > 18 else query
        print(f"  {short_query:<20s}", end="")
        for dim in DIMENSIONS:
            all_texts = documents + [query]
            all_embeddings = await get_embeddings_with_dim(all_texts, dimensions=dim)
            doc_embs = all_embeddings[:-1]
            query_emb = all_embeddings[-1]
            results = await search_with_dim(query_emb, doc_embs, documents, top_k=1)
            print(f"  {results[0]['score']:.4f}", end="")
        print()

    print()
    print("차원을 줄여도 1위 문서가 바뀌지 않으면 → 차원 축소 OK")
    print("1위 문서가 바뀌거나 점수 차이가 크면 → 해당 차원은 위험")


if __name__ == "__main__":
    asyncio.run(main())


'''
=== 차원별 검색 결과 비교 ===

--- 256차원 (저장 공간: 1024bytes/문서) ---

  질문: 휴가 며칠 쓸 수 있어?
    1. [0.3932] 경조사 휴가는 결혼 5일, 출산 10일, 사망 3일이 ...
    2. [0.2951] 연차는 입사일 기준으로 매년 15일이 부여됩니다....
    3. [0.2525] 점심시간은 12시부터 1시까지이며, 자율적으로 조정 가...

  질문: 재택근무 가능한가요?
    1. [0.4393] 원격 근무는 주 2회까지 가능하며, 팀장 승인이 필요합...
    2. [0.3546] 퇴직금은 근속 1년 이상 시 지급되며, 평균 임금 기준...
    3. [0.2814] 야근 식대는 1만원까지 법인카드로 결제 가능합니다....

  질문: 야근하면 밥값 나와?
    1. [0.4152] 야근 식대는 1만원까지 법인카드로 결제 가능합니다....
    2. [0.1876] 원격 근무는 주 2회까지 가능하며, 팀장 승인이 필요합...
    3. [0.144] 퇴직금은 근속 1년 이상 시 지급되며, 평균 임금 기준...

--- 512차원 (저장 공간: 2048bytes/문서) ---

  질문: 휴가 며칠 쓸 수 있어?
    1. [0.4529] 경조사 휴가는 결혼 5일, 출산 10일, 사망 3일이 ...
    2. [0.2993] 점심시간은 12시부터 1시까지이며, 자율적으로 조정 가...
    3. [0.2963] 연차는 입사일 기준으로 매년 15일이 부여됩니다....

  질문: 재택근무 가능한가요?
    1. [0.4757] 원격 근무는 주 2회까지 가능하며, 팀장 승인이 필요합...
    2. [0.3804] 퇴직금은 근속 1년 이상 시 지급되며, 평균 임금 기준...
    3. [0.35] 야근 식대는 1만원까지 법인카드로 결제 가능합니다....

  질문: 야근하면 밥값 나와?
    1. [0.4194] 야근 식대는 1만원까지 법인카드로 결제 가능합니다....
    2. [0.2142] 점심시간은 12시부터 1시까지이며, 자율적으로 조정 가...
    3. [0.204] 성과 평가는 반기별로 진행되며, OKR 기반으로 측정합...

--- 1024차원 (저장 공간: 4096bytes/문서) ---

  질문: 휴가 며칠 쓸 수 있어?
    1. [0.4472] 경조사 휴가는 결혼 5일, 출산 10일, 사망 3일이 ...
    2. [0.2985] 연차는 입사일 기준으로 매년 15일이 부여됩니다....
    3. [0.2582] 점심시간은 12시부터 1시까지이며, 자율적으로 조정 가...

  질문: 재택근무 가능한가요?
    1. [0.4436] 원격 근무는 주 2회까지 가능하며, 팀장 승인이 필요합...
    2. [0.3516] 퇴직금은 근속 1년 이상 시 지급되며, 평균 임금 기준...
    3. [0.3241] 야근 식대는 1만원까지 법인카드로 결제 가능합니다....

  질문: 야근하면 밥값 나와?
    1. [0.3851] 야근 식대는 1만원까지 법인카드로 결제 가능합니다....
    2. [0.1765] 퇴직금은 근속 1년 이상 시 지급되며, 평균 임금 기준...
    3. [0.168] 점심시간은 12시부터 1시까지이며, 자율적으로 조정 가...

--- 1536차원 (저장 공간: 6144bytes/문서) ---

  질문: 휴가 며칠 쓸 수 있어?
    1. [0.4536] 경조사 휴가는 결혼 5일, 출산 10일, 사망 3일이 ...
    2. [0.2875] 연차는 입사일 기준으로 매년 15일이 부여됩니다....
    3. [0.2482] 점심시간은 12시부터 1시까지이며, 자율적으로 조정 가...

  질문: 재택근무 가능한가요?
    1. [0.4208] 원격 근무는 주 2회까지 가능하며, 팀장 승인이 필요합...
    2. [0.3369] 퇴직금은 근속 1년 이상 시 지급되며, 평균 임금 기준...
    3. [0.3144] 야근 식대는 1만원까지 법인카드로 결제 가능합니다....

  질문: 야근하면 밥값 나와?
    1. [0.378] 야근 식대는 1만원까지 법인카드로 결제 가능합니다....
    2. [0.1811] 퇴직금은 근속 1년 이상 시 지급되며, 평균 임금 기준...
    3. [0.1666] 경조사 휴가는 결혼 5일, 출산 10일, 사망 3일이 ...

=== 차원별 1위 유사도 점수 비교 ===

  질문                    256차원  512차원  1024차원  1536차원
  ────────────────────────────────────────────────────────────
  휴가 며칠 쓸 수 있어?         0.3932  0.4526  0.4472  0.4536
  재택근무 가능한가요?           0.4393  0.4757  0.4437  0.4209
  야근하면 밥값 나와?           0.4153  0.4193  0.3850  0.3780

차원을 줄여도 1위 문서가 바뀌지 않으면 → 차원 축소 OK
1위 문서가 바뀌거나 점수 차이가 크면 → 해당 차원은 위험

'''