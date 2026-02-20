"""
top_k 값에 따른 검색 정확도 변화 실험

질문: top_k를 늘리면 검색 정확도가 얼마나 올라갈까?
그리고 그 대가(컨텍스트 길이, 비용)는 얼마일까?
"""
import asyncio
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from basic_rag import (
    load_documents,
    chunk_all_documents,
    cosine_similarity,
    embed_texts,
    DATA_PATH,
    INDEX_PATH,
    NUM_DOCS,
)

TOP_K_VALUES = [1, 3, 5, 10, 15, 20]


async def main():
    # 데이터 & 인덱스 로드
    docs = load_documents(DATA_PATH, NUM_DOCS)
    chunks, metadata = chunk_all_documents(docs)

    data = np.load(INDEX_PATH, allow_pickle=True)
    embeddings = data["embeddings"]

    # 평가용 QA 수집
    eval_qas = []
    for doc_idx, doc in enumerate(docs):
        for qa in doc["qas"]:
            eval_qas.append({
                "question": qa["question"],
                "answer": qa["answer_text"],
                "doc_idx": doc_idx,
            })
    eval_qas = eval_qas[:50]

    # 질문 임베딩 (한번만 하면 됨)
    print("질문 임베딩 중...")
    questions = [qa["question"] for qa in eval_qas]
    query_embeddings = await embed_texts(questions)

    # top_k별 검색 정확도 측정
    print(f"\n{'=' * 60}")
    print(f"{'top_k':>6} | {'검색 정확도':>10} | {'평균 컨텍스트 길이':>16} | {'예상 토큰 비용':>14}")
    print(f"{'-' * 60}")

    for top_k in TOP_K_VALUES:
        hits = 0
        total_context_len = 0

        for i, qa in enumerate(eval_qas):
            answer_clean = " ".join(qa["answer"].split())

            # 코사인 유사도 계산
            scores = cosine_similarity(query_embeddings[i], embeddings)
            top_indices = np.argsort(scores)[::-1][:top_k]

            # 검색된 청크에서 정답 찾기
            hit = False
            context_len = 0
            for idx in top_indices:
                chunk_clean = " ".join(chunks[idx].split())
                context_len += len(chunks[idx])
                if answer_clean in chunk_clean:
                    hit = True

            hits += hit
            total_context_len += context_len

        avg_context = total_context_len // len(eval_qas)
        hit_rate = hits / len(eval_qas) * 100
        # 한국어 대략 1자 ≈ 1.5 토큰으로 추정
        est_tokens = int(avg_context * 1.5)

        print(f"{top_k:>6} | {hits:>4}/{len(eval_qas)} ({hit_rate:>4.1f}%) | {avg_context:>12,}자 | ~{est_tokens:>10,} 토큰")

    print(f"\n참고: gpt-4o-mini 입력 비용 = $0.15 / 1M 토큰")
    print(f"50개 질문 기준 top_k=3: ~{int(750*1.5*50/1000)}K 토큰 ≈ $0.01 이하")
    print(f"50개 질문 기준 top_k=20: ~{int(5000*1.5*50/1000)}K 토큰 ≈ $0.06 이하")


if __name__ == "__main__":
    asyncio.run(main())
