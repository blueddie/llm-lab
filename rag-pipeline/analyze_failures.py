"""
검색 실패 케이스를 분석하는 스크립트

"왜 검색이 실패했는가?"를 직접 눈으로 확인한다.
실패 원인을 파악해야 개선 방향이 보인다.
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
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    NUM_DOCS,
)


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
                "title": doc["title"],
                "doc_idx": doc_idx,
            })
    eval_qas = eval_qas[:50]

    # 실패 케이스 분석
    print("=" * 60)
    print("검색 실패 케이스 분석")
    print("=" * 60)

    fail_count = 0
    fail_reasons = {"split": 0, "no_match": 0, "low_rank": 0}

    for i, qa in enumerate(eval_qas):
        answer = qa["answer"]
        answer_clean = " ".join(answer.split())

        # 검색 실행
        query_emb = await embed_texts([qa["question"]])
        scores = cosine_similarity(query_emb[0], embeddings)
        top_indices = np.argsort(scores)[::-1][:TOP_K]

        # top-k에 정답이 있는지 확인
        hit = False
        for idx in top_indices:
            chunk_clean = " ".join(chunks[idx].split())
            if answer_clean in chunk_clean:
                hit = True
                break

        if hit:
            continue

        # === 실패 케이스 ===
        fail_count += 1
        print(f"\n{'─' * 60}")
        print(f"[실패 #{fail_count}] {qa['title']}")
        print(f"  질문: {qa['question']}")
        print(f"  정답: {answer[:100]}")

        # 원인 분석 1: 정답이 아예 어떤 청크에도 없는가? (청킹에서 잘림)
        answer_in_any_chunk = False
        answer_chunk_idx = -1
        for j, chunk in enumerate(chunks):
            chunk_clean = " ".join(chunk.split())
            if answer_clean in chunk_clean:
                answer_in_any_chunk = True
                answer_chunk_idx = j
                break

        if not answer_in_any_chunk:
            # 정답이 청크 경계에서 잘렸는지 확인
            doc_text = docs[qa["doc_idx"]]["context_text"]
            doc_text_clean = " ".join(doc_text.split())
            if answer_clean in doc_text_clean:
                print(f"  원인: 정답이 청크 경계에서 잘림!")

                # 어디서 잘렸는지 보여주기
                pos = doc_text.find(answer[:20])
                if pos >= 0:
                    chunk_num = pos // (CHUNK_SIZE - CHUNK_OVERLAP)
                    chunk_start = chunk_num * (CHUNK_SIZE - CHUNK_OVERLAP)
                    chunk_end = chunk_start + CHUNK_SIZE
                    print(f"  정답 위치: {pos}번째 글자")
                    print(f"  해당 청크 범위: [{chunk_start}:{chunk_end}]")
                    if pos + len(answer) > chunk_end:
                        print(f"  → 정답이 청크 끝({chunk_end})을 넘어감! (정답 끝: {pos + len(answer)})")

                fail_reasons["split"] += 1
            else:
                print(f"  원인: 정답이 원본 문서에도 없음 (HTML 전처리 과정에서 손실?)")
                fail_reasons["no_match"] += 1
        else:
            # 정답 청크는 있는데 순위가 낮음
            answer_score = float(scores[answer_chunk_idx])
            top_scores = [float(scores[idx]) for idx in top_indices]
            rank = int(np.where(np.argsort(scores)[::-1] == answer_chunk_idx)[0][0]) + 1

            print(f"  원인: 정답 청크가 순위 밖 (rank={rank}, top-{TOP_K}만 사용)")
            print(f"  정답 청크 유사도: {answer_score:.4f}")
            print(f"  top-{TOP_K} 유사도: {top_scores}")
            fail_reasons["low_rank"] += 1

            # 정답 청크와 검색된 청크 비교
            print(f"\n  [정답이 있는 청크 (rank {rank})]")
            print(f"  {chunks[answer_chunk_idx][:200]}...")
            print(f"\n  [검색된 1위 청크]")
            print(f"  {chunks[top_indices[0]][:200]}...")

    # 요약
    print(f"\n{'=' * 60}")
    print(f"실패 원인 요약 (총 {fail_count}건)")
    print(f"{'=' * 60}")
    print(f"  청크 경계에서 잘림: {fail_reasons['split']}건")
    print(f"  순위 밖 (검색은 됐지만 top-k 밖): {fail_reasons['low_rank']}건")
    print(f"  원본에서 매칭 안 됨: {fail_reasons['no_match']}건")

    total = fail_count or 1
    print(f"\n비율:")
    print(f"  청킹 문제: {fail_reasons['split']/total*100:.0f}%")
    print(f"  검색 순위 문제: {fail_reasons['low_rank']/total*100:.0f}%")
    print(f"  데이터 문제: {fail_reasons['no_match']/total*100:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
