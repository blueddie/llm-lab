"""
Hybrid Search RAG - Dense (임베딩) + Sparse (BM25) 결합

기본 RAG에서 검색 정확도가 86%에서 멈췄다 (top_k=10 이상 올려도 동일).
원인: 임베딩 유사도만으로는 키워드 매칭이 약하다.

해결: BM25 키워드 검색을 추가하고, 두 결과를 RRF로 합친다.

새로 추가된 것:
- BM25 (직접 구현) — 키워드 기반 검색
- 한국어 형태소 분석 (kiwipiepy) — "임베딩을" → "임베딩"으로 정규화
- RRF (Reciprocal Rank Fusion) — 두 검색 결과 합치기
"""
import asyncio
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from openai import AsyncOpenAI

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from basic_rag import (
    load_documents,
    chunk_all_documents,
    cosine_similarity,
    embed_texts,
    build_index,
    evaluate_retrieval,
    generate,
    evaluate_answer,
    DATA_PATH,
    INDEX_PATH,
    NUM_DOCS,
    NUM_EVAL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)


# ============================================================
# 한국어 형태소 분석기
# ============================================================
# BM25는 단어 단위로 동작한다.
# 한국어는 "임베딩을", "임베딩은", "임베딩의"가 다 다른 단어로 취급되므로
# 형태소 분석으로 어근을 추출해야 한다.
#
# kiwipiepy는 한국어 형태소 분석기 중 설치가 가장 쉽고 성능도 좋다.

kiwi = Kiwi()


def tokenize(text: str) -> list[str]:
    """
    한국어 텍스트를 형태소 단위로 토큰화한다.

    "이명박은 서울시장 취임 후 서울시청 앞에 광장을 조성하고자 했다"
    → ["이명박", "서울", "시장", "취임", "후", "서울시청", "앞", "광장", "조성"]

    조사, 어미 등은 제거하고 의미 있는 품사(명사, 동사, 형용사 등)만 남긴다.
    """
    # 의미 있는 품사 태그 (NNG: 일반명사, NNP: 고유명사, VV: 동사, VA: 형용사, SL: 외국어, SN: 숫자)
    meaningful_tags = {"NNG", "NNP", "VV", "VA", "SL", "SN", "NR"}

    tokens = []
    for token in kiwi.tokenize(text):
        if token.tag in meaningful_tags and len(token.form) > 1:
            tokens.append(token.form)

    return tokens


# ============================================================
# BM25 (직접 구현)
# ============================================================
# BM25는 TF-IDF의 개선 버전이다.
#
# 핵심 아이디어:
# - TF (Term Frequency): 이 단어가 이 청크에 자주 나오면 관련성 높음
# - IDF (Inverse Document Frequency): 모든 청크에 나오는 단어는 중요하지 않음
# - 문서 길이 정규화: 긴 청크가 불공정하게 유리하지 않도록
#
# 공식:
# score(q, d) = Σ IDF(t) × (TF(t,d) × (k1+1)) / (TF(t,d) + k1 × (1-b+b×|d|/avgdl))
#
# k1: TF 포화도 조절 (보통 1.2~2.0)
# b: 문서 길이 정규화 강도 (보통 0.75)

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0
        self.doc_lengths = []        # 각 청크의 토큰 수
        self.doc_tokens = []         # 각 청크의 토큰 리스트
        self.doc_freqs = Counter()   # 각 단어가 몇 개의 청크에 등장하는지 (DF)

    def fit(self, chunks: list[str]):
        """청크들을 인덱싱한다. (오프라인 단계)"""
        print("BM25 인덱싱 중 (형태소 분석)...")
        start = time.time()

        self.doc_count = len(chunks)

        for chunk in chunks:
            tokens = tokenize(chunk)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))

            # DF 계산: 각 고유 단어가 이 청크에 등장했는지
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        self.avg_doc_len = sum(self.doc_lengths) / self.doc_count
        elapsed = time.time() - start
        print(f"BM25 인덱싱 완료 ({elapsed:.1f}초, 고유 단어 {len(self.doc_freqs):,}개)")

    def _idf(self, term: str) -> float:
        """IDF(Inverse Document Frequency) 계산"""
        df = self.doc_freqs.get(term, 0)
        # 표준 BM25 IDF 공식
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str) -> np.ndarray:
        """
        쿼리와 모든 청크 사이의 BM25 점수를 계산한다.

        반환: (청크 수,) shape의 점수 배열
        """
        query_tokens = tokenize(query)
        scores = np.zeros(self.doc_count, dtype=np.float32)

        for token in query_tokens:
            idf = self._idf(token)

            for i in range(self.doc_count):
                # TF: 이 청크에서 이 단어가 몇 번 나오는지
                tf = self.doc_tokens[i].count(token)
                if tf == 0:
                    continue

                # BM25 점수 공식
                doc_len = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                scores[i] += idf * numerator / denominator

        return scores


# ============================================================
# RRF (Reciprocal Rank Fusion)
# ============================================================
# 두 검색 결과의 순위를 합쳐서 최종 순위를 만든다.
#
# 왜 점수를 직접 합치지 않고 순위를 쓸까?
# - Dense 점수 범위: 0.0 ~ 1.0 (코사인 유사도)
# - BM25 점수 범위: 0.0 ~ 수십 (스케일이 완전히 다름)
# - 점수를 직접 합치면 스케일이 큰 쪽에 치우친다
# - 순위 기반이면 스케일 차이가 무관해진다
#
# RRF(d) = 1/(k + rank_dense(d)) + 1/(k + rank_bm25(d))
# k는 보통 60을 쓴다 (원논문 기본값)

def rrf_fusion(
    dense_scores: np.ndarray,
    bm25_scores: np.ndarray,
    top_k: int,
    rrf_k: int = 60,
) -> list[int]:
    """
    Dense와 BM25 점수를 RRF로 합쳐서 최종 top_k 인덱스를 반환한다.
    """
    n = len(dense_scores)

    # 각 검색의 순위 계산 (0-based)
    dense_ranks = np.argsort(np.argsort(-dense_scores))  # 점수 높은 순 → 순위
    bm25_ranks = np.argsort(np.argsort(-bm25_scores))

    # RRF 점수 계산
    rrf_scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        rrf_scores[i] = 1 / (rrf_k + dense_ranks[i]) + 1 / (rrf_k + bm25_ranks[i])

    # 최종 top_k
    top_indices = np.argsort(rrf_scores)[::-1][:top_k]
    return top_indices.tolist()


# ============================================================
# 메인: Dense vs BM25 vs Hybrid 비교 실험
# ============================================================
async def main():
    print("=" * 60)
    print("Hybrid Search 실험: Dense vs BM25 vs Hybrid")
    print("=" * 60)

    # 데이터 로드
    docs = load_documents(DATA_PATH, NUM_DOCS)
    chunks, metadata = chunk_all_documents(docs)

    # Dense 인덱스 로드
    embeddings = await build_index(chunks, metadata)

    # BM25 인덱스 생성
    bm25 = BM25()
    bm25.fit(chunks)

    # 평가용 QA 수집
    eval_qas = []
    for doc_idx, doc in enumerate(docs):
        for qa in doc["qas"]:
            eval_qas.append({
                "question": qa["question"],
                "answer": qa["answer_text"],
                "doc_idx": doc_idx,
            })
    eval_qas = eval_qas[:NUM_EVAL]

    # 질문 임베딩 (한번에)
    print("\n질문 임베딩 중...")
    questions = [qa["question"] for qa in eval_qas]
    query_embeddings = await embed_texts(questions)

    # top_k 값들에 대해 세 방법 비교
    top_k_values = [3, 5, 10]

    print(f"\n{'=' * 60}")
    print(f"{'방법':<20} | {'top_k':>5} | {'검색 정확도':>10}")
    print(f"{'-' * 60}")

    for top_k in top_k_values:
        dense_hits = 0
        bm25_hits = 0
        hybrid_hits = 0

        for i, qa in enumerate(eval_qas):
            answer_clean = " ".join(qa["answer"].split())

            # Dense 검색
            dense_scores = cosine_similarity(query_embeddings[i], embeddings)
            dense_top = np.argsort(dense_scores)[::-1][:top_k]

            # BM25 검색
            bm25_scores = bm25.score(qa["question"])
            bm25_top = np.argsort(bm25_scores)[::-1][:top_k]

            # Hybrid (RRF)
            hybrid_top = rrf_fusion(dense_scores, bm25_scores, top_k)

            # 각 방법의 히트 여부
            for idx in dense_top:
                if answer_clean in " ".join(chunks[idx].split()):
                    dense_hits += 1
                    break
            for idx in bm25_top:
                if answer_clean in " ".join(chunks[idx].split()):
                    bm25_hits += 1
                    break
            for idx in hybrid_top:
                if answer_clean in " ".join(chunks[idx].split()):
                    hybrid_hits += 1
                    break

        n = len(eval_qas)
        print(f"{'Dense (임베딩)':<20} | {top_k:>5} | {dense_hits:>4}/{n} ({dense_hits/n*100:.1f}%)")
        print(f"{'BM25 (키워드)':<20} | {top_k:>5} | {bm25_hits:>4}/{n} ({bm25_hits/n*100:.1f}%)")
        print(f"{'Hybrid (RRF)':<20} | {top_k:>5} | {hybrid_hits:>4}/{n} ({hybrid_hits/n*100:.1f}%)")
        print(f"{'-' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
