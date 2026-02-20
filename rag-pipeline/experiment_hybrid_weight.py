"""
Hybrid Search 가중치 최적화 실험 + 시각화

질문: Dense와 BM25의 비율을 어떻게 설정해야 최고 성능이 나올까?

방법: Weighted RRF
  RRF(d) = α/(k + rank_dense) + (1-α)/(k + rank_bm25)
  α = 0.0 → BM25만 사용
  α = 0.5 → 기존 동일 가중치
  α = 1.0 → Dense만 사용

시각화:
  1. α 값에 따른 검색 정확도 곡선 → 최적 가중치 찾기
  2. Dense vs BM25 vs Hybrid 비교 막대 그래프
  3. 질문별 Dense/BM25 성공 여부 히트맵 → 두 방법의 상호보완 관계 확인
"""
import asyncio
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
    build_index,
    DATA_PATH,
    INDEX_PATH,
    NUM_DOCS,
    NUM_EVAL,
)
from hybrid_rag import BM25, tokenize

# ============================================================
# 한글 폰트 설정
# ============================================================
def setup_korean_font():
    """matplotlib에서 한글이 깨지지 않도록 폰트를 설정한다."""
    # Windows의 맑은 고딕
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

setup_korean_font()


# ============================================================
# Weighted RRF
# ============================================================
def weighted_rrf(
    dense_scores: np.ndarray,
    bm25_scores: np.ndarray,
    top_k: int,
    alpha: float = 0.5,  # Dense 가중치 (0~1)
    rrf_k: int = 60,
) -> list[int]:
    """
    가중치를 적용한 RRF.
    alpha: Dense의 비중 (0이면 BM25만, 1이면 Dense만, 0.5면 동일)
    """
    n = len(dense_scores)
    dense_ranks = np.argsort(np.argsort(-dense_scores))
    bm25_ranks = np.argsort(np.argsort(-bm25_scores))

    rrf_scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        rrf_scores[i] = (
            alpha / (rrf_k + dense_ranks[i])
            + (1 - alpha) / (rrf_k + bm25_ranks[i])
        )

    return np.argsort(rrf_scores)[::-1][:top_k].tolist()


# ============================================================
# 실험 실행
# ============================================================
async def run_experiment():
    # 데이터 로드
    docs = load_documents(DATA_PATH, NUM_DOCS)
    chunks, metadata = chunk_all_documents(docs)
    embeddings = await build_index(chunks, metadata)

    bm25 = BM25()
    bm25.fit(chunks)

    # 평가 QA
    eval_qas = []
    for doc_idx, doc in enumerate(docs):
        for qa in doc["qas"]:
            eval_qas.append({
                "question": qa["question"],
                "answer": qa["answer_text"],
                "doc_idx": doc_idx,
            })
    eval_qas = eval_qas[:NUM_EVAL]
    n = len(eval_qas)

    # 질문 임베딩
    print("\n질문 임베딩 중...")
    questions = [qa["question"] for qa in eval_qas]
    query_embeddings = await embed_texts(questions)

    # 모든 질문에 대해 Dense/BM25 점수 미리 계산
    all_dense_scores = []
    all_bm25_scores = []
    for i in range(n):
        all_dense_scores.append(cosine_similarity(query_embeddings[i], embeddings))
        all_bm25_scores.append(bm25.score(eval_qas[i]["question"]))

    top_k = 5  # 앞선 실험에서 sweet spot

    # ── 실험 1: α 값에 따른 정확도 ──
    alphas = np.arange(0, 1.05, 0.05)  # 0.0, 0.05, 0.10, ..., 1.0
    alpha_results = []

    print("\nα 값별 검색 정확도 측정 중...")
    for alpha in alphas:
        hits = 0
        for i in range(n):
            answer_clean = " ".join(eval_qas[i]["answer"].split())
            top_indices = weighted_rrf(
                all_dense_scores[i], all_bm25_scores[i], top_k, alpha
            )
            for idx in top_indices:
                if answer_clean in " ".join(chunks[idx].split()):
                    hits += 1
                    break
        alpha_results.append(hits / n * 100)

    best_alpha_idx = np.argmax(alpha_results)
    best_alpha = alphas[best_alpha_idx]
    best_score = alpha_results[best_alpha_idx]
    print(f"최적 α = {best_alpha:.2f} → 검색 정확도 {best_score:.1f}%")

    # ── 실험 2: 질문별 Dense/BM25 히트 여부 ──
    dense_hits_per_q = []
    bm25_hits_per_q = []

    for i in range(n):
        answer_clean = " ".join(eval_qas[i]["answer"].split())

        # Dense top-5
        dense_top = np.argsort(all_dense_scores[i])[::-1][:top_k]
        d_hit = any(answer_clean in " ".join(chunks[idx].split()) for idx in dense_top)
        dense_hits_per_q.append(d_hit)

        # BM25 top-5
        bm25_top = np.argsort(all_bm25_scores[i])[::-1][:top_k]
        b_hit = any(answer_clean in " ".join(chunks[idx].split()) for idx in bm25_top)
        bm25_hits_per_q.append(b_hit)

    # 상호보완 분석
    both_hit = sum(1 for d, b in zip(dense_hits_per_q, bm25_hits_per_q) if d and b)
    dense_only = sum(1 for d, b in zip(dense_hits_per_q, bm25_hits_per_q) if d and not b)
    bm25_only = sum(1 for d, b in zip(dense_hits_per_q, bm25_hits_per_q) if not d and b)
    neither = sum(1 for d, b in zip(dense_hits_per_q, bm25_hits_per_q) if not d and not b)

    print(f"\n[상호보완 분석] (top_k={top_k})")
    print(f"  둘 다 성공: {both_hit}건")
    print(f"  Dense만 성공: {dense_only}건")
    print(f"  BM25만 성공: {bm25_only}건")
    print(f"  둘 다 실패: {neither}건")

    # ============================================================
    # 시각화
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 그래프 1: α vs 정확도 곡선 ──
    ax1 = axes[0]
    ax1.plot(alphas, alpha_results, "b-o", markersize=4, linewidth=2)
    ax1.axvline(x=best_alpha, color="red", linestyle="--", alpha=0.7,
                label=f"최적 α={best_alpha:.2f}")
    ax1.axhline(y=best_score, color="red", linestyle="--", alpha=0.3)

    # 주요 지점 표시
    ax1.axvline(x=0.0, color="green", linestyle=":", alpha=0.5)
    ax1.axvline(x=0.5, color="orange", linestyle=":", alpha=0.5)
    ax1.axvline(x=1.0, color="purple", linestyle=":", alpha=0.5)

    ax1.annotate("BM25만", xy=(0.0, alpha_results[0]), fontsize=9,
                 xytext=(0.08, alpha_results[0] - 3),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 color="green")
    ax1.annotate("동일 가중치", xy=(0.5, alpha_results[10]), fontsize=9,
                 xytext=(0.55, alpha_results[10] - 3),
                 arrowprops=dict(arrowstyle="->", color="orange"),
                 color="orange")
    ax1.annotate("Dense만", xy=(1.0, alpha_results[-1]), fontsize=9,
                 xytext=(0.82, alpha_results[-1] + 2),
                 arrowprops=dict(arrowstyle="->", color="purple"),
                 color="purple")

    ax1.set_xlabel("α (Dense 가중치)", fontsize=12)
    ax1.set_ylabel("검색 정확도 (%)", fontsize=12)
    ax1.set_title("Dense 가중치(α)에 따른 검색 정확도", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([60, 100])

    # ── 그래프 2: 방법별 비교 막대 그래프 ──
    ax2 = axes[1]
    methods = ["Dense\n(임베딩)", "BM25\n(키워드)", f"Hybrid\n(α=0.5)", f"Hybrid\n(α={best_alpha:.2f})"]

    # 각 방법의 정확도
    dense_acc = sum(dense_hits_per_q) / n * 100
    bm25_acc = sum(bm25_hits_per_q) / n * 100
    hybrid_050 = alpha_results[10]  # α=0.5
    hybrid_best = best_score

    values = [dense_acc, bm25_acc, hybrid_050, hybrid_best]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
    bars = ax2.bar(methods, values, color=colors, edgecolor="white", linewidth=1.5)

    # 막대 위에 수치 표시
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax2.set_ylabel("검색 정확도 (%)", fontsize=12)
    ax2.set_title(f"검색 방법별 정확도 비교 (top_k={top_k})", fontsize=13)
    ax2.set_ylim([60, 105])
    ax2.grid(True, alpha=0.3, axis="y")

    # ── 그래프 3: 상호보완 벤 다이어그램 (막대로 표현) ──
    ax3 = axes[2]
    categories = ["둘 다\n성공", "Dense만\n성공", "BM25만\n성공", "둘 다\n실패"]
    counts = [both_hit, dense_only, bm25_only, neither]
    colors3 = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    bars3 = ax3.bar(categories, counts, color=colors3, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars3, counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val}건", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax3.set_ylabel("질문 수", fontsize=12)
    ax3.set_title("Dense와 BM25의 상호보완 관계", fontsize=13)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path(__file__).parent / "hybrid_search_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n시각화 저장: {output_path}")
    plt.close()


if __name__ == "__main__":
    asyncio.run(run_experiment())
