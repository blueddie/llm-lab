"""
임베딩 차원 결정 — 실무 평가 방식

눈으로 "1위가 같네" 하고 보는 게 아니라,
평가 데이터셋 + 검색 품질 지표로 정량적으로 비교한다.

핵심 지표:
- Recall@K: 정답 문서가 상위 K개 안에 포함되는 비율 ("찾아야 할 걸 찾았는가?")
- MRR: 정답 문서가 몇 번째에 나왔는지의 역수 평균 (1위=1.0, 2위=0.5, 3위=0.33)

의사결정 기준:
- Recall@3이 95% 이상이면서 가장 낮은 차원 → 그게 최적 차원
"""

import asyncio
import os

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DIMENSIONS = [256, 512, 1024, 1536]


# --- 1. 평가 데이터셋 ---
# 실무에서는 이걸 사람이 직접 만들거나, LLM으로 생성 후 사람이 검증한다.
# 핵심: 각 질문에 대해 "정답 문서"가 뭔지 미리 정해두는 것.

DOCUMENTS = [
    # 인사/근태
    "연차는 입사일 기준으로 매년 15일이 부여됩니다.",
    "연차 사용 시 최소 하루 전에 팀장에게 신청해야 합니다.",
    "경조사 휴가는 결혼 5일, 출산 10일, 사망 3일이 지급됩니다.",
    "출근 시간은 오전 9시이며, 유연근무제 적용 시 8~10시 사이에 출근하면 됩니다.",
    "원격 근무는 주 2회까지 가능하며, 팀장 승인이 필요합니다.",
    "원격 근무 시 업무 시작과 종료를 슬랙으로 공유해야 합니다.",
    # 복리후생
    "야근 식대는 1만원까지 법인카드로 결제 가능합니다.",
    "점심시간은 12시부터 1시까지이며, 자율적으로 조정 가능합니다.",
    "건강검진은 연 1회 회사 부담으로 제공됩니다.",
    "자기개발비는 분기당 30만원까지 지원됩니다.",
    "통근버스는 강남역, 판교역에서 운행합니다.",
    # 평가/급여
    "성과 평가는 반기별로 진행되며, OKR 기반으로 측정합니다.",
    "연봉 협상은 매년 1월에 진행됩니다.",
    "인센티브는 팀 목표 달성률에 따라 분기별 지급됩니다.",
    "퇴직금은 근속 1년 이상 시 지급되며, 평균 임금 기준으로 산정됩니다.",
    # 온보딩/교육
    "신규 입사자는 첫 주에 온보딩 프로그램에 참여해야 합니다.",
    "사내 기술 세미나는 매주 금요일 오후 4시에 진행됩니다.",
    "교육 휴가는 연간 5일까지 사용 가능합니다.",
    # 보안/정책
    "사내 코드는 개인 저장소에 복사할 수 없습니다.",
    "VPN 접속은 사외에서 업무 시 필수입니다.",
]

# 평가 데이터: (질문, 정답 문서 인덱스 리스트)
# 하나의 질문에 정답이 여러 개일 수 있다
EVAL_SET = [
    ("휴가 며칠 쓸 수 있어?", [0, 2]),                    # 연차 + 경조사
    ("재택근무 가능한가요?", [4, 5]),                       # 원격 근무 관련 2개
    ("야근하면 밥값 나와?", [6]),                           # 야근 식대
    ("연봉은 언제 올라?", [12]),                            # 연봉 협상
    ("새로 입사했는데 뭐 해야 해?", [15]),                   # 온보딩
    ("공부하고 싶은데 지원되는 거 있어?", [9, 17]),          # 자기개발비 + 교육 휴가
    ("회사 코드 깃허브에 올려도 돼?", [18]),                 # 코드 보안
    ("보너스는 어떻게 받아?", [13]),                         # 인센티브
    ("몇 시에 출근해야 돼?", [3]),                          # 출근 시간
    ("건강검진 해주나요?", [8]),                             # 건강검진
    ("집에서 일할 때 VPN 써야 해?", [19]),                   # VPN
    ("평가는 어떻게 하나요?", [11]),                         # 성과 평가
]


# --- 2. 임베딩 ---

async def get_embeddings(texts: list[str], dimensions: int) -> list[list[float]]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions=dimensions,
    )
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- 3. 검색 품질 지표 ---

def evaluate(
    query_embeddings: list[list[float]],
    doc_embeddings: list[list[float]],
    eval_set: list[tuple[str, list[int]]],
    top_k: int = 3,
) -> dict:
    """
    Recall@K와 MRR을 계산한다.

    Recall@K: 정답 문서가 상위 K개 안에 하나라도 있으면 성공
    MRR: 정답 문서 중 가장 높은 순위의 역수 (1위=1.0, 2위=0.5, ...)
    """
    recall_hits = 0
    reciprocal_ranks = []

    for qi, (query, relevant_indices) in enumerate(eval_set):
        # 모든 문서와의 유사도 계산
        scores = []
        for di, doc_emb in enumerate(doc_embeddings):
            score = cosine_similarity(query_embeddings[qi], doc_emb)
            scores.append((di, score))

        # 점수 높은 순 정렬
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [idx for idx, _ in scores[:top_k]]

        # Recall@K: 정답 중 하나라도 top_k에 있는가?
        if any(idx in top_k_indices for idx in relevant_indices):
            recall_hits += 1

        # MRR: 정답 문서 중 가장 높은 순위 찾기
        best_rank = None
        for rank, (idx, _) in enumerate(scores, 1):
            if idx in relevant_indices:
                best_rank = rank
                break

        reciprocal_ranks.append(1.0 / best_rank if best_rank else 0.0)

    recall_at_k = recall_hits / len(eval_set)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return {
        "recall@k": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
        "recall_hits": recall_hits,
        "total": len(eval_set),
    }


# --- 4. 차원별 비교 ---

async def main():
    queries = [q for q, _ in EVAL_SET]

    print(f"문서 수: {len(DOCUMENTS)}")
    print(f"평가 질문 수: {len(EVAL_SET)}")
    print(f"검색 상위 K: 3")
    print()

    print("=== 차원별 검색 품질 비교 ===\n")
    print(f"  {'차원':>6s}  {'Recall@3':>10s}  {'MRR':>8s}  {'저장 공간':>12s}  {'판정'}")
    print("  " + "─" * 58)

    results = []

    for dim in DIMENSIONS:
        # 임베딩 생성 (문서 + 질문을 한 번에)
        all_texts = DOCUMENTS + queries
        all_embeddings = await get_embeddings(all_texts, dimensions=dim)

        doc_embeddings = all_embeddings[: len(DOCUMENTS)]
        query_embeddings = all_embeddings[len(DOCUMENTS) :]

        # 평가
        metrics = evaluate(query_embeddings, doc_embeddings, EVAL_SET, top_k=3)

        # 저장 공간 계산 (float32 = 4bytes)
        storage_per_doc = dim * 4
        total_storage = storage_per_doc * len(DOCUMENTS)

        # 판정
        if metrics["recall@k"] >= 0.95:
            verdict = "✓ 충분"
        elif metrics["recall@k"] >= 0.85:
            verdict = "△ 주의"
        else:
            verdict = "✗ 부족"

        print(
            f"  {dim:>6d}  "
            f"{metrics['recall@k']:>10.1%}  "
            f"{metrics['mrr']:>8.4f}  "
            f"{total_storage:>10,d}B  "
            f"{verdict}"
        )

        results.append({"dim": dim, **metrics, "storage": total_storage})

    # --- 질문별 상세 결과 (가장 낮은 차원 vs 최대 차원) ---
    print(f"\n=== 질문별 상세 비교 ({DIMENSIONS[0]}차원 vs {DIMENSIONS[-1]}차원) ===\n")

    for dim in [DIMENSIONS[0], DIMENSIONS[-1]]:
        all_texts = DOCUMENTS + queries
        all_embeddings = await get_embeddings(all_texts, dimensions=dim)
        doc_embeddings = all_embeddings[: len(DOCUMENTS)]
        query_embeddings = all_embeddings[len(DOCUMENTS) :]

        print(f"--- {dim}차원 ---")
        for qi, (query, relevant_indices) in enumerate(EVAL_SET):
            scores = []
            for di, doc_emb in enumerate(doc_embeddings):
                score = cosine_similarity(query_embeddings[qi], doc_emb)
                scores.append((di, score))
            scores.sort(key=lambda x: x[1], reverse=True)

            top1_idx, top1_score = scores[0]
            hit = "✓" if top1_idx in relevant_indices else "✗"
            print(f"  {hit} {query[:20]:<22s} → {DOCUMENTS[top1_idx][:30]}... ({top1_score:.4f})")
        print()

    # --- 의사결정 가이드 ---
    print("=== 의사결정 ===\n")
    print("실무 기준:")
    print("  1. Recall@3 ≥ 95% 인 차원 중 가장 낮은 것을 선택")
    print("  2. MRR이 크게 떨어지지 않는지 확인 (1위 정확도)")
    print("  3. 저장 공간 / 검색 속도와 트레이드오프 판단")
    print()

    best = None
    for r in results:
        if r["recall@k"] >= 0.95:
            best = r
            break

    if best:
        print(f"  → 이 데이터셋 기준 추천: {best['dim']}차원")
        print(f"    Recall@3: {best['recall@k']:.1%}, MRR: {best['mrr']:.4f}")
        baseline = results[-1]
        saving = (1 - best["storage"] / baseline["storage"]) * 100
        print(f"    저장 공간: 1536차원 대비 {saving:.0f}% 절감")
    else:
        print("  → Recall@3 95% 이상을 달성하는 차원이 없음. 1536차원 사용 권장")


if __name__ == "__main__":
    asyncio.run(main())

'''
문서 수: 20
평가 질문 수: 12
검색 상위 K: 3

=== 차원별 검색 품질 비교 ===

      차원    Recall@3       MRR         저장 공간  판정
  ──────────────────────────────────────────────────────────
     256      100.0%    0.9583      20,480B  ✓ 충분
     512      100.0%    0.9583      40,960B  ✓ 충분
    1024      100.0%    0.9583      81,920B  ✓ 충분
    1536      100.0%    0.9583     122,880B  ✓ 충분

=== 질문별 상세 비교 (256차원 vs 1536차원) ===

--- 256차원 ---
  ✗ 휴가 며칠 쓸 수 있어?          → 교육 휴가는 연간 5일까지 사용 가능합니다.... (0.4618)
  ✓ 재택근무 가능한가요?            → 원격 근무는 주 2회까지 가능하며, 팀장 승인이 필요합... (0.4393)
  ✓ 야근하면 밥값 나와?            → 야근 식대는 1만원까지 법인카드로 결제 가능합니다.... (0.4146)
  ✓ 연봉은 언제 올라?             → 연봉 협상은 매년 1월에 진행됩니다.... (0.4824)
  ✓ 새로 입사했는데 뭐 해야 해?       → 신규 입사자는 첫 주에 온보딩 프로그램에 참여해야 합니... (0.4520)
  ✓ 공부하고 싶은데 지원되는 거 있어?    → 자기개발비는 분기당 30만원까지 지원됩니다.... (0.4299)
  ✓ 회사 코드 깃허브에 올려도 돼?      → 사내 코드는 개인 저장소에 복사할 수 없습니다.... (0.4399)
  ✓ 보너스는 어떻게 받아?           → 인센티브는 팀 목표 달성률에 따라 분기별 지급됩니다.... (0.2840)
  ✓ 몇 시에 출근해야 돼?           → 출근 시간은 오전 9시이며, 유연근무제 적용 시 8~1... (0.4490)
  ✓ 건강검진 해주나요?             → 건강검진은 연 1회 회사 부담으로 제공됩니다.... (0.6122)
  ✓ 집에서 일할 때 VPN 써야 해?     → VPN 접속은 사외에서 업무 시 필수입니다.... (0.5810)
  ✓ 평가는 어떻게 하나요?           → 성과 평가는 반기별로 진행되며, OKR 기반으로 측정합... (0.3798)

--- 1536차원 ---
  ✗ 휴가 며칠 쓸 수 있어?          → 교육 휴가는 연간 5일까지 사용 가능합니다.... (0.4792)
  ✓ 재택근무 가능한가요?            → 원격 근무는 주 2회까지 가능하며, 팀장 승인이 필요합... (0.4207)
  ✓ 야근하면 밥값 나와?            → 야근 식대는 1만원까지 법인카드로 결제 가능합니다.... (0.3779)
  ✓ 연봉은 언제 올라?             → 연봉 협상은 매년 1월에 진행됩니다.... (0.4690)
  ✓ 새로 입사했는데 뭐 해야 해?       → 신규 입사자는 첫 주에 온보딩 프로그램에 참여해야 합니... (0.4190)
  ✓ 공부하고 싶은데 지원되는 거 있어?    → 자기개발비는 분기당 30만원까지 지원됩니다.... (0.4078)
  ✓ 회사 코드 깃허브에 올려도 돼?      → 사내 코드는 개인 저장소에 복사할 수 없습니다.... (0.3702)
  ✓ 보너스는 어떻게 받아?           → 인센티브는 팀 목표 달성률에 따라 분기별 지급됩니다.... (0.2321)
  ✓ 몇 시에 출근해야 돼?           → 출근 시간은 오전 9시이며, 유연근무제 적용 시 8~1... (0.4183)
  ✓ 건강검진 해주나요?             → 건강검진은 연 1회 회사 부담으로 제공됩니다.... (0.6406)
  ✓ 집에서 일할 때 VPN 써야 해?     → VPN 접속은 사외에서 업무 시 필수입니다.... (0.5866)
  ✓ 평가는 어떻게 하나요?           → 성과 평가는 반기별로 진행되며, OKR 기반으로 측정합... (0.4379)

=== 의사결정 ===

실무 기준:
  1. Recall@3 ≥ 95% 인 차원 중 가장 낮은 것을 선택
  2. MRR이 크게 떨어지지 않는지 확인 (1위 정확도)
  3. 저장 공간 / 검색 속도와 트레이드오프 판단

  → 이 데이터셋 기준 추천: 256차원
    Recall@3: 100.0%, MRR: 0.9583
    저장 공간: 1536차원 대비 83% 절감


=== 지표 정리 ===

Recall@K (재현율)
  - 정답 문서가 상위 K개 안에 하나라도 있으면 성공, 그 비율
  - 예: 12개 질문 중 12개가 top3에 정답 포함 → Recall@3 = 100%
  - "찾아야 할 걸 찾았는가?"를 측정
  - 언제 보나: RAG의 가장 기본 지표. 이게 낮으면 LLM에게 엉뚱한 문서를 전달하게 된다.
         Recall이 충분히 높은 상태에서 다른 최적화(차원 축소, 청킹 전략 등)를 시도해야 한다.

MRR (Mean Reciprocal Rank)
  - 정답 문서가 몇 번째에 나왔는지의 역수를 평균낸 것
  - 1위=1.0, 2위=0.5, 3위=0.33, ...
  - 예: 12개 중 11개가 1위, 1개가 2위 → (11×1.0 + 1×0.5) / 12 = 0.9583
  - "정답이 얼마나 위에 있는가?"를 측정
  - 언제 보나: Recall@K가 같을 때 품질을 더 세밀하게 비교할 때.
         top1만 사용하는 시스템이면 MRR이 Recall보다 중요하다.

실무에서의 지표 선택 기준:
  - top_k개를 LLM에 전부 넘기는 RAG → Recall@K 중심으로 판단
  - top1만 쓰거나, 순위가 중요한 경우 → MRR 중심으로 판단
  - 정답이 여러 개이고 전부 찾아야 하는 경우 → Recall@K를 높은 K에서 측정

참고:
  - "휴가 며칠 쓸 수 있어?" → 1위가 "교육 휴가"로 나옴 (256, 1536 모두 동일)
  - ground truth에 "교육 휴가"를 포함시키지 않아서 ✗로 찍힌 것
  - 이건 모델/차원의 문제가 아니라 평가 데이터셋의 문제
  - 실무 교훈: RAG 성능을 좌우하는 건 모델보다 평가 데이터셋의 품질인 경우가 많다
'''