"""
유사도 히트맵 — 텍스트 간 유사도를 색상 격자로 시각화

t-SNE가 "위치"로 유사도를 표현했다면, 히트맵은 "색상"으로 표현한다.
30×30 격자에서:
  - 빨간색/밝은색 → 유사도 높음
  - 파란색/어두운색 → 유사도 낮음

같은 카테고리 블록이 밝게 뜨면 → 임베딩이 의미를 잘 구분하고 있다는 증거
카테고리 간에도 밝은 영역이 있으면 → 의미적으로 겹치는 부분이 있다는 뜻
"""

import asyncio
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 시각화 파일과 동일한 데이터 사용
CATEGORY_TEXTS = {
    "프로그래밍": [
        "파이썬에서 리스트 컴프리헨션으로 데이터를 필터링했다",
        "깃허브에 풀 리퀘스트를 올리고 코드 리뷰를 받았다",
        "비동기 함수로 API 호출 성능을 개선했다",
        "도커 컨테이너에 서비스를 배포했다",
        "타입스크립트로 프론트엔드를 리팩토링했다",
        "데이터베이스 인덱스를 추가해서 쿼리 속도를 올렸다",
    ],
    "요리": [
        "마늘을 다져서 올리브오일에 볶았다",
        "반죽을 30분간 숙성시킨 뒤 오븐에 구웠다",
        "된장찌개에 두부와 호박을 넣고 끓였다",
        "소금과 후추로 간을 맞추고 레몬즙을 뿌렸다",
        "파스타 면을 알덴테로 삶아서 소스에 버무렸다",
        "냉장고에 있는 재료로 볶음밥을 만들었다",
    ],
    "스포츠": [
        "전반전에 프리킥으로 선제골을 넣었다",
        "마라톤 풀코스를 4시간 만에 완주했다",
        "3점 슛이 들어가면서 역전에 성공했다",
        "수영 자유형 100미터에서 개인 최고 기록을 세웠다",
        "9회말 투아웃에서 끝내기 홈런이 터졌다",
        "테니스 결승전에서 타이브레이크 끝에 우승했다",
    ],
    "음악": [
        "피아노 소나타를 연습하다가 어려운 패시지에서 막혔다",
        "기타 코드를 잡으면서 노래를 불렀다",
        "드럼 비트에 맞춰 베이스 라인을 연주했다",
        "콘서트홀에서 오케스트라 공연을 관람했다",
        "보컬 녹음을 마치고 믹싱 작업에 들어갔다",
        "새 앨범에 수록할 곡의 멜로디를 작곡했다",
    ],
    "여행": [
        "공항에서 체크인하고 면세점을 둘러봤다",
        "호텔 체크인 후 근처 야시장을 구경했다",
        "렌터카를 빌려서 해안도로를 드라이브했다",
        "현지 가이드와 함께 유적지를 투어했다",
        "배낭을 메고 기차를 타고 다음 도시로 이동했다",
        "숙소에서 일출을 보며 커피를 마셨다",
    ],
}


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


async def main():
    # 1. 데이터 준비
    all_texts = []
    labels = []
    categories = list(CATEGORY_TEXTS.keys())

    for category, texts in CATEGORY_TEXTS.items():
        all_texts.extend(texts)
        labels.extend([category] * len(texts))

    print(f"총 {len(all_texts)}개 텍스트\n")

    # 2. 임베딩
    print("임베딩 생성 중...")
    embeddings = await get_embeddings(all_texts)
    embeddings_array = np.array(embeddings)

    # 3. 유사도 행렬 계산 (30×30)
    # 정규화 후 내적 = 코사인 유사도
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized = embeddings_array / norms
    similarity_matrix = normalized @ normalized.T  # (30, 30)

    print(f"유사도 행렬 shape: {similarity_matrix.shape}\n")

    # 4. 히트맵 시각화
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(similarity_matrix, cmap="RdYlBu_r", vmin=0, vmax=1)

    # 카테고리 구분선
    n_per_cat = 6
    for i in range(1, len(categories)):
        pos = i * n_per_cat - 0.5
        ax.axhline(y=pos, color="black", linewidth=2)
        ax.axvline(x=pos, color="black", linewidth=2)

    # 카테고리 라벨
    for i, category in enumerate(categories):
        mid = i * n_per_cat + n_per_cat / 2 - 0.5
        ax.text(-1.5, mid, category, ha="right", va="center",
                fontsize=11, fontweight="bold", fontfamily="Malgun Gothic")
        ax.text(mid, -1.5, category, ha="center", va="bottom",
                fontsize=11, fontweight="bold", fontfamily="Malgun Gothic",
                rotation=45)

    # 텍스트 라벨 (짧게)
    short_labels = [t[:12] + "..." for t in all_texts]
    ax.set_xticks(range(len(all_texts)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7,
                       fontfamily="Malgun Gothic")
    ax.set_yticks(range(len(all_texts)))
    ax.set_yticklabels(short_labels, fontsize=7,
                       fontfamily="Malgun Gothic")

    # 컬러바
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("코사인 유사도", fontsize=11, fontfamily="Malgun Gothic")

    ax.set_title("임베딩 유사도 히트맵 — 같은 카테고리 블록이 밝을수록 잘 분리된 것",
                 fontsize=14, fontfamily="Malgun Gothic", pad=20)

    plt.tight_layout()

    output_path = "embedding_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"히트맵 저장: {output_path}")

    # 5. 카테고리 내부 vs 외부 유사도 비교
    print("\n=== 카테고리 내부 vs 외부 평균 유사도 ===\n")
    print(f"  {'카테고리':<10s}  {'내부 유사도':>10s}  {'외부 유사도':>10s}  {'차이':>8s}")
    print("  " + "─" * 44)

    for i, category in enumerate(categories):
        start = i * n_per_cat
        end = start + n_per_cat

        # 내부: 같은 카테고리끼리 (대각선 제외)
        internal_block = similarity_matrix[start:end, start:end]
        mask = ~np.eye(n_per_cat, dtype=bool)
        internal_avg = internal_block[mask].mean()

        # 외부: 다른 카테고리와
        external_scores = []
        for j in range(len(all_texts)):
            if j < start or j >= end:
                for k in range(start, end):
                    external_scores.append(similarity_matrix[k][j])
        external_avg = np.mean(external_scores)

        diff = internal_avg - external_avg
        print(f"  {category:<10s}  {internal_avg:>10.4f}  {external_avg:>10.4f}  {diff:>+8.4f}")

    print("\n차이가 클수록 해당 카테고리의 임베딩이 잘 분리된 것")


if __name__ == "__main__":
    asyncio.run(main())

'''
총 30개 텍스트

임베딩 생성 중...
유사도 행렬 shape: (30, 30)

히트맵 저장: embedding_heatmap.png

=== 카테고리 내부 vs 외부 평균 유사도 ===

  카테고리            내부 유사도      외부 유사도        차이
  ────────────────────────────────────────────
  프로그래밍           0.2940      0.2063   +0.0877
  요리              0.2716      0.1941   +0.0775
  스포츠             0.2426      0.1909   +0.0517
  음악              0.2371      0.2157   +0.0214
  여행              0.2512      0.1939   +0.0572

차이가 클수록 해당 카테고리의 임베딩이 잘 분리된 것

  - 내부 유사도 = 같은 카테고리 텍스트끼리의 평균 유사도. 예: 프로그래밍 6개 텍스트끼리 비교한 15쌍의 평균
  - 외부 유사도 = 다른 카테고리 텍스트와의 평균 유사도. 예: 프로그래밍 6개 vs 나머지 24개의 평균

  차이가 클수록 "같은 카테고리끼리는 가깝고, 다른 카테고리와는 멀다"는 뜻이니까 임베딩이 잘 분리된 거다.

  결과를 보면:

  프로그래밍  +0.0877  ← 가장 잘 분리됨
  요리      +0.0775
  여행      +0.0572
  스포츠    +0.0517
  음악      +0.0214  ← 가장 분리 안 됨

  음악이 +0.0214로 가장 낮다. 내부 유사도(0.2371)와 외부 유사도(0.2157)가 거의 차이가 없다는 건, 음악 텍스트끼리의 거리나 다른 카테고리와의 거리가 비슷하다는 뜻이다.
  t-SNE에서 흩어졌던 이유, 히트맵에서 블록이 덜 선명했던 이유가 전부 이 숫자 하나로 설명된다.


'''