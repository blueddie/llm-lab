"""
t-SNE vs UMAP — 두 차원 축소 알고리즘 비교

둘 다 고차원 벡터를 2D로 압축하는데, 방식이 다르다:

t-SNE:
  - 가까운 점들 사이의 거리를 잘 보존 (지역 구조)
  - 클러스터 "안"의 구조가 정확
  - 클러스터 "간" 거리는 신뢰할 수 없음
  - 느림 (O(N²))

UMAP:
  - 지역 구조 + 전역 구조 모두 보존하려고 시도
  - 클러스터 간 상대적 거리도 의미 있음
  - 빠름 (대규모 데이터에 유리)
  - 실무에서 더 많이 쓰이는 추세

같은 데이터에 두 알고리즘을 적용해서 결과를 나란히 비교한다.
"""

import asyncio
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.manifold import TSNE
import umap

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


def plot_2d(ax, coords, labels, categories, colors, title):
    """산점도 + 라벨을 그리는 공통 함수"""
    for i, category in enumerate(categories):
        mask = [l == category for l in labels]
        cat_coords = coords[mask]
        ax.scatter(
            cat_coords[:, 0], cat_coords[:, 1],
            c=colors[i], label=category, s=100, alpha=0.8,
            edgecolors="white", linewidth=0.5,
        )
    ax.legend(fontsize=9, loc="best", prop={"family": "Malgun Gothic"})
    ax.set_title(title, fontsize=13, fontfamily="Malgun Gothic", fontweight="bold")
    ax.grid(True, alpha=0.3)


async def main():
    # 1. 데이터 준비 + 임베딩
    all_texts = []
    labels = []
    categories = list(CATEGORY_TEXTS.keys())

    for category, texts in CATEGORY_TEXTS.items():
        all_texts.extend(texts)
        labels.extend([category] * len(texts))

    print(f"총 {len(all_texts)}개 텍스트\n")
    print("임베딩 생성 중...")
    embeddings = await get_embeddings(all_texts)
    embeddings_array = np.array(embeddings)

    # 2. t-SNE
    print("t-SNE 압축 중...")
    tsne = TSNE(n_components=2, perplexity=8, random_state=42, max_iter=1000)
    tsne_coords = tsne.fit_transform(embeddings_array)

    # 3. UMAP
    print("UMAP 압축 중...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=8,      # 이웃 수 (perplexity와 비슷한 역할)
        min_dist=0.3,        # 점들 사이 최소 거리 (작을수록 밀집)
        random_state=42,
    )
    umap_coords = reducer.fit_transform(embeddings_array)

    # 4. 나란히 시각화
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#F7DC6F", "#BB8FCE"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    plot_2d(ax1, tsne_coords, labels, categories, colors,
            "t-SNE — 지역 구조 중심 (클러스터 내부가 정확)")
    plot_2d(ax2, umap_coords, labels, categories, colors,
            "UMAP — 전역 구조도 보존 (클러스터 간 거리도 의미 있음)")

    ax1.set_xlabel("t-SNE 1", fontsize=10)
    ax1.set_ylabel("t-SNE 2", fontsize=10)
    ax2.set_xlabel("UMAP 1", fontsize=10)
    ax2.set_ylabel("UMAP 2", fontsize=10)

    plt.suptitle(
        "같은 임베딩, 다른 압축 — t-SNE vs UMAP",
        fontsize=15, fontfamily="Malgun Gothic", fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    output_path = "tsne_vs_umap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n시각화 저장: {output_path}")

    # 5. 정량 비교: 클러스터 분리도 (실루엣 스코어)
    from sklearn.metrics import silhouette_score

    # 라벨을 숫자로 변환
    label_to_num = {cat: i for i, cat in enumerate(categories)}
    numeric_labels = [label_to_num[l] for l in labels]

    tsne_silhouette = silhouette_score(tsne_coords, numeric_labels)
    umap_silhouette = silhouette_score(umap_coords, numeric_labels)

    print(f"\n=== 클러스터 분리도 (실루엣 스코어) ===")
    print(f"  -1 ~ 1 범위. 높을수록 클러스터가 잘 분리된 것\n")
    print(f"  t-SNE:  {tsne_silhouette:.4f}")
    print(f"  UMAP:   {umap_silhouette:.4f}")

    if umap_silhouette > tsne_silhouette:
        print(f"\n  → UMAP이 클러스터를 더 잘 분리했다 (차이: {umap_silhouette - tsne_silhouette:+.4f})")
    else:
        print(f"\n  → t-SNE가 클러스터를 더 잘 분리했다 (차이: {tsne_silhouette - umap_silhouette:+.4f})")

    print(f"\n참고: 실루엣 스코어는 2D 압축 결과에 대한 것이지, 원본 임베딩의 품질이 아님")


if __name__ == "__main__":
    asyncio.run(main())


'''
=== 클러스터 분리도 (실루엣 스코어) ===
  -1 ~ 1 범위. 높을수록 클러스터가 잘 분리된 것

  t-SNE:  -0.0051
  UMAP:   0.0489

  → UMAP이 클러스터를 더 잘 분리했다 (차이: +0.0540)

참고: 실루엣 스코어는 2D 압축 결과에 대한 것이지, 원본 임베딩의 품질이 아님
'''