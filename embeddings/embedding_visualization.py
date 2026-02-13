"""
임베딩 시각화 — 1536차원을 2D로 압축해서 눈으로 확인

임베딩은 "의미가 비슷하면 벡터가 가깝다"는 개념인데,
1536차원 공간에서 "가깝다"를 어떻게 확인할까?

t-SNE (t-distributed Stochastic Neighbor Embedding):
  - 고차원 벡터 간의 상대적 거리를 최대한 보존하면서 2D로 압축
  - 가까운 것은 가깝게, 먼 것은 멀게 유지
  - 시각화 전용. 압축된 좌표 자체는 의미 없고, 클러스터 형태만 본다

결과물: 카테고리별로 색이 다른 산점도.
같은 카테고리끼리 뭉쳐 있으면 → 임베딩이 의미를 잘 잡고 있다는 증거.
"""

import asyncio
import os

import matplotlib
matplotlib.use("Agg")  # GUI 없이 파일로 저장
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.manifold import TSNE

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 카테고리별 텍스트 ---
# 5개 카테고리, 각 6개 텍스트. 일부러 키워드가 겹치지 않게 구성했다.
# 임베딩이 단어가 아니라 "의미"를 잡는다면, 같은 카테고리끼리 뭉쳐야 한다.

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
    # 1. 모든 텍스트와 카테고리 라벨 준비
    all_texts = []
    labels = []
    for category, texts in CATEGORY_TEXTS.items():
        all_texts.extend(texts)
        labels.extend([category] * len(texts))

    print(f"총 {len(all_texts)}개 텍스트, {len(CATEGORY_TEXTS)}개 카테고리\n")

    # 2. 임베딩 생성
    print("임베딩 생성 중...")
    embeddings = await get_embeddings(all_texts)
    embeddings_array = np.array(embeddings)
    print(f"임베딩 shape: {embeddings_array.shape}")  # (30, 1536)

    # 3. t-SNE로 2D 압축
    print("t-SNE 압축 중...")
    tsne = TSNE(
        n_components=2,     # 2차원으로 압축
        perplexity=8,       # 이웃 고려 범위 (데이터 수가 적으므로 낮게)
        random_state=42,    # 재현성
        max_iter=1000,      # 반복 횟수
    )
    coords = tsne.fit_transform(embeddings_array)  # (30, 2)
    print(f"압축 후 shape: {coords.shape}\n")

    # 4. 시각화
    categories = list(CATEGORY_TEXTS.keys())
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#F7DC6F", "#BB8FCE"]

    plt.figure(figsize=(12, 8))

    for i, category in enumerate(categories):
        mask = [l == category for l in labels]
        cat_coords = coords[mask]
        plt.scatter(
            cat_coords[:, 0],
            cat_coords[:, 1],
            c=colors[i],
            label=category,
            s=100,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )
        # 텍스트 라벨 (앞 15자만)
        for j, (x, y) in enumerate(cat_coords):
            text_idx = [k for k, l in enumerate(labels) if l == category][j]
            short_text = all_texts[text_idx][:15] + "..."
            plt.annotate(
                short_text,
                (x, y),
                fontsize=7,
                alpha=0.7,
                textcoords="offset points",
                xytext=(5, 5),
                fontfamily="Malgun Gothic",
            )

    plt.legend(
        fontsize=11,
        loc="best",
        prop={"family": "Malgun Gothic"},
    )
    plt.title(
        "임베딩 시각화 (t-SNE) — 같은 카테고리끼리 뭉치는지 확인",
        fontsize=14,
        fontfamily="Malgun Gothic",
    )
    plt.xlabel("t-SNE 1", fontsize=11)
    plt.ylabel("t-SNE 2", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = "embedding_visualization.png"
    plt.savefig(output_path, dpi=150)
    print(f"시각화 저장: {output_path}")

    # 5. 카테고리 간 평균 유사도 (숫자로도 확인)
    print("\n=== 카테고리 간 평균 코사인 유사도 ===\n")

    # 카테고리별 평균 벡터
    category_means = {}
    for category in categories:
        mask = [l == category for l in labels]
        cat_embeddings = embeddings_array[mask]
        category_means[category] = cat_embeddings.mean(axis=0)

    # 카테고리 간 유사도 행렬
    print(f"{'':>10s}", end="")
    for cat in categories:
        print(f"  {cat:>8s}", end="")
    print()

    for cat_a in categories:
        print(f"{cat_a:>10s}", end="")
        for cat_b in categories:
            vec_a = category_means[cat_a]
            vec_b = category_means[cat_b]
            sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
            print(f"  {sim:>8.4f}", end="")
        print()

    print("\n대각선(자기 자신)이 가장 높고, 다른 카테고리는 낮을수록 임베딩이 잘 분리된 것")


if __name__ == "__main__":
    asyncio.run(main())

'''
총 30개 텍스트, 5개 카테고리

임베딩 생성 중...
임베딩 shape: (30, 1536)
t-SNE 압축 중...
압축 후 shape: (30, 2)

시각화 저장: embedding_visualization.png

=== 카테고리 간 평균 코사인 유사도 ===

               프로그래밍        요리       스포츠        음악        여행
     프로그래밍    1.0000    0.4428    0.5304    0.6114    0.5178
        요리    0.4428    1.0000    0.4591    0.6020    0.5095
       스포츠    0.5304    0.4591    1.0000    0.5510    0.4837
        음악    0.6114    0.6020    0.5510    1.0000    0.5299
        여행    0.5178    0.5095    0.4837    0.5299    1.0000

대각선(자기 자신)이 가장 높고, 다른 카테고리는 낮을수록 임베딩이 잘 분리된 것
'''