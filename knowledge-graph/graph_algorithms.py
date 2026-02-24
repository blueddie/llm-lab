"""
그래프 알고리즘 & 시각화

Neo4j에서 그래프를 가져와 Python으로 분석한다.
1. 커뮤니티 탐지 (Louvain) — GraphRAG의 핵심
2. PageRank — 중요한 노드 찾기
3. 매개 중심성 (Betweenness Centrality) — 브릿지 노드 찾기
4. 인터랙티브 시각화 (pyvis)

핵심 라이브러리:
- networkx: 그래프 알고리즘 실행
- community (python-louvain): 커뮤니티 탐지
- pyvis: 인터랙티브 그래프 시각화 (HTML)
"""
import community as community_louvain  # python-louvain
import networkx as nx
from neo4j import GraphDatabase
from pyvis.network import Network

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "testpassword")


def load_graph_from_neo4j(driver) -> nx.Graph:
    """
    Neo4j에서 전체 그래프를 가져와 networkx 그래프로 변환한다.

    왜 networkx로 변환하나?
    - Neo4j는 '저장소'이고, networkx는 '분석 도구'다
    - 실무에서도 Neo4j에서 데이터를 꺼내 Python으로 분석하는 패턴이 일반적
    """
    G = nx.Graph()  # 무방향 그래프 (커뮤니티 탐지용)

    with driver.session() as session:
        # 모든 노드 가져오기 (elementId 사용 — id()는 deprecated)
        result = session.run("""
            MATCH (n)
            RETURN elementId(n) AS id, labels(n)[0] AS label, n.name AS name
        """)
        for record in result:
            G.add_node(
                record["id"],
                label=record["label"],
                name=record["name"],
            )

        # 모든 관계 가져오기
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS rel_type
        """)
        for record in result:
            G.add_edge(
                record["source"],
                record["target"],
                rel_type=record["rel_type"],
            )

    print(f"그래프 로드 완료: {G.number_of_nodes()}개 노드, {G.number_of_edges()}개 엣지")
    return G


# ============================================================
# 1. 커뮤니티 탐지 (Louvain Algorithm)
# ============================================================
def detect_communities(G: nx.Graph) -> dict:
    """
    Louvain 알고리즘으로 커뮤니티를 탐지한다.

    커뮤니티란?
    - 서로 밀접하게 연결된 노드들의 그룹
    - "이 노드들은 같은 주제/분야에 속한다"를 자동으로 발견

    왜 GraphRAG에 중요한가?
    - Microsoft GraphRAG는 커뮤니티 단위로 지식을 요약한다
    - 질문이 들어오면, 관련 커뮤니티를 찾아서 그 요약으로 답변 생성
    """
    partition = community_louvain.best_partition(G)

    # 커뮤니티별 노드 그룹핑
    communities = {}
    for node_id, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(G.nodes[node_id]["name"])

    print(f"\n{'=' * 50}")
    print(f"커뮤니티 탐지 결과 (Louvain)")
    print(f"{'=' * 50}")
    print(f"발견된 커뮤니티: {len(communities)}개\n")

    for comm_id, members in sorted(communities.items()):
        print(f"  커뮤니티 {comm_id}: {members}")

    return partition


# ============================================================
# 2. PageRank — 중요한 노드 찾기
# ============================================================
def run_pagerank(G: nx.Graph):
    """
    PageRank로 그래프에서 가장 중요한 노드를 찾는다.

    PageRank란?
    - 구글이 웹페이지 중요도를 매기기 위해 만든 알고리즘
    - "많은 노드와 연결된 노드"가 높은 점수를 받음
    - 단순 연결 수가 아니라 "중요한 노드와 연결된 노드"가 더 높은 점수

    의료 KG에서의 의미:
    - PageRank 높은 노드 = 여러 질병/약물/증상과 관련된 핵심 개체
    """
    pr = nx.pagerank(G, max_iter=200, tol=1e-04)

    # 이름과 함께 정렬
    pr_named = []
    for node_id, score in pr.items():
        node_data = G.nodes[node_id]
        pr_named.append((node_data["name"], node_data["label"], score))

    pr_named.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'=' * 50}")
    print(f"PageRank (중요도 상위 10)")
    print(f"{'=' * 50}")
    for name, label, score in pr_named[:10]:
        bar = "█" * int(score * 200)
        print(f"  {name:15s} [{label:10s}] {score:.4f} {bar}")

    return pr


# ============================================================
# 3. 매개 중심성 (Betweenness Centrality) — 브릿지 노드
# ============================================================
def run_betweenness(G: nx.Graph):
    """
    매개 중심성(Betweenness Centrality)으로 브릿지 노드를 찾는다.

    매개 중심성이란?
    - "이 노드를 거치는 최단 경로가 얼마나 많은가?"
    - 높은 노드 = 서로 다른 그룹을 연결하는 '다리' 역할

    의료 KG에서의 의미:
    - 매개 중심성 높은 노드 = 여러 질병 영역을 잇는 공통 요소
    - 예: "피로감"은 당뇨병-고혈압-우울증을 연결하는 브릿지
    """
    bc = nx.betweenness_centrality(G)

    bc_named = []
    for node_id, score in bc.items():
        node_data = G.nodes[node_id]
        bc_named.append((node_data["name"], node_data["label"], score))

    bc_named.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'=' * 50}")
    print(f"매개 중심성 (브릿지 노드 상위 10)")
    print(f"{'=' * 50}")
    for name, label, score in bc_named[:10]:
        bar = "█" * int(score * 100)
        print(f"  {name:15s} [{label:10s}] {score:.4f} {bar}")

    return bc


# ============================================================
# 4. 인터랙티브 시각화 (pyvis)
# ============================================================
def visualize_graph(G: nx.Graph, partition: dict):
    """
    pyvis로 인터랙티브 그래프를 HTML로 생성한다.

    시각화 요소:
    - 노드 색상: 커뮤니티별로 다른 색
    - 노드 크기: PageRank 점수에 비례
    - 엣지 라벨: 관계 타입 표시
    """
    # 커뮤니티별 색상
    community_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
        "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    ]

    # 노드 타입별 모양 (shape)
    label_shapes = {
        "Disease": "dot",
        "Drug": "diamond",
        "Symptom": "star",
        "SideEffect": "triangle",
        "Test": "square",
    }

    # PageRank 계산 (노드 크기용)
    pr = nx.pagerank(G, max_iter=200, tol=1e-04)

    # pyvis 네트워크 생성
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
    )

    # 물리 시뮬레이션 설정 (노드 배치)
    net.barnes_hut(gravity=-3000, spring_length=150)

    # 노드 추가
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        name = node_data["name"]
        label = node_data["label"]
        comm_id = partition.get(node_id, 0)

        color = community_colors[comm_id % len(community_colors)]
        shape = label_shapes.get(label, "dot")
        size = 15 + pr.get(node_id, 0) * 500  # PageRank 기반 크기

        net.add_node(
            node_id,
            label=name,
            color=color,
            shape=shape,
            size=size,
            title=f"{label}: {name}\n커뮤니티: {comm_id}\nPageRank: {pr.get(node_id, 0):.4f}",
        )

    # 엣지 추가
    for source, target, data in G.edges(data=True):
        net.add_edge(
            source,
            target,
            title=data.get("rel_type", ""),
            label=data.get("rel_type", ""),
            color="rgba(255,255,255,0.3)",
            font={"size": 10, "color": "rgba(255,255,255,0.5)"},
        )

    output_path = "medical_kg_visualization.html"
    net.save_graph(output_path)
    print(f"\n시각화 저장: {output_path}")
    print("브라우저에서 열어보세요!")

    return output_path


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    print("Neo4j 연결 성공!")

    # Neo4j → networkx 변환
    G = load_graph_from_neo4j(driver)
    driver.close()

    # 알고리즘 실행
    partition = detect_communities(G)
    run_pagerank(G)
    run_betweenness(G)

    # 시각화
    visualize_graph(G, partition)


if __name__ == "__main__":
    main()
