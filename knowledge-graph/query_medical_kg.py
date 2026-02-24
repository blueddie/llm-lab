"""
의료 지식 그래프 질의

그래프에 다양한 Cypher 쿼리를 날려서 의미 있는 답을 얻는다.
SQL로는 복잡한 JOIN이 필요한 질문도, Cypher에서는 관계를 따라가면 간단하다.
"""
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "testpassword")


def run_query(driver, title, query):
    """쿼리 실행 후 결과 출력"""
    print(f"\n{'─' * 50}")
    print(f"Q: {title}")
    print(f"{'─' * 50}")
    with driver.session() as session:
        result = session.run(query)
        records = list(result)
        if not records:
            print("  (결과 없음)")
            return
        for record in records:
            values = [f"{v}" for v in record.values()]
            print(f"  → {', '.join(values)}")


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()

    # ============================================================
    # 1단계: 기본 조회 — 단순 관계 따라가기
    # ============================================================
    print("\n" + "=" * 50)
    print("1단계: 기본 조회")
    print("=" * 50)

    run_query(driver,
        "고혈압을 치료하는 약은?",
        """
        MATCH (d:Drug)-[:TREATS]->(dis:Disease {name: "고혈압"})
        RETURN d.name AS 약물명
        """
    )

    run_query(driver,
        "천식의 증상은?",
        """
        MATCH (s:Symptom)-[:SYMPTOM_OF]->(dis:Disease {name: "천식"})
        RETURN s.name AS 증상
        """
    )

    run_query(driver,
        "폐렴을 진단하는 검사는?",
        """
        MATCH (t:Test)-[:DIAGNOSES]->(dis:Disease {name: "폐렴"})
        RETURN t.name AS 검사
        """
    )

    # ============================================================
    # 2단계: 다중 관계 — 여러 관계를 한 번에 따라가기
    # ============================================================
    print("\n" + "=" * 50)
    print("2단계: 다중 관계 질의")
    print("=" * 50)

    run_query(driver,
        "당뇨병 치료약의 부작용은?",
        """
        MATCH (d:Drug)-[:TREATS]->(dis:Disease {name: "제2형 당뇨병"})
        MATCH (d)-[:HAS_SIDE_EFFECT]->(se:SideEffect)
        RETURN d.name AS 약물, collect(se.name) AS 부작용목록
        """
    )

    run_query(driver,
        "피로감이 증상인 모든 질병과 각 질병의 치료약은?",
        """
        MATCH (s:Symptom {name: "피로감"})-[:SYMPTOM_OF]->(dis:Disease)
        MATCH (d:Drug)-[:TREATS]->(dis)
        RETURN dis.name AS 질병, collect(d.name) AS 치료약
        """
    )

    # ============================================================
    # 3단계: 경로 탐색 — 그래프의 진짜 힘
    # ============================================================
    print("\n" + "=" * 50)
    print("3단계: 경로 탐색")
    print("=" * 50)

    run_query(driver,
        "기침과 호흡곤란이 동시에 나타나는 질병은?",
        """
        MATCH (s1:Symptom {name: "기침"})-[:SYMPTOM_OF]->(dis:Disease)
        MATCH (s2:Symptom {name: "호흡곤란"})-[:SYMPTOM_OF]->(dis)
        RETURN dis.name AS 질병
        """
    )

    run_query(driver,
        "오심/구토 부작용이 있는 약물은 어떤 질병 치료약인가?",
        """
        MATCH (d:Drug)-[:HAS_SIDE_EFFECT]->(se:SideEffect {name: "오심/구토"})
        MATCH (d)-[:TREATS]->(dis:Disease)
        RETURN d.name AS 약물, dis.name AS 치료대상
        """
    )

    run_query(driver,
        "'기침' 증상으로 의심되는 질병 → 진단 검사 → 치료약까지 한 번에",
        """
        MATCH (s:Symptom {name: "기침"})-[:SYMPTOM_OF]->(dis:Disease)
        MATCH (t:Test)-[:DIAGNOSES]->(dis)
        MATCH (d:Drug)-[:TREATS]->(dis)
        RETURN dis.name AS 질병,
               collect(DISTINCT t.name) AS 진단검사,
               collect(DISTINCT d.name) AS 치료약
        """
    )

    # ============================================================
    # 4단계: 고급 질의 — 집계, 패턴 분석
    # ============================================================
    print("\n" + "=" * 50)
    print("4단계: 고급 질의")
    print("=" * 50)

    run_query(driver,
        "부작용이 가장 많은 약물 TOP 3",
        """
        MATCH (d:Drug)-[:HAS_SIDE_EFFECT]->(se:SideEffect)
        RETURN d.name AS 약물, count(se) AS 부작용수
        ORDER BY 부작용수 DESC
        LIMIT 3
        """
    )

    run_query(driver,
        "여러 질병의 공통 증상 (2개 이상 질병에 나타나는 증상)",
        """
        MATCH (s:Symptom)-[:SYMPTOM_OF]->(dis:Disease)
        WITH s, collect(dis.name) AS 질병목록, count(dis) AS 질병수
        WHERE 질병수 >= 2
        RETURN s.name AS 공통증상, 질병목록, 질병수
        ORDER BY 질병수 DESC
        """
    )

    run_query(driver,
        "두 질병 사이의 최단 경로 (당뇨병 ↔ 우울증)",
        """
        MATCH path = shortestPath(
            (d1:Disease {name: "제2형 당뇨병"})-[*]-(d2:Disease {name: "우울증"})
        )
        RETURN [n IN nodes(path) |
            CASE WHEN n:Disease THEN '질병:' + n.name
                 WHEN n:Symptom THEN '증상:' + n.name
                 WHEN n:Drug THEN '약물:' + n.name
                 WHEN n:SideEffect THEN '부작용:' + n.name
                 ELSE n.name END
        ] AS 경로
        """
    )

    driver.close()
    print("\n완료!")


if __name__ == "__main__":
    main()
