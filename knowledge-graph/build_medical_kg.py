"""
의료 지식 그래프 구축

Python에서 Neo4j에 연결해서 의료 지식 그래프를 만든다.
질병 - 증상 - 약물 - 부작용 - 검사 간의 관계를 그래프로 표현.

핵심 개념:
- Neo4j Python Driver로 DB 연결
- Cypher 쿼리를 Python에서 실행
- 트랜잭션 단위로 데이터 생성
"""
from neo4j import GraphDatabase

# Neo4j 연결 설정 (Docker 컨테이너)
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "testpassword")


def clear_database(driver):
    """기존 데이터 전부 삭제"""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("DB 초기화 완료")


def build_knowledge_graph(driver):
    """의료 지식 그래프 생성"""

    # 하나의 트랜잭션에서 전체 그래프를 생성한다.
    # MERGE는 "이미 있으면 재사용, 없으면 생성" — CREATE의 중복 문제를 방지한다.
    query = """
    // === 질병 ===
    MERGE (diabetes:Disease {name: "제2형 당뇨병"})
    MERGE (hypertension:Disease {name: "고혈압"})
    MERGE (asthma:Disease {name: "천식"})
    MERGE (pneumonia:Disease {name: "폐렴"})
    MERGE (depression:Disease {name: "우울증"})

    // === 증상 ===
    MERGE (thirst:Symptom {name: "다음(갈증)"})
    MERGE (frequent_urination:Symptom {name: "빈뇨"})
    MERGE (fatigue:Symptom {name: "피로감"})
    MERGE (headache:Symptom {name: "두통"})
    MERGE (dizziness:Symptom {name: "어지러움"})
    MERGE (cough:Symptom {name: "기침"})
    MERGE (dyspnea:Symptom {name: "호흡곤란"})
    MERGE (wheezing:Symptom {name: "천명(쌕쌕거림)"})
    MERGE (fever:Symptom {name: "발열"})
    MERGE (chest_pain:Symptom {name: "흉통"})
    MERGE (insomnia:Symptom {name: "불면"})
    MERGE (appetite_loss:Symptom {name: "식욕저하"})

    // === 약물 ===
    MERGE (metformin:Drug {name: "메트포르민", type: "경구약"})
    MERGE (insulin:Drug {name: "인슐린", type: "주사"})
    MERGE (amlodipine:Drug {name: "암로디핀", type: "경구약"})
    MERGE (losartan:Drug {name: "로사르탄", type: "경구약"})
    MERGE (salbutamol:Drug {name: "살부타몰", type: "흡입제"})
    MERGE (amoxicillin:Drug {name: "아목시실린", type: "경구약"})
    MERGE (ssri:Drug {name: "SSRI(플루옥세틴)", type: "경구약"})

    // === 부작용 ===
    MERGE (hypoglycemia:SideEffect {name: "저혈당"})
    MERGE (renal:SideEffect {name: "신장기능저하"})
    MERGE (nausea:SideEffect {name: "오심/구토"})
    MERGE (edema:SideEffect {name: "부종"})
    MERGE (dry_cough:SideEffect {name: "마른기침"})
    MERGE (tremor:SideEffect {name: "떨림"})
    MERGE (diarrhea:SideEffect {name: "설사"})
    MERGE (weight_gain:SideEffect {name: "체중증가"})

    // === 검사 ===
    MERGE (hba1c:Test {name: "HbA1c (당화혈색소)"})
    MERGE (fbs:Test {name: "공복혈당검사"})
    MERGE (bp_test:Test {name: "혈압측정"})
    MERGE (ecg:Test {name: "심전도(ECG)"})
    MERGE (pft:Test {name: "폐기능검사(PFT)"})
    MERGE (chest_xray:Test {name: "흉부 X-ray"})
    MERGE (blood_culture:Test {name: "혈액배양검사"})
    MERGE (phq9:Test {name: "PHQ-9 (우울증 선별)"})

    // === 관계: 증상 → 질병 ===
    MERGE (thirst)-[:SYMPTOM_OF]->(diabetes)
    MERGE (frequent_urination)-[:SYMPTOM_OF]->(diabetes)
    MERGE (fatigue)-[:SYMPTOM_OF]->(diabetes)
    MERGE (headache)-[:SYMPTOM_OF]->(hypertension)
    MERGE (dizziness)-[:SYMPTOM_OF]->(hypertension)
    MERGE (fatigue)-[:SYMPTOM_OF]->(hypertension)
    MERGE (cough)-[:SYMPTOM_OF]->(asthma)
    MERGE (dyspnea)-[:SYMPTOM_OF]->(asthma)
    MERGE (wheezing)-[:SYMPTOM_OF]->(asthma)
    MERGE (fever)-[:SYMPTOM_OF]->(pneumonia)
    MERGE (cough)-[:SYMPTOM_OF]->(pneumonia)
    MERGE (dyspnea)-[:SYMPTOM_OF]->(pneumonia)
    MERGE (chest_pain)-[:SYMPTOM_OF]->(pneumonia)
    MERGE (insomnia)-[:SYMPTOM_OF]->(depression)
    MERGE (fatigue)-[:SYMPTOM_OF]->(depression)
    MERGE (appetite_loss)-[:SYMPTOM_OF]->(depression)

    // === 관계: 약물 → 질병 (치료) ===
    MERGE (metformin)-[:TREATS]->(diabetes)
    MERGE (insulin)-[:TREATS]->(diabetes)
    MERGE (amlodipine)-[:TREATS]->(hypertension)
    MERGE (losartan)-[:TREATS]->(hypertension)
    MERGE (salbutamol)-[:TREATS]->(asthma)
    MERGE (amoxicillin)-[:TREATS]->(pneumonia)
    MERGE (ssri)-[:TREATS]->(depression)

    // === 관계: 약물 → 부작용 ===
    MERGE (metformin)-[:HAS_SIDE_EFFECT]->(nausea)
    MERGE (metformin)-[:HAS_SIDE_EFFECT]->(renal)
    MERGE (metformin)-[:HAS_SIDE_EFFECT]->(diarrhea)
    MERGE (insulin)-[:HAS_SIDE_EFFECT]->(hypoglycemia)
    MERGE (insulin)-[:HAS_SIDE_EFFECT]->(weight_gain)
    MERGE (amlodipine)-[:HAS_SIDE_EFFECT]->(edema)
    MERGE (losartan)-[:HAS_SIDE_EFFECT]->(dizziness)
    MERGE (salbutamol)-[:HAS_SIDE_EFFECT]->(tremor)
    MERGE (amoxicillin)-[:HAS_SIDE_EFFECT]->(nausea)
    MERGE (amoxicillin)-[:HAS_SIDE_EFFECT]->(diarrhea)
    MERGE (ssri)-[:HAS_SIDE_EFFECT]->(nausea)
    MERGE (ssri)-[:HAS_SIDE_EFFECT]->(insomnia)
    MERGE (ssri)-[:HAS_SIDE_EFFECT]->(weight_gain)

    // === 관계: 검사 → 질병 (진단) ===
    MERGE (hba1c)-[:DIAGNOSES]->(diabetes)
    MERGE (fbs)-[:DIAGNOSES]->(diabetes)
    MERGE (bp_test)-[:DIAGNOSES]->(hypertension)
    MERGE (ecg)-[:DIAGNOSES]->(hypertension)
    MERGE (pft)-[:DIAGNOSES]->(asthma)
    MERGE (chest_xray)-[:DIAGNOSES]->(pneumonia)
    MERGE (blood_culture)-[:DIAGNOSES]->(pneumonia)
    MERGE (phq9)-[:DIAGNOSES]->(depression)

    RETURN count(*) AS total_operations
    """

    with driver.session() as session:
        result = session.run(query)
        record = result.single()
        print(f"그래프 생성 완료 (operations: {record['total_operations']})")


def print_stats(driver):
    """그래프 통계 출력"""
    with driver.session() as session:
        # 노드 수
        result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC")
        print("\n[노드 통계]")
        total_nodes = 0
        for record in result:
            print(f"  {record['label']:15s} {record['cnt']}개")
            total_nodes += record['cnt']
        print(f"  {'합계':15s} {total_nodes}개")

        # 관계 수
        result = session.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC")
        print("\n[관계 통계]")
        total_rels = 0
        for record in result:
            print(f"  {record['rel']:20s} {record['cnt']}개")
            total_rels += record['cnt']
        print(f"  {'합계':20s} {total_rels}개")


if __name__ == "__main__":
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # 연결 확인
    driver.verify_connectivity()
    print("Neo4j 연결 성공!\n")

    # 그래프 구축
    clear_database(driver)
    build_knowledge_graph(driver)
    print_stats(driver)

    driver.close()
