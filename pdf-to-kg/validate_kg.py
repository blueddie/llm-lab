"""
지식 그래프 후처리: 유효성 검사 및 정리.

LLM이 추출한 결과를 코드로 검증하고 자동 수정하는 단계.
프롬프트 개선만으로는 100% 깨끗한 결과를 보장할 수 없기 때문에,
규칙 기반 후처리로 나머지를 잡는다.

검증 항목:
  1. 노드 타입 유효성 — 스키마에 정의된 5가지 타입만 허용
  2. 유령 노드 탐지 — 관계에는 있지만 노드 목록에 없는 엔티티
  3. 중복 관계 제거 — 완전히 동일한 트리플 제거
  4. 고아 노드 탐지 — 어떤 관계에도 참여하지 않는 노드
"""

import json
from schema import NODE_TYPES, RELATION_TYPES

INPUT_PATH = "kg_result.json"
OUTPUT_PATH = "kg_validated.json"

VALID_NODE_TYPES = set(NODE_TYPES.keys())
VALID_RELATION_TYPES = set(RELATION_TYPES.keys())


def load_kg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_and_fix(kg: dict) -> dict:
    """유효성 검사 + 자동 수정을 수행한다."""
    nodes = kg.get("nodes", [])
    relations = kg.get("relations", [])
    fixes = []  # 수정 내역 기록

    # -------------------------------------------------------
    # 검증 1: 노드 타입 유효성
    # -------------------------------------------------------
    # 스키마에 없는 타입을 쓴 노드를 찾아낸다
    for node in nodes:
        if node["type"] not in VALID_NODE_TYPES:
            fixes.append(f"[타입 위반] '{node['name']}'의 타입 '{node['type']}'은 스키마에 없음")

    # -------------------------------------------------------
    # 검증 2: 유령 노드 탐지
    # -------------------------------------------------------
    # 관계의 source/target이 노드 목록에 있는지 확인
    node_names = {n["name"] for n in nodes}

    ghost_nodes = set()
    for rel in relations:
        if rel["source"] not in node_names:
            ghost_nodes.add(rel["source"])
        if rel["target"] not in node_names:
            ghost_nodes.add(rel["target"])

    for ghost in ghost_nodes:
        fixes.append(f"[유령 노드] '{ghost}'이 관계에 있지만 노드 목록에 없음")

    # -------------------------------------------------------
    # 자동 수정: 유령 노드를 가진 관계 제거
    # -------------------------------------------------------
    # 유령 노드가 포함된 관계는 신뢰할 수 없으므로 제거
    clean_relations = []
    removed_count = 0
    for rel in relations:
        if rel["source"] in ghost_nodes or rel["target"] in ghost_nodes:
            removed_count += 1
        else:
            clean_relations.append(rel)

    if removed_count > 0:
        fixes.append(f"[자동 수정] 유령 노드를 포함한 관계 {removed_count}개 제거")

    # -------------------------------------------------------
    # 검증 3: 관계 타입 유효성
    # -------------------------------------------------------
    invalid_rel_relations = []
    valid_relations = []
    for rel in clean_relations:
        if rel["relation"] not in VALID_RELATION_TYPES:
            invalid_rel_relations.append(rel)
            fixes.append(
                f"[관계 타입 위반] '{rel['source']} --{rel['relation']}--> {rel['target']}'"
            )
        else:
            valid_relations.append(rel)
    clean_relations = valid_relations

    # -------------------------------------------------------
    # 검증 4: 중복 관계 제거
    # -------------------------------------------------------
    seen = set()
    deduped_relations = []
    dup_count = 0
    for rel in clean_relations:
        key = (rel["source"], rel["relation"], rel["target"])
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
            deduped_relations.append(rel)

    if dup_count > 0:
        fixes.append(f"[자동 수정] 중복 관계 {dup_count}개 제거")

    # -------------------------------------------------------
    # 검증 5: 고아 노드 탐지
    # -------------------------------------------------------
    # 어떤 관계에도 참여하지 않는 노드 = 그래프에서 의미 없음
    used_names = set()
    for rel in deduped_relations:
        used_names.add(rel["source"])
        used_names.add(rel["target"])

    orphan_nodes = [n for n in nodes if n["name"] not in used_names]
    connected_nodes = [n for n in nodes if n["name"] in used_names]

    if orphan_nodes:
        names = [n["name"] for n in orphan_nodes]
        fixes.append(f"[고아 노드] 관계 없는 노드 {len(orphan_nodes)}개: {names}")

    # -------------------------------------------------------
    # 결과 조립
    # -------------------------------------------------------
    result = {
        "nodes": connected_nodes,
        "relations": deduped_relations,
    }

    return result, fixes


def print_summary(original: dict, validated: dict, fixes: list[str]):
    """원본과 검증 결과를 비교 출력."""
    orig_nodes = len(original.get("nodes", []))
    orig_rels = len(original.get("relations", []))
    val_nodes = len(validated.get("nodes", []))
    val_rels = len(validated.get("relations", []))

    print("=== 후처리 결과 ===\n")

    print(f"노드: {orig_nodes}개 -> {val_nodes}개 ({orig_nodes - val_nodes}개 제거)")
    print(f"관계: {orig_rels}개 -> {val_rels}개 ({orig_rels - val_rels}개 제거)")

    print(f"\n수정 내역 ({len(fixes)}건):")
    for fix in fixes:
        print(f"  {fix}")

    # 노드 타입별 분포
    print("\n노드 타입별 분포:")
    type_counts = {}
    for node in validated["nodes"]:
        t = node["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}개")

    # 관계 타입별 분포
    print("\n관계 타입별 분포:")
    rel_counts = {}
    for rel in validated["relations"]:
        r = rel["relation"]
        rel_counts[r] = rel_counts.get(r, 0) + 1
    for r, count in sorted(rel_counts.items()):
        print(f"  {r}: {count}개")


def main():
    original = load_kg(INPUT_PATH)
    validated, fixes = validate_and_fix(original)

    print_summary(original, validated, fixes)

    # 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(validated, f, ensure_ascii=False, indent=2)
    print(f"\n검증된 결과 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
