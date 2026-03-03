"""
LLM 기반 지식 그래프 추출 (C 전략: 절별 추출 → 병합)

전체 흐름:
  1. cleaned_text.txt를 4개 절로 분리
  2. 각 절에 대해 LLM으로 엔티티/관계 추출
  3. 4개 결과를 LLM에 보내 병합/보완
  4. 최종 결과를 JSON으로 저장

핵심 기법:
  - Structured Output: OpenAI의 response_format으로 JSON 응답 강제
  - 스키마 기반 프롬프트: schema.py의 정의를 프롬프트에 포함시켜
    LLM이 일관된 타입으로 추출하도록 유도
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from schema import NODE_TYPES, RELATION_TYPES

load_dotenv()
client = OpenAI()

# --- 설정 ---
MODEL = "gpt-4o-mini"  # 학습용이니 비용 효율적인 모델로
CLEANED_TEXT_PATH = "cleaned_text.txt"
OUTPUT_PATH = "kg_result.json"

# ============================================================
# 1단계: 텍스트를 절(section)별로 분리
# ============================================================
# 절 제목 패턴으로 텍스트를 잘라냄
# README에서 분석했던 4개 절의 시작 키워드를 사용

SECTION_MARKERS = [
    "아동의 개인정보 보호(법 제22조의2)",
    "민감정보의 처리 제한(법 제23조)",
    "고유식별정보의 처리 제한(법 제24조)",
    "주민등록번호 처리의 제한(법 제24조의2)",
]


def split_into_sections(text: str) -> list[dict]:
    """텍스트를 절 제목 기준으로 분리한다.

    왜 이렇게 하나?
      단순히 줄 수로 자르면 문맥이 끊길 수 있다.
      절 제목을 기준으로 자르면 각 절이 완전한 의미 단위가 된다.
    """
    sections = []
    for i, marker in enumerate(SECTION_MARKERS):
        start = text.find(marker)
        if start == -1:
            print(f"  [경고] 절 마커를 찾을 수 없음: {marker}")
            continue

        # 다음 절의 시작점 또는 텍스트 끝까지가 이 절의 범위
        if i + 1 < len(SECTION_MARKERS):
            next_start = text.find(SECTION_MARKERS[i + 1])
            end = next_start if next_start != -1 else len(text)
        else:
            end = len(text)

        section_text = text[start:end].strip()
        sections.append({
            "title": marker,
            "text": section_text,
            "char_count": len(section_text),
        })

    return sections


# ============================================================
# 2단계: 스키마를 프롬프트용 텍스트로 변환
# ============================================================
def build_schema_prompt() -> str:
    """schema.py의 정의를 LLM이 읽을 수 있는 텍스트로 변환.

    왜 스키마를 프롬프트에 넣나?
      LLM에게 "자유롭게 추출해라"하면 매번 다른 타입명을 쓴다.
      스키마를 명시하면 일관된 타입으로 추출하게 유도할 수 있다.
      (예: "정보 카테고리" 대신 항상 "정보유형"으로 뽑게 됨)
    """
    lines = ["## 노드 타입 (엔티티)"]
    for type_name, info in NODE_TYPES.items():
        examples = ", ".join(info["examples"][:5])
        lines.append(f"- **{type_name}**: {info['description']} (예: {examples})")

    lines.append("\n## 관계 타입")
    for rel_name, info in RELATION_TYPES.items():
        lines.append(f"- **{rel_name}**: {info['description']}")
        lines.append(f"  예시: {info['example']}")

    return "\n".join(lines)


# ============================================================
# 3단계: 절별 추출 (C 전략의 1단계)
# ============================================================
# 프롬프트 개선 포인트:
#   [문제1 해결] 노드 타입을 5가지로 한정하는 규칙을 더 강하게 명시
#   [문제2 해결] 관계 방향에 대한 명확한 가이드 추가
#   [문제3 해결] source/target은 반드시 nodes에 있어야 한다고 명시
#   [문제4 해결] 법조항 이름 형식 통일 규칙 추가
EXTRACTION_SYSTEM_PROMPT = """\
당신은 법률 문서에서 지식 그래프를 추출하는 전문가입니다.

주어진 스키마에 정의된 노드 타입과 관계 타입만 사용하여 엔티티와 관계를 추출하세요.

## 필수 규칙

### 노드 규칙
1. 노드의 type은 반드시 다음 5가지 중 하나: 정보유형, 주체, 법조항, 의무, 제재
   - 절대로 다른 타입을 사용하지 말 것 (예: "고유식별정보"는 타입이 아니라 name)
   - 주민등록번호, 여권번호 등은 type="정보유형"으로 분류
2. 노드의 name은 원문의 표현을 간결하게 사용 (불필요하게 긴 문장 금지)
3. 법조항 이름은 "제X조" 형식으로 통일 (예: "제22조의2", "제23조")
   - "법 제X조" 형식 사용 금지

### 관계 규칙
4. 관계의 source/target은 반드시 nodes 배열에 있는 name과 정확히 일치해야 함
5. 관계 방향을 정확히 지킬 것:
   - 하위유형: "민감정보 --하위유형--> 개인정보" (하위가 source, 상위가 target)
   - 보호주체: "아동 --보호주체--> 법정대리인" (보호받는 쪽이 source)
   - 필요의무: "고유식별정보 --필요의무--> 암호화" (정보/주체가 source, 의무가 target)
6. 모호하거나 불확실한 관계는 추출하지 말 것"""


def extract_from_section(section: dict, schema_prompt: str) -> dict:
    """한 절에서 엔티티/관계를 추출한다.

    핵심 포인트:
      - response_format={"type": "json_object"}로 JSON 응답을 강제
      - 스키마 + 텍스트를 함께 전달해서 LLM이 스키마에 맞게 추출하도록 유도
    """
    user_prompt = f"""\
아래 스키마와 텍스트를 읽고, 엔티티(노드)와 관계를 추출하여 JSON으로 반환하세요.

{schema_prompt}

---

## 텍스트
{section['text']}

---

다음 JSON 형식으로 응답하세요:
{{
  "nodes": [
    {{"name": "엔티티 이름", "type": "노드타입"}}
  ],
  "relations": [
    {{"source": "출발 엔티티 이름", "relation": "관계타입", "target": "도착 엔티티 이름"}}
  ]
}}"""

    print(f"  → LLM 호출 중: {section['title'][:30]}... ({section['char_count']:,}자)")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,  # 추출 작업은 창의성이 아니라 정확성이 중요 → 0
    )

    result = json.loads(response.choices[0].message.content)
    node_count = len(result.get("nodes", []))
    rel_count = len(result.get("relations", []))
    print(f"    [완료] 노드 {node_count}개, 관계 {rel_count}개 추출")

    return result


# ============================================================
# 4단계: 병합 및 보완 (C 전략의 2단계)
# ============================================================
MERGE_SYSTEM_PROMPT = """\
당신은 지식 그래프 병합 전문가입니다.

여러 절에서 각각 추출된 엔티티와 관계를 하나의 일관된 지식 그래프로 병합하세요.

## 수행할 작업
1. **중복 노드 통합**: 같은 대상인데 이름이 다른 노드를 하나로 통합
   (예: "별도의 동의"와 "별도 동의"는 같은 엔티티 → 하나로)
2. **절 간 관계 추가**: 서로 다른 절의 엔티티 사이에 관계가 있으면 추가
   (예: "주민등록번호"는 "고유식별정보"의 하위유형)
3. **누락 보완**: 명백히 빠진 중요 관계가 있으면 추가

## 필수 검증 규칙
4. 노드의 type은 반드시 다음 5가지 중 하나: 정보유형, 주체, 법조항, 의무, 제재
5. 관계의 source/target은 반드시 nodes에 존재하는 name과 일치해야 함
   - nodes에 없는 엔티티가 관계에 등장하면, 해당 노드를 nodes에 추가하거나 관계를 제거
6. 법조항 이름은 "제X조" 형식으로 통일 ("법 제X조" 형식 사용 금지)
7. 관계 방향을 정확히 지킬 것 (하위유형: 하위→상위, 보호주체: 보호받는쪽→보호자)"""


def merge_extractions(
    section_results: list[dict], schema_prompt: str
) -> dict:
    """4개 절의 추출 결과를 병합한다.

    왜 LLM에게 병합을 시키나?
      단순 합치기(union)로는 중복 제거나 절 간 관계 추가가 안 된다.
      LLM이 전체 결과를 보면서 "이건 같은 엔티티다", "이 둘 사이에 관계가 있다"를
      판단하게 하는 것이 핵심.
    """
    # 각 절의 결과를 텍스트로 정리
    sections_text = ""
    for i, result in enumerate(section_results):
        sections_text += f"\n### 절 {i + 1} 추출 결과\n"
        sections_text += json.dumps(result, ensure_ascii=False, indent=2)
        sections_text += "\n"

    user_prompt = f"""\
아래는 법률 문서의 4개 절에서 각각 추출한 지식 그래프입니다.
이를 하나의 통합된 지식 그래프로 병합하세요.

{schema_prompt}

---

{sections_text}

---

병합된 결과를 다음 JSON 형식으로 응답하세요:
{{
  "nodes": [
    {{"name": "엔티티 이름", "type": "노드타입"}}
  ],
  "relations": [
    {{"source": "출발 엔티티 이름", "relation": "관계타입", "target": "도착 엔티티 이름"}}
  ],
  "merge_notes": "병합 과정에서 수행한 작업 설명 (통합한 노드, 추가한 관계 등)"
}}"""

    print("  → LLM 호출 중: 결과 병합...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)
    node_count = len(result.get("nodes", []))
    rel_count = len(result.get("relations", []))
    print(f"    [완료] 병합 완료: 노드 {node_count}개, 관계 {rel_count}개")

    return result


# ============================================================
# 메인 실행
# ============================================================
def main():
    # 텍스트 로드
    with open(CLEANED_TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"텍스트 로드 완료: {len(text):,}자\n")

    # 1단계: 절 분리
    print("[1단계] 텍스트를 절별로 분리")
    sections = split_into_sections(text)
    for s in sections:
        print(f"  - {s['title'][:40]}  ({s['char_count']:,}자)")
    print()

    # 2단계: 절별 추출
    print("[2단계] 절별 엔티티/관계 추출")
    schema_prompt = build_schema_prompt()
    section_results = []
    for section in sections:
        result = extract_from_section(section, schema_prompt)
        section_results.append(result)
    print()

    # 절별 결과를 중간 파일로 저장 (디버깅용)
    with open("kg_sections.json", "w", encoding="utf-8") as f:
        json.dump(section_results, f, ensure_ascii=False, indent=2)
    print("  중간 결과 저장: kg_sections.json\n")

    # 3단계: 병합
    print("[3단계] 결과 병합 및 보완")
    merged = merge_extractions(section_results, schema_prompt)
    print()

    # 병합 노트 출력
    if "merge_notes" in merged:
        print(f"  병합 노트: {merged['merge_notes']}\n")

    # 최종 결과 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"최종 결과 저장: {OUTPUT_PATH}")

    # 요약 통계
    print(f"\n=== 최종 요약 ===")
    print(f"노드: {len(merged.get('nodes', []))}개")
    print(f"관계: {len(merged.get('relations', []))}개")

    # 노드 타입별 개수
    type_counts = {}
    for node in merged.get("nodes", []):
        t = node.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in sorted(type_counts.items()):
        print(f"  - {t}: {count}개")


if __name__ == "__main__":
    main()
