"""
스키마 검증을 위한 테스트 추출 스크립트.

목적:
  정의한 스키마(노드 타입 5개, 관계 타입 9개)가 실제 텍스트에서
  의미 있는 엔티티/관계를 뽑아낼 수 있는지 확인한다.
  샘플 텍스트 하나만 넣어서 빠르게 테스트.
"""

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from schema import NODE_TYPES, RELATION_TYPES

load_dotenv("../.env")

client = Anthropic()

# --- 샘플 텍스트 (아동 개인정보 보호 - 주요 내용 부분) ---
SAMPLE_TEXT = """
가. 아동을 대신한 법정대리인의 동의
개인정보처리자는 14세 미만 아동의 개인정보를 처리하기 위하여 이 법에 따른 동의를 받아야
할 때에는 그 법정대리인의 동의를 받아야 하며, 법정대리인이 동의하였는지를 확인하여야
한다. (법 제22조의2 제1항, 제3항, 영 제17조의2 제1항, 제3항)

아동은 개인정보 수집 목적의 진위를 평가하는 데에 어려움이 클 수 있어, 아동을 대상으로 한
무분별한 정보 수집을 방지하기 위하여 법정대리인이 아동을 대신해서 동의를 받도록 하였다.
따라서, 법정대리인의 동의를 받은 경우에는 아동의 동의를 중복하여 받을 필요는 없다.

다. 아동으로부터 직접 수집 가능한 정보
개인정보처리자가 법정대리인의 동의를 얻기 위해서는 법정대리인의 이름과 연락처를 알아야
하기 때문에, 법정대리인의 동의를 받기 위하여 필요한 최소한의 정보(법정대리인의 성명 및
연락처에 관한 정보)는 법정대리인의 동의 없이 해당 아동으로부터 직접 수집할 수 있다. (법
제22조의2 제2항, 영 제17조의2 제2항)

5. 제재 규정
법정대리인의 동의를 받지 아니하고 만 14세 미만인 아동의 개인정보를
처리한 경우 (제22조의2제1항 위반, 제26조제8항에 따라 준용되는 경우를 포함)
-> 전체 매출액의 100분의 3 이하의 과징금 (제64조의2제1항제2호)

법정대리인의 동의를 받지 아니하고 만 14세 미만 아동의 개인정보를
처리한 자 (제22조의2제1항 위반, 제26조제8항에 따라 준용되는 경우를 포함)
-> 5년 이하의 징역 또는 5천만원 이하의 벌금 (제71조제3호)
"""

# --- 프롬프트 구성 ---
# 스키마 정보를 프롬프트에 포함
node_desc = "\n".join(
    f"- {name}: {info['description']} (예: {', '.join(info['examples'][:3])})"
    for name, info in NODE_TYPES.items()
)

rel_desc = "\n".join(
    f"- {name}: {info['description']} (예: {info['example']})"
    for name, info in RELATION_TYPES.items()
)

prompt = f"""다음 텍스트에서 엔티티(노드)와 관계를 추출해주세요.

## 노드 타입
{node_desc}

## 관계 타입
{rel_desc}

## 규칙
1. 반드시 위에 정의된 노드 타입과 관계 타입만 사용하세요.
2. 텍스트에 명시적으로 나타난 정보만 추출하세요. 추론하지 마세요.
3. 동일한 엔티티는 같은 이름으로 통일하세요.

## 출력 형식
JSON으로 출력하세요:
{{
  "nodes": [
    {{"name": "엔티티 이름", "type": "노드 타입"}},
    ...
  ],
  "relations": [
    {{"source": "출발 노드 이름", "relation": "관계 타입", "target": "도착 노드 이름"}},
    ...
  ]
}}

## 텍스트
{SAMPLE_TEXT}
"""

# --- API 호출 ---
print("Anthropic API 호출 중...")
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2000,
    messages=[{"role": "user", "content": prompt}],
)

result_text = response.content[0].text
print(f"\n=== LLM 응답 ===\n{result_text}")

# --- JSON 파싱 시도 ---
try:
    # ```json ... ``` 블록에서 JSON 추출
    if "```json" in result_text:
        json_str = result_text.split("```json")[1].split("```")[0]
    elif "```" in result_text:
        json_str = result_text.split("```")[1].split("```")[0]
    else:
        json_str = result_text

    result = json.loads(json_str)
    print(f"\n=== 추출 결과 ===")
    print(f"노드: {len(result['nodes'])}개")
    for node in result["nodes"]:
        print(f"  [{node['type']}] {node['name']}")
    print(f"\n관계: {len(result['relations'])}개")
    for rel in result["relations"]:
        print(f"  {rel['source']} --{rel['relation']}--> {rel['target']}")

    # 결과 저장
    with open("test_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: test_result.json")

except json.JSONDecodeError as e:
    print(f"\nJSON 파싱 실패: {e}")
    print("LLM 응답을 직접 확인해주세요.")
