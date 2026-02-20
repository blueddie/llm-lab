"""
KorQuAD 2.0 데이터를 로컬에 저장하는 스크립트

HuggingFace에서 validation 셋을 받아서:
1. 200개 문서만 샘플링 (학습용으로 충분)
2. HTML → 텍스트 전처리
3. JSON으로 data/ 폴더에 저장
"""
import json
from pathlib import Path
from datasets import load_dataset
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def html_to_text(html: str) -> str:
    """HTML에서 텍스트만 추출"""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def clean_answer(answer: dict) -> str:
    """답변에서 HTML 태그 제거"""
    text = answer["text"]
    if "<" in text:
        return BeautifulSoup(text, "html.parser").get_text(strip=True)
    return text


# 1. 데이터 다운로드
print("KorQuAD 2.0 validation 셋 다운로드 중...")
val_files = [
    f"https://huggingface.co/datasets/KorQuAD/squad_kor_v2/resolve/refs%2Fconvert%2Fparquet/squad_kor_v2/partial-validation/{i:04d}.parquet"
    for i in range(4)
]
dataset = load_dataset("parquet", data_files={"validation": val_files}, split="validation")
print(f"전체 {len(dataset)}개 QA 쌍 로드 완료")

# 2. 문서 단위로 그룹핑
docs = {}
for item in dataset:
    title = item["title"]
    if title not in docs:
        docs[title] = {
            "title": title,
            "url": item["url"],
            "context_html": item["context"],
            "context_text": html_to_text(item["context"]),
            "qas": [],
        }
    docs[title]["qas"].append({
        "id": item["id"],
        "question": item["question"],
        "answer_text": clean_answer(item["answer"]),
        "answer_raw": item["answer"]["text"],
    })

print(f"고유 문서: {len(docs)}개")

# 3. 200개 문서 샘플링 (QA가 2개 이상인 문서 우선)
doc_list = sorted(docs.values(), key=lambda d: len(d["qas"]), reverse=True)
sampled = doc_list[:200]

total_qas = sum(len(d["qas"]) for d in sampled)
print(f"샘플링: {len(sampled)}개 문서, {total_qas}개 QA 쌍")

# 4. 저장
output = {
    "description": "KorQuAD 2.0 validation subset (200 docs)",
    "num_docs": len(sampled),
    "num_qas": total_qas,
    "documents": sampled,
}

output_path = DATA_DIR / "korquad_v2_subset.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n저장 완료: {output_path}")
print(f"파일 크기: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# 5. 간단한 통계 출력
text_lengths = [len(d["context_text"]) for d in sampled]
print(f"\n[저장된 데이터 요약]")
print(f"문서 수: {len(sampled)}")
print(f"QA 쌍 수: {total_qas}")
print(f"문서 텍스트 길이 - 평균: {sum(text_lengths)//len(text_lengths):,}자, 최소: {min(text_lengths):,}자, 최대: {max(text_lengths):,}자")
