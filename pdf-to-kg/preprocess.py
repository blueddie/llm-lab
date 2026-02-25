"""
추출된 텍스트에서 노이즈를 제거하는 전처리 스크립트.

제거 대상:
  1. 페이지 머리글: "Ⅴ. 특별한 보호  135" 같은 패턴
  2. 페이지 번호/문서명: "134  개인정보 처리 통합 안내서" 같은 패턴
  3. 특수 불릿 기호: ⬜, ⚫, ☞, ➡ → 의미 있는 마커로 치환
  4. 페이지 구분선: ====... [페이지 N] ====... 제거하고 연속 텍스트로 합침
"""

import re

INPUT_PATH = "extracted_text.txt"
OUTPUT_PATH = "cleaned_text.txt"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    raw = f.read()

# --- 1. 페이지 구분선 제거 ---
# "===...\n[페이지 N]\n===..." 패턴 제거
text = re.sub(r"=+\n\[페이지 \d+\]\n=+", "", raw)

# --- 2. 페이지 머리글 제거 ---
# "Ⅴ. 특별한 보호  135" (로마숫자 + 제목 + 페이지번호)
text = re.sub(r"Ⅴ\.\s*특별한 보호\s+\d+", "", text)

# --- 3. 페이지 하단 문서명 제거 ---
# "134  개인정보 처리 통합 안내서" (페이지번호 + 문서명)
text = re.sub(r"\d+\s+개인정보 처리 통합 안내서", "", text)

# --- 4. 특수 불릿 기호 정리 ---
# ⬜ → 빈 줄 없이 본문으로 (원래 문단 시작 마커)
text = text.replace("⬜", "")
# ⚫ → 불릿 포인트로
text = text.replace("⚫", "- ")
# ☞ → 화살표로
text = text.replace("☞", "-> ")
# ➡ → 화살표로
text = text.replace("➡", "-> ")
# ∙ → 불릿으로
text = text.replace("∙", "- ")

# --- 5. 연속 빈 줄 정리 ---
# 3줄 이상 빈 줄을 2줄로 축소
text = re.sub(r"\n{3,}", "\n\n", text)

# 앞뒤 공백 정리
text = text.strip()

# --- 저장 ---
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(text)

# --- 비교 출력 ---
raw_chars = len(raw)
clean_chars = len(text)
removed = raw_chars - clean_chars
print(f"전처리 완료:")
print(f"  원본: {raw_chars:,}자")
print(f"  정리: {clean_chars:,}자")
print(f"  제거: {removed:,}자 ({removed/raw_chars*100:.1f}%)")
print(f"저장: {OUTPUT_PATH}")
