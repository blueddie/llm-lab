"""
PDF에서 특정 페이지 범위의 텍스트를 추출하는 스크립트.

해결하는 문제:
  PDF는 비정형 데이터라서 LLM에 바로 넣을 수 없다.
  먼저 텍스트로 변환해야 청킹 → 엔티티 추출이 가능하다.

핵심 개념:
  - PyMuPDF(fitz)는 PDF를 페이지 단위로 접근한다
  - 페이지 번호는 0-indexed (PDF의 p.1 = doc[0])
  - get_text("blocks")는 텍스트를 블록 단위로 좌표와 함께 반환한다
  - 기본 get_text()는 블록 번호 순으로 추출하므로, 박스/표 안 텍스트의
    순서가 뒤바뀔 수 있다. y좌표(세로 위치)로 정렬하면 해결된다.
"""

import fitz

# --- 설정 ---
PDF_PATH = "../data/개인정보_처리_통합_안내서(2025.7.).pdf"
START_PAGE = 136  # V장 실제 내용 시작 (p.135는 챕터 표지)
END_PAGE = 159
OUTPUT_PATH = "extracted_text.txt"

# --- PDF 열기 ---
doc = fitz.open(PDF_PATH)
print(f"전체 페이지 수: {doc.page_count}")

# --- 텍스트 추출 ---
# PDF 페이지 번호는 1부터 시작하지만, PyMuPDF는 0부터 시작
# p.133 → doc[132], p.159 → doc[158]
extracted_pages = []

for page_num in range(START_PAGE - 1, END_PAGE):  # 135 ~ 158
    page = doc[page_num]

    # get_text("blocks")로 블록별 좌표를 얻어서 y좌표(세로 위치) 기준 정렬
    # 각 블록: (x0, y0, x1, y1, text, block_no, type)
    # type=0이 텍스트 블록, type=1이 이미지 블록
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0]  # 텍스트만
    text_blocks.sort(key=lambda b: (b[1], b[0]))  # y좌표 → x좌표 순 정렬
    text = "\n".join(b[4].strip() for b in text_blocks)

    extracted_pages.append({
        "page": page_num + 1,  # 사람이 읽는 페이지 번호로 저장
        "text": text,
    })
    # 각 페이지별 추출 결과 미리보기
    # Windows 콘솔(cp949)에서 출력 불가능한 유니코드 문자는 무시
    preview = text[:80].replace("\n", " ")
    safe_preview = preview.encode("cp949", errors="ignore").decode("cp949")
    print(f"  p.{page_num + 1}: {len(text)}자 | {safe_preview}...")

doc.close()

# --- 결과 저장 ---
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for p in extracted_pages:
        f.write(f"\n{'='*60}\n")
        f.write(f"[페이지 {p['page']}]\n")
        f.write(f"{'='*60}\n")
        f.write(p["text"])

total_chars = sum(len(p["text"]) for p in extracted_pages)
print(f"\n총 {len(extracted_pages)}페이지, {total_chars:,}자 추출 완료")
print(f"저장: {OUTPUT_PATH}")
