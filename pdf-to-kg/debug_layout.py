"""
PDF 텍스트 블록의 좌표를 확인하는 디버그 스크립트.

get_text("blocks")는 각 텍스트 블록을 (x0, y0, x1, y1, text, block_no, type) 형태로 반환한다.
이걸 통해 텍스트가 어떤 순서로 추출되는지, 박스 안 텍스트가 왜 섞이는지 확인할 수 있다.
"""

import fitz

PDF_PATH = "../data/개인정보_처리_통합_안내서(2025.7.).pdf"
doc = fitz.open(PDF_PATH)

# p.137 확인 (법령 박스가 있는 페이지)
page = doc[136]  # 0-indexed
blocks = page.get_text("blocks")

print(f"p.137 텍스트 블록 ({len(blocks)}개):\n")
for i, block in enumerate(blocks):
    x0, y0, x1, y1, text, block_no, block_type = block
    preview = text[:60].replace("\n", " ").strip()
    safe = preview.encode("cp949", errors="ignore").decode("cp949")
    print(f"  [{i}] y={y0:.0f}~{y1:.0f}, x={x0:.0f}~{x1:.0f} | {safe}")

doc.close()
