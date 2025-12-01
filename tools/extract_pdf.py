from pathlib import Path
from PyPDF2 import PdfReader

pdf_path = Path("Tugas Besar Deep Learning.pdf")
if not pdf_path.exists():
    pdf_path = Path("d:/ITERA/semester7/DeepLearningTubes/Tugas Besar Deep Learning.pdf")

if not pdf_path.exists():
    print("PDF not found at expected locations.")
    raise SystemExit(1)

reader = PdfReader(str(pdf_path))
text = []
for p in reader.pages:
    text.append(p.extract_text() or "")

full = "\n\n".join(text)
# Print first 3000 chars for safety
print(full[:3000])
# Save full extracted text to file for later review
out = Path("tools/pdf_extracted.txt")
out.write_text(full, encoding="utf-8")
print(f"\n\n[Saved full extracted text to {out}]")
