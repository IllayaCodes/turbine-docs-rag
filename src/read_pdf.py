import sys
import json
import pymupdf
import pdfplumber
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/extracted")
MIN_CHARS = 100
MAX_SINGLE_LETTER_RATIO = 0.10


def single_letter_ratio(pages):
    words = []
    for p in pages:
        words.extend(p["text"].split())
    if not words:
        return 0.0
    singles = [w for w in words if len(w) == 1 and w.lower() not in ("a", "i")]
    return len(singles) / len(words)


def extract_with_pymupdf(pdf_path):
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": page.get_text(),
            })
    return pages


def extract_with_pdfplumber(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": page.extract_text() or "",
            })
    return pages


def extract_pages(pdf_path):
    pages = extract_with_pymupdf(pdf_path)
    ratio = single_letter_ratio(pages)
    extractor = "pymupdf"

    if ratio > MAX_SINGLE_LETTER_RATIO:
        fallback = extract_with_pdfplumber(pdf_path)
        fallback_ratio = single_letter_ratio(fallback)
        if fallback_ratio < ratio:
            pages, ratio, extractor = fallback, fallback_ratio, "pdfplumber"

    return pages, extractor, ratio


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}")
        return

    for pdf_path in pdf_files:
        try:
            pages, extractor, ratio = extract_pages(pdf_path)
        except Exception as error:
            print(f"\nFAILED to read {pdf_path.name}: {error}")
            continue

        total_chars = sum(len(p["text"]) for p in pages)
        low_pages = [p["page"] for p in pages if len(p["text"].strip()) < MIN_CHARS]

        print(f"\n===== {pdf_path.name} =====")
        print(f"Extractor used: {extractor} | Stray single letters: {ratio:.1%}")
        if ratio > MAX_SINGLE_LETTER_RATIO:
            print("WARNING: text quality is still low for this document")
        print(f"Pages: {len(pages)} | Total characters: {total_chars}")
        print(f"Pages with almost no text: {low_pages if low_pages else 'none'}")

        middle_page = pages[len(pages) // 2]
        print(f"--- preview of page {middle_page['page']} ---")
        print(middle_page["text"][:400])

        out_file = OUT_DIR / f"{pdf_path.stem}.json"
        out_file.write_text(
            json.dumps(pages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()