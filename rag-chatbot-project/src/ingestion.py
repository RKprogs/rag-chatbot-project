import os
import json
import re
from typing import List
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pypdf."""
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            pages.append(t)
    return "\n".join(pages)

def clean_text(text: str) -> str:
    """Basic cleaning: collapse whitespace, remove repeated newlines."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chunk text into overlapping character chunks (safe approximation)."""
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(end - overlap, end)
    return chunks

def ingest_pdf(pdf_path: str, out_json: str, chunk_size: int = 1000, overlap: int = 200):
    """Extract, clean and chunk PDF text, then save as JSON list of chunks."""
    text = extract_text_from_pdf(pdf_path)
    text = clean_text(text)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Ingested {len(chunks)} chunks from {pdf_path} -> {out_json}")
    return chunks

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', required=True, help='Path to input PDF')
    parser.add_argument('--out', default='data/processed/chunks.json', help='Output JSON path')
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--overlap', type=int, default=200)
    args = parser.parse_args()
    ingest_pdf(args.pdf, args.out, args.chunk_size, args.overlap)
