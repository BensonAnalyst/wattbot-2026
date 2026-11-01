"""Documents: read metadata, download the PDFs, extract text page by page, and cut it into chunks.

The competition data may not be redistributed, so nothing here is committed: the PDFs are downloaded at run time
from the URLs in metadata.csv.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd


def load_metadata(path) -> pd.DataFrame:
    """Read metadata.csv, trying UTF-8 first, then Windows-1252 (the 2025 file used it)."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            meta = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    meta.columns = [c.strip() for c in meta.columns]
    meta["id"] = meta["id"].str.strip()
    meta["url"] = meta["url"].str.strip()
    return meta


def first_author(citation: str) -> str:
    """'Jesse Dodge, Taylor Prewitt, ...' -> 'dodge'; 'Amazon Staff. (2023)' -> 'amazon'."""
    head = re.split(r"[,(.&]", str(citation), maxsplit=1)[0].strip()
    words = re.findall(r"[A-Za-z\-']+", head)
    return words[-1].lower() if words else ""


def pdf_url(url: str) -> str:
    """arXiv abstract pages (/abs/) link to an HTML page, not the PDF: point them at /pdf/ instead."""
    return re.sub(r"arxiv\.org/abs/", "arxiv.org/pdf/", str(url).strip())


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_cached_pdfs(meta: pd.DataFrame, search_dirs) -> dict:
    """{doc_id: path} for documents already saved in any of search_dirs (e.g. a private Kaggle Dataset).

    Names are matched ignoring case and punctuation, because uploads get renamed (agl2025.pdf -> agl_2025.pdf)."""
    by_key = {}
    for d in search_dirs:
        for p in Path(d).glob("*.pdf"):
            if p.stat().st_size > 10_000:
                by_key.setdefault(_key(p.stem), p)
    return {row.id: by_key[_key(row.id)] for row in meta.itertuples() if _key(row.id) in by_key}


def download_pdfs(meta: pd.DataFrame, out_dir, cache_dirs=(), delay: float = 3.0, retries: int = 2) -> dict:
    """Fetch each document once, politely, and keep it.

    The competition rules ask entrants to fetch each document once, cache it, and not crawl arXiv or publishers.
    So: documents already in cache_dirs or out_dir are reused; the rest are fetched one at a time with a pause
    between requests and a descriptive User-Agent. Returns {doc_id: path or None if it failed}."""
    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = find_cached_pdfs(meta, list(cache_dirs) + [out_dir])
    agents = ["WattBot2026-entrant/1.0 (one-time fetch of the competition corpus)",
              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"]
    todo = [r for r in meta.itertuples() if r.id not in result]
    for i, row in enumerate(todo):
        result[row.id] = None
        for attempt in range(retries):
            try:
                # Some sites refuse non-browser clients, so the second attempt identifies as a browser.
                resp = requests.get(pdf_url(row.url), timeout=180, headers={"User-Agent": agents[min(attempt, 1)]})
                if resp.ok and resp.content[:4] == b"%PDF":
                    path = out_dir / f"{row.id}.pdf"
                    path.write_bytes(resp.content)
                    result[row.id] = path
                    break
            except requests.RequestException:
                pass
            time.sleep(10 * (attempt + 1))
        if i < len(todo) - 1:
            time.sleep(delay)
    return result


def extract_pages(pdf_path) -> list:
    """Text of each page (pdfplumber, MIT licence). Tables come out as text rows in reading order."""
    import pdfplumber

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            pages.append({"page": i, "text": _clean(text)})
    return pages


def _clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_pages(doc_id: str, title: str, pages: list, size: int = 220, overlap: int = 60) -> list:
    """Word windows within a page, so every chunk keeps its page number for the citation.

    Each chunk's search text starts with the document ID and title, so a question naming the paper
    (\"JetMoE\", \"Power Hungry Processing\") matches all of that paper's chunks."""
    chunks = []
    step = size - overlap
    for p in pages:
        words = p["text"].split()
        if not words:
            continue
        for start in range(0, max(len(words) - overlap, 1), step):
            body = " ".join(words[start:start + size])
            chunks.append({
                "chunk_id": f"{doc_id}:p{p['page']}:{start // step}",
                "doc_id": doc_id,
                "page": p["page"],
                "text": body,
                "search_text": f"{doc_id} {title}. {body}",
            })
    return chunks


def build_chunks(meta: pd.DataFrame, pdf_paths: dict, cache_path=None, out_path=None) -> list:
    """Parse PDFs into chunks, reusing a cache: only documents missing from the cache are parsed.

    cache_path may be read-only (a Kaggle Dataset); the merged result is written to out_path (default
    cache_path). Documents added later, such as a PDF fetched in a second run, are parsed without
    re-parsing the other documents."""
    chunks = json.loads(Path(cache_path).read_text()) if cache_path and Path(cache_path).exists() else []
    have = {c["doc_id"] for c in chunks}
    added = 0
    for row in meta.itertuples():
        path = pdf_paths.get(row.id)
        if path and row.id not in have:
            chunks += chunk_pages(row.id, row.title, extract_pages(path))
            added += 1
    out = out_path or cache_path
    if out and (added or not Path(out).exists()):
        Path(out).write_text(json.dumps(chunks))
    return chunks
