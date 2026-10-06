import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import config as c


def corpus_manifest():
    paths = sorted(c.DOCUMENTS.glob("*.pdf"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def build_index():
    manifest = corpus_manifest()
    if not manifest:
        raise ValueError("Place text-readable PDF reports in documents/ first.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c.CHUNK_SIZE, chunk_overlap=c.CHUNK_OVERLAP)
    records = []
    skipped = 0
    for name in manifest:
        reader = PdfReader(c.DOCUMENTS / name)
        aircraft = re.search(r"VT-[A-Z0-9]+", name.upper())
        status = "preliminary" if "preliminary" in name.lower() else (
            "final" if "final" in name.lower() else "unknown")
        for page, pdf_page in enumerate(reader.pages, start=1):
            text = (pdf_page.extract_text() or "").strip()
            if not text:
                skipped += 1
                continue
            for chunk in splitter.split_text(text):
                records.append(dict(id=len(records), text=chunk, source=name,
                                    page=page, aircraft=aircraft.group() if aircraft else "unknown",
                                    status=status))
    if not records:
        raise ValueError("No extracted text. Scanned reports require OCR first.")
    embedder = OpenAIEmbeddings(model=c.EMBEDDING_MODEL)
    vectors = []
    for start in range(0, len(records), 64):
        vectors.extend(embedder.embed_documents([r["text"] for r in records[start:start+64]]))
    c.INDEX.mkdir(parents=True, exist_ok=True)
    np.save(c.INDEX / "vectors.npy", np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    (c.INDEX / "chunks.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    metadata = dict(corpus=manifest, embedding_model=c.EMBEDDING_MODEL,
                    chunk_size=c.CHUNK_SIZE, chunk_overlap=c.CHUNK_OVERLAP,
                    chunks=len(records), skipped_empty_pages=skipped)
    (c.INDEX / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    argparse.ArgumentParser(description="Build the local PDF embedding cache").parse_args()
    print(json.dumps(build_index(), indent=2))
