import json
import re
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
import config as c
from ingest import corpus_manifest


class Retriever:
    def __init__(self):
        if not (c.INDEX / "manifest.json").exists():
            raise RuntimeError("Index missing. Run python ingest.py first.")
        meta = json.loads((c.INDEX / "manifest.json").read_text())
        if meta["embedding_model"] != c.EMBEDDING_MODEL:
            raise RuntimeError("Embedding model changed; rebuild the index.")
        if meta["corpus"] != corpus_manifest():
            raise RuntimeError("PDF corpus changed; rebuild the index.")
        if (meta["chunk_size"], meta["chunk_overlap"]) != (c.CHUNK_SIZE, c.CHUNK_OVERLAP):
            raise RuntimeError("Chunk settings changed; rebuild the index.")
        self.records = json.loads((c.INDEX / "chunks.json").read_text(encoding="utf-8"))
        self.vectors = np.load(c.INDEX / "vectors.npy", allow_pickle=False).astype("float32")
        if self.vectors.ndim != 2 or len(self.vectors) != len(self.records):
            raise ValueError("Invalid embedding cache")
        faiss.normalize_L2(self.vectors)
        self.embedder = OpenAIEmbeddings(model=c.EMBEDDING_MODEL)

    def search(self, question):
        aircraft = set(re.findall(r"VT-[A-Z0-9]+", question.upper()))
        query = np.asarray([self.embedder.embed_query(question)], dtype="float32")
        faiss.normalize_L2(query)
        groups = sorted(aircraft) if aircraft else [None]
        results = []
        for target in groups:
            ids = [i for i, r in enumerate(self.records) if target is None or r["aircraft"] == target]
            if not ids:
                continue
            subset = np.ascontiguousarray(self.vectors[ids])
            index = faiss.IndexFlatIP(subset.shape[1])
            index.add(subset)
            scores, hits = index.search(query, min(c.RETRIEVAL_K, len(ids)))
            for score, hit in zip(scores[0], hits[0]):
                results.append({**self.records[ids[int(hit)]], "similarity": float(score)})
        return results
