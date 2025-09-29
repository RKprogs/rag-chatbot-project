import faiss
import json
from typing import List, Dict

def load_faiss_index(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return index, meta

def search(index, meta, query_embedding, top_k: int = 5):
    import numpy as np
    # ensure shape (1, dim)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, top_k)
    results = []
    for idx in I[0]:
        if idx < 0:
            continue
        results.append(meta[int(idx)])
    return results
