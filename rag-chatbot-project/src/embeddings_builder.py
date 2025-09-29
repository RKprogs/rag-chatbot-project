import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

MODEL_NAME = 'all-MiniLM-L6-v2'

def build_embeddings(chunks_json: str, index_path: str = 'data/index.faiss', meta_path: str = 'data/index_meta.json', model_name: str = MODEL_NAME):
    os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
    with open(chunks_json, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    # normalize for cosine similarity using inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    meta = [{'id': i, 'text': chunks[i]} for i in range(len(chunks))]
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Built FAISS index at {index_path} with {len(chunks)} vectors (dim={dim})")
    return index_path, meta_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', default='data/processed/chunks.json')
    parser.add_argument('--index', default='data/index.faiss')
    parser.add_argument('--meta', default='data/index_meta.json')
    parser.add_argument('--model', default=MODEL_NAME)
    args = parser.parse_args()
    build_embeddings(args.chunks, args.index, args.meta, args.model)
