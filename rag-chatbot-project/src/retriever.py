from sentence_transformers import SentenceTransformer
from .vector_store import load_faiss_index, search
import numpy as np

EMBED_MODEL = 'all-MiniLM-L6-v2'

class Retriever:
    def __init__(self, index_path: str = 'data/index.faiss', meta_path: str = 'data/index_meta.json', model_name: str = EMBED_MODEL):
        self.index, self.meta = load_faiss_index(index_path, meta_path)
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 5):
        emb = self.model.encode([query], convert_to_numpy=True)
        results = search(self.index, self.meta, emb, top_k=top_k)
        return results

if __name__ == '__main__':
    r = Retriever()
    res = r.retrieve('What are the barriers to electrification?', top_k=3)
    for r in res:
        print(r['text'][:300])
