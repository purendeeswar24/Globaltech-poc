import json,os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np


OUT_DIR = "output"
INDEX_PATH = os.path.join(OUT_DIR,"index.faiss")
METADATA_PATH = os.path.join(OUT_DIR,"metadata.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBEDDING_MODEL)


def search(question, k = 5):
    idx = faiss.read_index(INDEX_PATH)
    with open (METADATA_PATH, "r",encoding="utf-8") as f:
        docs = json.load(f)
    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")
    D,I = idx.search(q_emb, k)
    result = []
    for i in I[10]:
        if i < 0 or i >= len(docs):
            continue
        result.append( docs[i])
    return result


if __name__ == "__main__":
    q = input("Ask a Question: ")
    res = search(q,k = 5)
    for i in res:
        print("-------/n")
        print("page",r["page"])
        print("text",r["text"][:500]+("...." if len(r["text"]) > 500 else ""))
        print("image")
        if r.get("image"):
            print("image files:",r["image"])
    