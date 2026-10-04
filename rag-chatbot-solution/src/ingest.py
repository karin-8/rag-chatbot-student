"""Build the local Chroma index from data/policies.txt.

Unlike a clean JSON array, this is one raw text file - real documents
usually arrive this way. You have to decide how to split it into
retrievable pieces before you can embed anything.

Run this once (and again any time the documents change):
    python -m src.ingest
"""
import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "policies.txt"
DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "policies"

# A short fingerprint of the source text. It changes whenever the documents
# change, and every answer logs it, so a wrong answer can be traced back to
# the exact knowledge it came from.
KB_VERSION = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()[:12]

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def chunk_documents():
    text = DATA_PATH.read_text(encoding="utf-8")
    raw_chunks = [c.strip() for c in text.split("\n\n")]
    return [c for c in raw_chunks if "\n" in c]


def build_index():
    chunks = chunk_documents()

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine", "kb_version": KB_VERSION}
    )

    embeddings = embedding_model.encode(chunks, normalize_embeddings=True).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, embeddings=embeddings)
    print(f"Indexed {len(chunks)} chunks into {DB_PATH} (kb_version {KB_VERSION})")


if __name__ == "__main__":
    build_index()
