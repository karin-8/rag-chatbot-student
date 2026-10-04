"""Build the local Chroma index from data/policies.txt.

Run this once (and again any time the documents change):
    python -m src.ingest
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "policies.txt"
DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "policies"

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def chunk_documents():
    """Split the raw policy text into chunks.

    Real documents rarely arrive pre-chunked. Here we split on blank lines,
    so each policy's heading + body becomes one chunk, and drop any chunk
    that's just a heading with no body (e.g. the file's title line).
    """
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
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    embeddings = embedding_model.encode(chunks, normalize_embeddings=True).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, embeddings=embeddings)
    print(f"Indexed {len(chunks)} chunks into {DB_PATH}")


if __name__ == "__main__":
    build_index()
