"""The agent: retrieval, a versioned prompt, a policy check, and generation.

Module 1.2 builds the retrieval half, Module 2.1 the generation half with a
versioned prompt, and Module 2.2 the policy check. rag_answer() at the
bottom wires them together and records what the monitoring in Module 5
needs.
"""
import time
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "chroma_db"
PROMPTS_DIR = ROOT / "prompts"
COLLECTION_NAME = "policies"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-base"

# Which prompt in prompts/ is live. Changing it is a release: run the
# quality gate first (Module 3.2) and record the change in prompts/CHANGELOG.md.
PROMPT_VERSION = "v1"

# What the bot says when its policy says not to answer (POLICY.md, P3-P4).
REFUSAL_MESSAGE = (
    "I don't have information about that. "
    "For anything else, please contact LumaBox technical support."
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------- Module 1.2: retrieval
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
_client = chromadb.PersistentClient(path=str(DB_PATH))
_collection = _client.get_collection(COLLECTION_NAME)


def retrieve(query, k=2):
    """Return the top-k chunks as dicts with "id", "text" and "similarity"."""
    query_vector = embedding_model.encode(query, normalize_embeddings=True).tolist()
    results = _collection.query(query_embeddings=[query_vector], n_results=k)
    hits = []
    for doc_id, text, distance in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        hits.append({"id": doc_id, "text": text, "similarity": 1 - distance})
    return hits


# ------------------------------------------------- Module 2.1: generation
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
generator = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME).to(device)
generator.eval()


def load_prompt(version):
    """Return the prompt template stored in prompts/answer_<version>.txt."""
    return (PROMPTS_DIR / f"answer_{version}.txt").read_text(encoding="utf-8").strip()


def build_prompt(question, context, version=None):
    template = load_prompt(version or PROMPT_VERSION)
    return template.format(context=context, question=question)


def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
    if inputs["input_ids"].shape[1] > 512:
        raise ValueError("Prompt exceeds the 512 token budget. Reduce k or passage length.")
    with torch.inference_mode():
        output = generator.generate(**inputs, max_new_tokens=64, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ------------------------------------------------- Module 2.2: policy
# Chosen by comparing top similarities of in-scope and out-of-scope
# questions (Module 1.2). Re-check it whenever the documents change.
SIMILARITY_THRESHOLD = 0.30


def should_refuse(hits):
    """POLICY.md P1: refuse when nothing retrieved is relevant enough."""
    return not hits or hits[0]["similarity"] < SIMILARITY_THRESHOLD


# ------------------------------------------------- the agent
def kb_version():
    """Which version of the knowledge base is being searched (set by ingest.py)."""
    return (_collection.metadata or {}).get("kb_version", "unknown")


def rag_answer(question, k=2, search_query=None, enforce_policy=True, prompt_version=None):
    """Answer a question and return everything a trace needs.

    enforce_policy=False skips the policy check (Module 2.2 compares both).
    prompt_version overrides PROMPT_VERSION (the quality gate uses it).
    """
    version = prompt_version or PROMPT_VERSION
    started = time.perf_counter()
    hits = retrieve(search_query or question, k)
    retrieved = time.perf_counter()

    result = {
        "question": question,
        "hits": hits,
        "refused": False,
        "context": "",
        "prompt": None,
        "prompt_version": version,
        "kb_version": kb_version(),
        "models": {"embedding": EMBEDDING_MODEL_NAME, "generator": GENERATOR_MODEL_NAME},
    }
    if enforce_policy and should_refuse(hits):
        # Deterministic: the generator is never called.
        result.update(answer=REFUSAL_MESSAGE, refused=True)
    else:
        context = "\n".join(h["text"] for h in hits)
        prompt = build_prompt(question, context, version)
        result.update(context=context, prompt=prompt, answer=generate_answer(prompt))

    finished = time.perf_counter()
    result["timings_ms"] = {
        "retrieve": round((retrieved - started) * 1000),
        "generate": round((finished - retrieved) * 1000),
    }
    return result
