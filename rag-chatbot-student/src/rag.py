"""The agent: retrieval, a versioned prompt, a policy check, and generation.

Module 1.2 builds the retrieval half (TODO R1-R2), Module 2.1 the
generation half with a versioned prompt (TODO R3-R4), and Module 2.2 the
policy check (TODO R5-R6). rag_answer() at the bottom wires them together
and records what the monitoring in Module 5 needs. It's already written.
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
# TODO R1: load the embedding model (EMBEDDING_MODEL_NAME, on the CPU), then
# connect to the collection ingest.py built: same path, same name, but get
# it rather than create it. If it says the collection doesn't exist, run
# `python -m src.ingest` first.
embedding_model = ...
_client = ...
_collection = ...


def retrieve(query, k=2):
    """Return the top-k chunks for a query.

    TODO R2: embed the query the same way the chunks were embedded, then ask
    _collection for the k nearest chunks instead of doing the NumPy
    dot-product search you used in Colab.

    Return the same shape your Colab retrieve() returned: a list of dicts,
    each with "id", "text" and "similarity" keys. Note that Chroma reports
    *distances*, not similarities.
    """
    raise NotImplementedError("TODO R2: query Chroma and shape the results")


# ------------------------------------------------- Module 2.1: generation
# TODO R3: load the GENERATOR_MODEL_NAME tokenizer and model, as in your
# Week 5 notebook. Move the model to `device` and put it in eval mode.
tokenizer = ...
generator = ...


def load_prompt(version):
    """Return the prompt template stored in prompts/answer_<version>.txt."""
    return (PROMPTS_DIR / f"answer_{version}.txt").read_text(encoding="utf-8").strip()


def build_prompt(question, context, version=None):
    """TODO R4: fill the template with the context and the question.

    The template comes from load_prompt(version or PROMPT_VERSION) and has
    {context} and {question} placeholders.
    """
    raise NotImplementedError("TODO R4: fill the prompt template")


def generate_answer(prompt):
    """TODO R4: paste this straight from your Week 5 notebook - no changes."""
    raise NotImplementedError("TODO R4: paste generate_answer from Colab")


# ------------------------------------------------- Module 2.2: policy
# TODO R5: choose the threshold from the similarity scores you printed in
# Module 1.2: below every in-scope question, above the clearly
# out-of-scope ones.
SIMILARITY_THRESHOLD = ...


def should_refuse(hits):
    """POLICY.md P1: refuse when nothing retrieved is relevant enough.

    TODO R6: return True when there are no hits, or when the top hit's
    similarity is below SIMILARITY_THRESHOLD.
    """
    return False  # placeholder: never refuses, i.e. no policy yet


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
