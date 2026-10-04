"""Retrieval-augmented generation over the local Chroma index.

Paste your Week 5 Colab functions in below - the only real change is that
`retrieve` now queries Chroma instead of a NumPy matrix. build_prompt,
generate_answer, and rag_answer are unchanged; you're moving them, not
rewriting them.
"""
from pathlib import Path

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "policies"

device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO R1: load the same three objects as your Week 5 notebook:
#   - the all-MiniLM-L6-v2 embedding model (CPU)
#   - the google/flan-t5-base tokenizer
#   - the google/flan-t5-base generator, moved to `device` and put in eval mode
embedding_model = ...
tokenizer = ...
generator = ...

# TODO R2: connect to the collection that ingest.py already built (same
# path, same name - but get it, don't create it). If it says the collection
# doesn't exist, run `python -m src.ingest` first.
_client = ...
_collection = ...


def retrieve(query, k=2):
    """Return the top-k chunks for a query.

    TODO R3: embed the query the same way the chunks were embedded, then
    ask _collection for the k nearest chunks instead of doing the NumPy
    dot-product search you used in Colab.

    Return the same shape your Colab retrieve() returned: a list of dicts,
    each with "id", "text" and "similarity" keys. Note that Chroma reports
    *distances*, not similarities.
    """
    raise NotImplementedError("TODO R3: query Chroma and shape the results")


def build_prompt(question, context=None):
    """TODO R4: paste this straight from your Week 5 notebook - no changes.

    Keep the plain prompt. Small models are very sensitive to prompt wording,
    and "more instructions" isn't always better - see the guide, Module 5.
    """
    raise NotImplementedError("TODO R4: paste build_prompt from Colab")


def generate_answer(prompt):
    """TODO R5: paste this straight from your Week 5 notebook - no changes."""
    raise NotImplementedError("TODO R5: paste generate_answer from Colab")


def rag_answer(question, k=2, search_query=None):
    """TODO R6: paste this straight from your Week 5 notebook - no changes."""
    raise NotImplementedError("TODO R6: paste rag_answer from Colab")
