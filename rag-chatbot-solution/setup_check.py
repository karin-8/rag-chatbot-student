"""Pre-class setup check. Run it from the project folder with your venv active.

At home, before class (downloads ~1.1 GB of model weights, once):
    python setup_check.py

In class (proves everything is already on disk - nothing gets downloaded):
    python setup_check.py --offline
"""
import os
import shutil
import sys

OFFLINE = "--offline" in sys.argv
if OFFLINE:
    # Must be set before any Hugging Face library is imported.
    os.environ["HF_HUB_OFFLINE"] = "1"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "google/flan-t5-base"


def ok(msg):
    print(f"[ok]   {msg}")


def warn(msg):
    print(f"[warn] {msg}")


def fail(msg, fix):
    print(f"[FAIL] {msg}")
    print(f"       fix: {fix}")
    sys.exit(1)


# 1. Python version
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10):
    fail(f"Python {major}.{minor} is too old", "install Python 3.11, 3.12 or 3.13 from python.org")
elif (major, minor) > (3, 13):
    warn(f"Python {major}.{minor} hasn't been tested with this project; 3.11-3.13 are known to work")
else:
    ok(f"Python {major}.{minor}")

# 2. Virtual environment (Codespaces installs globally, so it's exempt)
if sys.prefix != sys.base_prefix:
    ok(f"running inside a virtual environment ({sys.prefix})")
elif os.environ.get("CODESPACES"):
    ok("running in Codespaces")
else:
    warn("not inside a virtual environment - activate .venv first (see the guide's 'Before class' module)")

# 3. git
if shutil.which("git") is None:
    fail("git was not found", "install it from https://git-scm.com, then close and reopen VS Code")
ok("git is installed")

# 4. Python packages
try:
    import chromadb
    import sentence_transformers
    import torch
    import transformers
except ImportError as e:
    fail(f"missing package: {e.name}", "activate .venv, then run: pip install -r requirements.txt")
ok(
    f"packages import (torch {torch.__version__}, transformers {transformers.__version__}, "
    f"sentence-transformers {sentence_transformers.__version__}, chromadb {chromadb.__version__})"
)

# 5. Models actually load (and, with --offline, load without the network)
hint = (
    "run `python setup_check.py` once WITHOUT --offline on a good connection to download the models"
    if OFFLINE
    else "check your internet connection (or proxy) and run this again"
)

try:
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    vector = embedding_model.encode("hello world")
except Exception as e:
    fail(f"could not load {EMBEDDING_MODEL}: {e}", hint)
ok(f"embedding model loads ({EMBEDDING_MODEL}, vector size {len(vector)})")

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    generator = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)
    inputs = tokenizer("Answer briefly. Question: What colour is the sky? Answer:", return_tensors="pt")
    with torch.inference_mode():
        output = generator.generate(**inputs, max_new_tokens=8)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
except Exception as e:
    fail(f"could not load {GENERATOR_MODEL}: {e}", hint)
ok(f"generator loads and answers ({GENERATOR_MODEL} says: {reply!r})")

print()
print("All checks passed." + (" Models are cached - no downloads needed in class." if OFFLINE else ""))
