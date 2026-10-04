# RAG Chatbot (Reference Solution)

Local, offline RAG chatbot over a small set of fictional LumaBox support
policies. Same models as the Week 5 Colab activity (`all-MiniLM-L6-v2` for
embeddings, `flan-t5-base` for generation) - the only change is that the
vector index now persists to disk via Chroma instead of living in a NumPy
array that vanished on runtime restart.

This is the completed reference implementation, for facilitator use and for
comparison against the student scaffold.

## Setup

```
python -m venv .venv
```

Windows:
```
.venv\Scripts\activate
```
Mac/Linux:
```
source .venv/bin/activate
```

Then:
```
pip install -r requirements.txt
```

## Run

Build the index (do this first, and again whenever `data/policies.txt` changes):
```
python -m src.ingest
```

First run downloads model weights (`all-MiniLM-L6-v2` + `flan-t5-base`,
roughly 1GB combined) - expect this step and the first `python -m src.main`
question to be slow (a minute or more). After that, weights are cached
locally and things speed up a lot.

Chat via CLI:
```
python -m src.main
```

Run the verification test:
```
pytest
```

Stretch - chat via a minimal web UI:
```
streamlit run app.py
```

## Project layout

```
rag-chatbot/
├── .devcontainer/devcontainer.json   # Codespaces fallback, deps preinstalled
├── data/policies.txt                 # raw policy text, chunked at ingest time
├── src/ingest.py                     # chunk + embed + persist to Chroma
├── src/rag.py                        # retrieve / prompt / generate
├── src/main.py                       # CLI (required deliverable)
├── app.py                            # Streamlit (stretch goal)
├── tests/test_retrieval.py           # verification test
├── requirements.txt
└── .gitignore
```
