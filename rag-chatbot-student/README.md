# RAG Chatbot - Starter

Local, offline RAG chatbot over a small set of fictional LumaBox support
policies - the same use case, models, and functions from your Week 5 Colab
activity (`all-MiniLM-L6-v2` for embeddings, `flan-t5-base` for generation),
now organized as a real, runnable project instead of a notebook.

Today's job is mostly **moving code you already wrote**, plus one genuinely
new idea: chunking raw text and persisting the vector index to disk with
Chroma, so it survives after you close the terminal - unlike the NumPy array
in Colab, which vanished every time the runtime reset.

Look for `TODO` comments (`git grep -n TODO`): `I1-I4` in `src/ingest.py`,
`R1-R6` in `src/rag.py`, `T1` in `tests/test_retrieval.py`. That's everything
you need to fill in. `src/main.py`, `app.py`, and the rest are already done.

## Before class

You need VS Code (with the Python extension), Python 3.11-3.13, and git.
Then, from this folder:

```
python -m venv .venv
```

Windows (PowerShell):
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
python setup_check.py
```

`setup_check.py` downloads the two models (~1.1 GB, once) and confirms
everything loads. In class, `python setup_check.py --offline` proves nothing
needs downloading.

## Run (once the TODOs are filled in)

Always run from the project folder, as a module (`python -m ...`). VS Code's
Run button runs the file as a loose script and fails with
`No module named 'src'`.

Build the index:
```
python -m src.ingest
```

Chat via CLI - this is today's required deliverable:
```
python -m src.main
```

Run the verification test:
```
pytest
```

Stretch goal - chat via a minimal web UI:
```
streamlit run app.py
```

## Project layout

```
rag-chatbot/
├── .devcontainer/devcontainer.json   # Codespaces fallback if your install breaks
├── data/policies.txt                 # raw policy text - you decide how to chunk it
├── src/ingest.py                     # TODO I1-I4: chunk + embed + persist to Chroma
├── src/rag.py                        # TODO R1-R6: retrieve / prompt / generate
├── src/main.py                       # CLI - already done (required deliverable)
├── app.py                            # Streamlit - already done (stretch goal)
├── tests/test_retrieval.py           # TODO T1: one assertion
├── homework/                         # Homework.md + model_prices.xlsx
├── setup_check.py                    # pre-class environment check
├── pytest.ini                        # lets tests import from src/
├── requirements.txt
└── .gitignore
```

## Homework

See `homework/Homework.md` in this folder. Submit by zipping this whole
folder, including the `.git` folder, and uploading it to [LMS link].
