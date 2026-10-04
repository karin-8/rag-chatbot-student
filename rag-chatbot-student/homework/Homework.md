# Homework 1 — Hardening Your RAG Chatbot

You already have a working CLI backed by a real local database. This
homework doesn't add new features so much as it exposes ways a small RAG
system fails in practice, and asks you to fix them - the same way you'd
harden any system after the first version works.

**Commit after each exercise.** Your git history should show three (or
four, with the challenge) separate points of progress, not one giant commit
at the end - that history is part of what gets submitted.

Submit by zipping this whole project folder, **including the `.git`
folder**, and uploading it to [LMS link].

---

## Exercise 1 — Edge case handling (required)

Your prompt already tells the model: *"If the information is unavailable,
say 'I do not know'."* That's an instruction, not a guarantee -
`flan-t5-base` is small enough that it won't always obey it, and may
confidently invent an answer to a question your documents don't cover.

**Task:** make the refusal deterministic instead of hoping the model
follows the instruction.

**How to approach it:**
1. In a scratch script or the Python REPL, call `retrieve()` with a
   question your documents clearly answer (e.g. "How long does delivery
   take?") and one they clearly don't (e.g. "Does LumaBox ship to the
   moon?"). Print the similarity scores for both.
2. Look at the gap between the two. Pick a threshold that sits between
   them.
3. In `rag_answer` (or wherever makes sense), check the top hit's
   similarity *before* calling the generator. Below the threshold, skip
   generation entirely and return a fixed phrase, e.g. `"I don't have
   information about that."`

**Deliverable:** in your README (or a short `homework/exercise1_notes.md`),
show at least 2 in-scope and 2 out-of-scope example questions and the
bot's response to each, after your fix.

**What "done" looks like:** the refusal happens every time for an
out-of-scope question, not just most of the time.

---

## Exercise 2 — Multi-source RAG (required)

Real RAG systems are rarely single-sourced. `homework/model_prices.xlsx`
contains a small table of LumaBox model prices (`model`, `price_thb`).

**Task:** extend the system so it can answer questions from *either*
source - policies and prices - without mixing them up or guessing.

**How to approach it:**
1. Read `model_prices.xlsx` with `pandas.read_excel`. Note the `model`
   column is messy on purpose - it's the serial code and the product name
   run together (e.g. `"LMX-201 LumaBox X100"`), the way a real inventory
   export often is. You don't need to split it apart for this exercise.
2. Turn each row into a short sentence before embedding it, e.g.
   `f"The {model} costs {price} THB."` - embeddings work on sentences
   describing a fact, not on bare table cells.
3. Add these sentences into the **same** Chroma collection as your policy
   chunks (not a second collection), tagging each item with a metadata
   field like `source: "policy"` or `source: "price"` so you can tell them
   apart later if needed.
4. Re-run ingestion and confirm retrieval now returns relevant results for
   both kinds of question.

**Ensure they don't hallucinate:** reuse your Exercise 1 fix, so a question
neither source covers (e.g. the weather) is still refused.

Then test its limit: ask the price of a product that *isn't* in the table
(e.g. "How much is the LumaBox X999?"). Print the top similarity. A
neighbouring product's price row can score well above your threshold,
because to the embedding model "X999" means almost the same as "X200" - so
the threshold alone may let a wrong price through. You don't have to solve
this fully here (the challenge exercise is about exactly this), but your
notes must report what your bot does for this question, honestly.

**Deliverable:** in your notes, show at least one price question, one
policy question, one question neither source covers, and the unknown-product
price question above, with the bot's response to each.

---

## Challenge exercise — When embeddings can't tell models apart (optional)

Ask your bot "how much is the LumaBox X250?" a few times. You may notice it
sometimes answers with the X200 or X300 price instead. This isn't a bug in
your code - it's a limitation of embeddings themselves: `all-MiniLM-L6-v2`
represents *meaning*, and "LumaBox X200", "LumaBox X250", and "LumaBox X300"
mean nearly the same thing to it, even though they're different products
with different prices. Try "how much does model LMX-201 cost?" too (LMX-201
is the X100's serial code) - the serial codes are sequential and unrelated
to which model is which, so they carry even less semantic meaning than a
model name. This version of the problem is usually worse, not better.

**Task:** plan first, then implement.

1. **Plan (write 3-5 sentences):** why does this happen, and what kind of
   fix would actually solve it (not just "use a better embedding model" -
   assume you're stuck with this one)?
2. **Implement:** there is a completely different retrieval method from
   vector search that's good at exactly this: keyword/lexical search, which
   matches on exact tokens rather than meaning. Add a lexical check for
   product-model-like tokens in the query (a simple exact/substring match
   against the known model names is enough; `pip install rank_bm25` if you
   want to build something closer to real hybrid search). When a query
   contains an exact model name, prioritize the price entry for *that*
   model rather than relying on embedding similarity alone.

**Deliverable:** your written plan, the code, and a demonstration that
asking about each of the five LumaBox models now reliably returns *that
model's* price.
