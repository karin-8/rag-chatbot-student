# Homework 1 — Run It Like a Product

In class you built a bot with a knowledge base, a versioned prompt, a
policy, a golden dataset, a quality gate and tracing. This homework uses
all of them the way a team does after launch: add a second knowledge
source without breaking anything, then improve the bot from what real
users asked.

**Commit after each exercise.** Your git history should show separate
points of progress, not one giant commit at the end - that history is part
of what gets submitted. (Haven't run `git init` yet? See the guide's
Appendix B.)

Submit by zipping this whole project folder, **including the `.git`
folder**, and uploading it to [LMS link] by [due date].

---

## Exercise 1 — A second knowledge source: prices (required)

`homework/model_prices.xlsx` holds LumaBox's product prices (`model`,
`price`). Customers ask about prices constantly; today the bot refuses
or says "I do not know".

**Task:** answer price questions from the table, without breaking any
policy answer and without ever inventing a price.

**How to approach it:**
1. **Knowledge.** In `src/ingest.py`, read the sheet with
   `pandas.read_excel` and turn each row into a sentence, e.g.
   `f"The {model} costs {price} THB."` - embeddings work on sentences, not
   table cells. The `model` column is messy on purpose (serial code and
   name run together, e.g. `"LMX-201 LumaBox X100"`); keep it as is. Add
   the sentences to the **same** collection as the policy chunks, with a
   `metadatas` field like `{"source": "price"}`.
2. **Policy.** Update `POLICY.md`: what may the bot say about prices, and
   what must it do for a product that isn't in the table? Name how each
   rule is enforced and tested.
3. **Golden dataset.** Add at least three `"price"` rows (including one
   product that isn't a LumaBox), each with variants. Then look at the
   existing `out_of_scope_tricky` row: "How much is the LumaBox X999?" now
   retrieves a *real* price row. Run it and see what the bot says.
4. **Gate.** Run `pytest` and `python -m src.gate`. With more chunks
   competing, check every existing category still scores at least its
   baseline. Only then run `python -m src.gate --update-baseline`, and say
   in the commit message why the new baseline is acceptable.

**Deliverable** (in `homework/exercise1_notes.md`): the gate output before
and after, your policy changes, and what the bot now says for the X999
question - honestly, even if it's wrong. If it's wrong, keep the row with a
`known_issue` explaining why rather than deleting it.

**Won't count:** two collections with your own "does this look like a price
question?" routing; deleting or rewording golden rows until they pass.

---

## Exercise 2 — The production loop (required)

`homework/traffic.txt` is a day of (simulated) real customer messages:
paraphrases, typos, questions your documents don't cover, frustrated users.

**Task:** find out how the bot really performs, and fix what matters most.

1. **Collect.** Replay the traffic and review the traces:
   ```
   python -m src.main --replay homework/traffic.txt
   python -m src.review --sample 5
   ```
2. **Analyze.** In `homework/monitoring_report.md`, make a table of every
   flagged turn and every sampled turn: trace id, question, the bot's
   answer, and a **failure type**:
   - *retrieval miss* - the right chunk wasn't retrieved;
   - *generation error* - the right chunk was retrieved, the answer is wrong
     or missing;
   - *knowledge gap* - no document has the answer;
   - *wrongly refused* / *should have refused* - the policy decision was
     wrong;
   - *not a failure* - the bot did the right thing.

   Then summarize: how many of each type, and which one hurts customers
   most.
3. **Fix.** Fix at least **two failures of different types**. For every
   fix: first add the failing question to `tests/golden.json`, watch it
   fail, then fix, then run `pytest` and the gate. A knowledge gap is fixed
   by the content owner, not by the model, so for one of those, write the
   missing policy text into `data/policies.txt` as if you were LumaBox's
   support team, and re-run `python -m src.ingest`.
4. **Sentiment.** Look at the turns flagged with negative sentiment. What
   should the bot do for a frustrated customer? Add it to `POLICY.md` as a
   new rule, with how you would enforce and test it. (You don't have to
   implement it.)

**Deliverable:** `homework/monitoring_report.md` (the table, the summary,
what you fixed and the gate result after each fix), plus the updated
`POLICY.md`, `tests/golden.json` and baseline, each fix in its own commit.

---

## Challenge — Exact model names, and CI for real (optional)

1. **Hybrid search.** Ask the price of each LumaBox model and serial code
   (e.g. "How much does model LMX-201 cost?"). Some come back with a
   *sibling* product's price: to an embedding model, "X200" and "X250" mean
   nearly the same thing, and serial codes mean nothing at all. Add golden
   rows for all of them first, marking failures with `known_issue`. Write a
   3-5 sentence plan for why this happens, then add a lexical (exact or
   keyword) match alongside vector search (`pip install rank_bm25` if you
   want real BM25). When a fix works, pytest fails those rows as
   unexpectedly passing until you remove `known_issue`.
2. **CI.** Push your project to a GitHub repository and open the
   **Actions** tab. Make the `test` job pass (hint: CI has no baseline
   unless you committed one). Screenshot the green run.
