# pdfrag: hybrid-search question answering over technical manuals, with page citations

Reference solution for the **pdf-rag** capstone (mini lab 900.2). It answers questions about three short,
original technical manuals (a booster pump, a lighting controller and a field gateway). It parses pages by
layout, chunks without crossing pages or headings, searches with BM25 and dense vectors merged by reciprocal
rank fusion, reranks the top candidates, answers with a page citation on every sentence, and says
"Not in the documents." when retrieval is weak. A 50-question golden set and an eval script measure it.

Everything runs offline and free. The default model is a deterministic extractive fake, so tests repeat exactly.

## Quick start

```bash
uv sync
uv run pytest -q                     # 34 tests, under a second
uv run python -m pdfrag.cli demo     # three sample answers, then hybrid vs naive on the golden set
uv run python -m pdfrag.cli ask "What is the factory IP address of the LX-40?"
uv run python scripts/eval.py        # acceptance numbers only (add --json, or --mode naive)
uv run python scripts/make_golden.py # regenerate the synthetic half of the golden set
```

## Architecture

```
manual pages (layout JSON, or a real PDF via pdf.py)
  -> layout.reading_order     drop running headers, read column by column inside full-width bands
  -> chunking.chunk_document  chunks keep page, section and kind; a table is one chunk, one line per row
  -> store (InMemory, SQLite or Qdrant) + embed (hashing n-grams, or a real model) + bm25
question
  -> Pipeline.route           names one product? filter on doc_id (a metadata filter)
  -> BM25 top 20 + dense top 20 -> fusion.rrf -> rerank top 5
  -> abstain gate             top rerank score below 0.5: "Not in the documents."
  -> model.complete           prompt with [doc p.N] labelled blocks
  -> check_citations          every sentence must cite a context page; one repair, then refuse
  -> evaluate + judge         recall, faithfulness, citations, refusals
```

| File | Role |
| --- | --- |
| `src/pdfrag/layout.py`, `pdf.py` | Page and block model, reading order, optional pypdf loader |
| `src/pdfrag/chunking.py`, `corpus.py` | Layout-aware chunks with page numbers, bundled corpus loader |
| `src/pdfrag/store.py`, `embed.py` | Vector store interface with filters, local stores, embedders |
| `src/pdfrag/bm25.py`, `fusion.py`, `rerank.py` | Keyword search, reciprocal rank fusion, reranker |
| `src/pdfrag/models.py`, `pipeline.py` | Prompt, fake and real models, the answer pipeline |
| `src/pdfrag/golden.py`, `judge.py`, `evaluate.py`, `cli.py` | Golden set, faithfulness judge, metrics, CLI |
| `data/manuals/` | Three page-marked manuals: two-column pages, tables, running headers |
| `data/golden/` | `synthetic.json` (24, generated), `handwritten.json` (26, 7 unanswerable), `judge_calibration.json` |

## How each acceptance target is measured

| Target | Measured by | Test |
| --- | --- | --- |
| Context recall at least 0.85 | Each answerable item lists evidence quotes with doc and page. Recall is the share found in a reranked top-5 chunk on that page, averaged over the 43 answerable items. | `tests/test_accept_context_recall.py` |
| Faithfulness at least 0.90 | Each answer sentence is a claim, checked only against the chunks on the pages it cites by `ClaimSupportJudge`: every number must appear and at least 80% of content terms. Score is supported claims over all claims. | `tests/test_accept_faithfulness.py`, `tests/test_judge.py` |
| Every answer cites a page | Every answered question must cite at least one page, every sentence must carry a citation, and every cited `(doc, page)` must exist in the corpus. | `tests/test_accept_citations.py` |
| "Not in the documents" | All 7 unanswerable questions must get exactly `Not in the documents.`, and at least 95% of answerable ones must not. | `tests/test_accept_not_in_documents.py` |

Current numbers with the fake model: recall 1.00, faithfulness 1.00, citations 1.00, refusals 1.00. The
naive baseline (dense top 5, no routing, rerank or gate) matches on recall in a corpus this small, but answers
2 of the 7 unanswerable questions with confident, cited, irrelevant text: that is the stretch write-up in one line.
An extra sanity number, "answers containing the expected value", is 0.95: two answers are faithful to their
citations but quote the wrong sentence. Faithfulness does not catch that, which is why you keep both.

### Calibrating a model judge

The lexical judge is cheap and repeatable, but it reads words, not meaning: `judge_calibration.json` holds
18 hand-labelled claims and it agrees on 16. Its two misses are a negation flip and two numbers swapped
between facts. A model judge (`ModelJudge`) can catch those, but only trust it after calibration:

1. Label 50 to 100 real claims from your own answers by hand, with at least a third unsupported.
2. Run `calibrate(ModelJudge(model), labelled)`. Aim for agreement of at least 0.9 and watch `false_pass`
   (unsupported claims it lets through) more than `false_fail`.
3. Fix the judge prompt, not the labels. Keep the replies to one word and parse strictly: anything but
   `SUPPORTED` counts as unsupported.
4. Re-run calibration whenever you change the judge model or prompt, and keep the lexical judge as a
   cross-check: large disagreement between them on a new run is worth reading by hand.

## Plugging in real components

- **A real model**: `PDFRAG_PROVIDER=openai-compatible PDFRAG_MODEL=<name> uv run python -m pdfrag.cli demo`.
  `PDFRAG_BASE_URL` defaults to a local Ollama server (`http://localhost:11434/v1`); point it at vLLM, LM Studio
  or a hosted API instead. Set `PDFRAG_API_KEY` only if the server needs one. The key is read from the
  environment and never printed. Tests never use it.
- **Real PDFs**: `uv sync --extra pdf`, then `pdf.load_pdf(path)` returns the same `Document` model. pypdf does
  not detect tables; for table-heavy files use a layout model or pdfplumber and emit `Block(kind="table")`.
- **Embeddings and a cross-encoder**: `uv sync --extra models`, then pass `SentenceTransformerEmbedder()` and
  `CrossEncoderReranker()` to `Pipeline`. Recalibrate `threshold` for the new score scale: run the eval and pick
  the value that refuses all unanswerable items while keeping answered at least 0.95.
- **Qdrant**: `uv sync --extra qdrant`, then `Pipeline(chunks, store=QdrantStore(dim, location=":memory:"))`.
  pgvector or Chroma need the same three methods: `add`, `search(vector, k, where)` and `chunks(where)`.

## Milestones

These match the milestone guide on the site.

- **M1. Parse pages by layout and chunk with page numbers**: `src/pdfrag/layout.py`, `src/pdfrag/chunking.py`,
  `src/pdfrag/corpus.py`, `tests/test_layout.py`
- **M2. Store vectors with metadata filters**: `src/pdfrag/embed.py`, `src/pdfrag/store.py`, `tests/test_store.py`
- **M3. Combine BM25 and dense search with RRF, then rerank**: `src/pdfrag/bm25.py`, `src/pdfrag/fusion.py`,
  `src/pdfrag/rerank.py`, `src/pdfrag/pipeline.py` (`route`, `retrieve`), `tests/test_retrieval.py`
- **M4. Answer with citations, or refuse**: `src/pdfrag/models.py`, `src/pdfrag/pipeline.py` (`answer`,
  `check_citations`), `tests/test_models.py`
- **M5. Build the golden set and a calibrated judge**: `src/pdfrag/golden.py`, `scripts/make_golden.py`,
  `data/golden/`, `src/pdfrag/judge.py`, `tests/test_golden.py`, `tests/test_judge.py`
- **M6. Measure the acceptance targets**: `src/pdfrag/evaluate.py`, `src/pdfrag/cli.py`, `scripts/eval.py`,
  `tests/test_accept_*.py`
