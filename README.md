# WattBot 2026: RAG over AI and data-center sustainability research

**A retrieval-augmented pipeline that answers precise questions about the energy, water and carbon footprint of AI and data centers from 122 documents, cites its sources, and abstains when the documents cannot support an answer.** Entry to the [WattBot 2026](https://www.kaggle.com/) Kaggle community competition (UW–Madison ML+X).

> **Status:** v1 in development. Builds on my [WattBot 2025](https://github.com/BensonAnalyst/wattbot-rag) late submission (0.822 public / 0.752 private). The full report is in [WRITEUP.md](WRITEUP.md).

## What changed from 2025, and what the pipeline does about it

| 2026 change | Pipeline response |
|---|---|
| 122 documents (was 32), including long government and industry reports | Same page-level chunking and hybrid retrieval; PDFs fetched once, politely, and cached as a private dataset |
| Answering an unanswerable question now loses the value **and** the refusal component | Refusal stays allowed but is measured; majority voting abstains only when most votes refuse |
| Citations scored by **F1** (extra documents cost points) | The model cites only documents it took a fact from; voting keeps only documents most agreeing votes cited |
| New question types: two-paper, "reconcile" (sources that seem to disagree), figure-only, two-value answers | Prompt rules for scope checks (site vs source, year, region, company claim) and for `[low,high]` ranges vs `(a,b)` value pairs |
| 2025 winner's biggest gains: excerpts before the question, ensemble voting | Both adopted: context-first prompt, optional 3-vote majority with separate caching per vote |

## Evaluation

- **Official metric offline:** the notebook scores the 245 training questions with the organisers' `Score.py`; `src/wattbot/score.py` is a tested mirror (identical scores on the training file and on perturbed submissions) for running without competition files.
- **Breakdown by question type** (Quote, Table, Figure, Math, CrossPaper, Reconcile, unanswerable) and by cohort, because the 2026 questions are where the difficulty moved.
- Only 2 submissions a day, so leaderboard submissions are reserved for changes that beat the previous version offline.

## Rules followed

- Answers produced end to end by code; no manual lookups.
- No competition data in this repo (documents, CSVs and `Score.py` may not be redistributed); the notebook fetches documents from their source URLs once and caches them.
- Repo kept private during the competition, as Kaggle's rules require for competition code; published with a pinned tag at the close.
- Open-source components with permissive licences: pdfplumber (MIT), sentence-transformers (Apache 2.0), bge-small-en-v1.5 (MIT), gpt-oss-120b (Apache 2.0, served on Databricks).

## Run it

**Kaggle:** import `notebooks/wattbot2026_pipeline.ipynb`, add the competition data, turn Internet and a GPU on, and add three secrets: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `GITHUB_TOKEN`.

**Tests (offline, no data needed):** `pip install -r requirements-dev.txt && pytest -q`
