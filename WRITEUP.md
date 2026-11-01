# WattBot 2026 — team writeup

*Draft: filled in as the pipeline improves. Final version pinned to the submission tag.*

## Submission card

```nix
code_url: https://github.com/BensonAnalyst/wattbot-2026/tree/<tag>
writeup_url: https://github.com/BensonAnalyst/wattbot-2026/blob/<tag>/WRITEUP.md
generation_model: gpt-oss-120b (Databricks serving endpoint)
embedding_model: BAAI/bge-small-en-v1.5
reranker_model: none
other_models: none
leaderboard_score: <tbd>
external_data: none
hardware: Kaggle notebook, GPU T4 x2 (embeddings); Databricks Free Edition (generation)
cost_estimate: <tbd>
wall_clock: <tbd>
```

## Approach

- **Retrieval:** page-level chunks of ~220 words (60 overlap), each prefixed with its document ID and title; BM25 and bge-small dense retrieval fused by reciprocal rank fusion; top 8 excerpts, widened to 20 when the model refuses.
- **Generation:** gpt-oss-120b, excerpts before the question, JSON output; question type (number, true/false, name) stated explicitly; calculated values rounded to 3 significant figures; optional majority vote over 3 samples.
- **What I tried that didn't work:** <tbd>

## Knowing when not to answer

<tbd: refusal rule, retry, voting threshold, and the measured trade-off on the training set>

## Handling sources that disagree

<tbd>

## Citations

<tbd: cite only documents a fact was taken from; vote-level filtering of citations>

## What I'd do with another week

<tbd>
