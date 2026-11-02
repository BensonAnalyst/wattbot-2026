"""Hybrid retrieval: BM25 keywords + optional dense embeddings, fused with reciprocal rank fusion.

Questions often name the paper ("Dodge et al. (2022)", "Cottier et al.") or the model it studies. When the
question names an author of one of the 32 documents, retrieval is restricted to that author's documents.
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

STOP = set("""a an and are as at be by for from has have how in is it its of on or that the this to was were what
which with when where who whom why will would can could does did do than then there these those per into over
about after before between according reported estimated using used""".split())


def tokenize(text: str) -> list:
    """Keep compound names whole ("gshard-600b", "ml.energy") and also add their parts ("gshard", "600b")."""
    out = []
    for tok in re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower()):
        parts = re.split(r"[.\-]", tok)
        out += [t for t in ([tok] + parts if len(parts) > 1 else [tok]) if t not in STOP]
    return out


class BM25:
    def __init__(self, docs: list, k1: float = 1.4, b: float = 0.75):
        self.tokens = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        df = Counter(t for toks in self.tokens for t in set(toks))
        n = len(self.tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(toks) for toks in self.tokens]

    def scores(self, query: str) -> list:
        q = tokenize(query)
        out = []
        for tf, toks in zip(self.tf, self.tokens):
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl)
            out.append(sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf))
        return out


def author_hints(question: str, authors: dict) -> set:
    """authors: {doc_id: first-author surname}. Returns the doc IDs whose first author the question names."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    named = {doc for doc, name in authors.items() if name and len(name) > 2 and name in words
             and re.search(rf"\b{name}\b\s+(et al|\(|and|\d{{4}})", question.lower())}
    return named


def rrf(rankings: list, k: int = 60, weights: list = None) -> dict:
    """Reciprocal rank fusion; weights scale each ranking's contribution (e.g. favour keyword search)."""
    fused = defaultdict(float)
    for ranking, w in zip(rankings, weights or [1.0] * len(rankings)):
        for rank, idx in enumerate(ranking):
            fused[idx] += w / (k + rank + 1)
    return fused


class Retriever:
    def __init__(self, chunks: list, authors: dict, embed_fn=None, bm25_weight: float = 1.0,
                 dense_weight: float = 1.0, max_per_doc: int = None):
        """embed_fn(list[str]) -> list of unit-normalised vectors, or None for BM25 only.

        bm25_weight / dense_weight: how much each ranking counts in the fusion (dense_weight 0 = keyword only).
        max_per_doc: at most this many excerpts from one document, so a question that needs two papers is
        not crowded out by eight excerpts of the first one."""
        self.chunks, self.authors, self.embed_fn = chunks, authors, embed_fn
        self.bm25 = BM25([c["search_text"] for c in chunks])
        self.vectors = np.asarray(embed_fn([c["search_text"] for c in chunks])) if embed_fn else None
        self._qvec = {}
        self.configure(bm25_weight=bm25_weight, dense_weight=dense_weight, max_per_doc=max_per_doc)

    def configure(self, bm25_weight=1.0, dense_weight=1.0, max_per_doc=None):
        self.bm25_weight, self.dense_weight, self.max_per_doc = bm25_weight, dense_weight, max_per_doc
        return self

    def _query_vector(self, question):
        if question not in self._qvec:
            self._qvec[question] = np.asarray(self.embed_fn([question])[0])
        return self._qvec[question]

    def search(self, question: str, k: int = 8, pool: int = 50, **override) -> list:
        bw = override.get("bm25_weight", self.bm25_weight)
        dw = override.get("dense_weight", self.dense_weight)
        cap = override.get("max_per_doc", self.max_per_doc)
        allowed = author_hints(question, self.authors)
        idx = [i for i, c in enumerate(self.chunks) if not allowed or c["doc_id"] in allowed]
        bm = self.bm25.scores(question)
        rankings = [[i for i in sorted(idx, key=lambda i: -bm[i])[:pool] if bm[i] > 0]]
        weights = [bw]
        if self.vectors is not None and dw > 0:
            sims = self.vectors[idx] @ self._query_vector(question)
            rankings.append([idx[j] for j in np.argsort(-sims)[:pool]])
            weights.append(dw)
        fused = rrf(rankings, weights=weights)
        best, per_doc = [], defaultdict(int)
        for i in sorted(fused, key=lambda i: -fused[i]):
            doc = self.chunks[i]["doc_id"]
            if cap and per_doc[doc] >= cap:
                continue
            per_doc[doc] += 1
            best.append(i)
            if len(best) == k:
                break
        return [{**self.chunks[i], "score": round(fused[i], 5)} for i in best]


def tune(retriever, questions, k: int, grid: list, parse_refs) -> "pd.DataFrame":
    """Document recall@k on answerable training questions for each retrieval setting (no LLM calls)."""
    import pandas as pd

    rows = []
    ans = questions[questions["answer_value"] != "is_blank"]
    cross = (ans["CrossPaper"].astype(str) == "1").values if "CrossPaper" in ans else None
    for cfg in grid:
        rec = [doc_recall(retriever.search(q.question, k=k, **cfg), parse_refs(q.ref_id)) for q in ans.itertuples()]
        row = {**cfg, "recall": round(float(np.mean(rec)), 3)}
        if cross is not None and cross.any():
            row["recall_crosspaper"] = round(float(np.mean([r for r, c in zip(rec, cross) if c])), 3)
        rows.append(row)
    out = pd.DataFrame(rows)
    keys = [c for c in ("recall", "recall_crosspaper") if c in out]
    return out.sort_values(keys, ascending=False, kind="stable").reset_index(drop=True)


def doc_recall(retrieved: list, truth_refs: set) -> float:
    """Share of the ground-truth documents that appear in the retrieved chunks."""
    if not truth_refs or truth_refs == {"is_blank"}:
        return None
    got = {c["doc_id"].lower() for c in retrieved}
    return len(got & truth_refs) / len(truth_refs)
