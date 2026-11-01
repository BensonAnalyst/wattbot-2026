"""Offline WattBot 2026 score, mirroring the organisers' Score.py so it runs in tests without competition files.

The notebook prefers the official Score.py from the competition data when it is present; this module is the
fallback and the reference for the unit tests.

Per question:  0.75 * answer_value  +  0.20 * citation F1  +  0.05 * clean refusal
- answer_value: numbers within 0.1% (relative or 1e-3 absolute); text compared case- and punctuation-insensitive;
  gold [lo,hi] is a range (a single value inside it is correct); gold (a,b) is a list (every value required);
  gold is_blank means the question is unanswerable and only a blank answer is correct.
- citation F1 between the predicted and gold document-ID sets.
- clean refusal: on an unanswerable question, answer_value, ref_id, ref_url and supporting_materials must all be
  blank (the unit may stay). Always 1 on answerable questions.
"""

import json
import math
import re

BLANK = "is_blank"
TOL = 1e-3
_BLANK_TOKENS = {"", "na", "n/a", "is_blank", "nan", "none"}
_FALLBACK = "unable to answer with confidence based on the provided documents"
NA_FIELDS = ("answer_value", "ref_id", "ref_url", "supporting_materials")


def is_blank(value) -> bool:
    s = "" if value is None else str(value).strip().lower()
    return s in _BLANK_TOKENS or s.startswith(_FALLBACK)


def parse_number(value):
    try:
        return float(str(value).strip().replace(",", ""))
    except ValueError:
        return None


def canon_text(value) -> str:
    s = re.sub(r"[^\w\s]", " ", str(value).lower())
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^(?:the|a|an)\s+", "", s)


def list_kind(value):
    s = str(value).strip()
    if s.startswith("[") and s.endswith("]"):
        return "range"
    if s.startswith("(") and s.endswith(")"):
        return "list"
    return None


def parse_list(value):
    s = str(value).strip()
    if list_kind(s) is None:
        return None
    out = []
    for item in s[1:-1].split(","):
        item = item.strip().strip("'\"")
        n = parse_number(item)
        out.append(n if n is not None else item.lower())
    return out


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)


def _lists_equal(a: list, b: list) -> bool:
    if all(isinstance(x, float) for x in a + b):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(sorted(a), sorted(b)))
    return sorted(str(x).strip().lower() for x in a) == sorted(str(x).strip().lower() for x in b)


def _strip_unit(value, unit) -> str:
    u = str(unit or "").strip().lower()
    v = canon_text(value)
    if u and u != BLANK:
        for tok in re.split(r"[\s/]+", u):
            if tok and re.search(r"[a-z]", tok):
                v = re.sub(rf"\b{re.escape(tok)}\b", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def value_correct(pred, truth, unit=None) -> bool:
    if is_blank(truth):
        return is_blank(pred)
    if is_blank(pred):
        return False
    t_list, p_list = parse_list(truth), parse_list(pred)
    if t_list is not None:
        if p_list is not None:
            return _lists_equal(t_list, p_list)
        if list_kind(truth) == "range" and len(t_list) == 2 and all(isinstance(x, float) for x in t_list):
            x = parse_number(pred)
            lo, hi = sorted(t_list)
            return x is not None and lo - TOL <= x <= hi + TOL
        return False
    t, p = parse_number(truth), parse_number(pred)
    if t is not None and p is not None:
        return _close(p, t)
    t2, p2 = parse_number(_strip_unit(truth, unit)), parse_number(_strip_unit(pred, unit))
    if t2 is not None and p2 is not None:
        return _close(p2, t2)
    return canon_text(pred) == canon_text(truth)


def parse_refs(value) -> set:
    s = "" if value is None else str(value).strip()
    if not s:
        return set()
    if s.startswith("["):
        try:
            return {str(x).strip().lower() for x in json.loads(s.replace("'", '"'))}
        except ValueError:
            return set(re.findall(r"[a-z][a-z0-9]*\d{4}[a-z0-9]*", s.lower()))
    return {s.lower()}


def ref_f1(pred, truth) -> float:
    a, b = parse_refs(truth), parse_refs(pred)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    hit = len(a & b)
    if not hit:
        return 0.0
    p, r = hit / len(b), hit / len(a)
    return 2 * p * r / (p + r)


def score_row(pred: dict, truth: dict) -> dict:
    v = float(value_correct(pred.get("answer_value"), truth["answer_value"], truth.get("answer_unit")))
    r = ref_f1(pred.get("ref_id"), truth["ref_id"])
    n = float(all(is_blank(pred.get(f)) for f in NA_FIELDS)) if is_blank(truth["answer_value"]) else 1.0
    return {"value": v, "ref": round(r, 3), "na": n, "score": round(0.75 * v + 0.20 * r + 0.05 * n, 4)}


def score_frame(pred_df, truth_df):
    """Per-question DataFrame and the mean score, joined on id."""
    import pandas as pd

    preds = {r["id"]: r for r in pred_df.to_dict("records")}
    rows = []
    for t in truth_df.to_dict("records"):
        p = preds.get(t["id"], {})
        rows.append({"id": t["id"], "question": t["question"], "truth": t["answer_value"],
                     "pred": p.get("answer_value"), "truth_ref": t["ref_id"], "pred_ref": p.get("ref_id"),
                     **score_row(p, t)})
    out = pd.DataFrame(rows)
    return out, round(out["score"].mean(), 4)
