"""Prompt, LLM call, and turning the model's JSON into one submission row in the train_QA.csv format."""

import hashlib
import json
import re

from .score import BLANK, parse_number

NA_TEXT = "Unable to answer with confidence based on the provided documents."
COLUMNS = ["id", "question", "answer", "answer_value", "answer_unit", "ref_id", "ref_url",
           "supporting_materials", "explanation"]

PROMPT = """You answer questions about the energy, water and carbon footprint of AI and data centers, using ONLY
the excerpts below. Each excerpt starts with its document ID and page.

EXCERPTS:
{sources}

Rules:
1. Use only the excerpts; no outside knowledge, even for well-known facts.
   Answer whenever the excerpts contain the needed facts, directly or through simple arithmetic, even if the
   wording differs from the question. Set "answerable": false only when the excerpts do not cover the subject
   or lack a number you need. Answering a question the documents cannot support scores zero, so do not guess.
2. Prefer the most exact figure: if a table gives 5.439 and the text says "about 5.4", use 5.439. Do not
   compute a value yourself when a document reports it directly.
3. Check scope before choosing between sources that seem to disagree: site vs source water, one year vs
   another, one region vs global, a company's own claim vs an independent estimate. Pick the value that matches
   what the question asks for.
4. answer_value must be expressed in the required unit:
   - QUESTION TYPE number: digits only, no commas, symbols (~, <, >, %, $) or words; convert the unit if
     needed (5.439 million liters -> 5439000 when the unit is liters). A value stated in a document is copied
     exactly; a value you calculate is rounded to 3 significant figures (312.5 / 24 = 13.02 -> 13).
   - A range the question asks for: [low,high]. Two separate values the question asks for: (first,second).
   - QUESTION TYPE true/false (including yes/no questions): 1 if the excerpts support the statement or the
     answer is yes, 0 if they contradict it or the answer is no.
   - QUESTION TYPE name: the shortest exact name or term as the document writes it, without words the
     question already implies ("real income effect", not "real income"; "net zero", not "net zero
     commitments"; "induction", not "induction impacts"). Never 1 or 0. Two names: (first,second).
5. Arithmetic on numbers from the excerpts is allowed, including combining two documents. Show it in
   "explanation".
6. ref_ids: only the document IDs whose excerpts you actually took a number or statement from. Extra
   documents lower the score; usually this is one document, two when you combined two.
7. supporting_materials: a short verbatim quote (or "Table N" / "Figure N") from those excerpts.

QUESTION: {question}
QUESTION TYPE: {qtype}
REQUIRED UNIT FOR answer_value: {unit}

Reply with JSON only:
{{"answerable": true, "answer": "short natural-language answer with unit", "answer_value": "...",
"ref_ids": ["docid"], "supporting_materials": "...", "explanation": "..."}}"""


def format_sources(chunks: list) -> str:
    return "\n\n".join(f"[{c['doc_id']}, page {c['page']}]\n{c['text']}" for c in chunks)


_YES_NO = re.compile(r"^(is|are|was|were|does|do|did|can|could|has|have|had|will|would|should)\b", re.I)


def question_type(question: str, unit: str) -> str:
    """number, true/false or name. Yes/no questions ("Was the relationship ... linear?") are true/false too:
    the gold answer is 1 or 0 whether or not the question starts with "True or False"."""
    q = str(question).strip()
    no_unit = not unit or str(unit).strip() == BLANK
    if "true or false" in q.lower():  # often after a setup paragraph: "... True or False: the two findings ..."
        return "true/false"
    if no_unit and not re.match(r"^(what|which|who|whom|whose|where|when|why|how|name|list)\b", q, re.I):
        # the yes/no verb may follow a lead-in clause: "In experiments measuring X, was the relationship ...?"
        parts = re.split(r"[,.:;]\s+", q)
        starts = [i for i, c in enumerate(parts) if _YES_NO.match(c.strip())]
        # "...scale linearly, sublinearly, or superlinearly?" offers a choice, so it is a name question
        if starts and not re.search(r"\bor\b", " ".join(parts[starts[-1]:]), re.I):
            return "true/false"
    return "name" if no_unit else "number"


def build_messages(question: str, unit: str, chunks: list) -> list:
    return [{"role": "user", "content": PROMPT.format(question=question.strip(), qtype=question_type(question, unit),
                                                      unit=unit or BLANK, sources=format_sources(chunks))}]


def cache_key(messages: list, model: str) -> str:
    return hashlib.sha256((model + json.dumps(messages, sort_keys=True)).encode()).hexdigest()[:16]


def parse_reply(raw: str):
    text = re.sub(r"```(?:json)?", "", raw or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def normalise_value(value, unit: str) -> str:
    """Clean the model's answer_value into the competition's conventions."""
    if value is None:
        return BLANK
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return _fmt(float(value))
    if isinstance(value, list) and len(value) == 2:
        nums = [parse_number(normalise_value(v, unit)) for v in value]
        return f"[{_fmt(nums[0])},{_fmt(nums[1])}]" if None not in nums else BLANK
    text = str(value).strip()
    low = text.lower()
    if low in ("", "none", "null", "n/a", BLANK):
        return BLANK
    if low in ("true", "yes"):
        return "1"
    if low in ("false", "no"):
        return "0"
    if text.startswith("[") and text.endswith("]"):
        return normalise_value([p.strip() for p in text[1:-1].split(",")], unit) if text.count(",") == 1 else text
    if text.startswith("(") and text.endswith(")") and "," in text:  # two required values or names
        parts = [p.strip().strip("'\"") for p in text[1:-1].split(",")]
        cleaned = [normalise_value(p, unit) if parse_number(normalise_value(p, unit)) is not None else p
                   for p in parts]
        return "(" + ",".join(cleaned) + ")"
    num = re.fullmatch(r"[~<>≈$\s]*(-?[\d,]*\.?\d+(?:e[+-]?\d+)?)\s*(thousand|million|billion|trillion)?\s*[%x×]?",
                       low)
    if num:
        n = float(num.group(1).replace(",", ""))
        scale = num.group(2)
        if scale and scale not in (unit or "").lower():
            n *= _SCALE[scale]
        return _fmt(n)
    return text


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() and abs(n) < 1e15 else repr(float(n))


def na_row(qid: str, question: str, explanation: str = "") -> dict:
    return {"id": qid, "question": question, "answer": NA_TEXT, "answer_value": BLANK, "answer_unit": BLANK,
            "ref_id": BLANK, "ref_url": BLANK, "supporting_materials": BLANK, "explanation": explanation or BLANK}


def clean_refs(raw, valid_ids: set) -> list:
    """Keep the document IDs the model cited, even when written as 'li2025b, page 5' or 'Li2025b'."""
    if isinstance(raw, str):
        raw = [raw]
    found = []
    for item in raw or []:
        text = str(item).lower()
        found += [d for d in sorted(valid_ids, key=len, reverse=True) if re.search(rf"\b{re.escape(d.lower())}\b", text)]
    return list(dict.fromkeys(found))


def typed_value(reply: dict, question: str, unit: str) -> str:
    """Normalise answer_value, then check it fits the question type."""
    qtype = question_type(question, unit)
    value = normalise_value(reply.get("answer_value"), unit)
    if qtype == "true/false" and value not in ("0", "1"):
        first = re.match(r"\W*(\w+)", str(reply.get("answer") or ""))
        value = normalise_value(first.group(1) if first else None, unit)
        return value if value in ("0", "1") else BLANK
    if qtype == "name" and value in ("0", "1", BLANK):  # the model wrote a flag instead of the name
        name = str(reply.get("answer") or "").strip().rstrip(".")
        if normalise_value(re.sub(r"\W.*", "", name), unit) in ("0", "1"):  # the answer itself is yes/no
            return value if value != BLANK else normalise_value(re.sub(r"\W.*", "", name), unit)
        return name or BLANK
    return value


def to_row(qid: str, question: str, unit: str, reply, urls: dict, valid_ids: set, fallback_refs: list = ()) -> dict:
    """Build one submission row. Anything unusable becomes a full is_blank row (scored as unanswerable).

    If the model gives a value but no usable citation, the top retrieved document is cited instead of
    throwing the answer away: the value is worth 75% of the score, the citation 15%."""
    if not reply or not reply.get("answerable", True):
        return na_row(qid, question, (reply or {}).get("explanation", "No supporting evidence retrieved."))
    value = typed_value(reply, question, unit)
    refs = clean_refs(reply.get("ref_ids"), valid_ids) or [r for r in fallback_refs if r in valid_ids][:1]
    if value == BLANK or not refs:
        return na_row(qid, question, reply.get("explanation", ""))
    return {
        "id": qid, "question": question,
        "answer": str(reply.get("answer") or value),
        "answer_value": value,
        "answer_unit": unit or BLANK,
        "ref_id": "[" + ",".join(f"'{r}'" for r in refs) + "]",
        "ref_url": "[" + ",".join(f"'{urls[r]}'" for r in refs) + "]",
        "supporting_materials": str(reply.get("supporting_materials") or BLANK),
        "explanation": str(reply.get("explanation") or BLANK),
    }


def ask(q: dict, chunks: list, llm_fn, cache: dict, model: str, vote: int = 0):
    """One model reply. Vote 0 is deterministic (temperature 0); later votes sample at temperature 0.7.
    Every reply is cached under the prompt, model and vote number, so re-runs cost nothing."""
    messages = build_messages(q["question"], q.get("answer_unit"), chunks)
    key = cache_key(messages, model) + ("" if vote == 0 else f"-v{vote}")
    if key not in cache:
        cache[key] = llm_fn(messages) if vote == 0 else llm_fn(messages, temperature=0.7)
    return parse_reply(cache[key])


def _vote_key(value: str):
    """Group equivalent answers: numbers to 3 significant figures, text case- and punctuation-insensitive."""
    n = parse_number(value)
    if n is not None:
        return f"{n:.3g}"
    return re.sub(r"[^\w]+", " ", str(value).lower()).strip()


def vote_rows(rows: list, min_answer_votes: int = None) -> dict:
    """Majority vote over candidate rows for one question.

    The question is marked unanswerable when fewer than min_answer_votes (default: a majority) of the votes
    answered. Among the answers, the most common value wins (ties go to the earliest vote, the deterministic
    one). The citation is the documents cited by more than half of the winning votes: citation F1 punishes
    extra documents, so a paper only one of two agreeing votes cited is dropped."""
    n = len(rows)
    need = min_answer_votes if min_answer_votes is not None else n // 2 + 1
    answered = [r for r in rows if r["answer_value"] != BLANK]
    if len(answered) < need:
        return rows[0] if rows[0]["answer_value"] == BLANK else next(r for r in rows if r["answer_value"] == BLANK)
    groups = {}
    for r in answered:
        groups.setdefault(_vote_key(r["answer_value"]), []).append(r)
    winners = max(groups.values(), key=lambda g: (len(g), -rows.index(g[0])))
    best = dict(winners[0])
    counts = {}
    for r in winners:
        for ref in _refs(r["ref_id"]):
            counts[ref] = counts.get(ref, 0) + 1
    keep = [ref for ref, c in counts.items() if c * 2 > len(winners)] or _refs(best["ref_id"])
    urls = dict(zip(_refs(best["ref_id"]), _refs(best["ref_url"])))
    for r in winners:
        urls.update(zip(_refs(r["ref_id"]), _refs(r["ref_url"])))
    best["ref_id"] = "[" + ",".join(f"'{x}'" for x in keep) + "]"
    best["ref_url"] = "[" + ",".join(f"'{urls[x]}'" for x in keep) + "]"
    best["explanation"] = f"{best['explanation']} [votes: {len(winners)}/{n} agree]"
    return best


def _refs(field: str) -> list:
    return re.findall(r"'([^']+)'", str(field))


def answer_question(q: dict, chunks: list, llm_fn, urls: dict, cache: dict, model: str,
                    retry_chunks: list = None, n_votes: int = 1) -> tuple:
    """One submission row, plus whether the wider retry was used.

    If the deterministic first vote finds no answer in the top excerpts, every vote uses the wider set
    (retry_chunks) instead. With n_votes > 1, the votes are combined by majority (see vote_rows)."""
    used = chunks
    reply = ask(q, chunks, llm_fn, cache, model)
    retried = False
    if retry_chunks and len(retry_chunks) > len(chunks) and (not reply or not reply.get("answerable", True)):
        used, retried = retry_chunks, True
        reply = ask(q, retry_chunks, llm_fn, cache, model)
    fallback = [c["doc_id"] for c in used]
    make = lambda rep: to_row(q["id"], q["question"], q.get("answer_unit"), rep, urls, set(urls), fallback)
    rows = [make(reply)] + [make(ask(q, used, llm_fn, cache, model, vote=v)) for v in range(1, n_votes)]
    return (vote_rows(rows) if n_votes > 1 else rows[0]), retried
