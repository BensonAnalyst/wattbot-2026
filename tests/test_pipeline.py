"""Offline tests. No competition data is committed: everything below uses small invented examples."""

import json

import pandas as pd
import pytest

from wattbot.answer import NA_TEXT, build_messages, normalise_value, parse_reply, to_row
from wattbot.docs import chunk_pages, extract_pages, first_author, load_metadata
from wattbot.retrieve import Retriever, author_hints, doc_recall
from wattbot.score import ref_f1, score_frame, score_row, value_correct

URLS = {"dodge2022": "https://arxiv.org/pdf/2206.05229", "strubell2019": "https://arxiv.org/pdf/1906.02243"}


# --- scoring (2026 rules; the mirror is also checked against the official Score.py in the notebook) -------
@pytest.mark.parametrize("pred,truth,ok", [
    ("1438", "1438", True), ("1438.9", "1438", True), ("1440", "1438", False),
    ("[0.02,0.1]", "[0.02,0.1]", True), ("0.05", "[0.02,0.1]", True),   # a value inside a gold range is correct
    ("(1,5)", "(1,5)", True), ("[1,5]", "(1,5)", True), ("3", "(1,5)", False),  # a gold list needs every value
    ("(GPT-4,Claude-3.5)", "(Claude-3.5,GPT-4)", True),
    ("1", "1", True), ("0", "1", False), ("Water consumption.", "water consumption", True),
    ("Backblaze (B2)", "Backblaze B2", True), ("the watershed", "Watershed", True),
    ("is_blank", "is_blank", True), ("12", "is_blank", False), ("is_blank", "12", False),
])
def test_value_correct(pred, truth, ok):
    assert value_correct(pred, truth) is ok


def test_citation_f1_and_clean_refusal():
    assert ref_f1("['wu2021b']", "['wu2021b','patterson2021','a2020','b2021']") == pytest.approx(0.4)  # 1 of 4
    assert ref_f1("['wu2021b','x2020']", "['wu2021b']") == pytest.approx(2 / 3)  # an extra citation costs
    truth = {"answer_value": "is_blank", "answer_unit": "percent", "ref_id": "is_blank"}
    blank = {f: "is_blank" for f in ("answer_value", "ref_id", "ref_url", "supporting_materials")}
    assert score_row({**blank, "answer_unit": "percent"}, truth)["score"] == 1.0  # unit may stay
    bluff = {**blank, "ref_id": "['chen2024']"}
    assert score_row(bluff, truth)["score"] == 0.75  # refused, but cited a paper: loses citation and NA
    assert score_row({**blank, "answer_value": "42"}, truth)["value"] == 0.0  # answering costs the value


def test_score_frame_perfect():
    t = pd.DataFrame([{"id": "q1", "question": "x", "answer_value": "4.3", "answer_unit": "tCO2e",
                       "ref_id": "['patterson2021']"}])
    _, mean = score_frame(t.copy(), t)
    assert mean == 1.0


# --- answer formatting -----------------------------------------------------------------------------
@pytest.mark.parametrize("value,unit,out", [
    ("1,438", "lbs", "1438"), ("~13", "days", "13"), ("55%", "percent", "55"), ("5.439 million", "liters", "5439000"),
    ("2.5 billion", "billion cubic meters", "2.5"), ("TRUE", "is_blank", "1"), (False, "is_blank", "0"),
    ([80, 90], "percent", "[80,90]"), ("[0.02, 0.1]", "500 mL bottles", "[0.02,0.1]"),
    ("(1.65, 0.5)", "L/kWh", "(1.65,0.5)"), ("(GPT-4, Claude-3.5)", "is_blank", "(GPT-4,Claude-3.5)"),
    ("Water consumption", "is_blank", "Water consumption"), (8.720430108, "multiplier", "8.720430108"),
])
def test_normalise_value(value, unit, out):
    assert normalise_value(value, unit) == out


def test_to_row_answer_and_unanswerable():
    reply = parse_reply('```json\n{"answerable": true, "answer": "1438 lbs", "answer_value": "1,438", '
                        '"ref_ids": ["strubell2019", "made_up_doc"], "supporting_materials": "Table 3", '
                        '"explanation": "Quote"}\n```')
    row = to_row("q262", "BERT?", "lbs", reply, URLS, set(URLS))
    assert row["answer_value"] == "1438" and row["ref_id"] == "['strubell2019']" and row["answer_unit"] == "lbs"
    na = to_row("q164", "Elephant?", "lbs", {"answerable": False}, URLS, set(URLS))
    assert na["answer"] == NA_TEXT and all(na[c] == "is_blank" for c in ("answer_value", "answer_unit", "ref_id"))
    assert to_row("q1", "?", "lbs", None, URLS, set(URLS))["answer_value"] == "is_blank"  # unparseable reply


def test_prompt_contains_unit_and_sources():
    msg = build_messages("How long?", "days", [{"doc_id": "dodge2022", "page": 3, "text": "about 60 days"}])
    content = msg[0]["content"]
    assert "REQUIRED UNIT FOR answer_value: days" in content and "[dodge2022, page 3]" in content
    assert content.index("EXCERPTS:") < content.index("QUESTION: How long?")  # context before the question


# --- documents and retrieval -----------------------------------------------------------------------
def test_metadata_windows_1252(tmp_path):
    p = tmp_path / "metadata.csv"
    p.write_bytes("id,type,title,year,citation,url\ngriggs2024,paper,M\xe9lange,2024,\"Tyler Griggs, Ion Stoica\","
                  "https://arxiv.org/pdf/2404.14527\n".encode("cp1252"))
    meta = load_metadata(p)
    assert meta.loc[0, "title"] == "Mélange" and first_author(meta.loc[0, "citation"]) == "griggs"
    assert first_author("Amazon Staff. (2023). Amazon Sustainability Report.") == "staff"


def test_pdf_extraction_and_chunks(tmp_path):
    from reportlab.pdfgen import canvas
    pdf = tmp_path / "x.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "Training GShard-600B used 24 MWh and produced 4.3 net tCO2e.")
    c.showPage()
    c.drawString(72, 720, "Page two text.")
    c.save()
    pages = extract_pages(pdf)
    assert len(pages) == 2 and "4.3 net tCO2e" in pages[0]["text"]
    chunks = chunk_pages("patterson2021", "Carbon Emissions", pages)
    assert chunks[0]["chunk_id"] == "patterson2021:p1:0" and chunks[1]["page"] == 2
    assert chunks[0]["search_text"].startswith("patterson2021 Carbon Emissions")


def _chunks():
    texts = {
        "dodge2022": "The 6.1 billion parameter model would take 60 days to train fully and consumed 13.8 MWh.",
        "strubell2019": "Training BERT base on 64 V100 GPUs for 79 hours emitted 1438 lbs of CO2e.",
        "patterson2021": "Training GShard-600B used 24 MWh and produced 4.3 net tCO2e.",
    }
    return [{"chunk_id": f"{d}:p1:0", "doc_id": d, "page": 1, "text": t, "search_text": f"{d} {t}"}
            for d, t in texts.items()]


def test_bm25_finds_the_right_document():
    r = Retriever(_chunks(), {"dodge2022": "dodge", "strubell2019": "strubell", "patterson2021": "patterson"})
    top = r.search("What were the net CO2e emissions from training GShard-600B?", k=1)
    assert top[0]["doc_id"] == "patterson2021"
    assert doc_recall(top, {"patterson2021"}) == 1.0


def test_author_named_in_question_restricts_search():
    authors = {"dodge2022": "dodge", "strubell2019": "strubell", "patterson2021": "patterson"}
    assert author_hints("Dodge et al. (2022) estimate that training takes how many days?", authors) == {"dodge2022"}
    assert author_hints("What is the energy of BERT training?", authors) == set()
    r = Retriever(_chunks(), authors)
    assert {c["doc_id"] for c in r.search("Dodge et al. (2022): how many MWh for BERT?", k=3)} == {"dodge2022"}


def test_dense_fusion_runs_with_fake_embeddings():
    import numpy as np
    def embed(texts):
        return [np.array([1.0, 0.0]) if "GShard" in t else np.array([0.0, 1.0]) for t in texts]
    r = Retriever(_chunks(), {}, embed_fn=embed)
    assert r.search("GShard emissions", k=1)[0]["doc_id"] == "patterson2021"


def test_answer_all_with_fake_llm_and_cache(tmp_path):
    from wattbot.run import answer_all
    calls = []

    def llm(messages):
        calls.append(1)
        if "GShard" in messages[0]["content"].split("QUESTION:")[-1]:
            return json.dumps({"answerable": True, "answer": "4.3 tCO2e", "answer_value": "4.3",
                               "ref_ids": ["patterson2021"], "supporting_materials": "q", "explanation": "e"})
        return json.dumps({"answerable": False})

    urls = {"patterson2021": "u1", "dodge2022": "u2", "strubell2019": "u3"}
    qs = pd.DataFrame([{"id": "q1", "question": "Net CO2e of GShard-600B?", "answer_unit": "tCO2e"},
                       {"id": "q2", "question": "How many MWh did training use per day?", "answer_unit": "MWh"},
                       {"id": "q3", "question": "How many MWh did training use per day?", "answer_unit": "MWh"}])
    r = Retriever(_chunks(), {})
    sub, _, retried = answer_all(qs, r, llm, urls, tmp_path / "cache.json", "m", k=1, k_retry=3, threads=2)
    assert list(sub["id"]) == ["q1", "q2", "q3"]
    assert sub.loc[0, "answer_value"] == "4.3" and sub.loc[1, "answer_value"] == "is_blank"
    assert retried == {"q2", "q3"}  # refused at k=1, so retried with the wider set
    off_topic = pd.DataFrame([{"id": "q4", "question": "How much does an elephant weigh?", "answer_unit": "lbs"}])
    sub2, _, retried2 = answer_all(off_topic, r, llm, urls, tmp_path / "c2.json", "m", k=1, k_retry=3)
    assert sub2.loc[0, "answer_value"] == "is_blank" and not retried2  # nothing retrieved, nothing to widen
    n = len(calls)
    answer_all(qs, r, llm, urls, tmp_path / "cache.json", "m", k=1, k_retry=3, threads=2)
    assert len(calls) == n  # second run served from the cache


def test_retry_with_wider_context_finds_the_answer():
    from wattbot.answer import answer_question
    narrow = [{"doc_id": "dodge2022", "page": 1, "text": "irrelevant"}]
    wide = narrow + [{"doc_id": "strubell2019", "page": 2, "text": "household uses 10715 kWh per year"}]

    def llm(messages):
        if "10715" in messages[0]["content"]:
            return json.dumps({"answerable": True, "answer": "1.3", "answer_value": "1.3", "ref_ids": ["strubell2019"],
                               "supporting_materials": "q", "explanation": "13.8 / 10.7"})
        return json.dumps({"answerable": False})

    row, retried = answer_question({"id": "q272", "question": "household-years?", "answer_unit": "household-years"},
                                   narrow, llm, URLS, {}, "m", retry_chunks=wide)
    assert retried and row["answer_value"] == "1.3"


def test_question_type_guards_the_value():
    from wattbot.answer import question_type, typed_value
    assert question_type("True or False: X.", "is_blank") == "true/false"
    assert question_type("Name the benchmark?", "is_blank") == "name"
    assert question_type("How many days?", "days") == "number"
    # name question answered with a flag: fall back to the answer text (v2 bug on q003, q202)
    assert typed_value({"answer_value": "1", "answer": "ML.ENERGY Benchmark."}, "Name the suite?", "is_blank") \
        == "ML.ENERGY Benchmark"
    assert typed_value({"answer_value": "TRUE"}, "True or False: X.", "is_blank") == "1"
    # v3 bug on q166, q202, q282: the model left answer_value empty for name questions
    assert typed_value({"answer_value": "", "answer": "GPT-3"}, "Which model?", "is_blank") == "GPT-3"
    assert typed_value({"answer_value": None, "answer": "water consumption"}, "What term?", "is_blank") \
        == "water consumption"
    assert typed_value({"answer_value": "maybe", "answer": "False, it does not."}, "True or False: X.", "is_blank") == "0"


def test_citation_cleanup_and_fallback():
    from wattbot.answer import clean_refs
    ids = {"li2025b", "li2025a", "dodge2022"}
    assert clean_refs(["li2025b, page 5", "Dodge2022"], ids) == ["li2025b", "dodge2022"]
    assert clean_refs("li2025b", ids) == ["li2025b"]
    assert clean_refs(["Li et al."], ids) == []
    reply = {"answerable": True, "answer": "5.439 million L", "answer_value": "5439000", "ref_ids": ["Li et al."]}
    row = to_row("q124", "Water?", "liters", reply, {"li2025b": "u"}, {"li2025b"}, fallback_refs=["li2025b"])
    assert row["answer_value"] == "5439000" and row["ref_id"] == "['li2025b']"  # v2 bug: this answer was discarded


def test_prompt_states_type_and_rounding():
    content = build_messages("How long?", "days", [])[0]["content"]
    assert "QUESTION TYPE: number" in content and "3 significant figures" in content


def test_majority_vote():
    from wattbot.answer import na_row, vote_rows
    def row(v, refs):
        return {"id": "q", "question": "?", "answer": v, "answer_value": v, "answer_unit": "MWh",
                "ref_id": "[" + ",".join(f"'{r}'" for r in refs) + "]",
                "ref_url": "[" + ",".join(f"'u-{r}'" for r in refs) + "]", "supporting_materials": "s",
                "explanation": "e"}
    best = vote_rows([row("13.02", ["a2020"]), row("13", ["a2020", "b2021"]), row("15", ["c2022"])])
    assert best["answer_value"] == "13.02" and best["ref_id"] == "['a2020']"  # 13.02 and 13 agree at 3 s.f.
    assert "2/3 agree" in best["explanation"]
    two_blank = vote_rows([row("13", ["a2020"]), na_row("q", "?"), na_row("q", "?")])
    assert two_blank["answer_value"] == "is_blank"  # most votes refused: abstain
    one_blank = vote_rows([na_row("q", "?"), row("13", ["a2020"]), row("13", ["a2020"])])
    assert one_blank["answer_value"] == "13"


def test_votes_are_cached_separately_and_sampled(tmp_path):
    from wattbot.answer import answer_question
    temps = []
    def llm(messages, temperature=0.0):
        temps.append(temperature)
        return json.dumps({"answerable": True, "answer": "4.3", "answer_value": "4.3",
                           "ref_ids": ["patterson2021"], "supporting_materials": "q", "explanation": "e"})
    cache = {}
    chunks = [{"doc_id": "patterson2021", "page": 1, "text": "4.3 tCO2e"}]
    row, _ = answer_question({"id": "q", "question": "GShard?", "answer_unit": "tCO2e"}, chunks, llm,
                             {"patterson2021": "u"}, cache, "m", n_votes=3)
    assert temps == [0.0, 0.7, 0.7] and len(cache) == 3 and row["answer_value"] == "4.3"
    answer_question({"id": "q", "question": "GShard?", "answer_unit": "tCO2e"}, chunks, llm,
                    {"patterson2021": "u"}, cache, "m", n_votes=3)
    assert len(temps) == 3  # second run fully cached


def test_download_reuses_cached_copies(tmp_path):
    from wattbot.docs import find_cached_pdfs
    meta = pd.DataFrame([{"id": "a2020", "url": "u"}, {"id": "b2021", "url": "u"}])
    (tmp_path / "a2020.pdf").write_bytes(b"%PDF" + b"x" * 20_000)
    (tmp_path / "b2021.pdf").write_bytes(b"%PDF")  # too small: a failed download, fetch again
    assert set(find_cached_pdfs(meta, [tmp_path])) == {"a2020"}
    (tmp_path / "B_2021.pdf").write_bytes(b"%PDF" + b"x" * 20_000)  # renamed on upload
    assert set(find_cached_pdfs(meta, [tmp_path])) == {"a2020", "b2021"}


def test_arxiv_abstract_links_point_to_pdf():
    from wattbot.docs import pdf_url
    assert pdf_url("https://arxiv.org/abs/2601.19311v2") == "https://arxiv.org/pdf/2601.19311v2"
    assert pdf_url("https://www.gao.gov/assets/gao-25-107172.pdf") == "https://www.gao.gov/assets/gao-25-107172.pdf"


def test_chunk_cache_parses_only_new_documents(tmp_path, monkeypatch):
    import wattbot.docs as docs
    parsed = []
    monkeypatch.setattr(docs, "extract_pages", lambda p: parsed.append(p) or [{"page": 1, "text": "some words here"}])
    meta = pd.DataFrame([{"id": "a2020", "title": "A"}, {"id": "b2021", "title": "B"}])
    first = docs.build_chunks(meta, {"a2020": "a.pdf", "b2021": None}, cache_path=tmp_path / "c.json")
    assert {c["doc_id"] for c in first} == {"a2020"} and parsed == ["a.pdf"]
    out = tmp_path / "merged.json"  # the first cache acts as a read-only dataset
    second = docs.build_chunks(meta, {"a2020": "a.pdf", "b2021": "b.pdf"}, cache_path=tmp_path / "c.json", out_path=out)
    assert {c["doc_id"] for c in second} == {"a2020", "b2021"} and parsed == ["a.pdf", "b.pdf"]
    assert out.exists()


def test_yes_no_questions_are_boolean():
    from wattbot.answer import question_type, typed_value
    yn = "In experiments measuring inference, was the relationship between runtime and energy nearly linear?"
    assert question_type(yn, "is_blank") == "true/false"
    assert question_type("A popular claim puts a chat query at ten times a search. Is that claim well supported?",
                         "is_blank") == "true/false"
    assert question_type("Which preemption mechanism consumes less energy, Recomputation or Swapping?",
                         "is_blank") == "name"
    assert question_type("Does the energy cost scale linearly or sublinearly?", "is_blank") == "name"  # a choice
    assert question_type("As counts grow, does the cost scale linearly, sublinearly, or superlinearly?",
                         "is_blank") == "name"
    assert question_type("Is the PUE above 1.5, in PUE?", "PUE") == "number"
    assert question_type("One study says X. Another says Y. True or False: the findings contradict.", "is_blank") \
        == "true/false"
    # v1 bug: a correct 0 was replaced by the answer text "False"
    assert typed_value({"answer_value": "0", "answer": "False"}, yn, "is_blank") == "0"
    assert typed_value({"answer_value": "Yes", "answer": "Yes, nearly linear."}, yn, "is_blank") == "1"
    assert typed_value({"answer_value": "0", "answer": "False."}, "What term is used?", "is_blank") == "0"


def test_gateway_and_legacy_endpoints():
    from wattbot.llm import endpoint
    assert endpoint("https://h", "system.ai.deepseek-v4-flash-0731") == \
        ("https://h/ai-gateway/mlflow/v1/chat/completions", {"model": "system.ai.deepseek-v4-flash-0731"})
    assert endpoint("https://h", "databricks-gpt-oss-120b") == \
        ("https://h/serving-endpoints/databricks-gpt-oss-120b/invocations", {})


def test_cache_merges_saved_dataset(tmp_path):
    from wattbot.run import load_cache
    (tmp_path / "in" / "ds").mkdir(parents=True)
    (tmp_path / "in" / "ds" / "llm_cache.json").write_text(json.dumps({"a": "old", "b": "old"}))
    (tmp_path / "c.json").write_text(json.dumps({"b": "new"}))
    assert load_cache(tmp_path / "c.json", [tmp_path / "in"]) == {"a": "old", "b": "new"}


def test_think_tags_are_removed():
    from wattbot.llm import chat_text
    r = {"choices": [{"message": {"content": "<think>try {a: 1}</think>\n{\"answer_value\": \"4.3\"}"}}]}
    assert chat_text(r) == '{"answer_value": "4.3"}'
    blocks = {"choices": [{"message": {"content": [{"type": "reasoning", "text": "x"}, {"type": "text", "text": "ok"}]}}]}
    assert chat_text(blocks) == "ok"


def test_unreadable_replies_are_retried_not_cached():
    from wattbot.answer import ask
    replies = iter(["cut off {\"answer", json.dumps({"answerable": True, "answer_value": "4.3", "ref_ids": ["a"]})])
    cache = {}
    reply = ask({"question": "q", "answer_unit": "MWh"}, [], lambda m, **k: next(replies), cache, "m")
    assert reply["answer_value"] == "4.3" and len(cache) == 1
    stale = {}
    from wattbot.answer import build_messages, cache_key
    stale[cache_key(build_messages("q", "MWh", []), "m")] = "garbage"
    reply = ask({"question": "q", "answer_unit": "MWh"}, [], lambda m, **k: json.dumps({"answer_value": "1"}), stale, "m")
    assert reply == {"answer_value": "1"}  # a bad cached reply is replaced


def test_rate_limit_waits_and_retries(monkeypatch):
    import wattbot.llm as L
    class R:
        def __init__(self, code): self.status_code, self.ok, self.headers, self.text = code, code == 200, {}, "x"
        def json(self): return {"choices": [{"message": {"content": "ready"}}]}
    seq = iter([R(429), R(429), R(200)])
    monkeypatch.setattr("requests.post", lambda *a, **k: next(seq))
    sleeps = []
    monkeypatch.setattr(L.time, "sleep", sleeps.append)
    assert L.make_databricks_llm("m", host="h", token="t")([]) == "ready" and sleeps == [15, 30]
    monkeypatch.setattr("requests.post", lambda *a, **k: R(400))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        L.make_databricks_llm("m", host="h", token="t")([])  # not retried
