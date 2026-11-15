"""Tests for the extraction of permissions, prohibitions and duties."""
import json

import pandas as pd
import pytest

from eurlex_ontology_generation.pipelines.ontology_generation import nodes as gen_nodes
from eurlex_ontology_generation.pipelines.ontology_generation import rules_pass
from eurlex_ontology_generation.pipelines.ontology_review.ontology_checks import (
    run_deterministic_checks,
)
from eurlex_ontology_generation.utils.changelog import load_events
from eurlex_ontology_generation.utils.deontic import classify, covered, modal_sentences
from eurlex_ontology_generation.utils.eurlex_structure import chunk_act
from eurlex_ontology_generation.utils.llm_client import LLMResult
from eurlex_ontology_generation.utils.ontology_cleanup import empty_ontology

ACT = (
    "DECISION (EU) 2016/9999 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL THE EUROPEAN PARLIAMENT "
    "AND THE COUNCIL OF THE EUROPEAN UNION, Having regard to the Treaty, Whereas: (1) The Union "
    "should support Jordan. (2) The Commission shall be informed. HAVE ADOPTED THIS DECISION: "
    "Article 1 1. The Commission shall inform the European Parliament of the disbursement. "
    "2. The Commission may suspend the disbursement of the assistance. 3. The borrowing operations "
    "shall not expose the Union to any exchange rate risk. Article 2 This Decision shall enter into "
    "force on the day of its publication. Done at Strasbourg, 14 December 2016."
)
CHUNKS = {c["section_type"]: c for c in chunk_act(ACT)}
ARTICLE = CHUNKS["article"]["chunk_text"]
RECITALS = CHUNKS["recital"]["chunk_text"]


@pytest.mark.parametrize("sentence, kind", [
    ("The Commission shall inform the Council.", "duties"),
    ("The Commission may suspend the disbursement.", "permissions"),
    ("The operations shall not expose the Union to any risk.", "prohibitions"),
    ("Member States may not grant aid.", "prohibitions"),
    ("Imports of the product are prohibited.", "prohibitions"),
    ("Operators are entitled to request a refund.", "permissions"),
    ("This Decision shall enter into force on the day of its publication.", None),
    ("It shall apply from 1 January 2017.", None),
    ("Article 5 of Regulation (EU) No 182/2011 shall apply.", None),
    ("That committee shall be a committee within the meaning of Regulation (EU) No 182/2011.", None),
    ("This Regulation shall be binding in its entirety and directly applicable in all Member States.", None),
])
def test_classify_deontic_sentences(sentence, kind):
    hit = classify(sentence)
    assert (hit[0] if hit else None) == kind


def test_modal_sentences_and_coverage():
    modal = modal_sentences(ARTICLE)
    assert [m["kind"] for m in modal] == ["duties", "permissions", "prohibitions"]
    sentence = modal[1]
    rule = {"action_label": "suspend_disbursement", "source": {"chunk_start": sentence["start"] + 4,
                                                               "chunk_end": sentence["end"]}}
    assert covered(sentence, [rule]) and not covered(modal[0], [rule])
    # rules without a located source: their verb in the sentence counts
    assert covered(modal[0], [{"action_label": "inform_parliament"}])


def test_review_flags_rules_the_text_states():
    _, findings = run_deterministic_checks(json.dumps(empty_ontology()), ARTICLE)
    missing = [f for f in findings if f["code"] == "modal_without_rule"]
    assert sorted(f["data"]["rule_kind"] for f in missing) == ["duties", "permissions", "prohibitions"]
    assert any(f["code"] == "rights_model_empty" and f["severity"] == "warning" for f in findings)
    assert all("Add to" in f["fix"] for f in missing)
    # recitals do not bind: an empty rights model is right there
    _, rec_findings = run_deterministic_checks(json.dumps(empty_ontology()), RECITALS)
    assert not [f for f in rec_findings if f["code"] in ("rights_model_empty", "modal_without_rule")]


RULES_ANSWER = {
    "permissions": [{"action": "suspend", "action_label": "suspend_disbursement", "party": "Commission",
                     "assigner": "", "asset": "Disbursement", "provision": "article", "constraints": [],
                     "evidence": "The Commission may suspend the disbursement of the assistance."}],
    "prohibitions": [{"action": "other", "action_label": "expose_union", "party": "BorrowingOperations",
                      "assigner": "", "asset": "", "provision": "article", "constraints": [],
                      "evidence": "The borrowing operations shall not expose the Union to any exchange rate risk."}],
    "duties": [{"action": "inform", "action_label": "inform_european_parliament", "party": "Commission",
                "assigner": "", "asset": "", "provision": "article", "constraints": [], "evidence": ""}],
}
CONFIG = {"enabled": True, "sections": ["article"], "max_sentences": 12,
          "system_prompt": "rules {actions}", "user_prompt_template": "{text}\n{sentences}"}


def test_rules_pass_adds_missing_rules_and_their_entities(monkeypatch):
    prompts = []

    def fake_chat(backend, model, system, user, temperature, **kw):
        prompts.append(user)
        return LLMResult(json.dumps(RULES_ANSWER), "stop", 0, 0)

    monkeypatch.setattr(rules_pass, "llm_chat", fake_chat)
    onto = empty_ontology()
    onto["classes"] = [{"name": "Commission", "type": "Party", "is_a": "EUInstitution", "kind": "individual",
                        "description": ""}]
    onto["rights_model"]["duties"] = [dict(RULES_ANSWER["duties"][0])]     # already extracted
    events = rules_pass.run_rules_pass(onto, ARTICLE, CONFIG, llm_backend={}, model="m", temperature=0)
    rm = onto["rights_model"]
    assert len(rm["permissions"]) == 1 and len(rm["prohibitions"]) == 1 and len(rm["duties"]) == 1
    # only the uncovered sentences are sent
    assert "[permissions]" in prompts[0] and "[prohibitions]" in prompts[0] and "[duties]" not in prompts[0]
    names = {c["name"] for c in onto["classes"]}
    assert {"Disbursement", "BorrowingOperations"} <= names
    codes = [e["code"] for e in events]
    assert codes.count("rule_added_by_rules_pass") == 2  # noqa: PLR2004
    assert {e["actor"] for e in events if e["code"] == "rule_added_by_rules_pass"} == {"llm_generator"}


def test_rules_pass_skips_recitals_and_disabled_config(monkeypatch):
    monkeypatch.setattr(rules_pass, "llm_chat", lambda *a, **k: pytest.fail("no LLM call expected"))
    assert rules_pass.run_rules_pass(empty_ontology(), RECITALS, CONFIG, llm_backend={}, model="m",
                                     temperature=0) == []
    assert rules_pass.run_rules_pass(empty_ontology(), ARTICLE, {**CONFIG, "enabled": False},
                                     llm_backend={}, model="m", temperature=0) == []


def test_generation_runs_the_rules_pass_and_keeps_provenance(monkeypatch):
    main = empty_ontology()
    main["classes"] = [{"name": "Commission", "type": "Party", "is_a": "EUInstitution", "kind": "individual",
                        "description": "", "evidence": "The Commission shall inform"}]
    monkeypatch.setattr(gen_nodes, "llm_chat", lambda *a, **k: LLMResult(json.dumps(main), "stop", 0, 0))
    monkeypatch.setattr(rules_pass, "llm_chat",
                        lambda *a, **k: LLMResult(json.dumps(RULES_ANSWER), "stop", 0, 0))
    chunk = CHUNKS["article"]
    chunks = pd.DataFrame([{"chunk_id": "32016D9999_2", "CELEX": "32016D9999", "chunk_text": chunk["chunk_text"],
                            "segments": json.dumps(chunk["segments"])}])
    out = gen_nodes.generate_partial_ontologies(
        chunks, 0, "m", 0.1, "sys", "{text}", True, {"max_workers": 1}, structured_output=False,
        rules_pass=CONFIG)
    onto = json.loads(out.loc[0, "ontology"])
    perm = onto["rights_model"]["permissions"][0]
    assert perm["source"]["match"] == "exact" and perm["source"]["paragraph"] == 2  # noqa: PLR2004
    assert ACT[perm["source"]["start"]:perm["source"]["end"]] == perm["evidence"]
    assert onto["rights_model"]["prohibitions"][0]["action_label"] == "expose_union"
    events = load_events(out.loc[0, "change_log"])
    assert any(e["code"] == "rule_added_by_rules_pass" and e["stage"] == "generation" for e in events)
