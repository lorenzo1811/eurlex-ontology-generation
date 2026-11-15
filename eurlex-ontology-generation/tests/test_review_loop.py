"""Tests for the review/revision loop: missing content, blocking warnings,
unchanged revisions, revision history."""
import json

import pandas as pd
import pytest

from eurlex_ontology_generation.pipelines.ontology_merge_partition.nodes import (
    load_partition_revisions,
)
from eurlex_ontology_generation.pipelines.ontology_review import nodes as review_nodes
from eurlex_ontology_generation.pipelines.ontology_review.ontology_checks import (
    build_to_add,
    check_to_add,
    normalize_json_path,
    revision_brief,
    verify_weaknesses,
)
from eurlex_ontology_generation.pipelines.ontology_revision.nodes import (
    persist_ontology_revisions,
)
from eurlex_ontology_generation.utils.ontology_cleanup import empty_ontology

TEXT = ("§ Recitals (6)–(7)\n(6) The Memorandum of Understanding defines the policy conditions. "
        "(7) The Commission shall report to the European Parliament on the assistance.")


def _onto(*names):
    onto = empty_ontology()
    onto["classes"] = [{"name": n, "type": "Party", "is_a": "EUInstitution", "kind": "individual",
                        "description": ""} for n in names]
    return onto


def test_missing_content_claims_are_kept_and_verified():
    onto = _onto("Commission")
    weaknesses = [
        {"claim": "The ontology lacks the 'Memorandum of Understanding', a key concept.", "json_path": "",
         "quote": "", "suggested_fix": "add it"},
        {"claim": "The ontology does not capture the 'Commission'.", "json_path": "", "quote": "",
         "suggested_fix": ""},
        {"claim": "The ontology is missing the 'Syrian crisis'.", "json_path": "", "quote": "",
         "suggested_fix": ""},
    ]
    kept, discarded = verify_weaknesses(weaknesses, onto, json.dumps(onto), TEXT, [])
    assert [w["term"] for w in kept] == ["Memorandum of Understanding"] and kept[0]["kind"] == "missing"
    assert kept[0]["evidence"].startswith("(6) The Memorandum")
    reasons = [r for w in discarded for r in w["discard_reasons"]]
    assert reasons == ["missing concept is already in the ontology", "missing concept not in the source text"]


def test_paths_without_prefix_and_quotes_from_the_text():
    assert normalize_json_path("duties[0].action") == "rights_model.duties[0].action"
    assert normalize_json_path("$.classes[0].name") == "classes[0].name"
    onto = _onto("Commission")
    onto["rights_model"]["duties"] = [{"action": "notify", "action_label": "notify_council", "party": "Commission"}]
    w = {"claim": "The duty names the wrong addressee.", "json_path": "duties[0].action_label",
         "quote": "report to the European Parliament", "suggested_fix": "use inform_european_parliament"}
    kept, discarded = verify_weaknesses([w], onto, json.dumps(onto), TEXT, [])
    assert kept and kept[0]["json_path"] == "rights_model.duties[0].action_label" and not discarded


def test_to_add_and_its_check_after_the_revision():
    findings = [{"code": "modal_without_rule", "severity": "warning", "message": "m",
                 "data": {"rule_kind": "duties", "sentence": "The Commission shall report ..."}}]
    to_add = build_to_add(["Memorandum of Understanding"], [], findings, TEXT)
    assert [i["kind"] for i in to_add] == ["concept", "rule"]
    assert check_to_add(_onto("Commission"), to_add)[0]["code"] == "missing_not_addressed"
    assert check_to_add(_onto("Commission", "MemorandumOfUnderstanding"), to_add) == []


def test_revision_brief_keeps_only_what_the_reviser_needs():
    review = {"weaknesses": [{"claim": "c", "json_path": "classes[0]", "suggested_fix": "f", "quote": "q"}],
              "discarded_weaknesses": [{"claim": "noise"}], "strengths": ["s"], "overall_score": 3,
              "suggested_amendments": ["a"], "to_add": [{"kind": "concept", "term": "X"}],
              "hallucinations": [], "unsupported_concepts": []}
    brief = json.loads(revision_brief(json.dumps(review)))
    assert "discarded_weaknesses" not in brief and "strengths" not in brief and "overall_score" not in brief
    assert brief["to_add"] == [{"kind": "concept", "term": "X"}] and brief["weaknesses"][0]["claim"] == "c"


THRESHOLDS = {"min_overall_score": 1, "min_odrl_compliance": 1, "max_verified_weaknesses": 5,
              "max_warnings": 5, "count_by": "code", "blocking_codes": ["rule_from_recital"]}


def test_blocking_codes_block_approval():
    review = review_nodes.Review(overall_score=5, odrl_compliance=5, strengths=[], weaknesses=[],
                                 missing_concepts=[], unsupported_concepts=[], hallucinations=[])
    onto = _onto("Commission")
    warning = [{"code": "rule_from_recital", "severity": "warning", "message": "duty from a recital"}]
    _, approved = review_nodes._build_review(review, onto, json.dumps(onto), TEXT, warning, THRESHOLDS)
    assert approved is False
    other = [{"code": "party_far_from_action", "severity": "warning", "message": "x"}]
    review_dict, approved = review_nodes._build_review(review, onto, json.dumps(onto), TEXT, other, THRESHOLDS)
    assert approved is True and review_dict["blocking_findings"] == []


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_unchanged_revision_reuses_the_review_and_stalls(tmp_path, monkeypatch):
    onto = _onto("Commission")
    onto["classes"][0]["source"] = {"match": "exact"}         # provenance differs, content does not
    same = json.dumps(_onto("Commission"))
    changed = json.dumps(_onto("Commission", "Council"))
    _write(tmp_path / "reviews" / "part_000.csv", [
        {"batch_id": 0, "chunk_id": "a", "round": 0, "CELEX": "X", "ontology": json.dumps(onto),
         "review": json.dumps({"status": "ok", "to_add": []}), "status": "ok", "approved": False,
         "revision_required": True, "automatic_checks": "[]"},
        {"batch_id": 0, "chunk_id": "b", "round": 0, "CELEX": "X", "ontology": same,
         "review": json.dumps({"status": "ok"}), "status": "ok", "approved": False,
         "revision_required": True, "automatic_checks": "[]"}])
    _write(tmp_path / "revisions" / "part_000.csv", [
        {"batch_id": 0, "chunk_id": "a", "round": 1, "CELEX": "X", "text": TEXT, "ontology": same},
        {"batch_id": 0, "chunk_id": "b", "round": 1, "CELEX": "X", "text": TEXT, "ontology": changed}])
    judged = []
    empty_review = review_nodes.Review(overall_score=4, odrl_compliance=4, strengths=[], weaknesses=[],
                                       missing_concepts=[], unsupported_concepts=[], hallucinations=[])
    monkeypatch.setattr(review_nodes, "_call_judge", lambda **kw: (judged.append(1) or (empty_review, None)))
    out = review_nodes.review_ontologies(
        0, 1000, "unused", pd.DataFrame(), "m", 0.0, 1, 100, 100, 0, "sys",
        "{text}{ontology}{automatic_checks}", {**THRESHOLDS, "blocking_codes": []}, {"max_workers": 1},
        review_round=1, source="revisions", revisions_directory=str(tmp_path / "revisions"),
        reviews_directory=str(tmp_path / "reviews"))
    rows = out.set_index("chunk_id")
    assert rows.loc["a", "status"] == "stalled" and not rows.loc["a", "revision_required"]
    assert rows.loc["b", "status"] == "ok"
    assert len(judged) == 1                                    # only the changed chunk was judged


def test_revisions_keep_history_and_merge_reads_the_latest(tmp_path):
    rev_dir = tmp_path / "revisions"
    base = {"batch_id": 0, "chunk_id": "a", "CELEX": "X", "text": TEXT}
    persist_ontology_revisions(pd.DataFrame([{**base, "round": 1, "ontology": json.dumps(_onto("A"))}]),
                               0, 1000, str(rev_dir))
    persist_ontology_revisions(pd.DataFrame([{**base, "round": 2, "ontology": json.dumps(_onto("B"))}]),
                               0, 1000, str(rev_dir))
    stored = pd.read_csv(rev_dir / "part_000.csv")
    assert sorted(stored["round"]) == [1, 2]
    latest = load_partition_revisions(0, str(rev_dir))
    assert len(latest) == 1 and json.loads(latest.loc[0, "ontology"])["classes"][0]["name"] == "B"
    assert review_nodes.rows_of_round(stored, 1)["round"].tolist() == [1]


@pytest.mark.parametrize("claim", ["The ontology lacks X", "fails to capture X", "X is not represented"])
def test_missing_claim_patterns(claim):
    from eurlex_ontology_generation.pipelines.ontology_review.ontology_checks import (
        _MISSING_CLAIM,
    )
    assert _MISSING_CLAIM.search(claim)
