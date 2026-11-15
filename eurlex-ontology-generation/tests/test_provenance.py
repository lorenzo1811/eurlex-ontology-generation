"""Tests for traceability: source provenance of every element and the change log."""
import json

import pandas as pd
from rdflib import Literal
from rdflib.namespace import RDF

from eurlex_ontology_generation.pipelines.ontology_refinement.nodes import (
    refine_ontology,
)
from eurlex_ontology_generation.utils.changelog import (
    diff_ontologies,
    load_events,
    stamp,
)
from eurlex_ontology_generation.utils.eurlex_structure import chunk_act
from eurlex_ontology_generation.utils.ontology_cleanup import cleanup_ontology
from eurlex_ontology_generation.utils.ontology_merge_common.assembly import (
    assemble_merged_ontology,
)
from eurlex_ontology_generation.utils.ontology_merge_common.entity_resolution import (
    canonicalize_clusters,
    cluster_entities,
    extract_entities,
)
from eurlex_ontology_generation.utils.ontology_normalize import normalize_chunk_ontology
from eurlex_ontology_generation.utils.provenance import (
    attach_provenance,
    locate_quote,
    strip_provenance,
    unit_at,
)
from eurlex_ontology_generation.utils.rdf_export import OA, PROV, export_rdf, namespaces
from eurlex_ontology_generation.utils.rdf_validate import validate_graph

ACT = (
    "DECISION (EU) 2016/9999 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL THE EUROPEAN PARLIAMENT "
    "AND THE COUNCIL OF THE EUROPEAN UNION, Having regard to the proposal from the European "
    "Commission, Whereas: (1) The Fund supports workers. (2) Estonia submitted an application. "
    "HAVE ADOPTED THIS DECISION: Article 1 1. The Commission shall submit an annual report to the "
    "European Parliament by 30 June 2017. 2. Member States may request an advance of EUR 500 000. "
    "Article 2 This Decision shall enter into force on the day of its publication. "
    "Done at Strasbourg, 23 November 2016."
)
CHUNKS = {c["section_type"]: c for c in chunk_act(ACT)}
ART = CHUNKS["article"]


def _onto():
    return {
        "ontology_name": "X", "description": "",
        "odrl_alignment": {"core_entities": [], "extensions": []},
        "classes": [
            {"name": "European_Commission", "type": "Party", "is_a": "EUInstitution", "kind": "individual",
             "description": "", "evidence": "The Commission shall submit an annual report"},
            {"name": "AnnualReport", "type": "Asset", "is_a": "Document", "kind": "class", "description": ""},
        ],
        "relationships": [],
        "rights_model": {
            "permissions": [{"action": "request", "action_label": "request_advance", "party": "Member_States",
                             "asset": "", "provision": "article", "constraints": [],
                             "evidence": "member states MAY request an   advance"}],
            "prohibitions": [],
            "duties": [{"action": "submit", "action_label": "submit_annual_report", "party": "European_Commission",
                        "asset": "AnnualReport", "provision": "article",
                        "evidence": "The Commission shall submit an annual report to the European Parliament",
                        "constraints": [{"left_operand": "dateTime", "operator": "lteq",
                                         "right_operand": "30 June 2017"}]}],
        },
        "legal_patterns": [], "example_instances": [],
    }


def test_quotes_are_located_exactly_normalized_or_fuzzily():
    text = ART["chunk_text"]
    assert locate_quote("The Commission shall submit", text)[2] == "exact"
    assert locate_quote("the commission  SHALL submit", text)[2] == "normalized"
    assert locate_quote("Commission must submit an annual report to Parliament by 30 June 2017", text)[2] == "fuzzy"
    assert locate_quote("The Council adopts the budget of the agency", text) is None


def test_unit_at_finds_article_paragraph_and_recital():
    text = ART["chunk_text"]
    assert unit_at(text, text.index("The Commission shall"))["paragraph"] == 1
    assert unit_at(text, text.index("Member States may"))["paragraph"] == 2  # noqa: PLR2004
    assert unit_at(text, text.index("enter into force"))["article"] == "2"
    rec = CHUNKS["recital"]["chunk_text"]
    assert unit_at(rec, rec.index("Estonia"))["recital"] == 2  # noqa: PLR2004


def test_every_element_gets_a_source_with_act_offsets():
    onto = _onto()
    events = attach_provenance(onto, ART["chunk_text"], celex="32016D9999", chunk_id="32016D9999_2",
                               segments=ART["segments"])
    duty = onto["rights_model"]["duties"][0]
    perm = onto["rights_model"]["permissions"][0]
    assert duty["source"]["match"] == "exact" and duty["source"]["article"] == "1"
    assert perm["source"]["match"] == "normalized" and perm["source"]["paragraph"] == 2  # noqa: PLR2004
    for src in (duty["source"], perm["source"], duty["constraints"][0]["source"],
                onto["classes"][0]["source"]):
        assert ACT[src["start"]:src["end"]] == src["quote"]
        assert src["celex"] == "32016D9999" and src["chunk_id"] == "32016D9999_2"
    assert duty["constraints"][0]["source"]["quote"] == "30 June 2017"
    assert onto["classes"][1]["source"]["match"] == "inferred"      # no quote: found by its words
    assert events == []


def test_unverified_quotes_are_repaired_or_logged():
    onto = _onto()
    # a wrong quote, but the class name points to a sentence: the quote is replaced
    onto["classes"][0]["evidence"] = "The Council adopted a regulation on fisheries in 1990"
    # a wrong quote and nothing in the text: flagged
    onto["classes"].append({"name": "Fisheries_Fund", "evidence": "The fund for fisheries in 1990",
                            "type": "Asset", "is_a": "Fund", "kind": "individual", "description": ""})
    events = attach_provenance(onto, ART["chunk_text"])
    assert [e["code"] for e in events] == ["evidence_repaired", "evidence_unverified"]
    assert events[0]["before"].startswith("The Council")
    assert onto["classes"][0]["evidence"] == events[0]["after"] and "Commission" in events[0]["after"]


def test_normalisation_keeps_provenance_and_logs_before_after():
    onto = _onto()
    attach_provenance(onto, ART["chunk_text"], segments=ART["segments"])
    out, changes = normalize_chunk_ontology(json.loads(json.dumps(onto)), source_text=ART["chunk_text"])
    duty = out["rights_model"]["duties"][0]
    assert duty["evidence"].startswith("The Commission") and duty["source"]["match"] == "exact"
    assert duty["constraints"][0]["source"]["quote"] == "30 June 2017"
    assert all(e["actor"] == "normalizer" for e in changes)
    defined = [e for e in changes if e["code"] == "undefined_entity_defined"]
    assert defined and defined[0]["after"]["name"] == "Member_States"


def test_cleanup_events_have_before_after_and_target():
    onto = _onto()
    del onto["rights_model"]["duties"][0]["provision"]
    onto["rights_model"]["duties"].append(dict(onto["rights_model"]["duties"][0], action_label="levy_tax"))
    _, notes = cleanup_ontology(onto, ART["chunk_text"])
    by_code = {e["code"]: e for e in notes}
    assert by_code["rule_ungrounded_dropped"]["before"]["action_label"] == "levy_tax"
    # provision_defaulted is no longer silent: it says what was set, and where
    assert by_code["provision_defaulted"]["after"] == "article"
    assert by_code["provision_defaulted"]["target"].endswith(".provision")
    assert by_code["provision_defaulted"]["actor"] == "normalizer"


def test_diff_ontologies_reports_llm_changes():
    before = _onto()
    after = json.loads(json.dumps(before))
    after["classes"][1]["type"] = "Legal_extension"
    after["rights_model"]["permissions"] = []
    after["classes"].append({"name": "EuropeanParliament", "type": "Party", "is_a": "EUInstitution",
                             "kind": "individual", "description": ""})
    events = diff_ontologies(before, after, actor="llm_reviser", code_prefix="llm_revision")
    codes = sorted(e["code"] for e in events)
    assert codes == ["llm_revision_added", "llm_revision_modified", "llm_revision_removed"]
    modified = next(e for e in events if e["code"] == "llm_revision_modified")
    assert modified["before"] == {"type": "Asset"} and modified["after"] == {"type": "Legal_extension"}
    stamp(events, stage="revision", round=1, chunk_id="c1", celex="X")
    assert {e["round"] for e in events} == {1} and {e["actor"] for e in events} == {"llm_reviser"}
    assert load_events(json.dumps(events)) == events
    # provenance fields do not count as changes
    moved = json.loads(json.dumps(before))
    moved["classes"][0]["source"] = {"match": "fuzzy"}
    assert diff_ontologies(before, moved, actor="llm_reviser") == []


def test_strip_provenance_for_prompts():
    onto = _onto()
    attach_provenance(onto, ART["chunk_text"])
    stripped = json.loads(strip_provenance(json.dumps(onto)))
    assert "source" not in json.dumps(stripped) and stripped["classes"][0]["evidence"]


def _merged_from_chunks():
    onto = _onto()
    attach_provenance(onto, ART["chunk_text"], celex="32016D9999", chunk_id="32016D9999_2",
                      segments=ART["segments"])
    norm, _ = normalize_chunk_ontology(onto, source_text=ART["chunk_text"])
    rows = pd.DataFrame([{"chunk_id": "32016D9999_2", "CELEX": "32016D9999", "approved": True,
                          "ontology": json.dumps(norm)}])
    entities = extract_entities(rows)
    clusters = cluster_entities(entities, "none", 0.9, {"provider": "unavailable"})
    canonical, _ = canonicalize_clusters(clusters)
    merged, _ = assemble_merged_ontology(rows, canonical, "test")
    return merged


def test_merge_and_refinement_carry_provenance_to_the_turtle():
    merged = _merged_from_chunks()
    duty = merged["rights_model"]["duties"][0]
    assert duty["provenance"][0]["chunk_id"] == "32016D9999_2"
    assert duty["constraints"][0]["provenance"][0]["quote"] == "30 June 2017"

    refined, _ = refine_ontology(json.loads(json.dumps(merged)))
    r_duty = refined["rights_model"]["duties"][0]
    assert r_duty["provenance"][0]["article"] == "1"
    ents = {e["name"]: e for e in refined["classes"] + refined["individuals"]}
    assert ents["European_Commission"]["provenance"][0]["match"] == "exact"

    g, stats = export_rdf(refined)
    eo = namespaces()["eo"]
    evidence = list(g.subjects(RDF.type, eo.TextEvidence))
    assert evidence and stats["evidence_links"] >= 3  # noqa: PLR2004
    quotes = {str(o) for s in evidence for sel in g.objects(s, OA.hasSelector)
              for o in g.objects(sel, OA.exact)}
    assert "30 June 2017" in quotes
    assert (None, PROV.wasDerivedFrom, None) in g
    assert any(g.objects(None, OA.start))
    assert (None, eo.article, Literal("1")) in g
    ok, _, summary = validate_graph(g)
    assert ok, summary["results"]


def test_revision_logs_llm_and_normalizer_changes(tmp_path, monkeypatch):
    from eurlex_ontology_generation.pipelines.ontology_revision import (
        nodes as revision_nodes,
    )
    from eurlex_ontology_generation.utils.llm_client import LLMResult
    from eurlex_ontology_generation.utils.ontology_schema import EXPECTED_TOP_LEVEL_KEYS

    reviewed = _onto()
    attach_provenance(reviewed, ART["chunk_text"], segments=ART["segments"])
    revised = json.loads(strip_provenance(json.dumps(reviewed)))
    revised["rights_model"]["permissions"] = []               # the LLM drops a rule
    revised["classes"][1]["kind"] = "individual"              # ... and changes a class
    seen_prompts = []

    def fake_chat(backend, model, system, user, temperature, **kw):
        seen_prompts.append(user)
        return LLMResult(json.dumps(revised), "stop", 0, 0)

    monkeypatch.setattr(revision_nodes, "llm_chat", fake_chat)
    pd.DataFrame([{"batch_id": 0, "chunk_id": "32016D9999_2", "round": 0, "CELEX": "32016D9999",
                   "text": ART["chunk_text"], "ontology": json.dumps(reviewed), "review": "{}",
                   "revision_required": True, "segments": json.dumps(ART["segments"])}]
                 ).to_csv(tmp_path / "part_000.csv", index=False)
    out = revision_nodes.revise_ontologies(
        0, 1000, str(tmp_path), "m", 0.0, "sys", "{ontology}\n{text}\n{review}", False,
        EXPECTED_TOP_LEVEL_KEYS, 1000, 1000, {"max_workers": 1}, revision_round=1)
    assert '"source"' not in seen_prompts[0]                   # located spans are not sent to the LLM
    row = out.iloc[0]
    events = load_events(row["change_log"])
    llm = [e for e in events if e["actor"] == "llm_reviser"]
    assert {e["code"] for e in llm} == {"llm_revision_removed", "llm_revision_modified"}
    assert all(e["stage"] == "revision" and e["round"] == 1 and e["chunk_id"] == "32016D9999_2"
               for e in events)
    onto = json.loads(row["ontology"])
    assert onto["rights_model"]["duties"][0]["source"]["start"] == ACT.index("The Commission shall")
    assert json.loads(row["segments"]) == ART["segments"]


def test_shacl_warns_on_elements_without_evidence():
    merged = _merged_from_chunks()
    for c in merged["classes"]:
        c.pop("provenance", None)
    refined, _ = refine_ontology(merged)
    g, _ = export_rdf(refined)
    _, _, summary = validate_graph(g)
    warnings = [r for r in summary["results"] if r["severity"] == "Warning"]
    assert any("No text evidence" in r["message"] for r in warnings)
