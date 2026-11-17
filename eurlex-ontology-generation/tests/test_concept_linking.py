"""Tests for concept linking (utils/concept_linking.py). ReLiK, the encoder, the
Wikipedia API and the LLM are replaced by small fakes."""
import json
import re
import zlib

import numpy as np
import pandas as pd
from rdflib import URIRef
from rdflib.namespace import OWL, SKOS

from eurlex_ontology_generation.pipelines.ontology_refinement.nodes import (
    refine_ontology,
)
from eurlex_ontology_generation.utils.concept_linking import (
    ConceptLinker,
    LinkDecision,
    VocabularyIndex,
    WikidataSearch,
    apply_links,
    link_stats,
    load_skos_file,
    shares_content_word,
)
from eurlex_ontology_generation.utils.evaluation import (
    link_gold_metrics,
    link_sheet_rows,
)
from eurlex_ontology_generation.utils.rdf_export import WD, export_rdf
from eurlex_ontology_generation.utils.rdf_validate import validate_graph

RECORDS = [
    {"id": "http://eurovoc.europa.eu/1", "label": "macro-financial assistance", "source": "eurovoc",
     "alt": ["MFA"], "note": ""},
    {"id": "http://eurovoc.europa.eu/2", "label": "balance of payments", "source": "eurovoc", "alt": [], "note": ""},
    {"id": "http://eurovoc.europa.eu/3", "label": "public finance", "source": "eurovoc", "alt": [], "note": ""},
    {"id": "http://eurovoc.europa.eu/4", "label": "financial management", "source": "eurovoc", "alt": [], "note": ""},
    {"id": "http://publications.europa.eu/resource/authority/corporate-body/CBJ", "label": "Central Bank of Jordan",
     "source": "corporate_body", "alt": [], "note": ""},
]


def encode(texts):
    """Bag-of-words hashing encoder: deterministic, no model."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, zlib.crc32(w.encode()) % 64] += 1.0
    return out


def _index():
    return VocabularyIndex.build(RECORDS, encode, meta={"encoder": {"kind": "test"}})


def test_index_round_trip_and_exact_labels(tmp_path):
    index = _index()
    index.save(str(tmp_path / "ix"))
    loaded = VocabularyIndex.load(str(tmp_path / "ix"))
    assert len(loaded.records) == len(RECORDS) and loaded.meta["encoder"] == {"kind": "test"}
    assert [r["id"] for r in loaded.exact(["MacroFinancialAssistance"])] == ["http://eurovoc.europa.eu/1"]
    assert loaded.exact(["Central Bank of Jordan"], {"eurovoc"}) == []
    top = loaded.search(encode(["public finance management"])[0], 2)
    assert {r["label"] for r, _ in top} == {"public finance", "financial management"}


def _chooser(answer):
    calls = []

    def choose(concept, candidates):
        calls.append((concept, candidates))
        return answer(candidates) if callable(answer) else answer
    return choose, calls


def test_eurovoc_exact_label_then_llm_choice_with_prior():
    choose, calls = _chooser(LinkDecision(choice=1, relation="close", reason="r"))
    linker = ConceptLinker(index=_index(), encode=encode, choose=choose,
                           act_descriptors={"32016D2371": ["financial management"]}, k=2, prior_bonus=0.5)
    exact, _ = linker.eurovoc({"name": "MacroFinancialAssistance", "type": "Asset"})
    assert exact["method"] == "label_exact" and exact["relation"] == "exact" and not calls

    ev, cands = linker.eurovoc({"name": "PublicFinanceManagementSystems", "type": "Legal_extension",
                                "sources": ["32016D2371"]})
    assert cands[0]["label"] == "financial management"               # the act's descriptor wins
    assert ev["id"] == "http://eurovoc.europa.eu/4" and ev["method"] == "retriever_llm"
    assert ev["relation"] == "close" and ev["confidence"] == "low"
    assert calls[0][0]["descriptors"] == ["financial management"]

    # a corporate body is offered only to parties; a gazetteer IRI is kept as it is
    party, _ = linker.eurovoc({"name": "Central Bank of Jordan", "type": "Party"})
    assert party["source"] == "corporate_body"
    gaz, _ = linker.eurovoc({"name": "Jordan", "iri": "http://publications.europa.eu/resource/authority/country/JOR"})
    assert gaz["method"] == "gazetteer"


def test_llm_answer_must_name_a_candidate():
    choose, _ = _chooser(LinkDecision(choice=9, relation="exact"))
    linker = ConceptLinker(index=_index(), encode=encode, choose=choose, k=2)
    ev, cands = linker.eurovoc({"name": "PublicFinanceRules", "type": "Legal_extension"})
    assert ev is None and len(cands) == 2 and linker.stats["llm_none"] == 1  # noqa: PLR2004


def test_wikidata_from_relik_spans():
    chunks = {"32016D2371_4": "The Union shall make available macro-financial assistance to the "
                              "Hashemite Kingdom of Jordan through the Ministry of Finance."}
    spans = {"32016D2371_4": [
        {"start": 33, "end": 59, "text": "macro-financial assistance", "title": "Macro-financial assistance"},
        {"start": 101, "end": 120, "text": "Ministry of Finance", "title": "Ministry of Finance (Jordan)"}]}
    qids = {"Macro-financial assistance": "Q111", "Ministry of Finance (Jordan)": "Q222"}
    choose, calls = _chooser(LinkDecision(choice=1, relation="exact"))
    linker = ConceptLinker(qids=lambda titles: {t: qids.get(t) for t in titles}, choose=choose)
    prov = [{"chunk_id": "32016D2371_4"}]
    exact = linker.wikidata({"name": "MacroFinancialAssistance", "provenance": prov}, chunks, spans)
    assert (exact["id"], exact["method"], exact["confidence"]) == ("Q111", "relik_exact", "high") and not calls
    partial = linker.wikidata({"name": "JordanianMinistryOfFinance", "provenance": prov}, chunks, spans)
    assert partial["id"] == "Q222" and partial["method"] == "relik_llm"
    assert calls[0][1] == [{"id": "Q222", "label": "Ministry of Finance (Jordan)"}]
    gaz = linker.wikidata({"name": "Jordan", "wikidata": "Q810", "type_source": "gazetteer"}, chunks, spans)
    assert gaz["method"] == "gazetteer"


def _merged():
    def cls(name, type_, is_a, kind, desc=""):
        return {"name": name, "type": type_, "is_a": is_a, "kind": kind, "description": desc,
                "sources": ["32016D2371"], "support": {"chunks": 1, "documents": 1},
                "provenance": [{"celex": "32016D2371", "chunk_id": "32016D2371_4", "quote": f"the {name}"}]}
    return {"ontology_name": "t", "description": "", "classes": [
        cls("MacroFinancialAssistance", "Asset", "FinancialInstrument", "class", "Loans to third countries"),
        cls("JordanianMinistryOfFinance", "Party", "PublicAuthority", "individual"),
        cls("PublicFinanceManagementSystems", "Legal_extension", "Concept", "class")],
        "relationships": [], "rights_model": {"permissions": [], "prohibitions": [], "duties": []},
        "legal_patterns": [], "example_instances": []}


LINKS = {
    "MacroFinancialAssistance": {"wikidata": {"id": "Q111", "relation": "exact", "method": "relik_exact",
                                              "confidence": "high"},
                                 "eurovoc": {"id": "http://eurovoc.europa.eu/1", "label": "macro-financial assistance",
                                             "relation": "exact", "method": "label_exact", "confidence": "high",
                                             "source": "eurovoc"}},
    "JordanianMinistryOfFinance": {"wikidata": {"id": "Q222", "relation": "exact", "method": "relik_llm",
                                                "confidence": "medium"}, "eurovoc": None},
    "PublicFinanceManagementSystems": {"wikidata": None,
                                       "eurovoc": {"id": "http://eurovoc.europa.eu/4", "label": "financial management",
                                                   "relation": "close", "method": "retriever_llm",
                                                   "confidence": "low", "source": "eurovoc"}},
}


def test_refinement_applies_links_and_export_writes_them():
    refined, report = refine_ontology(_merged(), links=LINKS)
    ents = {e["name"]: e for e in refined["classes"] + refined["individuals"]}
    mfa = ents["MacroFinancialAssistance"]
    assert mfa["wikidata"] == "Q111" and mfa["eurovoc"]["id"].endswith("/1")
    assert mfa["link_method"] == {"wikidata": "relik_exact", "eurovoc": "label_exact"}
    assert mfa["link_confidence"] == "high"
    assert report["changes"]["concept_linked"] == 3  # noqa: PLR2004
    assert {e["actor"] for e in report["events"] if e["code"] == "concept_linked"} == {"normalizer", "llm_reviewer"}

    g, stats = export_rdf(refined)
    iri = {str(s).rsplit("/", 1)[-1]: s for s in g.subjects() if isinstance(s, URIRef)}
    assert (iri["JordanianMinistryOfFinance"], OWL.sameAs, WD["Q222"]) in g
    assert (iri["MacroFinancialAssistance"], SKOS.exactMatch, WD["Q111"]) in g
    assert (iri["PublicFinanceManagementSystems"], SKOS.closeMatch, URIRef("http://eurovoc.europa.eu/4")) in g
    assert stats["eurovoc_links"] == 2  # noqa: PLR2004
    _, _, shacl = validate_graph(g)
    assert shacl["conforms_without_violations"]

    s = link_stats(refined)
    assert s["concepts"] == 3 and s["linked"] == 3 and s["close_match"] == 1  # noqa: PLR2004


def test_gazetteer_identifiers_are_not_overwritten():
    record = {"name": "Jordan", "wikidata": "Q810"}
    apply_links(record, {"Jordan": {"wikidata": {"id": "Q999", "relation": "exact", "method": "relik_exact"},
                                    "eurovoc": None}})
    assert record["wikidata"] == "Q810" and "link_method" not in record


def test_link_sheet_and_precision(tmp_path):
    refined, _ = refine_ontology(_merged(), links=LINKS)
    rows = link_sheet_rows(refined, n=10, seed=1, unlinked_share=0.0)
    assert len(rows) == 4 and "method" not in rows[0]  # noqa: PLR2004
    verdict = {"Q111": "correct", "Q222": "wrong", "/1": "correct", "/4": "close"}
    for r in rows:
        r["judgement"] = next(v for k, v in verdict.items() if r["link_iri"].endswith(k))
    sheet = tmp_path / "links.csv"
    pd.DataFrame(rows).to_csv(sheet, index=False)
    m = link_gold_metrics(refined, str(sheet))
    assert m["judged_links"] == {"n": 4, "precision_strict": 50.0, "precision_lenient": 75.0}
    assert m["by_method"]["relik_llm"]["precision_strict"] == 0.0
    assert m["by_method"]["label_exact"]["precision_strict"] == 100.0  # noqa: PLR2004


def test_skos_file(tmp_path):
    ttl = tmp_path / "v.ttl"
    ttl.write_text("""@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
<http://eurovoc.europa.eu/1> a skos:Concept ; skos:inScheme <http://eurovoc.europa.eu/100141> ;
  skos:prefLabel "public finance"@en, "finances publiques"@fr ; skos:altLabel "state finances"@en .
<http://eurovoc.europa.eu/2> a skos:Concept ; skos:inScheme <http://eurovoc.europa.eu/100141> ;
  skos:prefLabel "old term"@en ; owl:deprecated true .
""", encoding="utf-8")
    recs = load_skos_file(str(ttl), "eurovoc", "http://eurovoc.europa.eu/100141")
    assert recs == [{"id": "http://eurovoc.europa.eu/1", "label": "public finance", "source": "eurovoc",
                     "alt": ["state finances"], "note": ""}]
    assert json.dumps(recs)


def test_wikidata_search_builds_candidates(tmp_path, monkeypatch):
    assert WikidataSearch.query_of("MacroFinancialAssistance") == "Macro Financial Assistance"
    assert WikidataSearch.query_of("EGF_Mobilization") == "EGF Mobilization"
    calls = []

    def fake_get(self, params):
        calls.append(params["action"])
        if params["action"] == "wbsearchentities":
            return {"search": [{"id": "Q8880", "label": "European Commission", "description": "EU executive"},
                               {"id": "Q1", "label": "Commission", "description": "other"}]}
        return {"entities": {"Q8880": {"sitelinks": {"enwiki": {"title": "European Commission"}}},
                             "Q1": {"sitelinks": {}}}}
    monkeypatch.setattr(WikidataSearch, "_get", fake_get)
    search = WikidataSearch(str(tmp_path / "s.json"))
    hits = search.search("European_Commission")
    assert [(h["qid"], h["title"]) for h in hits] == [("Q8880", "European Commission"), ("Q1", None)]
    search.search("European_Commission")                       # cached
    assert calls == ["wbsearchentities", "wbgetentities"]
    search.save()
    assert WikidataSearch(str(tmp_path / "s.json")).search("European Commission") == hits


def test_guards_against_loose_links():
    assert shares_content_word(["MultilateralDonors"], "multilateral aid")
    assert not shares_content_word(["Suspension"], "Structural Funds")
    assert not shares_content_word(["Decision_(EU)_2017/340"], "adequacy decision (EU)")   # 'decision' is weak
    index = VocabularyIndex.build([{"id": "x", "label": "COST", "source": "eurovoc", "alt": [], "note": ""}], encode)
    assert index.exact(["costs"]) == [] and index.exact(["COST"])[0]["id"] == "x"
    choose, _ = _chooser(LinkDecision(choice=1, relation="close"))
    linker = ConceptLinker(index=_index(), encode=encode, choose=choose, k=1)
    ev, _ = linker.eurovoc({"name": "Suspension", "type": "Legal_extension"})
    assert ev is None and linker.stats["llm_rejected_no_shared_word"] == 1
