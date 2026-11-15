"""Tests for the structural parsing and chunking of EUR-Lex acts."""
import json

import pandas as pd

from eurlex_ontology_generation.pipelines.eurlex_chunking.nodes import chunk_corpus
from eurlex_ontology_generation.utils.eurlex_structure import (
    chunk_act,
    chunk_to_act_offset,
    parse_act,
    section_of,
    window_chunks,
)
from eurlex_ontology_generation.utils.ontology_cleanup import apply_provision_evidence

# Decision (EU) 2016/2099 (batch 46), as it appears in the corpus: one line,
# footnotes at the end, inline footnote markers "(1)" and recitals "(1) ...".
ACT = (
    "1.12.2016 EN Official Journal of the European Union L 326/14 DECISION (EU) 2016/2099 OF THE "
    "EUROPEAN PARLIAMENT AND OF THE COUNCIL of 23 November 2016 on the mobilisation of the European "
    "Globalisation Adjustment Fund THE EUROPEAN PARLIAMENT AND THE COUNCIL OF THE EUROPEAN UNION, "
    "Having regard to Regulation (EU) No 1309/2013 of the European Parliament and of the Council (1), "
    "and in particular Article 15(4) thereof, Having regard to the proposal from the European "
    "Commission, Whereas: (1) The European Globalisation Adjustment Fund (EGF) aims to provide support "
    "for workers made redundant. (2) The EGF is not to exceed a maximum annual amount of EUR 150 million "
    "(2011 prices), as laid down in Article 12 of Council Regulation (EU, Euratom) No 1311/2013 (2). "
    "(3) On 11 May 2016, Estonia submitted an application to mobilise the EGF, in respect of "
    "redundancies in Eesti Energia AS, Nitrofert AS and Viru Keemia Grupp AS in Estonia. "
    "(4) In order to minimise the time taken to mobilise the EGF, this decision should apply from the "
    "date of its adoption, HAVE ADOPTED THIS DECISION: Article 1 For the general budget of the Union "
    "for the financial year 2016, the European Globalisation Adjustment Fund shall be mobilised to "
    "provide the amount of EUR 1 131 358 in commitment and payment appropriations. Article 2 This "
    "Decision shall enter into force on the day of its publication in the Official Journal of the "
    "European Union. It shall apply from 23 November 2016. Done at Strasbourg, 23 November 2016. "
    "For the European Parliament The President M. SCHULZ For the Council The President I. KORCOK "
    "(1) OJ L 347, 20.12.2013, p. 855. (2) Council Regulation (EU, Euratom) No 1311/2013 of "
    "2 December 2013 laying down the multiannual financial framework (OJ L 347, 20.12.2013, p. 884)."
)

PARAGRAPHS = (
    "COMMISSION REGULATION (EU) No 1/2020 THE EUROPEAN COMMISSION, Having regard to the Treaty, "
    "Whereas: (1) A recital. (2) Another recital. HAS ADOPTED THIS REGULATION: Article 1 Subject "
    "matter 1. Member States shall notify the Commission. 2. The Commission may publish the "
    "notifications referred to in paragraph 1. Article 2 1. Operators shall not export the products "
    "referred to in Article 1 of Regulation (EC) No 2/2019. 2. Article 5 of Regulation (EU) No "
    "182/2011 shall apply. Article 3 This Regulation shall enter into force on the twentieth day. "
    "Done at Brussels, 1 January 2020. For the Commission The President (1) OJ L 1, 1.1.2019, p. 1. "
    "ANNEX List of products 1. Product A. 2. Product B. (1) OJ L 2, 2.1.2019, p. 2."
)


def _body(chunk):
    return chunk["chunk_text"].split("\n", 1)[1]


def test_units_follow_the_act_structure():
    parsed = parse_act(ACT)
    kinds = [(u.kind, u.recital, u.article) for u in parsed.units]
    assert kinds == [("preamble", None, None), ("recital", 1, None), ("recital", 2, None),
                     ("recital", 3, None), ("recital", 4, None), ("article", None, "1"),
                     ("article", None, "2"), ("final", None, None)]
    assert set(parsed.footnotes) == {1, 2}
    # the inline marker "(1)" in the citations is not a recital
    assert parsed.units[1].start > ACT.index("Whereas") - 1


def test_chunks_drop_footnotes_and_signatures_and_keep_sections_apart():
    chunks = chunk_act(ACT, chunk_size=2000)
    labels = [c["section_label"] for c in chunks]
    assert labels == ["Preamble", "Recitals (1)–(4)", "Article 1, Article 2"]
    for c in chunks:
        assert "OJ L 347" not in c["chunk_text"]
        assert "SCHULZ" not in c["chunk_text"]
    assert section_of(chunks[1]["chunk_text"]) == "recital"
    assert section_of(chunks[2]["chunk_text"]) == "article"
    assert chunks[0]["footnotes"] == {"1": "OJ L 347, 20.12.2013, p. 855."}


def test_offsets_point_back_to_the_act_text():
    for text, size in ((ACT, 300), (PARAGRAPHS, 150)):
        for c in chunk_act(text, chunk_size=size):
            for a, b, pos in c["segments"]:
                assert c["chunk_text"][pos:pos + b - a] == text[a:b]
    c = chunk_act(ACT)[2]
    pos = c["chunk_text"].index("shall be mobilised")
    assert ACT[chunk_to_act_offset(pos, c["segments"]):].startswith("shall be mobilised")


def test_small_chunks_never_cut_words_or_mix_sections():
    for c in chunk_act(ACT, chunk_size=200):
        assert len(_body(c)) <= 200
        for a, b, _ in c["segments"]:
            assert not (a > 0 and ACT[a - 1].isalnum() and ACT[a].isalnum())
            assert not (b < len(ACT) and ACT[b - 1].isalnum() and ACT[b].isalnum())
    kinds = [c["section_type"] for c in chunk_act(ACT, chunk_size=200)]
    assert kinds == sorted(kinds, key=["preamble", "recital", "article"].index)
    assert any("(continued)" in c["section_label"] for c in chunk_act(ACT, chunk_size=200))


def test_paragraphs_cross_references_and_annex():
    parsed = parse_act(PARAGRAPHS)
    arts = [(u.article, u.paragraph) for u in parsed.units if u.kind == "article"]
    # "Article 1 of Regulation" and "Article 5 of Regulation" are references, not headings
    assert arts == [("1", 1), ("1", 2), ("2", 1), ("2", 2), ("3", None)]
    chunks = chunk_act(PARAGRAPHS, chunk_size=2000)
    labels = [c["section_label"] for c in chunks]
    assert labels == ["Preamble", "Recitals (1)–(2)", "Article 1(1)–(2), Article 2(1)–(2), Article 3",
                      "Annex"]
    annex = _body(chunks[-1])
    assert annex.startswith("ANNEX") and "OJ L" not in annex and "Product B." in annex
    assert [c["section_type"] for c in chunk_act(PARAGRAPHS, include_annexes=False)][-1] == "article"


def test_unnumbered_recitals_and_unstructured_text():
    old = ("Council Regulation of 1 May 1990 THE COUNCIL, Having regard to the Treaty, Whereas the "
           "market must be stabilised; whereas prices have fallen; HAS ADOPTED THIS REGULATION: "
           "Article 1 Aid shall be granted. Article 2 This Regulation shall enter into force.")
    rec = [u for u in parse_act(old).units if u.kind == "recital"]
    assert len(rec) == 2 and "prices have fallen" in old[rec[1].start:rec[1].end]
    plain = "The Parliament grants discharge. It notes the accounts. " * 60
    chunks = chunk_act(plain, chunk_size=500)
    assert {c["section_type"] for c in chunks} == {"text"} and chunks[0]["chunking"] == "sentences"
    assert all(len(_body(c)) <= 500 for c in chunks)


def test_window_chunks_cut_at_word_boundaries():
    chunks = window_chunks(ACT, 300, 50)
    for c in chunks:
        a, b = c["char_start"], c["char_end"]
        assert not (b < len(ACT) and ACT[b - 1].isalnum() and ACT[b].isalnum())
        assert not (a > 0 and ACT[a - 1].isalnum() and ACT[a].isalnum())
    assert chunks[-1]["char_end"] == len(ACT)


def test_chunk_corpus_node_writes_metadata_columns():
    corpus = pd.DataFrame([{"CELEX": "32016D2099", "Act_name": "x", "Act_type": "Decision",
                            "Subject_matter": "", "Date_document": "2016-11-23", "act_raw_text": ACT}])
    df = chunk_corpus(corpus, "act_raw_text", 2000, 200)
    assert df["chunk_id"].tolist() == ["32016D2099_0", "32016D2099_1", "32016D2099_2"]
    assert df.loc[2, "section_type"] == "article"
    assert json.loads(df.loc[2, "articles"]) == ["1", "2"]
    legacy = chunk_corpus(corpus, "act_raw_text", 500, 100, strategy="window")
    assert set(legacy["chunking"]) == {"window"}


def test_provision_comes_from_the_section_header():
    duty = {"action": "submit", "action_label": "mobilise_fund", "party": "EGF", "asset": "",
            "provision": "article", "constraints": []}
    recital_chunk = chunk_act(ACT)[1]["chunk_text"]
    onto = {"rights_model": {"permissions": [], "prohibitions": [], "duties": [dict(duty)]}}
    notes: list = []
    apply_provision_evidence(onto, recital_chunk, notes)
    assert onto["rights_model"]["duties"] == []
    assert [n["code"] for n in notes] == ["provision_corrected", "rule_from_recital_dropped"]

    article_chunk = chunk_act(ACT)[2]["chunk_text"]
    onto = {"rights_model": {"permissions": [], "prohibitions": [],
                             "duties": [dict(duty, provision="recital")]}}
    apply_provision_evidence(onto, article_chunk, notes := [])
    assert onto["rights_model"]["duties"][0]["provision"] == "article"
