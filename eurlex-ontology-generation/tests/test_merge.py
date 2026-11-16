"""Tests for the merge: concept coverage, no silent losses, change-log events.
Embeddings are unavailable here, so classes merge by exact key."""
import json

import pandas as pd
import pytest

from eurlex_ontology_generation.pipelines.ontology_merge_global.nodes import (
    merge_global,
)
from eurlex_ontology_generation.pipelines.ontology_merge_partition.nodes import (
    merge_partition,
    persist_partition_merge,
)
from eurlex_ontology_generation.pipelines.ontology_refinement.nodes import (
    refine_ontology,
)
from eurlex_ontology_generation.utils.ontology_cleanup import empty_ontology
from eurlex_ontology_generation.utils.ontology_merge_common.assembly import (
    check_coverage,
)
from eurlex_ontology_generation.utils.ontology_merge_common.entity_resolution import (
    class_disposition,
    constraint_drop_reason,
    group_key,
    legal_act_short,
)

NO_EMBEDDINGS = {"provider": "unavailable"}


@pytest.mark.parametrize(("name", "description", "expected"), [
    ("11 October 2016", "", ("drop", "date_value")),
    ("EUR 55 million", "", ("drop", "amount_value")),
    ("Article 5", "", ("drop", "provision_reference")),
    ("AuditDuty", "", ("drop", "metamodel_leakage")),
    ("Decision (EU) 2016/2371", "", ("individual", "legal_act_reference")),
    ("Regulation1822011", "", ("individual", "legal_act_reference")),
    ("Eesti Energia AS", "", ("individual", "organisation_name")),
    ("Nitrofert", "A company in Estonia.", ("individual", "organisation_name")),
    ("EGF/2016/003", "", ("individual", "instance_name")),
    ("MacroFinancialAssistance", "", ("class", None)),
])
def test_class_disposition(name, description, expected):
    assert class_disposition(name, description) == expected


def test_legal_act_titles_share_one_key():
    long = "DECISION (EU) 2016/1858 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL of 11 October 2016"
    assert legal_act_short(long) == "Decision (EU) 2016/1858"
    assert group_key("class", long) == group_key("class", "Decision (EU) 2016/1858")
    assert group_key("class", "Decision (EU) 2016/2371") != group_key("class", "Decision (EU) 2016/2099")


def test_constraint_drop_reasons():
    assert constraint_drop_reason({"left_operand": "dateTime", "operator": "lteq", "right_operand": "2017"}) is None
    assert constraint_drop_reason({"left_operand": "dateTime", "right_operand": "unknown"}) == "constraint_placeholder"
    assert constraint_drop_reason({"name": "deadline", "value": "30 June 2017"}) == "constraint_instance_value"


def _cls(name, type_="Party", is_a="EUInstitution", kind="class", description=""):
    return {"name": name, "type": type_, "is_a": is_a, "kind": kind, "description": description}


def _row(chunk_id, classes, duties=(), permissions=()):
    onto = empty_ontology()
    onto["classes"] = list(classes)
    onto["rights_model"]["duties"] = list(duties)
    onto["rights_model"]["permissions"] = list(permissions)
    return {"chunk_id": chunk_id, "CELEX": chunk_id.split("_")[0], "approved": True,
            "ontology": json.dumps(onto)}


def _rows():
    return pd.DataFrame([
        _row("32016D2099_1", [
            _cls("Eesti Energia AS", "Asset", "LegalAct", "individual", "A company in Estonia."),
            _cls("11 October 2016", "Legal_extension", "Concept"),
            _cls("EGF", "Asset", "Fund"),
            _cls("Member State"),
        ], duties=[
            {"action": "inform", "action_label": "inform_commission", "party": "Member State",
             "asset": "EGF", "provision": "article",
             "constraints": [{"name": "deadline", "value": "30 June 2017"}]},
            # 'Commission' is not declared in this chunk: it resolves to the alias canonical
            {"action": "report", "action_label": "report_to_parliament", "party": "Commission",
             "asset": "30 June 2017", "provision": "article"},
        ]),
        _row("32016D2371_3", [
            _cls("DECISION (EU) 2016/2371 OF THE EUROPEAN PARLIAMENT", "Asset", "LegalAct"),
            _cls("EGF", "Asset", "Fund"),
            _cls("Member State"),
        ], duties=[
            {"action": "inform", "action_label": "notify_commission", "party": "Member State",
             "asset": "EGF", "provision": "article"},
        ]),
    ])


def test_merge_maps_every_concept_or_gives_a_reason(tmp_path):
    merged, provenance, _ = merge_partition(_rows(), 0, "none", 0.86, NO_EMBEDDINGS)
    coverage = provenance["merge_coverage"]
    assert coverage["silent_losses"] == []
    assert coverage["dropped_by_reason"] == {"date_value": 1}
    by_name = {c["name"]: c for c in merged["classes"]}

    # companies and acts become individuals with the upper class their name says
    company = by_name["Eesti_Energia_AS"]
    assert (company["kind"], company["is_a"], company["type"]) == ("individual", "Undertaking", "Party")
    act = by_name["Decision_(EU)_2016/2371"]
    assert (act["kind"], act["is_a"]) == ("individual", "LegalAct")

    # the alias canonical used as a party is defined, not lost with its rule
    assert by_name["European_Commission"]["defined_by_merge"] == "class_defined_from_alias"
    duties = merged["rights_model"]["duties"]
    report = next(d for d in duties if d["action"] == "report")
    assert report["party"] == "European_Commission" and report.get("asset") is None

    # identical rules from two chunks keep both labels
    inform = next(d for d in duties if d["action"] == "inform")
    assert inform["action_labels"] == ["inform_commission", "notify_commission"]
    assert inform["support"]["chunks"] == 2  # noqa: PLR2004

    codes = {e["code"] for e in provenance["events"]}
    assert {"concept_dropped", "class_merged_into", "rule_merged", "rule_asset_cleared",
            "constraint_dropped", "individual_from_name"} <= codes
    assert all(e["actor"] == "merger" and e["stage"] == "merge" for e in provenance["events"])
    assert coverage["rules"]["input"] == 3  # noqa: PLR2004
    assert coverage["rules"]["asset_cleared"] == 1

    # the full mapping goes to the coverage file, the provenance keeps a summary
    persist_partition_merge(merged, provenance, pd.DataFrame(), 0, str(tmp_path / "m"),
                            str(tmp_path / "p"), str(tmp_path / "c"), str(tmp_path / "cov"))
    saved = json.loads((tmp_path / "cov" / "part_000.json").read_text(encoding="utf-8"))
    reasons = {m["name"]: m["reason"] for m in saved["concept_mapping"]}
    assert reasons["11 October 2016"] == "date_value"
    stored = json.loads((tmp_path / "p" / "part_000.json").read_text(encoding="utf-8"))
    assert "concept_mapping" not in json.dumps(stored) and stored["merge_coverage_summary"]["dropped"] == 1


def test_global_merge_and_refinement_keep_everything():
    merged, _, _ = merge_partition(_rows(), 0, "none", 0.86, NO_EMBEDDINGS)
    parts = pd.DataFrame([{"chunk_id": "part_000", "CELEX": "part_000", "ontology": json.dumps(merged)}])
    glob, provenance, _ = merge_global(parts, "none", 0.88, NO_EMBEDDINGS)
    assert provenance["merge_coverage"]["silent_losses"] == []
    assert provenance["merge_coverage"]["coverage"] == 1.0
    assert {c["name"] for c in glob["classes"]} == {c["name"] for c in merged["classes"]}

    refined, _ = refine_ontology(json.loads(json.dumps(glob)))
    inform = next(d for d in refined["rights_model"]["duties"] if d["action"] == "notify")  # normalised
    assert inform["action_labels"] == ["inform_commission", "notify_commission"]
    individuals = {i["name"] for i in refined["individuals"]}
    assert {"Eesti_Energia_AS", "Decision_(EU)_2016/2371"} <= individuals


def test_silent_loss_fails_the_run():
    coverage = {"source_label": "partition_000",
                "silent_losses": [{"name": "X", "why": "canonical missing from the merged ontology"}]}
    with pytest.raises(ValueError, match="lost without a reason"):
        check_coverage(coverage, fail_on_silent_loss=True)
    check_coverage(coverage, fail_on_silent_loss=False)
