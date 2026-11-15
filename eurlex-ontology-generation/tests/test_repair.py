"""Tests for repair-instead-of-delete (utils/repair.py)."""
import json

import pandas as pd

from eurlex_ontology_generation.utils.changelog import REVIEW_QUEUE_CODES
from eurlex_ontology_generation.utils.ontology_cleanup import (
    action_ungrounded_reason,
    cleanup_ontology,
)
from eurlex_ontology_generation.utils.ontology_normalize import normalize_chunk_ontology
from eurlex_ontology_generation.utils.persist import persist_removed_items
from eurlex_ontology_generation.utils.repair import (
    Repairer,
    amounts_in,
    dates_in,
    spelling_variant,
)

TEXT = (
    "§ Article 3(1)–(4)\n"
    "1. The Commission and the European External Action Service shall monitor the fulfilment of the "
    "pre-condition. 2. The Commission shall inform the European Parliament and the Council of developments. "
    "3. Where the conditions are not met, the Commission shall temporarily suspend or cancel the "
    "disbursement of the assistance. 4. The Union shall provide EUR 200 million to Jordan by "
    "31 December 2018."
)


def _rule(label, action="other", **kw):
    return {"action": action, "action_label": label, "party": "Commission", "asset": "",
            "provision": "article", "constraints": [], **kw}


def _repair(rule, repairer=None):
    repairer = repairer or Repairer()
    return repairer.repair_action(rule, TEXT, action_ungrounded_reason(rule, TEXT) or "",
                                  action_ungrounded_reason)


def test_text_helpers():
    assert spelling_variant("fulfillment", TEXT) == "fulfilment"
    assert dates_in("by 31 December 2018 or 1.2.2019") == ["2018-12-31", "2019-02-01"]
    assert amounts_in("EUR 200 million and EUR 1 131 358") == [("200000000", "EUR"), ("1131358", "EUR")]


def test_action_repairs_spelling_synonym_and_sentence():
    rule, method = _repair(_rule("monitor_fulfillment", "monitor"))
    assert (rule["action_label"], method) == ("monitor_fulfilment", "spelling variant")

    rule, method = _repair(_rule("notify_european_parliament", "notify",
                                 evidence="The Commission shall inform the European Parliament"))
    assert rule["action_label"] == "inform_european_parliament" and method.startswith("synonym")
    assert rule["action"] == "notify"                     # 'inform' is a synonym of notify

    rule, method = _repair(_rule("grant_disbursement", "grant",
                                 evidence="the Commission shall temporarily suspend or cancel the disbursement"))
    assert (rule["action_label"], method) == ("suspend_disbursement", "rebuilt from the sentence")
    assert rule["action"] == "suspend"

    # a word of the text that only shares a prefix is not a synonym ('authorities' is not 'authorise')
    assert _repair(_rule("levy_tax", "other")) is None


def test_llm_correction_is_accepted_only_if_found_in_the_text():
    answers = iter([
        json.dumps({"decision": "correct", "value": "levy_customs_duty", "evidence": "invented sentence"}),
        json.dumps({"decision": "correct", "value": "provide_eur", "evidence": "The Union shall provide EUR 200 million"}),
    ])
    llm = lambda system, user, schema: next(answers)  # noqa: E731
    repairer = Repairer(llm=llm)
    assert _repair(_rule("grant_loan", "grant"), repairer) is None          # evidence not in the text
    rule, method = _repair(_rule("grant_loan", "grant"), repairer)
    assert method == "llm" and rule["action_label"] == "provide_eur"


def test_constraint_values_are_repaired_from_the_rule_sentence():
    repairer = Repairer()
    rule = _rule("provide_eur", "provide", evidence="The Union shall provide EUR 200 million to Jordan")
    date, method = repairer.repair_constraint(
        {"left_operand": "dateTime", "operator": "lteq", "right_operand": "2018-12-30", "unit": ""},
        {**rule, "evidence": "The Union shall provide EUR 200 million to Jordan by 31 December 2018."},
        TEXT, "date 2018-12-30 not in text")
    assert date["right_operand"] == "2018-12-31" and "date" in method
    amount, _ = repairer.repair_constraint(
        {"left_operand": "payAmount", "operator": "eq", "right_operand": "150000000", "unit": "EUR"},
        rule, TEXT, "value 150000000 not in text")
    assert amount["right_operand"] == "200000000" and amount["unit"] == "EUR"
    unit, _ = repairer.repair_constraint(
        {"left_operand": "payAmount", "operator": "eq", "right_operand": "200000000", "unit": "USD"},
        rule, TEXT, "currency USD not in text")
    assert unit["unit"] == "EUR"


def test_cleanup_and_normaliser_repair_before_removing():
    onto = {"classes": [{"name": "Commission", "type": "Party", "is_a": "EUInstitution", "kind": "individual",
                         "description": ""}],
            "rights_model": {"permissions": [], "prohibitions": [], "duties": [
                _rule("monitor_fulfillment", "monitor"),
                _rule("levy_customs_duty", "other"),
                _rule("provide_eur", "provide", evidence="The Union shall provide EUR 200 million to Jordan",
                      constraints=[{"left_operand": "dateTime", "operator": "lteq",
                                    "right_operand": "2018-12-01", "unit": ""}])]},
            "example_instances": [{"entity": "Commission", "example_value": "External-Action Service",
                                   "source_context": "x"}]}
    out, notes = normalize_chunk_ontology(onto, source_text=TEXT, repairer=Repairer())
    codes = [e["code"] for e in notes]
    assert "rule_action_repaired" in codes and "constraint_value_repaired" in codes
    assert "example_value_repaired" in codes
    assert codes.count("rule_ungrounded_dropped") == 1           # only the unsupported rule is removed
    labels = [r["action_label"] for r in out["rights_model"]["duties"]]
    assert "monitor_fulfilment" in labels and "levy_customs_duty" not in labels
    duty = next(r for r in out["rights_model"]["duties"] if r["action_label"] == "provide_eur")
    assert duty["constraints"][0]["right_operand"] == "2018-12-31"
    repaired = next(e for e in notes if e["code"] == "rule_action_repaired")
    assert repaired["before"]["action_label"] == "monitor_fulfillment" and repaired["actor"] == "normalizer"
    # without a repairer the old behaviour stays: removal
    _, plain = cleanup_ontology(onto, TEXT)
    assert [e["code"] for e in plain].count("rule_ungrounded_dropped") == 2  # noqa: PLR2004


def test_removed_items_go_to_a_review_queue(tmp_path):
    events = [{"code": "rule_ungrounded_dropped", "actor": "normalizer", "target": "t1", "before": {"a": 1},
               "stage": "generation", "round": 0, "chunk_id": "c1"},
              {"code": "rule_duplicate_dropped", "actor": "normalizer", "target": "t2", "stage": "generation",
               "round": 0, "chunk_id": "c1"}]
    rows = pd.DataFrame([{"chunk_id": "c1", "change_log": json.dumps(events)}])
    path = persist_removed_items(rows, str(tmp_path), 0)
    queue = pd.read_csv(path)
    assert queue["code"].tolist() == ["rule_ungrounded_dropped"] and queue["status"].tolist() == ["to_review"]
    assert "rule_duplicate_dropped" not in REVIEW_QUEUE_CODES
    persist_removed_items(rows, str(tmp_path), 0)                 # idempotent on re-runs
    assert len(pd.read_csv(path)) == 1
