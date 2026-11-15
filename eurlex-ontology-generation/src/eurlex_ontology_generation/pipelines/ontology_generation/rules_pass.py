"""Rules-only generation pass for article chunks.

A small model asked for a whole ontology at once tends to return classes and
an empty or duty-only rights model. This pass asks a second, narrow question:
here are the article sentences that state a permission, prohibition or duty
(found in code, utils/deontic.py), each with the list it belongs to; return the
rules. The answer is constrained to the rights-model schema, merged into the
generated ontology without duplicates, and goes through the same cleanup and
grounding as the rest (the caller runs cleanup afterwards).
"""
from __future__ import annotations

import json
import re

from ...utils.changelog import change, class_target, rule_target
from ...utils.deontic import covered, modal_sentences
from ...utils.entity_linking import load_gazetteer
from ...utils.eurlex_structure import PROVISION_OF_SECTION, section_of
from ...utils.llm_client import llm_chat
from ...utils.ontology_normalize import type_entity
from ...utils.ontology_schema import build_rules_model
from ...utils.vocabulary import load_vocabulary, render_prompt

RULE_KINDS = ("permissions", "prohibitions", "duties")


def _key(rule: dict) -> tuple:
    return (str(rule.get("action_label") or rule.get("action") or "").lower(), rule.get("party"),
            rule.get("asset") or "")


def _parse(content: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|```$", "", (content or "").strip(), flags=re.M)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group()) if m else {}
        except json.JSONDecodeError:
            data = {}
    if isinstance(data, dict) and isinstance(data.get("rights_model"), dict):
        data = data["rights_model"]
    return data if isinstance(data, dict) else {}


def run_rules_pass(onto: dict, chunk_text: str, config: dict, *, llm_backend: dict, model: str,  # noqa: PLR0913
                   temperature: float, vocabulary_path: str | None = None,
                   structured_output: bool = True, num_ctx: int | None = None,
                   num_predict: int | None = None) -> list[dict]:
    """Add to `onto` (in place) the rules of the uncovered deontic sentences.
    Returns change events (actor llm_generator / normalizer)."""
    if not config or not config.get("enabled"):
        return []
    section = section_of(chunk_text)
    if section not in config.get("sections", ["article"]):
        return []
    rm = onto.setdefault("rights_model", {})
    existing = [r for k in RULE_KINDS for r in rm.get(k) or [] if isinstance(r, dict)]
    sentences = [s for s in modal_sentences(chunk_text) if not covered(s, existing)]
    sentences = sentences[: int(config.get("max_sentences", 12))]
    if not sentences:
        return []

    listing = "\n".join(f"{i + 1}. [{s['kind']}] {s['sentence']}" for i, s in enumerate(sentences))
    system = render_prompt(config["system_prompt"], vocabulary_path)
    user = render_prompt(config["user_prompt_template"], vocabulary_path, text=chunk_text,
                         sentences=listing)
    response = llm_chat(llm_backend, model, system, user, temperature,
                        schema_model=build_rules_model(vocabulary_path) if structured_output else None,
                        json_mode=True, num_ctx=num_ctx, max_tokens=num_predict)
    answer = _parse(response.content)

    events: list[dict] = []
    vocab = load_vocabulary(vocabulary_path)
    gazetteer = load_gazetteer(None, vocabulary_path)
    defined = {c.get("name") for c in onto.get("classes") or [] if isinstance(c, dict)}
    keys = {_key(r) for r in existing}
    provision = PROVISION_OF_SECTION.get(section, "article")
    for kind in RULE_KINDS:
        for raw in answer.get(kind) or []:
            if not isinstance(raw, dict) or not raw.get("party") or _key(raw) in keys:
                continue
            rule = {**raw, "provision": provision}
            keys.add(_key(rule))
            rm.setdefault(kind, []).append(rule)
            events.append(change("rule_added_by_rules_pass", f"{kind}", actor="llm_generator",
                                 target=rule_target(kind, rule), after=rule))
            # parties and assets of the new rules must be defined classes
            for slot, core in (("party", "Party"), ("asset", "Asset")):
                name = rule.get(slot)
                if not isinstance(name, str) or not name.strip() or name in defined:
                    continue
                typed = type_entity(name, "", vocab, gazetteer, declared_type=core)
                entry = {"name": name, "type": typed["type"], "is_a": typed["is_a"],
                         "kind": typed["kind"], "description": "",
                         "evidence": rule.get("evidence", "")}
                onto.setdefault("classes", []).append(entry)
                defined.add(name)
                events.append(change("rule_entity_defined", f"{slot} of a rules-pass rule",
                                     target=class_target(name), after=entry))
    return events
