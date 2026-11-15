"""Change log: every change made to an ontology, as one event.

An event says what changed (`target`, `before`, `after`), who changed it
(`actor`), why (`code`, the reason code, plus a free `detail`), and where in the
pipeline (`stage`, `round`, `chunk_id`, `celex`, added by `stamp` when the
event is stored with its row).

Actors
  llm_generator LLM of the generation step (rules added by the rules-only pass)
  auto_check   deterministic review check (`ontology_review/ontology_checks.py`)
  llm_reviewer LLM judge of the review step
  llm_reviser  LLM of the revision step (changes found by `diff_ontologies`)
  normalizer   deterministic code: cleanup, normalisation, provenance, refinement
  merger       partition / global merge
  human        accepted curation decision

The event dicts keep the old `{"code", "detail"}` keys, so code that reads
`cleanup_notes` / `normalization_changes` keeps working.
"""
from __future__ import annotations

import json
from typing import Any

ACTORS = ("llm_generator", "auto_check", "llm_reviewer", "llm_reviser", "normalizer", "merger", "human")
RULE_KINDS = ("permissions", "prohibitions", "duties")
_MAX_VALUE_CHARS = 600


def _compact(value: Any) -> Any:
    """Values are stored inside CSV cells: keep them small."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_VALUE_CHARS else value[:_MAX_VALUE_CHARS] + "…"
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= _MAX_VALUE_CHARS:
        return value
    return text[:_MAX_VALUE_CHARS] + "…"


def change(code: str, detail: str = "", *, actor: str = "normalizer", target: str | None = None,  # noqa: PLR0913
           before: Any = None, after: Any = None) -> dict:
    """One change event (unstamped)."""
    if actor not in ACTORS:
        raise ValueError(f"unknown actor {actor!r}")
    event = {"code": code, "detail": detail, "actor": actor}
    if target is not None:
        event["target"] = target
    if before is not None:
        event["before"] = _compact(before)
    if after is not None:
        event["after"] = _compact(after)
    return event


def stamp(events: list[dict], *, stage: str, round: int | None = None,  # noqa: A002
          chunk_id: str | None = None, celex: str | None = None) -> list[dict]:
    """Add the pipeline context to events (in place) and return them.
    Context already present on an event is kept."""
    ctx = {"stage": stage, "round": round, "chunk_id": chunk_id, "celex": celex}
    for e in events:
        e.setdefault("actor", "normalizer")
        for k, v in ctx.items():
            if v is not None:
                e.setdefault(k, v)
    return events


def rule_target(kind: str, rule: dict) -> str:
    label = rule.get("action_label") or rule.get("action") or "?"
    return f"rights_model.{kind}[{label} | {rule.get('party')} | {rule.get('asset') or ''}]"


def class_target(name: Any) -> str:
    return f"classes[{name}]"


# ------------------------------------------------------------------ diffs
def _strip(obj: dict, drop: tuple = ("source", "evidence")) -> dict:
    return {k: v for k, v in obj.items() if k not in drop}


def _index(items: Any, key) -> dict:
    out = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            out.setdefault(key(item), item)
    return out


def _rule_key(rule: dict) -> tuple:
    return (str(rule.get("action_label") or rule.get("action") or ""), rule.get("party"),
            rule.get("asset") or "")


_SECTIONS = {
    "classes": (lambda c: str(c.get("name")), class_target),
    "relationships": (lambda r: (r.get("source"), r.get("relation"), r.get("target")),
                      lambda k: f"relationships[{k[0]} {k[1]} {k[2]}]"),
    "example_instances": (lambda e: (e.get("entity"), e.get("example_value")),
                          lambda k: f"example_instances[{k[0]}: {k[1]}]"),
}


def diff_ontologies(before: dict | None, after: dict | None, *, actor: str,
                    code_prefix: str = "llm") -> list[dict]:
    """Element-level differences between two chunk ontologies, as events.

    Classes are matched by name, rules by (kind, action label, party, asset),
    relationships by triple, examples by (entity, value). Provenance fields are
    ignored. Codes: <prefix>_added, <prefix>_removed, <prefix>_modified."""
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    events: list[dict] = []

    def compare(old: dict, new: dict, target: str) -> None:
        for k in old.keys() - new.keys():
            events.append(change(f"{code_prefix}_removed", k, actor=actor, target=target,
                                 before=_strip(old[k]) if isinstance(old[k], dict) else old[k]))
        for k in new.keys() - old.keys():
            events.append(change(f"{code_prefix}_added", k, actor=actor, target=target,
                                 after=_strip(new[k]) if isinstance(new[k], dict) else new[k]))
        for k in old.keys() & new.keys():
            a, b = _strip(old[k]), _strip(new[k])
            if a != b:
                fields = sorted(f for f in a.keys() | b.keys() if a.get(f) != b.get(f))
                events.append(change(f"{code_prefix}_modified", ", ".join(fields), actor=actor,
                                     target=target,
                                     before={f: a.get(f) for f in fields},
                                     after={f: b.get(f) for f in fields}))

    for section, (key, fmt) in _SECTIONS.items():
        old = _index(before.get(section), key)
        new = _index(after.get(section), key)
        for k in sorted(old.keys() | new.keys(), key=str):
            sub_old = {str(k): old[k]} if k in old else {}
            sub_new = {str(k): new[k]} if k in new else {}
            compare(sub_old, sub_new, fmt(k))

    rm_old = before.get("rights_model") if isinstance(before.get("rights_model"), dict) else {}
    rm_new = after.get("rights_model") if isinstance(after.get("rights_model"), dict) else {}
    for kind in RULE_KINDS:
        old = _index(rm_old.get(kind), _rule_key)
        new = _index(rm_new.get(kind), _rule_key)
        for k in sorted(old.keys() | new.keys(), key=str):
            rule = old.get(k) or new.get(k)
            compare({"rule": old[k]} if k in old else {}, {"rule": new[k]} if k in new else {},
                    rule_target(kind, rule))
    # events carry the element key in `detail`; for rules it is the literal "rule"
    for e in events:
        if e["detail"] == "rule":
            e["detail"] = ""
    return events


def load_events(value: Any) -> list[dict]:
    """Events stored in a CSV cell (JSON list) -> list of dicts."""
    if isinstance(value, list):
        return [e for e in value if isinstance(e, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []
