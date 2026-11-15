"""Deterministic checks and post-processing for the ontology review node.

Everything that can be verified mechanically is verified here, so the LLM
judge only has to handle the semantic part of the review.
"""

import json
import math
import re
from typing import Any

from ...utils.deontic import covered, modal_sentences
from ...utils.entity_linking import load_gazetteer
from ...utils.eurlex_structure import NON_BINDING, section_of
from ...utils.ontology_cleanup import action_ungrounded_reason, example_problem, party_far_from_action, rule_key
from ...utils.ontology_normalize import type_entity
from ...utils.vocabulary import OTHER_ACTION, load_vocabulary, split_identifier

# Keep this list aligned with the convention used in the ontology-generation prompt (see the system prompt of the judge)
ODRL_CORE_NAMES = {
    "Policy",
    "Rule",
    "Asset",
    "Party",
    "Action",
    "Permission",
    "Prohibition",
    "Duty",
    "Constraint",
}

# Generic party names: the generator filled a slot without finding a real party in the text
PLACEHOLDER_PARTIES = {
    "targetparty",
    "responsibleparty",
    "party",
    "unknownparty",
    "genericparty",
}

# Fields that hold modelling vocabulary, not statements taken from the text.
VOCABULARY_KEYS = {"relation", "pattern_name", "ontology_name", "description"}

# Finding codes that count against ODRL compliance.
ODRL_CODES = {
    "misclassified_core_class",   # legacy "ODRL_core" type
    "invalid_class_type",
    "metamodel_leakage",
    "gazetteer_type_mismatch",
    "rule_slot_type",
    "rights_model_empty",
    "modal_without_rule",
    "placeholder_party",
    "undefined_entities",
}
# Uncovered deontic sentences reported per chunk (one finding each).
MAX_MODAL_FINDINGS = 8

# Severity 'info' findings are reported (and feed the metrics) but never
# affect scores or approval: they describe vocabulary drift that the
# deterministic post-processor fixes on its own.
INFO = "info"

# Words that mark a name as an actor (Party) or as a domain concept
# (Legal_extension). Domain words win when both appear
# (e.g. EuropeanUnionSolidarityFund is a fund, not an actor).
_ACTOR_WORDS = {
    "parliament", "council", "commission", "union", "state", "states",
    "authority", "authorities", "agency", "bank", "court", "committee",
    "institution", "institutions", "government", "member",
}
_DOMAIN_WORDS = {
    "fund", "instrument", "decision", "regulation", "programme", "program",
    "measures", "year", "period", "budget", "appropriations", "heading",
}

# Pattern matching claims of empty structures
_EMPTY_CLAIM = re.compile(
    r"\bempty\b|\blacks? any\b|\bwithout any\b",
    re.IGNORECASE,
)
# Claim phrasing typical of what the automatic checks already say. Used to
# tell "this weakness just restates an automatic finding" (discard it) apart
# from "this weakness happens to quote the same name but makes a different
# point" (keep it) - e.g. "X is undefined" vs "X is assigned the wrong role".
_UNDEFINED_ENTITY_CLAIM = re.compile(
    r"not defined|undefined|not declared|no.{0,15}class|ambigu",
    re.IGNORECASE,
)
_MISCLASSIFIED_CLAIM = re.compile(
    r"odrl_core|legal_extension|incorrectly typed|should be (a |an )?(party|legal_extension)",
    re.IGNORECASE,
)
_PLACEHOLDER_CLAIM = re.compile(
    r"placeholder|generic party|not (a party )?named|not referenced in the (source )?text",
    re.IGNORECASE,
)
_UNGROUNDED_ACTION_CLAIM = re.compile(
    r"unsupported action|hallucinat|no support|does not appear|not (mentioned|present)",
    re.IGNORECASE,
)
_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_WORD = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")



# Normalize strings for fuzzy string comparison by lowercasing and stripping whitespace/separators
def _norm(value: Any) -> str:
    """Lowercase and drop whitespace, commas and underscores.

    'EUR 178,715,475' matches 'EUR 178 715 475' and
    '1_January_2019' matches '1 January 2019'.
    """
    return re.sub(r"[\s,_]+", "", str(value).lower())


# Helper returning a list filtered exclusively for dictionary elements
def _dicts(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


# Determine candidate entity type (Party vs Legal_extension) using keyword heuristics
def _guess_entity_type(name: str) -> str:
    # Heuristic: 'Party' for actors, 'Legal_extension' for everything else
    words = {w.lower() for w in _WORD.findall(str(name))}
    if words & _DOMAIN_WORDS:
        return "Legal_extension"
    if words & _ACTOR_WORDS:
        return "Party"
    return "Legal_extension"


# Construct a finding dictionary containing error code, human message, and optional fix information
def _finding(
    code: str,
    severity: str,
    message: str,
    fix: str | None = None,
    data: dict | None = None,
) -> dict:
    """'fix' is a human-readable amendment, 'data' its machine-readable form
    (for the deterministic correction step)."""
    finding = {"code": code, "severity": severity, "message": message}
    if fix:
        finding["fix"] = fix
    if data:
        finding["data"] = data
    return finding


_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]


"""
True if the value appears in the text, also as an ISO date
(2019-01-01 matches '1 January 2019' / 'January 1, 2019')
"""
def _value_in_text(value: Any, text_norm: str) -> bool:
    
    raw = str(value).strip()
    if _norm(raw) in text_norm:
        return True
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if match:
        year, month, day = match.groups()
        if 1 <= int(month) <= 12:
            name = _MONTHS[int(month) - 1]
            variants = [f"{int(day)} {name} {year}", f"{name} {int(day)} {year}"]
            return any(_norm(v) in text_norm for v in variants)
    return False


# Yield all permissions, prohibitions, and duties present in the rights model
def _iter_rules(onto: dict):
    rights_model = onto.get("rights_model")
    if not isinstance(rights_model, dict):
        return
    for kind in ("permissions", "prohibitions", "duties"):
        for rule in _dicts(rights_model.get(kind)):
            yield kind, rule


# Collect all entity names referenced across relationships, rules, patterns, and examples
def _collect_entity_references(onto: dict) -> set[str]:
    refs: set[Any] = set()

    for rel in _dicts(onto.get("relationships")):
        refs.update([rel.get("source"), rel.get("target")])

    for _, rule in _iter_rules(onto):
        refs.update([rule.get("asset"), rule.get("party"), rule.get("assigner")])

    for pattern in _dicts(onto.get("legal_patterns")):
        involved = pattern.get("entities_involved")
        if isinstance(involved, list):
            refs.update(involved)

    for instance in _dicts(onto.get("example_instances")):
        refs.add(instance.get("entity"))

    return {ref for ref in refs if isinstance(ref, str) and ref}


"""Parse JSON and perform deterministic checks on ODRL schema structure and text grounding.

Returns:
    Tuple of (parsed dictionary or None on JSON error, list of finding dicts).
"""
def run_deterministic_checks(
    ontology_json: str,
    text: str,
) -> tuple[dict | None, list[dict]]:

    try:
        onto = json.loads(ontology_json)
    except (TypeError, json.JSONDecodeError) as exc:
        return None, [
            _finding(
                "ontology_json_invalid",
                "error",
                f"Candidate ontology is not valid JSON: {exc}",
            )
        ]

    if not isinstance(onto, dict):
        return None, [
            _finding(
                "ontology_json_invalid",
                "error",
                "Candidate ontology is not a JSON object.",
            )
        ]

    findings: list[dict] = []
    text = text or ""
    text_norm = _norm(text)
    text_lower = text.lower()

    classes = _dicts(onto.get("classes"))

    # Names that count as "defined": classes + ODRL alignment entries
    defined = {c.get("name") for c in classes if c.get("name")}
    alignment = onto.get("odrl_alignment")
    if isinstance(alignment, dict):
        for key in ("core_entities", "extensions"):
            values = alignment.get(key)
            if isinstance(values, list):
                defined.update(v for v in values if isinstance(v, str))

    # 1. Rules the text states but the ontology misses (utils/deontic.py).
    #    Recitals and the preamble do not bind: an empty rights model is right there.
    section = section_of(text)
    rules = [r for _, r in _iter_rules(onto)]
    modal = modal_sentences(text) if section not in NON_BINDING else []
    missing = [s for s in modal if not covered(s, rules)]
    if not rules and section not in NON_BINDING:
        findings.append(
            _finding(
                "rights_model_empty",
                "warning" if modal else "info",
                "rights_model contains no permissions, prohibitions or duties"
                + (f", but the text has {len(modal)} sentence(s) stating one." if modal else "."),
                "Add the permissions, prohibitions or duties expressed in the text to rights_model.",
            )
        )
    for s in missing[:MAX_MODAL_FINDINGS]:
        singular = s["kind"][:-1] if s["kind"] != "duties" else "duty"
        findings.append(
            _finding(
                "modal_without_rule",
                "warning",
                f"The text states a {singular} ('{s['cue']}') that no rule covers: \"{s['sentence'][:300]}\"",
                f"Add to {s['kind']} the rule stated by: \"{s['sentence'][:300]}\"",
                data={"rule_kind": s["kind"], "cue": s["cue"], "sentence": s["sentence"],
                      "chunk_start": s["start"], "chunk_end": s["end"]},
            )
        )

    # 2. Entities referenced but never defined (placeholder parties are
    #    reported by their own finding, so they are not listed here)
    refs = _collect_entity_references(onto)
    placeholders = {r for r in refs if r.lower().replace("_", "") in PLACEHOLDER_PARTIES}
    undefined = sorted(refs - defined - placeholders)
    if undefined:
        typed = [{"name": n, "type": _guess_entity_type(n)} for n in undefined]
        findings.append(
            _finding(
                "undefined_entities",
                "warning",
                "Entities used in relationships/rights_model/patterns/examples "
                "but not defined in classes or odrl_alignment: "
                + ", ".join(undefined),
                "Define these classes: "
                + ", ".join(f"{t['name']} (type {t['type']})" for t in typed)
                + ".",
                data={"classes_to_add": typed},
            )
        )

    # 3. Class-level checks against the controlled vocabulary and gazetteer
    vocab = load_vocabulary()
    gazetteer = load_gazetteer()
    class_type = {}
    for cls in classes:
        name = cls.get("name")
        if not isinstance(name, str) or not name:
            continue
        declared = cls.get("type")
        class_type[name] = declared

        leak = vocab.metamodel_leakage(name)
        if leak:
            findings.append(_finding(
                "metamodel_leakage", "warning",
                f"Class {leak}: ODRL structure must be expressed as a rule, not as a class.",
                f"Remove class '{name}' and express it as a permission, prohibition or duty in rights_model.",
                data={"class": name},
            ))
            continue

        expected = type_entity(name, str(cls.get("description") or ""), vocab, gazetteer,
                               declared, cls.get("is_a"))
        if declared == "ODRL_core" and name not in ODRL_CORE_NAMES:
            findings.append(_finding(
                "misclassified_core_class", "warning",
                f"Class '{name}' is typed ODRL_core but is not an ODRL "
                f"core concept (expected {expected['type']}).",
                f"Change the type of class '{name}' to {expected['type']}.",
                data={"class": name, "new_type": expected["type"], "is_a": expected["is_a"]},
            ))
        elif declared not in vocab.core_types:
            findings.append(_finding(
                "invalid_class_type", "warning",
                f"Class '{name}' has type '{declared}', which is not one of {vocab.core_types} "
                f"(expected {expected['type']}).",
                f"Change the type of class '{name}' to {expected['type']}.",
                data={"class": name, "new_type": expected["type"], "is_a": expected["is_a"]},
            ))
        elif expected.get("type_source") == "gazetteer" and (
                declared != expected["type"] or cls.get("kind") == "class"):
            findings.append(_finding(
                "gazetteer_type_mismatch", "warning",
                f"'{name}' is a known {expected['is_a']} individual (type {expected['type']}), "
                f"but the ontology declares type {declared}, kind {cls.get('kind')}.",
                f"Set '{name}' to type {expected['type']}, is_a {expected['is_a']}, kind individual.",
                data={"class": name, "new_type": expected["type"], "is_a": expected["is_a"],
                      "kind": "individual"},
            ))
        if cls.get("is_a") is None or cls.get("kind") is None:
            findings.append(_finding(
                "missing_is_a_or_kind", INFO,
                f"Class '{name}' has no is_a/kind (legacy schema).",
            ))

    # 4. Rule-level checks: placeholders, ungrounded actions and values
    for kind, rule in _iter_rules(onto):
        party = rule.get("party")
        if isinstance(party, str) and party.lower().replace("_", "") in PLACEHOLDER_PARTIES:
            findings.append(
                _finding(
                    "placeholder_party",
                    "error",
                    f"{kind}: party '{party}' is a generic placeholder, "
                    "not a party named in the text.",
                    f"Replace placeholder party '{party}' with a party named "
                    "in the text, or remove the rule.",
                    data={"rule_kind": kind, "party": party, "action": rule.get("action")},
                )
            )

        asset = rule.get("asset")
        if isinstance(party, str) and class_type.get(party) not in (None, "Party", "RightsHolder", "ODRL_core"):
            findings.append(_finding(
                "rule_slot_type", "warning",
                f"{kind}: party '{party}' is typed {class_type.get(party)}, not Party.",
                f"Use a Party as the party of this rule, or retype '{party}'.",
                data={"rule_kind": kind, "party": party},
            ))
        if isinstance(asset, str) and class_type.get(asset) == "Party":
            findings.append(_finding(
                "rule_slot_type", "warning",
                f"{kind}: asset '{asset}' is typed Party; an asset is what the action applies to.",
                f"Use an Asset or Legal_extension as the asset of this rule, or retype '{asset}'.",
                data={"rule_kind": kind, "asset": asset},
            ))
        if rule.get("provision") == "recital" and kind in ("duties", "prohibitions"):
            findings.append(_finding(
                "rule_from_recital", "warning",
                f"{kind}: rule '{rule.get('action_label') or rule.get('action')}' comes from a recital, "
                "which is not binding.",
                "Remove duties and prohibitions taken from recitals, or ground them in an article.",
                data={"rule_kind": kind, "action": rule.get("action")},
            ))

        action = rule.get("action")
        if isinstance(action, str) and action not in vocab.actions:
            norm = vocab.normalize_action(action, rule.get("action_label"))
            if not norm["controlled"] or action == OTHER_ACTION:
                findings.append(_finding(
                    "uncontrolled_action", INFO,
                    f"{kind}: action '{rule.get('action_label') or action}' is not in the controlled action vocabulary.",
                ))
        # ground the wording the generator used, not the canonical verb: the verb and
        # every word of the object must appear in the text (a single shared word such as
        # "financial" is not enough)
        grounded_on = rule.get("action_label") or action
        if isinstance(grounded_on, str) and grounded_on != OTHER_ACTION:
            why = action_ungrounded_reason(rule, text)
            if why:
                findings.append(
                    _finding(
                        "ungrounded_action",
                        "error",
                        f"{kind}: action '{grounded_on}' has no support in the text ({why}).",
                        f"Remove action '{action}' or replace it with an "
                        "action expressed in the text.",
                        data={"rule_kind": kind, "action": action, "party": rule.get("party")},
                    )
                )

        for constraint in _dicts(rule.get("constraints")):
            if "left_operand" not in constraint:
                findings.append(_finding(
                    "unstructured_constraint", INFO,
                    f"{kind}: constraint '{constraint.get('name')}' is free text, not "
                    "left_operand/operator/right_operand.",
                ))
            value = constraint.get("right_operand", constraint.get("value"))
            if (
                isinstance(value, (str, int, float))
                and str(value).strip()
                and not _value_in_text(value, text_norm)
            ):
                findings.append(
                    _finding(
                        "ungrounded_value",
                        "warning",
                        f"{kind}: constraint '{constraint.get('name') or constraint.get('left_operand')}' has "
                        f"value '{value}' which does not appear in the text.",
                        f"Check constraint '{constraint.get('name') or constraint.get('left_operand')}': its "
                        "value must be taken from the text.",
                    )
                )

    # 5. Relations outside the controlled set
    for rel in _dicts(onto.get("relationships")):
        _, _, _, controlled = vocab.normalize_relation(rel.get("relation"), rel.get("source"), rel.get("target"))
        if not controlled:
            findings.append(_finding(
                "uncontrolled_relation", INFO,
                f"Relation '{rel.get('relation')}' is not in the controlled relation vocabulary.",
            ))

    # 6. Example instances that just copy their own context
    for instance in _dicts(onto.get("example_instances")):
        example = instance.get("example_value")
        context = instance.get("source_context")
        if example and context and _norm(example) == _norm(context):
            findings.append(
                _finding(
                    "example_equals_context",
                    "warning",
                    f"Example for entity '{instance.get('entity')}' has "
                    "example_value identical to source_context.",
                    f"For entity '{instance.get('entity')}' use a short, "
                    "specific example_value and keep the surrounding passage "
                    "in source_context.",
                )
            )

    # 7. Examples that do not quote the text (placeholder, paraphrase, value not in the text)
    for instance in _dicts(onto.get("example_instances")):
        problem = example_problem(instance, text)
        if problem:
            findings.append(
                _finding(
                    "example_not_in_text",
                    "warning",
                    f"Example for entity '{instance.get('entity')}' does not quote the text ({problem}).",
                    f"For entity '{instance.get('entity')}' copy example_value and source_context "
                    "verbatim from the text, or remove the example.",
                )
            )

    # 8. Identical rules repeated
    seen_rules: set[str] = set()
    for kind, rule in _iter_rules(onto):
        key = rule_key(rule)
        if key in seen_rules:
            findings.append(
                _finding(
                    "duplicate_rule",
                    "warning",
                    f"{kind}: rule '{rule.get('action_label') or rule.get('action')}' is repeated.",
                    "Remove the repeated rules.",
                )
            )
        seen_rules.add(key)

    # 9. Party mentioned nowhere near the action in the text (heuristic, warning only)
    for kind, rule in _iter_rules(onto):
        if party_far_from_action(rule, text):
            findings.append(
                _finding(
                    "party_far_from_action",
                    "warning",
                    f"{kind}: party '{rule.get('party')}' is not mentioned near "
                    f"'{rule.get('action_label') or rule.get('action')}' in the text.",
                    "Check who actually holds this rule in the text.",
                )
            )

    # Remove exact duplicates, keep order
    seen = set()
    unique = []
    for finding in findings:
        key = (finding["code"], finding["message"])
        if key not in seen:
            seen.add(key)
            unique.append(finding)

    return onto, unique


# Render findings for the judge prompt
def format_findings(findings: list[dict]) -> str:
    
    # info findings are fixed in code later; keep them out of the judge prompt
    shown = [f for f in findings if f["severity"] != INFO]
    if not shown:
        return "(none)"
    return "\n".join(f"- [{f['severity']}] {f['message']}" for f in shown)


# Resolve 'a.b[0].c' or 'a.b.0.c' inside a parsed JSON object
def resolve_path(obj: Any, path: str) -> tuple[bool, Any]:
    
    current = obj
    for name, index in _PATH_TOKEN.findall(path):
        try:
            if index:
                current = current[int(index)]
            elif name.isdigit() and isinstance(current, list):
                current = current[int(name)]
            else:
                current = current[name]
        except (KeyError, IndexError, TypeError):
            return False, None
    return True, current


"""
Split the judge's weaknesses into (grounded, discarded).

A weakness is kept only if:
    - its json_path exists in the ontology ('a.0.b' and 'a[0].b' accepted),
    - the last key of the path is not a modelling-vocabulary field,
    - it does not claim that a non-empty element is empty,
    - its quote appears in the value found at json_path (or, when json_path
    is empty, in the source text),
    - it is not a duplicate of an automatic finding.
"""
def verify_weaknesses(
    weaknesses: list[dict],
    onto: dict | None,
    ontology_json: str,
    text: str,
    findings: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    
    text_norm = _norm(text or "")
    findings_norm = _norm(" ".join(f["message"] for f in (findings or [])))
    kept, discarded = [], []

    for weakness in weaknesses:
        reasons = []
        claim = weakness.get("claim", "")
        path = normalize_json_path(weakness.get("json_path") or "")
        quote = (weakness.get("quote") or "").strip()
        quote_norm = _norm(quote)
        path_found = True
        value_norm = ""

        # "The ontology lacks X": content the text has and the ontology misses. It has
        # no json_path by nature; it is kept when X is in the text and not yet modelled.
        if _MISSING_CLAIM.search(claim):
            term = _claimed_term(claim) or quote
            why = _missing_term_problem(term, quote, onto, text_norm)
            if why:
                discarded.append({**weakness, "discard_reasons": [why]})
            else:
                kept.append({**weakness, "kind": "missing", "term": term,
                             "evidence": _sentence_with(term if _norm(term) in text_norm else quote, text)})
            continue

        if path:
            path_found, value = resolve_path(onto or {}, path)
            if not path_found:
                reasons.append(f"json_path not found in ontology: {path}")
            else:
                keys = [k for k in re.findall(r"[^.\[\]]+", path) if not k.isdigit()]
                if keys and keys[-1] in VOCABULARY_KEYS:
                    reasons.append(
                        "modelling vocabulary field (not checked against the text)"
                    )
                value_norm = _norm(json.dumps(value, ensure_ascii=False))
                if value not in (None, "", [], {}) and _EMPTY_CLAIM.search(claim):
                    reasons.append(f"claims '{path}' is empty but it is not")

        if path_found:
            # the quote may come from the ontology value or, for claims that the value
            # contradicts the text, from the source text (at least a few words of it)
            from_text = len(quote.split()) >= MIN_TEXT_QUOTE_WORDS and quote_norm in text_norm
            if not quote:
                reasons.append("no supporting quote")
            elif path and quote_norm not in value_norm and not from_text:
                reasons.append("quote not found in the value at json_path nor in the source text")
            elif not path and quote_norm not in text_norm:
                reasons.append("quote not found in the source text")

        # A quote matching automatic-finding text is only a real duplicate
        # if the claim itself restates that finding's point, not just because
        # the same entity/class name is mentioned in some other automatic
        # finding (e.g. a party-misassignment claim naming an entity that
        # undefined_entities also lists, for an unrelated reason).
        codes_hit = {
            f["code"]
            for f in (findings or [])
            if quote_norm and quote_norm in _norm(f["message"])
        }
        restates = (
            ("undefined_entities" in codes_hit and _UNDEFINED_ENTITY_CLAIM.search(claim))
            or ("misclassified_core_class" in codes_hit and _MISCLASSIFIED_CLAIM.search(claim))
            or ("placeholder_party" in codes_hit and _PLACEHOLDER_CLAIM.search(claim))
            or ("ungrounded_action" in codes_hit and _UNGROUNDED_ACTION_CLAIM.search(claim))
        )
        if not reasons and len(quote_norm) >= 4 and restates:
            reasons.append("duplicate of an automatic finding")
        if (
            not reasons
            and path.endswith("example_value")
            and any(f["code"] == "example_equals_context" for f in (findings or []))
        ):
            reasons.append("duplicate of an automatic finding")

        if reasons:
            discarded.append({**weakness, "discard_reasons": reasons})
        else:
            kept.append({**weakness, "json_path": path} if path else weakness)

    return kept, discarded


# ------------------------------------------------------------ missing content
MIN_TEXT_QUOTE_WORDS = 3
MAX_TO_ADD = 12
_MISSING_CLAIM = re.compile(
    r"\b(?:lacks?|lacking|missing|omits?|absent|"
    r"(?:does|do)\s+not\s+(?:capture|represent|include|define|model|cover|mention)|"
    r"fails?\s+to\s+(?:capture|represent|include|define|model|cover|mention)|"
    r"(?:is|are)\s+not\s+(?:captured|represented|included|defined|modell?ed|covered))\b", re.I)
_QUOTED_TERM = re.compile(r"[‘'\"“]([^’'\"”]{3,80})[’'\"”]")


def normalize_json_path(path: str) -> str:
    """'duties[0].action' -> 'rights_model.duties[0].action'; strips '$.' / 'ontology.'."""
    p = (path or "").strip().strip("`")
    p = re.sub(r"^(?:\$\.?|ontology\.)", "", p)
    if re.match(r"(?:permissions|prohibitions|duties)\b", p):
        p = "rights_model." + p
    return p


def _claimed_term(claim: str) -> str | None:
    m = _QUOTED_TERM.search(claim or "")
    return m.group(1).strip() if m else None


def _class_names(onto: dict | None) -> set[str]:
    return {_norm(c.get("name", "")) for c in _dicts((onto or {}).get("classes"))}


def _is_modelled(term: str, onto: dict | None) -> bool:
    """The term is already a class (same name, or all its words in one class name)."""
    t = _norm(term)
    if not t:
        return False
    words = {w for w in split_identifier(term) if len(w) >= 3}  # noqa: PLR2004
    for c in _dicts((onto or {}).get("classes")):
        name = _norm(c.get("name", ""))
        if t == name or t in name:
            return True
        if words and words <= {w for w in split_identifier(c.get("name", ""))}:
            return True
    return False


def _missing_term_problem(term: str, quote: str, onto: dict | None, text_norm: str) -> str | None:
    if not term:
        return "missing content without a term or quote"
    if _norm(term) not in text_norm and not (quote and _norm(quote) in text_norm):
        return "missing concept not in the source text"
    if _is_modelled(term, onto):
        return "missing concept is already in the ontology"
    return None


def _sentence_with(term: str, text: str) -> str:
    """The sentence of `text` that contains `term` (case-insensitive), or ''."""
    t = _norm(term)
    for sentence in re.split(r"(?<=[.;:])\s+|\n", text or ""):
        if t and t in _norm(sentence):
            return sentence.strip()[:400]
    return ""


def build_to_add(missing_concepts: list[str], grounded: list[dict], findings: list[dict],
                 text: str) -> list[dict]:
    """What the revision must add: concepts the reviewer found missing (verified),
    and article sentences that state a rule no rule covers (modal_without_rule)."""
    items, seen = [], set()
    for term in list(missing_concepts) + [w["term"] for w in grounded if w.get("kind") == "missing"]:
        key = _norm(term)
        if key and key not in seen:
            seen.add(key)
            items.append({"kind": "concept", "term": term, "evidence": _sentence_with(term, text)})
    for f in findings:
        if f["code"] == "modal_without_rule" and f.get("data"):
            items.append({"kind": "rule", "list": f["data"]["rule_kind"], "sentence": f["data"]["sentence"]})
    return items[:MAX_TO_ADD]


def check_to_add(onto: dict | None, to_add: list[dict]) -> list[dict]:
    """After a revision: concepts the revision had to add and did not.
    (Rule sentences are re-checked by modal_without_rule on the revised ontology.)"""
    findings = []
    for item in to_add or []:
        if item.get("kind") == "concept" and not _is_modelled(item.get("term", ""), onto):
            findings.append(_finding(
                "missing_not_addressed", "warning",
                f"The revision did not add the missing concept '{item['term']}'.",
                f"Add '{item['term']}' as a class grounded in: \"{item.get('evidence', '')[:200]}\"",
                data=item,
            ))
    return findings


def revision_brief(review: dict | str) -> str:
    """The part of a review the reviser needs: verified weaknesses, amendments,
    what to add, hallucinations. Discarded claims, scores and strengths stay out."""
    if isinstance(review, str):
        try:
            review = json.loads(review)
        except json.JSONDecodeError:
            return review
    if not isinstance(review, dict):
        return json.dumps(review, ensure_ascii=False)
    brief = {
        "weaknesses": [{k: w.get(k) for k in ("claim", "json_path", "suggested_fix") if w.get(k)}
                       for w in review.get("weaknesses") or [] if w.get("kind") != "missing"],
        "suggested_amendments": review.get("suggested_amendments") or [],
        "to_add": review.get("to_add") or [],
        "hallucinations": review.get("hallucinations") or [],
        "unsupported_concepts": review.get("unsupported_concepts") or [],
    }
    return json.dumps(brief, ensure_ascii=False)


"""
Drop 'missing' concepts that are already defined (classes or
odrl_alignment), are just part of an existing class name, or do not
appear in the source text
"""
def filter_missing_concepts(
    missing: list[str],
    onto: dict | None,
    text: str = "",
) -> list[str]:
    
    if not onto:
        return list(missing)

    class_names = {_norm(c.get("name", "")) for c in _dicts(onto.get("classes"))}
    defined = set(class_names)
    alignment = onto.get("odrl_alignment")
    if isinstance(alignment, dict):
        for key in ("core_entities", "extensions"):
            values = alignment.get(key)
            if isinstance(values, list):
                defined.update(_norm(v) for v in values if isinstance(v, str))

    text_norm = _norm(text)
    return [
        concept
        for concept in missing
        if _norm(concept) not in defined
        and not any(_norm(concept) in name for name in class_names)
        and (not text_norm or _norm(concept) in text_norm)
    ]


"""
For 'unsupported_concepts' / 'hallucinations' reported by the judge:
keep only terms that really are in the ontology AND have no support in
the text (none of their words appear)
"""
def filter_ungrounded_terms(
    terms: list[str],
    ontology_json: str,
    text: str,
) -> list[str]:
    
    onto_norm = _norm(ontology_json or "")
    text_lower = (text or "").lower()
    kept = []
    for term in terms:
        if _norm(term) not in onto_norm:
            continue
        stems = [w.lower()[:5] for w in _WORD.findall(term) if len(w) >= 4]
        if stems and any(stem in text_lower for stem in stems):
            continue
        kept.append(term)
    return kept


"""
Scores derived from evidence, not from the LLM's opinion.

overall: 5 minus up to 3 for errors and up to 2 for (warnings + grounded
weaknesses, two per point). odrl_compliance: 5 minus one per ODRL-related
finding (max 4). Provisional: tune together with approval_thresholds.
"""
def count_warnings(findings: list[dict], count_by: str = "code") -> int:
    """Warnings counted per distinct code (one root cause = one penalty) or per finding."""
    warnings = [f for f in findings if f["severity"] == "warning"]
    return len({f["code"] for f in warnings}) if count_by == "code" else len(warnings)


def compute_scores(findings: list[dict], n_grounded_weaknesses: int,
                   count_by: str = "code") -> tuple[int, int]:
    """count_by='code': ten mistyped classes are one problem (the generator
    misunderstood typing), not ten. 'finding' reproduces the original scoring."""
    errors = sum(1 for f in findings if f["severity"] == "error")
    warnings = count_warnings(findings, count_by)
    overall = 5 - min(3, errors) - min(2, math.ceil((warnings + n_grounded_weaknesses) / 2))
    odrl_findings = [f for f in findings if f["code"] in ODRL_CODES and f["severity"] != INFO]
    n_odrl = len({f["code"] for f in odrl_findings}) if count_by == "code" else len(odrl_findings)
    odrl = 5 - min(4, n_odrl)
    return max(1, overall), max(1, odrl)