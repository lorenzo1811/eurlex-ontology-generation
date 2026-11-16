"""Deterministic assembly of one merged ontology from many partial ontologies"""

import json
import re
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from eurlex_ontology_generation.utils.changelog import (
    change,
    class_target,
    rule_target,
    stamp,
)
from eurlex_ontology_generation.utils.ontology_merge_common.entity_resolution import (
    ALIAS_DISPLAY,
    DROP,
    INDIVIDUAL,
    UPPER_OF_REASON,
    build_resolver,
    celex_of,
    class_disposition,
    clean_constraint,
    constraint_drop_reason,
    group_key,
    is_composite_class,
    is_leaky_action,
    is_placeholder,
    merge_name,
    normalize_action,
    normalize_relation,
    split_entities,
)
from eurlex_ontology_generation.utils.provenance import evidence_of, merge_evidence
from eurlex_ontology_generation.utils.vocabulary import load_vocabulary

# Core entity types expected by the ontology model
VALID_CORE_TYPES = {"Asset", "Party", "RightsHolder", "Permission", "Prohibition", "Duty", "Constraint"}

# Limits used to keep the merged ontology compact and deterministic
MAX_MERGED_DESCRIPTIONS_PER_CLASS = 3
MAX_CONSTRAINTS_PER_RULE = 10
MAX_EXAMPLES_PER_ENTITY = 3
MIN_PATTERN_ENTITIES = 2
RULE_KINDS = ("permissions", "duties", "prohibitions")
GENERIC_RELATIONS = {"applies_to", "related_to", "relates_to", "involves", "involved_in",
                     "associated_with", "has"}
MIN_GENERIC_RELATION_SUPPORT = 2   # generic relations seen only once are noise
MAX_PATTERN_ENTITIES = 8
MAX_EXAMPLE_LEN = 120

# Regular expression to extract source counts from partition descriptions
_SOURCE_COUNT_RE = re.compile(r"from (\d+) source ontologies covering (\d+) EUR-Lex")


# Safely return a list of dictionaries from an untyped JSON value
def _dicts(value: Any) -> list[dict]:
    return [i for i in value if isinstance(i, dict)] if isinstance(value, list) else []


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "1.0")


# Chunks of support that come from review-approved ontologies. Partition-level
# objects carry it in support.approved_chunks; chunk-level rows carry the
# row's final review verdict.
def _approved_support(obj: dict, row_approved: bool, chunks: int) -> int:
    s = obj.get("support")
    if isinstance(s, dict) and "approved_chunks" in s:
        try:
            return int(s["approved_chunks"])
        except (TypeError, ValueError):
            return 0
    return chunks if row_approved else 0


def _constraint_key(c: dict) -> tuple:
    if "left_operand" in c:
        return (c["left_operand"], c.get("operator"), c.get("right_operand"))
    return (c.get("name"), c.get("value"))


# Provenance: CELEX numbers of the acts a class or rule comes from.
# Chunk-level rows contribute their own CELEX; partition-level objects carry
# `sources` from the lower merge. Lists longer than MAX_SOURCES keep the first
# MAX_SOURCES (sorted) plus the true count in support.documents.
_CELEX_RE = re.compile(r"^[0-9CE]\d{4}[A-Z]{1,2}\d+")
MAX_SOURCES = 2000
# Evidence items (CELEX, chunk, article/paragraph, act offsets, quote) kept per
# class, rule and constraint; `sources` keeps the full list of acts.
MAX_EVIDENCE = 10


def _sources(obj: dict, celex: str) -> set[str]:
    carried = obj.get("sources")
    if isinstance(carried, list) and carried:
        return {str(c) for c in carried}
    return {celex} if _CELEX_RE.match(celex or "") else set()


# Extract (chunk_count, document_count) support metrics from an entity dictionary
def _support(obj: dict) -> tuple[int, int]:
    s = obj.get("support")
    if isinstance(s, dict):
        try:
            return int(s.get("chunks", 1)), int(s.get("documents", 1))
        except (TypeError, ValueError):
            pass
    return 1, 1


# Why an example instance is not merged, or None when it is kept
def _example_drop_reason(entity: str | None, value: Any, ctx: str) -> str | None:  # noqa: PLR0911
    if not entity:
        return "example_entity_unresolved"
    if not isinstance(value, str) or is_placeholder(value) or is_placeholder(ctx):
        return "example_placeholder"
    low = ctx.lower()
    if "text fragment" in low or "throughout" in low or re.match(r"\s*article\s*\d", ctx, re.I):
        return "example_context_not_a_quote"
    if len(value) > MAX_EXAMPLE_LEN:
        return "example_too_long"
    if value[:1].islower():
        return "example_lowercase"
    if re.search(r"\((EU|EC|EEC)\)\s*\d|\d{4}/\d+", value):
        return "example_is_an_act_reference"
    return None


# Mapping statuses (see concept_mapping in the provenance and merge_coverage())
MAPPED, SPLIT, DROPPED = "mapped", "split", "dropped"
# reasons that mean a bug rather than a decision: a concept that passed the
# filters but did not reach the merged ontology
SILENT_REASONS = {"class_unresolved"}


# Merges multiple partial JSON ontologies into a single consolidated ontology
def assemble_merged_ontology(  # noqa: PLR0912, PLR0915
    ontologies: pd.DataFrame,
    canonical_map: dict[str, dict[str, str]],
    source_label: str,
) -> tuple[dict, dict]:
    """Returns (merged, provenance). The provenance holds:
    - `concept_mapping`: one row per class mention (chunk, name, type) with its
      canonical name, or the reason it was dropped;
    - `events`: change-log events of the merge (actor `merger`);
    - `dropped`: counts per reason."""
    vocab = load_vocabulary()

    # Build entity resolver to map raw strings to canonical names
    _resolve = build_resolver(canonical_map)
    declared = set(canonical_map.get("class", {}).values())
    known_display = set(ALIAS_DISPLAY.values())
    defined: dict[str, str] = {}      # canonical -> why it was defined by the merge
    ctx: dict[str, Any] = {}          # the row being merged (celex, row events)

    # Resolve a raw name to its canonical class name. Names declared as classes resolve;
    # gazetteer/alias canonicals ("Commission" -> European_Commission) and, for rule
    # slots, any acceptable name are defined by the merge instead of being lost.
    def resolve(name, define_as: str | None = None):
        r = _resolve(name)
        if r and r in declared:
            return r
        if r and r in known_display:
            _define(r, "class_defined_from_alias", name)
            return r
        if define_as and isinstance(name, str) and class_disposition(name)[0] != DROP:
            canonical = re.sub(r"\s+", "_", merge_name(name))
            _define(canonical, define_as, name)
            return canonical
        return None

    def _define(canonical: str, code: str, raw: str) -> None:
        if canonical in classes:
            return
        classes[canonical] = {"description": ""}
        class_chunks[canonical] += 1
        class_docs[canonical].add(ctx["celex"])
        class_sources[canonical] |= _sources({}, ctx["celex"])
        defined[canonical] = code
        mapping.append({"source_id": ctx["source_id"], "name": raw, "type": None,
                        "status": MAPPED, "canonical": canonical, "reason": code})
        ctx["events"].append(change(code, str(raw), actor="merger", target=class_target(canonical),
                                    after=canonical))

    action_map = canonical_map.get("action", {})

    # Data structures for accumulating resolved entities and support metrics
    classes: dict[str, dict] = {}
    type_votes: dict[str, Counter] = defaultdict(Counter)
    is_a_votes: dict[str, Counter] = defaultdict(Counter)
    kind_votes: dict[str, Counter] = defaultdict(Counter)
    forced: dict[str, dict] = {}               # individual names: kind (and is_a) set by the merge
    class_iri: dict[str, str] = {}
    class_approved: Counter = Counter()
    class_chunks: Counter = Counter()
    class_docs: dict[str, set] = defaultdict(set)
    class_docs_n: Counter = Counter()          # documents carried over from lower-level support
    class_sources: dict[str, set] = defaultdict(set)
    class_prov: dict[str, list] = defaultdict(list)    # evidence items (utils/provenance.py)
    raw_class_names: set[str] = set()
    relationships: dict[tuple, dict] = {}
    rules: dict[str, dict[tuple, dict]] = {k: {} for k in RULE_KINDS}
    patterns: dict[tuple, dict] = {}
    examples: dict[tuple, dict] = {}
    source_ids: set[str] = set()
    source_celex: set[str] = set()
    dropped: Counter = Counter()
    mapping: list[dict] = []
    events: list[dict] = []
    merged_pairs: set[tuple] = set()           # (raw, canonical) already logged
    rules_in = 0
    carried_sources = carried_docs = 0     # counts declared by lower-level (partition) ontologies

    # Phase 1: Ingest and aggregate raw input ontologies
    for _, row in ontologies.iterrows():
        try:
            onto = json.loads(row["ontology"])
        except (TypeError, json.JSONDecodeError):
            dropped["invalid_json"] += 1
            continue
        if not isinstance(onto, dict):
            continue

        # Extract source lineage and document CELEX identifier
        source_id = str(row["chunk_id"])
        raw_celex = str(row.get("CELEX", "") or "")
        celex = raw_celex if raw_celex and raw_celex != source_id else celex_of(source_id)
        source_ids.add(source_id)
        source_celex.add(celex)
        row_approved = _is_true(row.get("approved", False))
        row_events: list[dict] = []
        ctx.update(source_id=source_id, celex=celex, events=row_events)

        def drop(code: str, detail: str, target: str, before: Any = None) -> None:
            dropped[code] += 1
            row_events.append(change(code, detail, actor="merger", target=target, before=before))

        # Parse inherited source/document metadata if available
        m = _SOURCE_COUNT_RE.search(str(onto.get("description", "")))
        if m:
            carried_sources += int(m.group(1))
            carried_docs += int(m.group(2))
        else:
            carried_sources += 1
            carried_docs += 0

        # 1a: Process classes
        for cls in _dicts(onto.get("classes")):
            raw = cls.get("name")
            if isinstance(raw, str):
                raw_class_names.add(raw)
            description = (cls.get("description") or "").strip()
            mention = {"source_id": source_id, "name": raw, "type": cls.get("type")}

            # Values, provision references and placeholders are not concepts: dropped with a reason
            verdict, reason = class_disposition(raw, description)
            if verdict != DROP and is_composite_class(raw):
                parts = [p for p in (resolve(x) for x in split_entities(raw)) if p]
                mapping.append({**mention, "status": SPLIT, "canonical": parts,
                                "reason": "composite_of_known_entities"})
                dropped["class_composite_split"] += 1
                continue
            if verdict == DROP:
                mapping.append({**mention, "status": DROPPED, "canonical": None, "reason": reason})
                drop("concept_dropped", reason, class_target(raw), before=raw)
                continue
            name = resolve(raw)
            if not name:
                mapping.append({**mention, "status": DROPPED, "canonical": None, "reason": "class_unresolved"})
                drop("concept_dropped", "class_unresolved", class_target(raw), before=raw)
                continue
            mapping.append({**mention, "status": MAPPED, "canonical": name, "reason": reason})
            if name != raw and (raw, name) not in merged_pairs:
                merged_pairs.add((raw, name))
                row_events.append(change("class_merged_into", str(raw), actor="merger",
                                         target=class_target(raw), before=raw, after=name))
            if verdict == INDIVIDUAL:
                forced.setdefault(name, {"kind": "individual", "reason": reason,
                                         "is_a": UPPER_OF_REASON.get(reason)})

            # Update class statistics and type voting
            chunks, docs = _support(cls)
            class_chunks[name] += chunks
            class_docs[name].add(celex)
            class_docs_n[name] += max(docs - 1, 0)
            class_sources[name] |= _sources(cls, celex)
            if cls.get("type"):
                type_votes[name][cls["type"]] += chunks
            if cls.get("is_a"):
                is_a_votes[name][cls["is_a"]] += chunks
            if cls.get("kind"):
                kind_votes[name][cls["kind"]] += chunks
            if cls.get("iri"):
                class_iri.setdefault(name, cls["iri"])
            class_approved[name] += _approved_support(cls, row_approved, chunks)
            class_prov[name] = merge_evidence(class_prov[name], evidence_of(cls), MAX_EVIDENCE)

            # Merge class descriptions
            existing = classes.setdefault(name, {"description": ""})
            if description:
                pieces = [p for p in existing["description"].split(" | ") if p]
                if description not in pieces and len(pieces) < MAX_MERGED_DESCRIPTIONS_PER_CLASS:
                    pieces.append(description)
                    existing["description"] = " | ".join(pieces)

        # 1b: Process relationships
        for rel in _dicts(onto.get("relationships")):
            src, tgt = resolve(rel.get("source")), resolve(rel.get("target"))
            relation = normalize_relation(rel.get("relation"))
            triple = f"relationships[{rel.get('source')} {rel.get('relation')} {rel.get('target')}]"

            # Filter unresolved entities or invalid self-loops
            if not (src and tgt and relation):
                drop("relationship_unresolved", "source, target or relation unresolved", triple)
                continue
            if src == tgt:
                drop("relationship_self_loop", src, triple)
                continue

            # Deduplicate by triple (source, relation, target) and increment support
            entry = relationships.setdefault(
                (src, relation, tgt),
                {"source": src, "relation": relation, "target": tgt,
                 "description": rel.get("description", "") or "", "support": 0},
            )
            entry["support"] += _support(rel)[0]

        # 1c: Process rights model
        rm = onto.get("rights_model")
        if isinstance(rm, dict):
            for kind in RULE_KINDS:
                for rule in _dicts(rm.get(kind)):
                    rules_in += 1
                    target = rule_target(kind, rule)

                    # Sanitize and resolve rule action
                    raw_action = rule.get("action")
                    if not isinstance(raw_action, str) or not raw_action.strip():
                        drop("rule_dropped", "rule_without_action", target)
                        continue
                    action = normalize_action(raw_action)
                    if not action or is_leaky_action(action):
                        drop("rule_leaky_action", str(raw_action), target)
                        continue
                    action = action_map.get(action, action)
                    action_object = normalize_action(rule.get("action_object") or "") or None

                    # Resolve parties and assets: a party named in a rule is a concept of the text
                    raw_party = rule.get("party")
                    parties = [resolve(p, "class_defined_from_rule") for p in split_entities(raw_party)] \
                        if isinstance(raw_party, str) else []
                    parties = [p for p in dict.fromkeys(parties) if p]
                    if not parties:
                        drop("rule_unresolved_party", str(raw_party), target,
                             before={k: rule.get(k) for k in ("action", "party", "asset")})
                        continue
                    asset = resolve(rule.get("asset"), "class_defined_from_rule") if rule.get("asset") else None
                    if rule.get("asset") and not asset:
                        # kept without its target (odrl:target is optional), like the refinement does
                        drop("rule_asset_cleared", str(rule.get("asset")), target, before=rule.get("asset"))
                    if kind == "permissions" and not asset:
                        # kept: odrl:target is optional; only counted for review
                        dropped["permission_without_asset_kept"] += 1

                    # Clean and collect constraints, keeping where each one was read
                    constraints = []
                    for raw_c in _dicts(rule.get("constraints")):
                        c = clean_constraint(raw_c)
                        if c:
                            c["provenance"] = evidence_of(raw_c)
                            constraints.append(c)
                        else:
                            drop("constraint_dropped", constraint_drop_reason(raw_c) or "", target + ".constraints",
                                 before={k: v for k, v in raw_c.items() if k not in ("source", "evidence")})
                    rule_prov = evidence_of(rule)
                    for party in parties:
                        key = (action, action_object, party, asset)
                        entry = rules[kind].setdefault(
                            key, {"action": action, "action_object": action_object,
                                  "party": party, "asset": asset, "action_labels": [], "provisions": [],
                                  "constraints": [], "support": 0, "approved_support": 0})
                        if entry["support"]:
                            row_events.append(change("rule_merged", str(rule.get("action_label") or action),
                                                     actor="merger", target=target,
                                                     after=rule_target(kind, entry)))
                        rule_chunks = _support(rule)[0]
                        entry["support"] += rule_chunks
                        entry.setdefault("sources", set()).update(_sources(rule, celex))
                        entry["approved_support"] += _approved_support(rule, row_approved, rule_chunks)
                        # on a collision every label and provision is kept, not just the first one
                        for field, values in (("action_labels", rule.get("action_labels") or [rule.get("action_label")]),
                                              ("provisions", rule.get("provisions") or [rule.get("provision")])):
                            for v in values:
                                if v and v not in entry[field]:
                                    entry[field].append(v)
                        entry["provenance"] = merge_evidence(entry.get("provenance", []), rule_prov,
                                                             MAX_EVIDENCE)

                        # Unique constraint aggregation per rule
                        by_key = {_constraint_key(c): c for c in entry["constraints"]}
                        for c in constraints:
                            k = _constraint_key(c)
                            if k in by_key:
                                by_key[k]["provenance"] = merge_evidence(
                                    by_key[k].get("provenance", []), c.get("provenance", []), MAX_EVIDENCE)
                            elif len(entry["constraints"]) < MAX_CONSTRAINTS_PER_RULE:
                                c = dict(c)
                                entry["constraints"].append(c)
                                by_key[k] = c
                            else:
                                drop("constraint_dropped", "constraints_at_cap", target + ".constraints",
                                     before={kk: v for kk, v in c.items() if kk != "provenance"})

        # 1d: Process legal patterns
        for pat in _dicts(onto.get("legal_patterns")):
            involved = pat.get("entities_involved")
            p_target = f"legal_patterns[{pat.get('pattern_name')}]"
            if not isinstance(involved, list):
                drop("pattern_dropped", "no entities_involved", p_target)
                continue
            ents = list(dict.fromkeys(e for e in (resolve(x) for x in involved if isinstance(x, str)) if e))
            if len(ents) < MIN_PATTERN_ENTITIES or not pat.get("pattern_name"):
                drop("pattern_dropped", f"{len(ents)} resolved entities", p_target, before=involved)
                continue

            # Group equivalent pattern names
            key = group_key("class", str(pat["pattern_name"]))      # EGF_Mobilization_Pattern == EGF_Mobilisation_Pattern
            entry = patterns.setdefault(key, {"pattern_name": pat["pattern_name"],
                                              "description": pat.get("description", "") or "",
                                              "entities_involved": [], "support": 0})
            for e in ents:
                if e not in entry["entities_involved"] and len(entry["entities_involved"]) < MAX_PATTERN_ENTITIES:
                    entry["entities_involved"].append(e)
            entry["support"] += _support(pat)[0]

        # 1e: Process example instances
        for inst in _dicts(onto.get("example_instances")):
            entity = resolve(inst.get("entity"))
            value = inst.get("example_value")
            ctx_text = inst.get("source_context") or ""

            # Filter out noisy or generic example values
            why = _example_drop_reason(entity, value, ctx_text)
            if why:
                drop("example_dropped", why, f"example_instances[{inst.get('entity')}: {value}]")
                continue
            examples.setdefault((entity, value.strip()), {**inst, "entity": entity})

        events.extend(stamp(row_events, stage="merge", chunk_id=source_id, celex=celex))

    # Phase 2: Finalize schemas, infer missing types, and format output
    final_classes = []
    for name, cls in classes.items():
        # Assign entity type via majority vote; default to 'Legal_extension'
        votes = type_votes.get(name)
        declared_type = votes.most_common(1)[0][0] if votes else "Legal_extension"

        if declared_type not in (VALID_CORE_TYPES | {"Legal_extension"}):
            declared_type = "Legal_extension"
        documents = len(class_docs[name]) + class_docs_n[name]
        if class_sources[name]:
            # exact count of distinct acts when provenance is complete
            documents = max(len(class_sources[name]), 1)
        entry = {
            "name": name, "type": declared_type, "description": cls["description"],
            "sources": sorted(class_sources[name])[:MAX_SOURCES],
            "support": {"chunks": class_chunks[name],
                        "documents": documents,
                        "approved_chunks": class_approved[name]},
        }
        if is_a_votes.get(name):
            entry["is_a"] = is_a_votes[name].most_common(1)[0][0]
        if kind_votes.get(name):
            entry["kind"] = kind_votes[name].most_common(1)[0][0]
        if name in forced:
            # acts and companies are individuals; their upper class follows the name
            f = forced[name]
            before = {k: entry.get(k) for k in ("type", "is_a", "kind")}
            entry["kind"] = "individual"
            if f.get("is_a") and f["is_a"] in vocab.upper:
                entry["is_a"], entry["type"] = f["is_a"], vocab.core_type_of(f["is_a"])
            after = {k: entry.get(k) for k in ("type", "is_a", "kind")}
            if after != before:
                events.append(stamp([change("individual_from_name", f["reason"], actor="merger",
                                            target=class_target(name), before=before, after=after)],
                                    stage="merge")[0])
        if name in class_iri:
            entry["iri"] = class_iri[name]
        if class_prov[name]:
            entry["provenance"] = class_prov[name]
        if name in defined:
            entry["defined_by_merge"] = defined[name]
        final_classes.append(entry)

    # a name that is used as party/asset in the rights model but typed Legal_extension stays as is:
    # we do not guess types, but we report the mismatch in provenance
    party_names = {r["party"] for k in RULE_KINDS for r in rules[k].values()}

    # a class used as party / asset in the rights model but typed Legal_extension gets that core type
    asset_names = {r["asset"] for k in RULE_KINDS for r in rules[k].values() if r.get("asset")}
    type_inferred = []
    for c in final_classes:
        if c["type"] == "Legal_extension":
            if c["name"] in party_names:
                c["type"] = "Party"
            elif c["name"] in asset_names:
                c["type"] = "Asset"
            else:
                continue
            type_inferred.append(c["name"])

    # Track unresolved type mismatches for audit trails/provenance
    type_by_name = {c["name"]: c["type"] for c in final_classes}
    party_type_mismatch = sorted(n for n in party_names if type_by_name.get(n) not in ("Party", "RightsHolder"))

    # Filter out low-support generic relationships
    final_relationships = []
    for r in relationships.values():
        if r["relation"] in GENERIC_RELATIONS and r["support"] < MIN_GENERIC_RELATION_SUPPORT:
            dropped["relationship_generic_low_support"] += 1
            events.append(stamp([change("relationship_generic_low_support", r["relation"], actor="merger",
                                        target=f"relationships[{r['source']} {r['relation']} {r['target']}]")],
                                stage="merge")[0])
            continue
        final_relationships.append(r)

    n_sources = carried_sources or len(source_ids)
    n_docs = carried_docs or len(source_celex)

    # Limit example instances to MAX_EXAMPLES_PER_ENTITY per entity type
    per_entity: Counter = Counter()
    final_examples = []
    for (entity, value), inst in examples.items():
        if per_entity[entity] < MAX_EXAMPLES_PER_ENTITY:
            per_entity[entity] += 1
            final_examples.append(inst)
        else:
            dropped["example_dropped"] += 1
            events.append(stamp([change("example_dropped", "examples_at_cap", actor="merger",
                                        target=f"example_instances[{entity}: {value}]")], stage="merge")[0])

    # Build consolidated JSON schema structure
    merged = {
        "ontology_name": f"EurLex ODRL Merged Ontology ({source_label})",
        "description": (f"Ontology merged from {n_sources} source ontologies "
                        f"covering {n_docs} EUR-Lex document(s)."),
        "odrl_alignment": {
            "core_entities": sorted(VALID_CORE_TYPES),
            "extensions": sorted(c["name"] for c in final_classes if c["type"] == "Legal_extension"),
        },
        "classes": final_classes,
        "relationships": final_relationships,
        "rights_model": {k: [_public(r, k) for r in rules[k].values()] for k in RULE_KINDS},
        "legal_patterns": list(patterns.values()),
        "example_instances": final_examples,
    }

    # Build metadata tracking execution, filtering stats, and type conversions
    provenance = {
        "source_label": source_label,
        "n_source_ontologies": n_sources,
        "n_source_documents": n_docs,
        "type_inferred": type_inferred,
        "source_ids": sorted(source_ids),
        "source_celex": sorted(source_celex),
        "n_classes_before_dedup": len(raw_class_names),
        "n_classes_after_dedup": len(final_classes),
        "n_rules_in": rules_in,
        "dropped": dict(dropped),
        "party_type_mismatch": party_type_mismatch,
        "concept_mapping": mapping,
        "events": events,
    }
    return merged, provenance


def merge_coverage(merged: dict, provenance: dict) -> dict:
    """Where every source concept went: canonical name, split into parts, or
    dropped with a reason. A `silent_loss` is a mapped concept whose canonical is
    missing from the merged ontology, or a drop whose reason is a bug."""
    mapping = provenance.get("concept_mapping") or []
    final = {c["name"] for c in _dicts(merged.get("classes"))}
    by_name: dict[str, dict] = {}
    for m in mapping:
        if not isinstance(m.get("name"), str):
            continue
        e = by_name.setdefault(m["name"], {"name": m["name"], "status": m["status"], "canonical": m["canonical"],
                                           "reason": m["reason"], "chunks": []})
        if m["status"] != DROPPED and e["status"] == DROPPED:      # mapped anywhere wins over dropped
            e.update(status=m["status"], canonical=m["canonical"], reason=m["reason"])
        if m["source_id"] not in e["chunks"]:
            e["chunks"].append(m["source_id"])

    silent = []
    for e in by_name.values():
        targets = e["canonical"] if isinstance(e["canonical"], list) else [e["canonical"]]
        if e["status"] == DROPPED and e["reason"] in SILENT_REASONS:
            silent.append({**e, "why": e["reason"]})
        elif e["status"] != DROPPED and not all(t in final for t in targets if t):
            silent.append({**e, "why": "canonical missing from the merged ontology"})
        elif e["status"] == SPLIT and not targets:
            silent.append({**e, "why": "composite with no resolvable part"})

    variants: dict[str, list] = defaultdict(list)
    for e in by_name.values():
        if e["status"] == MAPPED and e["canonical"] != e["name"]:
            variants[e["canonical"]].append(e["name"])
    status = Counter(e["status"] for e in by_name.values())
    rules_out = sum(len(v) for v in (merged.get("rights_model") or {}).values())
    dropped_counts = provenance.get("dropped") or {}
    return {
        "source_label": provenance.get("source_label"),
        "source_concepts": len(by_name),
        "mapped": status[MAPPED], "split": status[SPLIT], "dropped": status[DROPPED],
        "coverage": round((status[MAPPED] + status[SPLIT]) / len(by_name), 4) if by_name else 1.0,
        "dropped_by_reason": dict(Counter(e["reason"] for e in by_name.values() if e["status"] == DROPPED)),
        "dropped_concepts": sorted(({k: e[k] for k in ("name", "reason", "chunks")}
                                    for e in by_name.values() if e["status"] == DROPPED),
                                   key=lambda e: (e["reason"] or "", e["name"])),
        "defined_by_merge": sorted(c["name"] for c in _dicts(merged.get("classes")) if c.get("defined_by_merge")),
        "individuals_from_name": sorted(c["name"] for c in _dicts(merged.get("classes"))
                                        if c.get("kind") == "individual" and c["name"] in
                                        {e["canonical"] for e in by_name.values() if e["reason"] in
                                         ("legal_act_reference", "organisation_name", "instance_name")}),
        "silent_losses": silent,
        "variants": {k: sorted(v) for k, v in sorted(variants.items())},
        "rules": {"input": provenance.get("n_rules_in"), "output": rules_out,
                  "dropped": {k: v for k, v in dropped_counts.items()
                              if k.startswith("rule_") and k != "rule_asset_cleared"},
                  "asset_cleared": dropped_counts.get("rule_asset_cleared", 0),
                  "merged": sum(1 for e in provenance.get("events") or [] if e.get("code") == "rule_merged")},
    }


def check_coverage(coverage: dict, fail_on_silent_loss: bool) -> None:
    """Raise when concepts were lost without a recorded reason."""
    lost = coverage.get("silent_losses") or []
    if lost and fail_on_silent_loss:
        names = ", ".join(e["name"] for e in lost[:10])
        raise ValueError(f"Merge {coverage.get('source_label')}: {len(lost)} concept(s) lost without a "
                         f"reason: {names}. Set fail_on_silent_loss: false to only report them.")


# Format an internal rule entry into a standardized public JSON representation
def _public(rule: dict, kind: str) -> dict:
    out = {"action": rule["action"], "party": rule["party"], "constraints": rule["constraints"],
           "support": {"chunks": rule["support"], "approved_chunks": rule.get("approved_support", 0)}}
    if rule.get("sources"):
        out["sources"] = sorted(rule["sources"])[:MAX_SOURCES]
    if rule.get("action_object"):
        out["action_object"] = rule["action_object"]
    labels = rule.get("action_labels") or ([rule["action_label"]] if rule.get("action_label") else [])
    if labels:
        out["action_label"] = labels[0]
        if len(labels) > 1:
            out["action_labels"] = labels
    provisions = rule.get("provisions") or ([rule["provision"]] if rule.get("provision") else [])
    if provisions:
        out["provision"] = provisions[0]
        if len(provisions) > 1:
            out["provisions"] = provisions
    if rule.get("asset") or kind == "permissions":
        out["asset"] = rule.get("asset")
    if rule.get("provenance"):
        out["provenance"] = rule["provenance"]
    return out


# Validates the structure and referential integrity of a merged ontology dictionary
def validate_merged_ontology(ontology: dict) -> list[str]:
    issues: list[str] = []

    # 1: Verify schema top-level keys match expected signature exactly
    expected = {"ontology_name", "description", "odrl_alignment", "classes", "relationships",
                "rights_model", "legal_patterns", "example_instances"}
    if set(ontology) != expected:
        issues.append(f"schema_mismatch: missing={sorted(expected - set(ontology))}, extra={sorted(set(ontology) - expected)}")

    # 2: Check for unrecognized class types
    valid_types = VALID_CORE_TYPES | {"Legal_extension"}
    classes = _dicts(ontology.get("classes"))
    names = {c.get("name") for c in classes}
    for c in classes:
        if c.get("type") not in valid_types:
            issues.append(f"invalid_class_type:{c.get('name')}")

    # 3: Ensure 'extensions' in odrl_alignment matches classes typed as Legal_extension
    ext = set(ontology.get("odrl_alignment", {}).get("extensions", []) or [])
    legal = {c["name"] for c in classes if c.get("type") == "Legal_extension"}
    if ext != legal:
        issues.append(f"extensions_classes_mismatch: extensions_only={sorted(ext - legal)}, classes_only={sorted(legal - ext)}")

    # 4: Collect referenced entity names to check for broken cross-references or placeholder constraints
    refs: set[str] = set()
    for r in _dicts(ontology.get("relationships")):
        refs.update(v for v in (r.get("source"), r.get("target")) if isinstance(v, str))
        if r.get("source") == r.get("target"):
            issues.append(f"self_loop:{r.get('source')}")
    rm = ontology.get("rights_model") or {}
    for kind in RULE_KINDS:
        for rule in _dicts(rm.get(kind)):
            refs.update(v for v in (rule.get("party"), rule.get("asset")) if isinstance(v, str))
            for c in _dicts(rule.get("constraints")):
                value = c.get("right_operand") if "left_operand" in c else c.get("value")
                if is_placeholder(value):
                    issues.append(f"placeholder_constraint:{rule.get('action')}")
    for p in _dicts(ontology.get("legal_patterns")):
        refs.update(v for v in (p.get("entities_involved") or []) if isinstance(v, str))
    refs.update(i["entity"] for i in _dicts(ontology.get("example_instances")) if isinstance(i.get("entity"), str))

    # 5: Flag any referenced entity that is not defined in the classes list
    undefined = sorted(refs - names)
    if undefined:
        issues.append(f"undefined_entities:{undefined}")
    return issues
