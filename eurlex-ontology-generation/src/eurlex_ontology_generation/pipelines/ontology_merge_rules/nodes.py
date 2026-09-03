import json
import logging
from difflib import SequenceMatcher

""" Reuse building blocks already validated in the previous pipeline ontology_merging:
    - _call_ollama_json handles the Ollama call + JSON parsing (with num_ctx fix)
    - _apply_mapping_merge mechanically applies class_mappings/cross_relationships to two ontologies - it was already pure 
      Python with no LLM involvement, so it works unchanged as the execution backend for rule-based merging too"""
from ..ontology_merging.nodes import _apply_mapping_merge, _call_ollama_json

logger = logging.getLogger(__name__)

# Fixed taxonomy used both by the LLM classifier and by the rule engine.
# Keeping this as a single source of truth avoids drift between what the
# LLM is asked to produce and what the deterministic engine expects.
CONCEPT_CATEGORIES = [
    "LegalInstrument",
    "EconomicCharge",
    "PartyRole",
    "Asset",
    "Restriction",
    "Other",
]

"""Generic retry wrapper: call the LLM until the JSON response contains
    all required top-level keys, nudging temperature upward on each retry
    (same strategy already validated in ontology_merging)."""
def _call_llm_json_with_retry(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    json_mode: bool,
    required_keys: set,
    max_retries: int = 5,
) -> dict:    

    last_result = {}
    for attempt in range(1, max_retries + 1):
        attempt_temperature = min(temperature + 0.1 * (attempt - 1), 0.6)
        result = _call_ollama_json(system_prompt, user_prompt, model, attempt_temperature, json_mode)

        if required_keys.issubset(result.keys()):
            return result

        last_result = result
        logger.warning(
            "Attempt %d/%d (temp=%.2f) produced unexpected shape (keys: %s). Retrying.",
            attempt, max_retries, attempt_temperature, list(result.keys()),
        )

    raise ValueError(
        f"LLM call failed after {max_retries} attempts, missing keys {required_keys}. "
        f"Last output: {json.dumps(last_result, ensure_ascii=False)[:500]}"
    )


# Phase 1: classify every class of every ontology into a fixed category.
# This is one of only two LLM touchpoints in the whole rule-based pipeline -
# everything downstream (rule generation excluded) is deterministic.
def profile_concepts(
    reviewed_ontologies: list[dict],
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt_template: str,
    json_mode: bool,
) -> list[dict]:

    profile = []
    for onto in reviewed_ontologies:
        classes = [c["name"] for c in onto.get("classes", [])]
        onto_name = onto.get("ontology_name", "?")

        user_prompt = user_prompt_template.format(
            ontology_name=onto_name,
            classes=json.dumps(classes, ensure_ascii=False),
            categories=json.dumps(CONCEPT_CATEGORIES, ensure_ascii=False),
        )

        try:
            result = _call_llm_json_with_retry(
                system_prompt, user_prompt, model, temperature, json_mode,
                required_keys={"classifications"},
            )
            classifications = result.get("classifications", [])
        except ValueError as e:
            # Graceful degradation: one ontology's profiling failure should
            # not discard the whole pipeline run. Fall back to "Other" for
            # every class in this ontology - the safest possible default,
            # since no rule ever treats "Other" as compatible for merging.
            logger.error(
                "Profiling failed for ontology '%s' after retries - falling back "
                "to category 'Other' for all its classes. Flag for manual review. Error: %s",
                onto_name, e,
            )
            classifications = [
                {"class": c, "category": "Other", "reason": "Profiling failed - manual review needed."}
                for c in classes
            ]

        profile.append({
            "ontology_name": onto_name,
            "classifications": classifications,
        })
        logger.info("Profiled %d classes for '%s'.", len(classes), onto_name)

    return profile


# Phase 2: derive a general, reusable ruleset from the categories observed.
# Single LLM call for the whole batch - the ruleset, not the LLM, drives
# every future merge decision.
def generate_merge_rules(
    concept_profile: list[dict],
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt_template: str,
    json_mode: bool,
) -> dict:

    categories_seen = sorted({
        c.get("category", "Other")
        for onto in concept_profile
        for c in onto.get("classifications", [])
    })

    user_prompt = user_prompt_template.format(
        categories_seen=json.dumps(categories_seen, ensure_ascii=False),
        full_categories=json.dumps(CONCEPT_CATEGORIES, ensure_ascii=False),
    )

    rules = _call_llm_json_with_retry(
        system_prompt, user_prompt, model, temperature, json_mode,
        required_keys={"equivalence_rules", "prohibition_rules", "relationship_inference_rules"},
    )

    logger.info(
        "Generated ruleset: %d equivalence, %d prohibition, %d relationship-inference rules.",
        len(rules.get("equivalence_rules", [])),
        len(rules.get("prohibition_rules", [])),
        len(rules.get("relationship_inference_rules", [])),
    )
    return rules


# Phase 3: deterministic validation - no LLM. Catches malformed or
# contradictory rules before they are ever applied to real ontology data.
def validate_merge_rules(merge_rules_raw: dict) -> dict:

    errors = []
    valid_categories = set(CONCEPT_CATEGORIES)

    def check_category(cat: str, rule_id: str):
        if cat not in valid_categories:
            errors.append(f"{rule_id}: unknown category '{cat}'")

    prohibited_pairs = set()
    for r in merge_rules_raw.get("prohibition_rules", []):
        rule_id = r.get("rule_id", "?")
        pair = r.get("condition", {}).get("category_pair", [])
        if len(pair) != 2:
            errors.append(f"{rule_id}: prohibition_rules condition must have exactly 2 categories")
            continue
        for cat in pair:
            check_category(cat, rule_id)
        prohibited_pairs.add(tuple(sorted(pair)))

    for r in merge_rules_raw.get("equivalence_rules", []):
        rule_id = r.get("rule_id", "?")
        cats = r.get("condition", {}).get("category_in", [])
        for cat in cats:
            check_category(cat, rule_id)
        if len(cats) == 2 and tuple(sorted(cats)) in prohibited_pairs:
            errors.append(
                f"{rule_id}: contradicts a prohibition_rule for the same category pair {tuple(sorted(cats))}"
            )

    for r in merge_rules_raw.get("relationship_inference_rules", []):
        rule_id = r.get("rule_id", "?")
        pair = r.get("condition", {}).get("category_pair", [])
        if len(pair) != 2:
            errors.append(f"{rule_id}: relationship_inference_rules condition must have exactly 2 categories")
            continue
        for cat in pair:
            check_category(cat, rule_id)
        if not r.get("relation"):
            errors.append(f"{rule_id}: missing 'relation' label")

    if errors:
        raise ValueError("Rule validation failed:\n" + "\n".join(errors))

    logger.info("Ruleset validated successfully: no errors found.")
    return merge_rules_raw


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# Phase 4a: apply the validated ruleset to a SINGLE pair of ontologies.
# 100% deterministic - no LLM call, no retries, no temperature, no randomness.
def _apply_rules_to_pair(
    ontology_a: dict,
    ontology_b: dict,
    category_lookup: dict,
    rules: dict,
    name_similarity_threshold: float,
) -> tuple[dict, list[dict]]:

    a_classes = [c["name"] for c in ontology_a.get("classes", [])]
    b_classes = [c["name"] for c in ontology_b.get("classes", [])]

    # Same CELEX = same underlying legal document (ontologies from different
    # chunks of the same act). Different CELEX = different legal instruments,
    # even if generically named the same way (e.g. both called "Regulation").
    same_source_document = ontology_a.get("_source_celex") == ontology_b.get("_source_celex")

    prohibited_pairs = {
        tuple(sorted(r["condition"]["category_pair"]))
        for r in rules.get("prohibition_rules", [])
    }
    relationship_rules_ordered = {
        tuple(r["condition"]["category_pair"]): r["relation"]
        for r in rules.get("relationship_inference_rules", [])
    }

    class_mappings = []
    cross_relationships = []
    rule_hits = []

    for a_name in a_classes:
        for b_name in b_classes:
            cat_a = category_lookup.get(a_name, "Other")
            cat_b = category_lookup.get(b_name, "Other")

            if cat_a == cat_b:
                similarity = _name_similarity(a_name, b_name)
                # Require near-exact name match across different source
                # documents, to avoid merging unrelated legal instruments
                # that merely share a generic category and a loosely
                # similar name (e.g. "Regulation" vs "ImportLevyRegulation"
                # from two completely different CELEX acts).
                effective_threshold = (
                    name_similarity_threshold if same_source_document else 0.9
                )
                if similarity >= effective_threshold:
                    class_mappings.append({
                        "a_class": a_name,
                        "b_class": b_name,
                        "merged_name": a_name,
                        "reason": (
                            f"Same category ({cat_a}), name similarity {similarity:.2f}, "
                            f"{'same' if same_source_document else 'different'} source document."
                        ),
                    })
                    rule_hits.append({
                        "rule_type": "equivalence",
                        "pair_classes": [a_name, b_name],
                        "category": cat_a,
                        "similarity": round(similarity, 2),
                        "same_source_document": same_source_document,
                    })
                continue

            # (resto invariato: relationship_inference / prohibition_no_relationship)
            ordered_pair = (cat_a, cat_b)
            if ordered_pair in relationship_rules_ordered:
                relation = relationship_rules_ordered[ordered_pair]
                cross_relationships.append({
                    "source": a_name,
                    "relation": relation,
                    "target": b_name,
                    "description": f"Inferred via relationship_inference_rule for {ordered_pair}.",
                })
                rule_hits.append({
                    "rule_type": "relationship_inference",
                    "pair_classes": [a_name, b_name],
                    "category_pair": list(ordered_pair),
                })
            elif tuple(reversed(ordered_pair)) in relationship_rules_ordered:
                reversed_pair = tuple(reversed(ordered_pair))
                relation = relationship_rules_ordered[reversed_pair]
                cross_relationships.append({
                    "source": b_name,
                    "relation": relation,
                    "target": a_name,
                    "description": f"Inferred via relationship_inference_rule for {reversed_pair} (reversed loop order).",
                })
                rule_hits.append({
                    "rule_type": "relationship_inference",
                    "pair_classes": [b_name, a_name],
                    "category_pair": list(reversed_pair),
                })
            elif tuple(sorted(ordered_pair)) in prohibited_pairs:
                rule_hits.append({
                    "rule_type": "prohibition_no_relationship",
                    "pair_classes": [a_name, b_name],
                    "category_pair": list(ordered_pair),
                })

    merged = _apply_mapping_merge(ontology_a, ontology_b, {
        "class_mappings": class_mappings,
        "cross_relationships": cross_relationships,
    })

    return merged, rule_hits


# Phase 4b: fold all N ontologies into one (1+2 -> M1, M1+3 -> M2, M2+4 -> M3, M3+5 -> final), 
# but with zero LLM calls in the loop.
def apply_merge_rules_iteratively(
    reviewed_ontologies: list[dict],
    concept_profile: list[dict],
    merge_rules: dict,
    name_similarity_threshold: float = 0.6,
) -> tuple[dict, list[dict]]:

    if not reviewed_ontologies:
        raise ValueError("No ontologies to merge.")

    category_lookup = {
        c["class"]: c.get("category", "Other")
        for onto in concept_profile
        for c in onto.get("classifications", [])
    }

    accumulator = reviewed_ontologies[0]
    trace = []

    for i, next_onto in enumerate(reviewed_ontologies[1:], start=2):
        next_name = next_onto.get("ontology_name", f"ontology_{i}")
        logger.info("Rule-based merge step %d/%d: merging with '%s'.", i - 1, len(reviewed_ontologies) - 1, next_name)

        merged, rule_hits = _apply_rules_to_pair(
            accumulator, next_onto, category_lookup, merge_rules, name_similarity_threshold
        )

        merged.pop("merge_log", None)  # produced by _apply_mapping_merge but redundant with rule_hits here
        trace.append({
            "step": i - 1,
            "merged_with": next_name,
            "rule_hits": rule_hits,
            "resulting_class_count": len(merged.get("classes", [])),
        })

        accumulator = merged

    return accumulator, trace