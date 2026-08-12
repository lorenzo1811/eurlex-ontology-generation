import json
import logging

import pandas as pd
from ollama import chat

from ..ontology_generation.nodes import extract_json

logger = logging.getLogger(__name__)


# Parse the 'ontology' column (JSON string) of each row into a Python dict
def parse_ontology_candidates(ontology_candidates: pd.DataFrame) -> list[dict]:

    ontologies = []
    for _, row in ontology_candidates.iterrows():
        onto = json.loads(row["ontology"])
        onto["_source_chunk_id"] = row["chunk_id"]
        onto["_source_celex"] = row["CELEX"]
        ontologies.append(onto)

    logger.info("Parsed %d candidate ontologies.", len(ontologies))
    return ontologies


# Query the local LLM through Ollama and return the parsed JSON response
def _call_ollama_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    json_mode: bool,
    num_ctx: int = 8192,
) -> dict:

    response = chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,  # default context window is too small for long prompts
        },
        format="json" if json_mode else None,
    )

    return extract_json(response["message"]["content"])


# Step (a): for each ontology, check whether its own concepts should be merged
def review_intra_ontology_concepts(
    ontologies: list[dict],
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt_template: str,
    json_mode: bool,
) -> tuple[list[dict], dict]:

    reviewed = []
    report = {}

    for onto in ontologies:
        # Metadata is not meant to be seen/edited by the LLM
        source_chunk_id = onto.pop("_source_chunk_id")
        source_celex = onto.pop("_source_celex")
        display_name = onto.get("ontology_name", source_chunk_id)

        user_prompt = user_prompt_template.format(
            ontology_json=json.dumps(onto, ensure_ascii=False, indent=2)
        )

        result = _call_ollama_json(
            system_prompt, user_prompt, model, temperature, json_mode
        )

        # Restore provenance metadata
        result["_source_chunk_id"] = source_chunk_id
        result["_source_celex"] = source_celex

        review_log = result.pop("review_log", [])
        report[display_name] = review_log

        n_merges = sum(
            1 for a in review_log
            if a.get("action") in ("merged_concepts", "merged")
        )
        logger.info(
            "Ontology '%s': %d merge action(s) out of %d review entries.",
            display_name, n_merges, len(review_log),
        )

        reviewed.append(result)

    return reviewed, report


# Step (b) - phase 1: ask the LLM only for class-equivalence mappings and new
# cross-ontology relationships. Keeping the LLM's job small (no full ontology
# rewrite) makes the response far less likely to collapse into an empty "{}"
def _propose_class_mapping(
    ontology_a: dict,
    ontology_b: dict,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt_template: str,
    json_mode: bool,
    max_retries: int = 5,
) -> dict:

    a_classes = [c["name"] for c in ontology_a.get("classes", [])]
    b_classes = [c["name"] for c in ontology_b.get("classes", [])]

    user_prompt = user_prompt_template.format(
        ontology_a_classes=json.dumps(a_classes, ensure_ascii=False),
        ontology_a_relationships=json.dumps(ontology_a.get("relationships", []), ensure_ascii=False),
        ontology_b_classes=json.dumps(b_classes, ensure_ascii=False),
        ontology_b_relationships=json.dumps(ontology_b.get("relationships", []), ensure_ascii=False),
    )

    last_result = {}
    for attempt in range(1, max_retries + 1):
        # Nudge temperature upward on each retry, but keep the ceiling low:
        # a small increase helps escape a repeated "{}" collapse, but too high
        # (e.g. close to 1.0) makes the model prone to over-merging distinct
        # concepts instead of just fixing the empty-output problem
        attempt_temperature = min(temperature + 0.1 * (attempt - 1), 0.6)

        result = _call_ollama_json(system_prompt, user_prompt, model, attempt_temperature, json_mode)

        if "class_mappings" in result and "cross_relationships" in result:
            return result

        last_result = result
        logger.warning(
            "Class-mapping attempt %d/%d (temp=%.2f) produced unexpected shape (keys: %s). Retrying.",
            attempt, max_retries, attempt_temperature, list(result.keys()),
        )

    # Graceful degradation: after exhausting retries, don't crash the whole
    # pipeline run. Assume "no merge / no new relationship" for this pair and
    # flag it clearly so it can be reviewed manually afterwards
    logger.error(
        "Class mapping failed after %d attempts (ontology_a='%s', ontology_b='%s'). "
        "Falling back to no merges/relationships for this step - review manually. "
        "Last LLM output: %s",
        max_retries,
        ontology_a.get("ontology_name", "?"),
        ontology_b.get("ontology_name", "?"),
        json.dumps(last_result, ensure_ascii=False)[:500],
    )
    return {
        "class_mappings": [],
        "cross_relationships": [],
        "_llm_failed": True,  # internal marker, stripped before saving the final merge_log entry
    }


# Step (b) - phase 1.5 (judge): a second LLM call reviews the mapping proposed
# by _propose_class_mapping. The judge only gives approve/reject verdicts by
# index - it never restates the mapping objects itself. Python then filters
# the original proposal based on the verdicts. This keeps the judge's output
# tiny and avoids asking it to reconstruct complex JSON objects.
def _judge_class_mapping(
    ontology_a: dict,
    ontology_b: dict,
    proposed_mapping: dict,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt_template: str,
    json_mode: bool,
    max_retries: int = 7,
) -> dict:

    a_classes = [c["name"] for c in ontology_a.get("classes", [])]
    b_classes = [c["name"] for c in ontology_b.get("classes", [])]

    class_mappings = proposed_mapping.get("class_mappings", [])
    cross_relationships = proposed_mapping.get("cross_relationships", [])

    # Render proposals as a compact numbered list instead of raw JSON - easier
    # for the judge to reference by index, and much shorter than full objects.
    numbered_class_mappings = "\n".join(
        f'{i}: merge "{m.get("a_class")}" (A) with "{m.get("b_class")}" (B) -> "{m.get("merged_name")}"'
        for i, m in enumerate(class_mappings)
    ) or "(none proposed)"

    numbered_cross_relationships = "\n".join(
        f'{i}: "{r.get("source")}" {r.get("relation")} "{r.get("target")}"'
        for i, r in enumerate(cross_relationships)
    ) or "(none proposed)"

    user_prompt = user_prompt_template.format(
        ontology_a_classes=json.dumps(a_classes, ensure_ascii=False),
        ontology_b_classes=json.dumps(b_classes, ensure_ascii=False),
        numbered_class_mappings=numbered_class_mappings,
        numbered_cross_relationships=numbered_cross_relationships,
    )

    last_result = {}
    for attempt in range(1, max_retries + 1):
        attempt_temperature = min(temperature + 0.1 * (attempt - 1), 0.6)

        result = _call_ollama_json(system_prompt, user_prompt, model, attempt_temperature, json_mode)

        if "class_mapping_verdicts" in result and "relationship_verdicts" in result:
            break

        last_result = result
        logger.warning(
            "Judge attempt %d/%d (temp=%.2f) produced unexpected shape (keys: %s). Retrying.",
            attempt, max_retries, attempt_temperature, list(result.keys()),
        )
    else:
        # All retries exhausted: fall back to rejecting everything. This is
        # the SAFE default (rule 5: "when in doubt, reject") rather than
        # silently trusting the unjudged proposal.
        logger.error(
            "Judge failed after %d attempts (ontology_a='%s', ontology_b='%s'). "
            "Falling back to REJECT ALL proposed items - flag for manual review.",
            max_retries,
            ontology_a.get("ontology_name", "?"),
            ontology_b.get("ontology_name", "?"),
        )
        return {
            "class_mappings": [],
            "cross_relationships": [],
            "judge_notes": [],
            "_judge_failed": True,
        }

    # Apply verdicts: keep only approved items, taken from the ORIGINAL
    # proposal (never from the judge's own text) - deterministic filtering.
    approved_mapping_indices = {
        v["index"] for v in result.get("class_mapping_verdicts", [])
        if v.get("verdict") == "approve" and isinstance(v.get("index"), int)
    }
    approved_relationship_indices = {
        v["index"] for v in result.get("relationship_verdicts", [])
        if v.get("verdict") == "approve" and isinstance(v.get("index"), int)
    }

    approved_class_mappings = [
        m for i, m in enumerate(class_mappings) if i in approved_mapping_indices
    ]
    approved_cross_relationships = [
        r for i, r in enumerate(cross_relationships) if i in approved_relationship_indices
    ]

    judge_notes = [
        {
            "item": f'class_mapping[{v.get("index")}]',
            "verdict": v.get("verdict", "unknown"),
            "reason": v.get("reason", ""),
        }
        for v in result.get("class_mapping_verdicts", [])
    ] + [
        {
            "item": f'cross_relationship[{v.get("index")}]',
            "verdict": v.get("verdict", "unknown"),
            "reason": v.get("reason", ""),
        }
        for v in result.get("relationship_verdicts", [])
    ]

    return {
        "class_mappings": approved_class_mappings,
        "cross_relationships": approved_cross_relationships,
        "judge_notes": judge_notes,
        "_judge_failed": False,
    }


# Step (b) - phase 2: mechanically apply the LLM's mapping decisions in plain
# Python (no LLM call here). Deterministic: no risk of empty/malformed JSON
def _apply_mapping_merge(ontology_a: dict, ontology_b: dict, mapping_result: dict) -> dict:

    # True when _propose_class_mapping had to fall back after exhausting retries
    llm_failed = mapping_result.get("_llm_failed", False)

    judge_failed = mapping_result.get("_judge_failed", False)
    judge_notes = mapping_result.get("judge_notes", [])

    # Defensive filtering: keep only mapping entries that have all the fields
    # we actually need. A partial/malformed entry from the LLM is skipped
    # instead of crashing the node with a KeyError.
    class_mappings = [
        m for m in mapping_result.get("class_mappings", [])
        if m.get("a_class") and m.get("b_class") and m.get("merged_name")
    ]
    cross_relationships = [
        r for r in mapping_result.get("cross_relationships", [])
        if r.get("source") and r.get("target") and r.get("relation")
    ]

    # name_a -> merged_name / name_b -> merged_name, used to rewrite references
    rename_a_to_merged = {m["a_class"]: m["merged_name"] for m in class_mappings}
    rename_b_to_merged = {m["b_class"]: m["merged_name"] for m in class_mappings}

    def rewrite(name: str, side: str) -> str:
        table = rename_a_to_merged if side == "a" else rename_b_to_merged
        return table.get(name, name)

    # A cross-relationship endpoint could reference an original A-side name,
    # an original B-side name, or a name already correct - check both tables.
    def rewrite_any_side(name: str) -> str:
        if name in rename_a_to_merged:
            return rename_a_to_merged[name]
        if name in rename_b_to_merged:
            return rename_b_to_merged[name]
        return name

    # legal_patterns[].entities_involved may reference pre-merge class names;
    # rewrite each entry so it points to the actual merged class name.
    def rewrite_legal_patterns(patterns: list[dict]) -> list[dict]:
        return [
            {**p, "entities_involved": [rewrite_any_side(e) for e in p.get("entities_involved", [])]}
            for p in patterns
        ]

    # example_instances[].entity may also reference a pre-merge class name.
    def rewrite_example_instances(instances: list[dict]) -> list[dict]:
        return [
            {**inst, "entity": rewrite_any_side(inst.get("entity", ""))}
            for inst in instances
        ]

    # Classes: union of A and B, with mapped pairs collapsed into a single class
    merged_classes = []
    seen_names = set()

    for c in ontology_a.get("classes", []):
        new_name = rewrite(c["name"], "a")
        if new_name not in seen_names:
            merged_classes.append({**c, "name": new_name})
            seen_names.add(new_name)

    for c in ontology_b.get("classes", []):
        new_name = rewrite(c["name"], "b")
        if new_name not in seen_names:
            merged_classes.append({**c, "name": new_name})
            seen_names.add(new_name)

    # Relationships: original ones from A and B (with renamed references)
    def rewrite_relationship(r: dict, side: str) -> dict:
        return {
            **r,
            "source": rewrite(r.get("source", ""), side),
            "target": rewrite(r.get("target", ""), side),
        }

    # Cross-relationships proposed by the LLM: their endpoints must ALSO be
    # rewritten, in case the LLM referred to a class by its pre-merge name
    # (fixes dangling references to classes that no longer exist standalone).
    rewritten_cross_relationships = [
        {**r, "source": rewrite_any_side(r["source"]), "target": rewrite_any_side(r["target"])}
        for r in cross_relationships
    ]

    merged_relationships = (
        [rewrite_relationship(r, "a") for r in ontology_a.get("relationships", [])]
        + [rewrite_relationship(r, "b") for r in ontology_b.get("relationships", [])]
        + rewritten_cross_relationships
    )

    # Filter out self-loops (source == target after rewriting) and deduplicate
    # relationships that are semantically identical (same source/relation/target),
    # keeping only the first occurrence.
    def _dedupe_and_filter(relationships: list[dict]) -> list[dict]:
        cleaned = []
        seen_triples = set()
        for r in relationships:
            source, relation, target = r.get("source", ""), r.get("relation", ""), r.get("target", "")

            # Drop self-loops: a class doesn't need a relation to itself,
            # and these only appear as a side-effect of two merged names
            # collapsing onto the same merged_name.
            if source == target:
                continue

            key = (source, relation, target)
            if key in seen_triples:
                continue
            seen_triples.add(key)
            cleaned.append(r)
        return cleaned

    merged_relationships = _dedupe_and_filter(merged_relationships)

    # Human-readable trace of what happened during this merge step
    merge_log = (
        [
            {
                "action": "merged_concepts",
                "concepts_involved": [m["a_class"], m["b_class"]],
                "reason": m.get("reason", ""),
            }
            for m in class_mappings
        ]
        + [
            {
                "action": "added_cross_relationship",
                # Log the rewritten endpoints too, so the log matches what's
                # actually in the final relationships list
                "concepts_involved": [r["source"], r["target"]],
                "reason": r.get("description", ""),
            }
            for r in rewritten_cross_relationships
        ]
    )

    if llm_failed:
        merge_log.append({
            "action": "llm_failed_needs_manual_review",
            "concepts_involved": [],
            "reason": (
                "LLM did not return a usable mapping after retries; "
                "no merges/relationships were applied automatically for this step."
            ),
        })

    # include the judge's own reasoning in the log, so rejected/corrected
    # items are visible even though they never reach the approved lists
    for note in judge_notes:
        merge_log.append({
            "action": f"judge_{note.get('verdict', 'unknown')}",
            "concepts_involved": [note.get("item", "")],
            "reason": note.get("reason", ""),
        })

    if judge_failed:
        merge_log.append({
            "action": "judge_failed_using_unreviewed_proposal",
            "concepts_involved": [],
            "reason": (
                "The judge LLM did not return a usable review after retries; "
                "the original unjudged proposal was applied as-is - flag for manual review."
            ),
        })

    merged = {
        "ontology_name": f"{ontology_a.get('ontology_name', 'A')}_merged_{ontology_b.get('ontology_name', 'B')}",
        "description": (
            f"Merged ontology combining '{ontology_a.get('ontology_name')}' "
            f"and '{ontology_b.get('ontology_name')}'."
        ),
        "odrl_alignment": ontology_a.get("odrl_alignment", {}),
        "classes": merged_classes,
        "relationships": merged_relationships,
        "rights_model": {
            "permissions": (
                ontology_a.get("rights_model", {}).get("permissions", [])
                + ontology_b.get("rights_model", {}).get("permissions", [])
            ),
            "duties": (
                ontology_a.get("rights_model", {}).get("duties", [])
                + ontology_b.get("rights_model", {}).get("duties", [])
            ),
        },
        "legal_patterns": rewrite_legal_patterns(
            ontology_a.get("legal_patterns", []) + ontology_b.get("legal_patterns", [])
        ),
        "example_instances": rewrite_example_instances(
            ontology_a.get("example_instances", []) + ontology_b.get("example_instances", [])
        ),
        "merge_log": merge_log,
    }

    return merged


# Merge two ontologies into one: LLM proposes the mapping (phase 1),
# LLM judges/corrects it (phase 1.5), Python applies it deterministically (phase 2)
def _merge_two_ontologies(
    ontology_a: dict,
    ontology_b: dict,
    model: str,
    temperature: float,
    propose_system_prompt: str,
    propose_user_prompt_template: str,
    judge_system_prompt: str,
    judge_user_prompt_template: str,
    json_mode: bool,
) -> dict:

    # Strip internal bookkeeping fields before sending anything to the LLM
    clean_a = {k: v for k, v in ontology_a.items() if not k.startswith("_")}
    clean_b = {k: v for k, v in ontology_b.items() if not k.startswith("_")}

    proposed = _propose_class_mapping(
        clean_a, clean_b, model, temperature, propose_system_prompt, propose_user_prompt_template, json_mode
    )

    judged = _judge_class_mapping(
        clean_a, clean_b, proposed, model, temperature, judge_system_prompt, judge_user_prompt_template, json_mode
    )

    # Translate the judge's output shape into the shape _apply_mapping_merge expects
    mapping_result = {
        "class_mappings": judged["class_mappings"],
        "cross_relationships": judged["cross_relationships"],
        "_llm_failed": proposed.get("_llm_failed", False),
        "_judge_failed": judged.get("_judge_failed", False),
        "judge_notes": judged.get("judge_notes", []),
    }

    return _apply_mapping_merge(clean_a, clean_b, mapping_result)


# Step (b): iteratively fold all ontologies into a single merged one
# (1+2 -> M1, M1+3 -> M2, M2+4 -> M3, M3+5 -> final)
def iterative_merge_ontologies(
    reviewed_ontologies: list[dict],
    model: str,
    temperature: float,
    propose_system_prompt: str,
    propose_user_prompt_template: str,
    judge_system_prompt: str,
    judge_user_prompt_template: str,
    json_mode: bool,
) -> tuple[dict, list[dict]]:

    if not reviewed_ontologies:
        raise ValueError("No ontologies to merge.")

    accumulator = reviewed_ontologies[0]
    trace = []

    for i, next_onto in enumerate(reviewed_ontologies[1:], start=2):
        next_name = next_onto.get("ontology_name", f"ontology_{i}")
        logger.info(
            "Merge step %d/%d: merging with '%s'.",
            i - 1, len(reviewed_ontologies) - 1, next_name,
        )

        merged = _merge_two_ontologies(
            accumulator, next_onto, model, temperature, propose_system_prompt, propose_user_prompt_template,
            judge_system_prompt, judge_user_prompt_template, json_mode,
        )

        # Pull the per-step log out of the merged ontology and into the trace,
        # so the final ontology stays clean and the history is kept separately
        merge_log = merged.pop("merge_log", [])
        trace.append(
            {
                "step": i - 1,
                "merged_with": next_name,
                "merge_log": merge_log,
                "resulting_class_count": len(merged.get("classes", [])),
            }
        )

        accumulator = merged

    return accumulator, trace