"""Summarise one batch run (generation -> review -> revision -> merge/refinement).

The numbers are the baseline the pipeline changes are measured against:
rules per type, what the cleanup/normaliser changed or deleted, which reviewer
weaknesses were discarded and why, missing concepts, chunks re-reviewed
although their ontology did not change, and source concepts that cannot be
found in the refined ontology.

Example (files copied from a Colab run, see data/09_baseline/):
  uv run python scripts/batch_report.py \
      --candidates data/09_baseline/batch_046/ontology_candidates/part_000.csv \
      --reviews    data/09_baseline/batch_046/ontology_reviews/part_000.csv \
      --revisions  data/09_baseline/batch_046/ontology_revisions/part_000.csv \
      --refined    data/09_baseline/batch_046/eurlex_odrl_ontology_refined.json \
      --batch-id 46 --out data/09_baseline/batch_046/report.json
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eurlex_ontology_generation.utils.ontology_merge_common.entity_resolution import (  # noqa: E402
    group_key,
)

RULE_KINDS = ("permissions", "prohibitions", "duties")


def _load(value):
    if isinstance(value, dict | list):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _read_csv(path, batch_id):
    if not path:
        return None
    df = pd.read_csv(path)
    if "round" not in df.columns:  # outputs older than the re-review loop
        df["round"] = 0
    if batch_id is not None and "batch_id" in df.columns:
        df = df[df["batch_id"] == batch_id]
    return df


def _rule_counts(ontologies) -> dict:
    counts = Counter()
    empty = 0
    for onto in ontologies:
        rm = (onto or {}).get("rights_model") or {}
        n = 0
        for k in RULE_KINDS:
            m = len(rm.get(k) or [])
            counts[k] += m
            n += m
        empty += n == 0
    return {**{k: counts[k] for k in RULE_KINDS}, "chunks_without_rules": empty}


def _note_codes(series) -> dict:
    codes = Counter()
    for value in series.dropna():
        for note in _load(value) or []:
            if isinstance(note, dict):
                codes[note.get("code", "?")] += 1
    return dict(codes.most_common())


def _provenance(ontologies) -> dict:
    """How the source of classes, rules and constraints was found (utils/provenance.py)."""
    out = {k: Counter() for k in ("classes", "rules", "constraints")}
    for onto in ontologies:
        for c in onto.get("classes") or []:
            out["classes"][(c.get("source") or {}).get("match", "missing")] += 1
        for k in RULE_KINDS:
            for r in (onto.get("rights_model") or {}).get(k) or []:
                out["rules"][(r.get("source") or {}).get("match", "missing")] += 1
                for c in r.get("constraints") or []:
                    out["constraints"][(c.get("source") or {}).get("match", "missing")] += 1
    return {k: dict(v.most_common()) for k, v in out.items()}


def _change_log(series) -> dict:
    by_actor, by_code = Counter(), Counter()
    for value in series.dropna():
        for e in _load(value) or []:
            if isinstance(e, dict):
                by_actor[e.get("actor", "?")] += 1
                by_code[f"{e.get('actor', '?')}:{e.get('code', '?')}"] += 1
    return {"by_actor": dict(by_actor.most_common()), "by_code": dict(by_code.most_common())}


def _canon(onto) -> str:
    return json.dumps(onto, sort_keys=True, ensure_ascii=False)


def candidates_report(df) -> dict:
    ontos = [_load(v) for v in df["ontology"]]
    return {
        "chunks": len(df),
        "invalid_json": sum(1 for o in ontos if not isinstance(o, dict) or "error" in o),
        "rules": _rule_counts(o for o in ontos if isinstance(o, dict)),
        "cleanup_notes": _note_codes(df["cleanup_notes"]) if "cleanup_notes" in df else {},
        "provenance": _provenance(o for o in ontos if isinstance(o, dict)),
        "change_log": _change_log(df["change_log"]) if "change_log" in df else {},
    }


def reviews_report(df) -> dict:
    out = {}
    for rnd, part in df.groupby("round"):
        findings, kept, discarded, reasons = Counter(), 0, 0, Counter()
        missing_total, chunks_with_missing, missing_examples = 0, 0, []
        for _, row in part.iterrows():
            for f in _load(row.get("automatic_checks")) or []:
                if isinstance(f, dict):
                    findings[f"{f.get('severity')}:{f.get('code')}"] += 1
            review = _load(row.get("review")) or {}
            kept += len(review.get("weaknesses") or [])
            for w in review.get("discarded_weaknesses") or []:
                discarded += 1
                for r in w.get("discard_reasons") or []:
                    # group "json_path not found in ontology: <path>" etc. by kind
                    reasons[re.sub(r"'[^']*'", "<path>", str(r).split(":")[0])] += 1
            missing = review.get("missing_concepts") or []
            missing_total += len(missing)
            chunks_with_missing += bool(missing)
            missing_examples.extend(missing[: max(0, 10 - len(missing_examples))])
        out[str(int(rnd))] = {
            "chunks": len(part),
            "approved": int(part["approved"].fillna(False).astype(bool).sum()),
            "automatic_findings": dict(findings.most_common()),
            "weaknesses_kept": kept,
            "weaknesses_discarded": discarded,
            "discard_reasons": dict(reasons.most_common()),
            "missing_concepts": missing_total,
            "chunks_with_missing_concepts": chunks_with_missing,
            "missing_concepts_examples": missing_examples,
        }
    return out


def unchanged_rereviews(df) -> dict:
    """Chunks whose ontology was identical in consecutive review rounds."""
    by_chunk = defaultdict(dict)
    for _, row in df.iterrows():
        onto = _load(row.get("ontology"))
        by_chunk[row["chunk_id"]][int(row["round"])] = _canon(onto) if onto is not None else None
    pairs, unchanged, ids = 0, 0, []
    for chunk_id, rounds in by_chunk.items():
        ordered = sorted(rounds)
        for a, b in zip(ordered, ordered[1:]):
            pairs += 1
            if rounds[a] is not None and rounds[a] == rounds[b]:
                unchanged += 1
                ids.append(f"{chunk_id}@{b}")
    return {"round_pairs": pairs, "unchanged": unchanged, "examples": ids[:20]}


def revisions_report(df) -> dict:
    ontos = [_load(v) for v in df["ontology"]]
    return {
        "chunks": len(df),
        "status": dict(Counter(df.get("revision_status", pd.Series(dtype=str)).fillna("?"))),
        "rules": _rule_counts(o for o in ontos if isinstance(o, dict)),
        "normalization_changes": (_note_codes(df["normalization_changes"])
                                  if "normalization_changes" in df else {}),
        "provenance": _provenance(o for o in ontos if isinstance(o, dict)),
        "change_log": _change_log(df["change_log"]) if "change_log" in df else {},
    }


def _entity_keys(onto) -> dict:
    keys = {}
    for section in ("classes", "individuals"):
        for c in onto.get(section) or []:
            if not isinstance(c, dict) or not isinstance(c.get("name"), str):
                continue
            etype = c.get("type") or ""
            keys[group_key(etype, c["name"])] = c["name"]
            for alias in c.get("merged_from") or []:
                if isinstance(alias, str):
                    keys[group_key(etype, alias)] = c["name"]
    return keys


def coverage_report(source_ontos, refined) -> dict:
    """Source class names whose merge key is absent from the refined ontology.

    Names merged by embedding similarity under a different key show up here
    too, so this is an upper bound on what was lost, to be checked by hand.
    """
    target = _entity_keys(refined)
    source = {}
    for onto in source_ontos:
        source.update(_entity_keys(onto))
    missing = sorted(name for key, name in source.items() if key not in target)
    rm = refined.get("rights_model") or {}
    return {
        "source_concepts": len(source),
        "refined_concepts": len({v for v in target.values()}),
        "source_keys_not_found": len(missing),
        "not_found": missing,
        "refined_rules": {k: len(rm.get(k) or []) for k in RULE_KINDS},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates")
    p.add_argument("--reviews")
    p.add_argument("--revisions")
    p.add_argument("--refined", help="refined (or merged) ontology JSON")
    p.add_argument("--batch-id", type=int, help="keep only rows of this batch")
    p.add_argument("--out", help="write the report as JSON here")
    args = p.parse_args()

    report: dict = {}
    candidates = _read_csv(args.candidates, args.batch_id)
    reviews = _read_csv(args.reviews, args.batch_id)
    revisions = _read_csv(args.revisions, args.batch_id)
    if candidates is not None:
        report["candidates"] = candidates_report(candidates)
    if reviews is not None:
        report["reviews"] = reviews_report(reviews)
        report["unchanged_rereviews"] = unchanged_rereviews(reviews)
    if revisions is not None:
        report["revisions"] = revisions_report(revisions)
    if args.refined:
        refined = json.loads(Path(args.refined).read_text(encoding="utf-8"))
        source = revisions if revisions is not None else candidates
        if source is not None:
            ontos = [o for o in (_load(v) for v in source["ontology"]) if isinstance(o, dict)]
            report["merge_coverage"] = coverage_report(ontos, refined)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)  # noqa: T201


if __name__ == "__main__":
    main()
