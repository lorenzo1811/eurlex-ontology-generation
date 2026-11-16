import json
from pathlib import Path

import pandas as pd

from eurlex_ontology_generation.utils.ontology_merge_common.assembly import (
    assemble_merged_ontology,
    check_coverage,
    merge_coverage,
    validate_merged_ontology,
)
from eurlex_ontology_generation.utils.ontology_merge_common.entity_resolution import (
    canonicalize_clusters,
    cluster_entities,
    extract_entities,
)


# Load one revisions partition file (data/04_feature/ontology_revisions/part_XXX.csv).
# Works even if the file is only partially filled (e.g. only a few
# batches processed so far out of the 1000 the partition can hold) -
# useful to test the merge on a small slice while the full dataset is
# still being processed in the background.
#
# When reviews_directory is given, each revision gets the verdict of its
# latest review round as an `approved` column. The merge then records, for
# every class and rule, how much of its support comes from approved chunks
# (support.approved_chunks), so consumers can filter by confidence.
def load_partition_revisions(part_id: int, revisions_directory: str,
                             reviews_directory: str | None = None,
                             exclude_unapproved: bool = False) -> pd.DataFrame:
    path = Path(revisions_directory) / f"part_{part_id:03d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Revisions partition not found: {path}")

    revisions = pd.read_csv(path, encoding="utf-8")
    if revisions.empty:
        raise ValueError(f"Revisions partition {path} is empty.")
    # revisions are kept per (chunk, round): merge the latest round of each chunk
    if "round" in revisions.columns:
        revisions = (revisions.assign(_r=revisions["round"].fillna(0).astype(int))
                     .sort_values("_r").drop_duplicates("chunk_id", keep="last")
                     .drop(columns="_r").reset_index(drop=True))

    revisions["approved"] = False
    review_path = Path(reviews_directory) / f"part_{part_id:03d}.csv" if reviews_directory else None
    if review_path and review_path.exists():
        reviews = pd.read_csv(review_path, encoding="utf-8")
        if "round" not in reviews.columns:
            reviews["round"] = 0
        reviews["round"] = reviews["round"].fillna(0).astype(int)
        latest = reviews.sort_values("round").drop_duplicates("chunk_id", keep="last")
        verdict = latest.set_index("chunk_id")["approved"].astype(str).str.lower().eq("true")
        # a verdict counts only if that review saw the ontology being merged:
        # round 0 reviewed the candidate, so only rounds >= 1 judge a revision
        reviewed_revision = latest.set_index("chunk_id")["round"] >= 1
        revisions["approved"] = revisions["chunk_id"].map(verdict & reviewed_revision).fillna(False)
        if exclude_unapproved:
            revisions = revisions[revisions["approved"]].reset_index(drop=True)

    return revisions


# Merge every ontology in this partition into a single one: extract
# entities -> embed -> cluster -> canonicalize -> deterministic assembly.
def merge_partition(  # noqa: PLR0913
    revisions: pd.DataFrame,
    part_id: int,
    embedding_model: str,
    similarity_threshold: float,
    embedding_backend: dict | None = None,
    fail_on_silent_loss: bool = True,
) -> tuple[dict, dict, pd.DataFrame]:
    entities = extract_entities(revisions)
    entities_with_clusters = cluster_entities(entities, embedding_model, similarity_threshold, embedding_backend)
    canonical_map, cluster_report = canonicalize_clusters(entities_with_clusters)

    merged, provenance = assemble_merged_ontology(
        revisions, canonical_map, source_label=f"partition_{part_id:03d}"
    )
    provenance["validation_issues"] = validate_merged_ontology(merged)
    attach_coverage(merged, provenance, fail_on_silent_loss)

    return merged, provenance, cluster_report


# Coverage of the merge (every source concept -> canonical, or dropped with a reason).
# The full report, with the concept mapping, goes to its own file (see
# persist_coverage); the provenance keeps the summary.
def attach_coverage(merged: dict, provenance: dict, fail_on_silent_loss: bool) -> dict:
    coverage = merge_coverage(merged, provenance)
    coverage["concept_mapping"] = provenance.pop("concept_mapping", [])
    provenance["merge_coverage"] = coverage
    check_coverage(coverage, fail_on_silent_loss)
    return coverage


def persist_coverage(provenance: dict, path: Path) -> None:
    coverage = provenance.pop("merge_coverage", None)
    if coverage is None:
        return
    provenance["merge_coverage_summary"] = {
        k: coverage[k] for k in ("source_concepts", "mapped", "split", "dropped", "coverage",
                                 "dropped_by_reason", "rules")}
    provenance["merge_coverage_path"] = Path(path).as_posix()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")


# Persist the partition-level merged ontology, its provenance report and
# the cluster report (one row per canonical entity, useful to audit
# which chunk-level names were merged together and tune the similarity
# threshold).
def persist_partition_merge(  # noqa: PLR0913
    merged_ontology: dict,
    provenance: dict,
    cluster_report: pd.DataFrame,
    part_id: int,
    output_directory: str,
    provenance_directory: str,
    cluster_report_directory: str,
    coverage_directory: str | None = None,
) -> str:
    if coverage_directory:
        persist_coverage(provenance, Path(coverage_directory) / f"part_{part_id:03d}.json")
    onto_path = Path(output_directory) / f"part_{part_id:03d}.json"
    onto_path.parent.mkdir(parents=True, exist_ok=True)
    onto_path.write_text(
        json.dumps(merged_ontology, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    prov_path = Path(provenance_directory) / f"part_{part_id:03d}.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_path = Path(cluster_report_directory) / f"part_{part_id:03d}.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_report.to_csv(report_path, index=False, encoding="utf-8")

    return str(onto_path)
