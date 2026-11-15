import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from ...utils.llm_client import llm_chat, run_parallel
from ...utils.persist import append_dedup_csv
from ...utils.provenance import strip_provenance
from .ontology_checks import (
    build_to_add,
    check_to_add,
    compute_scores,
    count_warnings,
    filter_missing_concepts,
    filter_ungrounded_terms,
    format_findings,
    run_deterministic_checks,
    verify_weaknesses,
)


# Schema representing a single identified weakness in the candidate ontology
class Weakness(BaseModel):
    claim: str = Field(
        description="One-sentence description of the problem."
    )
    json_path: str = Field(
        description=(
            "Dot path in the candidate ontology where the problem is, e.g. "
            "'rights_model.duties' or 'classes[0].type'. Empty string if it "
            "concerns only the source text."
        )
    )
    quote: str = Field(
        description=(
            "Exact quote copied verbatim from the source text or from the "
            "candidate ontology that supports the claim."
        )
    )
    suggested_fix: str = Field(
        description="Concrete change to the ontology that would fix this problem."
    )


# Structured response schema expected from the LLM judge
class Review(BaseModel):
    overall_score: int = Field(ge=1, le=5)
    odrl_compliance: int = Field(ge=1, le=5)
    strengths: list[str]
    weaknesses: list[Weakness]
    missing_concepts: list[str]
    unsupported_concepts: list[str]
    hallucinations: list[str]


# Calculate the persistent group identifier for a batch
def get_batch_group_id(
    batch_id: int,
    batch_group_size: int,
) -> int:

    if batch_id < 0:
        raise ValueError(
            "batch_id must be greater than or equal to zero."
        )

    if batch_group_size <= 0:
        raise ValueError(
            "batch_group_size must be greater than zero."
        )

    return batch_id // batch_group_size


"""
Load only the ontology candidates belonging to a specific batch.

Args:
    batch_id: The ID of the target batch.
    batch_group_size: Number of batches grouped into a single partition file.
    candidates_directory: Path to the directory containing candidate files.

Returns:
    DataFrame containing candidate records for the specified batch.
"""
def load_ontology_candidates(
    batch_id: int,
    batch_group_size: int,
    candidates_directory: str,
) -> pd.DataFrame:

    group_id = get_batch_group_id(
        batch_id=batch_id,
        batch_group_size=batch_group_size,
    )

    candidates_path = (
        Path(candidates_directory)
        / f"part_{group_id:03d}.csv"
    )

    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Ontology candidates file not found: "
            f"{candidates_path}"
        )

    candidates = pd.read_csv(
        candidates_path,
        encoding="utf-8",
    )

    batch_candidates = candidates[
        candidates["batch_id"] == batch_id
    ].copy()

    if batch_candidates.empty:
        raise ValueError(
            f"No ontology candidates found for batch "
            f"{batch_id} in {candidates_path}."
        )

    return batch_candidates.reset_index(drop=True)


"""
Load the chunk partition containing text sources for the requested batch.

Args:
    batch_id: The target batch identifier.
    batch_group_size: Number of batches per partition file.
    partition_manifest: DataFrame tracking file locations per partition ID.

Returns:
    DataFrame containing source text chunks for the given partition.
"""
def load_source_chunks(
    batch_id: int,
    batch_group_size: int,
    partition_manifest: pd.DataFrame,
) -> pd.DataFrame:

    partition_id = batch_id // batch_group_size

    partition_rows = partition_manifest[
        partition_manifest["part_id"] == partition_id
    ]

    if partition_rows.empty:
        raise ValueError(
            f"Partition {partition_id} does not exist."
        )

    partition_info = partition_rows.iloc[0]

    partition_path = Path(
        partition_info["file_path"]
    )

    if not partition_path.exists():
        raise FileNotFoundError(
            f"Chunk partition file not found: "
            f"{partition_path}"
        )

    return pd.read_csv(
        partition_path,
        encoding="utf-8",
    )


# Queries the LLM judge with structured Pydantic schema validation and retry logic
def _call_judge(
    backend: dict,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    seed: int,
    num_ctx: int,
    num_predict: int,
    max_retries: int,
) -> tuple[Review | None, str | None]:

    last_error = "unknown error"
    is_local = backend.get("provider", "ollama") == "ollama"

    for attempt in range(max_retries + 1):
        try:
            response = llm_chat(
                backend, model, system_prompt, user_prompt, temperature,
                schema_model=Review,
                seed=seed + attempt,
                num_ctx=num_ctx,
                max_tokens=num_predict,
            )
        except Exception as exc:
            last_error = f"llm call failed: {exc}"
            continue

        # Check context window overflow for local Ollama deployments
        if is_local:
            used_tokens = response.prompt_tokens + response.output_tokens
            if used_tokens >= num_ctx:
                return None, (
                    f"context window exceeded "
                    f"({used_tokens} tokens >= num_ctx={num_ctx})"
                )

        # Retry if response was cut off due to max output length
        if response.done_reason == "length":
            last_error = "output truncated (done_reason=length)"
            continue

        try:
            return Review.model_validate_json(response.content), None
        except ValidationError as exc:
            last_error = f"invalid review JSON ({exc.error_count()} validation errors)"

    return None, last_error


# Merge the LLM review with the code-side verification and decide approval
def _build_review(
    review: Review,
    onto: dict,
    ontology_json: str,
    chunk_text: str,
    findings: list[dict],
    thresholds: dict,
) -> tuple[dict, bool]:

    grounded, discarded = verify_weaknesses(
        [w.model_dump() for w in review.weaknesses],
        onto,
        ontology_json,
        chunk_text,
        findings,
    )

    errors = [f for f in findings if f["severity"] == "error"]
    # 'code': several findings with the same code are one root cause
    count_by = thresholds.get("count_by", "code")
    n_warnings = count_warnings(findings, count_by)

    # Scores come from evidence (findings + grounded weaknesses), not from
    # the LLM's opinion, which is kept only for reference.
    overall_score, odrl_compliance = compute_scores(findings, len(grounded), count_by)

    # Warnings whose code is listed in blocking_codes block approval on their own
    # (e.g. a duty taken from a recital, a value not in the text, a missed rule).
    blocking = sorted({f["code"] for f in findings
                       if f["severity"] != "info" and f["code"] in set(thresholds.get("blocking_codes") or [])})
    approved = (
        overall_score >= thresholds["min_overall_score"]
        and odrl_compliance >= thresholds["min_odrl_compliance"]
        and len(grounded) <= thresholds["max_verified_weaknesses"]
        and n_warnings <= thresholds["max_warnings"]
        and not errors
        and not blocking
    )

    # Amendments come only from automatic findings and grounded weaknesses,
    # so fixes attached to discarded claims never reach the revision step.
    # Provenance is kept: "automatic" fixes are deterministic and safe to
    # apply in code, "llm" fixes are semantic and should be verified.
    amendments_detail: list[dict] = []
    seen_fixes: set[str] = set()
    for f in findings:
        if f["severity"] == "info":
            continue
        fix = f.get("fix")
        if fix and fix not in seen_fixes:
            seen_fixes.add(fix)
            amendments_detail.append(
                {"source": "automatic", "code": f["code"], "fix": fix, "data": f.get("data")}
            )
    for w in grounded:
        fix = w.get("suggested_fix")
        if fix and fix not in seen_fixes:
            seen_fixes.add(fix)
            amendments_detail.append(
                {"source": "llm", "code": None, "fix": fix, "json_path": w.get("json_path")}
            )

    # Hallucinated content (placeholder parties, unsupported actions) is better
    # regenerated than patched.
    recommended_action = "regenerate" if errors else "patch"

    missing = filter_missing_concepts(review.missing_concepts, onto, chunk_text)
    review_dict = {
        "status": "ok",
        "overall_score": overall_score,
        "odrl_compliance": odrl_compliance,
        "llm_overall_score": review.overall_score,
        "llm_odrl_compliance": review.odrl_compliance,
        "json_valid": True,
        "blocking_findings": blocking,
        "strengths": review.strengths,
        "weaknesses": grounded,
        "discarded_weaknesses": discarded,
        "missing_concepts": missing,
        # what the revision must add, checked again after it (missing_not_addressed)
        "to_add": build_to_add(missing, grounded, findings, chunk_text),
        "unsupported_concepts": filter_ungrounded_terms(
            review.unsupported_concepts, ontology_json, chunk_text
        ),
        "hallucinations": filter_ungrounded_terms(
            review.hallucinations, ontology_json, chunk_text
        )
        + [
            f["message"]
            for f in findings
            if f["code"] in ("ungrounded_action", "placeholder_party")
        ],
        "suggested_amendments": [d["fix"] for d in amendments_detail],
        "amendments_detail": amendments_detail,
        "recommended_action": recommended_action,
        "automatic_findings": findings,
    }

    return review_dict, approved


def _json_list(value) -> list:
    if isinstance(value, list):
        return value
    try:
        data = json.loads(value) if isinstance(value, str) and value.strip() else []
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _json_dict(value) -> dict:
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def same_content(a, b) -> bool:
    """Same ontology, ignoring provenance fields (re-located after each step)."""
    try:
        da = strip_provenance(json.loads(a) if isinstance(a, str) else a)
        db = strip_provenance(json.loads(b) if isinstance(b, str) else b)
    except (TypeError, json.JSONDecodeError):
        return False
    return json.dumps(da, sort_keys=True, ensure_ascii=False) == json.dumps(db, sort_keys=True, ensure_ascii=False)


def rows_of_round(rows: pd.DataFrame, review_round: int) -> pd.DataFrame:
    """The revisions of `review_round` (revisions are stored per chunk and round);
    files from before that change have no round column: their rows are kept."""
    if "round" not in rows.columns:
        return rows.reset_index(drop=True)
    r = rows["round"].fillna(0).astype(int)
    selected = rows[r == review_round]
    if selected.empty:      # older file: one row per chunk, latest round
        selected = rows.assign(_r=r).sort_values("_r").drop_duplicates("chunk_id", keep="last").drop(columns="_r")
    return selected.reset_index(drop=True)


def previous_reviews(batch_id: int, batch_group_size: int, reviews_directory: str,
                     review_round: int) -> dict[str, dict]:
    """chunk_id -> review row of `review_round` for this batch ({} if none)."""
    path = Path(reviews_directory) / f"part_{get_batch_group_id(batch_id, batch_group_size):03d}.csv"
    if not path.exists() or review_round < 0:
        return {}
    reviews = pd.read_csv(path, encoding="utf-8")
    if "round" not in reviews.columns:
        reviews["round"] = 0
    reviews = reviews[(reviews["batch_id"] == batch_id)
                      & (reviews["round"].fillna(0).astype(int) == review_round)]
    return {r["chunk_id"]: r.to_dict() for _, r in reviews.iterrows()}


# Executes deterministic checks and LLM judge reviews on candidate ontologies in parallel
def review_ontologies(
    batch_id: int,
    batch_group_size: int,
    candidates_directory: str,
    chunk_partition_manifest: pd.DataFrame,
    model: str,
    temperature: float,
    seed: int,
    num_ctx: int,
    num_predict: int,
    max_retries: int,
    system_prompt: str,
    user_prompt_template: str,
    approval_thresholds: dict,
    llm_backend: dict,
    review_round: int = 0,
    source: str = "candidates",
    revisions_directory: str | None = None,
    reviews_directory: str | None = None,
) -> pd.DataFrame:
    """Review one batch.

    round 0 reviews the generator's candidates (source='candidates').
    round >= 1 re-reviews the revisions of that round (source='revisions'), which
    closes the loop: a revision is only trusted once it has been reviewed.

    A revision identical to the ontology reviewed in the previous round (same
    content, provenance aside) is not sent to the LLM judge again: the previous
    review is reused (`reused_unchanged`). If it was not approved, the chunk is
    `stalled`: another round would not change it, so it is not revised again.
    """
    previous: dict[str, dict] = {}
    if source == "revisions":
        if not revisions_directory:
            raise ValueError("revisions_directory is required when source='revisions'.")
        ontology_candidates = rows_of_round(load_ontology_candidates(
            batch_id=batch_id,
            batch_group_size=batch_group_size,
            candidates_directory=revisions_directory,
        ), review_round)
        source_chunks = None          # revisions carry their own source text
        if reviews_directory:
            previous = previous_reviews(batch_id, batch_group_size, reviews_directory, review_round - 1)
    else:
        ontology_candidates = load_ontology_candidates(
            batch_id=batch_id,
            batch_group_size=batch_group_size,
            candidates_directory=candidates_directory,
        )
        source_chunks = load_source_chunks(
            batch_id=batch_id,
            batch_group_size=batch_group_size,
            partition_manifest=chunk_partition_manifest,
        )

    # Helper closure to evaluate deterministic checks and LLM judge for a single candidate
    def _review_row(row) -> dict:
        segments = row.get("segments") if isinstance(row.get("segments"), str) else None
        if source_chunks is None:
            chunk_text = "" if pd.isna(row.get("text")) else str(row["text"])
        else:
            matching_chunks = source_chunks[source_chunks["chunk_id"] == row["chunk_id"]]
            if matching_chunks.empty:
                raise ValueError(f"Source chunk not found for chunk_id {row['chunk_id']}.")
            chunk = matching_chunks.iloc[0]
            chunk_text = "" if pd.isna(chunk["chunk_text"]) else str(chunk["chunk_text"]).replace("\r\n", "\n")
            if segments is None and isinstance(chunk.get("segments"), str):
                segments = chunk["segments"]
        ontology_json = row["ontology"]
        to_add = _json_list(row.get("to_add"))

        # Unchanged since the last review: reuse it instead of asking the judge again
        prev = previous.get(row["chunk_id"])
        if prev is not None and same_content(prev["ontology"], ontology_json):
            prev_approved = str(prev.get("approved")).strip().lower() == "true"
            review_dict = _json_dict(prev.get("review"))
            review_dict.update(reused_from_round=int(prev["round"]), stalled=not prev_approved)
            return {
                "batch_id": batch_id, "chunk_id": row["chunk_id"], "round": review_round,
                "CELEX": row["CELEX"], "text": chunk_text, "ontology": ontology_json,
                "review": json.dumps(review_dict, ensure_ascii=False),
                "status": "reused_unchanged" if prev_approved else "stalled",
                "error_reason": None, "automatic_checks": prev.get("automatic_checks"),
                "approved": prev_approved, "revision_required": False,
                "segments": segments, "to_add": json.dumps(review_dict.get("to_add") or [], ensure_ascii=False),
            }

        # Run deterministic rule checks on candidate JSON; after a revision, also
        # check that what the review asked to add was added
        onto, findings = run_deterministic_checks(ontology_json, chunk_text)
        if onto is not None and to_add:
            findings += check_to_add(onto, to_add)

        status = "ok"
        error_reason = None
        approved = None
        revision_required = None

        if onto is None:
            status = "invalid_ontology"
            error_reason = findings[0]["message"]
            review_dict = {
                "status": status,
                "json_valid": False,
                "automatic_findings": findings,
            }
            approved = False
            revision_required = True
        else:
            # Format user prompt for judge
            user_prompt = user_prompt_template.format(
                text=chunk_text,
                ontology=strip_provenance(ontology_json),   # `source` dicts only cost context
                automatic_checks=format_findings(findings),
            )
            # Invoke LLM judge
            review, error_reason = _call_judge(
                backend=llm_backend,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                seed=seed,
                num_ctx=num_ctx,
                num_predict=num_predict,
                max_retries=max_retries,
            )
            # Handle judge failure or context overflow
            if review is None:
                status = "error"
                review_dict = {
                    "status": status,
                    "error_reason": error_reason,
                    "automatic_findings": findings,
                }
            else:
                # Build consolidated review and evaluate approval
                review_dict, approved = _build_review(
                    review=review,
                    onto=onto,
                    ontology_json=ontology_json,
                    chunk_text=chunk_text,
                    findings=findings,
                    thresholds=approval_thresholds,
                )
                revision_required = not approved

        return {
            "batch_id": batch_id,
            "chunk_id": row["chunk_id"],
            "round": review_round,
            "CELEX": row["CELEX"],
            "text": chunk_text,
            "ontology": ontology_json,
            "review": json.dumps(review_dict, ensure_ascii=False),
            "status": status,
            "error_reason": error_reason,
            "automatic_checks": json.dumps(findings, ensure_ascii=False),
            "approved": approved,
            "revision_required": revision_required,
            "segments": segments,      # chunk -> act offsets, for the provenance of revisions
            "to_add": json.dumps(review_dict.get("to_add") or [], ensure_ascii=False),
        }

    rows = [row for _, row in ontology_candidates.iterrows()]
    results = run_parallel(_review_row, rows, llm_backend.get("max_workers", 1))
    
    return pd.DataFrame(results)


# Append the current batch reviews to the persistent review file (unchanged)
def persist_ontology_reviews(
    ontology_reviews: pd.DataFrame,
    batch_id: int,
    batch_group_size: int,
    output_directory: str,
) -> str:

    group_id = get_batch_group_id(
        batch_id=batch_id,
        batch_group_size=batch_group_size,
    )

    output_path = Path(output_directory) / f"part_{group_id:03d}.csv"

    # one row per (chunk, round): the review history is kept
    return append_dedup_csv(output_path, ontology_reviews, key=["chunk_id", "round"])