import json
from pathlib import Path

import pandas as pd
from ollama import chat
from pydantic import BaseModel, Field, ValidationError

from .ontology_checks import (
    compute_scores,
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


# Call the judge with structured output, retrying on failure.
# Returns (Review, None) on success or (None, error_reason) on failure.
def _call_judge(
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

    for attempt in range(max_retries + 1):
        try:
            response = chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": temperature,
                    # different seed per attempt, otherwise a retry at
                    # temperature 0 would return the same failure
                    "seed": seed + attempt,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                },
                format=Review.model_json_schema(),
            )
        except Exception as exc:  # connection errors, model not found, ...
            last_error = f"ollama call failed: {exc}"
            continue

        # Ollama silently truncates prompts that exceed num_ctx.
        # Heuristic: if prompt + output fill the whole window, flag it.
        used_tokens = (
            (response.prompt_eval_count or 0)
            + (response.eval_count or 0)
        )
        if used_tokens >= num_ctx:
            return None, (
                f"context window exceeded "
                f"({used_tokens} tokens >= num_ctx={num_ctx})"
            )

        if response.done_reason == "length":
            last_error = "output truncated (done_reason=length)"
            continue

        try:
            return (
                Review.model_validate_json(response.message.content),
                None,
            )
        except ValidationError as exc:
            last_error = (
                f"invalid review JSON "
                f"({exc.error_count()} validation errors)"
            )

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
    warnings = [f for f in findings if f["severity"] == "warning"]

    # Scores come from evidence (findings + grounded weaknesses), not from
    # the LLM's opinion, which is kept only for reference.
    overall_score, odrl_compliance = compute_scores(findings, len(grounded))

    approved = (
        overall_score >= thresholds["min_overall_score"]
        and odrl_compliance >= thresholds["min_odrl_compliance"]
        and len(grounded) <= thresholds["max_verified_weaknesses"]
        and len(warnings) <= thresholds["max_warnings"]
        and not errors
    )

    # Amendments come only from automatic findings and grounded weaknesses,
    # so fixes attached to discarded claims never reach the revision step.
    # Provenance is kept: "automatic" fixes are deterministic and safe to
    # apply in code, "llm" fixes are semantic and should be verified.
    amendments_detail: list[dict] = []
    seen_fixes: set[str] = set()
    for f in findings:
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

    review_dict = {
        "status": "ok",
        "overall_score": overall_score,
        "odrl_compliance": odrl_compliance,
        "llm_overall_score": review.overall_score,
        "llm_odrl_compliance": review.odrl_compliance,
        "json_valid": True,
        "strengths": review.strengths,
        "weaknesses": grounded,
        "discarded_weaknesses": discarded,
        "missing_concepts": filter_missing_concepts(
            review.missing_concepts, onto, chunk_text
        ),
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


# Review all ontology candidates belonging to one batch
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
) -> pd.DataFrame:

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

    results = []

    for _, row in ontology_candidates.iterrows():

        matching_chunks = source_chunks[
            source_chunks["chunk_id"] == row["chunk_id"]
        ]

        if matching_chunks.empty:
            raise ValueError(
                f"Source chunk not found for chunk_id "
                f"{row['chunk_id']}."
            )

        chunk = matching_chunks.iloc[0]
        chunk_text = (
            ""
            if pd.isna(chunk["chunk_text"])
            else str(chunk["chunk_text"])
        )
        ontology_json = row["ontology"]

        # 1. Deterministic checks (no LLM)
        onto, findings = run_deterministic_checks(
            ontology_json,
            chunk_text,
        )

        status = "ok"
        error_reason = None
        approved = None
        revision_required = None

        if onto is None:
            # Invalid candidate: no point in asking the judge.
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
            # 2. LLM judge (structured output + retries)
            user_prompt = user_prompt_template.format(
                text=chunk_text,
                ontology=ontology_json,
                automatic_checks=format_findings(findings),
            )

            review, error_reason = _call_judge(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                seed=seed,
                num_ctx=num_ctx,
                num_predict=num_predict,
                max_retries=max_retries,
            )

            if review is None:
                # Judge failed: NOT a real review. approved stays empty so
                # downstream steps can filter on status == "ok".
                status = "error"
                review_dict = {
                    "status": status,
                    "error_reason": error_reason,
                    "automatic_findings": findings,
                }
            else:
                # 3. Verify the judge's claims and decide approval in code
                review_dict, approved = _build_review(
                    review=review,
                    onto=onto,
                    ontology_json=ontology_json,
                    chunk_text=chunk_text,
                    findings=findings,
                    thresholds=approval_thresholds,
                )
                revision_required = not approved

        results.append(
            {
                "batch_id": batch_id,
                "chunk_id": row["chunk_id"],
                "CELEX": row["CELEX"],
                "text": chunk_text,
                "ontology": ontology_json,
                "review": json.dumps(
                    review_dict,
                    ensure_ascii=False,
                ),
                "status": status,
                "error_reason": error_reason,
                "automatic_checks": json.dumps(
                    findings,
                    ensure_ascii=False,
                ),
                "approved": approved,
                "revision_required": revision_required,
            }
        )

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

    output_path = (
        Path(output_directory)
        / f"part_{group_id:03d}.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        existing = pd.read_csv(
            output_path,
            encoding="utf-8",
        )

        combined = pd.concat(
            [
                existing,
                ontology_reviews,
            ],
            ignore_index=True,
        )

        combined = combined.drop_duplicates(
            subset=["chunk_id"],
            keep="last",
        )
    else:
        combined = ontology_reviews.copy()

    combined.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return str(output_path)