from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd


MANIFEST_PATH = Path(
    "data/03_primary/chunk_batch_manifest.csv"
)


# Load the persistent batch manifest
def load_manifest() -> pd.DataFrame:

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Batch manifest not found: {MANIFEST_PATH}"
        )

    return pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8",
    )


# Persist the updated batch manifest
def save_manifest(manifest: pd.DataFrame) -> None:

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8",
    )


# Update the status of a single batch
def update_status(
    manifest: pd.DataFrame,
    batch_id: int,
    status: str,
) -> None:

    manifest.loc[
        manifest["batch_id"] == batch_id,
        "status",
    ] = status

    save_manifest(manifest)


# Execute one Kedro pipeline for a specific batch
def run_pipeline(
    pipeline_name: str,
    parameter_namespace: str,
    batch_id: int,
) -> int:

    command = [
        "uv",
        "run",
        "kedro",
        "run",
        "--pipelines",
        pipeline_name,
        "--params",
        f"{parameter_namespace}.batch_id={batch_id}",
    ]

    print()
    print("-" * 70)
    print(
        f"Running pipeline '{pipeline_name}' "
        f"for batch {batch_id}"
    )
    print("-" * 70)

    result = subprocess.run(
        command,
        check=False,
    )

    return result.returncode


"""
Check that ontology review produced a valid result for the batch.

The review pipeline can finish with exit code 0 even when the LLM
review itself failed. Therefore the persistent output must also
be inspected.
"""
def validate_review_output(
    batch_id: int,
    expected_count: int,
) -> bool:

    group_id = batch_id // 1000

    output_path = (
        Path("data/04_feature/ontology_reviews")
        / f"part_{group_id:03d}.csv"
    )

    if not output_path.exists():
        print(
            f"Review output not found: {output_path}"
        )
        return False

    reviews = pd.read_csv(
        output_path,
        encoding="utf-8",
    )

    batch_reviews = reviews[
        reviews["batch_id"] == batch_id
    ]

    if len(batch_reviews) != expected_count:
        print(
            f"Review output contains {len(batch_reviews)} rows, "
            f"expected {expected_count}."
        )
        return False

    failed_reviews = batch_reviews[
        batch_reviews["status"].isin(
            ["error"]
        )
    ]

    if not failed_reviews.empty:
        print(
            f"Review failed for "
            f"{len(failed_reviews)} chunk(s)."
        )
        return False

    return True


"""
Check that ontology revision produced a valid result for the batch.

A revision fallback means that the LLM revision failed, so the
batch must not be marked as completed.
"""
def validate_revision_output(
    batch_id: int,
    expected_count: int,
) -> bool:    

    group_id = batch_id // 1000

    output_path = (
        Path("data/04_feature/ontology_revisions")
        / f"part_{group_id:03d}.csv"
    )

    if not output_path.exists():
        print(
            f"Revision output not found: {output_path}"
        )
        return False

    revisions = pd.read_csv(
        output_path,
        encoding="utf-8",
    )

    batch_revisions = revisions[
        revisions["batch_id"] == batch_id
    ]

    if len(batch_revisions) != expected_count:
        print(
            f"Revision output contains "
            f"{len(batch_revisions)} rows, "
            f"expected {expected_count}."
        )
        return False

    failed_revisions = batch_revisions[
        batch_revisions["revision_status"].astype(str).str.startswith(
            "failed_"
        )
    ]

    if not failed_revisions.empty:
        print(
            f"Revision failed for "
            f"{len(failed_revisions)} chunk(s)."
        )
        return False

    return True


# Execute the complete ontology-processing pipeline for one batch
def run_batch(
    batch_id: int,
    expected_count: int,
) -> bool:   

    # Step 1: Generate candidate ontologies.
    return_code = run_pipeline(
        pipeline_name="ontology_generation",
        parameter_namespace="ontology_generation",
        batch_id=batch_id,
    )

    if return_code != 0:
        print(
            f"Ontology generation failed for batch {batch_id}."
        )
        return False

    # Step 2: Review generated ontologies.
    return_code = run_pipeline(
        pipeline_name="ontology_review",
        parameter_namespace="ontology_review",
        batch_id=batch_id,
    )

    if return_code != 0:
        print(
            f"Ontology review pipeline failed for batch {batch_id}."
        )
        return False

    # The review pipeline may exit successfully even if the LLM judge
    # returned an error for one or more chunks.
    if not validate_review_output(
        batch_id=batch_id,
        expected_count=expected_count,
    ):
        print(
            f"Ontology review validation failed for batch {batch_id}."
        )
        return False

    # Step 3: Revise ontologies that require revision.
    return_code = run_pipeline(
        pipeline_name="ontology_revision",
        parameter_namespace="ontology_revision",
        batch_id=batch_id,
    )

    if return_code != 0:
        print(
            f"Ontology revision pipeline failed for batch {batch_id}."
        )
        return False

    # The revision pipeline can keep the original ontology when
    # the LLM output is truncated or invalid. Detect that explicitly.
    if not validate_revision_output(
        batch_id=batch_id,
        expected_count=expected_count,
    ):
        print(
            f"Ontology revision validation failed for batch {batch_id}."
        )
        return False

    return True


"""
Convert interrupted 'running' batches back to 'pending'.

This allows the script to resume after an unexpected interruption.
"""
def recover_interrupted_batches(
    manifest: pd.DataFrame,
) -> pd.DataFrame:    

    running_mask = manifest["status"] == "running"

    if running_mask.any():
        recovered_count = int(running_mask.sum())

        print(
            f"Recovering {recovered_count} interrupted batch(es)."
        )

        manifest.loc[
            running_mask,
            "status",
        ] = "pending"

        save_manifest(manifest)

    return manifest


# Parse command-line arguments
def parse_arguments() -> argparse.Namespace:    

    parser = argparse.ArgumentParser(
        description=(
            "Run pending ontology-processing batches "
            "with checkpoint support."
        )
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help=(
            "Maximum number of batches to process "
            "in this execution."
        ),
    )

    return parser.parse_args()


# Process pending ontology batches sequentially
def main() -> None:   

    args = parse_arguments()

    if args.max_batches <= 0:
        raise ValueError(
            "--max-batches must be greater than zero."
        )

    manifest = load_manifest()

    # Recover batches left in 'running' state by an interrupted execution.
    manifest = recover_interrupted_batches(
        manifest
    )

    processed_batches = 0

    while processed_batches < args.max_batches:

        processable_batches = manifest[
            manifest["status"].isin(["pending", "failed"])
        ]

        if processable_batches.empty:
            print()
            print("No pending or failed batches remain.")
            return

        batch_row = processable_batches.iloc[0]

        batch_id = int(
            batch_row["batch_id"]
        )

        expected_count = int(
            batch_row["chunk_count"]
        )

        print()
        print("=" * 70)
        print(
            f"Starting complete pipeline for batch {batch_id}"
        )
        print(
            f"Expected chunks: {expected_count}"
        )
        print("=" * 70)

        update_status(
            manifest=manifest,
            batch_id=batch_id,
            status="running",
        )

        success = run_batch(
            batch_id=batch_id,
            expected_count=expected_count,
        )

        if success:
            update_status(
                manifest=manifest,
                batch_id=batch_id,
                status="completed",
            )

            print()
            print(
                f"Batch {batch_id} completed successfully."
            )

            processed_batches += 1

        else:
            update_status(
                manifest=manifest,
                batch_id=batch_id,
                status="failed",
            )

            print()
            print(
                f"Batch {batch_id} failed."
            )
            print(
                "Execution stopped."
            )

            sys.exit(1)

    print()
    print(
        f"Finished. Processed "
        f"{processed_batches} batch(es)."
    )


if __name__ == "__main__":
    main()