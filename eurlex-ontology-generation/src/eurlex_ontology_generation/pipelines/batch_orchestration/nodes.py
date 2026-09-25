from pathlib import Path
import pandas as pd


# Update the status of a specific batch in the persistent manifest.
def update_batch_status(
    batch_manifest: pd.DataFrame,
    batch_id: int,
    status: str,
    manifest_path: str,
) -> pd.DataFrame:    

    allowed_statuses = {
        "pending",
        "running",
        "completed",
        "failed",
    }

    if status not in allowed_statuses:
        raise ValueError(
            f"Invalid status '{status}'. "
            f"Allowed statuses: {sorted(allowed_statuses)}"
        )

    matching_rows = batch_manifest[
        batch_manifest["batch_id"] == batch_id
    ]

    if matching_rows.empty:
        raise ValueError(
            f"Batch {batch_id} does not exist in the manifest."
        )

    updated_manifest = batch_manifest.copy()

    updated_manifest.loc[
        updated_manifest["batch_id"] == batch_id,
        "status",
    ] = status

    output_path = Path(manifest_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    updated_manifest.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return updated_manifest