import json
import re
import pandas as pd
from ollama import chat
from pathlib import Path


# Load only the partition containing the requested batch.
def select_chunk_batch(
    batch_manifest: pd.DataFrame,
    batch_id: int,
    batch_group_size: int,
    partition_manifest: pd.DataFrame,
) -> pd.DataFrame:

    # Locate the target batch in the global batch manifest
    batch_rows = batch_manifest[
        batch_manifest["batch_id"] == batch_id
    ]

    # Validate existence of requested batch
    if batch_rows.empty:
        raise ValueError(
            f"Batch {batch_id} does not exist."
        )

    # Calculate the corresponding partition identifier
    partition_id = batch_id // batch_group_size

    # Locate partition metadata in partition manifest
    partition_rows = partition_manifest[
        partition_manifest["part_id"] == partition_id
    ]

    # Validate existence of calculated partition
    if partition_rows.empty:
        raise ValueError(
            f"Partition {partition_id} does not exist."
        )

    # Extract target partition record and file path
    partition_info = partition_rows.iloc[0]
    partition_path = Path(
        partition_info["file_path"]
    )

    # Ensure target partition CSV exists on disk
    if not partition_path.exists():
        raise FileNotFoundError(
            f"Partition file not found: {partition_path}"
        )

    # Load partition DataFrame from disk
    partition = pd.read_csv(
        partition_path,
        encoding="utf-8",
    )

    # Retrieve row boundary markers for global batch
    batch_info = batch_rows.iloc[0]
    start_row = int(batch_info["start_row"])
    end_row = int(batch_info["end_row"])

    # Retrieve global starting row index of current partition
    partition_start_row = int(
        partition_info["start_row"]
    )

    # Convert global row boundaries into relative local indices for current partition
    local_start = start_row - partition_start_row
    local_end = end_row - partition_start_row

    # Slice target batch chunks locally from partition
    selected_chunks = partition.iloc[
        local_start:local_end + 1
    ].copy()

    # Validate that sliced count matches expected batch size
    expected_count = int(
        batch_info["chunk_count"]
    )

    if len(selected_chunks) != expected_count:
        raise ValueError(
            f"Batch {batch_id} contains "
            f"{len(selected_chunks)} chunks, "
            f"expected {expected_count}."
        )

    # Return selected batch chunks with clean index
    return selected_chunks.reset_index(drop=True)


# Calculate the persistent group identifier for a batch.
def get_batch_group_id(
    batch_id: int,
    batch_group_size: int,
) -> int:    

    # Validate boundary conditions
    if batch_id < 0:
        raise ValueError("batch_id must be greater than or equal to zero.")

    if batch_group_size <= 0:
        raise ValueError("batch_group_size must be greater than zero.")

    # Compute zero-based partition group index via integer division
    return batch_id // batch_group_size


# Generate candidate ontologies for input text chunks using Ollama LLM
def generate_partial_ontologies(
    chunks: pd.DataFrame,
    batch_id: int,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt_template: str,
    json_mode: bool,
) -> pd.DataFrame:
    results = []

    # Iterate over every chunk present in the dataset batch
    for _, row in chunks.iterrows():
        # Build the specific prompt by injecting current chunk text
        user_prompt = user_prompt_template.format(
            text=row["chunk_text"]
        )

        # Send execution request to local Ollama service
        response = chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={"temperature": temperature},
            format="json" if json_mode else None,
        )

        # Extract LLM output content
        ontology_text = response["message"]["content"]
        ontology_json = extract_json(ontology_text)

        # Record the generated metadata along with the batch context
        results.append(
            {
                "batch_id": batch_id,
                "chunk_id": row["chunk_id"],
                "CELEX": row["CELEX"],
                "ontology": json.dumps(
                    ontology_json,
                    ensure_ascii=False,
                ),
            }
        )

    return pd.DataFrame(results)


# Append the current batch results to the corresponding persistent CSV file.    
def persist_ontology_candidates(
    ontology_candidates: pd.DataFrame,
    batch_id: int,
    batch_group_size: int,
    output_directory: str,
) -> str:    

    # Identify target partition file group
    group_id = get_batch_group_id(
        batch_id=batch_id,
        batch_group_size=batch_group_size,
    )

    # Construct partition filename formatted with 3-digit padding (e.g., part_000.csv)
    output_path = (
        Path(output_directory)
        / f"part_{group_id:03d}.csv"
    )

    # Ensure parent directories exist
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load existing file to merge results if it exists
    if output_path.exists():
        existing = pd.read_csv(output_path)

        # Concatenate prior file content with newly generated candidates
        combined = pd.concat(
            [existing, ontology_candidates],
            ignore_index=True,
        )

        # Deduplicate entries by chunk_id, preserving the latest run data
        combined = combined.drop_duplicates(
            subset=["chunk_id"],
            keep="last",
        )
    else:
        combined = ontology_candidates.copy()

    # Save consolidated dataset to CSV
    combined.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return str(output_path)


# Helper to safely extract JSON from LLM output
def extract_json(text: str) -> dict:

    try:
        return json.loads(text)     # Direct parsing

    except json.JSONDecodeError:
        # Regex fallback: find anything between the first '{' and last '}'
        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Return error payload if all parsing attempts fail
    return {
        "error": "Invalid JSON generated",
        "raw_output": text
    }