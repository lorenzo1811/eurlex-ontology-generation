import json
import logging
import re
from pathlib import Path

import pandas as pd

from ...utils.changelog import change, stamp
from ...utils.eurlex_structure import NON_BINDING, section_of
from ...utils.llm_client import llm_chat, run_parallel
from ...utils.ontology_cleanup import (
    cleanup_ontology,
    empty_ontology,
    footnote_share,
)
from ...utils.ontology_schema import build_ontology_model
from ...utils.persist import append_dedup_csv, persist_removed_items
from ...utils.provenance import attach_provenance
from ...utils.repair import build_repairer
from ...utils.vocabulary import render_prompt
from .rules_pass import run_rules_pass

logger = logging.getLogger(__name__)


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

    # CSVs written in text mode on Windows turn the "\n" after the chunk header
    # into "\r\n", which would shift the chunk offsets (`segments`) by one.
    selected_chunks["chunk_text"] = selected_chunks["chunk_text"].str.replace("\r\n", "\n", regex=False)

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


# Generates partial ontologies for each text chunk using the local LLM in parallel
def generate_partial_ontologies(
    chunks: pd.DataFrame,
    batch_id: int,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt_template: str,
    json_mode: bool,
    llm_backend: dict,
    structured_output: bool = False,
    vocabulary_path: str | None = None,
    skip_footnote_chunks: bool = True,
    footnote_threshold: float = 0.6,
    cleanup_candidates: bool = True,
    non_binding_note: str | None = None,
    rules_pass: dict | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    max_attempts: int = 2,
    repair: dict | None = None,
) -> pd.DataFrame:

    # The system prompt carries the controlled vocabulary; render it once.
    rendered_system_prompt = render_prompt(system_prompt, vocabulary_path)
    schema_model = build_ontology_model(vocabulary_path) if structured_output else None
    # corrects unsupported values before the cleanup would remove them (utils/repair.py)
    repairer = build_repairer(repair, vocabulary_path)

    # Helper closure to process a single chunk row through the LLM pipeline
    def _generate(row) -> dict:
        chunk_text = "" if pd.isna(row["chunk_text"]) else str(row["chunk_text"])
        segments_json = row.get("segments") if isinstance(row.get("segments"), str) else None
        base = {"batch_id": batch_id, "chunk_id": row["chunk_id"], "CELEX": row["CELEX"],
                "segments": segments_json}

        def stamped(events: list) -> str:
            return json.dumps(stamp(events, stage="generation", round=0, chunk_id=row["chunk_id"],
                                    celex=row["CELEX"]), ensure_ascii=False)

        # A fragment made only of footnotes ("(7) Council Regulation ... (OJ L ...)") has no
        # rights or duties: asking the model anyway only produces invented rules.
        share = footnote_share(chunk_text)
        if skip_footnote_chunks and share >= footnote_threshold:
            notes = [change("footnote_chunk_skipped", f"footnote share {share:.2f}", target="ontology")]
            return {**base, "ontology": json.dumps(empty_ontology(), ensure_ascii=False),
                    "cleanup_notes": json.dumps(notes), "change_log": stamped(notes)}

        user_prompt = render_prompt(user_prompt_template, vocabulary_path, text=chunk_text)
        section = section_of(chunk_text)
        if non_binding_note and section in NON_BINDING:
            label = "recitals" if section == "recital" else section
            user_prompt += "\n\n" + non_binding_note.replace("{section}", label)

        # Query the LLM. With structured_output the JSON schema (enums for types,
        # is_a, actions, operands, relations) is enforced by the decoder. A failed
        # or unparsable answer is retried a little warmer (at low temperature a
        # loop repeats itself); if every attempt fails the chunk gets an empty
        # ontology and a `generation_failed` event, instead of failing the batch.
        ontology_json, failure = None, ""
        for attempt in range(max_attempts):
            try:
                response = llm_chat(
                    llm_backend, model, rendered_system_prompt, user_prompt,
                    min(temperature + 0.3 * attempt, 0.9),
                    schema_model=schema_model, json_mode=json_mode,
                    num_ctx=num_ctx, max_tokens=num_predict,
                )
            except Exception as exc:  # timeouts, server errors after the client's retries
                failure = f"attempt {attempt + 1}: {type(exc).__name__}: {str(exc)[:200]}"
                logger.warning("chunk %s: generation %s", row["chunk_id"], failure)
                continue
            parsed = extract_json(response.content)
            if isinstance(parsed, dict) and "error" not in parsed:
                ontology_json = parsed
                break
            failure = (f"attempt {attempt + 1}: invalid JSON (done_reason={response.done_reason}, "
                       f"{len(response.content or '')} chars)")
            logger.warning("chunk %s: generation %s", row["chunk_id"], failure)
        if ontology_json is None:
            notes = [change("generation_failed", failure, actor="llm_generator", target="ontology")]
            return {**base, "ontology": json.dumps(empty_ontology(), ensure_ascii=False),
                    "cleanup_notes": json.dumps(notes), "change_log": stamped(notes)}

        # Checks against the source text, before the review sees the candidate (see utils/ontology_cleanup.py)
        # Second, narrow LLM call on the article sentences that state a rule the first
        # answer missed (rules_pass.py); its rules then go through the same cleanup.
        rules_events: list = []
        if rules_pass and "error" not in ontology_json:
            try:
                rules_events = run_rules_pass(
                    ontology_json, chunk_text, rules_pass, llm_backend=llm_backend, model=model,
                    temperature=temperature, vocabulary_path=vocabulary_path,
                    structured_output=structured_output, num_ctx=num_ctx, num_predict=num_predict)
            except Exception as exc:  # the main extraction is still usable
                rules_events = [change("rules_pass_failed", str(exc)[:200], actor="llm_generator",
                                       target="rights_model")]

        notes: list = []
        if cleanup_candidates and "error" not in ontology_json:
            ontology_json, notes = cleanup_ontology(ontology_json, chunk_text, repairer=repairer)
        notes = rules_events + notes
        # Source of every class, rule and constraint: CELEX, chunk, article/paragraph,
        # act offsets and the quoted span (utils/provenance.py)
        provenance_notes = []
        if "error" not in ontology_json:
            provenance_notes = attach_provenance(
                ontology_json, chunk_text, celex=row["CELEX"], chunk_id=row["chunk_id"],
                segments=json.loads(segments_json) if segments_json else None)
        return {
            **base,
            "ontology": json.dumps(ontology_json, ensure_ascii=False),
            "cleanup_notes": json.dumps(notes, ensure_ascii=False),
            "change_log": stamped(notes + provenance_notes),
        }

    rows = [row for _, row in chunks.iterrows()]
    results = run_parallel(_generate, rows, llm_backend.get("max_workers", 1))
    
    return pd.DataFrame(results)


# Append the current batch results to the corresponding persistent CSV file.    
def persist_ontology_candidates(
    ontology_candidates: pd.DataFrame,
    batch_id: int,
    batch_group_size: int,
    output_directory: str,
    removed_items_directory: str | None = None,
) -> str:

    # Identify target partition file group
    group_id = get_batch_group_id(
        batch_id=batch_id,
        batch_group_size=batch_group_size,
    )

    output_path = Path(output_directory) / f"part_{group_id:03d}.csv"
    # removals that could not be repaired go to a review queue (one row per event)
    persist_removed_items(ontology_candidates, removed_items_directory, group_id)

    return append_dedup_csv(output_path, ontology_candidates)


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