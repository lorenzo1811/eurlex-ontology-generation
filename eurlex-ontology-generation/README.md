# EurLex Ontology Generation

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Extracts an ODRL-aligned ontology (parties, assets, permissions, prohibitions, duties,
constraints) from EU legislation (EUR-Lex) with small local LLMs, merges the per-chunk
ontologies, refines them deterministically, links the concepts to Wikidata and EuroVoc, and
exports OWL 2 / ODRL Turtle validated with SHACL. Every class, rule and constraint keeps its
source (act, article, character offsets, quote) and every change is logged.

`QUALITY_HARNESS.md` explains the design (vocabulary, typing, provenance, repair, review loop,
merge coverage, linking, evaluation, export).

## Requirements

| What | Why | How |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | dependencies and virtual environment (`uv.lock`) | `pip install uv` or the uv installer |
| Python 3.10–3.14 (3.10–3.12 for concept linking) | runtime | uv downloads it if missing |
| [Ollama](https://ollama.com) | local LLMs and embeddings | install, then `ollama serve` |
| Ollama models | generation, review/revision/repair/linking choice, merge embeddings | `ollama pull qwen2.5:3b`, `ollama pull gemma3:4b`, `ollama pull all-minilm` |
| `data/01_raw/EurLex_all.csv` | the corpus (not in the repository) | the CEPS EurLex dataset (Harvard Dataverse); put the CSV in `data/01_raw/` |
| A GPU (optional, recommended) | the LLM loop and ReLiK are slow on CPU | any CUDA machine; Ollama and torch use it automatically |
| Internet (concept linking only) | Wikipedia API, Publications Office SPARQL endpoint, Hugging Face models | — |

Other LLM providers (any OpenAI-compatible server, Anthropic) can replace Ollama: see the
backend profiles in `conf/base/globals.yml`. API keys are read only from environment variables.

## Install

Run every command from this directory (the Kedro project; the git root is its parent).

```
uv sync                        # runtime dependencies
uv sync --extra dev            # + pytest, ruff
```

Optional extras: `reasoner` (owlready2 for the HermiT check, needs Java), `agent` (MCP
curation server), `linking` (ReLiK + torch for concept linking; Python <= 3.12, best on a GPU).
Extras are installed on demand by `uv run --extra <name> ...` as well.

`requirements.txt` lists the same direct dependencies for pip users, but `uv.lock` is the
reference: `pip install -r requirements.txt && pip install -e .` gives unpinned versions.

## Run

Always name the pipeline: a bare `kedro run` runs every pipeline, including all LLM steps.

1. **Setup** (once, or when the chunking changes; no LLM):
   ```
   uv run kedro run --pipelines eurlex_ingestion
   uv run kedro run --pipelines eurlex_chunking
   uv run kedro run --pipelines chunk_batching
   uv run kedro run --pipelines chunk_storage
   ```
   Structural chunks (one article / recital block / annex per chunk) and the batch manifest go
   to `data/02_intermediate/` and `data/03_primary/`.
2. **Generation → review → revision** per batch (LLM; checkpointed, resumable):
   ```
   uv run python scripts/run_ontology_batches.py --start-batch 0 --end-batch 9 --max-batches 10
   ```
   Flags: `--parallel`, `--review-rounds` (default 2). State in
   `data/03_primary/chunk_batch_manifest.csv`; outputs in `data/04_feature/`.
3. **Merge → refinement → export**:
   ```
   uv run python scripts/run_ontology_merge.py --force
   ```
   Refined ontology `data/07_model_output/eurlex_odrl_ontology_refined.json`, Turtle
   `data/07_model_output/eurlex_odrl_ontology.ttl`, reports in `data/08_reporting/`
   (SHACL, merge coverage, refinement review queue, removed items).
4. **Concept linking** (Wikidata + EuroVoc / EU corporate bodies; ReLiK, CC BY-NC-SA 4.0):
   ```
   uv run --extra linking python scripts/link_concepts.py index        # once
   uv run --extra linking python scripts/link_concepts.py link --apply # after every merge
   ```
   `--apply` re-runs refinement and export, so the links reach the refined JSON and the Turtle.
   Without ReLiK: `link --no-relik --apply` (exact Wikidata labels). Settings:
   `conf/base/parameters_concept_linking.yml`.

Parameters live in `conf/base/parameters_<pipeline>.yml` (namespaced:
`--params ontology_generation.batch_id=5`); local overrides go in `conf/local/` (git-ignored).

## Evaluate

```
uv run python scripts/batch_report.py --candidates ... --reviews ... --revisions ... --refined ... --out report.json
uv run python scripts/make_annotation_sheet.py --n 120 --annotator A          # typing gold set
uv run python scripts/make_annotation_sheet.py --links --n 60 --annotator A   # link gold set
uv run python scripts/evaluate_ontology.py data/07_model_output/eurlex_odrl_ontology_refined.json \
    --gold data/08_reporting/annotation/annotation_sheet_A.csv \
    --link-gold data/08_reporting/annotation/link_sheet_A.csv
```

`batch_report.py` gives per-run metrics (rules per type, repairs and removals, review loop,
merge coverage, linked concepts). The annotation sheets are filled by people; scores avoid
comparing the gazetteer with itself (see `QUALITY_HARNESS.md`).

## Human curation

`scripts/curation.py list|show|accept` reviews proposals written by the curation agent
(`uv run --extra agent python -m eurlex_ontology_generation.agent.mcp_server --root .`). The
agent never edits the ontology; only accepted decisions are applied by the refinement.

## Tests

```
uv run pytest tests/ -k "not test_run and not pipelines" --no-cov
uv run ruff check .
```

`tests/pipelines/*` are empty Kedro stubs and `tests/test_run.py` is template code.

## Layout

| Path | Content |
|---|---|
| `conf/base/` | parameters per pipeline, data catalog, LLM backend profiles (`globals.yml`) |
| `src/eurlex_ontology_generation/pipelines/` | Kedro pipelines (setup, generation, review, revision, merge, refinement, export) |
| `src/eurlex_ontology_generation/utils/` | shared code: chunking, LLM client, provenance, change log, repair, merge, linking, export |
| `src/eurlex_ontology_generation/resources/` | controlled vocabulary, gazetteer, SHACL shapes |
| `scripts/` | orchestration, reports, evaluation, curation, linking |
| `data/` | numbered Kedro data layers (mostly git-ignored) |
| `publish/w3id/` | redirects for the `https://w3id.org/eurlex-odrl/` namespace |
