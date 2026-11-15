# Ontology quality harness

This branch adds a harness around the LLM steps so that typing, vocabulary and
review quality are enforced and measured in code, not left to the prompts.

## What changed, by issue

### 0. Structural chunking
`utils/eurlex_structure.py` parses the one-line corpus text of an act into
preamble (title + citations), recitals, articles and their numbered paragraphs,
signatures, annexes and footnotes. Numbered units count only when they follow
the expected numbering and start a sentence, so inline footnote markers
("Agreement (2) with") and cross-references ("Article 5 of Regulation ...") are
not boundaries. `eurlex_chunking` (`strategy: structural`) packs consecutive
units of one section up to `chunk_size` characters, splits long units at
sentence ends, never mixes recitals and articles, and leaves footnotes and
signatures out. Each chunk starts with a header line (`§ Article 3(2)–(4)`,
`§ Recitals (1)–(6)`), so review and revision, which only see the text, know the
section: `provision` comes from it instead of the old regex guess, and recital
or preamble chunks are asked for concepts only (`non_binding_note`). Chunks
carry `section_type`, `articles`, `paragraphs`, `recitals`, `char_start`,
`char_end`, `segments` (chunk position -> act offset, see
`chunk_to_act_offset`) and the `footnotes` they reference. On a 2,000-act
sample: no word cut, 2 of ~13,900 chunks with a footnote left, offsets exact.
The full corpus gives 1.04M chunks instead of 0.89M; annexes are a third of
them (`include_annexes: false` drops them). `strategy: window` keeps the old
fixed windows, now cut at word boundaries.

### 0b. Traceability: source provenance and change log
**Provenance** (`utils/provenance.py`). The schema asks the LLM for an
`evidence` quote on every class and rule. After generation and after every
revision, `attach_provenance` locates it in the chunk (exact, then
whitespace/quote/case-normalised, then fuzzy: the sentence sharing most words)
or, without a usable quote, infers the sentence from the element's own words.
Each class, rule and constraint gets a `source`: CELEX, chunk_id, section,
article, paragraph or recital, chunk offsets, act offsets (through the chunk
`segments`), the quoted span and `match` (exact / normalized / fuzzy / inferred
/ none). A quote not found in the text is logged as `evidence_unverified`.
Merge and refinement turn `source` into `provenance`, a list of up to 10 such
items per element. The Turtle links every class, individual, rule and
constraint with `prov:wasDerivedFrom` to `eo:TextEvidence` nodes carrying
`dct:source` (the CELLAR CELEX IRI), `oa:TextQuoteSelector` and
`oa:TextPositionSelector`; `eosh:EvidenceShape` warns about elements without
evidence. Class links to Wikidata / EU authority IRIs are exported as
`skos:exactMatch` (individuals keep `owl:sameAs`). `source` fields are removed
from the ontology shown to the LLM judge and reviser (`strip_provenance`).

**Change log** (`utils/changelog.py`). Every change is an event with `code`
(reason), `detail`, `actor` (auto_check, llm_reviewer, llm_reviser, normalizer,
merger, human), `target` (JSON path or element key), `before`, `after`, and,
once stored, `stage`, `round`, `chunk_id`, `celex`. Cleanup and normalisation
steps emit before/after events (previously silent ones too:
`provision_defaulted`, `extensions_synced`, duplicate relationships, examples of
removed entities). The revision step records what the LLM changed with respect
to the reviewed ontology (`diff_ontologies`: `llm_revision_added/removed/
modified`, element by element). Rows of candidates and revisions carry a
`change_log` column; the refinement report has `events`, with accepted curation
decisions under actor `human`. `scripts/batch_report.py` summarises both
(provenance match per element type, events per actor and code).

### 0c. Missing permissions and prohibitions
Batch 46 and the first structural run (batches 49–53) gave duties only, although
the articles contain "may" and "shall not" sentences. Causes: the prompt
template showed an example under `duties` only, a 3B model filling a whole
ontology at once leaves most rule lists empty, and labels the model invents
("approve_release") are dropped by the grounding check.
- `utils/deontic.py` finds the deontic sentences of a chunk and the list each
  belongs to (prohibitions: shall/may/must not, prohibited; permissions: may,
  entitled/authorised to; duties: shall, must, required to), skipping formal
  clauses (entry into force, application, "shall apply", addressees, binding
  effect), definitions and short list lead-ins.
- Review check `modal_without_rule` (warning, ODRL code): an article sentence of
  that kind that no rule rests on (rule `source` span, or its verb). The finding
  quotes the sentence and names the list; the revision prompt tells the model
  to add exactly that rule. `rights_model_empty` is no longer raised on recital
  and preamble chunks, and is only informational when the chunk states no rule.
- Rules-only pass (`ontology_generation.rules_pass`, on by default for article
  chunks): a second, narrow call sends the uncovered deontic sentences with
  their list and gets back only a rights model (constrained to its schema).
  New rules get `provision` from the section and stub classes for undefined
  parties/assets, are logged as `rule_added_by_rules_pass` (actor
  `llm_generator`), then go through the normal cleanup, grounding and
  provenance.
- Generation and revision templates show one example per list, the prompt maps
  modal verbs to lists, and `permission_duty_conflict` keeps both rules.

### 1. Inconsistent typing
| Layer | Where | What it does |
|---|---|---|
| Controlled vocabulary | `src/.../resources/vocabulary.yml` | One file defines class types (`Party`, `Asset`, `Legal_extension`), a 20-class upper ontology (`is_a`), canonical actions, ODRL constraint operands/operators, canonical relations and reserved ODRL terms. |
| Constrained decoding | `utils/ontology_schema.py`, `ontology_generation` | The JSON schema built from the vocabulary is passed to Ollama's `format`, so `type`, `is_a`, `kind`, `action`, `left_operand`, `operator` and `relation` are enums the model cannot leave (`structured_output: true`). |
| Prompt fix | `parameters_ontology_generation.yml` | The example no longer says `"type": "ODRL_core"` (the generator copied it; the reviewer then penalised it on almost every chunk). Prohibitions are in the schema again (the revision schema had dropped them). Recitals are flagged as non-binding. |
| Gazetteer | `src/.../resources/gazetteer.yml`, `utils/entity_linking.py` | Countries (with EU Vocabularies country IRIs), EU institutions and bodies, international organisations and named funds/programmes. A match fixes type, `is_a` and `kind` deterministically and force-merges aliases. |
| Deterministic typer | `utils/ontology_normalize.py` | Gazetteer > declared `is_a` > head-noun keyword (`...Fund` → `Fund`) > declared type. |
| Checks | `ontology_review/ontology_checks.py` | New findings: `invalid_class_type`, `gazetteer_type_mismatch`, `rule_slot_type` (party not a Party, asset is a Party). |
| Refinement | `pipelines/ontology_refinement` | Re-types the merged ontology globally and reports slot-typing conflicts that remain. |

### 2. Classes that should be individuals
`utils/entity_linking.detect_kind` decides class vs individual, in order:
gazetteer hit → name built on a known individual (`Cyprus_Government` is an
individual, `EGF_Mobilization` is a class of mobilisations of the EGF) →
all-caps acronym (low confidence) → declared `kind`. Optionally, low-confidence
decisions are checked on Wikidata (`P31` instance-of vs `P279` subclass-of,
exact label match only, cached; `ontology_refinement.wikidata.enabled`). EU
sources come first because EU-specific constructs are rarely in Wikidata.
The refined ontology has separate `classes` and `individuals` (with
`instance_of`), and low-confidence decisions go to `review_queue` in the report.

### 3. Meta-model leakage and uncontrolled vocabulary
- Leakage: class names built on ODRL structure (`AuditDuty`,
  `AuditPermissionTarget`, `Assignee`) are flagged by the review, removed by
  the normaliser and rejected by the merge. `ImportDuty`, `RuleOfLaw`,
  `NetworkOperator` are allowed (see `reserved_terms` in the vocabulary).
- Actions: free-text actions are split into a canonical verb plus an object
  (`mobilize_egf` → `mobilise` + `egf`), each verb mapped to its ODRL parent
  action. Unknown verbs are kept but marked `controlled: false`.
- Constraints: converted to `{left_operand, operator, right_operand, unit}`;
  dates become ISO, amounts become digits plus currency. Unmatched qualitative
  conditions fall back to the extension operand `condition` with
  `controlled: false`.
- Relations: closed set; inverse forms are flipped (`funded_by` → `funds`).

### 4. Review verdicts
- Scoring counts findings per distinct code (`approval_thresholds.count_by:
  code`): ten mistyped classes are one root cause, not ten. `finding`
  restores the old behaviour.
- `info` findings (vocabulary drift fixed in code) never affect approval.
- **Re-review loop**: `scripts/run_ontology_batches.py --review-rounds N`
  runs generate → review(0) → revise(1) → review(1) → … and stops when every
  chunk is approved. Reviews are stored per `(chunk_id, round)`; the revision
  reads the latest round.
- The merge joins the latest verdict and records `support.approved_chunks`
  for every class and rule; `exclude_unapproved: true` merges only approved
  chunks.

## Evaluation harness
```bash
uv run python scripts/evaluate_ontology.py \
  data/06_models/eurlex_odrl_ontology_merged.json \
  data/07_model_output/eurlex_odrl_ontology_refined.json \
  --reviews-dir data/04_feature/ontology_reviews \
  --gold conf/base/evaluation/gold_entities.yml \
  --out data/08_reporting/ontology_evaluation.json
```
Reports typing, kind, leakage, action/constraint/relation control,
consistency, linking and review approval per round, plus gold-set accuracy.

### Avoiding circular numbers
The gazetteer, keyword typer and seed gold set were written together, so any
score that compares one of them with another proves nothing. The evaluation
therefore separates:

| Number | What it compares | Evidence? |
|---|---|---|
| `typing.pct_known_entities_correct_type` | entity types vs gazetteer facts, **excluding entities the gazetteer typed** | yes on raw generator output; not reported (`null`) on the refined ontology, where every known entity was typed by the lookup |
| Held-out typer test (`typer_ablation`) | the deterministic typer with the gazetteer **switched off**, on every gazetteer name and alias | yes: the gazetteer's labels are facts about the world, not typer output |
| Gold `held_out` | independent labels (blind annotation sheets or `source: expert`) on entities **not** typed by the gazetteer | yes, this is the headline accuracy |
| Gold `gazetteer_lookups` | independent labels on gazetteer-typed entities | checks the gazetteer's content only |
| Gold `seed_smoke_test` | `conf/base/evaluation/gold_entities.yml` (`source: seed`) | no: regression test only |

Held-out typer result on the current vocabulary, measured once and not tuned
afterwards: **19% of the 95 canonical gazetteer names get the right core type
from keywords alone** (6% of countries, 64% of funds). The refined ontology's
correct typing of known entities comes from the gazetteer, and unknown named
entities depend on the generator's declared type plus the keyword fallback,
which is weak. Only blind annotation can measure that path:

```bash
uv run python scripts/make_annotation_sheet.py --n 120 --annotator A   # same seed -> same sample
uv run python scripts/make_annotation_sheet.py --n 120 --annotator B
# two people fill type / is_a / kind from the text (the sheet hides our labels), then:
uv run python scripts/evaluate_ontology.py data/07_model_output/eurlex_odrl_ontology_refined.json \
  --gold data/08_reporting/annotation/annotation_sheet_A.csv \
  --agreement data/08_reporting/annotation/annotation_sheet_A.csv data/08_reporting/annotation/annotation_sheet_B.csv
```
Report held-out accuracy only together with inter-annotator kappa; adjudicate
disagreements into a third sheet with `source` = `adjudicated`.

## OWL/ODRL export and validation
`uv run kedro run --pipelines ontology_export` (also run by
`scripts/run_ontology_merge.py`) turns the refined JSON into
`data/07_model_output/eurlex_odrl_ontology.ttl` and checks it three ways:

| Check | Tool | Fails the run on |
|---|---|---|
| Turtle round trip | rdflib | any triple lost |
| SHACL (`resources/shapes.ttl`) | pySHACL | any `sh:Violation` (`fail_on_violation`) |
| OWL 2 consistency | HermiT via owlready2 (optional, needs Java) | inconsistency or unsatisfiable class |

Graph layout: `eo:Party ≡ odrl:Party`, `eo:Asset ≡ odrl:Asset` and
`eo:LegalExtension` are pairwise disjoint; the 20 upper classes sit under them;
domain classes are `owl:Class` + `rdfs:subClassOf` an upper class; individuals
are `owl:NamedIndividual` typed with an upper class, `owl:sameAs` their Wikidata
item and `skos:exactMatch` their EU Vocabularies IRI. Rules are
`odrl:Permission/Prohibition/Duty` with typed constraints (`xsd:date`,
`xsd:decimal`, `odrl:unit` = EU currency IRI), grouped in **one `odrl:Set` per
act** (`policy/act/<CELEX>`). Each act policy carries `eo:celex`, `eo:actType`
(Regulation / Directive / Decision / Recommendation, parsed from the CELEX
number), `eo:year`, `dct:title` and `dct:issued` (from `eurlex_corpus.csv`) and
`rdfs:seeAlso` the EUR-Lex page and the CELLAR CELEX IRI. A rule found in several
acts is attached to each of them; rules without a source go to
`policy/unattributed`. Provenance (`sources`, a list of CELEX numbers) is kept on
every class and rule through the partition and global merges.

**Class promotion.** A domain class is `eo:status "accepted"` when it occurs in
at least `min_documents_for_class` acts (default 3, in
`parameters_ontology_refinement.yml`), otherwise `"candidate"` (SHACL Info).
Candidates one act short of the threshold go to the review queue as
`class_promotion` items; a person can promote or demote any class by a curation
decision.

### Namespace
IRIs live under the persistent namespace `https://w3id.org/eurlex-odrl/`
(`ontology#` vocabulary, `entity/`, `action/`, `policy/`, `shapes#`). To make
them resolve:
1. merge this branch into `main` (the redirects point at files on `main`);
2. fork https://github.com/perma-id/w3id.org, copy `publish/w3id/eurlex-odrl/`
   (`.htaccess` + `README.md`) to the root of the fork as `eurlex-odrl/`, and
   open a pull request.
RDF clients are then redirected (303) to the Turtle files, browsers to this
document.

SHACL severities: **Violation** = structurally broken (rule without action or
assignee, non-ODRL operator, entity under two disjoint core classes, core type
contradicting its parent, meta-model term as a name, dangling reference);
**Warning** = modelling problem needing a human decision (assignee not a Party,
target is a Party, duty from a recital, free-text `condition` constraint,
undated `dateTime`, action outside the vocabulary, isolated entity);
**Info** = individual without an external identifier.
Results: `data/08_reporting/ontology_export_summary.json` and `shacl_report.ttl`.

## Identifiers
`resources/gazetteer.yml` now carries Wikidata QIDs for 91 entries (collected
from web-search results whose page title exactly matched the label; ambiguous
cases have a `wikidata_note`) and EU Vocabularies IRIs for all countries plus the
Commission's corporate-body IRI. On a machine with internet access:
```bash
uv run python scripts/link_identifiers.py --verify                       # label + P31 check of every QID
uv run python scripts/link_identifiers.py --corporate-body-rdf <file> --write   # fill corporate-body IRIs
```

## LLM backends
Each LLM step (`ontology_generation`, `ontology_review`, `ontology_revision`)
and the embedding step of both merges has its own `llm_backend` /
`embedding_backend`, chosen from the profiles in `conf/base/globals.yml`:

| Profile | Provider | Key (environment variable) |
|---|---|---|
| `local` (default) | Ollama on this machine | none |
| `ollama_remote` | Ollama on another host (`base_url`) | none |
| `vllm`, `lmstudio` | OpenAI-compatible server | none |
| `openrouter` | OpenAI-compatible | `OPENROUTER_API_KEY` |
| `openai` | OpenAI | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages API | `ANTHROPIC_API_KEY` |

Example `conf/local/parameters_ontology_review.yml` (git-ignored):
```yaml
ontology_review:
  model: anthropic/claude-sonnet-4.5
  llm_backend: ${globals:llm_backends.openrouter}
```
Keys are read from the environment only, never from parameter files. Structured
output uses a JSON schema where the server supports it (falls back to JSON mode,
then plain prompting, once per server); Anthropic uses a forced tool call.
Rate limits (429) and server errors are retried with backoff. Cursor has no
public inference API, so it is used as an MCP *client* (below), not a backend.

## Agent-assisted curation (MCP)
The deterministic pipeline leaves a review queue
(`data/08_reporting/ontology_refinement_report.json`: `entity_kind`,
`type_conflict`, `class_promotion`, `rule_slot` items). An agent works through it
with the MCP server in `agent/mcp_server.py`:

```bash
uv sync --extra agent
uv run python -m eurlex_ontology_generation.agent.mcp_server --root .
```
Client configuration (Claude Desktop `claude_desktop_config.json`, Cursor
`.cursor/mcp.json`, Claude Code `.mcp.json`):
```json
{"mcpServers": {"eurlex-curation": {
  "command": "uv",
  "args": ["run", "--directory", "/path/to/eurlex-ontology-generation",
           "python", "-m", "eurlex_ontology_generation.agent.mcp_server", "--root", "."],
  "env": {"CURATION_PROPOSER": "cursor-jane"}}}}
```
Tools: `review_queue`, `get_entity`, `get_rule`, `get_chunk`, `search_text`,
`act_metadata` (optionally CELLAR), `gazetteer_lookup`, `wikidata_lookup`,
`extract_chunk` (runs the generation step with the configured backend),
`check_ontology`, `validate_decision`, `propose_decision`, `list_decisions`.

Guard rails:
- the agent never edits the ontology; `propose_decision` appends to
  `data/05_model_input/curation/decisions.jsonl`;
- every proposal is validated first (re-refine, export, SHACL; HermiT with
  `reasoner=true`); one that adds violations or breaks consistency is stored as
  `rejected_by_validation`;
- only people accept proposals:
  ```bash
  uv run python scripts/curation.py list
  uv run python scripts/curation.py show <id>
  uv run python scripts/curation.py accept <id> --reviewer "Name"
  uv run kedro run --pipelines ontology_refinement,ontology_export
  ```
  The refinement pipeline applies accepted decisions only (`set_typing`,
  `merge_entities`, `remove_rule`, `promote_class`, `demote_class`).

## Running
```bash
uv run python scripts/run_ontology_batches.py --max-batches 10 --review-rounds 2
uv run python scripts/run_ontology_merge.py          # partitions, global, refinement
uv run kedro run --pipelines ontology_refinement     # refinement only (no LLM)
uv run kedro run --pipelines ontology_export          # Turtle + SHACL + HermiT
uv sync --extra reasoner                              # owlready2 for the HermiT check
uv run pytest tests/test_quality_harness.py tests/test_rdf_export.py tests/test_llm_backends.py tests/test_agent_workbench.py
```

## Not done here
- Corporate-body IRIs other than the Commission's (run `link_identifiers.py`
  with the downloaded authority table).
- The data outputs committed under `data/05`-`data/08` predate provenance
  tracking, so they have no per-act policies yet. Rerun
  `scripts/run_ontology_merge.py --force` with Ollama running (embedding-based
  merging) to regenerate them. A test run without Ollama (exact-key merging) gave
  224 merged classes and 505 rules, all with sources; after refinement 179 classes
  (12 accepted, 167 candidate), 45 individuals, 502 rules in 42 act policies;
  HermiT consistent, 0 SHACL violations.
- The new prompts and schema have not been run against Ollama in this branch;
  run a few batches and compare with `evaluate_ontology.py`.
