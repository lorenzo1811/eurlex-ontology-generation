"""Concept linking: refined concepts -> Wikidata and EuroVoc / EU authority tables.

Runs after the merge, on the refined ontology (scripts/link_concepts.py); the
refinement then applies the links file deterministically (`apply_links`), like
curation decisions, and the export writes them as owl:sameAs / skos:exactMatch /
skos:closeMatch.

Two targets, two methods:

* Wikidata: for the concepts of each chunk, candidate items are searched on
  Wikidata (`WikidataSearch`: QID, description, English Wikipedia title); ReLiK's
  reader (Sapienza NLP, `sapienzanlp/relik-reader-deberta-v3-*-aida`) then links the
  spans of the chunk text to those candidates. ReLiK's own Wikipedia retriever index
  (12 GB in memory) is not used. A span whose text is the concept's name (or a merged
  variant) is an exact link; spans that only share the head noun are candidates
  checked by the LLM. Without ReLiK, an exact Wikidata label match
  (`entity_linking.WikidataLinker`) is the fallback.
* EuroVoc + EU corporate bodies: a dense index (`VocabularyIndex`) of the SKOS
  concepts, encoded with ReLiK's retriever encoder (zero-shot) or any embedding
  backend. For each concept the top-k candidates are re-ranked with the act's
  own EUROVOC descriptors as a prior; an exact label match is linked directly,
  otherwise gemma3:4b picks exactMatch / closeMatch / none among the candidates
  (`make_llm_chooser`). The answer is accepted only if it names a candidate.

The gazetteer stays first: a concept typed by it keeps its Wikidata QID / EU IRI.
Every link records `method` and `confidence`, so the evaluation can report
precision per method.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .changelog import change, class_target
from .ontology_merge_common.entity_resolution import group_key, norm_tokens

logger = logging.getLogger(__name__)

EUROVOC_SCHEME = "http://eurovoc.europa.eu/100141"
CORPORATE_BODY_SCHEME = "http://publications.europa.eu/resource/authority/corporate-body"
OP_SPARQL = "https://publications.europa.eu/webapi/rdf/sparql"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
EU_IRI_PREFIXES = ("http://publications.europa.eu/", "http://eurovoc.europa.eu/")
DEFAULT_USER_AGENT = "EurLexOntologyGeneration/0.1 (research project)"
NIL_LABELS = {"", "--NME--", "NME", "O", "None"}

Encoder = Callable[[list[str]], np.ndarray]          # texts -> (n, d) vectors
Annotator = Callable[[str, list[str]], list[dict]]   # (text, candidate titles) -> [{start, end, text, title}]
Chooser = Callable[[dict, list[dict]], "LinkDecision | None"]
CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


# ---------------------------------------------------------------- vocabularies
def _sparql(query: str, endpoint: str, user_agent: str, timeout: float) -> list[dict]:
    url = f"{endpoint}?{urllib.parse.urlencode({'query': query, 'format': 'application/sparql-results+json'})}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent,
                                               "Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))["results"]["bindings"]


def _sparql_all(query: str, endpoint: str, user_agent: str, timeout: float, page: int) -> list[dict]:
    """All rows of a simple SELECT, page by page. No ORDER BY: Virtuoso refuses a sorted
    query whose OFFSET + LIMIT exceeds 10000 rows; duplicates are removed by the caller."""
    rows, offset = [], 0
    while True:
        got = _sparql(f"{query} LIMIT {page} OFFSET {offset}", endpoint, user_agent, timeout)
        rows += got
        if len(got) < page:
            return rows
        offset += page


def fetch_skos_concepts(scheme: str, source: str, endpoint: str = OP_SPARQL, lang: str = "en",  # noqa: PLR0913
                        page: int = 10000, user_agent: str = DEFAULT_USER_AGENT,
                        timeout: float = 300.0) -> list[dict]:
    """Every non-deprecated concept of a SKOS scheme from the Publications Office endpoint.

    Four simple queries (prefLabel, altLabel, deprecated, scopeNote) joined here: one
    query with OPTIONALs and GROUP_CONCAT times out on the endpoint."""
    prefix = ("PREFIX skos: <http://www.w3.org/2004/02/skos/core#> "
              "PREFIX owl: <http://www.w3.org/2002/07/owl#> ")

    def rows(prop: str) -> list[dict]:
        q = (f"{prefix}SELECT ?c ?l WHERE {{ ?c skos:inScheme <{scheme}> ; {prop} ?l . "
             f'FILTER(lang(?l) = "{lang}") }}')
        return _sparql_all(q, endpoint, user_agent, timeout, page)

    deprecated = {r["c"]["value"] for r in _sparql_all(
        f"{prefix}SELECT ?c WHERE {{ ?c skos:inScheme <{scheme}> ; owl:deprecated true }}",
        endpoint, user_agent, timeout, page)}
    records: dict[str, dict] = {}
    for r in rows("skos:prefLabel"):
        c = r["c"]["value"]
        if c not in deprecated and c not in records:
            records[c] = {"id": c, "label": r["l"]["value"], "source": source, "alt": [], "note": ""}
    for r in rows("skos:altLabel"):
        rec = records.get(r["c"]["value"])
        if rec and r["l"]["value"] not in rec["alt"]:
            rec["alt"].append(r["l"]["value"])
    for r in rows("skos:scopeNote"):
        rec = records.get(r["c"]["value"])
        if rec and not rec["note"]:
            rec["note"] = r["l"]["value"]
    return list(records.values())


def load_skos_file(path: str, source: str, scheme: str | None = None, lang: str = "en") -> list[dict]:
    """Concepts of a downloaded SKOS file (RDF/XML, Turtle, ...)."""
    from rdflib import Graph, Literal, URIRef  # noqa: PLC0415
    from rdflib.namespace import OWL, RDF, SKOS  # noqa: PLC0415

    g = Graph()
    g.parse(path)
    subjects = set(g.subjects(SKOS.inScheme, URIRef(scheme))) if scheme else set(g.subjects(RDF.type, SKOS.Concept))

    def labels(s, prop) -> list[str]:
        return [str(o) for o in g.objects(s, prop) if isinstance(o, Literal) and (o.language or lang) == lang]

    records = []
    for s in sorted(subjects, key=str):
        if (s, OWL.deprecated, Literal(True)) in g:
            continue
        pref = labels(s, SKOS.prefLabel)
        if pref:
            records.append({"id": str(s), "label": pref[0], "source": source,
                            "alt": labels(s, SKOS.altLabel), "note": (labels(s, SKOS.scopeNote) or [""])[0]})
    return records


def record_text(r: dict) -> str:
    alt = "; ".join(r.get("alt") or [])[:200]
    return ". ".join(p for p in (r["label"], alt, (r.get("note") or "")[:300]) if p)


# ---------------------------------------------------------------- encoders
def hf_encoder(model_name: str, pooling: str = "pooler", device: str | None = None,  # noqa: PLR0913
               batch_size: int = 64, prefix: str = "", max_length: int = 128,
               relik_model: bool = False) -> Encoder:
    """Dense encoder from a Hugging Face model.

    ReLiK's retriever encoder is a `GoldenRetrieverModel` (`relik_model=True`, loaded
    with relik's own class: transformers' AutoModel cannot load its remote code);
    its `pooler_output` is the embedding the retriever was trained with (pooling
    `pooler`). `mean` / `cls` pool the last hidden state of any other encoder."""
    import torch  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    if relik_model:
        from relik.retriever.pytorch_modules import hf as relik_hf  # noqa: PLC0415
        GoldenRetrieverModel = relik_hf.GoldenRetrieverModel  # noqa: N806
        model = GoldenRetrieverModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    model = model.to(device).eval()

    def encode(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = tok([prefix + t for t in texts[i:i + batch_size]], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(**batch)
            hidden = output.last_hidden_state
            if pooling == "pooler" and getattr(output, "pooler_output", None) is not None:
                vec = output.pooler_output
            elif pooling == "cls":
                vec = hidden[:, 0]
            else:
                mask = batch["attention_mask"].unsqueeze(-1).float()
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(vec.cpu().numpy())
            done = min(i + batch_size, len(texts))
            if len(texts) > 2000 and (done // batch_size) % max(1, 2000 // batch_size) == 0:  # noqa: PLR2004
                logger.info("encoded %d/%d", done, len(texts))
                print(f"encoded {done}/{len(texts)}", flush=True)  # noqa: T201
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)
    return encode


def backend_encoder(backend: dict | None, model: str, batch_size: int = 64) -> Encoder:
    """Encoder on top of llm_client.embed_texts (Ollama / OpenAI-compatible)."""
    from .llm_client import embed_texts  # noqa: PLC0415

    return lambda texts: np.array(embed_texts(backend, model, texts, batch_size), dtype=np.float32)


def make_encoder(cfg: dict) -> Encoder:
    """`cfg`: {kind: relik | hf | embed, model, pooling, prefix, backend}."""
    kind = cfg.get("kind", "relik")
    if kind in ("relik", "hf"):
        return hf_encoder(cfg["model"], cfg.get("pooling", "pooler" if kind == "relik" else "mean"),
                          cfg.get("device"), int(cfg.get("batch_size", 64)), cfg.get("prefix", ""),
                          relik_model=kind == "relik")
    if kind == "embed":
        return backend_encoder(cfg.get("backend"), cfg["model"], int(cfg.get("batch_size", 64)))
    raise ValueError(f"unknown encoder kind {kind!r}")


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1e-8
    return v / n


class VocabularyIndex:
    """Dense index of SKOS concepts: records + unit vectors, saved as jsonl + npy."""

    def __init__(self, records: list[dict], vectors: np.ndarray, meta: dict | None = None):
        self.records = records
        self.vectors = _unit(vectors) if len(records) else np.zeros((0, 1), dtype=np.float32)
        self.meta = meta or {}
        self._by_key: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            for lab in [r["label"], *(r.get("alt") or [])]:
                self._by_key.setdefault(group_key("class", lab), []).append(i)

    @classmethod
    def build(cls, records: list[dict], encode: Encoder, meta: dict | None = None) -> VocabularyIndex:
        return cls(records, encode([record_text(r) for r in records]), meta)

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.records),
                                         encoding="utf-8")
        np.save(d / "vectors.npy", self.vectors)
        (d / "meta.json").write_text(json.dumps({**self.meta, "size": len(self.records)}, indent=2),
                                     encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> VocabularyIndex:
        d = Path(directory)
        records = [json.loads(line) for line in (d / "records.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        return cls(records, np.load(d / "vectors.npy"), meta)

    def exact(self, names: Iterable[str], sources: set[str] | None = None) -> list[dict]:
        """Records whose label or altLabel is the name (same merge key). An all-caps
        label (an acronym: COST, EAGF) must also match case-sensitively."""
        hits = []
        for n in names:
            for i in self._by_key.get(group_key("class", n), []):
                r = self.records[i]
                labels = [lab for lab in [r["label"], *(r.get("alt") or [])]
                          if group_key("class", lab) == group_key("class", n)]
                if all(_ACRONYM.fullmatch(lab) and lab != n.strip() for lab in labels):
                    continue
                if (not sources or r["source"] in sources) and r not in hits:
                    hits.append(r)
        return hits

    def search(self, query: np.ndarray, k: int, sources: set[str] | None = None) -> list[tuple[dict, float]]:
        if not self.records:
            return []
        scores = self.vectors @ _unit(query).reshape(-1)
        order = np.argsort(-scores)
        out = []
        for i in order:
            r = self.records[int(i)]
            if sources and r["source"] not in sources:
                continue
            out.append((r, float(scores[int(i)])))
            if len(out) >= k:
                break
        return out


# ---------------------------------------------------------------- ReLiK + Wikipedia
def relik_annotator(reader_model: str, device: str | None = None, window_size: int = 32,  # noqa: PLR0913
                    window_stride: int = 16, precision: str | int | None = None) -> Annotator:
    """ReLiK entity linking with the candidates we give: (text, titles) -> spans.

    Only ReLiK's reader is loaded (`sapienzanlp/relik-reader-deberta-v3-*-aida`). The
    full model would also load its Wikipedia retriever index (12 GB, in memory),
    which does not fit a standard Colab runtime; here the candidates are the
    Wikipedia titles of the Wikidata items found for the concepts of the chunk
    (`WikidataSearch`), and the reader decides which text span denotes which."""
    import torch  # noqa: PLC0415
    from relik import Relik  # noqa: PLC0415
    from relik.reader.pytorch_modules import span as relik_span  # noqa: PLC0415

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"precision": precision} if precision else {}
    reader = relik_span.RelikReaderForSpanExtraction(reader_model, device=device, use_nme=True, **kwargs)
    model = Relik(reader=reader, retriever=None, index=None, task="span",
                  window_size=window_size, window_stride=window_stride)

    def annotate(text: str, candidates: list[str]) -> list[dict]:
        if not candidates:
            return []
        out = model(text, candidates=list(candidates))
        spans = []
        for s in getattr(out, "spans", None) or []:
            start, end = getattr(s, "start", None), getattr(s, "end", None)
            title = str(getattr(s, "label", "") or "")
            surface = getattr(s, "text", None)
            if surface is None and isinstance(start, int) and isinstance(end, int):
                surface = text[start:end]
            if title not in NIL_LABELS and surface:
                spans.append({"start": start, "end": end, "text": str(surface), "title": title.replace("_", " ")})
        return spans
    return annotate


class WikidataSearch:
    """Concept name -> candidate Wikidata items with their English Wikipedia title.

    wbsearchentities on the name (identifiers are split: 'MacroFinancialAssistance'
    -> 'Macro Financial Assistance'); it matches label prefixes only, so when it finds
    nothing a full-text search (list=search) is tried. wbgetentities then gives
    labels, descriptions and enwiki sitelinks. Cached in a JSON file, so each name
    is searched once. HTTP 429/5xx (Wikimedia rate limits, frequent from shared
    cloud IPs such as Colab) are retried with backoff; failures are counted in
    `failures` / `last_error` and are not cached."""

    API = "https://www.wikidata.org/w/api.php"

    def __init__(self, cache_path: str | None, user_agent: str = DEFAULT_USER_AGENT, limit: int = 5,  # noqa: PLR0913
                 timeout: float = 20.0, pause_seconds: float = 0.5, retries: int = 4):
        self.cache_path = Path(cache_path) if cache_path else None
        self.user_agent, self.limit, self.timeout, self.pause = user_agent, limit, timeout, pause_seconds
        self.retries = retries
        self.failures, self.last_error = 0, ""
        self.cache: dict[str, list[dict]] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _get(self, params: dict) -> dict:
        req = urllib.request.Request(f"{self.API}?{urllib.parse.urlencode(params)}",
                                     headers={"User-Agent": self.user_agent})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.retries:  # noqa: PLR2004
                    raise
                wait = float(exc.headers.get("Retry-After") or 2 ** (attempt + 1))
            time.sleep(min(wait, 20.0))
        raise RuntimeError("Wikidata API: too many retries")

    @staticmethod
    def query_of(name: str) -> str:
        words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", re.sub(r"[_]+", " ", name))
        return " ".join(re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", words).split())

    def search(self, name: str) -> list[dict]:
        query = self.query_of(name)
        if query.lower() in self.cache:
            return self.cache[query.lower()]
        try:
            ids = [f["id"] for f in self._get({"action": "wbsearchentities", "search": query, "language": "en",
                                               "format": "json", "limit": self.limit,
                                               "type": "item"}).get("search", [])]
            if not ids:                       # prefix search found nothing: full-text search
                ids = [h["title"] for h in self._get({"action": "query", "list": "search", "srsearch": query,
                                                      "srnamespace": 0, "srlimit": self.limit,
                                                      "format": "json"}).get("query", {}).get("search", [])]
            hits = []
            if ids:
                ents = self._get({"action": "wbgetentities", "ids": "|".join(ids), "props": "labels|descriptions|sitelinks",
                                  "languages": "en", "sitefilter": "enwiki", "format": "json"}).get("entities", {})
                for q in ids:
                    e = ents.get(q) or {}
                    hits.append({"qid": q, "label": (e.get("labels", {}).get("en") or {}).get("value", ""),
                                 "description": (e.get("descriptions", {}).get("en") or {}).get("value", ""),
                                 "title": (e.get("sitelinks", {}).get("enwiki") or {}).get("title")})
        except Exception as exc:  # no candidates for this name; not cached, so retried next run
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Wikidata search failed for %r (%s)", query, exc)
            return []
        self.cache[query.lower()] = hits
        time.sleep(self.pause)
        return hits

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")


class WikipediaQids:
    """Wikipedia title -> Wikidata QID (pageprops.wikibase_item), cached in a JSON file."""

    def __init__(self, cache_path: str | None, user_agent: str = DEFAULT_USER_AGENT,
                 timeout: float = 20.0, pause_seconds: float = 0.1):
        self.cache_path = Path(cache_path) if cache_path else None
        self.user_agent, self.timeout, self.pause = user_agent, timeout, pause_seconds
        self.cache: dict[str, str | None] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def resolve(self, titles: Iterable[str]) -> dict[str, str | None]:
        todo = [t for t in dict.fromkeys(titles) if t and t not in self.cache]
        for i in range(0, len(todo), 50):
            batch = todo[i:i + 50]
            params = {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item", "redirects": 1,
                      "format": "json", "titles": "|".join(batch)}
            req = urllib.request.Request(f"{WIKIPEDIA_API}?{urllib.parse.urlencode(params)}",
                                         headers={"User-Agent": self.user_agent})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                    data = json.loads(resp.read().decode("utf-8")).get("query", {})
            except Exception as exc:  # network problems: leave uncached, retry next run
                logger.warning("Wikipedia lookup failed (%s)", exc)
                continue
            alias = {n["from"]: n["to"] for n in data.get("normalized", []) + data.get("redirects", [])}
            qid_of = {p.get("title"): p.get("pageprops", {}).get("wikibase_item")
                      for p in data.get("pages", {}).values()}
            for t in batch:
                target = alias.get(alias.get(t, t), alias.get(t, t))
                self.cache[t] = qid_of.get(target)
            time.sleep(self.pause)
        return {t: self.cache.get(t) for t in titles}

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- LLM choice
class LinkDecision(BaseModel):
    choice: int = Field(description="number of the chosen candidate, 0 if none fits")
    relation: Literal["exact", "close", "none"] = Field(
        description="exact: same concept/entity; close: broader, narrower or related; none: no candidate fits")
    reason: str = Field(default="", description="one short sentence")


LINK_SYSTEM = (
    "You link concepts of an ontology extracted from EU legislation to an external vocabulary. "
    "Choose the candidate that denotes the same thing as the concept. Answer 'exact' only when the "
    "candidate is the same concept or the same named entity; 'close' only when it is a broader or "
    "narrower term for the same kind of thing (not merely the same topic, country or act). A company, "
    "a legal act or a place is never 'close' to a general term. If unsure, choice 0 and 'none'. "
    "Never invent a candidate."
)
LINK_USER = (
    "Concept: {name}\nKind: {kind} ({is_a})\nDescription: {description}\n"
    "Subjects of the act it comes from: {descriptors}\n\nCandidates ({target}):\n{candidates}\n\n"
    "Return JSON with choice (number, 0 for none), relation and reason."
)


def make_llm_chooser(chat, backend: dict, model: str, temperature: float = 0.0,  # noqa: PLR0913
                     num_ctx: int | None = 4096, max_tokens: int | None = 200) -> Chooser:
    """A `Chooser` on top of `chat` (utils/llm_client.llm_chat), e.g. gemma3:4b."""
    def choose(concept: dict, candidates: list[dict]) -> LinkDecision | None:
        lines = "\n".join(f"{i}. {c['label']}" + (f" — {c['description']}" if c.get("description") else "")
                          for i, c in enumerate(candidates, 1))
        user = LINK_USER.format(name=concept["name"], kind=concept.get("kind") or "",
                                is_a=concept.get("is_a") or "", description=(concept.get("description") or "")[:400],
                                descriptors="; ".join(concept.get("descriptors") or [])[:300] or "unknown",
                                target=concept.get("target", ""), candidates=lines)
        try:
            raw = chat(backend, model, LINK_SYSTEM, user, temperature, schema_model=LinkDecision,
                       json_mode=True, num_ctx=num_ctx, max_tokens=max_tokens).content
            data = json.loads(re.sub(r"^```(?:json)?\s*|```$", "", (raw or "").strip(), flags=re.M))
            return LinkDecision.model_validate(data)
        except Exception as exc:  # an unusable answer is no link, but it is logged
            logger.warning("Link LLM call failed (%s: %s)", type(exc).__name__, str(exc)[:200])
            return None
    return choose


# ---------------------------------------------------------------- linker
def _names(entity: dict) -> list[str]:
    return list(dict.fromkeys([entity["name"], *(entity.get("merged_from") or [])]))


_ACRONYM = re.compile(r"[A-Z0-9][A-Z0-9&\-]{1,9}")
# words that say nothing about what a concept is: no evidence of a match on their own
_WEAK_TOKENS = {"european", "union", "the", "and", "for", "with", "from", "this", "that", "other",
                "general", "decision", "regulation", "directive", "act", "article", "annex"}


def _content_stems(text: str) -> set[str]:
    return {t[:5] for t in norm_tokens(re.sub(r"[_]+", " ", text)) if len(t) >= 4 and t not in _WEAK_TOKENS}  # noqa: PLR2004


def shares_content_word(names: Iterable[str], label: str) -> bool:
    """True if the candidate label and one of the names share a content word (5-letter
    stem): an LLM choice without any lexical support is not accepted."""
    stems = set().union(*(_content_stems(n) for n in names)) if names else set()
    return bool(stems & _content_stems(label))


def _head(name: str) -> str:
    toks = norm_tokens(name)
    return toks[-1] if toks else ""


def _chunks_of(entity: dict, chunk_texts: dict[str, str]) -> list[str]:
    ids = [p.get("chunk_id") for p in entity.get("provenance") or [] if isinstance(p, dict)]
    ids = [i for i in dict.fromkeys(ids) if i in chunk_texts]
    if ids:
        return ids
    acts = set(entity.get("sources") or [])
    return [c for c in chunk_texts if c.split("_")[0] in acts]


class ConceptLinker:
    """Links the classes and individuals of a refined ontology (see module docstring)."""

    def __init__(self, index: VocabularyIndex | None = None, encode: Encoder | None = None,  # noqa: PLR0913
                 qids: Callable[[list[str]], dict[str, str | None]] | None = None,
                 choose: Chooser | None = None, act_descriptors: dict[str, list[str]] | None = None,
                 wikidata_label_lookup: Callable[[str], dict | None] | None = None,
                 k: int = 8, prior_bonus: float = 0.05):
        self.index, self.encode, self.qids, self.choose = index, encode, qids, choose
        self.descriptors = {c: [d.strip() for d in ds if d.strip()] for c, ds in (act_descriptors or {}).items()}
        self.wikidata_label_lookup = wikidata_label_lookup
        self.k, self.prior_bonus = k, prior_bonus
        self.stats: Counter = Counter()

    def _descriptors(self, entity: dict) -> list[str]:
        out = []
        for celex in entity.get("sources") or []:
            out += [d for d in self.descriptors.get(celex, []) if d not in out]
        return out

    # ------------------------------------------------------------ Wikidata
    def wikidata(self, entity: dict, chunk_texts: dict[str, str],
                 spans_of: dict[str, list[dict]] | None) -> dict | None:
        if entity.get("wikidata"):
            return {"id": entity["wikidata"], "relation": "exact", "method": "gazetteer"
                    if entity.get("type_source") == "gazetteer" else "existing", "confidence": "high"}
        names = _names(entity)
        keys = {group_key("class", n) for n in names}
        heads = {_head(n) for n in names} - {""}
        exact: Counter = Counter()
        partial: Counter = Counter()
        for cid in _chunks_of(entity, chunk_texts) if spans_of is not None else []:
            for s in spans_of.get(cid, []):
                if group_key("class", s["text"]) in keys or group_key("class", s["title"]) in keys:
                    exact[s["title"]] += 1
                elif _head(s["text"]) in heads:
                    partial[s["title"]] += 1
        if exact and self.qids:
            title = exact.most_common(1)[0][0]
            qid = self.qids([title]).get(title)
            if qid:
                return {"id": qid, "label": title, "relation": "exact", "method": "relik_exact",
                        "confidence": "high"}
        if partial and self.qids and self.choose:
            titles = [t for t, _ in partial.most_common(self.k)]
            qid_of = self.qids(titles)
            cands = [{"id": qid_of[t], "label": t} for t in titles if qid_of.get(t)]
            picked = self._pick(entity, cands, "Wikipedia/Wikidata")
            if picked:
                return {**picked, "method": "relik_llm"}
        if spans_of is None and self.wikidata_label_lookup:
            hit = self.wikidata_label_lookup(entity["name"])
            if hit and hit.get("qid"):
                return {"id": hit["qid"], "label": hit.get("label"), "relation": "exact",
                        "method": "wikidata_label", "confidence": "medium"}
        return None

    # ------------------------------------------------------------ EuroVoc
    def eurovoc(self, entity: dict) -> tuple[dict | None, list[dict]]:
        iri = entity.get("iri") or ""
        if iri.startswith(EU_IRI_PREFIXES):
            return {"id": iri, "relation": "exact", "method": "gazetteer", "confidence": "high"}, []
        if self.index is None:
            return None, []
        sources = {"eurovoc", "corporate_body"} if entity.get("type") == "Party" else {"eurovoc"}
        names = _names(entity)
        exact = self.index.exact(names, sources)
        if exact:
            r = exact[0]
            return {"id": r["id"], "label": r["label"], "source": r["source"], "relation": "exact",
                    "method": "label_exact", "confidence": "high"}, []
        if self.encode is None:
            return None, []
        query = " ".join(re.sub(r"[_]+", " ", entity["name"]).split())
        desc = (entity.get("description") or "").split(" | ")[0]
        vec = self.encode([f"{query}. {desc[:300]}" if desc else query])[0]
        descriptors = {d.lower() for d in self._descriptors(entity)}
        hits = [(r, s + (self.prior_bonus if r["label"].lower() in descriptors else 0.0))
                for r, s in self.index.search(vec, self.k * 2, sources)]
        hits = sorted(hits, key=lambda h: -h[1])[: self.k]
        cands = [{"id": r["id"], "label": r["label"], "source": r["source"], "score": round(s, 4),
                  "description": (r.get("note") or "")[:120]} for r, s in hits]
        if not self.choose:
            return None, cands
        picked = self._pick(entity, cands, "EuroVoc / EU authority tables")
        return ({**picked, "method": "retriever_llm"} if picked else None), cands

    def _pick(self, entity: dict, cands: list[dict], target: str) -> dict | None:
        if not cands or not self.choose:
            return None
        concept = {**entity, "descriptors": self._descriptors(entity), "target": target}
        decision = self.choose(concept, cands)
        self.stats["llm_calls"] += 1
        if decision is None:
            self.stats["llm_errors"] += 1
            return None
        if decision.relation == "none" or not 1 <= decision.choice <= len(cands):
            self.stats["llm_none"] += 1
            return None
        c = cands[decision.choice - 1]
        if not shares_content_word(_names(entity), c["label"]):
            self.stats["llm_rejected_no_shared_word"] += 1
            return None
        self.stats[f"llm_{decision.relation}"] += 1
        return {k: v for k, v in c.items() if k != "description"} | {
            "relation": decision.relation, "confidence": "medium" if decision.relation == "exact" else "low",
            "reason": decision.reason[:200]}

    # ------------------------------------------------------------ all
    def link(self, refined: dict, chunk_texts: dict[str, str] | None = None,
             spans_of: dict[str, list[dict]] | None = None) -> dict:
        chunk_texts = chunk_texts or {}
        concepts = {}
        for kind, section in (("class", "classes"), ("individual", "individuals")):
            for e in refined.get(section) or []:
                entity = {**e, "kind": kind, "is_a": e.get("is_a") or e.get("instance_of")}
                wd = self.wikidata(entity, chunk_texts, spans_of)
                ev, cands = self.eurovoc(entity)
                concepts[e["name"]] = {"kind": kind, "wikidata": wd, "eurovoc": ev,
                                       "eurovoc_candidates": cands[:3]}
                for target, link in (("wikidata", wd), ("eurovoc", ev)):
                    self.stats[f"{target}:{link['method'] if link else 'none'}"] += 1
        linked = sum(1 for c in concepts.values() if c["wikidata"] or c["eurovoc"])
        return {"concepts": concepts,
                "stats": {"concepts": len(concepts), "linked": linked,
                          "linked_share": round(linked / len(concepts), 4) if concepts else 0.0,
                          **dict(sorted(self.stats.items()))}}


# ---------------------------------------------------------------- refinement
def load_links(path: str | None) -> dict:
    """Links file written by scripts/link_concepts.py ({} when absent)."""
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8")).get("concepts", {})


def apply_links(record: dict, links: dict) -> list[dict]:
    """Copy the links of `record['name']` onto a refined entity; returns change events.
    Gazetteer identifiers already on the record are kept."""
    link = links.get(record["name"])
    if not link:
        return []
    events, methods, confidences = [], {}, []
    wd, ev = link.get("wikidata"), link.get("eurovoc")
    if wd:
        if not record.get("wikidata"):
            record["wikidata"] = wd["id"]
            events.append(change("concept_linked", f"wikidata {wd['method']}",
                                 actor="llm_reviewer" if wd["method"].endswith("_llm") else "normalizer",
                                 target=class_target(record["name"]) + ".wikidata", after=wd["id"]))
        if record["wikidata"] == wd["id"]:
            record["wikidata_relation"] = wd.get("relation", "exact")
            methods["wikidata"] = wd["method"]
            confidences.append(wd.get("confidence", "low"))
    if ev and ev["method"] != "gazetteer":
        record["eurovoc"] = {k: ev[k] for k in ("id", "label", "relation", "source") if ev.get(k)}
        events.append(change("concept_linked", f"eurovoc {ev['method']}",
                             actor="llm_reviewer" if ev["method"].endswith("_llm") else "normalizer",
                             target=class_target(record["name"]) + ".eurovoc", after=ev["id"]))
    if ev:
        methods["eurovoc"] = ev["method"]
        confidences.append(ev.get("confidence", "low"))
    if methods:
        record["link_method"] = methods
        record["link_confidence"] = max(confidences, key=lambda c: CONFIDENCE_RANK.get(c, 0))
    return events


def link_stats(refined: dict) -> dict:
    """Share of concepts with a link, per target and method (batch_report / evaluation)."""
    out: Counter = Counter()
    entities = [(e, "class") for e in refined.get("classes") or []] + \
               [(e, "individual") for e in refined.get("individuals") or []]
    for e, kind in entities:
        out[f"{kind}:total"] += 1
        has_wd = bool(e.get("wikidata"))
        has_ev = bool(e.get("eurovoc")) or str(e.get("iri") or "").startswith(EU_IRI_PREFIXES)
        if has_wd or has_ev:
            out[f"{kind}:linked"] += 1
        out["wikidata"] += has_wd
        out["eurovoc"] += has_ev
        for target, method in (e.get("link_method") or {}).items():
            out[f"method:{target}:{method}"] += 1
        if (e.get("eurovoc") or {}).get("relation") == "close" or e.get("wikidata_relation") == "close":
            out["close_match"] += 1
    total = out["class:total"] + out["individual:total"]
    linked = out["class:linked"] + out["individual:linked"]
    return {"concepts": total, "linked": linked, "linked_share": round(linked / total, 4) if total else 0.0,
            **{k: v for k, v in sorted(out.items())}}
