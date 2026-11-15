"""Source provenance of the elements of a chunk-level ontology.

Every class, rule and constraint gets a `source`:

    {"celex", "chunk_id", "section", "article", "paragraph", "recital",
     "chunk_start", "chunk_end",      # offsets in chunk_text
     "start", "end",                  # offsets in the act text (eurlex_corpus)
     "quote",                         # the text span the element rests on
     "match"}                         # how the span was found, see below

`match`
  exact       the `evidence` quoted by the LLM is in the chunk verbatim
  normalized  ... after normalising whitespace, quotes and case
  fuzzy       ... only approximately: the sentence that shares most words
  inferred    no usable quote; the sentence with most words of the element
              (name, action label, party, constraint value)
  none        nothing in the chunk supports the element

Merged ontologies replace `source` with `provenance`, a list of such items
(without the chunk offsets), one per chunk the element was extracted from.

Rules derived from the structural chunks (utils/eurlex_structure.py): the
article / paragraph / recital comes from the position in the chunk, the act
offsets from the chunk's `segments`.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .changelog import change, class_target, rule_target
from .eurlex_structure import HEADER_MARK, chunk_to_act_offset, section_of

RULE_KINDS = ("permissions", "prohibitions", "duties")
FUZZY_MIN_OVERLAP = 0.6
_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(‘'\"0-9])")
_PLAIN_QUOTES = {"’": "'", "‘": "'", "“": '"', "”": '"'}
_STOP = {"the", "of", "and", "to", "in", "a", "an", "for", "by", "on", "or", "with", "be", "is",
         "are", "shall", "that", "this", "from", "as", "at", "its", "which"}


# ------------------------------------------------------------------ text helpers
def _body_start(text: str) -> int:
    """Position after the '§ ...' header line, 0 for chunks without one."""
    if text.startswith(HEADER_MARK) and "\n" in text:
        return text.index("\n") + 1
    return 0


def sentence_spans(text: str) -> list[tuple[int, int]]:
    start = _body_start(text)
    cuts = [start] + [m.end() for m in _SENT.finditer(text, start)] + [len(text)]
    spans = []
    for a, b in zip(cuts, cuts[1:]):
        seg = text[a:b]
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg.rstrip())
        if trail > lead:
            spans.append((a + lead, a + trail))
    return spans


def _words(s: Any) -> list[str]:
    return _WORD.findall(str(s or "").lower().replace("_", " "))


def _stems(s: Any) -> set[str]:
    """Content words, cut to a stem ('mobilisation' ~ 'mobilise')."""
    out = set()
    for w in _words(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(s or ""))):
        if w in _STOP or len(w) < 3:  # noqa: PLR2004
            continue
        out.add(w[: max(4, len(w) - 3)] if len(w) > 5 else w)  # noqa: PLR2004
    return out


def _normalized_with_map(text: str) -> tuple[str, list[int]]:
    """Lower-cased text with collapsed whitespace and plain quotes, plus a map
    from each normalised character to its position in `text`."""
    out, pos = [], []
    prev_space = False
    for i, raw in enumerate(text):
        ch = _PLAIN_QUOTES.get(raw, raw)
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        out.append(ch.lower())
        pos.append(i)
    return "".join(out), pos


def locate_quote(quote: Any, text: str) -> tuple[int, int, str] | None:
    """(start, end, match) of `quote` in `text`: exact, normalized or fuzzy."""
    q = str(quote or "").strip()
    if len(q) < 8 or not text:  # noqa: PLR2004
        return None
    i = text.find(q)
    if i >= 0:
        return i, i + len(q), "exact"
    norm_text, mapping = _normalized_with_map(text)
    norm_q, _ = _normalized_with_map(q)
    norm_q = norm_q.strip().rstrip(".")
    i = norm_text.find(norm_q)
    if i >= 0 and norm_q:
        return mapping[i], mapping[i + len(norm_q) - 1] + 1, "normalized"
    q_words = set(_words(q)) - _STOP
    if not q_words:
        return None
    best, best_score = None, 0.0
    for a, b in sentence_spans(text):
        s_words = set(_words(text[a:b])) - _STOP
        score = len(q_words & s_words) / len(q_words)
        if score > best_score:
            best, best_score = (a, b), score
    if best and best_score >= FUZZY_MIN_OVERLAP:
        return best[0], best[1], "fuzzy"
    return None


def infer_span(terms: list[Any], text: str, within: tuple[int, int] | None = None
               ) -> tuple[int, int] | None:
    """The sentence (inside `within`, if given) containing most stems of `terms`."""
    stems = set().union(*(_stems(t) for t in terms)) if terms else set()
    if not stems or not text:
        return None
    best, best_hits = None, 0
    for a, b in sentence_spans(text):
        if within and not (a < within[1] and b > within[0]):
            continue
        sentence = text[a:b].lower()
        hits = sum(1 for s in stems if s in sentence)
        if hits > best_hits:
            best, best_hits = (a, b), hits
    return best


# ------------------------------------------------------------ position -> unit
_ART_HEADING = re.compile(r"(?:^|(?<=[.;:)]\s)|(?<=\n))Article\s*(\d{1,3}[a-z]?)(?![\da-z])")
_PARA = re.compile(r"(?:^|(?<=[.;:]\s)|(?<=\n))(\d{1,2})\.\s+(?=[A-Z(‘'\"])", re.M)
_PARA_ANY = re.compile(r"(?<!\S)(\d{1,2})\.\s+(?=[A-Z(‘'\"])")
_RECITAL = re.compile(r"(?:^|(?<=[.;:,]\s)|(?<=\n)|(?<=Whereas:\s))\((\d{1,3})\)\s")
_HEAD_ART = re.compile(r"Article\s+(\d{1,3}[a-z]?)(?:\((\d{1,2})\))?")
_HEAD_REC = re.compile(r"Recitals\s+\((\d{1,3})\)")


def unit_at(text: str, pos: int) -> dict:
    """Section, article, paragraph and recital at position `pos` of a chunk."""
    section = section_of(text)
    header = text[len(HEADER_MARK):_body_start(text) - 1] if section else ""
    out: dict[str, Any] = {"section": header or None}
    body = _body_start(text)
    if section == "article":
        first = _HEAD_ART.search(header)
        article = first.group(1) if first else None
        paragraph = int(first.group(2)) if first and first.group(2) else None
        last_heading = body
        for m in _ART_HEADING.finditer(text, body, pos + 1):
            article, paragraph, last_heading = m.group(1), None, m.start()
        for m in _PARA_ANY.finditer(text, last_heading, pos + 1):
            # "Article 1 1. The Union ...": paragraph 1 follows the heading (and its title)
            after_heading = int(m.group(1)) == 1 and m.start() - last_heading <= 150  # noqa: PLR2004
            if after_heading or _PARA.match(text, m.start()):
                paragraph = int(m.group(1))
        out.update(article=article, paragraph=paragraph)
    elif section == "recital":
        first = _HEAD_REC.search(header)
        recital = int(first.group(1)) if first else None
        for m in _RECITAL.finditer(text, body, pos + 1):
            recital = int(m.group(1))
        out["recital"] = recital
    return out


# ------------------------------------------------------------------- sources
def make_source(text: str, span: tuple[int, int] | None, match: str, *, celex: str | None,  # noqa: PLR0913
                chunk_id: str | None, segments: list | None) -> dict:
    src: dict[str, Any] = {"celex": celex, "chunk_id": chunk_id}
    if span is None:
        src.update(match="none", section=unit_at(text, 0)["section"] if text else None)
        return src
    a, b = span
    src.update(unit_at(text, a))
    src.update(chunk_start=a, chunk_end=b, quote=text[a:b][:600], match=match)
    if segments:
        start = chunk_to_act_offset(a, segments)
        end = chunk_to_act_offset(b - 1, segments)
        if start is not None and end is not None:
            src.update(start=start, end=end + 1)
    return {k: v for k, v in src.items() if v is not None}


def _find(evidence: Any, terms: list[Any], text: str,
          within: tuple[int, int] | None = None) -> tuple[tuple[int, int] | None, str]:
    hit = locate_quote(evidence, text)
    if hit:
        return (hit[0], hit[1]), hit[2]
    span = infer_span(terms, text, within)
    return (span, "inferred") if span else (None, "none")


def attach_provenance(onto: dict, text: str, *, celex: str | None = None,
                      chunk_id: str | None = None, segments: list | None = None) -> list[dict]:
    """Set `source` on every class, rule and constraint of `onto` (in place).

    Returns change events for quotes that could not be verified
    (`evidence_unverified`: the LLM quote is not in the text) and for elements
    with no support at all (`source_not_found`)."""
    events: list[dict] = []
    if not isinstance(onto, dict) or not isinstance(text, str) or not text:
        return events
    kw = {"celex": celex, "chunk_id": chunk_id, "segments": segments}

    def note(code: str, target: str, element: dict, match: str) -> None:
        if match in ("inferred", "none") and str(element.get("evidence") or "").strip():
            events.append(change("evidence_unverified", f"quote not in text; {match}",
                                 target=target, before=element.get("evidence")))
        elif match == "none":
            events.append(change(code, "no sentence of the chunk supports it", target=target))

    for cls in onto.get("classes") or []:
        if not isinstance(cls, dict):
            continue
        span, match = _find(cls.get("evidence"), [cls.get("name")], text)
        cls["source"] = make_source(text, span, match, **kw)
        note("source_not_found", class_target(cls.get("name")), cls, match)

    rm = onto.get("rights_model") if isinstance(onto.get("rights_model"), dict) else {}
    for kind in RULE_KINDS:
        for rule in rm.get(kind) or []:
            if not isinstance(rule, dict):
                continue
            terms = [rule.get("action_label") or rule.get("action"), rule.get("party"), rule.get("asset")]
            span, match = _find(rule.get("evidence"), terms, text)
            rule["source"] = make_source(text, span, match, **kw)
            target = rule_target(kind, rule)
            note("source_not_found", target, rule, match)
            for c in rule.get("constraints") or []:
                if not isinstance(c, dict):
                    continue
                value = c.get("right_operand") if c.get("right_operand") is not None else c.get("value")
                c_span = _value_span(value, text, span)
                if c_span:
                    c["source"] = make_source(text, c_span, "exact", **kw)
                else:
                    c_span = infer_span([value, c.get("left_operand")], text, span) or span
                    c["source"] = make_source(text, c_span, "inferred" if c_span else "none", **kw)
    return events


def _value_span(value: Any, text: str, near: tuple[int, int] | None) -> tuple[int, int] | None:
    """The span of a constraint value in the text, preferring the rule's sentence."""
    v = str(value or "").strip()
    if len(v) < 2:  # noqa: PLR2004
        return None
    hits = [m.start() for m in re.finditer(re.escape(v), text, re.I)]
    if not hits:
        return None
    if near:
        inside = [h for h in hits if near[0] <= h < near[1]]
        hits = inside or hits
    return hits[0], hits[0] + len(v)


def strip_provenance(ontology: Any) -> Any:
    """The ontology without `source` / `provenance` fields (JSON string or dict in,
    same type out), for LLM prompts: the located spans only cost context there.
    The LLM's own `evidence` quote is kept."""
    as_text = isinstance(ontology, str)
    try:
        data = json.loads(ontology) if as_text else ontology
    except (TypeError, ValueError):
        return ontology

    def walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items() if k not in ("source", "provenance")}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    out = walk(data)
    return json.dumps(out, ensure_ascii=False) if as_text else out


def source_to_evidence(src: Any) -> dict | None:
    """A chunk-level `source` as an evidence item for merged ontologies."""
    if not isinstance(src, dict) or src.get("match") in (None, "none"):
        return None
    keep = ("celex", "chunk_id", "section", "article", "paragraph", "recital", "start", "end",
            "quote", "match")
    return {k: src[k] for k in keep if src.get(k) is not None}


def evidence_of(obj: dict) -> list[dict]:
    """Evidence items of an element: its `provenance` list (merged levels) or its
    chunk-level `source`. (`evidence` is the LLM's quote, a string.)"""
    items = obj.get("provenance")
    if isinstance(items, list) and items:
        return [i for i in items if isinstance(i, dict)]
    item = source_to_evidence(obj.get("source"))
    return [item] if item else []


def merge_evidence(existing: list[dict], new: list[dict], cap: int = 10) -> list[dict]:
    """Union of evidence lists, without duplicates, at most `cap` items."""
    seen = {(e.get("chunk_id"), e.get("start"), e.get("end"), e.get("quote")) for e in existing}
    out = list(existing)
    for e in new:
        k = (e.get("chunk_id"), e.get("start"), e.get("end"), e.get("quote"))
        if k not in seen and len(out) < cap:
            out.append(e)
            seen.add(k)
    return out
