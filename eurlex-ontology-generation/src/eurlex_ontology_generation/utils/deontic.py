"""Deontic sentences: where an article states a permission, prohibition or duty.

`modal_sentences(text)` returns the sentences of a chunk that contain a deontic
cue, with the rule kind the cue points to:

  prohibitions  shall not, may not, must not, is/are prohibited, is not permitted
  permissions   may, is/are entitled to, is/are allowed/authorised to, can
  duties        shall, must, is/are required/obliged to, has/have to

Cues are checked in that order (`shall not` is a prohibition, not a duty).
Sentences that only state formal provisions (entry into force, application
date, addressees, "shall be binding in its entirety", definitions "shall mean")
are skipped: they bind nobody to an action.

Used by the review check `modal_without_rule` (a rule the generator missed) and
by the optional rules-only generation pass (ontology_generation).
"""
from __future__ import annotations

import re

from .provenance import sentence_spans

KIND_OF_CUE = (
    ("prohibitions", re.compile(
        r"\b(?:shall|may|must|should)\s+not\b|\b(?:is|are)\s+(?:hereby\s+)?prohibited\b"
        r"|\b(?:is|are)\s+not\s+(?:permitted|allowed|authori[sz]ed)\b|\bno\s+\w+\s+shall\b", re.I)),
    ("permissions", re.compile(
        r"\bmay\b(?!\s+not)|\b(?:is|are)\s+(?:entitled|allowed|permitted|authori[sz]ed)\s+to\b", re.I)),
    ("duties", re.compile(
        r"\bshall\b|\bmust\b|\b(?:is|are)\s+(?:required|obliged|bound)\s+to\b|\b(?:has|have)\s+to\b",
        re.I)),
)

# Formal provisions: binding, but not an action of a party.
FORMAL = re.compile(
    r"shall\s+(?:enter\s+into\s+force|apply\b|"
    r"be\s+binding|expire|cease\s+to\s+apply|be\s+addressed\s+to|mean\b|be\s+repealed)"
    r"|\bis\s+addressed\s+to\b|\bare\s+addressed\s+to\b"
    r"|\bshall\s+be\s+deemed\b"
    r"|\bshall\s+be\s+applied\s+in\s+accordance\s+with\b"
    r"|\bmay\s+be\s+cited\s+as\b",
    re.I)
# Definitions: the whole sentence says what something is, not what a party does.
DEFINITIONAL = re.compile(r"\bwithin\s+the\s+meaning\s+of\b|\bshall\s+mean\b|\bmeans\b", re.I)
MAX_SENTENCE_CHARS = 600
# "The report shall:" only introduces a list of points; the points carry the content.
MIN_LEAD_IN_CHARS = 80


def classify(sentence: str) -> tuple[str, str] | None:
    """(rule kind, cue) of a sentence, or None if it states no rule."""
    if DEFINITIONAL.search(sentence):
        return None
    if FORMAL.search(sentence):
        # a formal clause can share the sentence with a real rule: look past it
        rest = FORMAL.sub(" ", sentence)
        if not any(p.search(rest) for _, p in KIND_OF_CUE):
            return None
        sentence = rest
    for kind, pattern in KIND_OF_CUE:
        m = pattern.search(sentence)
        if m:
            return kind, m.group(0).lower()
    return None


def _verb_stem(rule: dict) -> str | None:
    words = [w for w in re.split(r"[_\W]+", str(rule.get("action_label") or rule.get("action") or "").lower())
             if w]
    return words[0][: max(4, len(words[0]) - 2)] if words and len(words[0]) >= 3 else None  # noqa: PLR2004


def covered(sentence: dict, rules: list[dict]) -> bool:
    """A rule rests on this sentence: its located source overlaps it, or (rules without
    a source) its verb occurs in it."""
    for r in rules:
        src = r.get("source") if isinstance(r.get("source"), dict) else {}
        a, b = src.get("chunk_start"), src.get("chunk_end")
        if isinstance(a, int) and isinstance(b, int):
            if a < sentence["end"] and b > sentence["start"]:
                return True
            continue
        stem = _verb_stem(r)
        if stem and stem in sentence["sentence"].lower():
            return True
    return False


def modal_sentences(text: str) -> list[dict]:
    """Deontic sentences of `text`: {start, end, sentence, kind, cue}."""
    out = []
    for a, b in sentence_spans(text or ""):
        sentence = text[a:b]
        if sentence.rstrip().endswith(":") and len(sentence) < MIN_LEAD_IN_CHARS:
            continue
        hit = classify(sentence)
        if hit:
            out.append({"start": a, "end": b, "sentence": sentence[:MAX_SENTENCE_CHARS],
                        "kind": hit[0], "cue": hit[1]})
    return out
