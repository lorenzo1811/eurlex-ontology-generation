"""Repair, instead of delete, values that the source text does not support.

The cleanup and the normaliser used to delete a rule whose action has no trace
in the text, a constraint whose date or amount is not in the text, and an
example whose value is not in the text. A `Repairer` is asked first, in three
steps (each logged, utils/changelog.py):

1. deterministic, from the text
   - action label: spelling variants (fulfillment / fulfilment, -ize / -ise,
     -or / -our, -er / -re), a vocabulary synonym of the verb that occurs in the
     rule's sentence, or the label rebuilt from the sentence (the verb after
     "shall" / "may" and its object);
   - constraint value: the only date / amount of the rule's sentence, the year
     when the text gives only a year, the currency the text names;
   - example value: the closest span of the text (same words, any case).
2. LLM check (optional): the element and the sentences around it are sent to a
   small model that answers `correct` (with the value and the quote it rests
   on) or `remove`. A correction is accepted only if the value and the quote
   are found in the text.
3. removal, the last resort: the caller removes the element and logs it, and
   the event goes to the removed-items review queue.

Repairs are events with actor `normalizer` (step 1) or `llm_reviser` (step 2)
and codes `*_repaired`; the method is in `detail`.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from .deontic import KIND_OF_CUE, classify
from .llm_client import llm_chat
from .ontology_grounding import (
    _date_in_text,
    _numbers_in_text,
    constraint_is_ungrounded,
)
from .provenance import infer_span, locate_quote, sentence_spans
from .vocabulary import OTHER_ACTION, Vocabulary, load_vocabulary, split_identifier

logger = logging.getLogger(__name__)

# --------------------------------------------------------------- spelling
_VARIANTS = (("ll", "l"), ("l", "ll"), ("ize", "ise"), ("ise", "ize"), ("yze", "yse"), ("yse", "yze"),
             ("ization", "isation"), ("isation", "ization"), ("or", "our"), ("our", "or"),
             ("er", "re"), ("re", "er"), ("og", "ogue"), ("ogue", "og"))
_STEM_MIN = 4
_SKIP_AFTER_CUE = {"be", "been", "also", "not", "in", "particular", "where", "necessary", "only",
                   "therefore", "however", "then", "thereafter", "jointly", "duly", "immediately"}
_OBJECT_SKIP = {"the", "a", "an", "its", "their", "his", "her", "this", "that", "these", "those",
                "on", "of", "to", "for", "with", "by", "in", "at", "from", "about", "any", "all",
                "each", "such", "and", "or", "it", "them", "out", "up", "regularly", "closely",
                "particular", "accordance", "behalf", "case", "cases", "order", "view", "where",
                "union", "whether", "which", "within", "without", "between", "under", "after",
                "before", "during", "upon", "into", "than", "other", "following", "referred"}
_DATE_IN_PROSE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{4})\b|\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", re.I)
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]
_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
_CURRENCY = re.compile(r"\b(EUR|ECU|USD|GBP|CHF|SEK|DKK|PLN|CZK|HUF|RON|BGN|NOK|JPY)\b")
_AMOUNT = re.compile(r"(?:EUR|ECU|USD|GBP|CHF|SEK|DKK|PLN|CZK|HUF|RON|BGN|NOK|JPY|€)\s*"
                     r"(\d{1,3}(?:[ ,.]\d{3})+|\d+)(?:[.,](\d+))?\s*(thousand|million|billion)?", re.I)
_MULT = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def _stem(word: str) -> str:
    return word[: max(_STEM_MIN, len(word) - 2)]


def _text_word(stem: str, text: str) -> str | None:
    """The word of `text` that starts with `stem` (lower case), if any."""
    m = re.search(r"\b(" + re.escape(stem) + r"[a-z]*)", text.lower())
    return m.group(1) if m else None


def spelling_variant(word: str, text: str) -> str | None:
    """The form of `word` used in `text` ('fulfillment' -> 'fulfilment')."""
    w = word.lower()
    for a, b in _VARIANTS:
        i = w.find(a)
        while i >= 0:
            cand = w[:i] + b + w[i + len(a):]
            found = _text_word(_stem(cand), text)
            if found:
                return found
            i = w.find(a, i + 1)
    return None


def _same_word(word: str, candidate: str) -> bool:
    """'inform' ~ 'informs' / 'informed'; not 'commit' ~ 'committee'."""
    w, c = word.lower(), candidate.lower()
    return w == c or (w.startswith(c) and len(w) - len(c) <= 3 and w[len(c):] in  # noqa: PLR2004
                      ("s", "es", "ed", "d", "ing", "ment"))


def _word_in(candidate: str, sentence: str) -> bool:
    return any(_same_word(w, candidate) for w in re.findall(r"[A-Za-z]+", sentence))


def _modal_verb_object(sentence: str, verbs: set[str]) -> tuple[str, str] | None:
    """('decide', 'release') from '... the Commission shall decide on the release of ...'.

    Only for active sentences that state a rule ('shall be provided' names no
    actor's verb; 'shall enter into force' is not a rule). The object is the first
    content word after the verb, skipping determiners, prepositions, adverbs and
    coordinated verbs ('suspend or cancel the disbursement' -> 'disbursement')."""
    if classify(sentence) is None:
        return None
    for _, pattern in KIND_OF_CUE:
        m = pattern.search(sentence)
        if not m:
            continue
        words = re.findall(r"[A-Za-z][A-Za-z-]*", sentence[m.end():])
        i = 0
        while i < len(words) and (words[i].lower() in _SKIP_AFTER_CUE - {"be", "been"}
                                  or words[i].lower().endswith("ly")):
            i += 1
        if i >= len(words) or words[i].lower() in ("be", "been", "have", "has"):
            return None
        verb = words[i].lower()
        obj = ""
        for w in words[i + 1: i + 10]:
            lw = w.lower()
            if lw in _OBJECT_SKIP or lw in _SKIP_AFTER_CUE or lw.endswith("ly") or lw in verbs or len(lw) < 4:  # noqa: PLR2004
                continue
            obj = lw
            break
        return (verb, obj) if obj else None
    return None


def _iso(day: str, month: str, year: str) -> str:
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def dates_in(text: str) -> list[str]:
    out = []
    for m in _DATE_IN_PROSE.finditer(text or ""):
        if m.group(2):
            out.append(_iso(m.group(1), str(_MONTHS.index(m.group(2).lower()) + 1), m.group(3)))
        else:
            out.append(_iso(m.group(4), m.group(5), m.group(6)))
    return list(dict.fromkeys(out))


def amounts_in(text: str) -> list[tuple[str, str]]:
    """[(value as digits, currency)] of the amounts with a currency in `text`."""
    out = []
    for m in _AMOUNT.finditer(text or ""):
        cur = re.match(r"[A-Z€]+", m.group(0).upper())
        currency = "EUR" if cur and cur.group(0) == "€" else (cur.group(0) if cur else "")
        value = float(re.sub(r"[ ,.]", "", m.group(1)) + ("." + m.group(2) if m.group(2) else ""))
        if m.group(3):
            value *= _MULT[m.group(3).lower()]
        digits = str(int(value)) if value == int(value) else str(value)
        out.append((digits, currency))
    return list(dict.fromkeys(out))


# --------------------------------------------------------------- LLM step
class RepairDecision(BaseModel):
    decision: Literal["correct", "remove"]
    value: str = Field("", description="The corrected value, copied from the text.")
    evidence: str = Field("", description="The words of the text the value rests on, verbatim.")


REPAIR_SYSTEM = (
    "You check one element extracted from an EU legal text against that text. The element has a value "
    "that does not appear in the text. If the text states a correct value for it, answer "
    "decision=correct, the value copied from the text and the evidence (the words of the text it rests "
    "on, verbatim). If the text does not support the element at all, answer decision=remove. Never "
    "invent a value; copy it from the text. Answer with a JSON object {decision, value, evidence}."
)
REPAIR_USER = (
    "Text:\n{context}\n\nElement ({what}):\n{element}\n\nProblem: {problem}\n\n"
    "What to return in `value`: {expected}"
)

LLMCall = Callable[[str, str, type[BaseModel]], str]


# ---------------------------------------------------------------- repairer
class Repairer:
    """Repair actions, constraints and examples against the chunk text.

    `llm` (optional) is a callable (system, user, schema_model) -> response
    text, e.g. built by `make_llm_call`; without it only the deterministic
    step runs."""

    def __init__(self, vocab: Vocabulary | None = None, llm: LLMCall | None = None,
                 context_chars: int = 1200):
        self.vocab = vocab or load_vocabulary()
        self.llm = llm
        self.context_chars = context_chars
        # outcome of the LLM step of the last repair_* call, written into the removal
        # event when the element is removed: llm_off, llm_error:<type>, remove,
        # llm_correct_not_verified, llm_accepted (None: the LLM step was not reached)
        self.last_llm: str | None = None
        # single-word verbs of the vocabulary: skipped when looking for an object
        self._verbs = {s for spec in self.vocab.actions.values() for s in spec.get("synonyms", [])
                       if "_" not in s} | set(self.vocab.actions)

    # ------------------------------------------------------------ helpers
    def _sentence(self, element: dict, terms: list, text: str) -> tuple[int, int] | None:
        """The whole sentence(s) the element's quote is in, or the sentence with most
        of `terms` (which should not include the party: it occurs everywhere)."""
        hit = locate_quote(element.get("evidence"), text)
        if hit:
            spans = [(a, b) for a, b in sentence_spans(text) if a < hit[1] and b > hit[0]]
            return (spans[0][0], spans[-1][1]) if spans else (hit[0], hit[1])
        return infer_span(terms, text)

    def _context(self, text: str, span: tuple[int, int] | None) -> str:
        if not span:
            return text[: self.context_chars]
        spans = sentence_spans(text)
        idx = next((i for i, (a, b) in enumerate(spans) if a <= span[0] < b), 0)
        a = spans[max(0, idx - 1)][0] if spans else 0
        b = spans[min(len(spans) - 1, idx + 1)][1] if spans else len(text)
        return text[a:b][: self.context_chars]

    def _ask(self, what: str, element: dict, problem: str, expected: str, text: str,  # noqa: PLR0913
             span: tuple[int, int] | None) -> RepairDecision | None:
        if self.llm is None:
            self.last_llm = "llm_off"
            return None
        user = REPAIR_USER.format(context=self._context(text, span), what=what,
                                  element=json.dumps(element, ensure_ascii=False)[:800],
                                  problem=problem, expected=expected)
        try:
            raw = self.llm(REPAIR_SYSTEM, user, RepairDecision)
            data = json.loads(re.sub(r"^```(?:json)?\s*|```$", "", (raw or "").strip(), flags=re.M))
            decision = RepairDecision.model_validate(data)
        except Exception as exc:  # an unusable answer is the same as no answer, but it is recorded
            self.last_llm = f"llm_error:{type(exc).__name__}"
            logger.warning("Repair LLM call failed (%s: %s)", type(exc).__name__, str(exc)[:200])
            return None
        self.last_llm = decision.decision
        return decision

    # ------------------------------------------------------------ actions
    def repair_action(self, rule: dict, text: str, why: str,
                      ungrounded: Callable[[dict, str], str | None]) -> tuple[dict, str] | None:
        """(repaired rule, method) or None. `ungrounded(rule, text)` is the grounding
        test the repaired rule must pass (ontology_cleanup.action_ungrounded_reason)."""
        self.last_llm = None
        label = str(rule.get("action_label") or rule.get("action") or "")
        words = split_identifier(label)
        if not words:
            return None
        # without a quote, only the label's own words may pick the sentence
        span = self._sentence(rule, [label, rule.get("asset")], text)
        sentence = text[span[0]:span[1]] if span else ""

        def accept(new_words: list[str], method: str) -> tuple[dict, str] | None:
            new_label = "_".join(w for w in new_words if w)
            if not new_label or new_label == label:
                return None
            candidate = {**rule, "action_label": new_label}
            norm = self.vocab.normalize_action(OTHER_ACTION, new_label)
            candidate["action"] = norm["action"] if norm["controlled"] else OTHER_ACTION
            if ungrounded(candidate, text):
                return None
            return candidate, method

        # 1a. spelling variants of the words that are not in the text
        fixed = list(words)
        for i, w in enumerate(words):
            if len(w) >= 3 and _stem(w) not in text.lower():  # noqa: PLR2004
                fixed[i] = spelling_variant(w, text) or w
        out = accept(fixed, "spelling variant")
        if out:
            return out
        # 1b. a synonym of the verb that the rule's sentence uses (whole word)
        canonical = self.vocab.normalize_action(OTHER_ACTION, label)
        if canonical["controlled"] and sentence:
            for syn in self.vocab.actions[canonical["action"]].get("synonyms", []):
                parts = syn.split("_")
                if all(_word_in(p, sentence) for p in parts):
                    out = accept([*parts, *words[1:]], f"synonym '{syn}' in the sentence")
                    if out:
                        return out
        # 1c. rebuild the label from the sentence: verb after the modal + object
        vo = _modal_verb_object(sentence, self._verbs) if sentence else None
        if vo:
            out = accept([vo[0], vo[1]], "rebuilt from the sentence")
            if out:
                return out
        # 2. LLM
        decision = self._ask("ODRL rule", {k: rule.get(k) for k in ("action", "action_label", "party", "asset")},
                             f"action_label '{label}': {why}",
                             "the action as worded in the text, snake_case verb_object (e.g. inform_council)",
                             text, span)
        if decision and decision.decision == "correct":
            self.last_llm = "llm_correct_not_verified"
            out = accept(split_identifier(decision.value), "llm") if locate_quote(decision.evidence, text) else None
            if out:
                self.last_llm = "llm_accepted"
                return out[0], "llm"
        return None

    # -------------------------------------------------------- constraints
    def repair_constraint(self, constraint: dict, rule: dict, text: str,
                          why: str) -> tuple[dict, str] | None:
        """(repaired constraint, method) or None."""
        self.last_llm = None
        span = self._sentence(rule, [rule.get("action_label"), rule.get("asset")], text)
        sentence = text[span[0]:span[1]] if span else text
        left = constraint.get("left_operand")
        value = str(constraint.get("right_operand") or "")

        def accept(new: dict, method: str) -> tuple[dict, str] | None:
            if new == constraint or constraint_is_ungrounded(new, text):
                return None
            return new, method

        if why.startswith("currency"):
            found = list(dict.fromkeys(_CURRENCY.findall(sentence) or _CURRENCY.findall(text)))
            new = {**constraint, "unit": found[0] if len(found) == 1 else ""}
            out = accept(new, f"currency {'from the text' if len(found) == 1 else 'cleared'}")
            if out:
                return out
        if left == "dateTime" or re.match(r"\d{4}-\d{2}-\d{2}", value):
            dates = dates_in(sentence)
            if len(dates) == 1:
                out = accept({**constraint, "right_operand": dates[0]}, "the date of the rule's sentence")
                if out:
                    return out
            year = value[:4]
            if _YEAR.fullmatch(year or "") and year in sentence and not dates:
                out = accept({**constraint, "right_operand": year}, "the text gives only the year")
                if out:
                    return out
        amounts = amounts_in(sentence)
        if left in ("payAmount", None) or re.fullmatch(r"[\d ,.]+", value):
            if len(amounts) == 1:
                digits, currency = amounts[0]
                out = accept({**constraint, "right_operand": digits, "unit": currency or constraint.get("unit", "")},
                             "the amount of the rule's sentence")
                if out:
                    return out
        decision = self._ask("ODRL constraint", constraint, why,
                             f"the value of '{left}' as stated in the text (dates as YYYY-MM-DD)", text, span)
        if decision and decision.decision == "correct":
            self.last_llm = "llm_correct_not_verified"
            out = accept({**constraint, "right_operand": decision.value.strip()}, "llm")                 if locate_quote(decision.evidence, text) else None
            if out:
                self.last_llm = "llm_accepted"
                return out
        return None

    # ----------------------------------------------------------- examples
    def repair_example(self, example: dict, text: str) -> tuple[dict, str] | None:
        """(example with a value copied from the text, method) or None."""
        value = str(example.get("example_value") or "").strip()
        if len(value) < 2:  # noqa: PLR2004
            return None
        low = text.lower()
        # case / spacing / punctuation differences
        words = re.findall(r"[a-z0-9]+", value.lower())
        if words:
            pattern = r"\b" + r"[\W_]+".join(map(re.escape, words)) + r"\b"
            m = re.search(pattern, low)
            if m:
                return {**example, "example_value": text[m.start():m.end()]}, "same words in the text"
        # the same number written differently ("1131358" vs "1 131 358")
        try:
            number = float(re.sub(r"[ ,]", "", value))
        except ValueError:
            number = None
        if number is not None and round(number, 6) in _numbers_in_text(text):
            for m in re.finditer(r"\d[\d ,.]*\d|\d", text):
                try:
                    if float(re.sub(r"[ ,]", "", m.group(0))) == number:
                        return {**example, "example_value": m.group(0)}, "same number in the text"
                except ValueError:
                    continue
        # a date written in prose
        iso = re.match(r"(\d{4})-(\d{2})-(\d{2})$", value)
        if iso and _date_in_text(*iso.groups(), text):
            for d in _DATE_IN_PROSE.finditer(text):
                found = dates_in(d.group(0))
                if found and found[0] == value:
                    return {**example, "example_value": d.group(0)}, "same date in the text"
        return None


def make_llm_call(chat, backend: dict, model: str, temperature: float = 0.0,  # noqa: PLR0913
                  num_ctx: int | None = 4096, max_tokens: int | None = 256) -> LLMCall:
    """An `LLMCall` for `Repairer` on top of `chat` (utils/llm_client.llm_chat)."""
    def call(system: str, user: str, schema_model: type[BaseModel]) -> str:
        return chat(backend, model, system, user, temperature, schema_model=schema_model,
                    json_mode=True, num_ctx=num_ctx, max_tokens=max_tokens).content
    return call


def build_repairer(config: dict | None, vocabulary_path: str | None = None) -> Repairer | None:
    """`repair` parameters -> Repairer (None when disabled)."""
    if not config or not config.get("enabled", True):
        return None
    llm = None
    llm_cfg = config.get("llm") or {}
    if llm_cfg.get("enabled"):
        llm = make_llm_call(llm_chat, llm_cfg.get("llm_backend") or {}, llm_cfg["model"],
                            float(llm_cfg.get("temperature", 0.0)), llm_cfg.get("num_ctx", 4096),
                            llm_cfg.get("num_predict", 256))
    return Repairer(load_vocabulary(vocabulary_path), llm)


def actor_of(method: str) -> str:
    """Event actor of a repair: the LLM check acts as a reviewer, the rest is code."""
    return "llm_reviewer" if method == "llm" else "normalizer"
