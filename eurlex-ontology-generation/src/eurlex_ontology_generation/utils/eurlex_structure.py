"""Structural parsing and chunking of EUR-Lex acts.

The corpus text of an act is a single line (no newlines): title, citations
("Having regard to ..."), recitals ("Whereas: (1) ... (2) ..."), the enacting
formula ("HAVE ADOPTED THIS DECISION:"), articles ("Article 1 1. ... 2. ..."),
the closing formula and signatures ("Done at ..."), annexes and, at the end,
the footnotes ("(1) OJ L 347, 20.12.2013, p. 855.").

`parse_act` splits the text into units (preamble, recital, article paragraph,
annex, final, footnote) with character offsets on the input text.
`chunk_act` packs consecutive units of the same section into chunks of at most
`chunk_size` characters without mixing recitals and articles, never cutting a
word, and leaves footnotes and signatures out of the chunk body.

Numbered units are recognised only when they follow the expected numbering and
start a sentence, so inline footnote references ("Agreement (2) with") and
cross-references ("Article 5 of Regulation ...") are not taken as boundaries.

Every chunk starts with a header line ("§ Article 1, Article 2") so that the
section survives through review and revision, which only see the text, and
carries `segments` mapping chunk positions back to the act text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADER_MARK = "§ "

# ------------------------------------------------------------------ patterns
_ENACTING = re.compile(
    r"\bHA(?:S|VE)\s+(?:ADOPTED|DECIDED|DETERMINED)\b[^:]{0,120}?:"
    r"|\bHA(?:S|VE)\s+DECIDED\s+AS\s+FOLLOWS\b\s*:?"
    r"|\bHEREBY\s+(?:DECIDES?|ADOPTS?)\b\s*:?"
)
_WHEREAS = re.compile(r"\bWhereas\b\s*[:,]?")
_WHEREAS_ANY = re.compile(r"[;:,.]\s*whereas\b", re.I)
_DONE = re.compile(r"\bDone\s+at\s+[A-Z]")
_ANNEX = re.compile(r"\bANNEX(?:ES)?(?:\s+[IVXLC]+\b)?")
# older texts drop the spaces: "Article 1For the ...", "Article 2Provision of ..."
_ARTICLE = re.compile(r"\bArticle\s*(\d{1,3})([a-z]?)(?![\da-z])")
_PARAGRAPH = re.compile(r"(?<!\S)(\d{1,2})\.\s+(?=[A-Z(‘'\"])")
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z(‘'\"])")
# "(3) OJ L 347, 20.12.2013, p. 884." anywhere in the text (old OCR'd acts put
# footnotes at page breaks); the tail block at the end is found separately.
_INLINE_FOOTNOTE = re.compile(
    r"\(\d{1,3}\)\s?OJ\s+[LCA]\s?\d+[^()]{0,60}?p\s?\.\s?\d+(?:\s?-\s?\d+)?\s?\.?")
_FOOTNOTE_ITEM = re.compile(r"\((\d{1,3})\)\s")

_REF_AFTER_ARTICLE = re.compile(r"\s*(?:of\b|\(|thereof\b|,|and\b|to\b|shall\b|in\b|or\b|is\b|are\b)")

NON_BINDING = ("preamble", "recital")
PROVISION_OF_SECTION = {"article": "article", "recital": "recital", "annex": "annex",
                        "preamble": "other"}


@dataclass
class Unit:
    kind: str                     # preamble | recital | article | annex | final | text
    start: int
    end: int
    recital: int | None = None
    article: str | None = None
    paragraph: int | None = None


@dataclass
class ParsedAct:
    units: list[Unit]
    footnotes: dict[int, str] = field(default_factory=dict)
    removed: list[tuple[int, int]] = field(default_factory=list)   # footnote spans
    structured: bool = True


# ------------------------------------------------------------------- helpers
def _prev_char(text: str, pos: int) -> str:
    i = pos - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    return text[i] if i >= 0 else ""


def _starts_sentence(text: str, pos: int, extra: str = "") -> bool:
    return _prev_char(text, pos) in ".;:" + extra or _prev_char(text, pos) == ""


def _footnote_tail(text: str, after: int, end: int | None = None) -> int | None:
    """Start of the footnote block '(1) OJ ... (2) ...' that runs to `end`, if any."""
    end = len(text) if end is None else end
    for m in re.finditer(r"\(1\)\s", text[after:end]):
        p = after + m.start()
        items = re.split(r"\(\d{1,3}\)\s", text[p:end])[1:]
        if items and sum("OJ" in it for it in items) / len(items) >= 0.6:  # noqa: PLR2004
            return p
    return None


def _parse_footnotes(block: str) -> dict[int, str]:
    out: dict[int, str] = {}
    marks = list(_FOOTNOTE_ITEM.finditer(block))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(block)
        out.setdefault(int(m.group(1)), block[m.end():end].strip())
    return out


def _numbered_recitals(text: str, start: int, end: int) -> list[Unit]:
    units, expected, pos = [], 1, start
    starts: list[tuple[int, int]] = []
    while True:
        # the next number, tolerating up to 3 missing ones (OCR / editorial gaps)
        found, number = None, None
        for k in range(expected, expected + 4):
            pat = re.compile(r"\(" + str(k) + r"\)\s")
            for m in pat.finditer(text, pos, end):
                if m.start() == start or _starts_sentence(text, m.start(), ","):
                    if found is None or m.start() < found.start():
                        found, number = m, k
                    break
        if not found:
            break
        starts.append((number, found.start()))
        pos, expected = found.end(), number + 1
    for i, (n, s) in enumerate(starts):
        e = starts[i + 1][1] if i + 1 < len(starts) else end
        units.append(Unit("recital", s, e, recital=n))
    return units


def _unnumbered_recitals(text: str, start: int, end: int) -> list[Unit]:
    cuts = [start] + [m.start() + 1 for m in _WHEREAS_ANY.finditer(text, start, end)] + [end]
    return [Unit("recital", a, b, recital=i + 1)
            for i, (a, b) in enumerate(zip(cuts, cuts[1:])) if text[a:b].strip()]


def _articles(text: str, start: int, end: int) -> list[tuple[str, int, int]]:
    """(number, heading_start, heading_end) of the articles, in order."""
    out: list[tuple[str, int, int]] = []
    current = 0
    for m in _ARTICLE.finditer(text, start, end):
        num, suffix = int(m.group(1)), m.group(2)
        follows = num == current + 1 or (num == current and suffix) or (not out and num >= 1)
        if not follows:
            continue
        if out and not _starts_sentence(text, m.start(), ")"):
            continue
        if not out and m.start() != start and not _starts_sentence(text, m.start(), ")"):
            continue
        if _REF_AFTER_ARTICLE.match(text, m.end()):
            continue
        out.append((f"{num}{suffix}", m.start(), m.end()))
        current = num
    return out


def _paragraphs(text: str, start: int, end: int, heading_end: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    expected = 1
    for m in _PARAGRAPH.finditer(text, heading_end, end):
        if int(m.group(1)) != expected:
            continue
        first_ok = expected == 1 and m.start() - heading_end <= 150  # noqa: PLR2004
        if not (first_ok or _starts_sentence(text, m.start())):
            continue
        out.append((expected, m.start()))
        expected += 1
    return out if len(out) >= 2 else []  # noqa: PLR2004


# --------------------------------------------------------------------- parse
def _footnote_blocks(text: str, after: int, annex) -> tuple[list[tuple[int, int]], dict[int, str]]:
    """Footnote blocks between the signatures and the annex, and at the very end."""
    n = len(text)
    ends = [annex.start(), n] if annex else [n]
    starts = [after, annex.end()] if annex else [after]
    removed, footnotes = [], {}
    for a, b in zip(starts, ends):
        p = _footnote_tail(text, a, b)
        if p is not None:
            removed.append((p, b))
            for k, v in _parse_footnotes(text[p:b]).items():
                footnotes.setdefault(k, v)
    return removed, footnotes


def _inline_footnotes(text: str, removed: list, footnotes: dict) -> None:
    """Footnotes in the middle of the text (old OCR'd acts), added to `removed`."""
    blocks = list(removed)
    for m in _INLINE_FOOTNOTE.finditer(text):
        if any(a <= m.start() < b for a, b in blocks):
            continue
        removed.append((m.start(), m.end()))
        mark = _FOOTNOTE_ITEM.match(m.group(0))
        if mark:
            footnotes.setdefault(int(mark.group(1)), m.group(0)[mark.end():].strip())


def _preamble_units(text: str, pre_end: int) -> list[Unit]:
    """Title + citations, then the recitals."""
    units: list[Unit] = []
    wh = _WHEREAS.search(text, 0, pre_end)
    cit_end = wh.start() if wh else pre_end
    if text[:cit_end].strip():
        units.append(Unit("preamble", 0, cit_end))
    if wh:
        rec = _numbered_recitals(text, wh.end(), pre_end)
        if len(rec) < 2:  # noqa: PLR2004
            rec = _unnumbered_recitals(text, wh.start(), pre_end)
        if rec and rec[0].start > wh.start():
            rec[0].start = wh.start()   # keep "Whereas:" with the first recital
        units.extend(rec)
    return units


def _article_units(text: str, arts: list, body_end: int) -> list[Unit]:
    units: list[Unit] = []
    for i, (num, h_start, h_end) in enumerate(arts):
        a_end = arts[i + 1][1] if i + 1 < len(arts) else body_end
        paras = _paragraphs(text, h_start, a_end, h_end)
        if not paras:
            units.append(Unit("article", h_start, a_end, article=num))
            continue
        for j, (p, p_start) in enumerate(paras):
            s = h_start if j == 0 else p_start
            e = paras[j + 1][1] if j + 1 < len(paras) else a_end
            units.append(Unit("article", s, e, article=num, paragraph=p))
    return units


def parse_act(text: str) -> ParsedAct:
    """Split an act into structural units with offsets on `text`."""
    if not isinstance(text, str) or not text.strip():
        return ParsedAct([], structured=False)
    n = len(text)

    enact = _ENACTING.search(text)
    done = _DONE.search(text, enact.end() if enact else 0)
    annex = _ANNEX.search(text, done.end()) if done else None
    after = done.start() if done else (enact.end() if enact else 0)
    removed, footnotes = _footnote_blocks(text, after, annex)
    first_end = annex.start() if annex else n
    tail = removed[0][0] if removed and removed[0][1] == first_end else None
    stop = tail if tail is not None else first_end

    body_start = enact.end() if enact else 0
    body_end = done.start() if done else stop
    arts = _articles(text, body_start, body_end)
    if not enact and arts:
        body_start = arts[0][1]
    structured = bool(enact or arts)

    if structured:
        pre_end = enact.start() if enact else arts[0][1]
        units = _preamble_units(text, pre_end) if pre_end > 0 else []
        units += _article_units(text, arts, body_end)
        if not arts and body_end > body_start:
            units.append(Unit("text", body_start, body_end))
        if done:   # closing formula / signatures, then annexes (footnotes cut out below)
            units.append(Unit("final", done.start(), stop))
            if annex:
                units.append(Unit("annex", annex.start(), n))
    else:
        units = [Unit("text", 0, n)]

    _inline_footnotes(text, removed, footnotes)
    units = [u for u in units if text[u.start:u.end].strip()]
    return ParsedAct(units, footnotes, sorted(removed), structured)


# --------------------------------------------------------------------- chunk
def _spans(start: int, end: int, removed: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """[start, end) minus the removed spans."""
    out, pos = [], start
    for a, b in removed:
        if b <= pos or a >= end:
            continue
        if a > pos:
            out.append((pos, a))
        pos = max(pos, b)
    if pos < end:
        out.append((pos, end))
    return out


def _pieces(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """Split [start, end) at sentence ends, then at spaces, into pieces <= size."""
    if end - start <= size:
        return [(start, end)]
    cuts = [start] + [m.end() for m in _SENTENCE_END.finditer(text, start, end)] + [end]
    sentences = list(zip(cuts, cuts[1:]))
    out: list[tuple[int, int]] = []
    for s_start, s_end in sentences:
        a = s_start
        while s_end - a > size:
            cut = text.rfind(" ", a, a + size)
            cut = cut if cut > a else a + size
            out.append((a, cut))
            a = cut
        out.append((a, s_end))
    # re-pack consecutive sentences up to `size`
    packed: list[tuple[int, int]] = []
    for a, b in out:
        if packed and b - packed[-1][0] <= size:
            packed[-1] = (packed[-1][0], b)
        else:
            packed.append((a, b))
    return packed


def _range(values: list) -> str:
    values = list(dict.fromkeys(v for v in values if v is not None))
    if not values:
        return ""
    return str(values[0]) if len(values) == 1 else f"{values[0]}–{values[-1]}"


def _header(kind: str, units: list[Unit], continued: bool) -> str:
    if kind == "recital":
        label = f"Recitals ({_range([u.recital for u in units])})".replace("–", ")–(")
    elif kind == "article":
        parts = []
        for art in dict.fromkeys(u.article for u in units):
            paras = _range([u.paragraph for u in units if u.article == art])
            parts.append(f"Article {art}" + (f"({paras})".replace("–", ")–(") if paras else ""))
        label = ", ".join(parts)
    else:
        label = {"preamble": "Preamble", "annex": "Annex", "text": "Text"}.get(kind, kind)
    return HEADER_MARK + label + (" (continued)" if continued else "")


def chunk_act(text: str, chunk_size: int = 2000, include_preamble: bool = True,
              include_annexes: bool = True, include_final: bool = False) -> list[dict]:
    """Chunks of one act: dicts with chunk_text, section_type, offsets and segments.

    `chunk_size` bounds the body; the one-line header comes on top of it."""
    parsed = parse_act(text)
    groups: list[tuple[str, list[Unit], list[tuple[int, int]], bool]] = []
    skip = {k for k, keep in (("final", include_final), ("preamble", include_preamble),
                              ("annex", include_annexes)) if not keep}

    for unit in parsed.units:
        if unit.kind in skip:
            continue
        spans = [p for a, b in _spans(unit.start, unit.end, parsed.removed)
                 for p in _pieces(text, a, b, chunk_size)]
        for i, span in enumerate(spans):
            length = span[1] - span[0]
            last = groups[-1] if groups else None
            same = last and last[0] == unit.kind and unit.kind != "preamble"
            size = sum(b - a for a, b in last[2]) if last else 0
            if same and size + length + 1 <= chunk_size:
                if unit not in last[1]:
                    last[1].append(unit)
                last[2].append(span)
            else:
                continued = i > 0 or bool(same and unit in last[1])
                groups.append((unit.kind, [unit], [span], continued))

    chunks = []
    for kind, units, spans, continued in groups:
        header = _header(kind, units, continued)
        body, segments = "", []
        offset = len(header) + 1
        for a, b in spans:
            piece = text[a:b].strip()
            if not piece:
                continue
            lead = len(text[a:b]) - len(text[a:b].lstrip())
            if body:
                body += " "
            segments.append([a + lead, a + lead + len(piece), offset + len(body)])
            body += piece
        if not body:
            continue
        chunks.append({
            "chunk_text": f"{header}\n{body}",
            "section_type": kind,
            "section_label": header[len(HEADER_MARK):],
            "articles": list(dict.fromkeys(u.article for u in units if u.article)),
            "paragraphs": [u.paragraph for u in units if u.paragraph],
            "recitals": [u.recital for u in units if u.recital],
            "char_start": segments[0][0],
            "char_end": segments[-1][1],
            "segments": segments,
            "footnotes": _referenced_footnotes(body, parsed.footnotes),
            "chunking": "structural" if parsed.structured else "sentences",
        })
    return chunks


def _referenced_footnotes(body: str, footnotes: dict[int, str]) -> dict[str, str]:
    """Footnotes whose marker ' (n)' appears in the body without starting a sentence
    ('Article 3(3)' is a cross-reference, not a footnote marker)."""
    out = {}
    for m in re.finditer(r"(?<=\s)\((\d{1,3})\)", body):
        k = int(m.group(1))
        if k in footnotes and not _starts_sentence(body, m.start()):
            out[str(k)] = footnotes[k]
    return out


def chunk_to_act_offset(pos: int, segments: list) -> int | None:
    """Map a position in `chunk_text` to the offset in the act text."""
    for a, b, c in segments:
        if c <= pos < c + (b - a):
            return a + (pos - c)
    return None


def section_of(text: str) -> str | None:
    """Section type from the chunk header ('§ Article 3' -> 'article'), or None."""
    if not isinstance(text, str) or not text.startswith(HEADER_MARK):
        return None
    label = text[len(HEADER_MARK):].split("\n", 1)[0].lower()
    for key, kind in (("recital", "recital"), ("article", "article"), ("annex", "annex"),
                      ("preamble", "preamble"), ("text", "text")):
        if label.startswith(key):
            return kind
    return None


# ------------------------------------------------------------- legacy windows
def window_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Fixed-size windows that never cut a word (legacy mode)."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size.")
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            cut = text.rfind(" ", start + 1, end)
            end = cut if cut > start else end
        piece = text[start:end]
        if piece.strip():
            chunks.append({"chunk_text": piece, "section_type": None, "section_label": None,
                           "articles": [], "paragraphs": [], "recitals": [],
                           "char_start": start, "char_end": end,
                           "segments": [[start, end, 0]], "footnotes": {},
                           "chunking": "window"})
        if end >= n:
            break
        nxt = max(end - chunk_overlap, start + 1)
        space = text.find(" ", nxt, end)
        start = space + 1 if space != -1 else nxt
    return chunks
