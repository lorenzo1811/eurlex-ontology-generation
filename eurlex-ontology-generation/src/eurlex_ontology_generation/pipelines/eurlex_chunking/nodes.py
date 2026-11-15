import json

import pandas as pd

from ...utils.eurlex_structure import chunk_act, window_chunks

META_COLUMNS = ["CELEX", "Act_name", "Act_type", "Subject_matter", "Date_document"]
JSON_COLUMNS = ["articles", "paragraphs", "recitals", "segments", "footnotes"]


def chunk_corpus(  # noqa: PLR0913
    corpus: pd.DataFrame,
    text_column: str,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str = "structural",
    include_preamble: bool = True,
    include_annexes: bool = True,
) -> pd.DataFrame:
    """Split each EUR-Lex act into chunks, keeping the act's metadata.

    strategy "structural" (default) follows the act's structure (preamble,
    recitals, articles/paragraphs, annexes; see utils/eurlex_structure.py),
    drops footnotes and signatures and never cuts a word; "window" is the
    legacy fixed-size window (now cut at word boundaries), where
    `chunk_overlap` applies. Offsets (`char_start`, `char_end`, `segments`)
    refer to the text in `eurlex_corpus`.
    """
    if strategy not in ("structural", "window"):
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size.")

    chunks = []
    for _, row in corpus.iterrows():
        text = row[text_column]
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        if strategy == "structural":
            pieces = chunk_act(text, chunk_size, include_preamble=include_preamble,
                               include_annexes=include_annexes)
        else:
            pieces = window_chunks(text, chunk_size, chunk_overlap)

        for number, piece in enumerate(pieces):
            record = {c: row[c] for c in META_COLUMNS if c in row}
            record.update({
                "chunk_id": f"{row['CELEX']}_{number}",
                "chunk_number": number,
                **{k: (json.dumps(v, ensure_ascii=False) if k in JSON_COLUMNS else v)
                   for k, v in piece.items()},
            })
            chunks.append(record)

    return pd.DataFrame(chunks)
