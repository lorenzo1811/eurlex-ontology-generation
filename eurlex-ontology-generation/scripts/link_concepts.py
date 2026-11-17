"""Link the concepts of the refined ontology to Wikidata and EuroVoc / EU corporate bodies.

Runs after the merge (utils/concept_linking.py explains the method). Settings in
conf/base/parameters_concept_linking.yml (override in conf/local/).

  # once, or when the encoder changes: download EuroVoc + corporate bodies
  # (SPARQL, or --eurovoc-skos / --corporate-body-skos files) and encode them
  uv run --extra linking python scripts/link_concepts.py index

  # link, then re-run refinement + export so the links reach the Turtle
  uv run --extra linking python scripts/link_concepts.py link --apply

  # without GPU / ReLiK: exact Wikidata labels + an Ollama embedding index
  uv run python scripts/link_concepts.py index --encoder-kind embed --encoder-model nomic-embed-text
  uv run python scripts/link_concepts.py link --no-relik --apply

The links file (default data/05_model_input/concept_links.json) is applied by
ontology_refinement (links_path); gazetteer identifiers always win.
"""
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eurlex_ontology_generation.utils.concept_linking import (  # noqa: E402
    ConceptLinker,
    VocabularyIndex,
    WikidataSearch,
    WikipediaQids,
    fetch_skos_concepts,
    load_skos_file,
    make_encoder,
    make_llm_chooser,
    relik_annotator,
)
from eurlex_ontology_generation.utils.entity_linking import WikidataLinker  # noqa: E402
from eurlex_ontology_generation.utils.llm_client import llm_chat  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    cfg = {}
    for path in (ROOT / "conf/base/parameters_concept_linking.yml", ROOT / "conf/local/parameters_concept_linking.yml"):
        if path.exists():
            data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("concept_linking", {})
            cfg = _deep_update(cfg, data)
    return cfg


def _deep_update(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in new.items():
        out[k] = _deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _encoder_cfg(cfg: dict, args) -> dict:
    enc = dict(cfg.get("encoder") or {})
    if args.encoder_kind:
        enc["kind"] = args.encoder_kind
    if args.encoder_model:
        enc["model"] = args.encoder_model
    if enc.get("kind") == "embed":
        enc.setdefault("backend", (cfg.get("llm") or {}).get("backend"))
    return enc


# ---------------------------------------------------------------- index
def cmd_index(cfg: dict, args) -> None:
    records = []
    for source, vcfg in (cfg.get("vocabularies") or {}).items():
        if source == "corporate_body" and args.no_corporate_bodies:
            continue
        local = getattr(args, f"{source}_skos", None) or vcfg.get("skos_file")
        if local:
            got = load_skos_file(local, source, vcfg.get("scheme"))
        else:
            print(f"downloading {source} from the Publications Office SPARQL endpoint ...")  # noqa: T201
            got = fetch_skos_concepts(vcfg["scheme"], source, user_agent=cfg.get("user_agent"))
        print(f"{source}: {len(got)} concepts")  # noqa: T201
        records += got
    enc_cfg = _encoder_cfg(cfg, args)
    index = VocabularyIndex.build(records, make_encoder(enc_cfg), meta={"encoder": enc_cfg})
    index.save(args.index or cfg["index_directory"])
    print(f"index: {len(records)} concepts -> {args.index or cfg['index_directory']}")  # noqa: T201


# ---------------------------------------------------------------- link
def chunk_texts(patterns: list[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(str(ROOT / pattern))):
            df = pd.read_csv(path)
            col = "text" if "text" in df else "chunk_text" if "chunk_text" in df else None
            if col is None:
                continue
            for cid, text in zip(df["chunk_id"], df[col]):
                if isinstance(text, str):
                    lines = text.replace("\r\n", "\n").split("\n", 1)     # drop the '§ section' header
                    texts[str(cid)] = lines[1] if lines[0].startswith("§") and len(lines) > 1 else text
    return texts


def act_descriptors(path: str | None) -> dict[str, list[str]]:
    if not path or not (ROOT / path).exists():
        return {}
    df = pd.read_csv(ROOT / path, usecols=["CELEX", "EUROVOC"])
    return {str(c): [d.strip() for d in str(v).split(";") if d.strip()]
            for c, v in zip(df["CELEX"], df["EUROVOC"]) if isinstance(v, str)}


def relik_spans(cfg: dict, texts: dict[str, str], names_of: dict[str, list[str]],
                search: WikidataSearch) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """ReLiK spans per chunk, with candidates from Wikidata for the chunk's concepts.
    Returns (spans per chunk, Wikipedia title -> QID of the candidates)."""
    rcfg = cfg.get("relik") or {}
    cache_path = ROOT / rcfg.get("spans_cache", "data/02_intermediate/linking/relik_spans.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    title_qid: dict[str, str] = {}
    candidates: dict[str, list[str]] = {}
    distinct = list(dict.fromkeys(n for names in names_of.values() for n in names))
    for i, name in enumerate(distinct, 1):           # search once per name, with progress
        search.search(name)
        if i % 10 == 0 or i == len(distinct):
            search.save()
            print(f"Wikidata search: {i}/{len(distinct)} names, {search.failures} failed", flush=True)  # noqa: T201
    for cid, names in names_of.items():
        titles = []
        for name in names:
            for hit in search.cache.get(search.query_of(name).lower(), []):
                if hit.get("title"):
                    title_qid.setdefault(hit["title"], hit["qid"])
                    if hit["title"] not in titles:
                        titles.append(hit["title"])
        candidates[cid] = titles[: int(rcfg.get("max_candidates", 60))]
    search.save()
    asked = sum(len(n) for n in names_of.values())
    if search.failures:
        print(f"Wikidata search: {search.failures} failed lookups (last: {search.last_error})")  # noqa: T201
        if search.failures > len(search.cache):
            sys.exit("Most Wikidata searches failed: Wikidata links would be meaningless. Re-run later "
                     "(the searches that worked are cached), or set a user_agent with a contact in "
                     "conf/local/parameters_concept_linking.yml (Wikimedia user-agent policy).")
    print(f"Wikidata search: {len(search.cache)} names cached, {asked} name mentions in chunks")  # noqa: T201
    # a chunk is annotated again only when its candidates change
    todo = [c for c in names_of if c in texts and (cache.get(c) or {}).get("candidates") != candidates[c]]
    if todo:
        annotate = relik_annotator(rcfg["reader_model"], rcfg.get("device"), int(rcfg.get("window_size", 32)),
                                   int(rcfg.get("window_stride", 16)), rcfg.get("precision"))
        for n, cid in enumerate(todo, 1):
            cache[cid] = {"candidates": candidates[cid], "spans": annotate(texts[cid], candidates[cid])}
            if n % 10 == 0 or n == len(todo):
                print(f"ReLiK: {n}/{len(todo)} chunks")  # noqa: T201
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return {c: (cache.get(c) or {}).get("spans", []) for c in names_of}, title_qid


def cmd_link(cfg: dict, args) -> None:  # noqa: PLR0912, PLR0915
    refined_path = ROOT / (args.refined or cfg["refined_path"])
    refined = json.loads(refined_path.read_text(encoding="utf-8"))
    texts = chunk_texts(cfg.get("chunk_text_files") or [])
    names_of: dict[str, list[str]] = {}      # chunk -> concept names (and variants) found in it
    for e in refined.get("classes", []) + refined.get("individuals", []):
        for p in e.get("provenance") or []:
            if isinstance(p, dict) and p.get("chunk_id") in texts:
                names = names_of.setdefault(p["chunk_id"], [])
                for n in [e["name"], *(e.get("merged_from") or [])]:
                    if n not in names:
                        names.append(n)

    index_dir = ROOT / (args.index or cfg["index_directory"])
    index = encode = None
    if args.no_eurovoc:
        pass
    elif (index_dir / "records.jsonl").exists():
        index = VocabularyIndex.load(str(index_dir))
        encode = make_encoder(index.meta.get("encoder") or _encoder_cfg(cfg, args))
    else:
        sys.exit(f"No EuroVoc index in {index_dir}: run `link_concepts.py index` first "
                 "(or pass --no-eurovoc to link to Wikidata only).")

    choose = None
    llm_cfg = cfg.get("llm") or {}
    if llm_cfg.get("enabled") and not args.no_llm:
        choose = make_llm_chooser(llm_chat, llm_cfg.get("backend") or {}, llm_cfg["model"],
                                  float(llm_cfg.get("temperature", 0.0)), llm_cfg.get("num_ctx", 4096),
                                  llm_cfg.get("num_predict", 200))

    spans, title_qid = None, {}
    wikipedia = WikipediaQids(str(ROOT / cfg["wikipedia_cache"]), cfg.get("user_agent"))
    if (cfg.get("relik") or {}).get("enabled", True) and not args.no_relik and not args.no_wikidata:
        search = WikidataSearch(str(ROOT / cfg["wikidata_search_cache"]), cfg.get("user_agent"))
        try:
            spans, title_qid = relik_spans(cfg, texts, names_of, search)
        except ImportError as exc:     # never a silent fallback: the run would not measure ReLiK
            sys.exit(f"ReLiK cannot be imported ({type(exc).__name__}: {exc}). Run with "
                     "`uv run --extra linking ...`, or pass --no-relik to link by exact Wikidata labels.")

    def qids(titles: list[str]) -> dict[str, str | None]:
        out = {t: title_qid.get(t) for t in titles}
        missing = [t for t in titles if not out[t]]
        if missing:
            out.update(wikipedia.resolve(missing))
        return out
    label_lookup = None
    if spans is None and not args.no_wikidata:
        linker = WikidataLinker(str(ROOT / "data/02_intermediate/wikidata_cache.json"), cfg.get("user_agent"),
                                max_lookups=1000)
        label_lookup = linker.lookup

    concept_linker = ConceptLinker(index=index, encode=encode, qids=qids, choose=choose,
                                   act_descriptors=act_descriptors(cfg.get("corpus_path")),
                                   wikidata_label_lookup=label_lookup, k=int(cfg.get("k", 8)),
                                   prior_bonus=float(cfg.get("prior_bonus", 0.05)))
    links = concept_linker.link(refined, texts, spans)
    wikipedia.save()
    if label_lookup is not None:
        linker.save()
    links["meta"] = {"refined": str(refined_path.relative_to(ROOT).as_posix()),
                     "relik": spans is not None, "eurovoc_index": index is not None,
                     "encoder": (index.meta.get("encoder") if index else None), "llm": choose is not None}
    out = ROOT / (args.out or cfg["links_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(links["stats"], indent=2))  # noqa: T201
    print(f"links -> {out}")  # noqa: T201

    if args.apply:
        for pipeline in ("ontology_refinement", "ontology_export"):
            code = subprocess.run(["uv", "run", "--no-sync", "kedro", "run", "--pipelines", pipeline],  # noqa: S607
                                  cwd=ROOT, check=False).returncode
            if code:
                sys.exit(f"{pipeline} failed (exit {code})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index", help="build the EuroVoc / corporate-body index")
    pi.add_argument("--eurovoc-skos", dest="eurovoc_skos")
    pi.add_argument("--corporate-body-skos", dest="corporate_body_skos")
    pi.add_argument("--no-corporate-bodies", action="store_true")
    pl = sub.add_parser("link", help="link the refined ontology")
    pl.add_argument("--refined")
    pl.add_argument("--out")
    pl.add_argument("--no-relik", action="store_true")
    pl.add_argument("--no-llm", action="store_true")
    pl.add_argument("--no-wikidata", action="store_true", help="no network lookups for Wikidata")
    pl.add_argument("--no-eurovoc", action="store_true", help="skip EuroVoc (no index needed)")
    pl.add_argument("--apply", action="store_true", help="re-run ontology_refinement + ontology_export")
    for sp in (pi, pl):
        sp.add_argument("--index", help="index directory")
        sp.add_argument("--encoder-kind", choices=["relik", "hf", "embed"])
        sp.add_argument("--encoder-model")
    args = p.parse_args()
    cfg = load_config()
    (cmd_index if args.cmd == "index" else cmd_link)(cfg, args)


if __name__ == "__main__":
    main()
