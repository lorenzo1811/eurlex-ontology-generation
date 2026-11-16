"""Export the refined ontology (JSON) to OWL 2 / ODRL 2.2 in RDF (Turtle).

Layout of the graph
  eo:   project vocabulary   <base>ontology#   core classes, upper ontology, properties
  ent:  entities             <base>entity/     domain classes and individuals
  act:  actions              <base>action/     canonical actions (odrl:Action)
  pol:  policy and rules     <base>policy/     one odrl:Set with all rules

TBox
  eo:Party ≡ odrl:Party, eo:Asset ≡ odrl:Asset, eo:LegalExtension; pairwise disjoint.
  Upper classes (eo:Fund, eo:Country, ...) are subclasses of their core class.
  Domain classes (kind=class) are owl:Class, rdfs:subClassOf their upper class.
ABox
  Individuals (kind=individual) are owl:NamedIndividual typed with their upper
  class; Wikidata IDs as owl:sameAs, EU Vocabularies IRIs as skos:exactMatch
  (authority-table IRIs denote SKOS concepts, not the entities themselves).
Rules
  odrl:Permission / odrl:Prohibition / odrl:Duty with odrl:action, odrl:assignee,
  odrl:assigner, odrl:target and odrl:constraint (leftOperand, operator,
  rightOperand typed as xsd:date / xsd:decimal where possible, odrl:unit).
Relationships
  eo:<relation> object-property assertions. When an endpoint is a class this
  relies on OWL 2 punning (allowed in OWL 2 DL).
Provenance
  Every class, individual, rule and constraint links with prov:wasDerivedFrom
  to eo:TextEvidence nodes (ev: namespace), one per quoted span: dct:source is
  the CELEX resource of the act, eo:chunkId / eo:section / eo:article /
  eo:paragraph / eo:recital locate it, and W3C Web Annotation selectors hold the
  quote (oa:TextQuoteSelector) and the character offsets in the act text of the
  project corpus (oa:TextPositionSelector). eo:extractionMatch tells how the
  span was found (exact, normalized, fuzzy, inferred; see utils/provenance.py).

Every entity also carries eo:entityKind ("class"/"individual") and eo:coreType
literals, which the SHACL shapes use as targets.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from .vocabulary import Vocabulary, load_vocabulary, parse_date

ODRL = Namespace("http://www.w3.org/ns/odrl/2/")
WD = Namespace("http://www.wikidata.org/entity/")
PROV = Namespace("http://www.w3.org/ns/prov#")
OA = Namespace("http://www.w3.org/ns/oa#")
CELLAR_CELEX = "http://publications.europa.eu/resource/celex/"
EU_CURRENCY = Namespace("http://publications.europa.eu/resource/authority/currency/")
DEFAULT_BASE_IRI = "https://w3id.org/eurlex-odrl/"
ONTOLOGY_VERSION = "0.1.0"
SOURCE_REPOSITORY = "https://github.com/jmrozanec/eurlex-ontology-generation"

CORE_CLASS_LOCAL = {"Party": "Party", "Asset": "Asset", "Legal_extension": "LegalExtension"}
# CELEX sectors and, for sector 3 (legislation), the act-type letters.
CELEX_SECTORS = {"1": "Treaties", "2": "International agreements", "3": "Legislation",
                 "4": "Complementary legislation", "5": "Preparatory documents", "6": "Case-law",
                 "7": "National transposition", "8": "National case-law", "9": "Parliamentary questions",
                 "0": "Consolidated texts", "C": "Official Journal C series", "E": "EFTA documents"}
CELEX_TYPES = {"3": {"R": "Regulation", "L": "Directive", "D": "Decision", "H": "Recommendation"}}

RULE_CLASS = {"permissions": ODRL.Permission, "prohibitions": ODRL.Prohibition, "duties": ODRL.Duty}
RULE_LINK = {"permissions": ODRL.permission, "prohibitions": ODRL.prohibition, "duties": ODRL.obligation}


def namespaces(base_iri: str = DEFAULT_BASE_IRI) -> dict[str, Namespace]:
    base = base_iri if base_iri.endswith(("/", "#")) else base_iri + "/"
    return {"eo": Namespace(base + "ontology#"), "ent": Namespace(base + "entity/"),
            "act": Namespace(base + "action/"), "pol": Namespace(base + "policy/"),
            "ev": Namespace(base + "evidence/")}


def add_evidence(g: Graph, ns: dict, node, items: Any) -> int:
    """prov:wasDerivedFrom links from `node` to one eo:TextEvidence per item."""
    eo, ev = ns["eo"], ns["ev"]
    n = 0
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not item.get("quote"):
            continue
        key = "|".join(str(item.get(k) or "") for k in ("celex", "chunk_id", "start", "end", "quote"))
        e = ev[hashlib.sha1(key.encode()).hexdigest()[:16]]
        g.add((node, PROV.wasDerivedFrom, e))
        n += 1
        if (e, RDF.type, eo.TextEvidence) in g:
            continue
        g.add((e, RDF.type, eo.TextEvidence))
        g.add((e, RDF.type, PROV.Entity))
        if item.get("celex"):
            g.add((e, DCTERMS.source, URIRef(CELLAR_CELEX + str(item["celex"]))))
            g.add((e, eo.celex, Literal(str(item["celex"]))))
        for key_, prop in (("chunk_id", eo.chunkId), ("section", eo.section), ("article", eo.article),
                           ("match", eo.extractionMatch)):
            if item.get(key_) is not None:
                g.add((e, prop, Literal(str(item[key_]))))
        for key_, prop in (("paragraph", eo.paragraph), ("recital", eo.recital)):
            if isinstance(item.get(key_), int):
                g.add((e, prop, Literal(item[key_], datatype=XSD.integer)))
        quote = BNode()
        g.add((e, OA.hasSelector, quote))
        g.add((quote, RDF.type, OA.TextQuoteSelector))
        g.add((quote, OA.exact, Literal(str(item["quote"]))))
        if isinstance(item.get("start"), int) and isinstance(item.get("end"), int):
            pos = BNode()
            g.add((e, OA.hasSelector, pos))
            g.add((pos, RDF.type, OA.TextPositionSelector))
            g.add((pos, OA.start, Literal(item["start"], datatype=XSD.nonNegativeInteger)))
            g.add((pos, OA.end, Literal(item["end"], datatype=XSD.nonNegativeInteger)))
    return n


def local_name(name: str) -> str:
    """IRI-safe local name: 'EU Solidarity Fund (2018)' -> 'EU_Solidarity_Fund_2018'."""
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name)).strip("_")
    if not s:
        s = "x"
    return s if s[0].isalpha() else f"e_{s}"


def _rule_id(kind: str, rule: dict) -> str:
    key = "|".join(str(rule.get(k) or "") for k in ("action", "action_object", "party", "asset"))
    prefix = {"permissions": "permission", "prohibitions": "prohibition", "duties": "duty"}[kind]
    return f"{prefix}_{hashlib.sha1(f'{kind}|{key}'.encode()).hexdigest()[:12]}"


def _typed_right_operand(left: str, value: Any) -> Literal:
    text = str(value if value is not None else "").strip()
    if left == "dateTime":
        iso = parse_date(text)
        if iso:
            return Literal(iso, datatype=XSD.date)
    if left in ("payAmount", "percentage", "count", "elapsedTime") and re.fullmatch(r"-?\d+(\.\d+)?", text):
        return Literal(text, datatype=XSD.decimal)
    return Literal(text)


def build_tbox(g: Graph, ns: dict, vocab: Vocabulary) -> None:
    eo = ns["eo"]
    onto = URIRef(str(eo).rstrip("#"))
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, OWL.versionInfo, Literal(ONTOLOGY_VERSION)))
    g.add((onto, OWL.versionIRI, URIRef(f"{str(eo).rstrip('#')}/{ONTOLOGY_VERSION}")))
    g.add((onto, DCTERMS.source, URIRef(SOURCE_REPOSITORY)))
    g.add((onto, RDFS.label, Literal("EurLex ODRL ontology", lang="en")))
    g.add((onto, RDFS.comment, Literal(
        "Ontology extracted from EUR-Lex with an LLM pipeline, refined and aligned to ODRL 2.2.", lang="en")))

    core = {}
    for core_type, localname in CORE_CLASS_LOCAL.items():
        c = eo[localname]
        core[core_type] = c
        g.add((c, RDF.type, OWL.Class))
        g.add((c, RDFS.label, Literal(core_type.replace("_", " "), lang="en")))
        g.add((c, eo.coreClass, Literal(True)))
    g.add((core["Party"], OWL.equivalentClass, ODRL.Party))
    g.add((core["Asset"], OWL.equivalentClass, ODRL.Asset))
    disjoint = BNode()
    g.add((disjoint, RDF.type, OWL.AllDisjointClasses))
    members = BNode()
    from rdflib.collection import Collection
    Collection(g, members, list(core.values()))
    g.add((disjoint, OWL.members, members))

    for name, spec in vocab.upper.items():
        c = eo[name]
        g.add((c, RDF.type, OWL.Class))
        g.add((c, RDFS.subClassOf, core[spec["core_type"]]))
        g.add((c, RDFS.label, Literal(name, lang="en")))
        g.add((c, SKOS.definition, Literal(spec["description"], lang="en")))
        g.add((c, eo.coreType, Literal(spec["core_type"])))
        g.add((c, eo.upperClass, Literal(True)))

    for p in ("entityKind", "coreType", "supportChunks", "supportDocuments", "supportApprovedChunks",
              "typeSource", "kindConfidence", "actionObject", "actionLabel", "provision",
              "mergedFrom", "upperClass", "coreClass", "status", "celex", "actType", "year",
              "chunkId", "section", "article", "paragraph", "recital", "extractionMatch"):
        g.add((eo[p], RDF.type, OWL.AnnotationProperty))
    g.add((eo.TextEvidence, RDF.type, OWL.Class))
    g.add((eo.TextEvidence, RDFS.label, Literal("Text evidence", lang="en")))
    g.add((eo.TextEvidence, RDFS.comment, Literal(
        "A quoted span of a source act that an extracted element rests on.", lang="en")))
    g.add((eo.relatedIndividual, RDF.type, OWL.ObjectProperty))
    g.add((eo.memberState, RDF.type, OWL.DatatypeProperty))

    for name in vocab.relations:
        g.add((eo[name], RDF.type, OWL.ObjectProperty))
        g.add((eo[name], RDFS.label, Literal(name.replace("_", " "), lang="en")))

    for name, spec in vocab.actions.items():
        a = ns["act"][name]
        g.add((a, RDF.type, ODRL.Action))
        g.add((a, RDFS.label, Literal(name, lang="en")))
        g.add((a, ODRL.includedIn, ODRL[spec["included_in"]]))

    for name, spec in vocab.left_operands.items():
        if not spec.get("odrl"):
            lo = eo[name]
            g.add((lo, RDF.type, ODRL.LeftOperand))
            g.add((lo, RDFS.label, Literal(name, lang="en")))
            g.add((lo, RDFS.comment, Literal("Project extension of the ODRL left operands.", lang="en")))


def _support(g: Graph, eo: Namespace, node, support: Any) -> None:
    if isinstance(support, dict):
        for key, prop in (("chunks", eo.supportChunks), ("documents", eo.supportDocuments),
                          ("approved_chunks", eo.supportApprovedChunks)):
            if isinstance(support.get(key), (int, float)):
                g.add((node, prop, Literal(int(support[key]), datatype=XSD.integer)))


def parse_celex(celex: str) -> dict:
    """Sector, year, act type and a readable label from a CELEX number.
    32018D1859 -> sector 3 (legislation), 2018, Decision, '(EU) 2018/1859'."""
    m = re.match(r"^([0-9CE])(\d{4})([A-Z]{1,2})(\d+)", celex or "")
    if not m:
        return {"act_type": "Unknown", "label": celex or "unknown act"}
    sector, year, letter, number = m.groups()
    types = CELEX_TYPES.get(sector, {})
    act_type = types.get(letter, f"{CELEX_SECTORS.get(sector, 'Sector ' + sector)} ({letter})")
    return {"sector": sector, "sector_label": CELEX_SECTORS.get(sector, sector), "year": year,
            "act_type": act_type, "number": str(int(number)),
            "label": f"{act_type} {year}/{int(number)}"}


def export_rdf(refined: dict, base_iri: str = DEFAULT_BASE_IRI,
               vocabulary_path: str | None = None,
               act_metadata: dict | None = None) -> tuple[Graph, dict]:
    """Return (graph, stats). `refined` is the output of the ontology_refinement pipeline."""
    vocab = load_vocabulary(vocabulary_path)
    ns = namespaces(base_iri)
    eo, ent, act, pol = ns["eo"], ns["ent"], ns["act"], ns["pol"]
    g = Graph()
    for prefix, n in (("eo", eo), ("ent", ent), ("act", act), ("pol", pol), ("ev", ns["ev"]),
                      ("odrl", ODRL), ("owl", OWL), ("skos", SKOS), ("wd", WD), ("dct", DCTERMS),
                      ("prov", PROV), ("oa", OA)):
        g.bind(prefix, n)
    build_tbox(g, ns, vocab)
    stats = {"classes": 0, "accepted_classes": 0, "candidate_classes": 0, "individuals": 0, "rules": 0,
             "relationships": 0, "constraints": 0, "policies": 0, "evidence_links": 0,
             "elements_without_evidence": 0, "skipped": []}

    def evidence(node, obj: dict) -> None:
        n = add_evidence(g, ns, node, obj.get("provenance"))
        stats["evidence_links"] += n
        stats["elements_without_evidence"] += n == 0
    EO = eo

    iri_of: dict[str, URIRef] = {}
    used_locals: set[str] = set()

    def entity_iri(name: str) -> URIRef:
        if name not in iri_of:
            base = local_name(name)
            loc, i = base, 2
            while loc in used_locals:
                loc, i = f"{base}_{i}", i + 1
            used_locals.add(loc)
            iri_of[name] = ent[loc]
        return iri_of[name]

    def describe(node, e: dict, kind: str) -> None:
        g.add((node, RDFS.label, Literal(e["name"].replace("_", " "), lang="en")))
        g.add((node, eo.entityKind, Literal(kind)))
        g.add((node, eo.coreType, Literal(e.get("type") or "")))
        if e.get("description"):
            for piece in str(e["description"]).split(" | "):
                if piece.strip():
                    g.add((node, SKOS.definition, Literal(piece.strip(), lang="en")))
        _support(g, eo, node, e.get("support"))
        if e.get("type_source"):
            g.add((node, eo.typeSource, Literal(str(e["type_source"]))))
        if e.get("kind_confidence"):
            g.add((node, eo.kindConfidence, Literal(str(e["kind_confidence"]))))
        for variant in e.get("merged_from") or []:
            g.add((node, eo.mergedFrom, Literal(variant)))
        evidence(node, e)

    for c in refined.get("classes", []):
        node = entity_iri(c["name"])
        g.add((node, RDF.type, OWL.Class))
        if c.get("is_a") in vocab.upper:
            g.add((node, RDFS.subClassOf, eo[c["is_a"]]))
        describe(node, c, "class")
        # a class is not the same thing as a Wikidata item / authority concept:
        # skos:exactMatch (mapping), not owl:sameAs (identity, used for individuals)
        if c.get("wikidata"):
            g.add((node, SKOS.exactMatch, WD[c["wikidata"]]))
        if c.get("iri"):
            g.add((node, SKOS.exactMatch, URIRef(c["iri"])))
        status = c.get("status") or "accepted"
        g.add((node, eo.status, Literal(status)))
        stats["classes"] += 1
        stats[f"{status}_classes"] += 1

    for i in refined.get("individuals", []):
        node = entity_iri(i["name"])
        g.add((node, RDF.type, OWL.NamedIndividual))
        if i.get("instance_of") in vocab.upper:
            g.add((node, RDF.type, eo[i["instance_of"]]))
        describe(node, i, "individual")
        if i.get("wikidata"):
            g.add((node, OWL.sameAs, WD[i["wikidata"]]))
        if i.get("iri"):
            g.add((node, SKOS.exactMatch, URIRef(i["iri"])))
        if i.get("member_state") is not None:
            g.add((node, eo.memberState, Literal(bool(i["member_state"]))))
        stats["individuals"] += 1

    for e in refined.get("classes", []) + refined.get("individuals", []):
        rel = e.get("related_individual")
        if rel and rel in iri_of:
            g.add((iri_of[e["name"]], eo.relatedIndividual, iri_of[rel]))

    for r in refined.get("relationships", []):
        s, t = r.get("source"), r.get("target")
        if s not in iri_of or t not in iri_of:
            stats["skipped"].append(f"relationship {s} {r.get('relation')} {t}: undefined endpoint")
            continue
        prop = eo[local_name(r["relation"])]
        if r["relation"] not in vocab.relations:
            g.add((prop, RDF.type, OWL.ObjectProperty))
            g.add((prop, RDFS.comment, Literal("Relation outside the controlled vocabulary.", lang="en")))
        g.add((iri_of[s], prop, iri_of[t]))
        stats["relationships"] += 1

    # One odrl:Set per legal act (CELEX): a rule belongs to every act it was
    # extracted from. Rules without provenance go to an explicit fallback set.
    policies: dict[str, URIRef] = {}

    def policy_for(celex: str | None) -> URIRef:
        key = celex or "unattributed"
        if key not in policies:
            # CELEX numbers are already IRI-safe (32018D1859); keep them verbatim
            segment = celex if celex and re.fullmatch(r"[A-Za-z0-9()]+", celex) else local_name(celex or "")
            node = pol[f"act/{segment}" if celex else "unattributed"]
            policies[key] = node
            g.add((node, RDF.type, ODRL.Set))
            g.add((node, ODRL.uid, node))
            if celex:
                info = parse_celex(celex)
                g.add((node, RDFS.label, Literal(f"Rules extracted from {info['label']} (CELEX {celex})", lang="en")))
                g.add((node, EO.celex, Literal(celex)))
                g.add((node, EO.actType, Literal(info["act_type"])))
                if info.get("year"):
                    g.add((node, EO.year, Literal(info["year"], datatype=XSD.gYear)))
                g.add((node, RDFS.seeAlso, URIRef(f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}")))
                g.add((node, RDFS.seeAlso, URIRef(f"http://publications.europa.eu/resource/celex/{celex}")))
                meta = (act_metadata or {}).get(celex) or {}
                if meta.get("title"):
                    g.add((node, DCTERMS.title, Literal(meta["title"], lang="en")))
                if meta.get("date"):
                    iso = parse_date(meta["date"]) or (meta["date"] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(meta["date"])) else None)
                    if iso:
                        g.add((node, DCTERMS.issued, Literal(iso, datatype=XSD.date)))
            else:
                g.add((node, RDFS.label, Literal("Rules without source-act provenance", lang="en")))
            stats["policies"] = len(policies)
        return policies[key]

    for kind in ("permissions", "prohibitions", "duties"):
        for r in (refined.get("rights_model") or {}).get(kind, []):
            node = pol[r.get("id") or _rule_id(kind, r)]
            for celex in (r.get("sources") or [None]):
                g.add((policy_for(celex), RULE_LINK[kind], node))
            g.add((node, RDF.type, RULE_CLASS[kind]))
            action = r.get("action")
            if action in vocab.actions:
                g.add((node, ODRL.action, act[action]))
            elif action:
                a = act[local_name(action)]
                g.add((a, RDF.type, ODRL.Action))
                g.add((a, RDFS.comment, Literal("Action outside the controlled vocabulary.", lang="en")))
                g.add((node, ODRL.action, a))
            if r.get("party") in iri_of:
                g.add((node, ODRL.assignee, iri_of[r["party"]]))
            elif r.get("party"):
                stats["skipped"].append(f"{kind} {action}: party '{r['party']}' undefined")
            if r.get("assigner") in iri_of:
                g.add((node, ODRL.assigner, iri_of[r["assigner"]]))
            if r.get("asset") in iri_of:
                g.add((node, ODRL.target, iri_of[r["asset"]]))
            for key, prop in (("action_object", eo.actionObject), ("action_label", eo.actionLabel),
                              ("provision", eo.provision)):
                if r.get(key):
                    g.add((node, prop, Literal(str(r[key]))))
            # rules merged from several chunks keep every label and provision
            for key, prop in (("action_labels", eo.actionLabel), ("provisions", eo.provision)):
                for v in r.get(key) or []:
                    g.add((node, prop, Literal(str(v))))
            _support(g, eo, node, r.get("support"))
            evidence(node, r)
            for c in r.get("constraints") or []:
                left = c.get("left_operand")
                if not left:
                    continue
                cn = BNode()
                g.add((node, ODRL.constraint, cn))
                g.add((cn, RDF.type, ODRL.Constraint))
                spec = vocab.left_operands.get(left)
                if spec is None:
                    lo = eo[local_name(left)]
                    g.add((lo, RDF.type, ODRL.LeftOperand))
                else:
                    lo = ODRL[left] if spec.get("odrl") else eo[left]
                g.add((cn, ODRL.leftOperand, lo))
                g.add((cn, ODRL.operator, ODRL[c.get("operator") or "eq"]))
                g.add((cn, ODRL.rightOperand, _typed_right_operand(left, c.get("right_operand"))))
                unit = c.get("unit")
                if unit and re.fullmatch(r"[A-Z]{3}", unit):
                    g.add((cn, ODRL.unit, EU_CURRENCY[unit]))
                stats["evidence_links"] += add_evidence(g, ns, cn, c.get("provenance"))
                stats["constraints"] += 1
            stats["rules"] += 1
    stats["triples"] = len(g)
    return g, stats
