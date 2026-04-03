from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import re

import pandas as pd


ROMAN_TO_ARABIC = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
    "vii": "7",
    "viii": "8",
    "ix": "9",
    "x": "10",
}

TOKEN_EQUIVALENTS = {
    "tumour": "neoplasm",
    "tumours": "neoplasm",
    "tumor": "neoplasm",
    "tumors": "neoplasm",
    "cancer": "neoplasm",
    "cancers": "neoplasm",
    "carcinoma": "neoplasm",
    "carcinomas": "neoplasm",
    "neoplasms": "neoplasm",
    "leukaemia": "leukemia",
    "leukaemias": "leukemia",
    "oesophagus": "esophagus",
    "haemorrhage": "hemorrhage",
    "haemorrhagic": "hemorrhagic",
    "oedema": "edema",
    "injuries": "injury",
    "defects": "defect",
    "diseases": "disease",
    "disorders": "disorder",
    "syndromes": "syndrome",
    "infections": "infection",
    "infectious": "infection",
    "metastases": "metastasis",
}

GENERIC_TOKENS = {
    "adult",
    "children",
    "childhood",
    "condition",
    "disease",
    "disorder",
    "human",
    "patients",
    "syndrome",
}

MATCH_PRIORITY = {
    "override": 6,
    "exact_name": 5,
    "exact_synonym": 4,
    "token_name": 3,
    "token_synonym": 2,
    "contains_name": 1,
    "contains_synonym": 0,
}


@dataclass(frozen=True)
class OntologyCandidate:
    doid: str
    name: str
    match_label: str
    match_source: str
    normalized_label: str
    token_key: tuple[str, ...]
    informative_token_count: int
    label_len: int


def ensure_disease_override_csv(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        pd.DataFrame(columns=["disease", "term", "note"]).to_csv(target, index=False)
    return target


def normalize_disease_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace("'", "")
    s = re.sub(r"[^a-z0-9,\s-]", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip(" ,")
    return s


def _normalize_token(token: str) -> str:
    t = str(token or "").strip().lower()
    if not t:
        return ""
    t = ROMAN_TO_ARABIC.get(t, t)
    t = TOKEN_EQUIVALENTS.get(t, t)
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("s") and len(t) > 4 and not t.endswith("ss"):
        t = t[:-1]
    t = TOKEN_EQUIVALENTS.get(t, t)
    return t


def informative_token_key(text: str) -> tuple[str, ...]:
    tokens = []
    for token in normalize_disease_text(text).split():
        norm = _normalize_token(token)
        if norm and norm not in GENERIC_TOKENS:
            tokens.append(norm)
    return tuple(sorted(dict.fromkeys(tokens)))


def disease_text_variants(text: str) -> list[str]:
    raw = str(text or "").strip()
    base = normalize_disease_text(raw)
    variants = {base}

    if "," in raw:
        parts = [normalize_disease_text(part) for part in raw.split(",") if normalize_disease_text(part)]
        if len(parts) >= 2:
            variants.add(" ".join(parts[1:] + parts[:1]))

    for value in list(variants):
        if " of " in value:
            pieces = [piece.strip() for piece in value.split(" of ") if piece.strip()]
            if len(pieces) == 2:
                variants.add(f"{pieces[1]} {pieces[0]}")

        tokens = value.split()
        singularized = " ".join(_normalize_token(tok) for tok in tokens if _normalize_token(tok))
        if singularized:
            variants.add(singularized)

    return [v for v in dict.fromkeys(v.strip() for v in variants if v.strip())]


def _parse_synonyms(row: pd.Series) -> list[str]:
    raw_json = str(row.get("synonyms_json", "") or "").strip()
    if raw_json:
        try:
            vals = json.loads(raw_json)
            return [str(v).strip() for v in vals if str(v).strip()]
        except Exception:
            pass
    raw = str(row.get("synonyms", "") or "")
    return [part.strip() for part in raw.split(" || ") if part.strip()]


def _build_lookup(do_terms_df: pd.DataFrame) -> tuple[dict[str, list[OntologyCandidate]], dict[tuple[str, ...], list[OntologyCandidate]], list[OntologyCandidate]]:
    exact_lookup: dict[str, list[OntologyCandidate]] = {}
    token_lookup: dict[tuple[str, ...], list[OntologyCandidate]] = {}
    universe: list[OntologyCandidate] = []

    for row in do_terms_df.fillna("").to_dict("records"):
        doid = str(row.get("doid", "")).strip()
        name = str(row.get("name", "")).strip()
        if not doid or not name:
            continue

        for source, label in [("name", name), *[("synonym", s) for s in _parse_synonyms(pd.Series(row))]]:
            for variant in disease_text_variants(label):
                token_key = informative_token_key(variant)
                candidate = OntologyCandidate(
                    doid=doid,
                    name=name,
                    match_label=label,
                    match_source=source,
                    normalized_label=variant,
                    token_key=token_key,
                    informative_token_count=len(token_key),
                    label_len=len(variant),
                )
                exact_lookup.setdefault(variant, []).append(candidate)
                if token_key:
                    token_lookup.setdefault(token_key, []).append(candidate)
                universe.append(candidate)

    return exact_lookup, token_lookup, universe


def _load_overrides(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    override_path = ensure_disease_override_csv(path)
    df = pd.read_csv(override_path, dtype=str).fillna("")
    if df.empty:
        return {}
    cols = {c.lower(): c for c in df.columns}
    if "disease" not in cols or "term" not in cols:
        raise ValueError(f"{override_path} must have columns disease,term")

    overrides = {}
    for _, row in df.iterrows():
        disease = normalize_disease_text(row[cols["disease"]])
        term = str(row[cols["term"]]).strip()
        if disease and term:
            overrides[disease] = term
    return overrides


def _dedupe_candidates(candidates: Iterable[OntologyCandidate]) -> list[OntologyCandidate]:
    deduped: dict[tuple[str, str, str], OntologyCandidate] = {}
    for candidate in candidates:
        key = (candidate.doid, candidate.normalized_label, candidate.match_source)
        deduped.setdefault(key, candidate)
    return list(deduped.values())


def _rank_candidate(candidate: OntologyCandidate, *, match_type: str, query_tokens: tuple[str, ...]) -> tuple[int, int, int, int, int]:
    overlap = len(set(query_tokens) & set(candidate.token_key))
    return (
        MATCH_PRIORITY[match_type],
        overlap,
        candidate.informative_token_count,
        candidate.label_len,
        1 if candidate.match_source == "name" else 0,
    )


def _resolve_disease(
    disease: str,
    *,
    exact_lookup: dict[str, list[OntologyCandidate]],
    token_lookup: dict[tuple[str, ...], list[OntologyCandidate]],
    universe: list[OntologyCandidate],
    overrides: dict[str, str],
) -> dict[str, str]:
    variants = disease_text_variants(disease)
    norm = normalize_disease_text(disease)

    if norm in overrides:
        return {
            "disease": disease,
            "term": overrides[norm],
            "selected_name": "",
            "match_type": "override",
            "candidate_count": 1,
            "alternative_terms": "",
            "alternative_names": "",
            "query_variants": " || ".join(variants),
        }

    collected: list[tuple[str, OntologyCandidate]] = []

    for variant in variants:
        for candidate in exact_lookup.get(variant, []):
            match_type = "exact_name" if candidate.match_source == "name" else "exact_synonym"
            collected.append((match_type, candidate))

    if not collected:
        for variant in variants:
            token_key = informative_token_key(variant)
            if token_key:
                for candidate in token_lookup.get(token_key, []):
                    match_type = "token_name" if candidate.match_source == "name" else "token_synonym"
                    collected.append((match_type, candidate))

    if not collected:
        query_token_sets = [informative_token_key(v) for v in variants]
        query_token_sets = [q for q in query_token_sets if q]
        for query_tokens in query_token_sets:
            query_set = set(query_tokens)
            if len(query_set) < 2:
                continue
            for candidate in universe:
                cand_set = set(candidate.token_key)
                if len(cand_set) < 2:
                    continue
                overlap = len(query_set & cand_set)
                if overlap < 2:
                    continue
                query_extra = len(query_set - cand_set)
                cand_extra = len(cand_set - query_set)
                if (query_set.issubset(cand_set) and cand_extra <= 1) or (
                    cand_set.issubset(query_set) and query_extra <= 1
                ):
                    match_type = "contains_name" if candidate.match_source == "name" else "contains_synonym"
                    collected.append((match_type, candidate))

    if not collected:
        return {
            "disease": disease,
            "term": "",
            "selected_name": "",
            "match_type": "unresolved",
            "candidate_count": 0,
            "alternative_terms": "",
            "alternative_names": "",
            "query_variants": " || ".join(variants),
        }

    # Deduplicate at DOID level while keeping the best matching evidence per term.
    by_doid: dict[str, tuple[str, OntologyCandidate]] = {}
    for match_type, candidate in collected:
        query_tokens = informative_token_key(disease)
        best = by_doid.get(candidate.doid)
        if best is None or _rank_candidate(candidate, match_type=match_type, query_tokens=query_tokens) > _rank_candidate(
            best[1],
            match_type=best[0],
            query_tokens=query_tokens,
        ):
            by_doid[candidate.doid] = (match_type, candidate)

    ranked = sorted(
        by_doid.values(),
        key=lambda item: _rank_candidate(item[1], match_type=item[0], query_tokens=informative_token_key(disease)),
        reverse=True,
    )

    selected_type, selected = ranked[0]
    alternatives = ranked[1:]
    return {
        "disease": disease,
        "term": selected.doid,
        "selected_name": selected.name,
        "match_type": selected_type,
        "candidate_count": len(ranked),
        "alternative_terms": " || ".join(item[1].doid for item in alternatives),
        "alternative_names": " || ".join(item[1].name for item in alternatives),
        "query_variants": " || ".join(variants),
    }


def build_disease_term_mapping(
    diseases: Iterable[str],
    do_terms_df: pd.DataFrame,
    *,
    overrides_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_lookup, token_lookup, universe = _build_lookup(do_terms_df)
    overrides = _load_overrides(overrides_path)

    review_rows = [
        _resolve_disease(
            str(disease or "").strip(),
            exact_lookup=exact_lookup,
            token_lookup=token_lookup,
            universe=universe,
            overrides=overrides,
        )
        for disease in diseases
        if str(disease or "").strip()
    ]

    review_df = pd.DataFrame(review_rows)
    map_df = review_df[["disease", "term"]].copy()
    return map_df, review_df
