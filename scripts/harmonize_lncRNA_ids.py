from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_V1_MATRIX = PROJECT_ROOT / "Data/output_data/sequences_for_oop.csv"
DEFAULT_V2_MATRIX = PROJECT_ROOT / "Data/output_data/website_full_matrix.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "Data/output_data/canonical_lncRNA_ids"

HGNC_FETCH_SYMBOL = "https://rest.genenames.org/fetch/symbol/{symbol}"
HGNC_SEARCH_ALIAS = "https://rest.genenames.org/search/alias_symbol/{symbol}"
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ELINK = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
ENSEMBL_XREF_ID = "https://rest.ensembl.org/xrefs/id/{ensembl_id}"

RETRY_STATUSES = [429, 500, 502, 503, 504]
MAP_COLUMNS = ["original_id", "hgnc_symbol", "ensembl_gene_id", "source", "status"]

ACCESSION_RE = re.compile(
    r"^(?:NR_|NM_|XR_|XM_|AK|AB|BC|AF|AJ|AL|AC|BX)[A-Z0-9_]+(?:\.\d+)?$",
    re.I,
)
ENSEMBL_GENE_RE = re.compile(r"^ENSG\d+(?:\.\d+)?$", re.I)
VERSION_SUFFIX_RE = re.compile(r"^(.+?)\.\d+$")
UNICODE_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


def set_max_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_requests():
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError:
        sys.exit("Missing dependency: pip install requests")
    return requests, HTTPAdapter, Retry


def make_session(delay: float):
    requests, HTTPAdapter, Retry = load_requests()
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=RETRY_STATUSES,
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "lncrna-canonical-id-harmonizer/1.0 (research)",
            "Accept": "application/json",
        }
    )
    session._delay = delay
    return session


def append_error_log(error_log: Path, *, url: str, message: str, status_code: int | None = None) -> None:
    error_log.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    code_text = f" status={status_code}" if status_code is not None else ""
    with error_log.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] {url}{code_text} {message}\n")


def request_json(session, url: str, *, error_log: Path, timeout: int = 20, **kwargs) -> dict[str, Any] | None:
    time.sleep(float(getattr(session, "_delay", 0.25)))
    try:
        response = session.get(url, timeout=timeout, **kwargs)
    except Exception as exc:
        append_error_log(error_log, url=url, message=f"exception={exc}")
        return None
    if response.status_code >= 400:
        append_error_log(
            error_log,
            url=response.url,
            status_code=response.status_code,
            message=response.text[:400].replace("\n", " "),
        )
        return None
    try:
        return response.json()
    except Exception as exc:
        append_error_log(error_log, url=response.url, message=f"json_error={exc}")
        return None


def load_cache(path: Path, resume: bool) -> dict[str, dict[str, str]]:
    if not resume or not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def save_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)


def normalize_id(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().translate(UNICODE_DASH_TRANSLATION))


def strip_version(value: str) -> str:
    match = VERSION_SUFFIX_RE.match(value)
    return match.group(1) if match else value


def first_scalar(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value or "").strip()


def empty_row(original_id: str, source: str, status: str = "unresolved") -> dict[str, str]:
    return {
        "original_id": original_id,
        "hgnc_symbol": "",
        "ensembl_gene_id": "",
        "source": source,
        "status": status,
    }


def row_from_hgnc_doc(original_id: str, doc: dict[str, Any], source: str) -> dict[str, str]:
    symbol = first_scalar(doc.get("symbol"))
    ensembl = first_scalar(doc.get("ensembl_gene_id"))
    if not symbol:
        return empty_row(original_id, f"{source}:missing_symbol")
    return {
        "original_id": original_id,
        "hgnc_symbol": symbol,
        "ensembl_gene_id": ensembl,
        "source": source,
        "status": "resolved",
    }


def hgnc_docs(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    response = payload.get("response")
    if not isinstance(response, dict):
        return []
    docs = response.get("docs")
    return docs if isinstance(docs, list) else []


def resolve_hgnc_symbol(
    original_id: str,
    symbol: str,
    *,
    session,
    cache: dict[str, dict[str, str]],
    error_log: Path,
) -> dict[str, str]:
    candidate = normalize_id(symbol)
    if not candidate:
        return empty_row(original_id, "hgnc_symbol:empty")

    cache_key = f"hgnc:{candidate}"
    cached = cache.get(cache_key)
    if cached is not None:
        out = dict(cached)
        out["original_id"] = original_id
        return out

    direct_url = HGNC_FETCH_SYMBOL.format(symbol=quote(candidate, safe=""))
    docs = hgnc_docs(request_json(session, direct_url, error_log=error_log))
    if docs:
        row = row_from_hgnc_doc(original_id, docs[0], "hgnc_fetch_symbol")
        cache[cache_key] = {k: row[k] for k in MAP_COLUMNS if k != "original_id"}
        return row

    alias_url = HGNC_SEARCH_ALIAS.format(symbol=quote(candidate, safe=""))
    docs = hgnc_docs(request_json(session, alias_url, error_log=error_log))
    if docs:
        row = row_from_hgnc_doc(original_id, docs[0], "hgnc_search_alias_symbol")
        cache[cache_key] = {k: row[k] for k in MAP_COLUMNS if k != "original_id"}
        return row

    row = empty_row(original_id, "hgnc_symbol_not_found")
    cache[cache_key] = {k: row[k] for k in MAP_COLUMNS if k != "original_id"}
    return row


def esearch_nucleotide(accession: str, *, session, error_log: Path) -> list[str]:
    payload = request_json(
        session,
        NCBI_ESEARCH,
        error_log=error_log,
        params={
            "db": "nuccore",
            "term": f"{accession}[Accession]",
            "retmode": "json",
        },
    )
    result = (payload or {}).get("esearchresult", {})
    ids = result.get("idlist", []) if isinstance(result, dict) else []
    return [str(x) for x in ids if str(x).strip()]


def elink_nuccore_to_gene(nuccore_ids: list[str], *, session, error_log: Path) -> list[str]:
    gene_ids: list[str] = []
    for nuccore_id in nuccore_ids:
        payload = request_json(
            session,
            NCBI_ELINK,
            error_log=error_log,
            params={
                "dbfrom": "nuccore",
                "db": "gene",
                "id": nuccore_id,
                "retmode": "json",
            },
        )
        for linkset in (payload or {}).get("linksets", []):
            for linkdb in linkset.get("linksetdbs", []) or []:
                if str(linkdb.get("dbto", "")).lower() != "gene":
                    continue
                for link in linkdb.get("links", []) or []:
                    if isinstance(link, dict):
                        link_id = str(link.get("id") or "").strip()
                    else:
                        link_id = str(link or "").strip()
                    if link_id:
                        gene_ids.append(link_id)
    return sorted(set(gene_ids))


def gene_summary_symbol(gene_ids: list[str], *, session, error_log: Path) -> str:
    if not gene_ids:
        return ""
    payload = request_json(
        session,
        NCBI_ESUMMARY,
        error_log=error_log,
        params={"db": "gene", "id": ",".join(gene_ids), "retmode": "json"},
    )
    result = (payload or {}).get("result", {})
    if not isinstance(result, dict):
        return ""

    for uid in result.get("uids", []):
        doc = result.get(str(uid), {})
        organism = doc.get("organism", {}) if isinstance(doc, dict) else {}
        sci_name = str(organism.get("scientificname", "")).lower() if isinstance(organism, dict) else ""
        if sci_name and sci_name != "homo sapiens":
            continue
        for key in ("nomenclaturesymbol", "name"):
            value = str(doc.get(key, "")).strip()
            if value:
                return value
    return ""


def resolve_accession(
    original_id: str,
    *,
    session,
    cache: dict[str, dict[str, str]],
    error_log: Path,
) -> dict[str, str]:
    accession = normalize_id(original_id)
    variants = [accession]
    stripped = strip_version(accession)
    if stripped != accession:
        variants.append(stripped)

    cache_key = f"accession:{accession}"
    cached = cache.get(cache_key)
    if cached is not None:
        out = dict(cached)
        out["original_id"] = original_id
        return out

    for variant in variants:
        nuccore_ids = esearch_nucleotide(variant, session=session, error_log=error_log)
        gene_ids = elink_nuccore_to_gene(nuccore_ids, session=session, error_log=error_log)
        symbol = gene_summary_symbol(gene_ids, session=session, error_log=error_log)
        if not symbol:
            continue
        row = resolve_hgnc_symbol(
            original_id,
            symbol,
            session=session,
            cache=cache,
            error_log=error_log,
        )
        if row["status"] == "resolved":
            row["source"] = f"ncbi_nuccore_elink_gene+{row['source']}"
        else:
            row = {
                "original_id": original_id,
                "hgnc_symbol": symbol,
                "ensembl_gene_id": "",
                "source": "ncbi_nuccore_elink_gene_no_hgnc_record",
                "status": "resolved",
            }
        cache[cache_key] = {k: row[k] for k in MAP_COLUMNS if k != "original_id"}
        return row

    row = empty_row(original_id, "ncbi_nuccore_to_gene_not_found")
    cache[cache_key] = {k: row[k] for k in MAP_COLUMNS if k != "original_id"}
    return row


def resolve_ensembl_gene(
    original_id: str,
    *,
    session,
    cache: dict[str, dict[str, str]],
    error_log: Path,
) -> dict[str, str]:
    ensembl_id = strip_version(normalize_id(original_id))
    cache_key = f"ensembl_gene:{ensembl_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        out = dict(cached)
        out["original_id"] = original_id
        return out

    payload = request_json(
        session,
        ENSEMBL_XREF_ID.format(ensembl_id=quote(ensembl_id, safe="")),
        error_log=error_log,
        headers={"Content-Type": "application/json"},
    )
    docs = payload if isinstance(payload, list) else []
    for doc in docs:
        dbname = str(doc.get("dbname", "")).lower()
        display_id = str(doc.get("display_id", "")).strip()
        if dbname == "hgnc" and display_id:
            row = resolve_hgnc_symbol(
                original_id,
                display_id,
                session=session,
                cache=cache,
                error_log=error_log,
            )
            if row["status"] == "resolved" and not row["ensembl_gene_id"]:
                row["ensembl_gene_id"] = ensembl_id
            row["source"] = f"ensembl_xrefs_id+{row['source']}"
            cache[cache_key] = {k: row[k] for k in MAP_COLUMNS if k != "original_id"}
            return row

    row = empty_row(original_id, "ensembl_gene_xref_hgnc_not_found")
    cache[cache_key] = {k: row[k] for k in MAP_COLUMNS if k != "original_id"}
    return row


def resolve_one(
    original_id: str,
    *,
    version: str,
    session,
    cache: dict[str, dict[str, str]],
    error_log: Path,
) -> dict[str, str]:
    value = normalize_id(original_id)
    if not value:
        return empty_row(original_id, "empty_id")

    if ACCESSION_RE.match(value):
        return resolve_accession(original_id, session=session, cache=cache, error_log=error_log)
    if ENSEMBL_GENE_RE.match(value):
        return resolve_ensembl_gene(original_id, session=session, cache=cache, error_log=error_log)

    # V2 is mostly symbol space, but V1 can contain a few non-accession IDs too.
    row = resolve_hgnc_symbol(original_id, value, session=session, cache=cache, error_log=error_log)
    if row["status"] == "resolved":
        row["source"] = f"{version}_symbol+{row['source']}"
    return row


def load_ids(matrix_path: Path) -> list[str]:
    df = pd.read_csv(matrix_path, dtype=str)
    if "ID" not in df.columns:
        raise ValueError(f"{matrix_path} is missing required ID column.")
    return df["ID"].astype(str).str.strip().loc[lambda s: s.ne("")].drop_duplicates().tolist()


def resolve_ids(
    ids: list[str],
    *,
    version: str,
    session,
    cache: dict[str, dict[str, str]],
    error_log: Path,
    max_ids: int | None,
    checkpoint_every: int,
    cache_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    limit = len(ids) if max_ids is None else min(len(ids), max_ids)
    for idx, original_id in enumerate(ids[:limit], start=1):
        rows.append(
            resolve_one(
                original_id,
                version=version,
                session=session,
                cache=cache,
                error_log=error_log,
            )
        )
        if checkpoint_every > 0 and idx % checkpoint_every == 0:
            save_cache(cache_path, cache)
            print(f"[{version}] processed {idx}/{limit}")
    save_cache(cache_path, cache)

    if limit < len(ids):
        for original_id in ids[limit:]:
            rows.append(empty_row(original_id, "not_processed_due_to_max_ids", status="unresolved"))
    return pd.DataFrame(rows, columns=MAP_COLUMNS)


def canonical_key(row: pd.Series, key: str) -> str:
    value = str(row.get(key, "") or "").strip()
    status = str(row.get("status", "") or "").strip()
    return value if status == "resolved" else ""


def collapse_matrix_by_canonical(
    matrix_path: Path,
    map_df: pd.DataFrame,
    common_keys: set[str],
    *,
    key_column: str,
    common_diseases: list[str],
) -> pd.DataFrame:
    matrix = pd.read_csv(matrix_path, dtype=str).fillna("")
    mapping = map_df.copy()
    mapping["_canonical_key"] = mapping.apply(lambda row: canonical_key(row, key_column), axis=1)
    mapping = mapping[mapping["_canonical_key"].isin(common_keys)]
    joined = matrix.merge(mapping, left_on="ID", right_on="original_id", how="inner")
    if joined.empty:
        return pd.DataFrame(
            columns=["canonical_id", "hgnc_symbol", "ensembl_gene_id", "original_ids", "seqs", *common_diseases]
        )

    for disease in common_diseases:
        joined[disease] = pd.to_numeric(joined[disease], errors="coerce").fillna(0).astype(int)

    rows: list[dict[str, Any]] = []
    for canonical_id, group in joined.groupby("_canonical_key", sort=True):
        first = group.iloc[0]
        row: dict[str, Any] = {
            "canonical_id": canonical_id,
            "hgnc_symbol": str(first.get("hgnc_symbol", "") or ""),
            "ensembl_gene_id": str(first.get("ensembl_gene_id", "") or ""),
            "original_ids": "|".join(sorted(set(group["ID"].astype(str)))),
            "seqs": next((str(x) for x in group.get("seqs", pd.Series(dtype=str)).astype(str) if x.strip()), ""),
        }
        for disease in common_diseases:
            row[disease] = int(group[disease].max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_intersection_outputs(
    *,
    v1_matrix: Path,
    v2_matrix: Path,
    v1_map: pd.DataFrame,
    v2_map: pd.DataFrame,
    outdir: Path,
) -> dict[str, int]:
    v1_df = pd.read_csv(v1_matrix, dtype=str).fillna("")
    v2_df = pd.read_csv(v2_matrix, dtype=str).fillna("")
    common_diseases = sorted((set(v1_df.columns) & set(v2_df.columns)) - {"ID", "seqs", "Unnamed: 0"})

    v1_hgnc = {canonical_key(row, "hgnc_symbol") for _, row in v1_map.iterrows()}
    v2_hgnc = {canonical_key(row, "hgnc_symbol") for _, row in v2_map.iterrows()}
    v1_ens = {canonical_key(row, "ensembl_gene_id") for _, row in v1_map.iterrows()}
    v2_ens = {canonical_key(row, "ensembl_gene_id") for _, row in v2_map.iterrows()}
    hgnc_common = {x for x in (v1_hgnc & v2_hgnc) if x}
    ens_common = {x for x in (v1_ens & v2_ens) if x}

    hgnc_rows = []
    for symbol in sorted(hgnc_common):
        v1_hits = v1_map[(v1_map["status"] == "resolved") & (v1_map["hgnc_symbol"] == symbol)]
        v2_hits = v2_map[(v2_map["status"] == "resolved") & (v2_map["hgnc_symbol"] == symbol)]
        ensembl_values = sorted(
            set(v1_hits["ensembl_gene_id"].astype(str)) | set(v2_hits["ensembl_gene_id"].astype(str))
        )
        hgnc_rows.append(
            {
                "hgnc_symbol": symbol,
                "ensembl_gene_id": "|".join(x for x in ensembl_values if x),
                "v1_original_ids": "|".join(sorted(v1_hits["original_id"].astype(str))),
                "v2_original_ids": "|".join(sorted(v2_hits["original_id"].astype(str))),
            }
        )
    pd.DataFrame(hgnc_rows).to_csv(outdir / "canonical_intersection_by_hgnc.csv", index=False)

    ens_rows = []
    for ensembl_id in sorted(ens_common):
        v1_hits = v1_map[(v1_map["status"] == "resolved") & (v1_map["ensembl_gene_id"] == ensembl_id)]
        v2_hits = v2_map[(v2_map["status"] == "resolved") & (v2_map["ensembl_gene_id"] == ensembl_id)]
        symbol_values = sorted(set(v1_hits["hgnc_symbol"].astype(str)) | set(v2_hits["hgnc_symbol"].astype(str)))
        ens_rows.append(
            {
                "ensembl_gene_id": ensembl_id,
                "hgnc_symbol": "|".join(x for x in symbol_values if x),
                "v1_original_ids": "|".join(sorted(v1_hits["original_id"].astype(str))),
                "v2_original_ids": "|".join(sorted(v2_hits["original_id"].astype(str))),
            }
        )
    pd.DataFrame(ens_rows).to_csv(outdir / "canonical_intersection_by_ensembl.csv", index=False)

    v1_hgnc_matrix = collapse_matrix_by_canonical(
        v1_matrix,
        v1_map,
        hgnc_common,
        key_column="hgnc_symbol",
        common_diseases=common_diseases,
    )
    v2_hgnc_matrix = collapse_matrix_by_canonical(
        v2_matrix,
        v2_map,
        hgnc_common,
        key_column="hgnc_symbol",
        common_diseases=common_diseases,
    )
    v1_hgnc_matrix.to_csv(outdir / "v1_hgnc_intersection_common_diseases.csv", index=False)
    v2_hgnc_matrix.to_csv(outdir / "v2_hgnc_intersection_common_diseases.csv", index=False)

    return {
        "common_diseases": len(common_diseases),
        "hgnc_intersection": len(hgnc_common),
        "ensembl_intersection": len(ens_common),
    }


def write_summary(
    *,
    outdir: Path,
    v1_ids: list[str],
    v2_ids: list[str],
    v1_map: pd.DataFrame,
    v2_map: pd.DataFrame,
    intersection_counts: dict[str, int],
) -> None:
    raw_overlap = len(set(v1_ids) & set(v2_ids))
    v1_resolved = int((v1_map["status"] == "resolved").sum())
    v2_resolved = int((v2_map["status"] == "resolved").sum())
    rows = [
        {
            "strategy": "raw_original_id",
            "v1_lncRNA_count": len(set(v1_ids)),
            "v2_lncRNA_count": len(set(v2_ids)),
            "v1_resolved": "",
            "v2_resolved": "",
            "overlap_count": raw_overlap,
        },
        {
            "strategy": "canonical_hgnc_symbol",
            "v1_lncRNA_count": len(set(v1_ids)),
            "v2_lncRNA_count": len(set(v2_ids)),
            "v1_resolved": v1_resolved,
            "v2_resolved": v2_resolved,
            "overlap_count": intersection_counts["hgnc_intersection"],
        },
        {
            "strategy": "canonical_ensembl_gene_id",
            "v1_lncRNA_count": len(set(v1_ids)),
            "v2_lncRNA_count": len(set(v2_ids)),
            "v1_resolved": int(v1_map["ensembl_gene_id"].astype(str).str.strip().ne("").sum()),
            "v2_resolved": int(v2_map["ensembl_gene_id"].astype(str).str.strip().ne("").sum()),
            "overlap_count": intersection_counts["ensembl_intersection"],
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "canonical_overlap_summary.csv", index=False)

    report = [
        "# Canonical lncRNA ID Harmonization",
        "",
        "This report is generated without modifying the original V1/V2 matrices or existing feature outputs.",
        "",
        f"- V1 input: `{DEFAULT_V1_MATRIX.relative_to(PROJECT_ROOT)}`",
        f"- V2 input: `{DEFAULT_V2_MATRIX.relative_to(PROJECT_ROOT)}`",
        f"- V1 resolved: {v1_resolved}/{len(set(v1_ids))}",
        f"- V2 resolved: {v2_resolved}/{len(set(v2_ids))}",
        f"- Raw original-ID overlap: {raw_overlap}",
        f"- Canonical HGNC-symbol overlap: {intersection_counts['hgnc_intersection']}",
        f"- Canonical Ensembl-gene overlap: {intersection_counts['ensembl_intersection']}",
        f"- Common disease labels in intersection matrices: {intersection_counts['common_diseases']}",
        "",
        "Generated files:",
        "",
        "- `v1_canonical_map.csv`",
        "- `v2_canonical_map.csv`",
        "- `canonical_intersection_by_hgnc.csv`",
        "- `canonical_intersection_by_ensembl.csv`",
        "- `v1_hgnc_intersection_common_diseases.csv`",
        "- `v2_hgnc_intersection_common_diseases.csv`",
        "- `canonical_overlap_summary.csv`",
    ]
    (outdir / "canonical_harmonization_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve V1/V2 lncRNA identifiers to canonical HGNC symbols and Ensembl gene IDs "
            "without rewriting existing matrices."
        )
    )
    parser.add_argument("--v1", default=str(DEFAULT_V1_MATRIX), help="V1 matrix with ID column")
    parser.add_argument("--v2", default=str(DEFAULT_V2_MATRIX), help="V2 matrix with ID column")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory for canonical maps")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between HTTP requests")
    parser.add_argument("--resume", action="store_true", help="Reuse the canonical ID cache if present")
    parser.add_argument("--max-ids", type=int, default=None, help="Process only the first N IDs per dataset")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save cache every N IDs")
    return parser.parse_args()


def main() -> None:
    set_max_csv_field_size_limit()
    args = parse_args()
    v1_matrix = Path(args.v1).resolve()
    v2_matrix = Path(args.v2).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cache_path = outdir / "canonical_harmonization_cache.json"
    error_log = outdir / "canonical_harmonization_errors.log"
    cache = load_cache(cache_path, args.resume)
    session = make_session(args.delay)

    v1_ids = load_ids(v1_matrix)
    v2_ids = load_ids(v2_matrix)
    print(f"Loaded V1 IDs: {len(v1_ids)} from {v1_matrix}")
    print(f"Loaded V2 IDs: {len(v2_ids)} from {v2_matrix}")
    print(f"Raw original-ID overlap: {len(set(v1_ids) & set(v2_ids))}")

    v1_map = resolve_ids(
        v1_ids,
        version="v1",
        session=session,
        cache=cache,
        error_log=error_log,
        max_ids=args.max_ids,
        checkpoint_every=args.checkpoint_every,
        cache_path=cache_path,
    )
    v2_map = resolve_ids(
        v2_ids,
        version="v2",
        session=session,
        cache=cache,
        error_log=error_log,
        max_ids=args.max_ids,
        checkpoint_every=args.checkpoint_every,
        cache_path=cache_path,
    )

    v1_map.to_csv(outdir / "v1_canonical_map.csv", index=False)
    v2_map.to_csv(outdir / "v2_canonical_map.csv", index=False)
    counts = write_intersection_outputs(
        v1_matrix=v1_matrix,
        v2_matrix=v2_matrix,
        v1_map=v1_map,
        v2_map=v2_map,
        outdir=outdir,
    )
    write_summary(
        outdir=outdir,
        v1_ids=v1_ids,
        v2_ids=v2_ids,
        v1_map=v1_map,
        v2_map=v2_map,
        intersection_counts=counts,
    )
    print(f"\nSaved canonical harmonization outputs to: {outdir}")


if __name__ == "__main__":
    main()
