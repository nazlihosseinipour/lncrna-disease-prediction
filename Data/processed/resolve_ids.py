from __future__ import annotations

"""
Recover real nucleotide sequences for unresolved lncRNA identifiers.

This script reads unresolved IDs, routes them to the appropriate data source,
only counts a hit as resolved when an actual nucleotide sequence is recovered,
and writes successful recoveries into Data/raw/sequence_id_overrides.csv so the
existing override/refetch pipeline can consume them directly.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.parse import quote

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def load_requests():
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError:
        sys.exit("Missing dependency: pip install requests")
    return requests, HTTPAdapter, Retry


def find_project_root(marker_rel: Path = Path("mainfolder/utils/sequence_overrides.py")) -> Path:
    script_dir = Path(__file__).resolve().parent
    for base in (Path.cwd().resolve(), *Path.cwd().resolve().parents, script_dir, *script_dir.parents):
        if (base / marker_rel).exists():
            return base
    raise FileNotFoundError(f"Could not locate project root containing {marker_rel}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mainfolder.utils.sequence_overrides import OVERRIDE_COLUMNS, ensure_sequence_override_csv


RNACENTRAL_EXTERNAL_ID = "https://rnacentral.org/api/v1/rna/"
ENSEMBL_XREF_SYMBOL = "https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{value}"
ENSEMBL_LOOKUP_ID = "https://rest.ensembl.org/lookup/id/{value}"
ENSEMBL_SEQUENCE_ID = "https://rest.ensembl.org/sequence/id/{value}"
ENSEMBL_ARCHIVE_POST = "https://rest.ensembl.org/archive/id"
HGNC_FETCH_SYMBOL = "https://rest.genenames.org/fetch/symbol/{value}"
HGNC_SEARCH_ALIAS_SYMBOL = "https://rest.genenames.org/search/alias_symbol/{value}"
UCSC_TRACK = "https://api.genome.ucsc.edu/getData/track"
NCBI_EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nucleotide&id={value}&rettype=fasta&retmode=text"
)

RETRY_STATUSES = [429, 500, 502, 503, 504]
VALID_NUCLEOTIDE_CHARS = set("ACGTUNRYSWKMBDHVX")
RNACENTRAL_ID_RE = re.compile(r"^NON(?:HSAT|MMUT)\d+(?:\.\d+)?$", re.I)
LNCIPEDIA_ID_RE = re.compile(r"^(?:LNC[-_].+|lnc[-_].+|Lnc[-_].+|lnr.+)$")
ENSEMBL_CLONE_RE = re.compile(r"^(?:AC|AL)\d{5,6}(?:\.\d+)?(?:[-:]\d+)?$", re.I)
ENSEMBL_ARCHIVE_RE = re.compile(r"^(?:ENSG|ENST)\d+(?:\.\d+)?$", re.I)
NCBI_ACCESSION_RE = re.compile(r"^(?:AJ|AP|AF)\d+(?:\.\d+)?(?:[-:]\d+)?$", re.I)
UCSC_RE = re.compile(r"^uc\d{3}[a-z]{3}(?:\.\d+)?$", re.I)
REFSEQ_RNA_RE = re.compile(r"\b(?:NR|NM|XR|XM)_\d+(?:\.\d+)?\b", re.I)
ENSEMBL_STABLE_RE = re.compile(r"\bENS(?:G|T)\d+(?:\.\d+)?\b", re.I)
VERSIONED_ENS_RE = re.compile(r"^(ENS(?:G|T)\d+)\.\d+$", re.I)
TRAILING_PLUS_MINUS_RE = re.compile(r"^(.+?)[+-]+$")
TRAILING_TRANSCRIPT_SUFFIX_RE = re.compile(r"^(.+?)(?:-\d{3,}|:\d+)$")
TRAILING_VERSION_RE = re.compile(r"^(.+?)\.\d+$")
TRAILING_NUMERIC_SUFFIX_RE = re.compile(r"^(.+?)-\d+$")
TRAILING_LOWERCASE_RE = re.compile(r"^(.+?)[a-z]$")


@dataclass(frozen=True)
class SequenceHit:
    query_id: str
    matched_query: str
    route: str
    resolved_id: str
    resolved_id_type: str
    sequence: str
    source: str
    confidence: str

    def override_row(self) -> dict[str, str]:
        notes = json.dumps(
            {
                "confidence": self.confidence,
                "resolved_id_type": self.resolved_id_type,
                "matched_query": self.matched_query,
                "route": self.route,
            },
            sort_keys=True,
        )
        return {
            "query_id": self.query_id,
            "resolved_id": self.resolved_id,
            "sequence": self.sequence,
            "source": self.source,
            "notes": notes,
        }


@dataclass(frozen=True)
class ResidualHit:
    query_id: str
    id_type: str
    route: str
    reason: str

    def residual_row(self) -> dict[str, str]:
        return {
            "id": self.query_id,
            "type": self.id_type,
            "status": "unresolved",
            "reason": f"{self.route}; {self.reason}",
            "resolved_id": "",
        }


def make_session(delay: float) -> requests.Session:
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
    session.headers.update({"User-Agent": "lncrna-sequence-recovery/1.0 (research)"})
    session._delay = delay
    return session


def append_error_log(error_log: Path, *, url: str, message: str, status_code: int | None = None) -> None:
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("a", encoding="utf-8") as fh:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        code_text = f" status={status_code}" if status_code is not None else ""
        fh.write(f"[{timestamp}] {url}{code_text} {message}\n")


def request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    error_log: Path,
    timeout: int = 20,
    **kwargs,
) -> requests.Response | None:
    time.sleep(float(getattr(session, "_delay", 0.25)))
    try:
        response = session.request(method, url, timeout=timeout, **kwargs)
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
    return response


def parse_fasta_text(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip() and not line.startswith(">")]
    return "".join(lines).upper()


def normalize_sequence_value(value: str) -> str:
    seq = parse_fasta_text(str(value or ""))
    if not seq:
        return ""
    if set(seq).issubset(VALID_NUCLEOTIDE_CHARS):
        return seq
    return ""


def load_fasta_index(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    index: dict[str, str] = {}
    current_id = ""
    chunks: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id and chunks:
                    seq = normalize_sequence_value("".join(chunks))
                    if seq:
                        index[current_id] = seq
                current_id = line[1:].split()[0]
                chunks = []
                continue
            chunks.append(line)
    if current_id and chunks:
        seq = normalize_sequence_value("".join(chunks))
        if seq:
            index[current_id] = seq
    return index


def build_variants(raw_id: str) -> list[str]:
    value = str(raw_id or "").strip()
    variants = [value]
    current = value
    for pattern in (
        TRAILING_PLUS_MINUS_RE,
        TRAILING_TRANSCRIPT_SUFFIX_RE,
        TRAILING_VERSION_RE,
        TRAILING_NUMERIC_SUFFIX_RE,
        TRAILING_LOWERCASE_RE,
    ):
        match = pattern.match(current)
        if not match:
            continue
        candidate = str(match.group(1) or "").strip()
        if candidate and candidate not in variants:
            variants.append(candidate)
            current = candidate
    return variants


def classify(raw_id: str) -> str:
    value = str(raw_id or "").strip()
    if RNACENTRAL_ID_RE.match(value):
        return "rnacentral_noncode"
    if LNCIPEDIA_ID_RE.match(value):
        return "lncipedia_local"
    if ENSEMBL_CLONE_RE.match(value):
        return "ensembl_clone"
    if ENSEMBL_ARCHIVE_RE.match(value):
        return "ensembl_archive"
    if NCBI_ACCESSION_RE.match(value):
        return "ncbi_accession"
    if UCSC_RE.match(value):
        return "ucsc_locator"
    return "hgnc_symbol"


def print_classification_summary(rows: list[dict[str, str]]) -> None:
    counts = Counter(classify(row.get("id", "")) for row in rows)
    labels = {
        "rnacentral_noncode": "RNAcentral NONCODE",
        "lncipedia_local": "LNCipedia local FASTA",
        "ensembl_clone": "Ensembl clone symbol",
        "ensembl_archive": "Ensembl archive",
        "ncbi_accession": "NCBI accession",
        "ucsc_locator": "UCSC locator",
        "hgnc_symbol": "HGNC symbol",
    }
    print("\nClassification summary")
    print("-" * 54)
    print(f"{'Route':<28} {'Count':>8}")
    print("-" * 54)
    for route in (
        "rnacentral_noncode",
        "lncipedia_local",
        "ensembl_clone",
        "ensembl_archive",
        "ncbi_accession",
        "ucsc_locator",
        "hgnc_symbol",
    ):
        print(f"{labels[route]:<28} {counts.get(route, 0):>8}")
    print("-" * 54)
    print(f"{'Total':<28} {len(rows):>8}\n")


def first_doc_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("results", "data", "response"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict) and isinstance(value.get("docs"), list):
                return [item for item in value["docs"] if isinstance(item, dict)]
        return [data]
    return []


def get_json(
    session: requests.Session,
    url: str,
    *,
    error_log: Path,
    timeout: int = 20,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any | None:
    response = request(session, "GET", url, error_log=error_log, timeout=timeout, params=params, headers=headers)
    if response is None or response.status_code != 200:
        return None
    try:
        return response.json()
    except Exception as exc:
        append_error_log(error_log, url=response.url, message=f"json_decode_exception={exc}")
        return None


def post_json(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    *,
    error_log: Path,
    timeout: int = 30,
    headers: dict[str, str] | None = None,
) -> Any | None:
    response = request(session, "POST", url, error_log=error_log, timeout=timeout, json=payload, headers=headers)
    if response is None or response.status_code != 200:
        return None
    try:
        return response.json()
    except Exception as exc:
        append_error_log(error_log, url=response.url, message=f"json_decode_exception={exc}")
        return None


def get_text(
    session: requests.Session,
    url: str,
    *,
    error_log: Path,
    timeout: int = 20,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    response = request(session, "GET", url, error_log=error_log, timeout=timeout, params=params, headers=headers)
    if response is None or response.status_code != 200:
        return ""
    return str(response.text or "")


def fetch_ensembl_cdna(
    transcript_id: str,
    session: requests.Session,
    *,
    error_log: Path,
) -> str:
    url = ENSEMBL_SEQUENCE_ID.format(value=quote(transcript_id))
    text = get_text(
        session,
        url,
        error_log=error_log,
        headers={"Accept": "text/plain"},
        params={"type": "cdna"},
    )
    return normalize_sequence_value(text)


def transcript_length(record: dict[str, Any]) -> int:
    if isinstance(record.get("length"), int):
        return int(record["length"])
    start = record.get("start")
    end = record.get("end")
    if isinstance(start, int) and isinstance(end, int):
        return abs(end - start) + 1
    return 0


def longest_transcript_id(lookup_data: dict[str, Any]) -> str:
    transcripts = lookup_data.get("Transcript", [])
    if not isinstance(transcripts, list):
        return ""
    ranked = [item for item in transcripts if isinstance(item, dict) and item.get("id")]
    if not ranked:
        return ""
    ranked.sort(key=transcript_length, reverse=True)
    return str(ranked[0]["id"])


def fetch_ensembl_from_stable_id(
    stable_id: str,
    session: requests.Session,
    *,
    error_log: Path,
) -> tuple[str, str, str]:
    base_id = VERSIONED_ENS_RE.sub(r"\1", str(stable_id or "").strip())
    if not base_id:
        return "", "", ""
    if base_id.startswith("ENST"):
        seq = fetch_ensembl_cdna(base_id, session, error_log=error_log)
        return seq, base_id, "ensembl_transcript"
    lookup = get_json(
        session,
        ENSEMBL_LOOKUP_ID.format(value=quote(base_id)),
        error_log=error_log,
        headers={"Accept": "application/json"},
        params={"expand": 1},
    )
    if not isinstance(lookup, dict):
        return "", "", ""
    transcript_id = longest_transcript_id(lookup)
    if not transcript_id:
        return "", "", ""
    seq = fetch_ensembl_cdna(transcript_id, session, error_log=error_log)
    return seq, transcript_id, "ensembl_transcript"


def resolve_hgnc_symbol_to_gene(
    candidate: str,
    session: requests.Session,
    *,
    error_log: Path,
) -> tuple[str, str]:
    for url in (
        HGNC_FETCH_SYMBOL.format(value=quote(candidate)),
        HGNC_SEARCH_ALIAS_SYMBOL.format(value=quote(candidate)),
    ):
        data = get_json(
            session,
            url,
            error_log=error_log,
            headers={"Accept": "application/json"},
        )
        docs = first_doc_list(data)
        if not docs:
            continue
        doc = docs[0]
        gene_id = str(doc.get("ensembl_gene_id") or "").strip()
        symbol = str(doc.get("symbol") or candidate).strip()
        if gene_id:
            return gene_id, symbol
    return "", ""


def resolve_clone_symbol_to_ensembl(
    candidate: str,
    session: requests.Session,
    *,
    error_log: Path,
) -> str:
    data = get_json(
        session,
        ENSEMBL_XREF_SYMBOL.format(value=quote(candidate)),
        error_log=error_log,
        headers={"Accept": "application/json"},
    )
    docs = first_doc_list(data)
    if not docs:
        return ""
    for doc in docs:
        ens_id = str(doc.get("id") or "").strip()
        if ens_id.startswith(("ENSG", "ENST")):
            return ens_id
    return str(docs[0].get("id") or "").strip()


def fetch_ncbi_accession(
    accession: str,
    session: requests.Session,
    *,
    error_log: Path,
) -> tuple[str, str]:
    clean = str(accession or "").strip()
    if not clean:
        return "", ""
    url = NCBI_EFETCH.format(value=quote(clean))
    text = get_text(session, url, error_log=error_log, headers={"Accept": "text/plain"})
    seq = normalize_sequence_value(text)
    return seq, clean


def recursive_string_values(blob: Any) -> list[str]:
    values: list[str] = []
    if isinstance(blob, dict):
        for value in blob.values():
            values.extend(recursive_string_values(value))
    elif isinstance(blob, list):
        for item in blob:
            values.extend(recursive_string_values(item))
    elif isinstance(blob, str):
        values.append(blob)
    return values


def ucsc_candidate_ids(row: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    text_blob = json.dumps(row, sort_keys=True)
    ensembl_ids = []
    refseq_ids = []
    for match in ENSEMBL_STABLE_RE.findall(text_blob):
        base = VERSIONED_ENS_RE.sub(r"\1", match)
        if base not in ensembl_ids:
            ensembl_ids.append(base)
    for match in REFSEQ_RNA_RE.findall(text_blob):
        if match not in refseq_ids:
            refseq_ids.append(match)
    symbols: list[str] = []
    for key in ("geneName2", "name2", "gene_name", "symbol"):
        value = str(row.get(key) or "").strip()
        if value and value not in symbols:
            symbols.append(value)
    return ensembl_ids, refseq_ids, symbols


def resolve_ucsc_rows(
    candidate: str,
    session: requests.Session,
    *,
    error_log: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for genome in ("hg38", "hg19"):
        data = get_json(
            session,
            UCSC_TRACK,
            error_log=error_log,
            params={"genome": genome, "track": "knownGene", "name": candidate},
            headers={"Accept": "application/json"},
        )
        if not isinstance(data, dict):
            continue
        value = data.get("knownGene")
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            rows.append(value)
    return rows


def rnacentral_sequence_from_candidate(
    candidate: str,
    session: requests.Session,
    *,
    error_log: Path,
) -> tuple[str, str]:
    data = get_json(
        session,
        RNACENTRAL_EXTERNAL_ID,
        error_log=error_log,
        params={"external_id": candidate, "format": "json"},
        headers={"Accept": "application/json"},
    )
    docs = first_doc_list(data)
    for doc in docs:
        sequence = normalize_sequence_value(str(doc.get("sequence") or ""))
        if sequence:
            resolved_id = str(doc.get("rnacentral_id") or doc.get("rna_id") or candidate).strip()
            return sequence, resolved_id
    return "", ""


def confidence_for_match(raw_id: str, matched_query: str) -> str:
    return "high" if str(raw_id or "").strip() == str(matched_query or "").strip() else "low"


def resolve_noncode(
    raw_id: str,
    variants: list[str],
    *,
    session: requests.Session,
    error_log: Path,
    noncode_index: dict[str, str],
) -> tuple[SequenceHit | None, str]:
    attempted: list[str] = []
    for candidate in variants:
        if noncode_index:
            seq = normalize_sequence_value(noncode_index.get(candidate, ""))
            if seq:
                return (
                    SequenceHit(
                        query_id=raw_id,
                        matched_query=candidate,
                        route="rnacentral_noncode",
                        resolved_id=candidate,
                        resolved_id_type="noncode_transcript",
                        sequence=seq,
                        source="noncode_fasta",
                        confidence=confidence_for_match(raw_id, candidate),
                    ),
                    "",
                )
            attempted.append(f"{candidate}:noncode_fasta_empty")
        seq, resolved_id = rnacentral_sequence_from_candidate(candidate, session, error_log=error_log)
        if seq:
            return (
                SequenceHit(
                    query_id=raw_id,
                    matched_query=candidate,
                    route="rnacentral_noncode",
                    resolved_id=resolved_id,
                    resolved_id_type="noncode_transcript",
                    sequence=seq,
                    source="rnacentral",
                    confidence=confidence_for_match(raw_id, candidate),
                ),
                "",
            )
        attempted.append(f"{candidate}:rnacentral_empty")
    return None, "; ".join(attempted) if attempted else "rnacentral_empty"


def resolve_lncipedia(
    raw_id: str,
    variants: list[str],
    *,
    lncipedia_index: dict[str, str],
) -> tuple[SequenceHit | None, str]:
    if not lncipedia_index:
        return None, "lncipedia_no_local_fasta"
    attempted: list[str] = []
    for candidate in variants:
        seq = normalize_sequence_value(lncipedia_index.get(candidate, ""))
        if seq:
            return (
                SequenceHit(
                    query_id=raw_id,
                    matched_query=candidate,
                    route="lncipedia_local",
                    resolved_id=candidate,
                    resolved_id_type="lncipedia_transcript",
                    sequence=seq,
                    source="lncipedia_fasta",
                    confidence=confidence_for_match(raw_id, candidate),
                ),
                "",
            )
        attempted.append(f"{candidate}:lncipedia_fasta_empty")
    return None, "; ".join(attempted)


def resolve_ensembl_clone(
    raw_id: str,
    variants: list[str],
    *,
    session: requests.Session,
    error_log: Path,
) -> tuple[SequenceHit | None, str]:
    attempted: list[str] = []
    for candidate in variants:
        stable_id = resolve_clone_symbol_to_ensembl(candidate, session, error_log=error_log)
        if not stable_id:
            attempted.append(f"{candidate}:ensembl_xref_empty")
            continue
        seq, transcript_id, resolved_type = fetch_ensembl_from_stable_id(stable_id, session, error_log=error_log)
        if seq:
            return (
                SequenceHit(
                    query_id=raw_id,
                    matched_query=candidate,
                    route="ensembl_clone",
                    resolved_id=transcript_id,
                    resolved_id_type=resolved_type,
                    sequence=seq,
                    source="ensembl_clone_cdna",
                    confidence=confidence_for_match(raw_id, candidate),
                ),
                "",
            )
        attempted.append(f"{candidate}:ensembl_cdna_empty")
    return None, "; ".join(attempted) if attempted else "ensembl_clone_empty"


def build_archive_map(
    ids: list[str],
    session: requests.Session,
    *,
    error_log: Path,
) -> dict[str, str]:
    clean_ids: list[str] = []
    for raw_id in ids:
        for candidate in build_variants(raw_id):
            if candidate.startswith(("ENSG", "ENST")):
                clean = VERSIONED_ENS_RE.sub(r"\1", candidate)
                if clean and clean not in clean_ids:
                    clean_ids.append(clean)
    archive_map: dict[str, str] = {}
    batch_size = 50
    for idx in range(0, len(clean_ids), batch_size):
        batch = clean_ids[idx : idx + batch_size]
        data = post_json(
            session,
            ENSEMBL_ARCHIVE_POST,
            {"ids": batch},
            error_log=error_log,
            headers={"Accept": "application/json"},
        )
        if not isinstance(data, dict):
            continue
        for stable_id, info in data.items():
            if not isinstance(info, dict):
                continue
            latest = str(info.get("latest") or stable_id).strip()
            if latest:
                archive_map[stable_id] = latest
    return archive_map


def resolve_ensembl_archive(
    raw_id: str,
    variants: list[str],
    *,
    archive_map: dict[str, str],
    session: requests.Session,
    error_log: Path,
) -> tuple[SequenceHit | None, str]:
    attempted: list[str] = []
    for candidate in variants:
        if not candidate.startswith(("ENSG", "ENST")):
            continue
        clean = VERSIONED_ENS_RE.sub(r"\1", candidate)
        current_id = archive_map.get(clean, clean)
        seq, transcript_id, resolved_type = fetch_ensembl_from_stable_id(current_id, session, error_log=error_log)
        if seq:
            return (
                SequenceHit(
                    query_id=raw_id,
                    matched_query=candidate,
                    route="ensembl_archive",
                    resolved_id=transcript_id,
                    resolved_id_type=resolved_type,
                    sequence=seq,
                    source="ensembl_archive_cdna",
                    confidence=confidence_for_match(raw_id, candidate),
                ),
                "",
            )
        attempted.append(f"{candidate}:ensembl_archive_cdna_empty")
    return None, "; ".join(attempted) if attempted else "ensembl_archive_empty"


def resolve_ncbi_accession(
    raw_id: str,
    variants: list[str],
    *,
    session: requests.Session,
    error_log: Path,
) -> tuple[SequenceHit | None, str]:
    attempted: list[str] = []
    for candidate in variants:
        seq, accession = fetch_ncbi_accession(candidate, session, error_log=error_log)
        if seq:
            return (
                SequenceHit(
                    query_id=raw_id,
                    matched_query=candidate,
                    route="ncbi_accession",
                    resolved_id=accession,
                    resolved_id_type="ncbi_accession",
                    sequence=seq,
                    source="ncbi_nucleotide",
                    confidence=confidence_for_match(raw_id, candidate),
                ),
                "",
            )
        attempted.append(f"{candidate}:ncbi_efetch_empty")
    return None, "; ".join(attempted) if attempted else "ncbi_efetch_empty"


def resolve_hgnc_symbol(
    raw_id: str,
    variants: list[str],
    *,
    session: requests.Session,
    error_log: Path,
    route: str = "hgnc_symbol",
    source: str = "hgnc_ensembl_cdna",
) -> tuple[SequenceHit | None, str]:
    attempted: list[str] = []
    for candidate in variants:
        gene_id, _symbol = resolve_hgnc_symbol_to_gene(candidate, session, error_log=error_log)
        if not gene_id:
            attempted.append(f"{candidate}:hgnc_symbol_empty")
            continue
        seq, transcript_id, resolved_type = fetch_ensembl_from_stable_id(gene_id, session, error_log=error_log)
        if seq:
            return (
                SequenceHit(
                    query_id=raw_id,
                    matched_query=candidate,
                    route=route,
                    resolved_id=transcript_id,
                    resolved_id_type=resolved_type,
                    sequence=seq,
                    source=source,
                    confidence=confidence_for_match(raw_id, candidate),
                ),
                "",
            )
        attempted.append(f"{candidate}:ensembl_cdna_empty")
    return None, "; ".join(attempted) if attempted else "hgnc_symbol_empty"


def resolve_ucsc(
    raw_id: str,
    variants: list[str],
    *,
    session: requests.Session,
    error_log: Path,
) -> tuple[SequenceHit | None, str]:
    attempted: list[str] = []
    for candidate in variants:
        rows = resolve_ucsc_rows(candidate, session, error_log=error_log)
        if not rows:
            attempted.append(f"{candidate}:ucsc_lookup_empty")
            continue
        for row in rows:
            ensembl_ids, refseq_ids, symbols = ucsc_candidate_ids(row)
            for stable_id in ensembl_ids:
                seq, transcript_id, resolved_type = fetch_ensembl_from_stable_id(stable_id, session, error_log=error_log)
                if seq:
                    return (
                        SequenceHit(
                            query_id=raw_id,
                            matched_query=candidate,
                            route="ucsc_locator",
                            resolved_id=transcript_id,
                            resolved_id_type=resolved_type,
                            sequence=seq,
                            source="ucsc_to_ensembl",
                            confidence=confidence_for_match(raw_id, candidate),
                        ),
                        "",
                    )
            for accession in refseq_ids:
                seq, resolved_accession = fetch_ncbi_accession(accession, session, error_log=error_log)
                if seq:
                    return (
                        SequenceHit(
                            query_id=raw_id,
                            matched_query=candidate,
                            route="ucsc_locator",
                            resolved_id=resolved_accession,
                            resolved_id_type="ncbi_accession",
                            sequence=seq,
                            source="ucsc_to_ncbi",
                            confidence=confidence_for_match(raw_id, candidate),
                        ),
                        "",
                    )
            for symbol in symbols:
                hit, _reason = resolve_hgnc_symbol(
                    raw_id,
                    [symbol],
                    session=session,
                    error_log=error_log,
                    route="ucsc_locator",
                    source="ucsc_to_hgnc_ensembl_cdna",
                )
                if hit:
                    return hit, ""
        attempted.append(f"{candidate}:ucsc_no_supported_link_id")
    return None, "; ".join(attempted) if attempted else "ucsc_no_supported_link_id"


def resolve_one(
    row: dict[str, str],
    *,
    session: requests.Session,
    error_log: Path,
    noncode_index: dict[str, str],
    lncipedia_index: dict[str, str],
    archive_map: dict[str, str],
) -> tuple[SequenceHit | None, ResidualHit]:
    raw_id = str(row.get("id") or "").strip()
    id_type = str(row.get("type") or "").strip()
    route = classify(raw_id)
    variants = build_variants(raw_id)

    if route == "rnacentral_noncode":
        hit, reason = resolve_noncode(raw_id, variants, session=session, error_log=error_log, noncode_index=noncode_index)
    elif route == "lncipedia_local":
        hit, reason = resolve_lncipedia(raw_id, variants, lncipedia_index=lncipedia_index)
    elif route == "ensembl_clone":
        hit, reason = resolve_ensembl_clone(raw_id, variants, session=session, error_log=error_log)
    elif route == "ensembl_archive":
        hit, reason = resolve_ensembl_archive(
            raw_id,
            variants,
            archive_map=archive_map,
            session=session,
            error_log=error_log,
        )
    elif route == "ncbi_accession":
        hit, reason = resolve_ncbi_accession(raw_id, variants, session=session, error_log=error_log)
    elif route == "ucsc_locator":
        hit, reason = resolve_ucsc(raw_id, variants, session=session, error_log=error_log)
    else:
        hit, reason = resolve_hgnc_symbol(raw_id, variants, session=session, error_log=error_log)

    if hit is not None:
        return hit, ResidualHit(query_id=raw_id, id_type=id_type, route=route, reason="")
    return None, ResidualHit(query_id=raw_id, id_type=id_type, route=route, reason=reason)


def checkpoint_path_for(input_path: Path, checkpoint_dir: Path) -> Path:
    digest = hashlib.sha1(str(input_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return checkpoint_dir / f"resolve_checkpoint_{digest}.json"


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def load_override_rows(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    override_path = ensure_sequence_override_csv(path)
    ordered_ids: list[str] = []
    rows: dict[str, dict[str, str]] = {}
    with override_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or OVERRIDE_COLUMNS
        missing = [column for column in OVERRIDE_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"{override_path} must have columns {', '.join(OVERRIDE_COLUMNS)}")
        for row in reader:
            query_id = str(row.get("query_id") or "").strip()
            if not query_id:
                continue
            ordered_ids.append(query_id)
            rows[query_id] = {column: str(row.get(column) or "").strip() for column in OVERRIDE_COLUMNS}
    return ordered_ids, rows


def update_override_rows(
    path: Path,
    hits: list[SequenceHit],
) -> tuple[int, int]:
    ordered_ids, rows = load_override_rows(path)
    updated_existing = 0
    added_new = 0
    for hit in hits:
        row = hit.override_row()
        query_id = row["query_id"]
        if query_id in rows:
            updated_existing += 1
        else:
            ordered_ids.append(query_id)
            added_new += 1
        rows[query_id] = row
    with ensure_sequence_override_csv(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OVERRIDE_COLUMNS)
        writer.writeheader()
        seen: set[str] = set()
        for query_id in ordered_ids:
            if query_id in rows and query_id not in seen:
                writer.writerow(rows[query_id])
                seen.add(query_id)
    return updated_existing, added_new


def write_residual_csv(path: Path, residuals: list[ResidualHit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "type", "status", "reason", "resolved_id"])
        writer.writeheader()
        for residual in residuals:
            writer.writerow(residual.residual_row())


def load_input_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [{str(k): str(v or "") for k, v in row.items()} for row in reader]


def make_default_path(project_rel: str) -> Path:
    return PROJECT_ROOT / project_rel


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Recover real sequences for unresolved lncRNA IDs.")
    parser.add_argument("--input", default=str(make_default_path("Data/output_data/unresolved_ids.csv")))
    parser.add_argument("--overrides", default=str(make_default_path("Data/raw/sequence_id_overrides.csv")))
    parser.add_argument("--residual", default=str(make_default_path("Data/output_data/unresolved_ids_recovery_residual.csv")))
    parser.add_argument("--error-log", default=str(make_default_path("Data/output_data/resolve_errors.log")))
    parser.add_argument("--checkpoint-dir", default=str(make_default_path("Data/output_data")))
    parser.add_argument("--noncode-fasta", help="Optional local NONCODE FASTA. Used before RNAcentral when provided.")
    parser.add_argument("--lncipedia-fasta", help="Required local FASTA for LNCipedia-style IDs if you want them resolved.")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true", help="Resume from an existing input-specific checkpoint.")
    args = parser.parse_args(argv)

    input_path = Path(args.input).resolve()
    overrides_path = Path(args.overrides).resolve()
    residual_path = Path(args.residual).resolve()
    error_log_path = Path(args.error_log).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_path = checkpoint_path_for(input_path, checkpoint_dir)

    rows = load_input_rows(input_path)
    print(f"Loaded {len(rows)} IDs from {input_path}")
    print_classification_summary(rows)

    noncode_index = load_fasta_index(Path(args.noncode_fasta).resolve()) if args.noncode_fasta else {}
    lncipedia_index = load_fasta_index(Path(args.lncipedia_fasta).resolve()) if args.lncipedia_fasta else {}
    if noncode_index:
        print(f"[index] NONCODE FASTA entries: {len(noncode_index)}")
    if lncipedia_index:
        print(f"[index] LNCipedia FASTA entries: {len(lncipedia_index)}")

    session = make_session(args.delay)
    checkpoint_payload = load_checkpoint(checkpoint_path) if args.resume else {}
    done = checkpoint_payload.get("done", {}) if isinstance(checkpoint_payload, dict) else {}
    if done:
        print(f"[resume] loaded {len(done)} processed IDs from {checkpoint_path}")

    archive_ids = [
        str(row.get("id") or "").strip()
        for row in rows
        if classify(row.get("id", "")) == "ensembl_archive"
    ]
    archive_map = build_archive_map(archive_ids, session, error_log=error_log_path) if archive_ids else {}
    if archive_map:
        print(f"[archive] mapped {len(archive_map)} Ensembl archive candidates")

    iterable = tqdm(rows, unit="id") if HAS_TQDM else rows
    processed_since_checkpoint = 0

    for row in iterable:
        raw_id = str(row.get("id") or "").strip()
        if not raw_id or raw_id in done:
            continue
        hit, residual = resolve_one(
            row,
            session=session,
            error_log=error_log_path,
            noncode_index=noncode_index,
            lncipedia_index=lncipedia_index,
            archive_map=archive_map,
        )
        if hit is not None:
            done[raw_id] = {"kind": "resolved", "data": asdict(hit)}
        else:
            done[raw_id] = {"kind": "residual", "data": asdict(residual)}
        processed_since_checkpoint += 1
        if processed_since_checkpoint % 100 == 0:
            save_checkpoint(checkpoint_path, {"done": done})

    save_checkpoint(checkpoint_path, {"done": done})

    resolved_hits: list[SequenceHit] = []
    residual_hits: list[ResidualHit] = []
    for row in rows:
        raw_id = str(row.get("id") or "").strip()
        result = done.get(raw_id)
        if not isinstance(result, dict):
            residual_hits.append(
                ResidualHit(
                    query_id=raw_id,
                    id_type=str(row.get("type") or "").strip(),
                    route=classify(raw_id),
                    reason="missing_from_checkpoint",
                )
            )
            continue
        payload = result.get("data", {})
        if result.get("kind") == "resolved" and isinstance(payload, dict):
            resolved_hits.append(
                SequenceHit(
                    query_id=str(payload.get("query_id") or raw_id),
                    matched_query=str(payload.get("matched_query") or raw_id),
                    route=str(payload.get("route") or classify(raw_id)),
                    resolved_id=str(payload.get("resolved_id") or ""),
                    resolved_id_type=str(payload.get("resolved_id_type") or ""),
                    sequence=normalize_sequence_value(str(payload.get("sequence") or "")),
                    source=str(payload.get("source") or ""),
                    confidence=str(payload.get("confidence") or "high"),
                )
            )
        else:
            residual_hits.append(
                ResidualHit(
                    query_id=str(payload.get("query_id") or raw_id),
                    id_type=str(payload.get("id_type") or row.get("type") or ""),
                    route=str(payload.get("route") or classify(raw_id)),
                    reason=str(payload.get("reason") or "unresolved"),
                )
            )

    updated_existing, added_new = update_override_rows(overrides_path, resolved_hits)
    write_residual_csv(residual_path, residual_hits)

    source_counts = Counter(hit.source for hit in resolved_hits)
    low_confidence = sum(1 for hit in resolved_hits if hit.confidence == "low")

    print("\nFinal summary")
    print("-" * 54)
    print(f"Resolved with sequence : {len(resolved_hits)}")
    print(f"Residual unresolved    : {len(residual_hits)}")
    print(f"Updated override rows  : {updated_existing}")
    print(f"Added override rows    : {added_new}")
    print(f"Low-confidence hits    : {low_confidence}")
    print(f"Overrides file         : {overrides_path}")
    print(f"Residual file          : {residual_path}")
    print(f"Error log              : {error_log_path}")
    print(f"Checkpoint             : {checkpoint_path}")
    if source_counts:
        print("\nResolved by source")
        print("-" * 54)
        for source, count in source_counts.most_common():
            print(f"{source:<28} {count:>8}")


if __name__ == "__main__":
    main()
