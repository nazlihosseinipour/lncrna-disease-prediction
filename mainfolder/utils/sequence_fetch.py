from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import quote


ENSEMBL_LOOKUP = "https://rest.ensembl.org/lookup/symbol/homo_sapiens/{symbol}"
ENSEMBL_XREF = "https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{symbol}"
ENSEMBL_XREF_NAME = "https://rest.ensembl.org/xrefs/name/homo_sapiens/{symbol}"
ENSEMBL_SEQ = "https://rest.ensembl.org/sequence/id/{ensembl_id}"
NCBI_EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id={accession}&rettype=fasta&retmode=text"
)
VALID_NUCLEOTIDE_CHARS = set("ACGTUNRYSWKMBDHVX")


@dataclass(frozen=True)
class SequenceFetchResult:
    query_id: str
    id_type: str
    sequence: str
    status: str
    detail: str
    resolved_id: str = ""

    def report_row(self, id_column: str = "ID") -> dict[str, str]:
        return {
            id_column: self.query_id,
            "type": self.id_type,
            "status": self.status,
            "detail": self.detail,
            "resolved_id": self.resolved_id,
        }

    def review_row(self) -> dict[str, str]:
        return {
            "id": self.query_id,
            "type": self.id_type,
            "status": self.status,
            "reason": self.detail,
            "resolved_id": self.resolved_id,
        }


def parse_fasta_text(text: str) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln and not ln.startswith(">")]
    return "".join(lines).replace(" ", "").strip().upper()


def _is_probable_nucleotide_sequence(seq: str) -> bool:
    value = str(seq or "").strip().upper()
    return bool(value) and set(value).issubset(VALID_NUCLEOTIDE_CHARS)


def detect_id_kind(raw_id: str) -> str:
    rid = str(raw_id or "").strip().upper()
    if not rid:
        return "empty"
    if rid.startswith("ENS"):
        return "ensembl"
    accession_patterns = (
        r"^[A-Z]{2}_\d+(\.\d+)?$",
        r"^[A-Z]{1}\d{5}(\.\d+)?$",
        r"^[A-Z]{2}\d{6,}(\.\d+)?$",
        r"^[A-Z]{4}\d{8,}(\.\d+)?$",
    )
    if any(re.match(pattern, rid) for pattern in accession_patterns):
        return "accession"
    return "symbol"


def _result(
    query_id: str,
    id_type: str,
    *,
    sequence: str = "",
    status: str,
    detail: str,
    resolved_id: str = "",
) -> SequenceFetchResult:
    return SequenceFetchResult(
        query_id=str(query_id or "").strip(),
        id_type=id_type,
        sequence=str(sequence or "").strip().upper(),
        status=status,
        detail=str(detail or "").strip(),
        resolved_id=str(resolved_id or "").strip(),
    )


def fetch_ensembl_sequence(session, ensembl_id: str, *, timeout: int = 10, query_id: str | None = None) -> SequenceFetchResult:
    eid = str(ensembl_id or "").strip()
    qid = str(query_id or eid).strip()
    try:
        response = session.get(
            ENSEMBL_SEQ.format(ensembl_id=quote(eid, safe="")),
            headers={"Accept": "text/plain"},
            timeout=timeout,
        )
    except Exception as exc:
        return _result(qid, "ensembl", status="unresolved", detail=f"ensembl_seq_err:{exc}", resolved_id=eid)
    if response.status_code != 200:
        return _result(qid, "ensembl", status="unresolved", detail=f"ensembl_seq_status:{response.status_code}", resolved_id=eid)
    seq = "".join(str(response.text or "").split()).upper()
    if not seq or seq.startswith("{"):
        return _result(qid, "ensembl", status="unresolved", detail=f"ensembl_seq_empty:{eid}", resolved_id=eid)
    if not _is_probable_nucleotide_sequence(seq):
        return _result(qid, "ensembl", status="unresolved", detail=f"ensembl_seq_invalid_payload:{eid}", resolved_id=eid)
    return _result(qid, "ensembl", sequence=seq, status="ensembl_fetched", detail=f"ensembl_seq:{eid}", resolved_id=eid)


def fetch_ncbi_accession(session, accession: str, *, timeout: int = 10, query_id: str | None = None) -> SequenceFetchResult:
    acc = str(accession or "").strip()
    qid = str(query_id or acc).strip()
    try:
        response = session.get(NCBI_EFETCH.format(accession=quote(acc, safe="")), timeout=timeout)
    except Exception as exc:
        return _result(qid, "accession", status="unresolved", detail=f"ncbi_err:{exc}")
    if response.status_code != 200:
        return _result(qid, "accession", status="unresolved", detail=f"ncbi_status:{response.status_code}")
    seq = parse_fasta_text(response.text)
    if not seq:
        return _result(qid, "accession", status="unresolved", detail=f"ncbi_empty:{acc}")
    if not _is_probable_nucleotide_sequence(seq):
        return _result(qid, "accession", status="unresolved", detail=f"ncbi_invalid_payload:{acc}")
    return _result(qid, "accession", sequence=seq, status="accession_fetched", detail=f"ncbi:{acc}", resolved_id=acc)


def _xref_candidates(session, symbol: str, *, timeout: int = 10) -> tuple[list[str], list[str]]:
    symbol_q = quote(symbol, safe="")
    candidates: list[str] = []
    notes: list[str] = []
    for label, template in (("xrefs_symbol", ENSEMBL_XREF), ("xrefs_name", ENSEMBL_XREF_NAME)):
        try:
            response = session.get(
                template.format(symbol=symbol_q),
                headers={"Accept": "application/json"},
                timeout=timeout,
            )
        except Exception as exc:
            notes.append(f"{label}_err:{exc}")
            continue
        if response.status_code != 200:
            notes.append(f"{label}_status:{response.status_code}")
            continue
        try:
            payload = response.json() or []
        except Exception as exc:
            notes.append(f"{label}_json_err:{exc}")
            continue
        for item in payload:
            eid = str(item.get("id") or "").strip()
            typ = str(item.get("type") or "").strip().lower()
            if eid.startswith("ENS") or typ in {"gene", "transcript"}:
                candidates.append(eid)
    unique_candidates = sorted(set(candidates))
    return unique_candidates, notes


def resolve_symbol(session, symbol: str, *, timeout: int = 10) -> SequenceFetchResult:
    sym = str(symbol or "").strip()
    symbol_q = quote(sym, safe="")
    try:
        response = session.get(
            ENSEMBL_LOOKUP.format(symbol=symbol_q),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
    except Exception as exc:
        return _result(sym, "symbol", status="unresolved", detail=f"symbol_lookup_err:{exc}")
    if response.status_code == 200:
        try:
            payload = response.json() or {}
        except Exception as exc:
            return _result(sym, "symbol", status="unresolved", detail=f"symbol_lookup_json_err:{exc}")
        resolved_id = str(payload.get("id") or "").strip()
        if not resolved_id:
            return _result(sym, "symbol", status="unresolved", detail="symbol_lookup_missing_id")
        seq_result = fetch_ensembl_sequence(session, resolved_id, timeout=timeout, query_id=sym)
        if seq_result.sequence:
            return _result(
                sym,
                "symbol",
                sequence=seq_result.sequence,
                status="symbol_resolved",
                detail=f"symbol_lookup:{sym}->{resolved_id}",
                resolved_id=resolved_id,
            )
        return _result(
            sym,
            "symbol",
            status="unresolved",
            detail=f"symbol_lookup_resolved:{resolved_id}; {seq_result.detail}",
            resolved_id=resolved_id,
        )
    candidates, notes = _xref_candidates(session, sym, timeout=timeout)
    if len(candidates) > 1:
        return _result(
            sym,
            "symbol",
            status="ambiguous_symbol",
            detail=f"ambiguous_symbol:{len(candidates)}_candidates",
            resolved_id="",
        )
    if response.status_code in {400, 404}:
        detail = f"symbol_lookup_status:{response.status_code}"
        if notes:
            detail = f"{detail}; {'; '.join(notes)}"
        return _result(sym, "symbol", status="unresolved", detail=detail)
    return _result(sym, "symbol", status="unresolved", detail=f"symbol_lookup_status:{response.status_code}")


def fetch_sequence_by_id(session, raw_id: str, *, timeout: int = 10) -> SequenceFetchResult:
    rid = str(raw_id or "").strip()
    kind = detect_id_kind(rid)
    if kind == "empty":
        return _result(rid, "empty", status="unresolved", detail="empty_id")
    if kind == "ensembl":
        return fetch_ensembl_sequence(session, rid, timeout=timeout)
    if kind == "accession":
        return fetch_ncbi_accession(session, rid, timeout=timeout)
    return resolve_symbol(session, rid, timeout=timeout)


def needs_manual_review(result: SequenceFetchResult) -> bool:
    return result.status in {"unresolved", "ambiguous_symbol"}
