from __future__ import annotations

from dataclasses import dataclass
import re
import time
from urllib.parse import quote


ENSEMBL_LOOKUP = "https://rest.ensembl.org/lookup/symbol/homo_sapiens/{symbol}"
ENSEMBL_LOOKUP_ID = "https://rest.ensembl.org/lookup/id/{ensembl_id}?expand=1"
ENSEMBL_XREF = "https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{symbol}"
ENSEMBL_XREF_NAME = "https://rest.ensembl.org/xrefs/name/homo_sapiens/{symbol}"
ENSEMBL_SEQ = "https://rest.ensembl.org/sequence/id/{ensembl_id}"
ENSEMBL_ARCHIVE = "https://rest.ensembl.org/archive/id/{ensembl_id}"
NCBI_EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id={accession}&rettype=fasta&retmode=text"
)
NCBI_ESEARCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    "?db=nuccore&term={term}&retmode=json&retmax={retmax}"
)
VALID_NUCLEOTIDE_CHARS = set("ACGTUNRYSWKMBDHVX")
INVALID_SEQUENCE_MARKERS = (
    "FAILED TO UNDERSTAND ID",
    "ERROR:",
    "<!DOCTYPE",
    "<HTML",
    "</HTML",
    "<?XML",
)
EMPTY_SEQUENCE_MARKERS = {"", "nan", "none", "null"}
CAUTIOUS_PREFIXES = ("AC", "AL", "AJ", "AF", "AP")
SAFE_VERSIONED_ID_RE = re.compile(
    r"^((?:ENS[A-Z0-9]*\d+)|(?:(?:NM|NR|XM|XR|NC|NG|NT|NW|NZ)_\d+))\.(\d+)$",
    re.I,
)
CAUTIOUS_SYMBOL_VERSION_RE = re.compile(
    r"^(((?:AC|AL|AJ|AF|AP)[A-Z0-9_-]*))\.(\d+)$",
    re.I,
)
SYMBOL_TRANSCRIPT_SUFFIX_RE = re.compile(r"^(.+)-(\d{3,})$")
SYMBOL_COLON_SUFFIX_RE = re.compile(r"^(.+):(\d+)$")
SYMBOL_DOT_V_SUFFIX_RE = re.compile(r"^(.+)\.v(\d+)$", re.I)
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
REQUEST_RETRIES = 2
REQUEST_RETRY_SLEEP = 0.2


@dataclass(frozen=True)
class SequenceFetchResult:
    query_id: str
    id_type: str
    sequence: str
    status: str
    detail: str
    resolved_id: str = ""
    alternative_ids: tuple[str, ...] = ()
    source: str = ""

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

    def alternatives_row(self) -> dict[str, str]:
        return {
            "query_id": self.query_id,
            "selected_id": self.resolved_id,
            "alternative_ids": "|".join(self.alternative_ids),
            "source": self.source,
        }


@dataclass(frozen=True)
class IdentifierRoute:
    raw_id: str
    family: str
    namespace: str
    id_type: str
    normalized_id: str


def parse_fasta_text(text: str) -> str:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln and not ln.startswith(">")]
    return "".join(lines).replace(" ", "").strip().upper()


def normalize_sequence_value(seq: str) -> str:
    raw = str(seq or "").strip()
    if not raw or raw.lower() in EMPTY_SEQUENCE_MARKERS:
        return ""
    upper_raw = raw.upper()
    if any(marker in upper_raw for marker in INVALID_SEQUENCE_MARKERS):
        return ""
    compact = parse_fasta_text(raw) if raw.startswith(">") else "".join(raw.split()).upper()
    return compact if _is_probable_nucleotide_sequence(compact) else ""


def has_usable_sequence_value(seq: str) -> bool:
    return bool(normalize_sequence_value(seq))


def _is_probable_nucleotide_sequence(seq: str) -> bool:
    value = str(seq or "").strip().upper()
    return bool(value) and set(value).issubset(VALID_NUCLEOTIDE_CHARS)


def _request_with_retry(session, url: str, *, headers=None, timeout: int = 10, retries: int = REQUEST_RETRIES):
    last_response = None
    last_exc = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(REQUEST_RETRY_SLEEP * (attempt + 1))
                continue
            return None, exc
        last_response = response
        if response.status_code in RETRYABLE_HTTP_STATUSES and attempt < retries:
            time.sleep(REQUEST_RETRY_SLEEP * (attempt + 1))
            continue
        return response, None
    return last_response, last_exc


def strip_safe_version_suffix(raw_id: str) -> tuple[str, bool]:
    rid = str(raw_id or "").strip()
    match = SAFE_VERSIONED_ID_RE.match(rid)
    if not match:
        return rid, False
    return match.group(1), True


def strip_cautious_symbol_version_suffix(raw_id: str) -> tuple[str, bool]:
    rid = str(raw_id or "").strip()
    match = CAUTIOUS_SYMBOL_VERSION_RE.match(rid)
    if not match:
        return rid, False
    return match.group(1), True


def symbol_fallback_variants(raw_id: str) -> list[str]:
    rid = str(raw_id or "").strip()
    variants: list[str] = []
    for pattern in (SYMBOL_COLON_SUFFIX_RE, SYMBOL_TRANSCRIPT_SUFFIX_RE, SYMBOL_DOT_V_SUFFIX_RE):
        match = pattern.match(rid)
        if match:
            candidate = str(match.group(1) or "").strip()
            if candidate and candidate != rid and candidate not in variants:
                variants.append(candidate)
    return variants


def detect_id_route(raw_id: str) -> IdentifierRoute:
    rid = str(raw_id or "").strip()
    upper = rid.upper()
    if not rid:
        return IdentifierRoute(rid, "empty", "empty", "empty", rid)
    if SAFE_VERSIONED_ID_RE.match(rid):
        family = "ensembl_versioned" if upper.startswith("ENS") else "refseq_versioned"
        namespace = "ensembl" if upper.startswith("ENS") else "ncbi_nuccore"
        id_type = "ensembl" if upper.startswith("ENS") else "accession"
        return IdentifierRoute(rid, family, namespace, id_type, rid)
    if upper.startswith("ENS"):
        return IdentifierRoute(rid, "ensembl", "ensembl", "ensembl", rid)
    if re.match(r"^(?:NM|NR|XM|XR|NC|NG|NT|NW|NZ)_\d+(\.\d+)?$", upper):
        return IdentifierRoute(rid, "refseq_accession", "ncbi_nuccore", "accession", rid)
    if re.match(r"^[A-Z]\d{5}(\.\d+)?$", upper):
        return IdentifierRoute(rid, "genbank_accession", "ncbi_nuccore", "accession", rid)
    if re.match(r"^[A-Z]{2}\d{6,}(\.\d+)?$", upper) and not upper.startswith(CAUTIOUS_PREFIXES):
        return IdentifierRoute(rid, "genbank_accession", "ncbi_nuccore", "accession", rid)
    if CAUTIOUS_SYMBOL_VERSION_RE.match(rid):
        return IdentifierRoute(rid, "cautious_clone_symbol_versioned", "ensembl_symbol", "symbol", rid)
    if re.match(r"^(?:LOC)\d+$", upper):
        return IdentifierRoute(rid, "loc_symbol", "ncbi_symbol_search", "symbol", rid)
    if re.match(r"^(?:LNC-|LNCV|LNCRNA|LNC_)", upper):
        return IdentifierRoute(rid, "catalog_symbol", "ensembl_symbol", "symbol", rid)
    if re.match(r"^[A-Z0-9-]{2,6}$", upper):
        return IdentifierRoute(rid, "family_symbol", "ensembl_symbol", "symbol", rid)
    return IdentifierRoute(rid, "generic_symbol", "ensembl_symbol", "symbol", rid)


def detect_id_kind(raw_id: str) -> str:
    return detect_id_route(raw_id).id_type


def _symbol_route_variants(route: IdentifierRoute) -> list[str]:
    variants: list[str] = []
    if route.family == "cautious_clone_symbol_versioned":
        base_symbol, can_retry = strip_cautious_symbol_version_suffix(route.normalized_id)
        if can_retry and base_symbol != route.normalized_id:
            variants.append(base_symbol)
    if route.family in {"catalog_symbol", "generic_symbol", "family_symbol", "cautious_clone_symbol_versioned"}:
        for candidate in symbol_fallback_variants(route.normalized_id):
            if candidate not in variants:
                variants.append(candidate)
    return variants


def _result(
    query_id: str,
    id_type: str,
    *,
    sequence: str = "",
    status: str,
    detail: str,
    resolved_id: str = "",
    alternative_ids: tuple[str, ...] = (),
    source: str = "",
) -> SequenceFetchResult:
    return SequenceFetchResult(
        query_id=str(query_id or "").strip(),
        id_type=id_type,
        sequence=normalize_sequence_value(sequence),
        status=status,
        detail=str(detail or "").strip(),
        resolved_id=str(resolved_id or "").strip(),
        alternative_ids=tuple(str(x).strip() for x in (alternative_ids or ()) if str(x).strip()),
        source=str(source or "").strip(),
    )


def fetch_ensembl_sequence(session, ensembl_id: str, *, timeout: int = 10, query_id: str | None = None) -> SequenceFetchResult:
    eid = str(ensembl_id or "").strip()
    qid = str(query_id or eid).strip()
    response, exc = _request_with_retry(
        session,
        ENSEMBL_SEQ.format(ensembl_id=quote(eid, safe="")),
        headers={"Accept": "text/plain"},
        timeout=timeout,
    )
    if exc is not None:
        return _result(qid, "ensembl", status="unresolved", detail=f"ensembl_seq_err:{exc}", resolved_id=eid)
    if response.status_code != 200:
        return _result(qid, "ensembl", status="unresolved", detail=f"ensembl_seq_status:{response.status_code}", resolved_id=eid)
    seq = normalize_sequence_value(response.text)
    if not seq:
        return _result(qid, "ensembl", status="unresolved", detail=f"ensembl_seq_empty:{eid}", resolved_id=eid)
    return _result(qid, "ensembl", sequence=seq, status="ensembl_fetched", detail=f"ensembl_seq:{eid}", resolved_id=eid)


def fetch_ncbi_accession(session, accession: str, *, timeout: int = 10, query_id: str | None = None) -> SequenceFetchResult:
    acc = str(accession or "").strip()
    qid = str(query_id or acc).strip()
    response, exc = _request_with_retry(
        session,
        NCBI_EFETCH.format(accession=quote(acc, safe="")),
        timeout=timeout,
    )
    if exc is not None:
        return _result(qid, "accession", status="unresolved", detail=f"ncbi_err:{exc}")
    if response.status_code != 200:
        return _result(qid, "accession", status="unresolved", detail=f"ncbi_status:{response.status_code}")
    seq = normalize_sequence_value(response.text)
    if not seq:
        return _result(qid, "accession", status="unresolved", detail=f"ncbi_empty:{acc}")
    return _result(qid, "accession", sequence=seq, status="accession_fetched", detail=f"ncbi:{acc}", resolved_id=acc)


def _dedupe_nonempty(values) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def _collect_ensembl_ids(obj, out: list[str], *, depth: int = 0, max_depth: int = 4) -> None:
    if depth > max_depth:
        return
    if isinstance(obj, str):
        value = obj.strip()
        if value.upper().startswith("ENS"):
            out.append(value)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _collect_ensembl_ids(value, out, depth=depth + 1, max_depth=max_depth)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_ensembl_ids(value, out, depth=depth + 1, max_depth=max_depth)


def _lookup_ensembl_candidates(session, ensembl_id: str, *, timeout: int = 10) -> tuple[list[str], list[str]]:
    eid = str(ensembl_id or "").strip()
    response, exc = _request_with_retry(
        session,
        ENSEMBL_LOOKUP_ID.format(ensembl_id=quote(eid, safe="")),
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    if exc is not None:
        return [], [f"ensembl_lookup_id_err:{exc}"]
    if response.status_code != 200:
        return [], [f"ensembl_lookup_id_status:{response.status_code}"]
    try:
        payload = response.json() or {}
    except Exception as exc:
        return [], [f"ensembl_lookup_id_json_err:{exc}"]

    candidates: list[str] = []
    canonical = str(payload.get("canonical_transcript") or "").strip()
    if canonical:
        candidates.append(strip_safe_version_suffix(canonical)[0])

    if eid.upper().startswith("ENSG"):
        for item in payload.get("Transcript") or payload.get("transcripts") or []:
            candidates.append(str(item.get("id") or "").strip())
    resolved_id = str(payload.get("id") or "").strip()
    if resolved_id and resolved_id != eid:
        candidates.append(resolved_id)
    return _dedupe_nonempty([cid for cid in candidates if cid != eid]), []


def _archive_ensembl_candidates(session, ensembl_id: str, *, timeout: int = 10) -> tuple[list[str], list[str]]:
    eid = str(ensembl_id or "").strip()
    response, exc = _request_with_retry(
        session,
        ENSEMBL_ARCHIVE.format(ensembl_id=quote(eid, safe="")),
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    if exc is not None:
        return [], [f"ensembl_archive_err:{exc}"]
    if response.status_code != 200:
        return [], [f"ensembl_archive_status:{response.status_code}"]
    try:
        payload = response.json() or {}
    except Exception as exc:
        return [], [f"ensembl_archive_json_err:{exc}"]

    candidates: list[str] = []
    _collect_ensembl_ids(payload, candidates)
    filtered = []
    tried = {eid, strip_safe_version_suffix(eid)[0]}
    for candidate in _dedupe_nonempty(candidates):
        if candidate not in tried:
            filtered.append(candidate)
    return filtered, []


def _resolve_ensembl_fallback_candidates(
    session,
    query_id: str,
    candidates: list[str],
    *,
    timeout: int,
    status_base: str,
    detail_prefix: str,
) -> SequenceFetchResult:
    candidate_ids = _dedupe_nonempty(candidates)
    attempt_notes: list[str] = []
    for idx, candidate_id in enumerate(candidate_ids):
        seq_result = fetch_ensembl_sequence(session, candidate_id, timeout=timeout, query_id=query_id)
        if seq_result.sequence:
            alternatives = tuple(cid for cid in candidate_ids if cid != candidate_id)
            status = f"{status_base}_first_candidate_used" if len(candidate_ids) > 1 else status_base
            detail = f"{detail_prefix}:{query_id}->{candidate_id}"
            if idx:
                detail = f"candidate_index:{idx}; {detail}"
            if alternatives:
                detail = f"{detail}; alternatives:{'|'.join(alternatives)}"
            return _result(
                query_id,
                "ensembl",
                sequence=seq_result.sequence,
                status=status,
                detail=detail,
                resolved_id=candidate_id,
                alternative_ids=alternatives,
                source=detail_prefix,
            )
        attempt_notes.append(f"{candidate_id}:{seq_result.detail}")
    detail = f"{detail_prefix}_candidates:{'|'.join(candidate_ids)}"
    if attempt_notes:
        detail = f"{detail}; {'; '.join(attempt_notes)}"
    return _result(query_id, "ensembl", status="unresolved", detail=detail)


def _ncbi_symbol_search_terms(route: IdentifierRoute, symbol: str) -> list[tuple[str, str]]:
    exact = f'"{symbol}"'
    human_rna = '"Homo sapiens"[Organism] AND biomol_rna[PROP]'
    gene_name = f"{exact}[Gene Name] AND {human_rna}"
    all_fields = f"{exact}[All Fields] AND {human_rna}"
    queries: list[tuple[str, str]] = []
    if route.family == "loc_symbol":
        queries.extend(
            [
                ("ncbi_gene_name_refseq_rna", f"{gene_name} AND srcdb_refseq[PROP]"),
                ("ncbi_gene_name_rna", gene_name),
                ("ncbi_all_fields_refseq_rna", f"{all_fields} AND srcdb_refseq[PROP]"),
                ("ncbi_all_fields_rna", all_fields),
            ]
        )
    else:
        queries.extend(
            [
                ("ncbi_gene_name_refseq_rna", f"{gene_name} AND srcdb_refseq[PROP]"),
                ("ncbi_gene_name_rna", gene_name),
                ("ncbi_all_fields_refseq_rna", f"{all_fields} AND srcdb_refseq[PROP]"),
                ("ncbi_all_fields_rna", all_fields),
            ]
        )
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, query in queries:
        if query not in seen:
            seen.add(query)
            deduped.append((label, query))
    return deduped


def resolve_symbol_via_ncbi_search(
    session,
    symbol: str,
    *,
    timeout: int = 10,
    route: IdentifierRoute | None = None,
) -> SequenceFetchResult:
    sym = str(symbol or "").strip()
    route = route or detect_id_route(sym)
    notes: list[str] = []
    candidate_ids: list[str] = []
    search_label = ""
    for label, query in _ncbi_symbol_search_terms(route, sym):
        response, exc = _request_with_retry(
            session,
            NCBI_ESEARCH.format(term=quote(query, safe=""), retmax=5),
            timeout=timeout,
        )
        if exc is not None:
            notes.append(f"{label}_err:{exc}")
            continue
        if response.status_code != 200:
            notes.append(f"{label}_status:{response.status_code}")
            continue
        try:
            payload = response.json() or {}
        except Exception as exc:
            notes.append(f"{label}_json_err:{exc}")
            continue
        id_list = payload.get("esearchresult", {}).get("idlist") or []
        candidate_ids = _dedupe_nonempty(id_list)
        if candidate_ids:
            search_label = label
            break
        notes.append(f"{label}_empty")

    if not candidate_ids:
        detail = "ncbi_symbol_search_empty"
        if notes:
            detail = f"{detail}; {'; '.join(notes)}"
        return _result(sym, "symbol", status="unresolved", detail=detail, source="ncbi_esearch_nuccore")

    attempt_notes: list[str] = []
    for idx, candidate_id in enumerate(candidate_ids):
        seq_result = fetch_ncbi_accession(session, candidate_id, timeout=timeout, query_id=sym)
        if seq_result.sequence:
            alternatives = tuple(cid for cid in candidate_ids if cid != candidate_id)
            status = "symbol_ncbi_search_first_candidate_used" if len(candidate_ids) > 1 else "symbol_ncbi_search_resolved"
            detail = f"{search_label}:{sym}->{candidate_id}"
            if idx:
                detail = f"candidate_index:{idx}; {detail}"
            if alternatives:
                detail = f"{detail}; alternatives:{'|'.join(alternatives)}"
            return _result(
                sym,
                "symbol",
                sequence=seq_result.sequence,
                status=status,
                detail=detail,
                resolved_id=candidate_id,
                alternative_ids=alternatives,
                source="ncbi_esearch_nuccore",
            )
        attempt_notes.append(f"{candidate_id}:{seq_result.detail}")

    detail = f"{search_label}:{sym}; {'; '.join(attempt_notes)}"
    return _result(sym, "symbol", status="unresolved", detail=detail, source="ncbi_esearch_nuccore")


def _xref_candidates(session, symbol: str, *, timeout: int = 10) -> tuple[list[tuple[str, str]], list[str]]:
    symbol_q = quote(symbol, safe="")
    candidates: list[tuple[str, str]] = []
    notes: list[str] = []
    seen: set[str] = set()
    for label, template in (("xrefs_symbol", ENSEMBL_XREF), ("xrefs_name", ENSEMBL_XREF_NAME)):
        response, exc = _request_with_retry(
            session,
            template.format(symbol=symbol_q),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        if exc is not None:
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
            if (eid.startswith("ENS") or typ in {"gene", "transcript"}) and eid not in seen:
                seen.add(eid)
                candidates.append((eid, label))
    return candidates, notes


def resolve_symbol(
    session,
    symbol: str,
    *,
    timeout: int = 10,
    route: IdentifierRoute | None = None,
    allow_ncbi_fallback: bool = True,
) -> SequenceFetchResult:
    sym = str(symbol or "").strip()
    route = route or detect_id_route(sym)
    symbol_q = quote(sym, safe="")
    response, exc = _request_with_retry(
        session,
        ENSEMBL_LOOKUP.format(symbol=symbol_q),
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    if exc is not None:
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
                source="lookup_symbol",
            )
        return _result(
            sym,
            "symbol",
            status="unresolved",
            detail=f"symbol_lookup_resolved:{resolved_id}; {seq_result.detail}",
            resolved_id=resolved_id,
            source="lookup_symbol",
        )
    candidates, notes = _xref_candidates(session, sym, timeout=timeout)
    if candidates:
        selected_id, source = candidates[0]
        alternative_ids = tuple(eid for eid, _ in candidates[1:])
        seq_result = fetch_ensembl_sequence(session, selected_id, timeout=timeout, query_id=sym)
        detail = f"{source}:{sym}->{selected_id}"
        if alternative_ids:
            detail = f"{detail}; alternatives:{'|'.join(alternative_ids)}"
        if seq_result.sequence:
            return _result(
                sym,
                "symbol",
                sequence=seq_result.sequence,
                status="ambiguous_symbol_first_candidate_used" if alternative_ids else "symbol_resolved",
                detail=detail,
                resolved_id=selected_id,
                alternative_ids=alternative_ids,
                source=source,
            )
        return _result(
            sym,
            "symbol",
            status="unresolved",
            detail=f"{detail}; {seq_result.detail}",
            resolved_id=selected_id,
            alternative_ids=alternative_ids,
            source=source,
        )
    detail = f"symbol_lookup_status:{response.status_code}"
    if notes:
        detail = f"{detail}; {'; '.join(notes)}"
    if allow_ncbi_fallback and route.family in {
        "loc_symbol",
        "catalog_symbol",
        "generic_symbol",
        "family_symbol",
        "cautious_clone_symbol_versioned",
    }:
        ncbi_result = resolve_symbol_via_ncbi_search(session, sym, timeout=timeout, route=route)
        if ncbi_result.sequence:
            return ncbi_result
        detail = f"{detail}; {ncbi_result.detail}"
    return _result(sym, "symbol", status="unresolved", detail=detail)


def fetch_sequence_by_id(session, raw_id: str, *, timeout: int = 10) -> SequenceFetchResult:
    route = detect_id_route(raw_id)
    rid = route.normalized_id
    if route.namespace == "empty":
        return _result(rid, "empty", status="unresolved", detail="empty_id")
    if route.namespace == "ensembl":
        result = fetch_ensembl_sequence(session, rid, timeout=timeout)
        base_id, can_retry = strip_safe_version_suffix(rid)
        if result.sequence:
            return result
        lookup_target = rid
        if can_retry and base_id != rid:
            retry = fetch_ensembl_sequence(session, base_id, timeout=timeout, query_id=rid)
            if retry.sequence:
                return _result(
                    rid,
                    "ensembl",
                    sequence=retry.sequence,
                    status="ensembl_version_retry_fetched",
                    detail=f"retry_without_version:{rid}->{base_id}",
                    resolved_id=base_id,
                )
            fallback_result = _result(
                rid,
                "ensembl",
                status="unresolved",
                detail=f"{result.detail}; retry_without_version:{rid}->{base_id}; {retry.detail}",
            )
            lookup_target = base_id
        else:
            fallback_result = result
        lookup_candidates, lookup_notes = _lookup_ensembl_candidates(session, lookup_target, timeout=timeout)
        if lookup_candidates:
            lookup_result = _resolve_ensembl_fallback_candidates(
                session,
                rid,
                lookup_candidates,
                timeout=timeout,
                status_base="ensembl_lookup_transcript_resolved",
                detail_prefix="ensembl_lookup_transcript",
            )
            if lookup_result.sequence:
                return lookup_result
            fallback_result = _result(
                rid,
                "ensembl",
                status="unresolved",
                detail=f"{fallback_result.detail}; {lookup_result.detail}",
            )
        elif lookup_notes:
            fallback_result = _result(
                rid,
                "ensembl",
                status="unresolved",
                detail=f"{fallback_result.detail}; {'; '.join(lookup_notes)}",
            )
        archive_candidates, archive_notes = _archive_ensembl_candidates(session, lookup_target, timeout=timeout)
        if archive_candidates:
            archive_result = _resolve_ensembl_fallback_candidates(
                session,
                rid,
                archive_candidates,
                timeout=timeout,
                status_base="ensembl_archive_resolved",
                detail_prefix="ensembl_archive",
            )
            if archive_result.sequence:
                return archive_result
            return _result(
                rid,
                "ensembl",
                status="unresolved",
                detail=f"{fallback_result.detail}; {archive_result.detail}",
            )
        if archive_notes:
            return _result(
                rid,
                "ensembl",
                status="unresolved",
                detail=f"{fallback_result.detail}; {'; '.join(archive_notes)}",
            )
        return fallback_result
    if route.namespace == "ncbi_nuccore":
        result = fetch_ncbi_accession(session, rid, timeout=timeout)
        base_id, can_retry = strip_safe_version_suffix(rid)
        if result.sequence or not can_retry or base_id == rid:
            return result
        retry = fetch_ncbi_accession(session, base_id, timeout=timeout, query_id=rid)
        if retry.sequence:
            return _result(
                rid,
                "accession",
                sequence=retry.sequence,
                status="accession_version_retry_fetched",
                detail=f"retry_without_version:{rid}->{base_id}",
                resolved_id=base_id,
            )
        return _result(
            rid,
            "accession",
            status="unresolved",
            detail=f"{result.detail}; retry_without_version:{rid}->{base_id}; {retry.detail}",
        )
    if route.namespace == "ncbi_symbol_search":
        result = resolve_symbol_via_ncbi_search(session, rid, timeout=timeout, route=route)
        if result.sequence:
            return result
        variants = _symbol_route_variants(route)
        if not variants:
            return result
        variant_result = _retry_symbol_variants(session, route, result, timeout=timeout, seen={rid})
        return variant_result if variant_result is not None else result
    return _fetch_symbol_like_sequence(session, route, timeout=timeout, seen={rid})


def _fetch_symbol_like_sequence(session, route: IdentifierRoute, *, timeout: int, seen: set[str]) -> SequenceFetchResult:
    rid = route.normalized_id
    result = resolve_symbol(session, rid, timeout=timeout, route=route)
    variants = _symbol_route_variants(route)
    if result.sequence or not variants:
        variant_result = _retry_symbol_variants(session, route, result, timeout=timeout, seen=seen)
        return variant_result if variant_result is not None else result
    next_variant = variants[0]
    next_route = detect_id_route(next_variant)
    retry = _fetch_symbol_like_sequence(session, next_route, timeout=timeout, seen=seen | {next_variant})
    if retry.sequence:
        retry_label = "retry_without_version" if route.family == "cautious_clone_symbol_versioned" else "retry_variant"
        return _result(
            rid,
            "symbol",
            sequence=retry.sequence,
            status="symbol_version_retry_resolved" if route.family == "cautious_clone_symbol_versioned" else "symbol_variant_retry_resolved",
            detail=f"{retry_label}:{rid}->{next_variant}; {retry.detail}",
            resolved_id=retry.resolved_id,
            alternative_ids=retry.alternative_ids,
            source=retry.source,
        )
    variant_result = _retry_symbol_variants(session, route, result, timeout=timeout, seen=seen | {next_variant})
    if variant_result is not None:
        return variant_result
    retry_label = "retry_without_version" if route.family == "cautious_clone_symbol_versioned" else "retry_variant"
    return _result(
        rid,
        "symbol",
        status="unresolved",
        detail=f"{result.detail}; {retry_label}:{rid}->{next_variant}; {retry.detail}",
        resolved_id=retry.resolved_id,
        alternative_ids=retry.alternative_ids,
        source=retry.source,
    )


def _retry_symbol_variants(session, route: IdentifierRoute, result: SequenceFetchResult, *, timeout: int, seen: set[str]) -> SequenceFetchResult | None:
    rid = route.normalized_id
    for variant in _symbol_route_variants(route):
        if variant in seen:
            continue
        retry = _fetch_symbol_like_sequence(session, detect_id_route(variant), timeout=timeout, seen=seen | {variant})
        if retry.sequence:
            return _result(
                rid,
                "symbol",
                sequence=retry.sequence,
                status="symbol_variant_retry_resolved",
                detail=f"retry_variant:{rid}->{variant}; {retry.detail}",
                resolved_id=retry.resolved_id,
                alternative_ids=retry.alternative_ids,
                source=retry.source,
            )
    return None


def needs_manual_review(result: SequenceFetchResult) -> bool:
    return result.status == "unresolved"
