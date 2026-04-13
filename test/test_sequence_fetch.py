import sys
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mainfolder.utils.sequence_fetch import (
    ENSEMBL_LOOKUP,
    ENSEMBL_LOOKUP_ID,
    ENSEMBL_SEQ,
    ENSEMBL_XREF,
    ENSEMBL_XREF_NAME,
    ENSEMBL_ARCHIVE,
    NCBI_EFETCH,
    NCBI_ESEARCH,
    NCBI_GENE_ESEARCH,
    NCBI_GENE_EFETCH,
    detect_id_kind,
    detect_id_route,
    extract_alias_candidates,
    fetch_sequence_by_id,
    normalize_sequence_value,
)
from mainfolder.utils.sequence_overrides import SequenceOverride


class FakeResponse:
    def __init__(self, status_code=200, text="", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    def json(self):
        if self._json_data is None:
            raise ValueError("no json payload")
        return self._json_data


class FakeSession:
    def __init__(self, responses):
        self.responses = responses

    def get(self, url, headers=None, timeout=None):
        if url not in self.responses:
            raise AssertionError(f"unexpected url: {url}")
        value = self.responses[url]
        if isinstance(value, list):
            if not value:
                raise AssertionError(f"no more queued responses for url: {url}")
            return value.pop(0)
        return value


def ncbi_search_url(term: str, retmax: int = 5) -> str:
    return NCBI_ESEARCH.format(term=quote(term, safe=""), retmax=retmax)


def ncbi_gene_search_url(term: str, retmax: int = 5) -> str:
    return NCBI_GENE_ESEARCH.format(term=quote(term, safe=""), retmax=retmax)


def ncbi_symbol_search_urls(symbol: str, *, loc: bool = False) -> list[str]:
    exact = f'"{symbol}"'
    human_rna = '"Homo sapiens"[Organism] AND biomol_rna[PROP]'
    gene_name = f"{exact}[Gene Name] AND {human_rna}"
    all_fields = f"{exact}[All Fields] AND {human_rna}"
    queries = [
        f"{gene_name} AND srcdb_refseq[PROP]",
        gene_name,
        f"{all_fields} AND srcdb_refseq[PROP]",
        all_fields,
    ]
    return [ncbi_search_url(query) for query in queries]


def ncbi_gene_lookup_urls(symbol: str, *, loc: bool = False) -> list[str]:
    exact = f'"{symbol}"'
    queries = [f'{exact}[sym] AND "Homo sapiens"[orgn]']
    if loc:
        queries.append(f'{exact}[Preferred Symbol] AND "Homo sapiens"[orgn]')
    queries.append(f'{exact}[All Fields] AND "Homo sapiens"[orgn]')
    return [ncbi_gene_search_url(query) for query in queries]


def test_detect_id_kind_classifies_expected_cases():
    assert detect_id_kind("ENSG00000251562") == "ensembl"
    assert detect_id_kind("DQ600483") == "accession"
    assert detect_id_kind("NR_024031") == "accession"
    assert detect_id_kind("DLEU1") == "symbol"
    assert detect_id_kind("AC002480.5") == "symbol"


def test_detect_id_route_assigns_expected_families_and_namespaces():
    ensembl = detect_id_route("ENSG00000251562.5")
    assert ensembl.family == "ensembl_versioned"
    assert ensembl.namespace == "ensembl"
    assert ensembl.id_type == "ensembl"

    refseq = detect_id_route("NR_024031.1")
    assert refseq.family == "refseq_versioned"
    assert refseq.namespace == "ncbi_nuccore"
    assert refseq.id_type == "accession"

    cautious = detect_id_route("AC002480.5")
    assert cautious.family == "cautious_clone_symbol_versioned"
    assert cautious.namespace == "ensembl_symbol"
    assert cautious.id_type == "symbol"

    loc = detect_id_route("LOC100127888")
    assert loc.family == "loc_symbol"
    assert loc.namespace == "ncbi_symbol_search"
    assert loc.id_type == "symbol"

    catalog = detect_id_route("lnc-ABCC5-2:1")
    assert catalog.family == "catalog_symbol"
    assert catalog.namespace == "ensembl_symbol"
    assert catalog.id_type == "symbol"

    family = detect_id_route("E2F")
    assert family.family == "family_symbol"
    assert family.namespace == "ensembl_symbol"
    assert family.id_type == "symbol"


def test_fetch_sequence_by_id_fetches_ensembl_ids_directly():
    ensembl_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_SEQ.format(ensembl_id=ensembl_id): FakeResponse(text="AUGCUU"),
        }
    )
    result = fetch_sequence_by_id(session, ensembl_id)
    assert result.status == "ensembl_fetched"
    assert result.sequence == "AUGCUU"
    assert result.resolved_id == ensembl_id


def test_fetch_sequence_by_id_retries_versioned_ensembl_ids_without_suffix():
    ensembl_id = "ENSG00000251562.5"
    base_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_SEQ.format(ensembl_id=ensembl_id): FakeResponse(status_code=404, text="not found"),
            ENSEMBL_SEQ.format(ensembl_id=base_id): FakeResponse(text="AUGCUU"),
        }
    )
    result = fetch_sequence_by_id(session, ensembl_id)
    assert result.status == "ensembl_version_retry_fetched"
    assert result.sequence == "AUGCUU"
    assert result.resolved_id == base_id
    assert result.detail == f"retry_without_version:{ensembl_id}->{base_id}"


def test_fetch_sequence_by_id_fetches_accessions_from_ncbi():
    accession = "NR_024031"
    session = FakeSession(
        {
            NCBI_EFETCH.format(accession=accession): FakeResponse(text=">x\nAUGC\nUU"),
        }
    )
    result = fetch_sequence_by_id(session, accession)
    assert result.status == "accession_fetched"
    assert result.sequence == "AUGCUU"
    assert result.resolved_id == accession


def test_fetch_sequence_by_id_retries_versioned_refseq_ids_without_suffix():
    accession = "NR_024031.1"
    base_id = "NR_024031"
    session = FakeSession(
        {
            NCBI_EFETCH.format(accession=accession): FakeResponse(status_code=404, text="not found"),
            NCBI_EFETCH.format(accession=base_id): FakeResponse(text=">x\nAUGC\nUU"),
        }
    )
    result = fetch_sequence_by_id(session, accession)
    assert result.status == "accession_version_retry_fetched"
    assert result.sequence == "AUGCUU"
    assert result.resolved_id == base_id
    assert result.detail == f"retry_without_version:{accession}->{base_id}"


def test_fetch_sequence_by_id_rejects_ncbi_error_payload():
    accession = "NR_024031"
    session = FakeSession(
        {
            NCBI_EFETCH.format(accession=accession): FakeResponse(text="Error: accession not found"),
        }
    )
    result = fetch_sequence_by_id(session, accession)
    assert result.status == "unresolved"
    assert result.sequence == ""


def test_normalize_sequence_value_rejects_cached_error_payloads():
    assert normalize_sequence_value("ERROR: F A I L E D TO UNDERSTAND ID") == ""
    assert normalize_sequence_value(">id\nA U G C") == "AUGC"


def test_extract_alias_candidates_handles_wrappers_and_description_aliases():
    assert extract_alias_candidates("lnc-ADD3-AS1", []) == ["ADD3-AS1"]
    assert extract_alias_candidates("LICN00520", []) == ["LINC00520"]
    assert extract_alias_candidates(
        "HMGA1-lnc",
        ["We identified the lncRNA RP11.513I15.6, which we refer to as HMGA1-lnc."],
    ) == ["RP11.513I15.6"]
    assert extract_alias_candidates(
        "LncKLHDC7B",
        ["We discovered that LncKLHDC7B (ENSG00000226738) acts as a transcriptional modulator."],
    ) == ["ENSG00000226738"]
    assert extract_alias_candidates(
        "LINC0638",
        ["LINC01638 lncRNA was significantly upregulated in melanoma."],
    ) == ["LINC01638"]
    assert extract_alias_candidates(
        "DLEUI",
        ["DLEU1 levels were higher in PTC cell lines than in controls."],
    ) == ["DLEU1"]


def test_fetch_sequence_by_id_rejects_ensembl_error_payload():
    ensembl_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_SEQ.format(ensembl_id=ensembl_id): FakeResponse(text="No sequence available"),
        }
    )
    result = fetch_sequence_by_id(session, ensembl_id)
    assert result.status == "unresolved"
    assert result.sequence == ""


def test_fetch_sequence_by_id_retries_transient_ensembl_sequence_errors():
    ensembl_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_SEQ.format(ensembl_id=ensembl_id): [
                FakeResponse(status_code=500, text="server error"),
                FakeResponse(text="AUGC"),
            ],
        }
    )
    result = fetch_sequence_by_id(session, ensembl_id)
    assert result.status == "ensembl_fetched"
    assert result.sequence == "AUGC"
    assert result.resolved_id == ensembl_id


def test_fetch_sequence_by_id_resolves_symbols_through_ensembl_lookup():
    symbol = "MALAT1"
    resolved_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=symbol): FakeResponse(json_data={"id": resolved_id}),
            ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol)
    assert result.status == "symbol_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id


def test_fetch_sequence_by_id_treats_ac_style_ids_as_symbol_like():
    symbol_like = "AC002480.5"
    resolved_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=symbol_like): FakeResponse(json_data={"id": resolved_id}),
            ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol_like)
    assert result.status == "symbol_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id


def test_fetch_sequence_by_id_uses_alias_candidates_after_primary_failure():
    symbol = "HMGA1-lnc"
    alias = "RP11.513I15.6"
    resolved_id = "ENSG00000251562"
    responses = {
        ENSEMBL_LOOKUP.format(symbol=symbol): FakeResponse(status_code=404, json_data={}),
        ENSEMBL_XREF.format(symbol=symbol): FakeResponse(json_data=[]),
        ENSEMBL_XREF_NAME.format(symbol=symbol): FakeResponse(json_data=[]),
        ENSEMBL_LOOKUP.format(symbol=alias): FakeResponse(json_data={"id": resolved_id}),
        ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
    }
    for url in ncbi_gene_lookup_urls(symbol):
        responses[url] = FakeResponse(json_data={"esearchresult": {"idlist": []}})
    for url in ncbi_symbol_search_urls(symbol):
        responses[url] = FakeResponse(json_data={"esearchresult": {"idlist": []}})
    session = FakeSession(responses)
    result = fetch_sequence_by_id(
        session,
        symbol,
        alias_candidates=[alias],
    )
    assert result.status == "alias_candidate_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id
    assert result.detail == f"alias_candidate:{symbol}->{alias}; symbol_lookup:{alias}->{resolved_id}"


def test_fetch_sequence_by_id_uses_direct_sequence_override():
    symbol = "NONHSAT000612.2"
    result = fetch_sequence_by_id(
        FakeSession({}),
        symbol,
        overrides={
            symbol: SequenceOverride(
                query_id=symbol,
                sequence="AUGCUU",
                source="manual_override",
                notes="validated externally",
            )
        },
    )
    assert result.status == "override_sequence_resolved"
    assert result.sequence == "AUGCUU"
    assert result.resolved_id == symbol


def test_fetch_sequence_by_id_uses_resolved_id_override():
    symbol = "LNCV6_100111_PI430048170"
    target = "NR_024031"
    session = FakeSession(
        {
            NCBI_EFETCH.format(accession=target): FakeResponse(text=">x\nAUGCUU"),
        }
    )
    result = fetch_sequence_by_id(
        session,
        symbol,
        overrides={
            symbol: SequenceOverride(
                query_id=symbol,
                resolved_id=target,
                source="manual_override",
            )
        },
    )
    assert result.status == "override_resolved_id_fetched"
    assert result.sequence == "AUGCUU"
    assert result.resolved_id == target


def test_fetch_sequence_by_id_retries_transient_symbol_lookup_errors():
    symbol = "MALAT1"
    resolved_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=symbol): [
                FakeResponse(status_code=500, json_data={}),
                FakeResponse(json_data={"id": resolved_id}),
            ],
            ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol)
    assert result.status == "symbol_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id


def test_fetch_sequence_by_id_retries_cautious_symbol_without_suffix():
    symbol_like = "AC002480.5"
    base_symbol = "AC002480"
    resolved_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=symbol_like): FakeResponse(status_code=404, json_data={}),
            ENSEMBL_XREF.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_XREF_NAME.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_LOOKUP.format(symbol=base_symbol): FakeResponse(json_data={"id": resolved_id}),
            ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol_like)
    assert result.status == "symbol_version_retry_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id
    assert result.detail == f"retry_without_version:{symbol_like}->{base_symbol}; symbol_lookup:{base_symbol}->{resolved_id}"


def test_fetch_sequence_by_id_retries_symbol_variants_with_transcript_suffix():
    symbol_like = "AC084816.1-205"
    versioned_symbol = "AC084816.1"
    base_symbol = "AC084816"
    resolved_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=symbol_like): FakeResponse(status_code=400, json_data={}),
            ENSEMBL_XREF.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_XREF_NAME.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_LOOKUP.format(symbol=versioned_symbol): FakeResponse(status_code=404, json_data={}),
            ENSEMBL_XREF.format(symbol=versioned_symbol): FakeResponse(json_data=[]),
            ENSEMBL_XREF_NAME.format(symbol=versioned_symbol): FakeResponse(json_data=[]),
            ENSEMBL_LOOKUP.format(symbol=base_symbol): FakeResponse(json_data={"id": resolved_id}),
            ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol_like)
    assert result.status == "symbol_variant_retry_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id
    assert result.detail == (
        f"retry_variant:{symbol_like}->{versioned_symbol}; "
        f"retry_without_version:{versioned_symbol}->{base_symbol}; symbol_lookup:{base_symbol}->{resolved_id}"
    )


def test_fetch_sequence_by_id_retries_symbol_variants_with_colon_suffix():
    symbol_like = "lnc-ABCC5-2:1"
    base_symbol = "lnc-ABCC5-2"
    resolved_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=quote(symbol_like, safe="")): FakeResponse(status_code=400, json_data={}),
            ENSEMBL_XREF.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_XREF_NAME.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_LOOKUP.format(symbol=base_symbol): FakeResponse(json_data={"id": resolved_id}),
            ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol_like)
    assert result.status == "symbol_variant_retry_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id
    assert result.detail == f"retry_variant:{symbol_like}->{base_symbol}; symbol_lookup:{base_symbol}->{resolved_id}"


def test_fetch_sequence_by_id_retries_symbol_variants_with_small_dash_suffix():
    symbol_like = "CASC9-1"
    base_symbol = "CASC9"
    resolved_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=symbol_like): FakeResponse(status_code=400, json_data={}),
            ENSEMBL_XREF.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_XREF_NAME.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_LOOKUP.format(symbol=base_symbol): FakeResponse(json_data={"id": resolved_id}),
            ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol_like)
    assert result.status == "symbol_variant_retry_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id
    assert result.detail == f"retry_variant:{symbol_like}->{base_symbol}; symbol_lookup:{base_symbol}->{resolved_id}"


def test_fetch_sequence_by_id_retries_symbol_variants_with_small_dash_and_colon_suffix():
    symbol_like = "LNC-MANSC4-8:1"
    base_symbol = "LNC-MANSC4"
    resolved_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=quote(symbol_like, safe="")): FakeResponse(status_code=400, json_data={}),
            ENSEMBL_XREF.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_XREF_NAME.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_LOOKUP.format(symbol=base_symbol): FakeResponse(json_data={"id": resolved_id}),
            ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol_like)
    assert result.status == "symbol_variant_retry_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id
    assert result.detail == f"retry_variant:{symbol_like}->{base_symbol}; symbol_lookup:{base_symbol}->{resolved_id}"


def test_fetch_sequence_by_id_retries_symbol_variants_with_bare_v_suffix():
    symbol_like = "DMTF1v4"
    base_symbol = "DMTF1"
    resolved_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=symbol_like): FakeResponse(status_code=400, json_data={}),
            ENSEMBL_XREF.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_XREF_NAME.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_LOOKUP.format(symbol=base_symbol): FakeResponse(json_data={"id": resolved_id}),
            ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol_like)
    assert result.status == "symbol_variant_retry_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id
    assert result.detail == f"retry_variant:{symbol_like}->{base_symbol}; symbol_lookup:{base_symbol}->{resolved_id}"


def test_fetch_sequence_by_id_retries_compact_antisense_symbol_variants():
    symbol_like = "E2F4as"
    base_symbol = "E2F4-AS1"
    resolved_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=symbol_like): FakeResponse(status_code=400, json_data={}),
            ENSEMBL_XREF.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_XREF_NAME.format(symbol=symbol_like): FakeResponse(json_data=[]),
            ENSEMBL_LOOKUP.format(symbol=base_symbol): FakeResponse(json_data={"id": resolved_id}),
            ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol_like)
    assert result.status == "symbol_variant_retry_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id
    assert result.detail == f"retry_variant:{symbol_like}->{base_symbol}; symbol_lookup:{base_symbol}->{resolved_id}"


def test_fetch_sequence_by_id_does_not_retry_without_suffix_for_ac_style_symbols():
    symbol_like = "AC002480.5"
    responses = {
        ENSEMBL_LOOKUP.format(symbol=symbol_like): FakeResponse(status_code=404, json_data={}),
        ENSEMBL_XREF.format(symbol=symbol_like): FakeResponse(json_data=[]),
        ENSEMBL_XREF_NAME.format(symbol=symbol_like): FakeResponse(json_data=[]),
    }
    for url in ncbi_gene_lookup_urls(symbol_like):
        responses[url] = FakeResponse(json_data={"esearchresult": {"idlist": []}})
    for url in ncbi_symbol_search_urls(symbol_like):
        responses[url] = FakeResponse(json_data={"esearchresult": {"idlist": []}})
    session = FakeSession(responses)
    result = fetch_sequence_by_id(session, symbol_like)
    assert result.status == "unresolved"
    assert result.sequence == ""


def test_fetch_sequence_by_id_does_not_retry_without_suffix_for_non_cautious_symbols():
    symbol_like = "MSTRG.255299"
    responses = {
        ENSEMBL_LOOKUP.format(symbol=symbol_like): FakeResponse(status_code=404, json_data={}),
        ENSEMBL_XREF.format(symbol=symbol_like): FakeResponse(json_data=[]),
        ENSEMBL_XREF_NAME.format(symbol=symbol_like): FakeResponse(json_data=[]),
    }
    for url in ncbi_gene_lookup_urls(symbol_like):
        responses[url] = FakeResponse(json_data={"esearchresult": {"idlist": []}})
    for url in ncbi_symbol_search_urls(symbol_like):
        responses[url] = FakeResponse(json_data={"esearchresult": {"idlist": []}})
    session = FakeSession(responses)
    result = fetch_sequence_by_id(session, symbol_like)
    assert result.status == "unresolved"
    assert result.sequence == ""


def test_fetch_sequence_by_id_uses_first_xref_candidate_for_ambiguous_symbols():
    symbol = "E2F"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=symbol): FakeResponse(status_code=404, json_data={}),
            ENSEMBL_XREF.format(symbol=symbol): FakeResponse(
                json_data=[
                    {"id": "ENSG00000101412", "type": "gene"},
                    {"id": "ENSG00000112242", "type": "gene"},
                ]
            ),
            ENSEMBL_XREF_NAME.format(symbol=symbol): FakeResponse(json_data=[]),
            ENSEMBL_SEQ.format(ensembl_id="ENSG00000101412"): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol)
    assert result.status == "ambiguous_symbol_first_candidate_used"
    assert result.sequence == "AUGC"
    assert result.resolved_id == "ENSG00000101412"
    assert result.alternative_ids == ("ENSG00000112242",)
    assert result.source == "xrefs_symbol"


def test_fetch_sequence_by_id_returns_primary_symbol_resolution_without_variant_retries():
    symbol = "lnc-ABCC5-2:1"
    resolved_id = "ENSG00000251562"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=quote(symbol, safe="")): FakeResponse(json_data={"id": resolved_id}),
            ENSEMBL_SEQ.format(ensembl_id=resolved_id): FakeResponse(text="AUGC"),
        }
    )
    result = fetch_sequence_by_id(session, symbol)
    assert result.status == "symbol_resolved"
    assert result.sequence == "AUGC"
    assert result.resolved_id == resolved_id


def test_fetch_sequence_by_id_keeps_alternatives_when_first_candidate_fails():
    symbol = "E2F"
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=symbol): FakeResponse(status_code=404, json_data={}),
            ENSEMBL_XREF.format(symbol=symbol): FakeResponse(
                json_data=[
                    {"id": "ENSG00000101412", "type": "gene"},
                    {"id": "ENSG00000112242", "type": "gene"},
                ]
            ),
            ENSEMBL_XREF_NAME.format(symbol=symbol): FakeResponse(json_data=[]),
            ENSEMBL_SEQ.format(ensembl_id="ENSG00000101412"): FakeResponse(text="No sequence available"),
        }
    )
    result = fetch_sequence_by_id(session, symbol)
    assert result.status == "unresolved"
    assert result.sequence == ""
    assert result.resolved_id == "ENSG00000101412"
    assert result.alternative_ids == ("ENSG00000112242",)


def test_fetch_sequence_by_id_marks_missing_symbols_unresolved():
    symbol = "NOT_A_REAL_SYMBOL"
    responses = {
        ENSEMBL_LOOKUP.format(symbol=symbol): FakeResponse(status_code=404, json_data={}),
        ENSEMBL_XREF.format(symbol=symbol): FakeResponse(json_data=[]),
        ENSEMBL_XREF_NAME.format(symbol=symbol): FakeResponse(json_data=[]),
    }
    for url in ncbi_gene_lookup_urls(symbol):
        responses[url] = FakeResponse(json_data={"esearchresult": {"idlist": []}})
    for url in ncbi_symbol_search_urls(symbol):
        responses[url] = FakeResponse(json_data={"esearchresult": {"idlist": []}})
    session = FakeSession(responses)
    result = fetch_sequence_by_id(session, symbol)
    assert result.status == "unresolved"
    assert result.sequence == ""


def test_fetch_sequence_by_id_resolves_loc_symbols_via_ncbi_search():
    symbol = "LOC100127888"
    nuccore_id = "123456"
    responses = {
        ncbi_gene_search_url('"LOC100127888"[sym] AND "Homo sapiens"[orgn]'): FakeResponse(
            json_data={"esearchresult": {"idlist": ["999"]}}
        ),
        NCBI_GENE_EFETCH.format(gene_id="999"): FakeResponse(text="<Gene-commentary_accession>NR_024031.1</Gene-commentary_accession>"),
        NCBI_EFETCH.format(accession="NR_024031.1"): FakeResponse(text=">x\nAUGCUU"),
    }
    session = FakeSession(responses)
    result = fetch_sequence_by_id(session, symbol)
    assert result.status == "symbol_ncbi_gene_resolved"
    assert result.sequence == "AUGCUU"
    assert result.resolved_id == "NR_024031.1"


def test_fetch_sequence_by_id_falls_back_to_nuccore_search_when_gene_lookup_has_no_rna():
    symbol = "LOC100127888"
    nuccore_id = "123456"
    responses = {
        ncbi_gene_search_url('"LOC100127888"[sym] AND "Homo sapiens"[orgn]'): FakeResponse(
            json_data={"esearchresult": {"idlist": ["999"]}}
        ),
        NCBI_GENE_EFETCH.format(gene_id="999"): FakeResponse(text="<Entrezgene></Entrezgene>"),
        ncbi_search_url(
            '"LOC100127888"[Gene Name] AND "Homo sapiens"[Organism] AND biomol_rna[PROP] AND srcdb_refseq[PROP]'
        ): FakeResponse(json_data={"esearchresult": {"idlist": [nuccore_id]}}),
        NCBI_EFETCH.format(accession=nuccore_id): FakeResponse(text=">x\nAUGCUU"),
    }
    session = FakeSession(responses)
    result = fetch_sequence_by_id(session, symbol)
    assert result.status == "symbol_ncbi_search_resolved"
    assert result.sequence == "AUGCUU"
    assert result.resolved_id == nuccore_id


def test_fetch_sequence_by_id_uses_ensembl_lookup_transcript_fallback():
    gene_id = "ENSG00000230971"
    transcript_id = "ENST00000456789"
    session = FakeSession(
        {
            ENSEMBL_SEQ.format(ensembl_id=gene_id): FakeResponse(status_code=400, text="bad request"),
            ENSEMBL_LOOKUP_ID.format(ensembl_id=gene_id): FakeResponse(
                json_data={
                    "id": gene_id,
                    "canonical_transcript": f"{transcript_id}.2",
                    "Transcript": [{"id": transcript_id}],
                }
            ),
            ENSEMBL_SEQ.format(ensembl_id=transcript_id): FakeResponse(text="AUGCUU"),
        }
    )
    result = fetch_sequence_by_id(session, gene_id)
    assert result.status == "ensembl_lookup_transcript_resolved"
    assert result.sequence == "AUGCUU"
    assert result.resolved_id == transcript_id


def test_fetch_sequence_by_id_uses_ensembl_archive_fallback():
    transcript_id = "ENST00000412204.1"
    base_id = "ENST00000412204"
    replacement_id = "ENST00000622222"
    session = FakeSession(
        {
            ENSEMBL_SEQ.format(ensembl_id=transcript_id): FakeResponse(status_code=400, text="bad request"),
            ENSEMBL_SEQ.format(ensembl_id=base_id): FakeResponse(status_code=400, text="bad request"),
            ENSEMBL_LOOKUP_ID.format(ensembl_id=base_id): FakeResponse(status_code=404, json_data={}),
            ENSEMBL_ARCHIVE.format(ensembl_id=base_id): FakeResponse(json_data={"id": base_id, "possible_replacement": replacement_id}),
            ENSEMBL_SEQ.format(ensembl_id=replacement_id): FakeResponse(text="AUGCUU"),
        }
    )
    result = fetch_sequence_by_id(session, transcript_id)
    assert result.status == "ensembl_archive_resolved"
    assert result.sequence == "AUGCUU"
    assert result.resolved_id == replacement_id
