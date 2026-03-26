import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mainfolder.utils.sequence_fetch import (
    ENSEMBL_LOOKUP,
    ENSEMBL_SEQ,
    ENSEMBL_XREF,
    ENSEMBL_XREF_NAME,
    NCBI_EFETCH,
    detect_id_kind,
    fetch_sequence_by_id,
)


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
        return self.responses[url]


def test_detect_id_kind_classifies_expected_cases():
    assert detect_id_kind("ENSG00000251562") == "ensembl"
    assert detect_id_kind("DQ600483") == "accession"
    assert detect_id_kind("NR_024031") == "accession"
    assert detect_id_kind("DLEU1") == "symbol"
    assert detect_id_kind("AC002480.5") == "symbol"


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
    session = FakeSession(
        {
            ENSEMBL_LOOKUP.format(symbol=symbol): FakeResponse(status_code=404, json_data={}),
            ENSEMBL_XREF.format(symbol=symbol): FakeResponse(json_data=[]),
            ENSEMBL_XREF_NAME.format(symbol=symbol): FakeResponse(json_data=[]),
        }
    )
    result = fetch_sequence_by_id(session, symbol)
    assert result.status == "unresolved"
    assert result.sequence == ""
