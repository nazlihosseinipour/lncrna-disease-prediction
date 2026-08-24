"""Replace cached Ensembl gene-region sequences with canonical RNA transcripts.

The historical V2 cache called ``sequence/id`` with ENSG identifiers, whose
default response is genomic sequence.  This script resolves those genes to
canonical ENST identifiers in batches, fetches cDNA, and writes a new input tree
without modifying the historical data or server results.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mainfolder.utils.sequence_fetch import normalize_sequence_value, strip_safe_version_suffix


LOOKUP_URL = "https://rest.ensembl.org/lookup/id"
SEQUENCE_URL = "https://rest.ensembl.org/sequence/id?type=cdna"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


def batched(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def post_json(
    session: requests.Session,
    url: str,
    ids: list[str],
    *,
    timeout: int,
    retries: int,
):
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = session.post(url, headers=HEADERS, json={"ids": ids}, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:160]}"
        except Exception as exc:  # network failures are recorded and retried
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(f"Ensembl request failed after {retries + 1} attempts: {last_error}")


def resolve_gene_transcripts(
    session: requests.Session,
    gene_ids: list[str],
    *,
    batch_size: int,
    timeout: int,
    retries: int,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if checkpoint_path is not None and checkpoint_path.exists():
        rows = pd.read_csv(checkpoint_path, dtype=str).fillna("").to_dict("records")
    completed = {row["gene_id"] for row in rows}
    pending = [gene_id for gene_id in gene_ids if gene_id not in completed]
    batches = list(batched(pending, batch_size))
    for number, batch in enumerate(batches, start=1):
        payload = post_json(
            session, LOOKUP_URL, batch, timeout=timeout, retries=retries
        )
        for gene_id in batch:
            record = payload.get(gene_id) or {}
            transcript = strip_safe_version_suffix(
                str(record.get("canonical_transcript") or "").strip()
            )[0]
            rows.append(
                {
                    "gene_id": gene_id,
                    "transcript_id": transcript,
                    "lookup_status": "resolved" if transcript else "missing_canonical_transcript",
                }
            )
        if checkpoint_path is not None:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        print(f"[lookup] batch {number}/{len(batches)} ({len(batch)} genes)", flush=True)
    result = pd.DataFrame(rows)
    return result[result["gene_id"].isin(gene_ids)].drop_duplicates("gene_id")


def fetch_transcript_sequences(
    session: requests.Session,
    transcript_ids: list[str],
    *,
    batch_size: int,
    timeout: int,
    retries: int,
    workers: int = 1,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if checkpoint_path is not None and checkpoint_path.exists():
        rows = pd.read_csv(checkpoint_path, dtype=str).fillna("").to_dict("records")
    completed = {row["transcript_id"] for row in rows}
    pending = [transcript_id for transcript_id in transcript_ids if transcript_id not in completed]
    batches = list(batched(pending, batch_size))

    def fetch_batch(batch: list[str]):
        request_session = session if workers == 1 else requests.Session()
        try:
            return post_json(
                request_session,
                SEQUENCE_URL,
                batch,
                timeout=timeout,
                retries=retries,
            )
        finally:
            if request_session is not session:
                request_session.close()

    if workers == 1:
        fetched_batches = (
            (number, batch, fetch_batch(batch))
            for number, batch in enumerate(batches, start=1)
        )
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {
            executor.submit(fetch_batch, batch): (number, batch)
            for number, batch in enumerate(batches, start=1)
        }
        fetched_batches = (
            (*futures[future], future.result()) for future in as_completed(futures)
        )

    try:
        for completed_number, (number, batch, payload) in enumerate(
            fetched_batches, start=1
        ):
            by_id = {
                str(record.get("id") or record.get("query") or "").split(".")[0]: record
                for record in payload
                if isinstance(record, dict)
            }
            for transcript_id in batch:
                record = by_id.get(transcript_id) or {}
                sequence = normalize_sequence_value(record.get("seq") or "")
                rows.append(
                    {
                        "transcript_id": transcript_id,
                        "seqs": sequence,
                        "sequence_status": "resolved" if sequence else "missing_sequence",
                    }
                )
            if checkpoint_path is not None:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
            print(
                f"[sequence] completed {completed_number}/{len(batches)} "
                f"(request {number}, {len(batch)} transcripts)",
                flush=True,
            )
    finally:
        if workers > 1:
            executor.shutdown(wait=False, cancel_futures=True)
    result = pd.DataFrame(rows)
    return result[result["transcript_id"].isin(transcript_ids)].drop_duplicates(
        "transcript_id"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-cache", default="Data/output_data/website_sequences.csv"
    )
    parser.add_argument(
        "--full-matrix", default="Data/output_data/website_full_matrix.csv"
    )
    parser.add_argument(
        "--fetch-report", default="Data/output_data/sequence_fetch_report.csv"
    )
    parser.add_argument(
        "--outdir", default="Data/corrected_inputs/v2_transcript_sequences"
    )
    parser.add_argument("--lookup-batch-size", type=int, default=200)
    parser.add_argument("--sequence-batch-size", type=int, default=40)
    parser.add_argument("--sequence-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sequence_cache = pd.read_csv(PROJECT_ROOT / args.sequence_cache, dtype=str).fillna("")
    full = pd.read_csv(PROJECT_ROOT / args.full_matrix, dtype=str).fillna("")
    report = pd.read_csv(PROJECT_ROOT / args.fetch_report, dtype=str).fillna("")
    for frame in (sequence_cache, full, report):
        frame["ID"] = frame["ID"].astype(str).str.strip()

    gene_rows = report[report["resolved_id"].str.upper().str.startswith("ENSG")].copy()
    gene_rows["gene_id"] = gene_rows["resolved_id"].map(
        lambda value: strip_safe_version_suffix(value)[0]
    )
    gene_ids = gene_rows["gene_id"].drop_duplicates().tolist()
    if args.limit is not None:
        gene_ids = gene_ids[: args.limit]
        gene_rows = gene_rows[gene_rows["gene_id"].isin(gene_ids)]

    session = requests.Session()
    lookup = resolve_gene_transcripts(
        session,
        gene_ids,
        batch_size=args.lookup_batch_size,
        timeout=args.timeout,
        retries=args.retries,
        checkpoint_path=outdir / "gene_to_canonical_transcript.checkpoint.csv",
    )
    transcript_ids = (
        lookup.loc[lookup["transcript_id"].ne(""), "transcript_id"]
        .drop_duplicates()
        .tolist()
    )
    sequences = fetch_transcript_sequences(
        session,
        transcript_ids,
        batch_size=args.sequence_batch_size,
        timeout=args.timeout,
        retries=args.retries,
        workers=max(1, args.sequence_workers),
        checkpoint_path=outdir / "transcript_sequences.checkpoint.csv",
    )

    resolution = gene_rows[["ID", "gene_id"]].merge(lookup, on="gene_id", how="left")
    resolution = resolution.merge(sequences, on="transcript_id", how="left")
    resolution[["transcript_id", "seqs", "sequence_status"]] = resolution[
        ["transcript_id", "seqs", "sequence_status"]
    ].fillna("")
    replacement = resolution.set_index("ID")["seqs"].to_dict()

    corrected = sequence_cache[["ID", "seqs"]].copy()
    corrected["seqs"] = corrected["seqs"].map(normalize_sequence_value)
    affected = corrected["ID"].isin(gene_rows["ID"])
    corrected.loc[affected, "seqs"] = corrected.loc[affected, "ID"].map(replacement).fillna("")

    corrected_full = full.drop(columns=["seqs"], errors="ignore").merge(
        corrected, on="ID", how="left", validate="one_to_one"
    )
    label_cols = [c for c in full.columns if c not in {"ID", "seqs"}]
    corrected_full = corrected_full[["ID", "seqs", *label_cols]]
    oop = corrected.loc[corrected["seqs"].ne(""), ["ID", "seqs"]].rename(
        columns={"ID": "id", "seqs": "seq"}
    )

    sequence_groups = (
        corrected.loc[corrected["seqs"].ne("")]
        .groupby("seqs", sort=False)["ID"]
        .agg(list)
    )
    duplicate_rows = [
        {"group_size": len(ids), "member_ids": "|".join(ids)}
        for ids in sequence_groups
        if len(ids) > 1
    ]

    lookup.to_csv(outdir / "gene_to_canonical_transcript.csv", index=False)
    resolution.to_csv(outdir / "transcript_refetch_report.csv", index=False)
    corrected.to_csv(outdir / "website_sequences.csv", index=False)
    corrected_full.to_csv(outdir / "website_full_matrix.csv", index=False)
    oop.to_csv(outdir / "website_sequences_for_oop.csv", index=False)
    pd.DataFrame(duplicate_rows, columns=["group_size", "member_ids"]).to_csv(
        outdir / "duplicate_sequence_groups.csv", index=False
    )

    resolved = int(resolution["seqs"].ne("").sum())
    print(
        f"[done] replaced {resolved}/{len(resolution)} gene-region rows; "
        f"usable sequences={len(oop)}/{len(corrected)}; output={outdir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
