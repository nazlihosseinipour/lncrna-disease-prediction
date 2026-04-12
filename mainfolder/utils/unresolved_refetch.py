from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd
import requests

from mainfolder.utils.nonhuman_ids import split_nonhuman_unresolved_df
from mainfolder.utils.sequence_fetch import (
    SequenceFetchResult,
    detect_id_kind,
    extract_alias_candidates,
    fetch_sequence_by_id,
    has_usable_sequence_value,
    needs_manual_review,
    normalize_sequence_value,
)


TEXT_ALIAS_COLUMNS = ["Description", "Clinical Application", "Causal Description"]


@dataclass(frozen=True)
class RefetchPaths:
    raw_website: Path
    website_sequences: Path
    website_full: Path
    website_oop: Path
    fetch_report: Path
    unresolved_ids: Path
    non_human_unresolved_ids: Path
    ambiguous_alternatives: Path
    website_missing_ids: Path


def load_alias_map(raw_website: Path, target_ids: list[str]) -> dict[str, list[str]]:
    if not raw_website.exists():
        return {}
    flt = pd.read_csv(raw_website, dtype=str).fillna("")
    if "ncRNA Symbol" not in flt.columns:
        return {}
    text_cols = [c for c in TEXT_ALIAS_COLUMNS if c in flt.columns]
    if not text_cols:
        return {}

    target_set = {str(x or "").strip() for x in target_ids if str(x or "").strip()}
    alias_map: dict[str, list[str]] = {}
    for rid, grp in flt.groupby("ncRNA Symbol", dropna=False):
        rid = str(rid or "").strip()
        if not rid or rid not in target_set:
            continue
        texts: list[str] = []
        for col in text_cols:
            texts.extend(grp[col].fillna("").astype(str).tolist())
        aliases = extract_alias_candidates(rid, texts)
        if aliases:
            alias_map[rid] = aliases
    return alias_map


def load_human_and_nonhuman_unresolved(
    unresolved_csv: Path,
    nonhuman_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not unresolved_csv.exists():
        empty = pd.DataFrame(columns=["id", "type", "status", "reason", "resolved_id"])
        empty_nonhuman = empty.assign(species_hint=pd.Series(dtype=str), non_human_reason=pd.Series(dtype=str))
        return empty, empty_nonhuman

    unresolved_df = pd.read_csv(unresolved_csv, dtype=str).fillna("")
    human_df, nonhuman_df = split_nonhuman_unresolved_df(unresolved_df)
    if not human_df.empty:
        human_df = human_df.drop_duplicates(subset=["id"], keep="first")
    if not nonhuman_df.empty:
        nonhuman_df = nonhuman_df.drop_duplicates(subset=["id"], keep="first")
    human_df.to_csv(unresolved_csv, index=False)
    nonhuman_df.to_csv(nonhuman_csv, index=False)
    return human_df, nonhuman_df


def refetch_unresolved_ids(
    ids: list[str],
    *,
    alias_map: dict[str, list[str]],
    timeout: int,
    sleep_sec: float,
    progress_every: int,
) -> list[SequenceFetchResult]:
    if not ids:
        return []

    session = requests.Session()
    results: list[SequenceFetchResult] = []
    total = len(ids)
    print(f"[refetch] human unresolved IDs to retry: {total}")
    for idx, rid in enumerate(ids, start=1):
        result = fetch_sequence_by_id(
            session,
            rid,
            timeout=timeout,
            alias_candidates=alias_map.get(rid),
        )
        results.append(result)
        if idx == 1 or idx % progress_every == 0 or idx == total:
            print(
                f"[refetch] {idx}/{total} | id={rid} | status={result.status}"
            )
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return results


def update_website_sequences(website_sequences: Path, results: list[SequenceFetchResult]) -> pd.DataFrame:
    if not website_sequences.exists():
        raise FileNotFoundError(f"Missing sequence table: {website_sequences}")

    seq_df = pd.read_csv(website_sequences, dtype=str).fillna("")
    if "ID" not in seq_df.columns or "seqs" not in seq_df.columns:
        raise ValueError(f"{website_sequences} must have columns ID,seqs")
    seq_df["ID"] = seq_df["ID"].astype(str).str.strip()
    seq_df["seqs"] = seq_df["seqs"].map(normalize_sequence_value)

    fetched = {
        result.query_id: normalize_sequence_value(result.sequence)
        for result in results
        if result.sequence
    }
    if fetched:
        mask = seq_df["ID"].isin(fetched)
        seq_df.loc[mask, "seqs"] = seq_df.loc[mask, "ID"].map(fetched)
    seq_df["seqs"] = seq_df["seqs"].map(normalize_sequence_value)
    seq_df = seq_df.sort_values(["ID", "seqs"], ascending=[True, False]).drop_duplicates(subset=["ID"], keep="first")
    seq_df.to_csv(website_sequences, index=False)
    return seq_df


def update_fetch_report(
    fetch_report: Path,
    *,
    seq_df: pd.DataFrame,
    results: list[SequenceFetchResult],
    nonhuman_df: pd.DataFrame,
) -> pd.DataFrame:
    if fetch_report.exists():
        report_df = pd.read_csv(fetch_report, dtype=str).fillna("")
    else:
        report_df = pd.DataFrame(columns=["ID", "type", "status", "detail", "resolved_id"])

    if "ID" not in report_df.columns:
        report_df["ID"] = pd.Series(dtype=str)
    for col in ["type", "status", "detail", "resolved_id"]:
        if col not in report_df.columns:
            report_df[col] = ""
    report_df["ID"] = report_df["ID"].astype(str).str.strip()
    report_df = report_df.set_index("ID", drop=False)

    for rid in seq_df["ID"].astype(str).tolist():
        if rid not in report_df.index:
            report_df.loc[rid, ["ID", "type", "status", "detail", "resolved_id"]] = [
                rid,
                detect_id_kind(rid),
                "cached" if has_usable_sequence_value(seq_df.loc[seq_df["ID"] == rid, "seqs"].iloc[0]) else "pending",
                "",
                "",
            ]

    for result in results:
        row = result.report_row(id_column="ID")
        report_df.loc[result.query_id, ["ID", "type", "status", "detail", "resolved_id"]] = [
            row["ID"],
            row["type"],
            row["status"],
            row["detail"],
            row["resolved_id"],
        ]

    if not nonhuman_df.empty:
        for row in nonhuman_df.to_dict("records"):
            rid = str(row.get("id", "")).strip()
            if not rid:
                continue
            detail = str(row.get("reason", "")).strip()
            species = str(row.get("species_hint", "")).strip()
            nonhuman_reason = str(row.get("non_human_reason", "")).strip()
            prefix = f"non_human:{species}:{nonhuman_reason}".strip(":")
            detail = f"{prefix}; {detail}" if detail else prefix
            report_df.loc[rid, ["ID", "type", "status", "detail", "resolved_id"]] = [
                rid,
                str(row.get("type", detect_id_kind(rid))),
                "unresolved_non_human",
                detail,
                str(row.get("resolved_id", "")),
            ]

    report_df = report_df.reset_index(drop=True)
    report_df = report_df.sort_values("ID").drop_duplicates(subset=["ID"], keep="first")
    report_df.to_csv(fetch_report, index=False)
    return report_df


def update_unresolved_outputs(
    unresolved_csv: Path,
    nonhuman_csv: Path,
    ambiguous_csv: Path,
    results: list[SequenceFetchResult],
    existing_nonhuman: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    review_rows = [r.review_row() for r in results if not r.sequence and needs_manual_review(r)]
    unresolved_df = pd.DataFrame(review_rows, columns=["id", "type", "status", "reason", "resolved_id"])
    if not unresolved_df.empty:
        unresolved_df = unresolved_df.drop_duplicates(subset=["id"], keep="first")
    unresolved_df, newly_nonhuman_df = split_nonhuman_unresolved_df(unresolved_df)

    nonhuman_frames = [df for df in (existing_nonhuman, newly_nonhuman_df) if not df.empty]
    if nonhuman_frames:
        nonhuman_df = pd.concat(nonhuman_frames, ignore_index=True).drop_duplicates(subset=["id"], keep="first")
    else:
        nonhuman_df = pd.DataFrame(columns=["id", "type", "status", "reason", "resolved_id", "species_hint", "non_human_reason"])

    unresolved_df.to_csv(unresolved_csv, index=False)
    nonhuman_df.to_csv(nonhuman_csv, index=False)

    alternative_rows = [r.alternatives_row() for r in results if r.alternative_ids]
    if ambiguous_csv.exists():
        existing_alt = pd.read_csv(ambiguous_csv, dtype=str).fillna("")
    else:
        existing_alt = pd.DataFrame(columns=["query_id", "selected_id", "alternative_ids", "source"])
    new_alt = pd.DataFrame(alternative_rows, columns=["query_id", "selected_id", "alternative_ids", "source"])
    alt_frames = [df for df in (existing_alt, new_alt) if not df.empty]
    if alt_frames:
        alternatives_df = pd.concat(alt_frames, ignore_index=True).drop_duplicates(subset=["query_id"], keep="last")
    else:
        alternatives_df = pd.DataFrame(columns=["query_id", "selected_id", "alternative_ids", "source"])
    alternatives_df.to_csv(ambiguous_csv, index=False)

    return unresolved_df, nonhuman_df


def rebuild_sequence_outputs(
    *,
    seq_df: pd.DataFrame,
    website_full: Path,
    website_oop: Path,
    website_missing_ids: Path,
) -> tuple[int, int]:
    if website_full.exists():
        full_df = pd.read_csv(website_full, dtype=str).fillna("")
        if {"ID", "seqs"}.issubset(full_df.columns):
            full_df = full_df.drop(columns=["seqs"]).merge(seq_df[["ID", "seqs"]], on="ID", how="left")
            disease_cols = [c for c in full_df.columns if c not in {"seqs"}]
            if "ID" in disease_cols:
                disease_cols.remove("ID")
            full_df["seqs"] = full_df["seqs"].map(normalize_sequence_value)
            full_df = full_df[["ID", "seqs", *disease_cols]]
            full_df.to_csv(website_full, index=False)
            oop_df = full_df.loc[full_df["seqs"].map(has_usable_sequence_value), ["ID", "seqs"]].rename(
                columns={"ID": "id", "seqs": "seq"}
            )
        else:
            oop_df = seq_df.loc[seq_df["seqs"].map(has_usable_sequence_value), ["ID", "seqs"]].rename(
                columns={"ID": "id", "seqs": "seq"}
            )
    else:
        oop_df = seq_df.loc[seq_df["seqs"].map(has_usable_sequence_value), ["ID", "seqs"]].rename(
            columns={"ID": "id", "seqs": "seq"}
        )

    oop_df.to_csv(website_oop, index=False)

    missing_ids = seq_df.loc[~seq_df["seqs"].map(has_usable_sequence_value), "ID"].astype(str).tolist()
    website_missing_ids.write_text("\n".join(missing_ids) + ("\n" if missing_ids else ""), encoding="utf-8")
    return len(oop_df), len(missing_ids)


def refetch_human_unresolved_only(
    paths: RefetchPaths,
    *,
    timeout: int = 10,
    sleep_sec: float = 0.05,
    progress_every: int = 25,
    max_ids: int | None = None,
) -> dict[str, int]:
    human_unresolved_df, nonhuman_unresolved_df = load_human_and_nonhuman_unresolved(
        paths.unresolved_ids,
        paths.non_human_unresolved_ids,
    )
    human_ids = human_unresolved_df["id"].astype(str).str.strip().tolist()
    if max_ids is not None:
        human_ids = human_ids[:max_ids]

    alias_map = load_alias_map(paths.raw_website, human_ids)
    if alias_map:
        print(f"[info] alias candidates prepared for {len(alias_map)} unresolved IDs")

    results = refetch_unresolved_ids(
        human_ids,
        alias_map=alias_map,
        timeout=timeout,
        sleep_sec=sleep_sec,
        progress_every=progress_every,
    )

    seq_df = update_website_sequences(paths.website_sequences, results)
    unresolved_df, nonhuman_df = update_unresolved_outputs(
        paths.unresolved_ids,
        paths.non_human_unresolved_ids,
        paths.ambiguous_alternatives,
        results,
        nonhuman_unresolved_df,
    )
    update_fetch_report(
        paths.fetch_report,
        seq_df=seq_df,
        results=results,
        nonhuman_df=nonhuman_df,
    )
    oop_rows, missing_count = rebuild_sequence_outputs(
        seq_df=seq_df,
        website_full=paths.website_full,
        website_oop=paths.website_oop,
        website_missing_ids=paths.website_missing_ids,
    )

    resolved_now = sum(1 for r in results if r.sequence)
    return {
        "retried_human_ids": len(human_ids),
        "resolved_now": resolved_now,
        "remaining_human_unresolved": len(unresolved_df),
        "non_human_unresolved": len(nonhuman_df),
        "oop_rows": oop_rows,
        "missing_sequences_total": missing_count,
    }
