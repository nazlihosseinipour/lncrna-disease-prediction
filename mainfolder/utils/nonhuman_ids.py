from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


NON_HUMAN_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^ENSRNO[GPT]\d+", re.I), "rat", "rat_ensembl"),
    (re.compile(r"^ENSMUS[GPT]\d+", re.I), "mouse", "mouse_ensembl"),
    (re.compile(r"^ENSGALG[GPT]?\d*", re.I), "chicken", "chicken_ensembl"),
    (re.compile(r"^ENSBTA[GTP]?\d*", re.I), "cow", "cow_ensembl"),
    (re.compile(r"^ENSSSC[GTP]?\d*", re.I), "pig", "pig_ensembl"),
    (re.compile(r"^ENSCAFG[TP]?\d*", re.I), "dog", "dog_ensembl"),
    (re.compile(r"^ENSDARG[TP]?\d*", re.I), "zebrafish", "zebrafish_ensembl"),
    (re.compile(r"^NONRATT", re.I), "rat", "rat_noncode"),
    (re.compile(r"^NONMMUT", re.I), "mouse", "mouse_noncode"),
    (re.compile(r"^mmu[-_]", re.I), "mouse", "mouse_prefix"),
    (re.compile(r"^rno[-_]", re.I), "rat", "rat_prefix"),
    (re.compile(r"Rik$", re.I), "mouse", "mouse_rik_symbol"),
]


def classify_nonhuman_id(raw_id: str) -> dict[str, str] | None:
    rid = str(raw_id or "").strip()
    if not rid:
        return None

    for pattern, species, rule in NON_HUMAN_PATTERNS:
        if pattern.search(rid):
            return {
                "species_hint": species,
                "non_human_reason": rule,
            }
    return None


def split_nonhuman_unresolved_df(unresolved_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if unresolved_df.empty:
        empty_nonhuman = unresolved_df.copy()
        if "species_hint" not in empty_nonhuman.columns:
            empty_nonhuman["species_hint"] = pd.Series(dtype=str)
        if "non_human_reason" not in empty_nonhuman.columns:
            empty_nonhuman["non_human_reason"] = pd.Series(dtype=str)
        return unresolved_df.copy(), empty_nonhuman

    df = unresolved_df.copy()
    annotations = df["id"].map(classify_nonhuman_id)
    nonhuman_mask = annotations.notna()

    human_df = df.loc[~nonhuman_mask].copy()
    nonhuman_df = df.loc[nonhuman_mask].copy()
    if not nonhuman_df.empty:
        nonhuman_df["species_hint"] = annotations.loc[nonhuman_mask].map(lambda x: x["species_hint"])
        nonhuman_df["non_human_reason"] = annotations.loc[nonhuman_mask].map(lambda x: x["non_human_reason"])
    else:
        nonhuman_df["species_hint"] = pd.Series(dtype=str)
        nonhuman_df["non_human_reason"] = pd.Series(dtype=str)

    return human_df, nonhuman_df


def split_nonhuman_unresolved_csv(
    unresolved_csv: str | Path,
    *,
    nonhuman_csv: str | Path,
    dedupe_on: str = "id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unresolved_path = Path(unresolved_csv)
    nonhuman_path = Path(nonhuman_csv)

    df = pd.read_csv(unresolved_path, dtype=str).fillna("")
    human_df, nonhuman_df = split_nonhuman_unresolved_df(df)
    if not human_df.empty and dedupe_on in human_df.columns:
        human_df = human_df.drop_duplicates(subset=[dedupe_on], keep="first")
    if not nonhuman_df.empty and dedupe_on in nonhuman_df.columns:
        nonhuman_df = nonhuman_df.drop_duplicates(subset=[dedupe_on], keep="first")

    human_df.to_csv(unresolved_path, index=False)
    nonhuman_df.to_csv(nonhuman_path, index=False)
    return human_df, nonhuman_df
