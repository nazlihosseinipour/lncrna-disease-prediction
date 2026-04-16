from __future__ import annotations

from pathlib import Path

import pandas as pd

from mainfolder.utils.disease_mapping import ensure_disease_override_csv, normalize_disease_text


def _load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), dtype=str).fillna("")


def merge_manual_review_into_overrides(
    review_path: str | Path,
    overrides_path: str | Path,
    *,
    do_terms_path: str | Path | None = None,
) -> dict[str, int]:
    review_path = Path(review_path)
    overrides_path = ensure_disease_override_csv(overrides_path)

    review_df = _load_csv(review_path)
    required = {"disease", "chosen_term"}
    missing = sorted(required - set(review_df.columns))
    if missing:
        raise ValueError(f"{review_path} is missing required columns: {', '.join(missing)}")

    review_df["disease"] = review_df["disease"].astype(str).str.strip()
    review_df["chosen_term"] = review_df["chosen_term"].astype(str).str.strip()
    if "chosen_name" not in review_df.columns:
        review_df["chosen_name"] = ""
    if "note" not in review_df.columns:
        review_df["note"] = ""

    selected_df = review_df.loc[review_df["disease"].ne("") & review_df["chosen_term"].ne("")].copy()
    if selected_df.empty:
        return {"selected_rows": 0, "added": 0, "updated": 0, "unchanged": 0, "total_overrides": 0}

    if do_terms_path is not None:
        valid_doids = set(_load_csv(do_terms_path)["doid"].astype(str).str.strip())
        invalid = selected_df.loc[~selected_df["chosen_term"].isin(valid_doids), ["disease", "chosen_term"]]
        if not invalid.empty:
            sample = ", ".join(
                f"{row.disease} -> {row.chosen_term}" for row in invalid.head(10).itertuples(index=False)
            )
            raise ValueError(f"Found {len(invalid)} invalid DOIDs not present in {do_terms_path}: {sample}")

    overrides_df = _load_csv(overrides_path)
    for col in ("disease", "term", "note"):
        if col not in overrides_df.columns:
            overrides_df[col] = ""
    overrides_df = overrides_df[["disease", "term", "note"]].copy()

    rows: list[dict[str, str]] = []
    row_index: dict[str, int] = {}
    for row in overrides_df.to_dict("records"):
        disease = str(row.get("disease", "")).strip()
        term = str(row.get("term", "")).strip()
        note = str(row.get("note", "")).strip()
        if not disease:
            continue
        normalized = normalize_disease_text(disease)
        if not normalized:
            continue
        row_index[normalized] = len(rows)
        rows.append({"disease": disease, "term": term, "note": note})

    added = 0
    updated = 0
    unchanged = 0

    for row in selected_df.to_dict("records"):
        disease = str(row.get("disease", "")).strip()
        term = str(row.get("chosen_term", "")).strip()
        chosen_name = str(row.get("chosen_name", "")).strip()
        note = str(row.get("note", "")).strip()
        if chosen_name and f"chosen_name={chosen_name}" not in note:
            note = f"chosen_name={chosen_name}" if not note else f"chosen_name={chosen_name} | {note}"

        normalized = normalize_disease_text(disease)
        if not normalized:
            continue

        payload = {"disease": disease, "term": term, "note": note}
        existing_idx = row_index.get(normalized)
        if existing_idx is None:
            row_index[normalized] = len(rows)
            rows.append(payload)
            added += 1
            continue

        existing = rows[existing_idx]
        if existing["term"] == payload["term"] and existing["note"] == payload["note"] and existing["disease"] == payload["disease"]:
            unchanged += 1
            continue

        rows[existing_idx] = payload
        updated += 1

    out_df = pd.DataFrame(rows, columns=["disease", "term", "note"])
    out_df.to_csv(overrides_path, index=False)

    return {
        "selected_rows": len(selected_df),
        "added": added,
        "updated": updated,
        "unchanged": unchanged,
        "total_overrides": len(out_df),
    }
