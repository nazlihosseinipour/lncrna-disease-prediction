"""Independent per-column audit of the released 827-column V1 feature matrix.

Question: can every released column be reproduced from documented sequence
operations applied to the raw sequences?

The audit
1. maps each released row to the raw row whose reconstructed sequence
   descriptors match it (separately for the DCC block and the k-mer block),
2. classifies every released column against a library of candidate columns
   (DCC lag 1, k-mer k=2/3 normalized and raw counts, and negative-control
   families: DAC, DACC, mismatch, PC-PseDNC-General, SC-PseDNC-General),
3. audits the 285-column binary block against the raw association matrix,
4. derives contiguous family boundaries and a verdict.

No model fitting, no network access, no parameter search: each negative-control
family is computed once (or at the few fixed settings listed below) with
textbook definitions, only to test whether the family is present at all.

The frozen reconstruction ``scripts/reconstruct_paper_sequence_features.py`` is
imported and never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reconstruct_paper_sequence_features import (  # noqa: E402
    DEFAULT_PROPERTIES,
    DINUCLEOTIDES,
    TRINUCLEOTIDES,
    dcc_vector,
    feature_columns,
    kmer_vector,
    load_properties,
    normalize_sequence,
    sha256_file,
)

RELEASED_CSV = PROJECT_ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/features/features_preprocessed.csv"
RAW_CSV = PROJECT_ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv"
FROZEN_SCRIPT = PROJECT_ROOT / "scripts/reconstruct_paper_sequence_features.py"
DEFAULT_OUT = PROJECT_ROOT / "results/felipe_paper_exact_2026-09-29T0014/feature_gate"
EXPECTED_RELEASED_SHA256 = "b264dd75545ca424f1db96622555066791c52ba6105001770ae869e211428887"

ATOL = 1e-6
RTOL = 1e-6
BINARY_BLOCK = (462, 747)  # half-open column range of the 285-column binary block
OUTPUT_FILES = (
    "column_audit.csv",
    "row_mapping.csv",
    "binary_block_audit.csv",
    "negative_controls.csv",
    "feature_gate_summary.json",
)
BINARY_WORDING = (
    "numerically matches the raw association matrix; its provenance conflicts "
    "with the paper's description and remains unresolved"
)


# --------------------------------------------------------------------------
# Generic matching / tolerance logic (unit-tested on synthetic data)
# --------------------------------------------------------------------------
def classify(max_abs: float, max_rel: float, *, atol: float = ATOL, rtol: float = RTOL) -> str:
    """Status of a comparison given max |diff| and max relative diff."""
    if not np.isfinite(max_abs):
        return "unmatched"
    if max_abs == 0.0:
        return "exact"
    if max_abs <= atol:
        return "within_abs_1e-6"
    if max_rel <= rtol:
        return "within_rel_1e-6"
    return "unmatched"


def rel_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise |a-b| / max(|a|,|b|), 0 where both are zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(a), np.abs(b))
    diff = np.abs(a - b)
    out = np.zeros(np.broadcast(a, b).shape)
    np.divide(diff, denom, out=out, where=denom > 0)
    return out


def best_row_matches(released: np.ndarray, candidates: np.ndarray, *, atol: float = ATOL) -> pd.DataFrame:
    """For each released row, find the candidate row with the smallest max |diff|.

    Returns best index, its max |diff|, second-best max |diff|, number of
    candidate rows within ``atol`` (ties), and the identity-row max |diff|.
    """
    released = np.asarray(released, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    rows = []
    for i, vec in enumerate(released):
        dist = np.abs(candidates - vec).max(axis=1)
        order = np.argsort(dist, kind="stable")
        best = int(order[0])
        rows.append(
            {
                "best_row": best,
                "best_max_abs": float(dist[best]),
                "second_max_abs": float(dist[order[1]]) if len(order) > 1 else np.inf,
                "n_within_atol": int((dist <= atol).sum()),
                "identity_max_abs": float(dist[i]) if i < len(candidates) else np.inf,
            }
        )
    return pd.DataFrame(rows)


def classify_rows(best: pd.DataFrame, *, atol: float = ATOL) -> list[str]:
    out = []
    for i, row in best.iterrows():
        if row.best_max_abs > atol:
            out.append("unmatched")
        elif row.identity_max_abs <= atol:
            out.append("identity")
        else:
            out.append("shifted")
    return out


def column_matches(
    released: np.ndarray,
    candidates: np.ndarray,
    mapping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare every released column with every candidate column.

    ``released`` is (n_released_rows, n_released_cols); ``candidates`` is
    (n_raw_rows, n_candidate_cols); ``mapping[i]`` is the raw row for released
    row i, or -1 to exclude that row. Returns, per released column, the index
    of the best candidate, its max |diff|, its max relative diff, and the full
    (n_released_cols, n_candidate_cols) max-|diff| matrix.
    """
    mapping = np.asarray(mapping)
    keep = mapping >= 0
    rel = np.asarray(released, dtype=float)[keep]
    cand = np.asarray(candidates, dtype=float)[mapping[keep]]
    n_rel, n_cand = rel.shape[1], cand.shape[1]
    maxabs = np.empty((n_rel, n_cand))
    chunk = 64
    for start in range(0, n_cand, chunk):
        block = cand[:, start : start + chunk]
        maxabs[:, start : start + chunk] = np.abs(rel[:, :, None] - block[:, None, :]).max(axis=0)
    best = maxabs.argmin(axis=1)
    best_abs = maxabs[np.arange(n_rel), best]
    best_rel = np.array([rel_diff(rel[:, c], cand[:, b]).max() for c, b in enumerate(best)])
    return best, best_abs, best_rel, maxabs


# --------------------------------------------------------------------------
# Candidate descriptor families
# --------------------------------------------------------------------------
def kmer_counts(sequence: str, k: int) -> np.ndarray:
    kmers = ["".join(x) for x in itertools.product("ACGU", repeat=k)]
    counts = Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1))
    return np.asarray([counts.get(x, 0) for x in kmers], dtype=float)


def mismatch_counts(sequence: str, k: int = 3, m: int = 1) -> np.ndarray:
    """Mismatch profile: for each k-mer, occurrences of k-mers within Hamming m."""
    kmers = ["".join(x) for x in itertools.product("ACGU", repeat=k)]
    observed = Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1))
    out = np.zeros(len(kmers))
    for j, target in enumerate(kmers):
        out[j] = sum(
            n for kmer, n in observed.items() if sum(a != b for a, b in zip(kmer, target)) <= m
        )
    return out


def encode(sequence: str, properties: np.ndarray) -> np.ndarray:
    idx = {d: i for i, d in enumerate(DINUCLEOTIDES)}
    return np.asarray([properties[idx[sequence[i : i + 2]]] for i in range(len(sequence) - 1)])


def dac_vector(sequence: str, properties: np.ndarray, lag: int = 1, *, style: str) -> np.ndarray:
    """DAC lag 1 for every property.

    style="pse106": same arithmetic as the frozen DCC (mean divided by L).
    style="textbook": mean over the L-1 dinucleotide values.
    """
    enc = encode(sequence, properties)
    n = len(sequence) - lag - 1
    if style == "pse106":
        mean = enc[:n].sum(axis=0) / len(sequence)
    else:
        mean = enc.mean(axis=0)
    left, right = enc[:n] - mean, enc[lag : lag + n] - mean
    return np.round((left * right).sum(axis=0) / n, 8)


def standardize(properties: np.ndarray) -> np.ndarray:
    sd = properties.std(axis=0)
    sd[sd == 0] = 1.0
    return (properties - properties.mean(axis=0)) / sd


def pc_psednc(sequence: str, std_props: np.ndarray, lam: int, w: float) -> np.ndarray:
    enc = encode(sequence, std_props)
    freq = kmer_counts(sequence, 2)
    freq = freq / freq.sum()
    theta = np.array(
        [((enc[: len(enc) - j] - enc[j:]) ** 2).mean(axis=1).mean() for j in range(1, lam + 1)]
    )
    denom = freq.sum() + w * theta.sum()
    return np.round(np.concatenate([freq / denom, w * theta / denom]), 8)


def sc_psednc(sequence: str, std_props: np.ndarray, lam: int, w: float) -> np.ndarray:
    enc = encode(sequence, std_props)
    freq = kmer_counts(sequence, 2)
    freq = freq / freq.sum()
    theta = np.concatenate(
        [(enc[: len(enc) - j] * enc[j:]).mean(axis=0) for j in range(1, lam + 1)]
    )
    denom = freq.sum() + w * theta.sum()
    return np.round(np.concatenate([freq / denom, w * theta / denom]), 8)


def build_candidates(sequences: list[str], prop_names: list[str], props: np.ndarray):
    """Return {family_key: (DataFrame n_raw x n_cols, role, params)}."""
    std_props = standardize(props)
    dcc_names = feature_columns(prop_names)[:462]
    tri = ["".join(x) for x in itertools.product("ACGU", repeat=3)]
    fams: dict[str, tuple[list[str], list[np.ndarray], str, str]] = {}

    def add(key, names, role, params, fn):
        fams[key] = (names, [fn(s) for s in sequences], role, params)

    add("dcc_lag1", dcc_names, "documented", "Pse-in-One 1.0.6 RNA DCC, 22 props, lag=1 (frozen script)",
        lambda s: dcc_vector(s, props))
    add("kmer2_norm", [f"kmer2::{x}" for x in DINUCLEOTIDES], "documented", "k=2 normalized (frozen script)",
        lambda s: kmer_vector(s, 2))
    add("kmer3_norm", [f"kmer3::{x}" for x in TRINUCLEOTIDES], "documented", "k=3 normalized (frozen script)",
        lambda s: kmer_vector(s, 3))
    add("kmer2_count", [f"kmer2_count::{x}" for x in DINUCLEOTIDES], "documented", "k=2 raw counts",
        lambda s: kmer_counts(s, 2))
    add("kmer3_count", [f"kmer3_count::{x}" for x in tri], "documented", "k=3 raw counts",
        lambda s: kmer_counts(s, 3))
    for style in ("pse106", "textbook"):
        add(f"dac_lag1_{style}", [f"dac_lag1_{style}::{p}" for p in prop_names], "negative_control",
            f"DAC lag=1, 22 raw props, mean={'sum/L' if style == 'pse106' else 'mean over L-1'}",
            lambda s, st=style: dac_vector(s, props, 1, style=st))
    # DACC = DAC + DCC concatenated; the DCC half is identical to dcc_lag1.
    add("dacc_lag1", [f"dacc_lag1::dac::{p}" for p in prop_names] + [f"dacc_lag1::{n}" for n in dcc_names],
        "negative_control", "DACC lag=1 = DAC(pse106) ++ DCC; DCC half duplicates dcc_lag1",
        lambda s: np.concatenate([dac_vector(s, props, 1, style="pse106"), dcc_vector(s, props)]))
    add("mismatch_k3_m1", [f"mismatch_k3_m1::{x}" for x in tri], "negative_control", "mismatch k=3 m=1 counts",
        lambda s: mismatch_counts(s, 3, 1))
    for lam in (1, 2, 3):
        for w in (0.05, 0.1):
            add(f"pc_psednc_l{lam}_w{w}",
                [f"pc_psednc_l{lam}_w{w}::{x}" for x in DINUCLEOTIDES] + [f"pc_psednc_l{lam}_w{w}::theta{j}" for j in range(1, lam + 1)],
                "negative_control", f"PC-PseDNC-General lambda={lam} w={w}, 22 z-scored props",
                lambda s, l=lam, ww=w: pc_psednc(s, std_props, l, ww))
    for lam in (1, 2):
        add(f"sc_psednc_l{lam}_w0.05",
            [f"sc_psednc_l{lam}_w0.05::{x}" for x in DINUCLEOTIDES]
            + [f"sc_psednc_l{lam}_w0.05::theta{j}::{p}" for j in range(1, lam + 1) for p in prop_names],
            "negative_control", f"SC-PseDNC-General lambda={lam} w=0.05, 22 z-scored props",
            lambda s, l=lam: sc_psednc(s, std_props, l, 0.05))
    return {
        key: (pd.DataFrame(np.vstack(rows), columns=names), role, params)
        for key, (names, rows, role, params) in fams.items()
    }


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
def contiguous_runs(labels: list[str]) -> list[dict]:
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append({"start": start, "end_inclusive": i - 1, "n": i - start, "family": labels[start]})
            start = i
    return runs


def run_audit(out_dir: Path) -> dict:
    existing = [name for name in OUTPUT_FILES if (out_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs in {out_dir}: {existing}")

    released_sha = sha256_file(RELEASED_CSV)
    X = pd.read_csv(RELEASED_CSV, index_col=0)
    raw = pd.read_csv(RAW_CSV)
    assoc = raw.iloc[:, 3:]
    prop_names, props = load_properties(DEFAULT_PROPERTIES)

    sequences, cleaned_rows = [], {}
    for i, value in enumerate(raw["seqs"]):
        seq, removed = normalize_sequence(value)
        sequences.append(seq)
        if removed:
            cleaned_rows[i] = dict(removed)
    fams = build_candidates(sequences, prop_names, props)

    Xv = X.to_numpy(dtype=float)
    n_rows = len(X)
    dcc_rel, kmer_rel = Xv[:, :462], Xv[:, 747:]
    dcc_rec = fams["dcc_lag1"][0].to_numpy()
    kmer_rec = np.hstack([fams["kmer2_norm"][0].to_numpy(), fams["kmer3_norm"][0].to_numpy()])
    bin_rel = Xv[:, BINARY_BLOCK[0] : BINARY_BLOCK[1]]
    A = assoc.to_numpy(dtype=float)

    # ---- 1. row mapping ----------------------------------------------------
    dcc_best = best_row_matches(dcc_rel, dcc_rec)
    kmer_best = best_row_matches(kmer_rel, kmer_rec)
    both_best = best_row_matches(np.hstack([dcc_rel, kmer_rel]), np.hstack([dcc_rec, kmer_rec]))
    both_class = classify_rows(both_best)
    mapping = np.where(np.array(both_class) != "unmatched", both_best.best_row.to_numpy(), -1)

    rm = pd.DataFrame(
        {
            "released_row": np.arange(n_rows),
            "released_index_id": X.index.astype(str),
            "raw_id_same_position": raw["ID"].astype(str).to_numpy(),
            "raw_unnamed0_same_position": raw["Unnamed: 0"].to_numpy(),
            "index_id_equals_raw_id": X.index.astype(str) == raw["ID"].astype(str).to_numpy(),
            "dcc_best_raw_row": dcc_best.best_row,
            "dcc_best_max_abs": dcc_best.best_max_abs,
            "dcc_second_max_abs": dcc_best.second_max_abs,
            "dcc_n_raw_rows_within_1e-6": dcc_best.n_within_atol,
            "dcc_identity_max_abs": dcc_best.identity_max_abs,
            "dcc_mapping_class": classify_rows(dcc_best),
            "kmer_best_raw_row": kmer_best.best_row,
            "kmer_best_max_abs": kmer_best.best_max_abs,
            "kmer_n_raw_rows_within_1e-6": kmer_best.n_within_atol,
            "kmer_identity_max_abs": kmer_best.identity_max_abs,
            "kmer_mapping_class": classify_rows(kmer_best),
            "sequence_mapping_raw_row": mapping,
            "sequence_mapping_offset": np.where(mapping >= 0, mapping - np.arange(n_rows), np.nan),
            "sequence_mapping_class": both_class,
            "sequence_mapping_status": [classify(a, np.inf) if a <= ATOL else "unmatched" for a in both_best.best_max_abs],
            "mapped_raw_id": [raw["ID"].iloc[m] if m >= 0 else "" for m in mapping],
            "mapped_raw_unnamed0": [int(raw["Unnamed: 0"].iloc[m]) if m >= 0 else -1 for m in mapping],
            "released_id_equals_mapped_raw_id": [m >= 0 and str(X.index[i]) == str(raw["ID"].iloc[m]) for i, m in enumerate(mapping)],
        }
    )
    bin_eq_identity = (bin_rel == A).all(axis=1)
    bin_eq_mapped = np.array([m >= 0 and bool((bin_rel[i] == A[m]).all()) for i, m in enumerate(mapping)])
    bin_n_equal = np.array([int((A == bin_rel[i]).all(axis=1).sum()) for i in range(n_rows)])
    rm["binary_equals_raw_assoc_identity_row"] = bin_eq_identity
    rm["binary_equals_raw_assoc_sequence_mapped_row"] = bin_eq_mapped
    rm["binary_n_raw_rows_with_identical_assoc_vector"] = bin_n_equal
    rm["binary_row_sum"] = bin_rel.sum(axis=1).astype(int)
    rm["released_row_ambiguous_seq_at_same_position"] = [i in cleaned_rows for i in range(n_rows)]

    raw_unmatched = sorted(set(range(len(raw))) - set(int(m) for m in mapping if m >= 0))
    released_unmatched = [int(i) for i in np.where(mapping < 0)[0]]
    # diagnostics for unmatched released rows: nearest raw rows per block
    unmatched_diag = []
    for i in released_unmatched:
        d_dcc = np.abs(dcc_rec - dcc_rel[i]).max(axis=1)
        d_k = np.abs(kmer_rec - kmer_rel[i]).max(axis=1)
        unmatched_diag.append(
            {
                "released_row": i,
                "released_index_id": str(X.index[i]),
                "nearest_raw_row_dcc": int(d_dcc.argmin()),
                "nearest_dcc_max_abs": float(d_dcc.min()),
                "nearest_raw_row_kmer": int(d_k.argmin()),
                "nearest_kmer_max_abs": float(d_k.min()),
                "dcc_max_abs_vs_unmatched_raw_rows": {int(r): float(d_dcc[r]) for r in raw_unmatched},
                "kmer_max_abs_vs_unmatched_raw_rows": {int(r): float(d_k[r]) for r in raw_unmatched},
                "binary_row_sum": int(bin_rel[i].sum()),
                "binary_n_raw_rows_identical": int(bin_n_equal[i]),
                "binary_identical_raw_rows": [int(r) for r in np.where((A == bin_rel[i]).all(axis=1))[0]],
            }
        )
    offset_runs = contiguous_runs(
        [("unmatched" if m < 0 else f"offset{int(m - i):+d}") for i, m in enumerate(mapping)]
    )

    # ---- 2. per-column classification --------------------------------------
    cand_frames, cand_family, cand_role = [], [], []
    for key, (frame, role, _params) in fams.items():
        cand_frames.append(frame)
        cand_family += [key] * frame.shape[1]
        cand_role += [role] * frame.shape[1]
    assoc_frame = assoc.copy()
    assoc_frame.columns = [f"raw_assoc::{c}" for c in assoc.columns]
    cand_frames.append(assoc_frame.astype(float))
    cand_family += ["raw_association_matrix"] * assoc.shape[1]
    cand_role += ["reference_non_sequence"] * assoc.shape[1]
    cand = pd.concat(cand_frames, axis=1)
    cand_names = cand.columns.tolist()
    cand_family = np.array(cand_family)
    cand_role = np.array(cand_role)
    C = cand.to_numpy(dtype=float)

    best, best_abs, best_rel, maxabs = column_matches(Xv, C, mapping)
    ident_map = np.arange(n_rows)
    _, _, _, maxabs_ident = column_matches(Xv, C, ident_map)
    positional = feature_columns(prop_names)
    pos_names = positional[:462] + [None] * 285 + positional[462:]
    rel_keep = Xv[mapping >= 0]
    cand_keep = C[mapping[mapping >= 0]]

    col_rows, fam_labels = [], []
    for c in range(Xv.shape[1]):
        b = int(best[c])
        status = classify(best_abs[c], best_rel[c])
        within = maxabs[c] <= ATOL
        fam = cand_family[b] if status != "unmatched" else "unmatched"
        if fam == "raw_association_matrix":
            fam = "binary_block_numerically_matches_raw_association_matrix"
        fam_labels.append(fam)
        if pos_names[c] is not None:
            p = cand_names.index(pos_names[c])
            pos_abs = float(maxabs[c, p])
            pos_rel = float(rel_diff(rel_keep[:, c], cand_keep[:, p]).max())
            pos_ident = float(maxabs_ident[c, p])
        else:
            p = cand_names.index(assoc_frame.columns[c - BINARY_BLOCK[0]])
            pos_abs, pos_ident = float(maxabs[c, p]), float(maxabs_ident[c, p])
            pos_rel = float(rel_diff(rel_keep[:, c], cand_keep[:, p]).max())
        vals = np.unique(Xv[:, c])
        col_rows.append(
            {
                "col_index": c,
                "released_name_or_position": X.columns[c],
                "family_assigned": fam,
                "best_match": cand_names[b],
                "best_match_family": cand_family[b],
                "max_abs_diff": float(best_abs[c]),
                "max_rel_diff": float(best_rel[c]),
                "status": status,
                "n_candidate_columns_within_1e-6": int(within.sum()),
                "n_negative_control_columns_within_1e-6": int((within & (cand_role == "negative_control")).sum()),
                "positional_expected_match": cand_names[p],
                "positional_max_abs_diff_sequence_mapping": pos_abs,
                "positional_status_sequence_mapping": classify(pos_abs, pos_rel),
                "positional_max_abs_diff_identity_rows": pos_ident,
                "positional_status_identity_rows": "exact" if pos_ident == 0 else ("within_abs_1e-6" if pos_ident <= ATOL else "unmatched"),
                "n_unique_values": int(len(vals)),
                "is_binary_valued": bool(set(vals.tolist()) <= {0.0, 1.0}),
                "n_rows_compared": int((mapping >= 0).sum()),
            }
        )
    col_df = pd.DataFrame(col_rows)

    # ---- negative controls --------------------------------------------------
    dcc_idx = np.where(cand_family == "dcc_lag1")[0]
    nc_rows = []
    for key, (frame, role, params) in list(fams.items()) + [("raw_association_matrix", (assoc_frame, "reference_non_sequence", "raw association columns (not a sequence family)"))]:
        idx = np.where(cand_family == key)[0]
        within = maxabs[:, idx] <= ATOL
        # candidate columns numerically identical (on mapped rows) to some DCC column
        dup_dcc = np.zeros(len(idx), dtype=bool)
        if key != "dcc_lag1":
            for j, ci in enumerate(idx):
                dup_dcc[j] = bool((np.abs(cand_keep[:, dcc_idx] - cand_keep[:, [ci]]).max(axis=0) <= ATOL).any())
        matched_rel = np.where(within.any(axis=1))[0]
        matched_rel_nodup = np.where(within[:, ~dup_dcc].any(axis=1))[0] if (~dup_dcc).any() else np.array([], int)
        nc_rows.append(
            {
                "family": key,
                "role": role,
                "params": params,
                "n_candidate_columns": len(idx),
                "n_released_columns_matched": int(len(matched_rel)),
                "n_candidate_columns_matched": int(within.any(axis=0).sum()),
                "n_candidate_columns_identical_to_a_dcc_column": int(dup_dcc.sum()),
                "n_released_columns_matched_excluding_dcc_duplicates": int(len(matched_rel_nodup)),
                "released_col_ranges_matched": json.dumps([[r["start"], r["end_inclusive"]] for r in contiguous_runs(
                    ["m" if k in set(matched_rel.tolist()) else "-" for k in range(Xv.shape[1])]) if r["family"] == "m"]),
                "released_col_ranges_matched_excluding_dcc_duplicates": json.dumps([[r["start"], r["end_inclusive"]] for r in contiguous_runs(
                    ["m" if k in set(matched_rel_nodup.tolist()) else "-" for k in range(Xv.shape[1])]) if r["family"] == "m"]),
                "min_max_abs_diff_any_released_col": float(maxabs[:, idx].min()),
            }
        )
    nc_df = pd.DataFrame(nc_rows)

    # ---- 3. binary block ------------------------------------------------------
    shift_counts = {}
    for s in range(-3, 4):
        n = 0
        for i in range(n_rows):
            j = i + s
            if 0 <= j < len(A) and (bin_rel[i] == A[j]).all():
                n += 1
        shift_counts[f"{s:+d}"] = n
    raw_bin_block = X.iloc[:, BINARY_BLOCK[0] : BINARY_BLOCK[1]]
    bb_rows = []
    for j in range(A.shape[1]):
        c = BINARY_BLOCK[0] + j
        col = bin_rel[:, j]
        mapped_eq = int(sum(col[i] == A[m, j] for i, m in enumerate(mapping) if m >= 0))
        bb_rows.append(
            {
                "col_index": c,
                "released_name": X.columns[c],
                "released_dtype": str(raw_bin_block.dtypes.iloc[j]),
                "released_value_set": json.dumps(sorted(np.unique(col).astype(int).tolist())),
                "released_sum": int(col.sum()),
                "raw_assoc_position": j,
                "raw_assoc_name": assoc.columns[j],
                "raw_assoc_sum": int(A[:, j].sum()),
                "sums_equal": bool(col.sum() == A[:, j].sum()),
                "rows_equal_identity": int((col == A[:, j]).sum()),
                "rows_equal_sequence_mapping": mapped_eq,
                "n_rows_sequence_mapping": int((mapping >= 0).sum()),
                "column_equal_identity_all_rows": bool((col == A[:, j]).all()),
                "column_equal_sequence_mapping_all_mapped_rows": mapped_eq == int((mapping >= 0).sum()),
                "best_raw_column_sequence_mapping": col_df.loc[c, "best_match"],
                "best_raw_column_max_abs": col_df.loc[c, "max_abs_diff"],
                "status_sequence_mapping": col_df.loc[c, "status"],
            }
        )
    bb_df = pd.DataFrame(bb_rows)

    # value sets of sequence blocks
    def value_info(block):
        return {
            "min": float(block.min()),
            "max": float(block.max()),
            "n_unique": int(len(np.unique(block))),
            "n_binary_valued_columns": int(sum(set(np.unique(block[:, k]).tolist()) <= {0.0, 1.0} for k in range(block.shape[1]))),
        }

    # ---- 4/5. boundaries + verdict -------------------------------------------
    runs = contiguous_runs(fam_labels)
    status_counts = col_df.status.value_counts().to_dict()
    blocks = {
        "dcc_0_461": (0, 462),
        "binary_462_746": (462, 747),
        "kmer_747_826": (747, 827),
    }
    per_block = {}
    for name, (a, b) in blocks.items():
        sub = col_df.iloc[a:b]
        per_block[name] = {
            "n_columns": b - a,
            "status_counts": sub.status.value_counts().to_dict(),
            "family_counts": sub.family_assigned.value_counts().to_dict(),
            "positional_status_sequence_mapping": sub.positional_status_sequence_mapping.value_counts().to_dict(),
            "positional_status_identity_rows": sub.positional_status_identity_rows.value_counts().to_dict(),
            "best_match_equals_positional": int((sub.best_match == sub.positional_expected_match).sum()),
            "max_abs_diff_max": float(sub.max_abs_diff.max()),
        }
    seq_family_cols = col_df.family_assigned.isin(["dcc_lag1", "kmer2_norm", "kmer3_norm"])
    unmatched_cols = col_df.loc[col_df.status == "unmatched", "col_index"].tolist()

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "released_csv": RELEASED_CSV.relative_to(PROJECT_ROOT).as_posix(),
            "released_sha256": released_sha,
            "released_sha256_matches_expected": released_sha == EXPECTED_RELEASED_SHA256,
            "raw_sequences_csv": RAW_CSV.relative_to(PROJECT_ROOT).as_posix(),
            "raw_sequences_sha256": sha256_file(RAW_CSV),
            "properties_csv": Path(DEFAULT_PROPERTIES).relative_to(PROJECT_ROOT).as_posix(),
            "properties_sha256": sha256_file(Path(DEFAULT_PROPERTIES)),
            "frozen_reconstruction_script": FROZEN_SCRIPT.relative_to(PROJECT_ROOT).as_posix(),
            "frozen_reconstruction_script_sha256": sha256_file(FROZEN_SCRIPT),
        },
        "script": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "tolerances": {"exact": "max|diff| == 0", "abs": ATOL, "rel": RTOL,
                       "row_match": f"max|diff| over block <= {ATOL}"},
        "shape_released": list(X.shape),
        "shape_raw": list(raw.shape),
        "row_mapping": {
            "released_index_equals_raw_ID_by_position": int(rm.index_id_equals_raw_id.sum()),
            "raw_unnamed0_is_0_to_n_minus_1": bool((raw["Unnamed: 0"].to_numpy() == np.arange(len(raw))).all()),
            "raw_unnamed0_range": [int(raw["Unnamed: 0"].min()), int(raw["Unnamed: 0"].max())],
            "raw_unnamed0_missing_values": sorted(set(range(int(raw["Unnamed: 0"].max()) + 1)) - set(raw["Unnamed: 0"].astype(int))),
            "raw_unnamed0_first_row_differing_from_position": int(np.argmax(raw["Unnamed: 0"].to_numpy() != np.arange(len(raw)))),
            "raw_rows_with_ambiguous_symbols_removed_by_normalize_sequence": {str(k): v for k, v in cleaned_rows.items()},
            "dcc_block_class_counts": rm.dcc_mapping_class.value_counts().to_dict(),
            "kmer_block_class_counts": rm.kmer_mapping_class.value_counts().to_dict(),
            "sequence_blocks_combined_class_counts": rm.sequence_mapping_class.value_counts().to_dict(),
            "released_rows_whose_index_ID_equals_content_matched_raw_ID": int(rm.released_id_equals_mapped_raw_id.sum()),
            "dcc_and_kmer_best_row_agree_on_matched_rows": int(((rm.dcc_best_raw_row == rm.kmer_best_raw_row) & (rm.sequence_mapping_raw_row >= 0)).sum()),
            "offset_runs": offset_runs,
            "released_rows_unmatched": released_unmatched,
            "raw_rows_not_matched_by_any_released_row": raw_unmatched,
            "unmatched_released_row_diagnostics": unmatched_diag,
            "matched_rows_with_multiple_raw_candidates": int((both_best.n_within_atol[mapping >= 0] > 1).sum()),
            "binary_block_equal_raw_assoc_identity_rows": int(bin_eq_identity.sum()),
            "binary_block_equal_raw_assoc_sequence_mapped_rows": int(bin_eq_mapped.sum()),
            "binary_block_sequence_mapped_rows_total": int((mapping >= 0).sum()),
        },
        "column_status_counts_all_827": status_counts,
        "per_block": per_block,
        "family_boundaries": runs,
        "unmatched_columns": unmatched_cols,
        "value_sets": {
            "dcc_block": value_info(Xv[:, :462]),
            "binary_block": {"values": sorted(np.unique(bin_rel).astype(int).tolist()),
                             "dtypes": sorted(set(raw_bin_block.dtypes.astype(str)))},
            "kmer_block": value_info(Xv[:, 747:]),
            "released_dtype_counts": X.dtypes.astype(str).value_counts().to_dict(),
        },
        "binary_block": {
            "n_columns": 285,
            "total_positives_released": int(bin_rel.sum()),
            "total_positives_raw_assoc": int(A.sum()),
            "column_sums_equal_by_position": int(bb_df.sums_equal.sum()),
            "columns_equal_identity_all_rows": int(bb_df.column_equal_identity_all_rows.sum()),
            "columns_equal_under_sequence_mapping": int(bb_df.column_equal_sequence_mapping_all_mapped_rows.sum()),
            "row_equality_counts_by_shift": shift_counts,
            "released_names_are_positional_integers": bool(all(str(X.columns[c]) == str(c) for c in range(827))),
            "wording": BINARY_WORDING,
        },
        "negative_controls": nc_df[["family", "role", "n_candidate_columns", "n_released_columns_matched",
                                    "n_released_columns_matched_excluding_dcc_duplicates"]].to_dict(orient="records"),
        "verdict": {
            "n_exact": int(status_counts.get("exact", 0)),
            "n_within_tolerance_not_exact": int(status_counts.get("within_abs_1e-6", 0) + status_counts.get("within_rel_1e-6", 0)),
            "n_unmatched": len(unmatched_cols),
            "n_columns_reproduced_by_documented_sequence_operations": int((seq_family_cols & (col_df.status != "unmatched")).sum()),
            "n_columns_matched_only_by_raw_association_matrix": int((col_df.family_assigned == "binary_block_numerically_matches_raw_association_matrix").sum()),
            "full_827_reproducible_from_documented_sequence_operations": bool((seq_family_cols & (col_df.status != "unmatched")).sum() == 827),
            "note": (
                "Sequence-derived columns are compared on released rows that match a raw row "
                "(row mapping from the DCC+k-mer blocks); unmatched released rows are excluded. "
                "The 285-column binary block " + BINARY_WORDING + "."
            ),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    col_df.to_csv(out_dir / "column_audit.csv", index=False)
    rm.to_csv(out_dir / "row_mapping.csv", index=False)
    bb_df.to_csv(out_dir / "binary_block_audit.csv", index=False)
    nc_df.to_csv(out_dir / "negative_controls.csv", index=False)
    summary["output_sha256"] = {n: sha256_file(out_dir / n) for n in OUTPUT_FILES[:-1]}
    (out_dir / "feature_gate_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = run_audit(args.out_dir)
    print(json.dumps({k: summary[k] for k in ("column_status_counts_all_827", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
