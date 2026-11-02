"""Independently revalidate the 73-concept shared V1/V2 disease map against the raw columns.

Read-only audit. Checks (all recorded in disease_map_audit.json; exit code 1 if any fails):
  1 source_columns_exist         every listed source column exists byte-exactly in its raw header
  2 source_columns_normalize     normalized_label(source column) == canonical_name
  3 completeness                 normalized(V1 raw) & normalized(V2 raw) == the 73 canonical names,
                                 and listed columns == all raw columns with that normalized name
  4 no_column_reuse              no raw column is assigned to two canonical names
  5 y_rebuild_equal              OR-rebuilt shared73 Y equals existing v1/v2_shared_73_Y.csv exactly
  6 positives_match              map positives == recomputed (V1 all rows; V2 on the Y-row subset)
Informational: OR-combined concepts (or_combinations.csv) and unmerged near-miss names
(near_misses.csv).
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_felipe_basics_inputs import normalized_label  # noqa: E402
from scripts.prediction_io import atomic_write_text, sha256_file  # noqa: E402

SEP = " | "
DEFAULTS = {
    "v1_raw": ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv",
    "v2_raw": ROOT / "Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv",
    "label_map": ROOT / "config/felipe_basics_shared_73_diseases.csv",
    "v1_y": ROOT / "results/felipe_basics_2026-09-27/inputs/v1_shared_73_Y.csv",
    "v2_y": ROOT / "results/felipe_basics_2026-09-27/inputs/v2_shared_73_Y.csv",
    "outdir": ROOT / "results/felipe_shared_sequence_2026-09-28/disease_map_audit",
}
NON_DISEASE = {"v1": ("", "Unnamed: 0", "ID", "seqs"), "v2": ("ID", "seqs")}
NEAR_MISS_RATIO = 0.9


def raw_header(path: Path) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def load_raw(path: Path, version: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (disease frame indexed by ID, exact disease header names, duplicate header names)."""
    header = raw_header(path)
    duplicates = sorted({name for name in header if header.count(name) > 1})
    frame = pd.read_csv(path, dtype={"ID": str})
    if list(frame.columns) != header:
        # pandas mangles duplicate names; keep going but the duplicate list reports it.
        pass
    frame = frame.set_index("ID")
    disease_cols = [c for c in header if c not in NON_DISEASE[version]]
    frame = frame[[c for c in frame.columns if c not in NON_DISEASE[version]]]
    return frame, disease_cols, duplicates


def split_cols(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "":
        return []
    return str(value).split(SEP)


def depluralize(name: str) -> str:
    return " ".join(tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok
                    for tok in name.split())


def near_misses(v1_names: set[str], v2_names: set[str]) -> pd.DataFrame:
    shared = v1_names & v2_names
    rows = []
    seen = set()
    for side, only, other in (("v1_only", v1_names - shared, v2_names), ("v2_only", v2_names - shared, v1_names)):
        for name in sorted(only):
            for cand in sorted(other):
                if cand == name:
                    continue
                ratio = difflib.SequenceMatcher(None, name, cand).ratio()
                plural = depluralize(name) == depluralize(cand)
                if ratio >= NEAR_MISS_RATIO or plural:
                    v1n, v2n = (name, cand) if side == "v1_only" else (cand, name)
                    if (v1n, v2n) in seen:
                        continue
                    seen.add((v1n, v2n))
                    rows.append({
                        "v1_normalized": v1n,
                        "v2_normalized": v2n,
                        "v1_is_shared": v1n in shared,
                        "v2_is_shared": v2n in shared,
                        "difflib_ratio": round(ratio, 4),
                        "equal_after_plural_strip": plural,
                    })
    cols = ["v1_normalized", "v2_normalized", "v1_is_shared", "v2_is_shared",
            "difflib_ratio", "equal_after_plural_strip"]
    return pd.DataFrame(rows, columns=cols)


def rebuild_y(raw: pd.DataFrame, sources: dict[str, list[str]], order: list[str], ids: list[str]) -> pd.DataFrame:
    sub = raw.loc[ids]
    out = pd.DataFrame(
        {name: (sub[sources[name]].to_numpy() > 0).any(axis=1).astype(int) for name in order},
        index=pd.Index(ids, name="ID"),
    )
    return out


def compare_y(rebuilt: pd.DataFrame, existing: pd.DataFrame) -> dict:
    ids_equal = list(rebuilt.index) == list(existing.index)
    cols_equal = list(rebuilt.columns) == list(existing.columns)
    values_equal = False
    n_diff = None
    if ids_equal and cols_equal:
        diff = rebuilt.to_numpy() != existing.to_numpy()
        n_diff = int(diff.sum())
        values_equal = n_diff == 0
    ex_vals = sorted(int(v) for v in np.unique(existing.to_numpy())) if existing.size else []
    return {
        "ids_equal": ids_equal,
        "columns_equal": cols_equal,
        "values_equal": values_equal,
        "n_value_differences": n_diff,
        "existing_shape": list(existing.shape),
        "rebuilt_shape": list(rebuilt.shape),
        "existing_unique_values": ex_vals,
        "pass": bool(ids_equal and cols_equal and values_equal and set(ex_vals) <= {0, 1}),
    }


def audit(v1_raw_path: Path, v2_raw_path: Path, map_path: Path, v1_y_path: Path, v2_y_path: Path,
          n_expected: int = 73) -> dict:
    v1_raw, v1_cols, v1_dup = load_raw(v1_raw_path, "v1")
    v2_raw, v2_cols, v2_dup = load_raw(v2_raw_path, "v2")
    raws = {"v1": v1_raw, "v2": v2_raw}
    raw_cols = {"v1": v1_cols, "v2": v2_cols}
    dmap = pd.read_csv(map_path, dtype={"canonical_name": str, "v1_source_columns": str, "v2_source_columns": str},
                       keep_default_na=False)
    order = dmap["canonical_name"].tolist()
    sources = {v: {row.canonical_name: split_cols(getattr(row, f"{v}_source_columns")) for row in dmap.itertuples()}
               for v in ("v1", "v2")}

    groups = {}
    for v in ("v1", "v2"):
        g: dict[str, list[str]] = {}
        for c in raw_cols[v]:
            g.setdefault(normalized_label(c), []).append(c)
        groups[v] = g

    checks: dict[str, dict] = {}
    per = {name: {} for name in order}

    # 1 existence
    missing = {v: [] for v in ("v1", "v2")}
    for v in ("v1", "v2"):
        header_set = set(raw_cols[v])
        for name in order:
            miss = [c for c in sources[v][name] if c not in header_set]
            per[name][f"{v}_cols_exist"] = not miss and bool(sources[v][name])
            missing[v] += [{"canonical_name": name, "column": c} for c in miss]
    empty = [(v, n) for v in ("v1", "v2") for n in order if not sources[v][n]]
    checks["1_source_columns_exist"] = {
        "pass": not missing["v1"] and not missing["v2"] and not empty and not v1_dup and not v2_dup,
        "missing": missing, "concepts_with_no_source_columns": empty,
        "duplicate_raw_header_names": {"v1": v1_dup, "v2": v2_dup},
        "n_listed_columns": {v: sum(len(s) for s in sources[v].values()) for v in ("v1", "v2")},
    }

    # 2 normalization
    bad_norm = []
    for v in ("v1", "v2"):
        for name in order:
            ok = True
            for c in sources[v][name]:
                if normalized_label(c) != name:
                    ok = False
                    bad_norm.append({"version": v, "canonical_name": name, "column": c,
                                     "normalized": normalized_label(c)})
            per[name][f"{v}_cols_normalize"] = ok
    checks["2_source_columns_normalize"] = {"pass": not bad_norm, "failures": bad_norm}

    # 3 completeness
    inter = set(groups["v1"]) & set(groups["v2"])
    canon = set(order)
    incomplete = []
    for name in order:
        ok = True
        for v in ("v1", "v2"):
            want = sorted(groups[v].get(name, []))
            have = sorted(sources[v][name])
            if want != have:
                ok = False
                incomplete.append({"version": v, "canonical_name": name,
                                   "unlisted_raw_columns": sorted(set(want) - set(have)),
                                   "extra_listed_columns": sorted(set(have) - set(want))})
        per[name]["complete"] = ok
    checks["3_completeness"] = {
        "pass": inter == canon and len(order) == len(canon) == n_expected and not incomplete,
        "n_v1_raw_disease_columns": len(v1_cols), "n_v2_raw_disease_columns": len(v2_cols),
        "n_v1_normalized_names": len(groups["v1"]), "n_v2_normalized_names": len(groups["v2"]),
        "n_intersection": len(inter), "n_map_concepts": len(order),
        "n_unique_map_concepts": len(canon), "n_expected": n_expected,
        "intersection_minus_map": sorted(inter - canon), "map_minus_intersection": sorted(canon - inter),
        "column_set_mismatches": incomplete,
    }

    # 4 no reuse
    reuse = []
    for v in ("v1", "v2"):
        owner: dict[str, list[str]] = {}
        for name in order:
            for c in sources[v][name]:
                owner.setdefault(c, []).append(name)
        reuse += [{"version": v, "column": c, "canonical_names": ns} for c, ns in owner.items() if len(ns) > 1]
    checks["4_no_column_reuse"] = {"pass": not reuse, "reused": reuse}

    # value domain of raw columns
    nonbinary = []
    for v in ("v1", "v2"):
        for c in raw_cols[v]:
            if c not in raws[v].columns:
                continue
            vals = pd.unique(raws[v][c].to_numpy().ravel())
            bad = sorted(str(x) for x in vals if not (pd.notna(x) and x in (0, 1)))
            if bad:
                nonbinary.append({"version": v, "column": c, "non_binary_values": bad})

    # 5 Y rebuild (only possible when columns exist)
    v1_y = pd.read_csv(v1_y_path, dtype={"ID": str}).set_index("ID")
    v2_y = pd.read_csv(v2_y_path, dtype={"ID": str}).set_index("ID")
    v2_ids = list(v2_y.index)
    rebuild_ok = checks["1_source_columns_exist"]["pass"] and set(v2_ids) <= set(v2_raw.index)
    y5 = {"raw_non_binary_columns": nonbinary, "v1_population": "all raw V1 rows",
          "v2_population": "IDs of existing v2_shared_73_Y.csv joined by ID",
          "v2_ids_missing_from_raw": sorted(set(v2_ids) - set(v2_raw.index))[:20]}
    if rebuild_ok:
        r1 = rebuild_y(v1_raw, sources["v1"], order, list(v1_raw.index))
        r2 = rebuild_y(v2_raw, sources["v2"], order, v2_ids)
        y5["v1"] = compare_y(r1, v1_y)
        y5["v2"] = compare_y(r2, v2_y)
        y5["pass"] = y5["v1"]["pass"] and y5["v2"]["pass"] and not nonbinary
        for name in order:
            per[name]["y_v1_equal"] = bool(name in v1_y.columns and y5["v1"]["ids_equal"]
                                           and (r1[name].to_numpy() == v1_y[name].to_numpy()).all())
            per[name]["y_v2_equal"] = bool(name in v2_y.columns and y5["v2"]["ids_equal"]
                                           and (r2[name].to_numpy() == v2_y[name].to_numpy()).all())
    else:
        y5["pass"] = False
        y5["reason"] = "rebuild impossible: source columns missing or V2 IDs absent from raw"
    checks["5_y_rebuild_equal"] = y5

    # 6 positives
    pos_rows = []
    all_v2 = list(v2_raw.index)
    for row in dmap.itertuples():
        name = row.canonical_name
        entry = {}
        if rebuild_ok:
            def count(raw, v, ids):
                return int((raw.loc[ids, sources[v][name]].to_numpy() > 0).any(axis=1).sum())
            entry = {
                "v1_positives_full": count(v1_raw, "v1", list(v1_raw.index)),
                "v2_positives_full_raw": count(v2_raw, "v2", all_v2),
                "v2_positives_y_subset": count(v2_raw, "v2", v2_ids),
            }
            entry["match_v1"] = entry["v1_positives_full"] == int(row.v1_positives)
            entry["match_v2_full_raw"] = entry["v2_positives_full_raw"] == int(row.v2_positives)
            entry["match_v2_y_subset"] = entry["v2_positives_y_subset"] == int(row.v2_positives)
        per[name].update(entry)
        pos_rows.append({"canonical_name": name, **entry})
    if rebuild_ok:
        n = len(order)
        mv1 = sum(r["match_v1"] for r in pos_rows)
        mfull = sum(r["match_v2_full_raw"] for r in pos_rows)
        msub = sum(r["match_v2_y_subset"] for r in pos_rows)
        pop = ("v2_y_subset" if msub == n else "v2_full_raw" if mfull == n else "neither")
        checks["6_positives_match"] = {
            "pass": mv1 == n and msub == n,
            "n_concepts": n, "v1_matches_full_raw": mv1,
            "v2_matches_full_raw": mfull, "v2_matches_y_subset": msub,
            "n_v2_full_raw_rows": len(all_v2), "n_v2_y_subset_rows": len(v2_ids),
            "n_v1_rows": len(v1_raw),
            "map_v2_positive_population": pop,
            "mismatches": [r for r in pos_rows if not (r["match_v1"] and r["match_v2_y_subset"])],
        }
    else:
        checks["6_positives_match"] = {"pass": False, "reason": "rebuild impossible"}

    # 7 OR combinations (informational)
    or_rows = []
    for name in order:
        if len(sources["v1"][name]) > 1 or len(sources["v2"][name]) > 1:
            rec = {"canonical_name": name,
                   "v1_source_columns": SEP.join(sources["v1"][name]),
                   "v2_source_columns": SEP.join(sources["v2"][name]),
                   "n_v1_cols": len(sources["v1"][name]), "n_v2_cols": len(sources["v2"][name])}
            if rebuild_ok:
                for label, v, raw, ids in (("v1_full", "v1", v1_raw, list(v1_raw.index)),
                                           ("v2_full_raw", "v2", v2_raw, all_v2),
                                           ("v2_y_subset", "v2", v2_raw, v2_ids)):
                    m = raw.loc[ids, sources[v][name]].to_numpy() > 0
                    rec[f"{label}_first_col_positives"] = int(m[:, 0].sum())
                    rec[f"{label}_or_positives"] = int(m.any(axis=1).sum())
                    rec[f"{label}_rows_changed_by_or"] = int((m.any(axis=1) & ~m[:, 0]).sum())
            or_rows.append(rec)
    or_df = pd.DataFrame(or_rows)

    nm = near_misses(set(groups["v1"]), set(groups["v2"]))

    table = []
    for row in dmap.itertuples():
        name = row.canonical_name
        p = per[name]
        table.append({
            "order": int(row.order), "canonical_name": name,
            "v1_source_columns": SEP.join(sources["v1"][name]),
            "v2_source_columns": SEP.join(sources["v2"][name]),
            "n_v1_cols": len(sources["v1"][name]), "n_v2_cols": len(sources["v2"][name]),
            "map_v1_positives": int(row.v1_positives), "map_v2_positives": int(row.v2_positives),
            "v1_positives_full": p.get("v1_positives_full"),
            "v2_positives_full_raw": p.get("v2_positives_full_raw"),
            "v2_positives_y_subset": p.get("v2_positives_y_subset"),
            "v1_cols_exist": p.get("v1_cols_exist"), "v2_cols_exist": p.get("v2_cols_exist"),
            "v1_cols_normalize": p.get("v1_cols_normalize"), "v2_cols_normalize": p.get("v2_cols_normalize"),
            "complete": p.get("complete"),
            "y_v1_equal": p.get("y_v1_equal"), "y_v2_equal": p.get("y_v2_equal"),
            "match_v1_positives": p.get("match_v1"),
            "match_v2_positives_y_subset": p.get("match_v2_y_subset"),
            "match_v2_positives_full_raw": p.get("match_v2_full_raw"),
        })
    return {
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks.values()),
        "table": pd.DataFrame(table),
        "or_combinations": or_df,
        "near_misses": nm,
        "summary_counts": {"n_or_combined": len(or_rows), "n_near_misses": len(nm)},
    }


def write_outputs(result: dict, outdir: Path, inputs: dict[str, Path]) -> dict:
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    files = {
        "disease_map_audit.csv": result["table"],
        "or_combinations.csv": result["or_combinations"],
        "near_misses.csv": result["near_misses"],
    }
    out_hashes = {}
    for fname, df in files.items():
        atomic_write_text(outdir / fname, df.to_csv(index=False))
        out_hashes[fname] = sha256_file(outdir / fname)

    def rel(p: Path) -> str:
        p = Path(p).resolve()
        try:
            return p.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return str(p)

    payload = {
        "schema": "shared_disease_map_audit/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_pass": result["all_pass"],
        "checks": result["checks"],
        "summary_counts": result["summary_counts"],
        "near_miss_rule": f"difflib ratio >= {NEAR_MISS_RATIO} or equal after stripping plural 's' per token",
        "inputs": {rel(p): sha256_file(p) for p in inputs.values()},
        "outputs": out_hashes,
        "code": {rel(Path(__file__)): sha256_file(Path(__file__)),
                 rel(ROOT / "scripts/prepare_felipe_basics_inputs.py"):
                     sha256_file(ROOT / "scripts/prepare_felipe_basics_inputs.py")},
    }
    atomic_write_text(outdir / "disease_map_audit.json",
                      json.dumps(payload, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)) + "\n")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path, default=value)
    args = parser.parse_args(argv)
    inputs = {k: getattr(args, k) for k in ("v1_raw", "v2_raw", "label_map", "v1_y", "v2_y")}
    result = audit(args.v1_raw, args.v2_raw, args.label_map, args.v1_y, args.v2_y)
    payload = write_outputs(result, args.outdir, inputs)
    for name, check in payload["checks"].items():
        print(f"{name}: {'PASS' if check['pass'] else 'FAIL'}")
    print(f"all_pass={payload['all_pass']}  outdir={args.outdir}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
