"""Build the shared-sequence V1/V2 paired cohort (Interface 1 of
Reports/felipe_audit_2026-09-27/SHARED_SEQUENCE_SPEC_2026-09-28.md).

Each RNA is identified by the SHA-256 of its normalized sequence (the frozen
``normalize_sequence`` from ``scripts/reconstruct_paper_sequence_features.py``).
Sequences present in both V1 and V2 form the test cohort; every row whose hash
is shared is removed from both training sets.  Every invariant is asserted and
recorded; a failed invariant raises ``CohortAssertionError`` and nothing is
adjusted to make it pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Callable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prediction_io import atomic_write_text, sha256_file  # noqa: E402
from scripts.reconstruct_paper_sequence_features import (  # noqa: E402
    DEFAULT_PROPERTIES,
    dcc_vector,
    feature_columns,
    kmer_vector,
    load_properties,
    normalize_sequence,
)

SCHEMA = "shared_sequence_cohort/1"
FLOAT_FORMAT = "%.17g"
JOIN = " | "
DEFAULT_ROOT = Path("results/felipe_shared_sequence_2026-09-28")
DEFAULT_V1_RAW = Path("lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv")
DEFAULT_V2_RAW = Path("Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv")
DEFAULT_MANIFEST = Path("config/felipe_basics_transfer_73_manifest.csv")
DEFAULT_LABEL_MAP = Path("config/felipe_basics_shared_73_diseases.csv")
DEFAULT_REPORT_COPY = Path("Reports/felipe_audit_2026-09-27/shared_sequence_correspondence.csv")

REAL_EXPECTED = {
    "n_shared_hashes": 48,
    "n_v2_rows_shared": 52,
    "n_one_to_one": 44,
    "n_train_v1": 307,
    "n_train_v2": 4078,
}

ReconstructFn = Callable[[str], np.ndarray]


class CohortAssertionError(AssertionError):
    """A cohort invariant did not hold."""


class CohortRefusal(RuntimeError):
    """The builder refused to write (existing output, bad inputs)."""


@dataclass
class Assertions:
    results: dict[str, bool] = field(default_factory=dict)

    def check(self, name: str, ok: bool, message: str) -> None:
        self.results[name] = bool(ok)
        if not ok:
            raise CohortAssertionError(f"cohort assertion '{name}' failed: {message}")


# --------------------------------------------------------------------------- helpers


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def removed_string(removed: Counter) -> str:
    return ";".join(f"{symbol}:{count}" for symbol, count in sorted(removed.items()))


def frozen_reconstruction(properties_csv: Path = DEFAULT_PROPERTIES) -> tuple[ReconstructFn, list[str]]:
    """Return the frozen 542-feature function and its column names."""
    names, properties = load_properties(properties_csv)

    def _fn(sequence: str) -> np.ndarray:
        return np.concatenate(
            [dcc_vector(sequence, properties), kmer_vector(sequence, 2), kmer_vector(sequence, 3)]
        )

    return _fn, feature_columns(names)


def read_raw(path: Path) -> pd.DataFrame:
    """Read ``ID`` and ``seqs`` as strings; empty cells stay empty strings."""
    frame = pd.read_csv(path, usecols=["ID", "seqs"], dtype=str, keep_default_na=False)
    if frame["ID"].duplicated().any():
        dup = frame.loc[frame["ID"].duplicated(), "ID"].tolist()[:5]
        raise CohortRefusal(f"{path} has duplicate IDs, e.g. {dup}")
    if (frame["ID"] == "").any():
        raise CohortRefusal(f"{path} has empty IDs")
    return frame


def hash_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """ID -> normalized sequence hash; rows whose normalized sequence is empty are dropped."""
    records = []
    for sample_id, raw in frame[["ID", "seqs"]].itertuples(index=False):
        seq, removed = normalize_sequence(raw)
        if not seq:
            continue
        records.append((sample_id, sequence_hash(seq), seq, len(seq), removed_string(removed)))
    return pd.DataFrame(records, columns=["ID", "hash", "seq", "length", "removed"]).set_index("ID")


def read_x(path: Path) -> pd.DataFrame:
    index_name = pd.read_csv(path, nrows=0).columns[0]
    frame = pd.read_csv(
        path,
        index_col=0,
        dtype={index_name: str},
        keep_default_na=False,
        float_precision="round_trip",
    )
    frame.index = frame.index.astype(str)
    if frame.index.duplicated().any():
        raise CohortRefusal(f"{path} has duplicate row IDs")
    frame = frame.astype(np.float64)
    return frame


def read_y(path: Path) -> pd.DataFrame:
    index_name = pd.read_csv(path, nrows=0).columns[0]
    frame = pd.read_csv(path, index_col=0, dtype={index_name: str}, keep_default_na=False)
    frame.index = frame.index.astype(str)
    if frame.index.duplicated().any():
        raise CohortRefusal(f"{path} has duplicate row IDs")
    values = frame.to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise CohortRefusal(f"{path} labels must be 0/1")
    return frame.astype(np.int64)


def row_bytes(frame: pd.DataFrame) -> list[bytes]:
    values = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
    return [values[i].tobytes() for i in range(values.shape[0])]


def csv_text(frame: pd.DataFrame, *, index: bool = True) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=index, float_format=FLOAT_FORMAT, lineterminator="\n")
    return buffer.getvalue()


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


# --------------------------------------------------------------------------- fromserver


RNA_KEYWORDS = ("overlap", "shared", "mapping", "map", "correspond", "match", "sequence", "seqs", "seq")
PAIR_ID_HINTS = (("v1", "id"), ("v2", "id"))
MAX_CANDIDATE_BYTES = 50 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024


def _is_id_col(name: str, version: str) -> bool:
    low = name.lower()
    return version in low and ("id" in low or "rna" in low or "name" in low) and "disease" not in low


def inspect_fromserver(
    fromserver: Path,
    correspondence: pd.DataFrame,
) -> dict[str, object]:
    """Read-only scan of the server bundle for V1<->V2 RNA mappings."""
    result: dict[str, object] = {
        "fromserver_path": str(fromserver.resolve()) if fromserver.exists() else str(fromserver),
        "exists": fromserver.exists(),
        "read_only": True,
        "method": (
            "walked every file; read only the first line of CSV/TSV files; parsed JSON files "
            f"<= {MAX_JSON_BYTES} bytes for keys mentioning overlap/mapping/shared/removed ids; "
            "a CSV counts as an RNA pair mapping only if it has a V1 ID column and a V2 ID column "
            "(disease-name maps are listed separately)."
        ),
    }
    if not fromserver.exists():
        result["conclusion"] = "fromserver directory not found; nothing scanned."
        return result

    v1_ids = set(correspondence["v1_id"])
    pairs_new: set[tuple[str, str]] = set()
    all_v2_ids: set[str] = set()
    for row in correspondence.itertuples(index=False):
        for v2_id in row.v2_ids.split(JOIN):
            pairs_new.add((row.v1_id, v2_id))
            all_v2_ids.add(v2_id)

    files = sorted(p for p in fromserver.rglob("*") if p.is_file())
    tabular_scanned = 0
    json_scanned = 0
    pair_candidates: list[dict[str, object]] = []
    rna_overlap_count_files: list[dict[str, object]] = []
    disease_level_files: list[dict[str, object]] = []
    id_list_findings: dict[str, dict[str, object]] = {}
    keyword_named_files = 0

    for path in files:
        rel = path.relative_to(fromserver).as_posix()
        low_rel = rel.lower()
        if any(k in path.name.lower() for k in RNA_KEYWORDS):
            keyword_named_files += 1
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            tabular_scanned += 1
            sep = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                header_line = handle.readline().rstrip("\r\n")
            try:
                columns = pd.read_csv(io.StringIO(header_line), sep=sep, nrows=0).columns.tolist()
            except Exception:  # pragma: no cover - malformed header
                columns = header_line.split(sep)
            low_cols = [str(c).lower() for c in columns]
            v1_cols = [c for c in columns if _is_id_col(str(c), "v1")]
            v2_cols = [c for c in columns if _is_id_col(str(c), "v2")]
            has_keyword_col = any(
                k in c for c in low_cols for k in ("overlap", "mapping", "shared", "seq")
            )
            is_disease = any("disease" in c or c == "canonical_name" for c in low_cols)
            if v1_cols and v2_cols and not is_disease:
                entry: dict[str, object] = {"path": rel, "columns": [str(c) for c in columns]}
                if path.stat().st_size <= MAX_CANDIDATE_BYTES:
                    frame = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
                    entry["n_rows"] = int(len(frame))
                    found = {(a, b) for a, b in zip(frame[v1_cols[0]], frame[v2_cols[0]])}
                    agree = sorted(found & pairs_new)
                    entry["id_columns_used"] = [str(v1_cols[0]), str(v2_cols[0])]
                    entry["n_pairs_agree_with_new_correspondence"] = len(agree)
                    entry["pairs_not_in_new_correspondence"] = [list(p) for p in sorted(found - pairs_new)]
                    entry["new_pairs_missing_from_file"] = [list(p) for p in sorted(pairs_new - found)]
                else:
                    entry["n_rows"] = None
                    entry["note"] = "file too large; not read beyond header"
                pair_candidates.append(entry)
            elif is_disease and any(k in low_rel for k in ("shared", "map", "alias", "canonical")):
                disease_level_files.append({"path": rel, "columns": [str(c) for c in columns]})
            elif has_keyword_col and "overlap" in " ".join(low_cols):
                frame = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False) if path.stat().st_size <= MAX_CANDIDATE_BYTES else None
                overlap_cols = [c for c in columns if "overlap" in str(c).lower()]
                summary = {"path": rel, "columns": [str(c) for c in columns]}
                if frame is not None:
                    summary["n_rows"] = int(len(frame))
                    summary["overlap_column_values"] = {
                        str(c): sorted(set(frame[c].tolist())) for c in overlap_cols
                    }
                summary["contains_id_pairs"] = False
                rna_overlap_count_files.append(summary)
        elif suffix == ".json" and path.stat().st_size <= MAX_JSON_BYTES:
            json_scanned += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except ValueError:
                continue
            stack = [payload]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    for key, value in node.items():
                        lk = str(key).lower()
                        if (
                            any(k in lk for k in ("overlap", "mapping", "shared_rna", "removed"))
                            and isinstance(value, list)
                            and all(isinstance(v, str) for v in value)
                        ):
                            ids = tuple(sorted(value))
                            finding = id_list_findings.setdefault(
                                f"{key}={json.dumps(list(ids))}",
                                {"key": key, "ids": list(ids), "n_files": 0, "example_file": rel},
                            )
                            finding["n_files"] += 1
                        elif isinstance(value, (dict, list)):
                            stack.append(value)
                elif isinstance(node, list):
                    stack.extend(v for v in node if isinstance(v, (dict, list)))

    for finding in id_list_findings.values():
        ids = set(finding["ids"])
        finding["ids_that_are_new_v1_ids"] = sorted(ids & v1_ids)
        finding["ids_that_are_new_v2_ids"] = sorted(ids & all_v2_ids)
        finding["ids_not_in_new_correspondence"] = sorted(ids - v1_ids - all_v2_ids)
        finding["note"] = (
            "ID list only (no V1->V2 pairs); covers "
            f"{len(ids & v1_ids & all_v2_ids)} of the {len(correspondence)} new shared-sequence pairs "
            "by identical ID."
        )

    result.update(
        {
            "n_files_scanned": len(files),
            "n_tabular_headers_read": tabular_scanned,
            "n_json_parsed": json_scanned,
            "n_files_with_keyword_in_name": keyword_named_files,
            "rna_pair_mapping_files": pair_candidates,
            "rna_overlap_count_only_files": rna_overlap_count_files,
            "json_id_lists": sorted(id_list_findings.values(), key=lambda d: d["example_file"]),
            "disease_level_map_files": disease_level_files,
            "n_new_correspondence_pairs": len(pairs_new),
        }
    )
    if pair_candidates:
        result["conclusion"] = f"{len(pair_candidates)} candidate V1<->V2 RNA mapping file(s) found; see agreement fields."
    else:
        result["conclusion"] = (
            "No file in fromserver maps V1 IDs to V2 IDs or records shared sequences. "
            "Only overlap counts and ID lists exist (see rna_overlap_count_only_files, json_id_lists); "
            "the 48-hash sequence correspondence cannot be cross-checked pairwise against the server bundle."
        )
    return result


# --------------------------------------------------------------------------- core


def build_cohort(
    *,
    repo_root: Path,
    out_root: Path,
    v1_raw: Path,
    v2_raw: Path,
    transfer_manifest: Path,
    label_map: Path,
    fromserver: Path | None,
    expected: Mapping[str, int] = REAL_EXPECTED,
    reconstruct_fn: ReconstructFn | None = None,
    expected_feature_columns: list[str] | None = None,
    extra_inputs: Mapping[str, Path] | None = None,
    report_copy: Path | None = None,
    builder_path: Path = Path(__file__).resolve(),
) -> dict[str, object]:
    """Build ``<out_root>/cohort``. Returns the cohort manifest dict.

    ``reconstruct_fn`` defaults to the frozen reconstruction; tests may inject a
    synthetic one.  ``expected`` holds the exact counts asserted.
    """
    cohort_dir = out_root / "cohort"
    if cohort_dir.exists() and any(cohort_dir.iterdir()):
        raise CohortRefusal(f"refusing to write into non-empty cohort directory {cohort_dir}")
    if report_copy is not None and report_copy.exists():
        raise CohortRefusal(f"refusing to overwrite existing report copy {report_copy}")

    extra_inputs = dict(extra_inputs or {})
    if reconstruct_fn is None:
        properties_csv = DEFAULT_PROPERTIES
        reconstruct_fn, expected_feature_columns = frozen_reconstruction(properties_csv)
        extra_inputs.setdefault("properties_csv", properties_csv)
        reconstruction_mode = "frozen:scripts/reconstruct_paper_sequence_features.py"
    else:
        reconstruction_mode = "injected"

    checks = Assertions()

    # ---- manifest, X, Y, label map
    manifest = pd.read_csv(transfer_manifest, dtype=str, keep_default_na=False)
    manifest = manifest.set_index("version")
    for version in ("v1", "v2"):
        if version not in manifest.index:
            raise CohortRefusal(f"{transfer_manifest} has no row for {version}")
    x_paths = {v: resolve(repo_root, manifest.at[v, "x_path"]) for v in ("v1", "v2")}
    y_paths = {v: resolve(repo_root, manifest.at[v, "y_path"]) for v in ("v1", "v2")}
    for v in ("v1", "v2"):
        checks.check(
            f"manifest_x_sha256_{v}",
            sha256_file(x_paths[v]) == manifest.at[v, "x_sha256"],
            f"{x_paths[v]} hash differs from transfer manifest",
        )
        checks.check(
            f"manifest_y_sha256_{v}",
            sha256_file(y_paths[v]) == manifest.at[v, "y_sha256"],
            f"{y_paths[v]} hash differs from transfer manifest",
        )
        if "label_map_sha256" in manifest.columns and manifest.at[v, "label_map_sha256"]:
            checks.check(
                f"manifest_label_map_sha256_{v}",
                sha256_file(label_map) == manifest.at[v, "label_map_sha256"],
                f"{label_map} hash differs from transfer manifest",
            )

    label_frame = pd.read_csv(label_map, dtype={"canonical_name": str}, keep_default_na=False)
    label_frame["order"] = label_frame["order"].astype(int)
    label_order = label_frame.sort_values("order")["canonical_name"].tolist()

    X = {v: read_x(x_paths[v]) for v in ("v1", "v2")}
    Y = {v: read_y(y_paths[v]) for v in ("v1", "v2")}
    for v in ("v1", "v2"):
        checks.check(
            f"y_columns_equal_label_map_{v}",
            Y[v].columns.tolist() == label_order,
            f"{y_paths[v]} columns differ from label-map order",
        )
        checks.check(
            f"y_index_equals_x_index_{v}",
            Y[v].index.tolist() == X[v].index.tolist(),
            f"{y_paths[v]} row IDs differ from {x_paths[v]}",
        )
    checks.check(
        "x_columns_equal_v1_v2",
        X["v1"].columns.tolist() == X["v2"].columns.tolist(),
        "V1 and V2 X column orders differ",
    )
    feature_cols = X["v1"].columns.tolist()
    if expected_feature_columns is not None:
        checks.check(
            "x_columns_equal_reconstruction_columns",
            feature_cols == list(expected_feature_columns),
            "X columns differ from reconstruction feature_columns",
        )

    # ---- raw sequences and hashes
    raw = {"v1": read_raw(v1_raw), "v2": read_raw(v2_raw)}
    hashed = {v: hash_raw(raw[v]) for v in ("v1", "v2")}
    n_missing = {v: int(len(raw[v]) - len(hashed[v])) for v in ("v1", "v2")}
    for v in ("v1", "v2"):
        missing_x = sorted(set(X[v].index) - set(hashed[v].index))
        checks.check(
            f"every_x_id_has_raw_sequence_{v}",
            not missing_x,
            f"{len(missing_x)} {v} X IDs lack a non-empty raw sequence, e.g. {missing_x[:5]}",
        )

    shared = sorted(set(hashed["v1"]["hash"]) & set(hashed["v2"]["hash"]))
    checks.check(
        "n_shared_hashes",
        len(shared) == expected["n_shared_hashes"],
        f"found {len(shared)} shared hashes, expected {expected['n_shared_hashes']}",
    )
    shared_set = set(shared)
    groups = {
        v: hashed[v][hashed[v]["hash"].isin(shared_set)]
        .reset_index()
        .groupby("hash")["ID"]
        .apply(lambda s: sorted(s.tolist()))
        .to_dict()
        for v in ("v1", "v2")
    }
    multi_v1 = {h: ids for h, ids in groups["v1"].items() if len(ids) != 1}
    checks.check(
        "v1_one_row_per_shared_hash",
        not multi_v1,
        f"{len(multi_v1)} shared hashes have != 1 V1 row: {multi_v1}",
    )
    n_v2_rows = sum(len(ids) for ids in groups["v2"].values())
    checks.check(
        "n_v2_rows_shared",
        n_v2_rows == expected["n_v2_rows_shared"],
        f"found {n_v2_rows} V2 rows for shared hashes, expected {expected['n_v2_rows_shared']}",
    )
    n_one = sum(1 for ids in groups["v2"].values() if len(ids) == 1)
    checks.check(
        "n_one_to_one",
        n_one == expected["n_one_to_one"],
        f"found {n_one} one-to-one hashes, expected {expected['n_one_to_one']}",
    )

    excluded = {v: sorted(i for ids in groups[v].values() for i in ids) for v in ("v1", "v2")}
    for v in ("v1", "v2"):
        not_in_x = [i for i in excluded[v] if i not in X[v].index]
        checks.check(
            f"shared_ids_in_x_{v}",
            not not_in_x,
            f"{v} shared-hash IDs missing from X: {not_in_x}",
        )

    x_pos = {v: {sample_id: i for i, sample_id in enumerate(X[v].index)} for v in ("v1", "v2")}

    # ---- per-hash correspondence and feature checks
    records = []
    all_v1_v2_equal = True
    all_recon = True
    recon_failures: list[str] = []
    v1_label_rows: dict[str, pd.Series] = {}
    v2_primary_rows: dict[str, pd.Series] = {}
    v2_or_rows: dict[str, pd.Series] = {}
    conflict_rows: list[tuple[str, str, str, int]] = []
    for h in shared:
        v1_id = groups["v1"][h][0]
        v2_ids = groups["v2"][h]
        seq = hashed["v1"].at[v1_id, "seq"]
        v1_vec = X["v1"].loc[v1_id].to_numpy(dtype=np.float64)
        v2_vecs = [X["v2"].loc[i].to_numpy(dtype=np.float64) for i in v2_ids]
        equal_v1_v2 = all(v1_vec.tobytes() == vec.tobytes() for vec in v2_vecs)
        all_v1_v2_equal &= equal_v1_v2
        rebuilt = np.asarray(reconstruct_fn(seq), dtype=np.float64)
        recon_ok = rebuilt.shape == v1_vec.shape and all(
            np.allclose(vec, rebuilt, rtol=0, atol=1e-12) for vec in [v1_vec, *v2_vecs]
        )
        # every V2 row of this hash must also rebuild from its own normalized sequence
        for i in v2_ids:
            recon_ok &= hashed["v2"].at[i, "seq"] == seq
        if not recon_ok:
            recon_failures.append(h)
        all_recon &= recon_ok

        y1 = Y["v1"].loc[v1_id, label_order]
        y2 = Y["v2"].loc[v2_ids, label_order]
        y2_or = y2.max(axis=0).astype(np.int64)
        conflict_diseases = [d for d in label_order if y2[d].nunique() > 1]
        for d in conflict_diseases:
            for i in v2_ids:
                conflict_rows.append((h, d, i, int(y2.at[i, d])))
        one_to_one = len(v2_ids) == 1
        y2_primary = y2.iloc[0] if one_to_one else y2_or
        disagreements = int((y1.to_numpy() != y2_primary.to_numpy()).sum())

        v1_label_rows[h] = y1
        v2_or_rows[h] = y2_or
        if one_to_one:
            v2_primary_rows[h] = y2.iloc[0]
        v2_removed = [hashed["v2"].at[i, "removed"] for i in v2_ids]
        records.append(
            {
                "sequence_sha256": h,
                "sequence_length": int(len(seq)),
                "v1_id": v1_id,
                "v1_x_row_index": x_pos["v1"][v1_id],
                "v2_ids": JOIN.join(v2_ids),
                "n_v2_ids": len(v2_ids),
                "v2_x_row_indices": JOIN.join(str(x_pos["v2"][i]) for i in v2_ids),
                "status": "one_to_one" if one_to_one else "ambiguous_multi_v2_id",
                "same_id_v1_v2": v1_id in v2_ids,
                "features_v1_equal_v2": equal_v1_v2,
                "features_equal_reconstruction": bool(recon_ok),
                "v1_ambiguity_removed": hashed["v1"].at[v1_id, "removed"],
                "v2_ambiguity_removed": JOIN.join(v2_removed) if any(v2_removed) else "",
                "n_v2_label_conflicts": len(conflict_diseases),
                "v2_conflict_diseases": JOIN.join(conflict_diseases),
                "n_v1_v2_label_disagreements_primary": disagreements,
                "in_primary": one_to_one,
                "in_sensitivity": True,
            }
        )
    correspondence = pd.DataFrame(records)
    checks.check(
        "features_v1_equal_v2_all",
        all_v1_v2_equal,
        "V1 and V2 542-vectors differ for: "
        + str(correspondence.loc[~correspondence["features_v1_equal_v2"], "sequence_sha256"].tolist()),
    )
    checks.check(
        "features_equal_reconstruction_all",
        all_recon,
        f"reconstructed 542-vectors differ from X rows for hashes {recon_failures}",
    )

    # ---- training split
    train_ids: dict[str, list[str]] = {}
    for v, key in (("v1", "n_train_v1"), ("v2", "n_train_v2")):
        drop = set(excluded[v])
        train_ids[v] = [i for i in X[v].index if i not in drop]
        checks.check(
            f"train_plus_excluded_equals_x_{v}",
            sorted(train_ids[v] + excluded[v]) == sorted(X[v].index)
            and len(train_ids[v]) + len(excluded[v]) == len(X[v]),
            f"{v} train+excluded does not partition the X index",
        )
        checks.check(
            key,
            len(train_ids[v]) == expected[key],
            f"{v} has {len(train_ids[v])} training rows, expected {expected[key]}",
        )
        leaked = sorted(set(hashed[v].loc[train_ids[v], "hash"]) & shared_set)
        checks.check(
            f"no_train_test_hash_overlap_{v}",
            not leaked,
            f"{v} training IDs share {len(leaked)} sequence hashes with test",
        )

    test_X = X["v1"].loc[correspondence["v1_id"].tolist()].copy()
    test_X.index = pd.Index(correspondence["sequence_sha256"].tolist(), name="sequence_sha256")
    test_bytes = set(row_bytes(test_X))
    for v in ("v1", "v2"):
        train_bytes = set(row_bytes(X[v].loc[train_ids[v]]))
        overlap = len(train_bytes & test_bytes)
        checks.check(
            f"no_exact_x_row_overlap_{v}",
            overlap == 0,
            f"{overlap} test X rows are byte-identical to {v} training X rows",
        )

    # ---- assemble outputs
    def label_frame_from(rows: dict[str, pd.Series], keys: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame([rows[k].to_numpy() for k in keys], columns=label_order, dtype=np.int64)
        frame.index = pd.Index(keys, name="sequence_sha256")
        return frame

    primary_keys = correspondence.loc[correspondence["in_primary"], "sequence_sha256"].tolist()
    labels_v1 = label_frame_from(v1_label_rows, shared)
    labels_v2_primary = label_frame_from(v2_primary_rows, primary_keys)
    labels_v2_or = label_frame_from(v2_or_rows, shared)
    cohorts = correspondence[["sequence_sha256", "in_primary", "in_sensitivity"]]
    conflicts = pd.DataFrame(conflict_rows, columns=["sequence_sha256", "disease", "v2_id", "label"])
    test_x_text = csv_text(test_X)

    reread = pd.read_csv(io.StringIO(test_x_text), index_col=0, float_precision="round_trip")
    checks.check(
        "test_x_equals_v1_x_rows",
        reread.columns.tolist() == feature_cols
        and reread.index.tolist() == shared
        and reread.to_numpy(dtype=np.float64).tobytes()
        == X["v1"].loc[correspondence["v1_id"]].to_numpy(dtype=np.float64).tobytes(),
        "written test_X does not round-trip to the V1 X rows",
    )

    fromserver_info = (
        inspect_fromserver(fromserver, correspondence)
        if fromserver is not None
        else {"exists": False, "conclusion": "no fromserver path given"}
    )

    correspondence_text = csv_text(correspondence, index=False)
    outputs_text = {
        "correspondence.csv": correspondence_text,
        "test_X.csv": test_x_text,
        "labels_v1.csv": csv_text(labels_v1),
        "labels_v2_primary.csv": csv_text(labels_v2_primary),
        "labels_v2_or.csv": csv_text(labels_v2_or),
        "cohorts.csv": csv_text(cohorts, index=False),
        "train_ids_v1.txt": "".join(f"{i}\n" for i in train_ids["v1"]),
        "train_ids_v2.txt": "".join(f"{i}\n" for i in train_ids["v2"]),
        "excluded_ids_v1.txt": "".join(f"{i}\n" for i in excluded["v1"]),
        "excluded_ids_v2.txt": "".join(f"{i}\n" for i in excluded["v2"]),
        "v2_label_conflicts.csv": csv_text(conflicts, index=False),
        "fromserver_check.json": json.dumps(fromserver_info, indent=2, sort_keys=False) + "\n",
    }

    def rel(path: Path) -> str:
        path = Path(path).resolve()
        try:
            return path.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    inputs = {
        rel(v1_raw): sha256_file(v1_raw),
        rel(v2_raw): sha256_file(v2_raw),
        rel(x_paths["v1"]): sha256_file(x_paths["v1"]),
        rel(x_paths["v2"]): sha256_file(x_paths["v2"]),
        rel(y_paths["v1"]): sha256_file(y_paths["v1"]),
        rel(y_paths["v2"]): sha256_file(y_paths["v2"]),
        rel(label_map): sha256_file(label_map),
        rel(transfer_manifest): sha256_file(transfer_manifest),
    }
    for path in extra_inputs.values():
        inputs[rel(path)] = sha256_file(Path(path))

    manifest_out = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {
            "n_v1_raw_rows": int(len(raw["v1"])),
            "n_v2_raw_rows": int(len(raw["v2"])),
            "n_v1_missing_sequence": n_missing["v1"],
            "n_v2_missing_sequence": n_missing["v2"],
            "n_v1_x_rows": int(len(X["v1"])),
            "n_v2_x_rows": int(len(X["v2"])),
            "n_shared_hashes": len(shared),
            "n_v1_rows_shared": len(excluded["v1"]),
            "n_v2_rows_shared": len(excluded["v2"]),
            "n_one_to_one": n_one,
            "n_ambiguous_multi_v2_id": len(shared) - n_one,
            "n_same_id_v1_v2": int(correspondence["same_id_v1_v2"].sum()),
            "n_train_v1": len(train_ids["v1"]),
            "n_train_v2": len(train_ids["v2"]),
            "n_labels": len(label_order),
            "n_features": len(feature_cols),
            "n_v2_label_conflict_diseases_total": int(correspondence["n_v2_label_conflicts"].sum()),
            "n_v1_v2_label_disagreements_primary_total": int(
                correspondence["n_v1_v2_label_disagreements_primary"].sum()
            ),
        },
        "expected_counts": dict(expected),
        "hash_definition": "sha256(normalize_sequence(seqs)[0].encode('utf-8')); empty normalized sequence = missing, not hashed",
        "reconstruction_check": {
            "mode": reconstruction_mode,
            "comparison": "np.allclose(x_row, rebuilt, rtol=0, atol=1e-12)",
        },
        "x_read": "pandas.read_csv(float_precision='round_trip'); row-overlap compared on float64 row bytes",
        "inputs": inputs,
        "outputs": {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in outputs_text.items()},
        "assertions": checks.results,
        "code": {
            rel(builder_path): sha256_file(builder_path),
            rel(ROOT / "scripts/reconstruct_paper_sequence_features.py"): sha256_file(
                ROOT / "scripts/reconstruct_paper_sequence_features.py"
            ),
        },
    }

    cohort_dir.mkdir(parents=True, exist_ok=True)
    for name, text in outputs_text.items():
        atomic_write_text(cohort_dir / name, text)
    for name, digest in manifest_out["outputs"].items():
        if sha256_file(cohort_dir / name) != digest:
            raise CohortAssertionError(f"written file {name} hash mismatch")
    atomic_write_text(cohort_dir / "cohort_manifest.json", json.dumps(manifest_out, indent=2) + "\n")
    if report_copy is not None:
        report_copy.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(report_copy, correspondence_text)
    return manifest_out


# --------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--v1-raw", type=Path, default=DEFAULT_V1_RAW)
    parser.add_argument("--v2-raw", type=Path, default=DEFAULT_V2_RAW)
    parser.add_argument("--transfer-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--fromserver", type=Path, default=ROOT.parent / "fromserver")
    parser.add_argument(
        "--report-copy",
        type=Path,
        default=None,
        help=f"optional copy of correspondence.csv (e.g. {DEFAULT_REPORT_COPY})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_cohort(
        repo_root=ROOT,
        out_root=resolve(ROOT, args.root),
        v1_raw=resolve(ROOT, args.v1_raw),
        v2_raw=resolve(ROOT, args.v2_raw),
        transfer_manifest=resolve(ROOT, args.transfer_manifest),
        label_map=resolve(ROOT, args.label_map),
        fromserver=resolve(ROOT, args.fromserver),
        report_copy=resolve(ROOT, args.report_copy) if args.report_copy else None,
    )
    print(json.dumps({"counts": manifest["counts"], "assertions": manifest["assertions"]}, indent=2))


if __name__ == "__main__":
    main()
