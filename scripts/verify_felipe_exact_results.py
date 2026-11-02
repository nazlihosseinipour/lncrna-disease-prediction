#!/usr/bin/env python3
"""Verify inputs and prediction cells for Felipe's V1/V2 experiment.

Checks (``--check``):
  manifests  frozen 542-feature manifests, V2 population ledger, splits, label map,
             transfer overlap, and frozen source hashes (gate G2/G3)
  cells      every prediction cell under --results-root is intact and matches the
             current manifests, frozen sources, and paper-profile parameters
  matrix     the expected cells for --scope exist and pass the cell check (gate G6)
  report     a final report states the required facts and no stale claims (gate G7)
  reference  released 827-column V1 artifact reference check (decision D9); this is
             reproduction evidence only and is not part of four-direction acceptance.
             ``protocol`` is kept as an alias.

It reads artifacts only; it never fits models or modifies results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from scripts.experiment_contract import validate_manifest_row  # noqa: E402
from scripts.prediction_io import (  # noqa: E402
    CONTRACT_SUFFIX,
    FrozenSourceError,
    PredictionCellError,
    read_prediction_cell,
    sha256_lines,
    verify_frozen_sources,
)

WITHIN_MANIFEST = "config/felipe_basics_sequence_542_manifest.csv"
TRANSFER_MANIFEST = "config/felipe_basics_transfer_73_manifest.csv"
LABEL_MAP = "config/felipe_basics_shared_73_diseases.csv"
LEDGER = "config/v2_missing_sequence_ledger.csv"
FROZEN = "config/felipe_frozen_sources.csv"
V2_RAW = "Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv"
V2_GROUP_SPLIT = "results/felipe_basics_2026-09-27/inputs/v2_all_436_group_splits.csv"
V1_SPLIT = "lncRNA_CIBCB2025-main/create_dataset/datasets/splits/inductive/inductive_splits.csv"
V1_SPLIT_SHA256 = "e9fe2eb5fc4ec928a005b4d000dc3ba8623868498b157b21cfb03579356e1e5a"
MODELS = ("rf", "ipcarf", "rflda")
EXPECTED_WITHIN = {
    ("v1", "all44"): (355, 44, "v1_paper_355"),
    ("v2", "all436"): (4130, 436, "v2_sequence_available_4130"),
    ("v1", "shared73"): (355, 73, "v1_paper_355"),
    ("v2", "shared73"): (4130, 73, "v2_sequence_available_4130"),
}
EXPECTED_OVERLAP = ["NR_024015"]
SCOPES = {
    "within6": [("within", "v1", "all44"), ("within", "v2", "all436")],
    "within": [("within", v, ls) for v, ls in EXPECTED_WITHIN],
    "all": [("within", v, ls) for v, ls in EXPECTED_WITHIN]
    + [("transfer", "v1", "v2"), ("transfer", "v2", "v1")],
}

# Reference-artifact constants (decision D9).
AUDIT = ROOT / "results/audit/felipe_exact_results_2026-09-27"
PAPER = ROOT / "Inductive_models_for_structured_output_prediction_of_lncRNA-disease_associations.pdf"
X_PATH = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/features/features_preprocessed.csv"
Y_PATH = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/flat_labels/labels_flat_preprocessed.csv"
SPLIT_PATH = ROOT / V1_SPLIT
RAW_PATH = ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv"
EXPECTED_HASHES = {
    PAPER: "078b6b13a098c631adef6efe5661f6f02fb2fc5976144a839b36ab095d7c008d",
    X_PATH: "b264dd75545ca424f1db96622555066791c52ba6105001770ae869e211428887",
    Y_PATH: "53c3573b72be6a893eba2a31e59b29899fba837986c9545e3115133048817e9e",
    SPLIT_PATH: V1_SPLIT_SHA256,
}


class VerificationError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    return frame


# --------------------------------------------------------------------------- manifests


def _split_test_ids(split_path: Path, ids: list[str], n_splits: int = 10) -> dict[int, list[str]]:
    splits = pd.read_csv(split_path)
    return {
        fold: [ids[i] for i in splits[f"test_fold{fold}"].dropna().astype(int)]
        for fold in range(1, n_splits + 1)
    }


def check_manifests(root: Path = ROOT) -> dict[str, object]:
    within = pd.read_csv(root / WITHIN_MANIFEST)
    require(
        sorted(zip(within.version, within.label_space)) == sorted(EXPECTED_WITHIN),
        f"{WITHIN_MANIFEST} must have exactly the rows {sorted(EXPECTED_WITHIN)}",
    )
    feature_columns = None
    loaded: dict[tuple[str, str], object] = {}
    for row in within.to_dict("records"):
        key = (row["version"], row["label_space"])
        n_rows, n_labels, population = EXPECTED_WITHIN[key]
        require(row["population"] == population, f"{key} population is {row['population']}")
        require(row["feature_key"] == "paper_sequence_542", f"{key} feature_key changed")
        require(row["n_features"] == 542, f"{key} must have 542 features")
        for field in ("x_sha256", "y_sha256", "split_sha256"):
            require(isinstance(row.get(field), str) and len(row[field]) == 64, f"{key} lacks {field}")
        experiment = validate_manifest_row(row, project_root=root, n_splits=10)
        require(experiment.x.shape == (n_rows, 542), f"{key} X shape {experiment.x.shape}")
        require(experiment.y.shape == (n_rows, n_labels), f"{key} Y shape {experiment.y.shape}")
        columns = experiment.x.columns.tolist()
        feature_columns = feature_columns or columns
        require(columns == feature_columns, f"{key} feature names/order differ from V1 all44")
        expected_split = V1_SPLIT if row["version"] == "v1" else V2_GROUP_SPLIT
        require(row["split_path"] == expected_split, f"{key} must use {expected_split}")
        loaded[key] = experiment

    require(sha256(root / V1_SPLIT) == V1_SPLIT_SHA256, "released V1 split hash changed")
    for version in ("v1", "v2"):
        require(
            loaded[(version, "all44" if version == "v1" else "all436")].x.index.tolist()
            == loaded[(version, "shared73")].x.index.tolist(),
            f"{version} label spaces use different rows",
        )

    # V2 group split: no normalized-sequence group or identical X row crosses a fold.
    metadata = json.loads((root / (V2_GROUP_SPLIT + ".metadata.json")).read_text())
    require(metadata["output"]["sha256"] == sha256(root / V2_GROUP_SPLIT), "group split hash drifted")
    v2 = loaded[("v2", "all436")]
    require(metadata["inputs"]["x"]["sha256"] == sha256(v2.x_path), "group split built from other X")
    row_hash = pd.Series(
        [hashlib.sha256(r.tobytes()).hexdigest() for r in v2.x.to_numpy(dtype=float)],
        index=v2.x.index,
    )
    for fold, test_ids in _split_test_ids(root / V2_GROUP_SPLIT, v2.x.index.tolist()).items():
        test = set(row_hash.loc[test_ids])
        train = set(row_hash.drop(index=test_ids))
        require(not test & train, f"V2 fold {fold}: identical X rows cross train/test")

    # Label map and transfer manifest.
    label_map = pd.read_csv(root / LABEL_MAP)
    require(len(label_map) == 73 and label_map["order"].tolist() == list(range(73)), "label map")
    canonical = label_map["canonical_name"].tolist()
    for version in ("v1", "v2"):
        require(
            loaded[(version, "shared73")].y.columns.tolist() == canonical,
            f"{version} shared73 columns differ from the label map order",
        )
        require(
            [int(v) for v in loaded[(version, "shared73")].y.sum()] == label_map[f"{version}_positives"].tolist(),
            f"{version} shared73 positive counts differ from the label map",
        )
    transfer = pd.read_csv(root / TRANSFER_MANIFEST)
    require(transfer.version.tolist() == ["v1", "v2"], "transfer manifest rows")
    for row in transfer.to_dict("records"):
        key = (row["version"], "shared73")
        match = within[(within.version == row["version"]) & (within.label_space == "shared73")].iloc[0]
        for field in ("x_path", "y_path", "x_sha256", "y_sha256", "population", "feature_key"):
            require(row[field] == match[field], f"transfer {row['version']} {field} differs from within")
        require(row["label_space"] == "shared73" and row["n_labels"] == 73, "transfer label space")
        require(row["label_map_path"] == LABEL_MAP, "transfer label map path")
        require(row["label_map_sha256"] == sha256(root / LABEL_MAP), "transfer label map hash")
        require(loaded[key].x.shape[0] == row["n_samples"], "transfer row count")
    overlap = sorted(set(loaded[("v1", "all44")].x.index) & set(v2.x.index))
    require(overlap == EXPECTED_OVERLAP, f"V1/V2 exact overlap is {overlap}, expected {EXPECTED_OVERLAP}")

    # V2 population ledger (gate G3).
    ledger = pd.read_csv(root / LEDGER, dtype=str, keep_default_na=False)
    raw_ids = pd.read_csv(root / V2_RAW, usecols=["ID"], dtype=str)["ID"]
    included = set(v2.x.index)
    excluded = set(ledger["rna_id"])
    require(len(ledger) == 1208 and ledger["rna_id"].is_unique, "ledger must list 1,208 unique IDs")
    require(not included & excluded, "an excluded ID is present in V2 X")
    require(included | excluded == set(raw_ids) and len(raw_ids) == 5338, "ledger + X != raw V2")
    require(set(ledger["reason"]) == {"missing_sequence"}, "unexpected ledger reasons")
    require(set(ledger["population"]) == {"v2_sequence_available_4130"}, "ledger population")
    require((ledger["flags"] == "species_mismatch_suspected").sum() == 19, "rat ID flags")

    print(f"inputs verified: {len(loaded)} manifest rows, ledger 1,208, overlap {overlap}")
    frozen = verify_frozen_sources(root / FROZEN, root)
    print(f"frozen sources: {len(frozen)} files match")
    print("manifest verification passed")
    return {"loaded": loaded, "feature_columns": feature_columns, "overlap": overlap}


# --------------------------------------------------------------------------- cells


def _expected_effective_params() -> dict[str, dict]:
    professor = (ROOT / "lncRNA_CIBCB2025-main").as_posix()
    if professor not in sys.path:
        sys.path.insert(0, professor)
    from scripts.felipe_paper_models import build_paper_model, effective_params

    return {key: effective_params(key, build_paper_model(key)) for key in MODELS}


def discover_cells(results_root: Path) -> list[tuple[Path, str, dict]]:
    cells = []
    for contract_path in sorted(Path(results_root).rglob(f"*{CONTRACT_SUFFIX}")):
        if "_rejected" in contract_path.parts:
            continue
        stem = contract_path.name[: -len(CONTRACT_SUFFIX)]
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            contract = {}
        cells.append((contract_path.parent, stem, contract))
    return cells


def _cell_key(contract: dict) -> tuple:
    if contract.get("kind") == "within":
        return ("within", contract.get("version"), contract.get("label_space"),
                contract.get("model"), contract.get("fold"))
    return ("transfer", contract.get("source"), contract.get("target"), contract.get("model"), None)


def check_cells(results_root: Path, root: Path = ROOT, *, quiet: bool = False) -> dict[tuple, str]:
    """Return {cell key: 'accepted' | error message}; raise if any cell fails."""
    manifests = check_manifests(root) if not quiet else _quiet(check_manifests, root)
    loaded = manifests["loaded"]
    frozen = verify_frozen_sources(root / FROZEN, root)
    params = _expected_effective_params()
    within_manifest_sha = sha256(root / WITHIN_MANIFEST)
    transfer_manifest_sha = sha256(root / TRANSFER_MANIFEST)
    within = pd.read_csv(root / WITHIN_MANIFEST).set_index(["version", "label_space"])
    split_ids = {
        key: _split_test_ids(root / within.loc[key, "split_path"], exp.x.index.tolist())
        for key, exp in loaded.items()
    }
    status: dict[tuple, str] = {}
    failures: list[str] = []
    for directory, stem, contract in discover_cells(results_root):
        key = _cell_key(contract)
        try:
            frame, contract = read_prediction_cell(directory, stem)
            model = contract["model"]
            require(model in MODELS, f"unknown model {model}")
            require(contract["model_profile"] == "paper", "model_profile must be paper")
            require(contract["effective_params"] == params[model], "effective_params differ from paper profile")
            require(contract["code_hashes"] == frozen, "code_hashes differ from the frozen table")
            require(contract["feature_key"] == "paper_sequence_542", "feature_key")
            require(contract["seed"] == 0, "seed")
            if contract["kind"] == "within":
                mkey = (contract["version"], contract["label_space"])
                require(mkey in loaded, f"unexpected within label space {mkey}")
                row = within.loc[mkey]
                require(
                    contract["input_hashes"] == {
                        "x": row["x_sha256"], "y": row["y_sha256"],
                        "split": row["split_sha256"], "manifest": within_manifest_sha,
                    },
                    "input_hashes differ from the current within manifest",
                )
                require(contract["truth_y_path"] == row["y_path"], "truth_y_path")
                require(contract["truth_y_sha256"] == row["y_sha256"], "truth_y_sha256")
                require(contract["n_folds_total"] == 10, "n_folds_total")
                fold = int(contract["fold"])
                require(frame.index.tolist() == split_ids[mkey][fold], f"fold {fold} IDs differ from the split")
                require(frame.columns.tolist() == loaded[mkey].y.columns.tolist(), "label order")
            elif contract["kind"] == "transfer":
                source, target = contract["source"], contract["target"]
                require({source, target} == {"v1", "v2"}, "transfer direction")
                require(contract["label_space"] == "shared73", "transfer label space")
                src, tgt = loaded[(source, "shared73")], loaded[(target, "shared73")]
                removed = sorted(set(src.x.index) & set(tgt.x.index))
                require(contract["removed_exact_target_ids"] == removed == EXPECTED_OVERLAP, "overlap removal")
                expected_ids = [i for i in tgt.x.index if i not in set(removed)]
                require(frame.index.tolist() == expected_ids, "transfer target IDs")
                require(frame.columns.tolist() == tgt.y.columns.tolist(), "label order")
                tm = pd.read_csv(root / TRANSFER_MANIFEST).set_index("version")
                require(
                    contract["input_hashes"] == {
                        "source_x": tm.loc[source, "x_sha256"], "source_y": tm.loc[source, "y_sha256"],
                        "target_x": tm.loc[target, "x_sha256"], "target_y": tm.loc[target, "y_sha256"],
                        "manifest": transfer_manifest_sha, "label_map": tm.loc[target, "label_map_sha256"],
                    },
                    "input_hashes differ from the current transfer manifest",
                )
                require(contract["truth_y_sha256"] == tm.loc[target, "y_sha256"], "truth_y_sha256")
                partition = contract["evaluation_partition"]
                folds = partition["folds"]
                flat = [i for fold_ids in folds for i in fold_ids]
                require(sorted(flat) == sorted(expected_ids) and len(flat) == len(set(flat)), "partition coverage")
                digest = sha256_lines([f"{n}:" + ",".join(ids) for n, ids in enumerate(folds, start=1)])
                require(partition["sha256"] == digest, "evaluation_partition sha256")
            else:
                raise VerificationError(f"unknown kind {contract.get('kind')}")
            status[key] = "accepted"
        except (PredictionCellError, VerificationError, KeyError, TypeError, ValueError) as error:
            status[key] = f"rejected: {error}"
            failures.append(f"{directory}/{stem}: {error}")
    if failures:
        raise VerificationError("prediction cell verification failed:\n  " + "\n  ".join(failures))
    print(f"cell verification passed ({len(status)} cells)")
    return status


def _quiet(function, *args):
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args)


def expected_cells(scope: str) -> list[tuple]:
    keys: list[tuple] = []
    for kind, first, second in SCOPES[scope]:
        for model in MODELS:
            if kind == "within":
                keys.extend(("within", first, second, model, fold) for fold in range(1, 11))
            else:
                keys.append(("transfer", first, second, model, None))
    return keys


def check_matrix(results_root: Path, scope: str, root: Path = ROOT) -> None:
    status = check_cells(results_root, root, quiet=True)
    missing = [key for key in expected_cells(scope) if status.get(key) != "accepted"]
    if missing:
        preview = "\n  ".join(map(str, missing[:20]))
        raise VerificationError(
            f"{len(missing)} of {len(expected_cells(scope))} expected cells for scope "
            f"{scope!r} are missing:\n  {preview}"
        )
    print(f"result matrix verification passed (scope={scope}, {len(expected_cells(scope))} cells)")


# --------------------------------------------------------------------------- report

REPORT_REQUIRED = (
    "v2_sequence_available_4130",
    "542",
    "44",
    "436",
    "73",
    "paper_comparison",
    "primary_clean",
    "paper_youden",
    "NR_024015",
)
REPORT_FORBIDDEN = (
    "did not build the same paper feature construction for V2",
    "missing Pse-in-One generation configuration",
    "V2 must be higher",
)


def check_report(report_path: Path) -> None:
    require(report_path is not None and Path(report_path).is_file(), f"report not found: {report_path}")
    text = Path(report_path).read_text(encoding="utf-8")
    missing = [phrase for phrase in REPORT_REQUIRED if phrase not in text]
    stale = [phrase for phrase in REPORT_FORBIDDEN if phrase in text]
    require(not missing, f"report is missing required facts: {missing}")
    require(not stale, f"report contains stale or forbidden claims: {stale}")
    print("report verification passed")


# --------------------------------------------------------------------------- reference


def check_reference() -> None:
    """Released-artifact reference check (D9); not part of four-direction acceptance."""
    protocol = (AUDIT / "protocol.md").read_text(encoding="utf-8")
    for phrase in (
        "355 lncRNAs and 44 disease outputs",
        "827 lncRNA features",
        "released 10 inductive folds",
        "150 trees",
        "Youden threshold per disease",
        "285 binary columns",
    ):
        require(phrase in protocol, f"protocol is missing: {phrase}")
    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file(), f"missing source: {path}")
        require(sha256(path) == expected, f"source hash changed: {path}")

    x_rows = read_csv_rows(X_PATH)
    y_rows = read_csv_rows(Y_PATH)
    raw_rows = read_csv_rows(RAW_PATH)
    require(len(x_rows) == 356 and len(x_rows[0]) == 828, "X is not 355 x 827 plus index")
    require(len(y_rows) == 356 and len(y_rows[0]) == 45, "Y is not 355 x 44 plus index")
    require(len(raw_rows) == 356 and len(raw_rows[0]) == 288, "raw matrix shape changed")
    block_sum = raw_sum = exact_first_231 = 0
    for index, (x_row, raw_row) in enumerate(zip(x_rows[1:], raw_rows[1:])):
        block = [int(float(value)) for value in x_row[1 + 462 : 1 + 747]]
        associations = [int(float(value)) for value in raw_row[3:]]
        require(len(block) == len(associations) == 285, "association block width")
        block_sum += sum(block)
        raw_sum += sum(associations)
        if index < 231 and block == associations:
            exact_first_231 += 1
    require(block_sum == raw_sum == 1132, "association block sums")
    require(exact_first_231 == 231, "first 231 rows must carry the association block")
    print(
        "released-artifact reference check passed "
        "(reproduction evidence only; not part of four-direction acceptance)"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check", required=True,
        choices=("manifests", "cells", "matrix", "report", "reference", "protocol"),
    )
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--scope", choices=tuple(SCOPES), default="all")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.check in {"cells", "matrix"} and args.results_root is None:
        parser.error(f"--check {args.check} requires --results-root")
    results_root = args.results_root
    if results_root is not None and not results_root.is_absolute():
        results_root = ROOT / results_root
    try:
        if args.check == "manifests":
            check_manifests()
        elif args.check == "cells":
            check_cells(results_root)
        elif args.check == "matrix":
            check_matrix(results_root, args.scope)
        elif args.check == "report":
            check_report(args.report)
        else:
            check_reference()
    except (VerificationError, FrozenSourceError, FileNotFoundError) as error:
        print(f"VERIFICATION FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
