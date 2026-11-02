"""Generate RNA features from one or more sequence CSVs.

Notes:
- ``--feature_set paper_sequence_542`` is the verified Felipe comparison mode. It
  emits only the reconstructible sequence portion of the released paper input:
  462 DCC values plus normalized k=2 and k=3 frequencies.
- ``--feature_set felipe_released_827`` validates the V1 RNA population and
  copies Felipe's released 827-column artifact byte for byte. It does not claim
  that the unresolved 285-column block was regenerated from sequence.
- ``--feature_set all`` retains the exploratory run-every-method workflow.
- Methods 3–6 need dinucleotide properties. The bundled 22-property
  Pse-in-One table is the default; another table may be supplied explicitly.
- Accepts multiple inputs; if a file has columns ID/seqs, it is normalized to id/seq.
- Feature-level normalization can be disabled only in exploratory ``all`` mode.

Verified Felipe-compatible example:
  python run_all_rna_features.py \
    --seqs_csv Data/raw/sequences.csv \
    --outdir results/paper-sequence-features --version_name V1 \
    --feature_set paper_sequence_542
"""

from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd

from mainfolder.utils.output_summary import add_shape_record, shape_from_result, write_shape_summary
from scripts.reconstruct_paper_sequence_features import (
    DEFAULT_PROPERTIES,
    reconstruct,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASED_FEATURES = (
    PROJECT_ROOT
    / "lncRNA_CIBCB2025-main/create_dataset/datasets/features/features_preprocessed.csv"
)
EXPECTED_RELEASED_FEATURES_SHA256 = (
    "b264dd75545ca424f1db96622555066791c52ba6105001770ae869e211428887"
)


def load_props_csv(path: str):
    """Load dinucleotide properties from CSV with columns: dinuc,p1,p2,..."""
    from mainfolder.utils.utils import normalize_rna_dinuc_props

    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if "dinuc" not in cols:
        raise ValueError("props_csv must include a 'dinuc' column")
    df["dinuc"] = df["dinuc"].str.upper().str.replace("T", "U")
    props = {}
    for _, row in df.iterrows():
        dinuc = row["dinuc"]
        vals = [float(x) for x in row.drop(labels="dinuc").tolist()]
        props[dinuc] = vals
    if not props:
        raise ValueError("props_csv contained no rows")
    return normalize_rna_dinuc_props(props)


def normalize_id_seq(csv_path: Path):
    """
    Load a CSV that may be id/seq or ID/seqs and return ids, seqs lists.
    Extra columns (disease labels, etc.) are ignored.
    """
    df = pd.read_csv(csv_path, dtype=str)
    cols_lower = [c.lower() for c in df.columns]
    df.columns = cols_lower
    if "seqs" in df.columns and "seq" not in df.columns:
        df = df.rename(columns={"seqs": "seq"})
    if "id" not in df.columns:
        raise ValueError(f"{csv_path} must have an id/ID column")
    if "seq" not in df.columns:
        raise ValueError(f"{csv_path} must have a seq/seqs column")
    df = df[["id", "seq"]].dropna(subset=["id", "seq"])
    return df["id"].tolist(), df["seq"].tolist()


def detect_id_sequence_columns(csv_path: Path) -> tuple[str, str]:
    """Return the original ID and sequence column names without changing IDs."""
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    by_lower = {str(column).lower(): str(column) for column in columns}
    id_column = by_lower.get("id")
    sequence_column = by_lower.get("seq") or by_lower.get("seqs")
    if id_column is None:
        raise ValueError(f"{csv_path} must have an id/ID column")
    if sequence_column is None:
        raise ValueError(f"{csv_path} must have a seq/seqs column")
    return id_column, sequence_column


def validate_paper_mode_args(args: argparse.Namespace) -> None:
    """Reject flags that would silently change a fixed Felipe contract."""
    if args.no_normalize:
        raise ValueError(
            f"--no_normalize is incompatible with --feature_set {args.feature_set}; "
            "fixed Felipe feature contracts cannot be changed by this flag."
        )


def run_paper_sequence_542(
    seqs_csvs: list[str | Path],
    *,
    outdir: str | Path,
    version_name: str,
    properties_csv: str | Path = DEFAULT_PROPERTIES,
) -> list[Path]:
    """Write the verified 542-column sequence feature matrix and audit metadata."""
    base_out = Path(outdir) / version_name / "rna"
    base_out.mkdir(parents=True, exist_ok=True)
    properties_path = Path(properties_csv)
    shape_records: list[dict[str, object]] = []
    outputs: list[Path] = []

    for seqs_csv in seqs_csvs:
        seqs_path = Path(seqs_csv)
        id_column, sequence_column = detect_id_sequence_columns(seqs_path)
        matrix, metadata = reconstruct(
            seqs_path,
            properties_path,
            id_column=id_column,
            sequence_column=sequence_column,
        )
        out_path = base_out / f"{seqs_path.stem}_paper_sequence_542.csv"
        matrix.to_csv(out_path)
        metadata.update(
            {
                "pipeline_feature_set": "paper_sequence_542",
                "output_csv": out_path.resolve().as_posix(),
                "output_sha256": sha256_file(out_path),
            }
        )
        metadata_path = out_path.with_suffix(out_path.suffix + ".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        add_shape_record(
            shape_records,
            feature_group="rna",
            feature_name="paper_sequence_542",
            source_name=seqs_path.stem,
            output_path=out_path,
            rows=matrix.shape[0],
            cols=matrix.shape[1],
        )
        outputs.append(out_path)
        print(
            f"[saved] {out_path}: {matrix.shape[0]} rows x {matrix.shape[1]} features; "
            f"cleaned={metadata['cleaned_sequence_count']}; "
            f"missing={metadata['missing_sequence_count']}"
        )

    summary_path = write_shape_summary(shape_records, base_out / "rna_feature_shapes.csv")
    print(f"[saved] {summary_path}")
    return outputs


def run_felipe_released_827(
    seqs_csvs: list[str | Path],
    *,
    outdir: str | Path,
    version_name: str,
    released_features_csv: str | Path = DEFAULT_RELEASED_FEATURES,
) -> list[Path]:
    """Validate the V1 population and copy Felipe's released 827 features."""
    base_out = Path(outdir) / version_name / "rna"
    base_out.mkdir(parents=True, exist_ok=True)
    released_path = Path(released_features_csv)
    released_hash = sha256_file(released_path)
    if released_hash != EXPECTED_RELEASED_FEATURES_SHA256:
        raise ValueError(
            "released 827-feature artifact hash differs from the verified Felipe file: "
            f"{released_path}"
        )

    released = pd.read_csv(released_path, index_col=0)
    expected_columns = [str(index) for index in range(827)]
    if released.shape != (355, 827):
        raise ValueError(f"released feature matrix must be 355x827, got {released.shape}")
    if released.index.astype(str).duplicated().any():
        raise ValueError("released feature matrix contains duplicate RNA IDs")
    if released.columns.astype(str).tolist() != expected_columns:
        raise ValueError("released feature columns must be the ordered labels 0..826")
    values = released.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("released feature matrix contains non-finite values")

    shape_records: list[dict[str, object]] = []
    outputs: list[Path] = []
    released_ids = released.index.astype(str).tolist()
    for seqs_csv in seqs_csvs:
        seqs_path = Path(seqs_csv)
        id_column, _ = detect_id_sequence_columns(seqs_path)
        source_ids = pd.read_csv(
            seqs_path,
            usecols=[id_column],
            dtype=str,
            keep_default_na=False,
        )[id_column].tolist()
        if len(source_ids) != 355 or len(set(source_ids)) != 355:
            raise ValueError("felipe_released_827 requires 355 unique V1 RNA IDs")
        if source_ids != released_ids:
            raise ValueError(
                "sequence RNA IDs/order do not exactly match Felipe's released V1 feature artifact"
            )

        out_path = base_out / f"{seqs_path.stem}_felipe_released_827.csv"
        shutil.copyfile(released_path, out_path)
        metadata = {
            "pipeline_feature_set": "felipe_released_827",
            "generation_method": "validated_byte_copy_of_felipe_released_artifact",
            "input_sequences_csv": seqs_path.resolve().as_posix(),
            "input_sequences_sha256": sha256_file(seqs_path),
            "released_features_csv": released_path.resolve().as_posix(),
            "released_features_sha256": released_hash,
            "output_csv": out_path.resolve().as_posix(),
            "output_sha256": sha256_file(out_path),
            "output_rows": 355,
            "output_features": 827,
            "rna_id_order_match": True,
            "released_artifact_columns": 827,
            "sequence_generated_columns": 0,
            "sequence_reconstructible_columns": 542,
            "unresolved_binary_block_columns": 285,
            "provenance_note": (
                "All columns are imported unchanged from Felipe's released artifact. "
                "The 285-column binary block is not regenerated from sequence."
            ),
        }
        metadata_path = out_path.with_suffix(out_path.suffix + ".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        add_shape_record(
            shape_records,
            feature_group="rna",
            feature_name="felipe_released_827",
            source_name=seqs_path.stem,
            output_path=out_path,
            rows=355,
            cols=827,
        )
        outputs.append(out_path)
        print(f"[saved] {out_path}: 355 rows x 827 released features")

    summary_path = write_shape_summary(shape_records, base_out / "rna_feature_shapes.csv")
    print(f"[saved] {summary_path}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run all RNA feature methods over one or more CSVs.")
    p.add_argument(
        "--seqs_csv",
        nargs="+",
        required=True,
        help="One or more CSVs with id/seq or ID/seqs (e.g. sequences_for_oop.csv, website_full_matrix.csv)",
    )
    p.add_argument("--outdir", required=True, help="Base folder to save outputs")
    p.add_argument("--version_name", required=True, help="Data version label (e.g., v1, v2)")
    p.add_argument(
        "--feature_set",
        choices=("all", "paper_sequence_542", "felipe_released_827"),
        default="all",
        help=(
            "Feature contract to generate. paper_sequence_542 is the verified Felipe "
            "sequence-only contract; felipe_released_827 validates and copies the "
            "released V1 artifact; all is the exploratory legacy workflow."
        ),
    )
    p.add_argument("--k", type=int, default=3, help="k for k-mer based methods (1,2,8,9)")
    p.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        help="Optional list of k values for methods 1 and 2 (e.g. --k_values 1 2 3 4). "
             "If omitted, the single --k value is used.",
    )
    p.add_argument("--lam", type=int, default=2, help="lam for PseDNC (method 3)")
    p.add_argument("--weight", type=float, default=0.5, help="w for PseDNC (method 3)")
    p.add_argument("--L", type=int, default=3, help="lag for DAC/DCC/DACC (methods 4,5,6)")
    p.add_argument("--k_gap", type=int, default=1, help="gap for monoMonoKGap/monoDiKGap (methods 11,12)")
    p.add_argument(
        "--no_normalize",
        action="store_true",
        help=(
            "Disable feature-value normalization in exploratory all mode. "
            "Rejected by paper_sequence_542."
        ),
    )
    p.add_argument(
        "--props_csv",
        default=str(DEFAULT_PROPERTIES),
        help=(
            "CSV with dinucleotide properties. Defaults to the bundled 16x22 "
            "Pse-in-One RNA table. paper_sequence_542 strictly requires that shape."
        ),
    )
    p.add_argument(
        "--released_features_csv",
        default=str(DEFAULT_RELEASED_FEATURES),
        help=(
            "Felipe's released 355x827 V1 feature artifact. Used only by "
            "felipe_released_827 and verified by SHA-256."
        ),
    )
    return p


def main(argv: list[str] | None = None):
    p = build_parser()
    args = p.parse_args(argv)
    if args.feature_set in {"paper_sequence_542", "felipe_released_827"}:
        try:
            validate_paper_mode_args(args)
        except ValueError as error:
            p.error(str(error))
    if args.feature_set == "paper_sequence_542":
        run_paper_sequence_542(
            args.seqs_csv,
            outdir=args.outdir,
            version_name=args.version_name,
            properties_csv=args.props_csv,
        )
        return
    if args.feature_set == "felipe_released_827":
        run_felipe_released_827(
            args.seqs_csv,
            outdir=args.outdir,
            version_name=args.version_name,
            released_features_csv=args.released_features_csv,
        )
        return

    print(
        "[warn] feature_set=all is an exploratory descriptor sweep and does not "
        "reproduce Felipe's released 827-column input. Use "
        "--feature_set paper_sequence_542 for the verified sequence-only contract."
    )
    from mainfolder.core.feature_extractor import FeatureExtractor
    from mainfolder.features.rna_features import RnaFeatures
    from mainfolder.utils.loader import preprocess_sequences
    from mainfolder.utils.utils import ALPHABET

    normalize_features = not args.no_normalize
    k_values = args.k_values or [args.k]
    bad_k = [k for k in k_values if k < 1]
    if bad_k:
        raise ValueError(f"All k values must be >= 1, got: {bad_k}")

    # place results under <outdir>/<version_name>/rna/
    base_out = Path(args.outdir) / args.version_name / "rna"
    base_out.mkdir(parents=True, exist_ok=True)

    props = load_props_csv(args.props_csv)
    shape_records = []

    # method_id -> kwargs (props always provided so nothing is skipped)
    method_params = {
        1: {"k": args.k},
        2: {"k": args.k},
        3: {"lam": args.lam, "w": args.weight, "props": props},
        4: {"L": args.L, "props": props},
        5: {"L": args.L, "props": props},
        6: {"L": args.L, "props": props},
        7: {},
        8: {},   # di_composition uses k=2 internally
        9: {},   # tri_composition uses k=3 internally
        10: {},
        11: {"k_gap": args.k_gap},
        12: {"k_gap": args.k_gap},
    }

    for seqs_csv in args.seqs_csv:
        seqs_path = Path(seqs_csv)
        ids, seqs = normalize_id_seq(seqs_path)
        # Drop sequences with invalid chars (including N) to avoid downstream errors
        ids2, seqs2 = preprocess_sequences(ids, seqs, valid_alphabet=set(ALPHABET), strict=False)
        if not seqs2:
            raise ValueError(
                f"{seqs_path} produced 0 valid sequences after preprocessing. "
                "Check seq/seqs content and allowed RNA alphabet."
            )

        stem = seqs_path.stem
        for mid, name in sorted(RnaFeatures.METHOD_MAP.items()):
            run_specs = []
            if mid in (1, 2):
                for k in k_values:
                    run_specs.append(
                        {
                            "feature_name": f"{name}_k{k}",
                            "kwargs": {"k": k},
                            "display_suffix": f", k={k}",
                        }
                    )
            else:
                run_specs.append(
                    {
                        "feature_name": name,
                        "kwargs": method_params.get(mid, {}).copy(),
                        "display_suffix": "",
                    }
                )

            for spec in run_specs:
                kwargs = spec["kwargs"]
                kwargs.update({"return_format": "dataframe", "sample_ids": ids2})
                # Only pass normalize to methods that accept it
                if mid in (1, 2, 7, 8, 9, 10, 11, 12):
                    kwargs["normalize"] = normalize_features
                print(
                    f"[run] {args.version_name}/{stem} -> method {mid} ({spec['feature_name']}) "
                    f"| normalize={normalize_features if mid in (1, 2, 7, 8, 9, 10, 11, 12) else 'n/a'}"
                    f"{spec['display_suffix']}"
                )
                cols, df = FeatureExtractor.run("rna", mid, seqs2, **kwargs)
                out_path = base_out / f"{stem}_{spec['feature_name']}.csv"
                if isinstance(df, pd.DataFrame):
                    df.to_csv(out_path, index=False)
                    rows, ncols = shape_from_result(df)
                else:
                    pd.DataFrame(df, columns=cols).to_csv(out_path, index=False)
                    rows, ncols = shape_from_result(df, cols)
                add_shape_record(
                    shape_records,
                    feature_group="rna",
                    feature_name=spec["feature_name"],
                    method_id=mid,
                    source_name=stem,
                    output_path=out_path,
                    rows=rows,
                    cols=ncols,
                )
                print(f"[saved] {out_path}")

    summary_path = write_shape_summary(shape_records, base_out / "rna_feature_shapes.csv")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
