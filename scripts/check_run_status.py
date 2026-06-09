from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PIPELINE_EXPECTED = {
    "rna": {
        "dir_candidates": ["rna", "Rna"],
        "shape_csv": "rna_feature_shapes.csv",
        "required_files": {
            "v1": [
                "sequences_for_oop_kmer_matrix_k4.csv",
                "sequences_for_oop_rc_kmer_matrix_k4.csv",
                "sequences_for_oop_psednc_matrix.csv",
            ],
            "v2": [
                "website_full_matrix_kmer_matrix_k4.csv",
                "website_full_matrix_rc_kmer_matrix_k4.csv",
                "website_full_matrix_psednc_matrix.csv",
            ],
        },
    },
    "disease": {
        "dir_candidates": ["disease", "Dis"],
        "shape_csv": "disease_feature_shapes.csv",
        "required_files": {
            "v1": [
                "lfs_from_Y.csv",
                "disease_similarity_bma.csv",
                "term_similarity_wang.csv",
            ],
            "v2": [
                "lfs_from_Y.csv",
                "disease_similarity_bma.csv",
                "term_similarity_wang.csv",
            ],
        },
    },
    "cross": {
        "dir_candidates": ["cross", "Cross"],
        "shape_csv": "cross_feature_shapes.csv",
        "required_files": {
            "v1": [
                "sequences_for_oop_gip_lncRNA.csv",
                "sequences_for_oop_gip_disease.csv",
                "sequences_for_oop_svd_lncRNA.csv",
                "sequences_for_oop_svd_disease.csv",
            ],
            "v2": [
                "website_full_matrix_gip_lncRNA.csv",
                "website_full_matrix_gip_disease.csv",
                "website_full_matrix_svd_lncRNA.csv",
                "website_full_matrix_svd_disease.csv",
            ],
        },
    },
    "nn": {
        "dir_candidates": ["nn", "NN"],
        "shape_csv": "nn_feature_shapes.csv",
        "required_files": {
            "v1": [
                "sequences_for_oop_mp_sequence.csv",
                "sequences_for_oop_mp_sequence_chunked.csv",
                "sequences_for_oop_mp_tokens.csv",
            ],
            "v2": [
                "website_full_matrix_mp_sequence.csv",
                "website_full_matrix_mp_sequence_chunked.csv",
                "website_full_matrix_mp_tokens.csv",
            ],
        },
    },
}


def find_first_existing(base: Path, names: list[str]) -> Path | None:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def resolve_final_output_root() -> Path | None:
    return find_first_existing(PROJECT_ROOT, ["final_output", "Final_output"])


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def pipeline_status() -> None:
    out_root = resolve_final_output_root()
    if out_root is None:
        print("final_output directory: missing")
        return

    print_section("Pipeline Outputs")
    print(f"output_root: {out_root}")

    version_candidates = {"v1": ["v1", "V1"], "v2": ["v2", "V2"]}

    for version, candidates in version_candidates.items():
        version_dir = find_first_existing(out_root, candidates)
        print(f"\n[{version}]")
        if version_dir is None:
            print(" version_dir: missing")
            continue
        print(f" version_dir: {version_dir}")

        for domain, cfg in PIPELINE_EXPECTED.items():
            domain_dir = find_first_existing(version_dir, cfg["dir_candidates"])
            if domain_dir is None:
                print(f"  {domain:8} dir missing")
                continue

            shape_csv = domain_dir / cfg["shape_csv"]
            shape_rows = "missing"
            if shape_csv.exists():
                try:
                    shape_rows = str(len(pd.read_csv(shape_csv)))
                except Exception as exc:
                    shape_rows = f"unreadable ({exc})"

            required = cfg["required_files"][version]
            missing = [name for name in required if not (domain_dir / name).exists()]
            status = "OK" if not missing else f"missing {len(missing)}"
            print(f"  {domain:8} {status}; shape_rows={shape_rows}")
            for name in missing:
                print(f"           - missing: {domain_dir / name}")


def sanity_report_status() -> None:
    report_dir = PROJECT_ROOT / "Data" / "output_data" / "comparison_reports"
    print_section("Sanity Reports")
    if not report_dir.exists():
        print("comparison_reports: missing")
        return

    sanity_csv = report_dir / "feature_sanity_check.csv"
    if not sanity_csv.exists():
        print("feature_sanity_check.csv: missing")
        return

    df = pd.read_csv(sanity_csv)
    if df.empty:
        print("feature_sanity_check.csv: empty")
        return

    flagged = df[
        (df["row_status"].fillna("ok") != "ok")
        | (df["col_status"].fillna("ok") != "ok")
        | (df["sample_status"].fillna("ok") != "ok")
    ]

    print(f"feature_sanity_check rows: {len(df)}")
    print(f"flagged rows: {len(flagged)}")
    if not flagged.empty:
        grouped = (
            flagged.groupby(["version", "domain"])
            .size()
            .reset_index(name="count")
            .sort_values(["version", "domain"])
        )
        print("flagged by version/domain:")
        for _, row in grouped.iterrows():
            print(f" - {row['version']} {row['domain']}: {row['count']}")


def inductive_status() -> None:
    project_dir = PROJECT_ROOT / "lncRNA_CIBCB2025-main"
    print_section("Inductive Experiments")
    if not project_dir.exists():
        print("lncRNA_CIBCB2025-main: missing")
        return

    for model_name in ["RFLDA", "IPCARF"]:
        model_dir = project_dir / model_name
        if not model_dir.exists():
            print(f"{model_name}: directory missing")
            continue

        pred_files = sorted(model_dir.glob("*_predictions_test_fold*.csv"))
        if not pred_files:
            print(f"{model_name}: no prediction files found")
            continue

        pattern = re.compile(r"(.+)_predictions_test_fold(\d+)\.csv$")
        groups: dict[str, set[int]] = {}
        for path in pred_files:
            match = pattern.match(path.name)
            if not match:
                continue
            prefix, fold = match.groups()
            groups.setdefault(prefix, set()).add(int(fold))

        print(f"{model_name}:")
        for prefix in sorted(groups):
            folds = groups[prefix]
            status = "OK" if len(folds) == 10 and min(folds) == 1 and max(folds) == 10 else "incomplete"
            print(f" - {prefix}: {len(folds)}/10 folds ({status})")

    parse_dir = project_dir / "parse_results"
    perf_files = sorted(parse_dir.glob("*performance.csv"))
    if perf_files:
        print("performance CSVs:")
        for path in perf_files:
            print(f" - {path.name}")
    else:
        print("performance CSVs: none found")

    fr_summary = parse_dir / "feature_representations" / "inductive_feature_representation_summary.csv"
    if fr_summary.exists():
        try:
            df = pd.read_csv(fr_summary)
            print(f"feature representation summary rows: {len(df)}")
        except Exception as exc:
            print(f"feature representation summary unreadable: {exc}")
    else:
        print("feature representation summary: missing")


def main() -> None:
    pipeline_status()
    sanity_report_status()
    inductive_status()


if __name__ == "__main__":
    main()
