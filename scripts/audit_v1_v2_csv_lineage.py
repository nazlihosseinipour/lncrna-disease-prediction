"""Build an evidence-backed V1/V2 CSV and feature-provenance inventory."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


LINEAGE_FILES = [
    # V1: supplied paper artifact and the locally normalized source.
    ("v1", "raw", "paper association + sequence matrix", "lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv"),
    ("v1", "raw mirror", "local mirror of paper association + sequence matrix", "Data/raw/sequences.csv"),
    ("v1", "normalized", "normalized association + sequence matrix", "Data/output_data/sequences_for_oop.csv"),
    ("v1", "raw", "paper-supplied feature artifact", "lncRNA_CIBCB2025-main/create_dataset/raw_data/features.csv"),
    ("v1", "paper preprocessing", "paper model X", "lncRNA_CIBCB2025-main/create_dataset/datasets/features/features_preprocessed.csv"),
    ("v1", "paper preprocessing", "paper model Y", "lncRNA_CIBCB2025-main/create_dataset/datasets/flat_labels/labels_flat_preprocessed.csv"),
    ("v1", "final clean model", "clean RF X", "inductive_inputs/corrected_10fold_2026-08-17/v1/v1_all_safe_concatenated_X.csv"),
    ("v1", "final clean model", "clean RF Y", "inductive_inputs/corrected_10fold_2026-08-17/v1/v1_all_safe_concatenated_Y_gt9.csv"),
    ("v1", "final clean model", "10-fold split indices", "inductive_inputs/corrected_10fold_2026-08-17/v1/v1_all_safe_concatenated_X_splits.csv"),
    # V2: raw long-form download, matrices, historical gene-region sequences,
    # corrected transcript sequences, and the final deduplicated model inputs.
    ("v2", "raw", "website long-form association records", "Data/raw/website_alldata.csv"),
    ("v2", "preprocessed", "wide disease-label matrix", "Data/output_data/website_disease_matrix.csv"),
    ("v2", "preprocessed historical", "wide labels + historical gene-region sequences", "Data/output_data/website_full_matrix.csv"),
    ("v2", "preprocessed historical", "historical gene-region sequence table", "Data/output_data/website_sequences.csv"),
    ("v2", "feature extraction historical", "valid historical gene-region sequences", "Data/output_data/website_sequences_for_oop.csv"),
    ("v2", "transcript correction audit", "gene-to-canonical-transcript mapping", "Data/corrected_inputs/v2_transcript_sequences/gene_to_canonical_transcript.csv"),
    ("v2", "transcript correction audit", "transcript refetch report", "Data/corrected_inputs/v2_transcript_sequences/transcript_refetch_report.csv"),
    ("v2", "corrected preprocessing", "wide labels + canonical transcript cDNA", "Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv"),
    ("v2", "corrected preprocessing", "canonical transcript sequence table", "Data/corrected_inputs/v2_transcript_sequences/website_sequences.csv"),
    ("v2", "corrected feature extraction", "valid canonical transcript sequences", "Data/corrected_inputs/v2_transcript_sequences/website_sequences_for_oop.csv"),
    ("v2", "corrected preprocessing audit", "duplicate transcript sequence groups", "Data/corrected_inputs/v2_transcript_sequences/duplicate_sequence_groups.csv"),
    ("v2", "final clean model", "deduplicated clean RF X", "inductive_inputs/corrected_v2_transcripts_10fold_2026-08-17/v2_all_safe_concatenated_X.csv"),
    ("v2", "final clean model", "deduplicated clean RF Y", "inductive_inputs/corrected_v2_transcripts_10fold_2026-08-17/v2_all_safe_concatenated_Y_ge10.csv"),
    ("v2", "final clean model", "10-fold split indices", "inductive_inputs/corrected_v2_transcripts_10fold_2026-08-17/v2_all_safe_concatenated_X_splits.csv"),
]


FEATURE_SUMMARIES = [
    ("v1", "current", "Final_output/V1/rna/rna_feature_shapes.csv"),
    ("v1", "current", "Final_output/V1/nn/nn_feature_shapes.csv"),
    ("v1", "current", "Final_output/V1/Dis/disease_feature_shapes.csv"),
    ("v1", "current", "Final_output/V1/Cross/cross_feature_shapes.csv"),
    ("v2", "historical_gene_region", "Final_output/V2/Rna/rna_feature_shapes.csv"),
    ("v2", "historical_gene_region", "Final_output/V2/Dis/disease_feature_shapes.csv"),
    ("v2", "historical_gene_region", "Final_output/V2/Cross/cross_feature_shapes.csv"),
    ("v2", "corrected_transcript", "Final_output/corrected_v2_transcript_features/Rna/rna_feature_shapes.csv"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def classify_columns(df: pd.DataFrame, role: str) -> tuple[int, int, int, int]:
    columns = list(df.columns)
    index_like = {
        column for column in columns
        if str(column).lower() in {"id", "sample_id", "lncrna_id"}
        or str(column).lower().startswith("unnamed:")
    }
    sequence_like = {
        column for column in columns if str(column).lower() in {"seq", "seqs", "sequence"}
    }
    administrative = index_like | sequence_like
    remaining = [column for column in columns if column not in administrative]

    label_columns = 0
    feature_columns = 0
    role_lower = role.lower()
    if (
        "model y" in role_lower
        or "clean rf y" in role_lower
        or "label matrix" in role_lower
        or "labels +" in role_lower
        or "association + sequence" in role_lower
    ):
        label_columns = len(remaining)
    elif "model x" in role_lower or "feature" in role_lower or "clean rf x" in role_lower:
        feature_columns = len(remaining)
    return len(index_like), len(sequence_like), label_columns, feature_columns


def positive_count(df: pd.DataFrame, label_columns: int) -> int | None:
    if label_columns <= 0:
        return None
    administrative = [
        column for column in df.columns
        if str(column).lower() in {"id", "sample_id", "lncrna_id", "seq", "seqs", "sequence"}
        or str(column).lower().startswith("unnamed:")
    ]
    values = df.drop(columns=administrative, errors="ignore")
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.notna().all(axis=None) and numeric.isin([0, 1]).all(axis=None):
        return int(numeric.to_numpy().sum())
    return None


def build_lineage() -> pd.DataFrame:
    rows = []
    for version, stage, role, relative in LINEAGE_FILES:
        path = ROOT / relative
        if not path.exists():
            rows.append({"version": version, "stage": stage, "role": role, "path": relative, "exists": False})
            continue
        df = read_csv(path)
        id_cols, seq_cols, label_cols, feature_cols = classify_columns(df, role)
        rows.append(
            {
                "version": version,
                "stage": stage,
                "role": role,
                "path": path.as_posix(),
                "exists": True,
                "rows": len(df),
                "columns_on_disk": len(df.columns),
                "id_or_index_columns": id_cols,
                "sequence_columns": seq_cols,
                "label_columns": label_cols,
                "feature_columns": feature_cols,
                "positive_associations": positive_count(df, label_cols),
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def feature_status(version: str, source: str, domain: str, name: str) -> tuple[str, str]:
    if source == "historical_gene_region":
        return "superseded", "generated from historical gene-region sequences, not transcript cDNA"
    if name in {"lfs_from_Y", "gip_lncRNA", "gip_disease", "svd_lncRNA", "svd_disease"}:
        return "excluded", "association-derived; invalid as inductive X"
    if domain.lower() == "disease":
        return "not_sample_X", "disease-by-disease representation, not one row per lncRNA"
    if name == "mp_tokens":
        return "not_sample_X", "token-level output, not one row per lncRNA"
    if name in {"psednc_matrix", "di_auto_cov_matrix", "di_cross_cov_matrix", "di_acc_matrix"}:
        return "quarantined", "property CSV contains corpus count/frequency, not validated physicochemical properties"
    if name in {"mono_composition_matrix", "di_composition_matrix", "tri_composition_matrix"}:
        return "redundant", "exact duplicate of normalized k-mer k=1, k=2, or k=3"
    return "eligible", "sequence-derived sample-level representation"


def build_feature_inventory() -> pd.DataFrame:
    manifests = {
        "v1": pd.read_csv(ROOT / "inductive_inputs/corrected_10fold_2026-08-17/v1/feature_representation_manifest.csv").iloc[0],
        "v2": pd.read_csv(ROOT / "inductive_inputs/corrected_v2_transcripts_10fold_2026-08-17/feature_representation_manifest.csv").iloc[0],
    }
    selected = {
        version: set(str(row["components"]).split("+")) for version, row in manifests.items()
    }
    rows: list[dict[str, object]] = []
    for version, source, relative in FEATURE_SUMMARIES:
        summary_path = ROOT / relative
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path).fillna("")
        for record in summary.to_dict("records"):
            name = str(record["feature_name"])
            domain = str(record["feature_group"])
            status, reason = feature_status(version, source, domain, name)
            output_path = Path(str(record["output_path"]))
            if not output_path.is_absolute():
                output_path = ROOT / output_path
            # macOS workspace is case-insensitive; normalize to the existing path.
            columns_on_disk = int(record["cols"])
            rows.append(
                {
                    "version": version,
                    "sequence_source": source,
                    "domain": domain,
                    "feature_name": name,
                    "method_id": record.get("method_id", ""),
                    "rows": int(record["rows"]),
                    "columns_on_disk": columns_on_disk,
                    "feature_dimensions": columns_on_disk - 1 if domain.lower() in {"rna", "nn"} else columns_on_disk,
                    "status": status,
                    "selected_in_final_clean_X": source != "historical_gene_region" and name in selected[version],
                    "reason": reason,
                    "path": output_path.as_posix(),
                }
            )

    # The paper artifact is inventoried as observed slices because its generation
    # script is not present in the supplied repository.
    paper_path = ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/features.csv"
    rows.extend(
        [
            {
                "version": "v1",
                "sequence_source": "paper_supplied_unknown_provenance",
                "domain": "paper_artifact",
                "feature_name": "paper_features_complete",
                "rows": 355,
                "columns_on_disk": 828,
                "feature_dimensions": 827,
                "status": "quarantined",
                "selected_in_final_clean_X": False,
                "reason": "contains an observed 285-column association-like binary block",
                "path": paper_path.as_posix(),
            },
            {
                "version": "v1",
                "sequence_source": "paper_supplied_unknown_provenance",
                "domain": "paper_artifact_slice",
                "feature_name": "paper_continuous_slice_columns_0_461_and_747_826",
                "rows": 355,
                "columns_on_disk": np.nan,
                "feature_dimensions": 542,
                "status": "audit_only",
                "selected_in_final_clean_X": False,
                "reason": "non-binary remainder used only in matched ablation; extractor provenance unavailable",
                "path": paper_path.as_posix(),
            },
            {
                "version": "v1",
                "sequence_source": "paper_supplied_unknown_provenance",
                "domain": "paper_artifact_slice",
                "feature_name": "paper_binary_slice_columns_462_746",
                "rows": 355,
                "columns_on_disk": np.nan,
                "feature_dimensions": 285,
                "status": "excluded",
                "selected_in_final_clean_X": False,
                "reason": "binary block has the same width and positive total as the raw 285-label matrix",
                "path": paper_path.as_posix(),
            },
        ]
    )
    return pd.DataFrame(rows)


def build_findings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": 1,
                "finding": "Paper-supplied V1 feature artifact has unresolved target-like content",
                "evidence": "827 features split into 542 non-binary columns plus a contiguous 285-column binary block; block sum=1,132 equals the raw 285-label matrix sum; 231/355 aligned rows are exactly equal and 352/355 rows match some label row.",
                "interpretation": "The public artifact is not a trustworthy pure sequence-feature matrix. This does not by itself prove which file produced the published scores.",
                "action": "Quarantine all 827-column runs; report leakage-free ablations separately; request original feature-generation provenance from the authors.",
            },
            {
                "priority": 2,
                "finding": "Fixed 0.5 is badly calibrated for sparse multilabel RF scores",
                "evidence": "Previous predictions average 14 predicted positives vs 61 true positives per V1 fold, and 24 vs 781 per V2 fold.",
                "interpretation": "This explains the very low F1 but cannot change AUROC or AUPRC.",
                "action": "Learn one global threshold only from each outer training fold's RF OOB predictions; retain fixed 0.5 as a transparent baseline.",
            },
            {
                "priority": 3,
                "finding": "Historical V2 used gene-region sequence instead of canonical transcript cDNA",
                "evidence": "Corrected full matrix changes the sequence field for 2,441/5,338 rows; valid feature-extraction rows change from 4,131 to 4,130.",
                "interpretation": "Old and corrected V2 results are not directly comparable feature reruns.",
                "action": "Use only corrected transcript inputs for final V2 reporting.",
            },
            {
                "priority": 4,
                "finding": "Identical V2 transcript sequences were previously separate samples",
                "evidence": "205 duplicate-sequence groups collapse 281 rows, from 4,130 aligned rows to 3,849 independent sequences.",
                "interpretation": "Without group collapse, identical biological inputs can cross train/test boundaries.",
                "action": "Keep sequence deduplication and OR labels before split generation.",
            },
            {
                "priority": 5,
                "finding": "Four apparent RNA feature families lack a valid property source",
                "evidence": "Data/output_data/dinuc_props.csv contains columns dinuc,count,freq; count reaches 24,577,439 and produces covariance magnitudes near 1e13.",
                "interpretation": "PseDNC/DAC/DCC/DACC are not validated physicochemical descriptors in this workspace, so their absence from corrected V2 is intentional, not a missing-feature bug.",
                "action": "Keep them quarantined until a cited, normalized physicochemical property table is supplied.",
            },
            {
                "priority": 6,
                "finding": "V1 all-feature concatenation shows overfitting risk",
                "evidence": "V1 has 353 samples and 2,023 retained features. On the same first three outer folds, the 583-feature core gives AUROC/AUPRC 0.658/0.143 versus 0.623/0.112 for all 2,023 features.",
                "interpretation": "The correlated 1,440 embedding dimensions reduce ranking quality in this probe; this is feature-selection risk, not leakage.",
                "action": "Run the 583-feature core on all 10 existing outer folds before replacing the final V1 estimate.",
            },
        ]
    )


def paper_block_evidence() -> pd.DataFrame:
    feature_path = ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/features.csv"
    label_path = ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv"
    features = pd.read_csv(feature_path, index_col=0)
    labels = pd.read_csv(label_path, index_col=0).drop(columns=["ID", "seqs"])
    block = features.iloc[:, 462:747].to_numpy(dtype=np.uint8)
    target = labels.to_numpy(dtype=np.uint8)
    exact_aligned = np.all(block == target, axis=1)
    cell_equal_fraction = float((block == target).mean())
    target_rows = {row.tobytes() for row in target}
    any_row_matches = np.array([row.tobytes() in target_rows for row in block])
    return pd.DataFrame(
        [
            {"measure": "paper_feature_dimensions", "value": features.shape[1]},
            {"measure": "non_binary_remainder_dimensions", "value": features.shape[1] - block.shape[1]},
            {"measure": "contiguous_binary_block_start_zero_based", "value": 462},
            {"measure": "contiguous_binary_block_end_zero_based_inclusive", "value": 746},
            {"measure": "contiguous_binary_block_dimensions", "value": block.shape[1]},
            {"measure": "raw_label_dimensions", "value": target.shape[1]},
            {"measure": "binary_block_positive_total", "value": int(block.sum())},
            {"measure": "raw_label_positive_total", "value": int(target.sum())},
            {"measure": "aligned_exact_rows", "value": int(exact_aligned.sum())},
            {"measure": "aligned_exact_row_fraction", "value": float(exact_aligned.mean())},
            {"measure": "aligned_cell_equality_fraction", "value": cell_equal_fraction},
            {"measure": "rows_matching_any_raw_label_row", "value": int(any_row_matches.sum())},
            {"measure": "rows_matching_any_raw_label_row_fraction", "value": float(any_row_matches.mean())},
        ]
    )


def build_probe_comparison() -> pd.DataFrame:
    cases = [
        ("V1 core sequence", 583, "results/audit/rf_probe_v1_core583_oob_2026-08-24/within_version_cv_summary.csv"),
        ("V1 all clean features", 2023, "results/audit/rf_probe_v1_all2023_oob_2026-08-24/within_version_cv_summary.csv"),
    ]
    rows = []
    for representation, dimensions, relative in cases:
        path = ROOT / relative
        if not path.exists():
            continue
        row = pd.read_csv(path).iloc[0]
        rows.append(
            {
                "representation": representation,
                "feature_dimensions": dimensions,
                "outer_folds_completed": int(row["completed_folds"]),
                "auroc": float(row["micro_roc_mean"]),
                "auprc": float(row["micro_auprc_mean"]),
                "f1_fixed_0_5": float(row["fixed_baseline_fscore_mean"]),
                "f1_training_oob_threshold": float(row["fscore_mean"]),
                "precision_training_oob_threshold": float(row["precision_mean"]),
                "recall_training_oob_threshold": float(row["recall_mean"]),
                "result_path": path.as_posix(),
            }
        )
    return pd.DataFrame(rows)


def build_rerun_reproducibility() -> pd.DataFrame:
    cases = [
        (
            "v1",
            "results/corrected_validation_2026-08-17/predictions/v1_all_safe_concatenated/rf",
            "results/rerun_corrected_v1_rf_2026-08-23/predictions/v1_all_safe_concatenated/rf",
        ),
        (
            "v2",
            "results/corrected_v2_transcript_validation_2026-08-17/predictions/v2_all_safe_concatenated/rf",
            "results/rerun_corrected_v2_rf_2026-08-23/predictions/v2_all_safe_concatenated/rf",
        ),
    ]
    rows = []
    for version, old_relative, new_relative in cases:
        old_dir, new_dir = ROOT / old_relative, ROOT / new_relative
        for fold in range(1, 11):
            old_path = old_dir / f"test_fold{fold}.csv"
            new_path = new_dir / f"test_fold{fold}.csv"
            old_hash = sha256(old_path) if old_path.exists() else ""
            new_hash = sha256(new_path) if new_path.exists() else ""
            if not old_hash or not new_hash:
                comparison_status = "pending"
                identical: object = pd.NA
            elif old_hash == new_hash:
                comparison_status = "identical"
                identical = True
            else:
                comparison_status = "mismatch"
                identical = False
            rows.append(
                {
                    "version": version,
                    "fold": fold,
                    "old_prediction_exists": old_path.exists(),
                    "fresh_prediction_exists": new_path.exists(),
                    "comparison_status": comparison_status,
                    "sha256_identical": identical,
                    "old_sha256": old_hash,
                    "fresh_sha256": new_hash,
                }
            )
    return pd.DataFrame(rows)


def write_report(
    outdir: Path,
    lineage: pd.DataFrame,
    features: pd.DataFrame,
    probes: pd.DataFrame,
) -> None:
    v1_manifest = pd.read_csv(ROOT / "inductive_inputs/corrected_10fold_2026-08-17/v1/feature_representation_manifest.csv").iloc[0]
    v2_manifest = pd.read_csv(ROOT / "inductive_inputs/corrected_v2_transcripts_10fold_2026-08-17/feature_representation_manifest.csv").iloc[0]
    probe_text = "The controlled OOB probe is still pending."
    if len(probes) == 2:
        core = probes.loc[probes["feature_dimensions"] == 583].iloc[0]
        full = probes.loc[probes["feature_dimensions"] == 2023].iloc[0]
        probe_text = (
            f"Across the same {int(core.outer_folds_completed)} outer folds, the 583-feature core "
            f"gave AUROC {core.auroc:.3f}, AUPRC {core.auprc:.3f}, and training-OOB-calibrated "
            f"F1 {core.f1_training_oob_threshold:.3f}; the 2,023-feature set gave AUROC "
            f"{full.auroc:.3f}, AUPRC {full.auprc:.3f}, and calibrated F1 "
            f"{full.f1_training_oob_threshold:.3f}."
        )
    text = f"""# V1/V2 CSV and feature-provenance audit

Generated from files present in the workspace. Checksums and dimensions are in `csv_lineage.csv`; every collected feature family is in `feature_inventory.csv`.

## Final clean model matrices

| Version | Samples | Features | Labels | Positive associations |
|---|---:|---:|---:|---:|
| V1 strict | {int(v1_manifest.n_samples):,} | {int(v1_manifest.n_features):,} | {int(v1_manifest.n_labels):,} | 613 |
| V2 transcripts | {int(v2_manifest.n_samples):,} | {int(v2_manifest.n_features):,} | {int(v2_manifest.n_labels):,} | 7,806 |

## Actual problems

1. **The supplied paper V1 feature CSV has unresolved provenance and contains a target-like 285-column binary block.** This is a defect in the public artifact. The evidence is strong enough to quarantine the file, but it is not enough to assert that the published experiment used the same contaminated artifact.
2. **The current F1 problem is threshold calibration.** A fixed 0.5 cutoff produces almost no positive predictions in these sparse multilabel tasks. The implemented correction learns a single global threshold from training-fold RF out-of-bag predictions only.
3. **The historical V2 biological input was wrong for the intended transcript representation.** Canonical transcript cDNA replaces gene-region sequences, and identical sequences are collapsed before splitting.
4. **The missing PseDNC/covariance families should not be restored from the current property file.** It contains corpus counts/frequencies, not a validated physicochemical table. Three other omitted composition families are exact duplicates of k-mer k=1/2/3.
5. **V1's 2,023-feature concatenation shows overfitting risk on 353 samples.** In the paired three-fold probe, the 583-feature core improves AUROC/AUPRC from 0.623/0.112 to 0.658/0.143. A complete 10-fold core run is still required before replacing the final estimate.

## Controlled RF probe

{probe_text}

## Interpretation of the earlier “better result”

The matched 44-label ablation remains valid as an artifact audit: removing the observed binary block lowers the paper artifact to AUROC 0.633 / AUPRC 0.082 / F1 0.009, while the clean V1 features give 0.661 / 0.097 / 0.032. That comparison supports the clean feature work. It does **not** establish that the published paper's run contained leakage, because the repository does not include enough feature-generation provenance to connect the supplied CSV to the reported run.

## Inventory scope

The lineage inventory contains {len(lineage)} input/intermediate/final CSVs that directly feed or document V1/V2 construction. The feature inventory contains {len(features)} collected feature-family records. Generated comparison reports, checkpoints, and unrelated review tables are not counted as model inputs.
"""
    (outdir / "audit_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/audit/v1_v2_csv_lineage_2026-08-24")
    args = parser.parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    lineage = build_lineage()
    features = build_feature_inventory()
    findings = build_findings()
    block = paper_block_evidence()
    probes = build_probe_comparison()
    reproducibility = build_rerun_reproducibility()
    lineage.to_csv(outdir / "csv_lineage.csv", index=False)
    features.to_csv(outdir / "feature_inventory.csv", index=False)
    findings.to_csv(outdir / "audit_findings.csv", index=False)
    block.to_csv(outdir / "paper_binary_block_evidence.csv", index=False)
    probes.to_csv(outdir / "rf_probe_comparison.csv", index=False)
    reproducibility.to_csv(outdir / "rerun_reproducibility.csv", index=False)
    write_report(outdir, lineage, features, probes)
    print(f"Saved audit to {outdir}")


if __name__ == "__main__":
    main()
