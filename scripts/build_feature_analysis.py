"""Feature-analysis outputs (Supervisor item 9).

Two tables under results/audit/:
  feature_catalogue.csv    - every discovered feature, domain, dims, compatibility,
                             and leakage status (from the feature inventory + shapes).
  feature_statistics.csv   - basic stats (n_features, %zero, mean variance) for the
                             prepared, leakage-free modelling inputs in
                             inductive_inputs/feature_representations/.

Cheap: reads metadata and the already-prepared X matrices; no training.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEAKY = {"svd_lncRNA", "svd_disease", "gip_lncRNA", "gip_disease", "lfs_from_Y"}


def build_catalogue() -> pd.DataFrame:
    rows = []
    for shapes in PROJECT_ROOT.glob("Final_output/*/*/*_feature_shapes.csv"):
        df = pd.read_csv(shapes)
        version = shapes.parts[-3]
        for r in df.to_dict("records"):
            name = str(r.get("feature_name", ""))
            rows.append({
                "version": version,
                "domain": r.get("feature_group", ""),
                "feature_name": name,
                "rows": r.get("rows", ""),
                "cols": r.get("cols", ""),
                "leaky": name in LEAKY,
                "note": "association-derived (built from Y) — excluded"
                        if name in LEAKY else "sequence/ontology-derived — usable",
            })
    return pd.DataFrame(rows).sort_values(["version", "domain", "feature_name"], kind="stable")


def build_statistics() -> pd.DataFrame:
    rows = []
    mani = PROJECT_ROOT / "inductive_inputs/feature_representations/feature_representation_manifest.csv"
    if not mani.exists():
        return pd.DataFrame()
    for r in pd.read_csv(mani).to_dict("records"):
        x = pd.read_csv(Path(r["x_path"]))
        feat = x.drop(columns=["sample_id"], errors="ignore").select_dtypes("number")
        rows.append({
            "version": r["version"],
            "feature_set": r["feature_set"],
            "n_samples": len(x),
            "n_features": feat.shape[1],
            "pct_zero": round(float((feat == 0).to_numpy().mean()) * 100, 2),
            "mean_variance": round(float(feat.var(axis=0).mean()), 6),
            "n_zero_variance_cols": int((feat.var(axis=0) == 0).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    outdir = PROJECT_ROOT / "results/audit"
    outdir.mkdir(parents=True, exist_ok=True)

    cat = build_catalogue()
    cat_path = outdir / "feature_catalogue.csv"
    cat.to_csv(cat_path, index=False)

    stats = build_statistics()
    stats_path = outdir / "feature_statistics.csv"
    stats.to_csv(stats_path, index=False)

    print(f"Saved {cat_path} ({len(cat)} rows; {int(cat['leaky'].sum())} leaky flagged)")
    print(f"Saved {stats_path} ({len(stats)} prepared modelling inputs)")
    if not stats.empty:
        print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
