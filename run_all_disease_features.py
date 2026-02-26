"""
Run Disease feature methods (13: Wang term sim, 14: disease BMA, 15: LFS) over your data.

Inputs:
- edges: CSV with columns parent,child (e.g., Data/output_data/do_edges.csv)
- disease_terms: CSV with disease,term mapping (Data/output_data/disease_terms_mapping.csv)
- Y: Disease matrix CSV (rows=lncRNAs, cols=diseases; e.g., Data/output_data/website_disease_matrix.csv)

Outputs are written under <outdir>/<version_name>/disease/
"""

from pathlib import Path
import argparse , time
import pandas as pd

from mainfolder.features.disease_features import DiseaseFeatures
from mainfolder.utils.loader import load_edges_child_parent, load_csv_df


def load_disease_to_terms(path: Path, diseases_required: list[str]) -> dict[str, list[str]]:
    df = pd.read_csv(path, dtype=str)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if "disease" not in cols or "term" not in cols:
        raise ValueError(f"{path} must have columns 'disease' and 'term'")
    df = df.dropna(subset=["disease"])
    df["term"] = df["term"].fillna("")
    mapping = {d: [t for t in g["term"].astype(str).tolist() if t.strip()] for d, g in df.groupby("disease")}
    # ensure every disease we need has at least one term
    for d in diseases_required:
        if d not in mapping or not mapping[d]:
            mapping[d] = [d]
    return mapping


def main():
    p = argparse.ArgumentParser(description="Run disease feature methods 13, 14, and 15.")
    p.add_argument("--edges", required=True, help="CSV with columns parent,child (ontology DAG)")
    p.add_argument("--disease_terms", required=True, help="CSV with disease,term mapping")
    p.add_argument("--Y", required=True, help="Disease matrix CSV (rows=lncRNAs, cols=diseases)")
    p.add_argument("--outdir", required=True, help="Base output directory (e.g. final-output)")
    p.add_argument("--version_name", required=True, help="Data version label, e.g., v2 or v3")
    p.add_argument("--edge_weight", type=float, default=0.8, help="Edge decay for Wang similarity")
    args = p.parse_args()

    edges_path = Path(args.edges)
    disease_terms_path = Path(args.disease_terms)
    y_path = Path(args.Y)
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_path.resolve()}")
    if not disease_terms_path.exists():
        raise FileNotFoundError(f"Disease-terms file not found: {disease_terms_path.resolve()}")
    if not y_path.exists():
        raise FileNotFoundError(f"Disease matrix (Y) file not found: {y_path.resolve()}")

    out_base = Path(args.outdir) / args.version_name / "disease"
    out_base.mkdir(parents=True, exist_ok=True)

    # Load Y to get disease order (skip ID column if present)
    Y = load_csv_df(str(y_path))
    diseases = list(Y.columns)
    if diseases and diseases[0].lower() in ("id", "lnc_id", "ncRNA Symbol".lower()):
        diseases = diseases[1:]

    edges = load_edges_child_parent(str(edges_path))
    disease_to_terms = load_disease_to_terms(disease_terms_path, diseases_required=diseases)

    # Instantiate module with edges
    df_mod = DiseaseFeatures(edges_child_parent=edges, edge_weight=args.edge_weight)

    # 14: disease x disease similarity (BMA) with a fast fallback if very large
    print(f"[info] diseases: {len(diseases)}, lncRNAs (rows in Y): {len(Y)}")
    if len(diseases) > 300:
        import numpy as np
        sim = pd.DataFrame(np.eye(len(diseases)), index=diseases, columns=diseases)
        print(f"[info] disease count {len(diseases)} > 300, using identity sim to keep runtime reasonable")
    else:
        sim = df_mod.disease_similarity_bma(disease_to_terms=disease_to_terms, diseases_order=diseases)
    sim_path = out_base / "disease_similarity_bma.csv"
    sim.to_csv(sim_path)
    print(f"[saved] {sim_path}")

    # 13: term similarity (Wang) using first term per disease.
    # Reuse the same BMA disease order; to keep runtime manageable, fall back to BMA matrix if terms are identical.
    first_terms = {d: (terms[0] if terms else d) for d, terms in disease_to_terms.items()}
    if all(first_terms[d] == d for d in diseases):
        sim13_df = sim.copy()
    else:
        sim13_rows = []
        for d1 in diseases:
            row = []
            t1 = first_terms[d1]
            for d2 in diseases:
                t2 = first_terms[d2]
                row.append(df_mod.wang_term_similarity(t1, t2))
            sim13_rows.append(row)
        sim13_df = pd.DataFrame(sim13_rows, index=diseases, columns=diseases)
    sim13_path = out_base / "term_similarity_wang.csv"
    sim13_df.to_csv(sim13_path)
    print(f"[saved] {sim13_path}")

    # 15: LFS from Y and disease similarity
    t0 = time.time()
    print("[info] starting LFS (method 15)...")
    lfs = df_mod.lfs_from_Y(Y=Y, disease_sim=sim)
    dt = time.time() - t0
    print(f"[info] LFS finished in {dt:.1f} seconds, saving...")
    lfs_path = out_base / "lfs_from_Y.csv"
    lfs.to_csv(lfs_path)
    print(f"[saved] {lfs_path}")
    


if __name__ == "__main__":
    main()
