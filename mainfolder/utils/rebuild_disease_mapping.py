from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def find_project_root(marker_rel: Path = Path("Data/raw/website_alldata.csv")) -> Path:
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if (base / marker_rel).exists():
            return base
    raise FileNotFoundError(f"Could not locate project root containing {marker_rel}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mainfolder.utils.disease_mapping import build_disease_term_mapping, ensure_disease_override_csv


def extract_quoted_synonym(line: str):
    q1 = line.find('"')
    if q1 == -1:
        return None

    buf = []
    escape = False
    for ch in line[q1 + 1 :]:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            return "".join(buf)
        buf.append(ch)
    return None


def parse_do_obo(obo_path: Path):
    terms = []
    edges = []
    current = None

    def flush_term(t):
        if not t or not t.get("id"):
            return

        is_obsolete = str(t.get("is_obsolete", "false")).lower() == "true"
        if is_obsolete:
            return

        syns = [str(x).strip() for x in t.get("synonyms", []) if str(x).strip()]
        fallback_syn = str(t.get("name", "")).strip() or str(t.get("id", "")).strip()
        if not syns and fallback_syn:
            syns = [fallback_syn]

        terms.append(
            {
                "doid": t.get("id", ""),
                "name": t.get("name", ""),
                "synonyms": " || ".join(syns),
                "synonyms_json": json.dumps(syns, ensure_ascii=False),
                "synonym_count": len(syns),
            }
        )

        for parent in t.get("is_a", []):
            edges.append({"child": t["id"], "parent": parent})

    with obo_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            if line == "[Term]":
                flush_term(current)
                current = {"synonyms": [], "is_a": []}
                continue

            if line.startswith("[") and line != "[Term]":
                flush_term(current)
                current = None
                continue

            if current is None:
                continue

            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("is_obsolete: "):
                current["is_obsolete"] = line.split(":", 1)[1].strip()
            elif line.startswith("is_a: "):
                parent = line[6:].split(" ! ", 1)[0].strip()
                if parent:
                    current["is_a"].append(parent)
            elif line.startswith("synonym: "):
                txt = extract_quoted_synonym(line)
                if txt is not None:
                    current["synonyms"].append(txt)

    flush_term(current)

    terms_df = pd.DataFrame(terms).drop_duplicates(subset=["doid"])
    name_by_doid = terms_df.set_index("doid")["name"].to_dict()
    edges_df = pd.DataFrame(edges).drop_duplicates()
    if not edges_df.empty:
        edges_df["child_name"] = edges_df["child"].map(name_by_doid).fillna(edges_df["child"])
        edges_df["parent_name"] = edges_df["parent"].map(name_by_doid).fillna(edges_df["parent"])
    return terms_df, edges_df


def main():
    p = argparse.ArgumentParser(description="Rebuild DO term helpers and disease-term mapping without rerunning the full notebook.")
    p.add_argument("--obo", default="Data/raw/HumanDO.obo", help="Path to HumanDO.obo")
    p.add_argument(
        "--Y",
        default="Data/output_data/website_disease_matrix.csv",
        help="Disease matrix CSV used only to read disease column names",
    )
    p.add_argument("--outdir", default="Data/output_data", help="Output directory for do_terms/do_edges/mapping files")
    p.add_argument(
        "--overrides",
        default="Data/raw/disease_term_overrides.csv",
        help="CSV with columns disease,term,note for manual mapping overrides",
    )
    args = p.parse_args()

    obo_path = PROJECT_ROOT / args.obo
    y_path = PROJECT_ROOT / args.Y
    outdir = PROJECT_ROOT / args.outdir
    overrides_path = PROJECT_ROOT / args.overrides

    if not obo_path.exists():
        raise FileNotFoundError(f"Missing ontology file: {obo_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing disease matrix file: {y_path}")

    outdir.mkdir(parents=True, exist_ok=True)
    ensure_disease_override_csv(overrides_path)

    do_terms_path = outdir / "do_terms.csv"
    do_edges_path = outdir / "do_edges.csv"
    do_map_path = outdir / "disease_terms_mapping.csv"
    do_review_path = outdir / "disease_term_mapping_review.csv"

    terms_df, edges_df = parse_do_obo(obo_path)
    terms_df.to_csv(do_terms_path, index=False)
    edges_df.to_csv(do_edges_path, index=False)
    print(f"[saved] {do_terms_path} shape={terms_df.shape}")
    print(f"[saved] {do_edges_path} shape={edges_df.shape}")

    disease_mat = pd.read_csv(y_path, dtype=str)
    diseases = [c for c in disease_mat.columns if c != "ID"]
    map_df, review_df = build_disease_term_mapping(
        diseases=diseases,
        do_terms_df=terms_df,
        overrides_path=overrides_path,
    )
    map_df.to_csv(do_map_path, index=False)
    review_df.to_csv(do_review_path, index=False)

    mapped_n = int((map_df["term"].fillna("") != "").sum())
    print(f"[saved] {do_map_path} shape={map_df.shape} (mapped={mapped_n}/{len(map_df)})")
    print(f"[saved] {do_review_path} shape={review_df.shape}")
    print(f"[info] overrides file: {overrides_path}")

    ambiguous = review_df.loc[review_df["candidate_count"] > 1, ["disease", "selected_name", "match_type"]]
    if not ambiguous.empty:
        print("sample ambiguous mappings:", ambiguous.head(10).to_dict("records"))

    unmatched = map_df.loc[map_df["term"].fillna("") == "", "disease"]
    if not unmatched.empty:
        print("sample unmatched diseases:", unmatched.head(20).tolist())


if __name__ == "__main__":
    main()
