"""
process_website_alldata_v2.py  (Notebook-friendly cell)

Goals:
- Keep your same outputs in Data/output_data/
- Use a library for Disease Ontology parsing (obonet)
- Remove duplicates + make the pipeline one-pass
- Ensure we don't miss: do_terms.csv, do_edges.csv, disease_terms_mapping.csv

Outputs (Data/output_data/):
- ncrna_symbol_list.txt
- website_sequences.csv
- sequence_fetch_report.csv
- website_disease_matrix.csv
- website_full_matrix.csv
- dinuc_props.csv
- do_terms.csv
- do_edges.csv
- disease_terms_mapping.csv
"""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Iterable

import obonet
import pandas as pd
import requests


# =========================
# Helpers
# =========================

def find_project_file(rel_path: Path | str) -> Path:
    """Find a file/dir by walking up from cwd. Works in notebooks."""
    rel_path = Path(rel_path)
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        cand = base / rel_path
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find {rel_path} from {cwd}")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def norm_text(s: str) -> str:
    """Normalization for matching (lowercase, strip punctuation, collapse whitespace)."""
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


def first_existing(paths: Iterable[Path]) -> Path:
    for p in paths:
        try:
            return find_project_file(p)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"None of the input files found: {paths}")


# =========================
# 1) Load your website input files
# =========================

IN_WEBSITE = first_existing(
    [
        Path("Data/raw/website_data.csv"),  # preferred if present
        Path("Data/raw/website_alldata.csv"),
        Path("Data/raw/website_alldata.tsv"),
    ]
)

# Outputs are always under Data/output_data relative to the raw input
OUT_DIR = ensure_dir(IN_WEBSITE.parent.parent / "output_data")

sep = "\t" if IN_WEBSITE.suffix.lower() == ".tsv" else ","
website_df = safe_read_csv(IN_WEBSITE, dtype=str, sep=sep).fillna("")

# Expected columns vary by project; these are common patterns.
# We need: symbol (lncRNA symbol), species, type, disease, and optionally ensembl_id / transcript_id.
colmap = {c.lower(): c for c in website_df.columns}


def pick_col(*names: str) -> str:
    for n in names:
        if n.lower() in colmap:
            return colmap[n.lower()]
    raise KeyError(f"Could not find any of columns: {names}. Available: {list(website_df.columns)}")


COL_SYMBOL = pick_col("ncrna_symbol", "symbol", "lncrna_symbol", "lnc_symbol")
COL_SPECIES = pick_col("species", "organism")
COL_TYPE = pick_col("type", "rna_type", "category", "ncrna category")
COL_DISEASE = pick_col("disease", "disease_name", "disease name")

# optional id used for fetching sequences:
COL_ENSEMBL = None
for candidate in ("ensembl_id", "ensembl_gene_id", "gene_id", "transcript_id"):
    if candidate.lower() in colmap:
        COL_ENSEMBL = colmap[candidate.lower()]
        break

# Filter: Homo sapiens + LncRNA
df_hs = website_df[
    website_df[COL_SPECIES].str.lower().str.strip().eq("homo sapiens")
    & website_df[COL_TYPE].str.lower().str.contains("lncrna")
].copy()

if df_hs.empty:
    raise RuntimeError("Filtered DataFrame is empty; check species/type column names/values.")

# =========================
# 2) Save ncrna_symbol_list.txt
# =========================
symbols = sorted(set(s for s in df_hs[COL_SYMBOL].astype(str).tolist() if s.strip()))
(OUT_DIR / "ncrna_symbol_list.txt").write_text("\n".join(symbols), encoding="utf-8")

print(f"[saved] {OUT_DIR/'ncrna_symbol_list.txt'} ({len(symbols)} symbols)")

# =========================
# 3) Fetch sequences -> website_sequences.csv + sequence_fetch_report.csv
# =========================

# Basic Ensembl REST fetcher.
ENSEMBL_BASE = "https://rest.ensembl.org"
HEADERS_JSON = {"Content-Type": "application/json", "Accept": "application/json"}
HEADERS_TEXT = {"Content-Type": "text/plain"}
SESSION = requests.Session()


def fetch_ensembl_seq(ensembl_id: str, timeout: int = 20) -> tuple[str | None, str]:
    url = f"{ENSEMBL_BASE}/sequence/id/{ensembl_id}"
    try:
        r = SESSION.get(url, headers=HEADERS_TEXT, timeout=timeout)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        seq = (r.text or "").strip().upper()
        if not seq or "{" in seq[:2]:  # sometimes JSON error
            return None, "empty_or_json"
        return seq, "ok"
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception:
        return None, "error"


def lookup_symbol(symbol: str, timeout: int = 10) -> tuple[str | None, str]:
    """Resolve a symbol to an Ensembl ID using lookup then xref."""
    lookup_url = f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{symbol}"
    xref_url = f"{ENSEMBL_BASE}/xrefs/symbol/homo_sapiens/{symbol}"
    try:
        r = SESSION.get(lookup_url, headers=HEADERS_JSON, timeout=timeout)
        if r.status_code == 200 and isinstance(r.json(), dict):
            eid = r.json().get("id")
            if eid:
                return eid, "lookup"
    except Exception:
        pass
    try:
        rx = SESSION.get(xref_url, headers=HEADERS_JSON, timeout=timeout)
        if rx.status_code == 200:
            data = rx.json()
            for item in data:
                if item.get("id"):
                    return item["id"], "xref"
    except Exception:
        pass
    return None, "no_match"


# Cache to avoid re-fetching
SEQ_CACHE = OUT_DIR / "seq_cache.csv"
cache: dict[str, str | None] = {}
if SEQ_CACHE.exists():
    _c = pd.read_csv(SEQ_CACHE, dtype=str).fillna("")
    for _, row in _c.iterrows():
        cache[row["key"]] = row["seq"] if row["seq"] != "" else None

# Build unique targets (ID + optional Ensembl)
targets = df_hs[[COL_SYMBOL] + ([COL_ENSEMBL] if COL_ENSEMBL else [])].copy()
targets.columns = ["id", *(["ensembl_id"] if COL_ENSEMBL else [])]
targets["id"] = targets["id"].astype(str).str.strip()
if "ensembl_id" in targets.columns:
    targets["ensembl_id"] = targets["ensembl_id"].astype(str).str.strip()
targets = targets[targets["id"] != ""].drop_duplicates(subset=["id"])

seq_rows: list[dict] = []
rep_rows: list[dict] = []

TIME_CAP_SECONDS = 60 * 25  # adjust if needed
t0 = time.time()

for i, row in targets.iterrows():
    elapsed = time.time() - t0
    if elapsed > TIME_CAP_SECONDS:
        rep_rows.append({"ID": row["id"], "status": "time_cap_stop", "elapsed_s": int(elapsed)})
        break

    sym = row["id"]
    ens = row.get("ensembl_id", "") or ""
    cache_keys = [sym]
    if ens:
        cache_keys.append(ens)

    cached_key = next((k for k in cache_keys if k in cache), None)
    if cached_key is not None:
        cached_seq = cache[cached_key]
        rep_rows.append({"ID": sym, "status": "cached_ok" if cached_seq else "cached_missing", "elapsed_s": int(elapsed)})
        if cached_seq:
            seq_rows.append({"ID": sym, "seqs": cached_seq})
        continue

    used_id = ens
    seq = status = None
    if ens:
        seq, status = fetch_ensembl_seq(ens)
    if not seq:
        resolved, lookup_status = lookup_symbol(sym)
        used_id = resolved or sym
        if resolved:
            seq, status = fetch_ensembl_seq(resolved)
            status = status or "resolved"
        else:
            status = f"lookup_failed:{lookup_status}"

    cache_value = seq if seq else None
    for k in cache_keys + ([used_id] if used_id else []):
        if k:
            cache[k] = cache_value

    rep_rows.append({"ID": sym, "status": status or "missing", "elapsed_s": int(time.time() - t0)})
    if seq:
        seq_rows.append({"ID": sym, "seqs": seq})

seq_df = pd.DataFrame(seq_rows).drop_duplicates(subset=["ID"])
rep_df = pd.DataFrame(rep_rows)

seq_df.to_csv(OUT_DIR / "website_sequences.csv", index=False)
rep_df.to_csv(OUT_DIR / "sequence_fetch_report.csv", index=False)

# update cache file
pd.DataFrame([{"key": k, "seq": (v or "")} for k, v in cache.items()]).to_csv(SEQ_CACHE, index=False)

print(f"[saved] {OUT_DIR/'website_sequences.csv'} ({len(seq_df)} rows)")
print(f"[saved] {OUT_DIR/'sequence_fetch_report.csv'} ({len(rep_df)} rows)")
print(f"[saved] {SEQ_CACHE} ({len(cache)} cached keys)")

# =========================
# 4) Build website_disease_matrix.csv
# =========================

# One-hot from (symbol, disease).
pairs = df_hs[[COL_SYMBOL, COL_DISEASE]].copy()
pairs.columns = ["ID", "disease"]
pairs["ID"] = pairs["ID"].astype(str).str.strip()
pairs["disease"] = pairs["disease"].astype(str).str.strip()
pairs = pairs[(pairs["ID"] != "") & (pairs["disease"] != "")].drop_duplicates()

pairs["value"] = 1
y = pairs.pivot_table(index="ID", columns="disease", values="value", aggfunc="max", fill_value=0)
y.reset_index(inplace=True)

y.to_csv(OUT_DIR / "website_disease_matrix.csv", index=False)
print(f"[saved] {OUT_DIR/'website_disease_matrix.csv'} shape={y.shape}")

# =========================
# 5) Build website_full_matrix.csv (ONLY ONCE)
# =========================
# Keep IDs that have sequences (inner join).
full = pd.merge(seq_df, y, on="ID", how="inner")
full.to_csv(OUT_DIR / "website_full_matrix.csv", index=False)
print(f"[saved] {OUT_DIR/'website_full_matrix.csv'} shape={full.shape}")

# =========================
# 6) Dinucleotide properties from sequences
# =========================

def dinuc_props(seq: str) -> dict:
    seq = (seq or "").upper()
    n = len(seq)
    if n < 2:
        return {}
    dinucs = [a + b for a in "ACGT" for b in "ACGT"]
    counts = {d: 0 for d in dinucs}
    total = 0
    for i in range(n - 1):
        d = seq[i : i + 2]
        if d in counts:
            counts[d] += 1
            total += 1
    props = {f"p_{d}": (counts[d] / total if total else 0.0) for d in dinucs}
    props["len"] = n
    return props


dinuc_rows = []
for _, r in seq_df.iterrows():
    props = dinuc_props(r["seqs"])
    if props:
        props["ID"] = r["ID"]
        dinuc_rows.append(props)

dinuc_df = pd.DataFrame(dinuc_rows)
dinuc_df.to_csv(OUT_DIR / "dinuc_props.csv", index=False)
print(f"[saved] {OUT_DIR/'dinuc_props.csv'} shape={dinuc_df.shape}")

# =========================
# 7) Disease Ontology -> do_terms.csv + do_edges.csv using obonet
# =========================

DO_OBO = find_project_file(Path("Data/raw/HumanDO.obo"))

G = obonet.read_obo(DO_OBO)  # MultiDiGraph

# Extract is_a edges robustly
edge_rows = []
for u, v, k, d in G.edges(keys=True, data=True):
    rel = d.get("relation") or d.get("typedef") or k
    if rel == "is_a":
        edge_rows.append({"parent": v, "child": u})

do_edges = pd.DataFrame(edge_rows).drop_duplicates()

# Build parents/children maps for terms.csv
children_map: dict[str, set[str]] = {}
parents_map: dict[str, set[str]] = {}

for _, er in do_edges.iterrows():
    p = er["parent"]
    c = er["child"]
    children_map.setdefault(p, set()).add(c)
    parents_map.setdefault(c, set()).add(p)


def extract_syns(node_data) -> list[str]:
    syns = node_data.get("synonym", [])
    if isinstance(syns, str):
        syns = [syns]
    out = []
    for s in syns:
        if '"' in s:
            parts = s.split('"')
            if len(parts) >= 2:
                out.append(parts[1])
            else:
                out.append(s)
        else:
            out.append(s)
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


term_rows = []
for doid, data in G.nodes(data=True):
    if data.get("is_obsolete") == "true":
        continue
    name = data.get("name", "")
    syns = extract_syns(data)
    parents = sorted(parents_map.get(doid, []))
    children = sorted(children_map.get(doid, []))
    term_rows.append(
        {
            "doid": doid,
            "name": name,
            "parents": ";".join(parents),
            "children": ";".join(children),
            "synonyms": ";".join(syns),
        }
    )

do_terms = pd.DataFrame(term_rows)

do_terms.to_csv(OUT_DIR / "do_terms.csv", index=False)
do_edges.to_csv(OUT_DIR / "do_edges.csv", index=False)

print(f"[saved] {OUT_DIR/'do_terms.csv'} ({len(do_terms)} terms)")
print(f"[saved] {OUT_DIR/'do_edges.csv'} ({len(do_edges)} edges)")

# =========================
# 8) disease_terms_mapping.csv (baseline normalized exact match)
# =========================

disease_cols = list(y.columns)
diseases = disease_cols[1:] if disease_cols and disease_cols[0].lower() == "id" else disease_cols

terms = do_terms.copy()
terms["name_norm"] = terms["name"].apply(norm_text)
terms["syn_norms"] = terms["synonyms"].apply(lambda s: [norm_text(x) for x in str(s).split(";") if norm_text(x)])

# Build lookup dicts
name_to_doid: dict[str, str] = {}
syn_to_doid: dict[str, str] = {}

for _, row in terms.iterrows():
    doid = row["doid"]
    nn = row["name_norm"]
    if nn and nn not in name_to_doid:
        name_to_doid[nn] = doid
    for sn in row["syn_norms"]:
        if sn and sn not in syn_to_doid:
            syn_to_doid[sn] = doid

mapping_rows = []
for d in diseases:
    dn = norm_text(d)
    doid = name_to_doid.get(dn, "") or syn_to_doid.get(dn, "")
    mapping_rows.append({"disease": d, "term": doid})

map_df = pd.DataFrame(mapping_rows)
map_df.to_csv(OUT_DIR / "disease_terms_mapping.csv", index=False)
print(f"[saved] {OUT_DIR/'disease_terms_mapping.csv'} ({len(map_df)} rows)")

# =========================
# 9) Sanity checks (so you instantly see if anything is missing)
# =========================

expected = [
    "ncrna_symbol_list.txt",
    "website_sequences.csv",
    "sequence_fetch_report.csv",
    "website_disease_matrix.csv",
    "website_full_matrix.csv",
    "dinuc_props.csv",
    "do_terms.csv",
    "do_edges.csv",
    "disease_terms_mapping.csv",
]
missing = [f for f in expected if not (OUT_DIR / f).exists()]
if missing:
    raise RuntimeError(f"Missing expected outputs: {missing}")

print("\n✅ All expected outputs exist in:", OUT_DIR)
print("Full matrix rows:", len(full), "| seq rows:", len(seq_df), "| disease-matrix rows:", len(y))
print("DO terms:", len(do_terms), "| DO edges:", len(do_edges), "| mapping rows:", len(map_df))
