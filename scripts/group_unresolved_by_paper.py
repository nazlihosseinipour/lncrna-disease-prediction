from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


UNICODE_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
    }
)


def find_project_root(marker_rel: Path = Path("Data/raw/website_alldata.csv")) -> Path:
    script_dir = Path(__file__).resolve().parent
    for base in (Path.cwd().resolve(), *Path.cwd().resolve().parents, script_dir, *script_dir.parents):
        if (base / marker_rel).exists():
            return base
    raise FileNotFoundError(f"Could not locate project root containing {marker_rel}")


PROJECT_ROOT = find_project_root()


def normalize_id(value: str) -> str:
    return str(value or "").strip().translate(UNICODE_DASH_TRANSLATION)


def classify_bucket(raw_id: str) -> str:
    value = normalize_id(raw_id)
    if re.match(r"^NON(?:HSAT|MMUT)\d+(?:\.\d+)?$", value, re.I):
        return "noncode"
    if re.match(r"^fantom(?:3|5)_[A-Za-z0-9._-]+$", value, re.I):
        return "fantom"
    if re.match(r"^(?:hsa_)?circ[_-][A-Za-z0-9._-]+$", value, re.I):
        return "circrna"
    if re.match(r"^(?:LNC[-_].+|LNCV.+|lnc[-_].+|Lnc[-_].+|lnr.+)$", value):
        return "lncipedia_like"
    if re.match(r"^(?:TCONS?[_A-Za-z0-9.-]+|XLOC[_A-Za-z0-9.-]+)$", value, re.I):
        return "assembler"
    if re.match(r"^LOC\d+(?:[._:-].+)?$", value, re.I):
        return "loc_symbol"
    if re.match(r"^(?:(?:AC|AL)\d{5,6}|RP\d+-[A-Za-z0-9]+|(?:CTD|CTB|CTC|CTA)-[A-Za-z0-9]+|CITF\d+-[A-Za-z0-9]+)(?:\.\d+)?(?:[-:]\d+)?$", value, re.I):
        return "clone_like"
    if re.match(r"^(?:uc\d{3}[a-z]{3}(?:\.\d+)?|uc\.\d+(?:[+-])?)$", value, re.I):
        return "ucsc"
    if re.match(r"^(?:ENSG|ENST)\d+(?:\.\d+)?$", value, re.I):
        return "ensembl"
    return "other"


def load_unresolved_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return [str(row.get("id") or "").strip() for row in reader if str(row.get("id") or "").strip()]


def truncate(text: str, limit: int = 180) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Group unresolved IDs by source paper to support supplement-based mapping.")
    parser.add_argument("--unresolved", default=str(PROJECT_ROOT / "Data/output_data/unresolved_ids.csv"))
    parser.add_argument("--website", default=str(PROJECT_ROOT / "Data/raw/website_alldata.csv"))
    parser.add_argument("--detail-out", default=str(PROJECT_ROOT / "Data/output_data/unresolved_ids_by_paper.csv"))
    parser.add_argument("--summary-out", default=str(PROJECT_ROOT / "Data/output_data/unresolved_paper_summary.csv"))
    args = parser.parse_args()

    unresolved_path = Path(args.unresolved).resolve()
    website_path = Path(args.website).resolve()
    detail_out = Path(args.detail_out).resolve()
    summary_out = Path(args.summary_out).resolve()

    unresolved_ids = load_unresolved_ids(unresolved_path)
    unresolved_set = set(unresolved_ids)
    if not unresolved_set:
        print(f"No unresolved IDs found in {unresolved_path}")
        return

    detail_index: dict[tuple[str, str], dict[str, object]] = {}
    paper_summary: dict[str, dict[str, object]] = {}
    unmatched_ids = set(unresolved_set)

    with website_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_id = str(row.get("ncRNA Symbol") or "").strip()
            if raw_id not in unresolved_set:
                continue
            unmatched_ids.discard(raw_id)
            pubmed_id = str(row.get("PubMed ID") or "").strip() or "missing_pubmed"
            key = (raw_id, pubmed_id)
            bucket = classify_bucket(raw_id)

            if key not in detail_index:
                detail_index[key] = {
                    "id": raw_id,
                    "normalized_id": normalize_id(raw_id),
                    "bucket": bucket,
                    "pubmed_id": pubmed_id,
                    "row_count": 0,
                    "diseases": set(),
                    "category": str(row.get("ncRNA Category") or "").strip(),
                    "description_excerpt": truncate(row.get("Description") or ""),
                }
            detail_index[key]["row_count"] = int(detail_index[key]["row_count"]) + 1
            disease_name = str(row.get("Disease Name") or "").strip()
            if disease_name:
                detail_index[key]["diseases"].add(disease_name)

            if pubmed_id not in paper_summary:
                paper_summary[pubmed_id] = {
                    "pubmed_id": pubmed_id,
                    "unresolved_ids": set(),
                    "buckets": Counter(),
                    "rows": 0,
                    "example_ids": [],
                }
            paper_summary[pubmed_id]["unresolved_ids"].add(raw_id)
            paper_summary[pubmed_id]["buckets"][bucket] += 1
            paper_summary[pubmed_id]["rows"] = int(paper_summary[pubmed_id]["rows"]) + 1
            if len(paper_summary[pubmed_id]["example_ids"]) < 8 and raw_id not in paper_summary[pubmed_id]["example_ids"]:
                paper_summary[pubmed_id]["example_ids"].append(raw_id)

    detail_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    with detail_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "normalized_id",
                "bucket",
                "pubmed_id",
                "row_count",
                "unique_disease_count",
                "example_diseases",
                "category",
                "description_excerpt",
            ],
        )
        writer.writeheader()
        for (_raw_id, _pubmed), payload in sorted(detail_index.items(), key=lambda item: (item[0][1], item[0][0])):
            diseases = sorted(payload["diseases"])
            writer.writerow(
                {
                    "id": payload["id"],
                    "normalized_id": payload["normalized_id"],
                    "bucket": payload["bucket"],
                    "pubmed_id": payload["pubmed_id"],
                    "row_count": payload["row_count"],
                    "unique_disease_count": len(diseases),
                    "example_diseases": " | ".join(diseases[:5]),
                    "category": payload["category"],
                    "description_excerpt": payload["description_excerpt"],
                }
            )

    with summary_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "pubmed_id",
                "unresolved_id_count",
                "website_row_count",
                "bucket_counts_json",
                "example_ids",
            ],
        )
        writer.writeheader()
        for pubmed_id, payload in sorted(
            paper_summary.items(),
            key=lambda item: (-len(item[1]["unresolved_ids"]), item[0]),
        ):
            writer.writerow(
                {
                    "pubmed_id": pubmed_id,
                    "unresolved_id_count": len(payload["unresolved_ids"]),
                    "website_row_count": payload["rows"],
                    "bucket_counts_json": json.dumps(dict(sorted(payload["buckets"].items())), sort_keys=True),
                    "example_ids": " | ".join(payload["example_ids"]),
                }
            )

    bucket_counts = Counter(classify_bucket(rid) for rid in unresolved_ids)
    print(f"Unresolved IDs loaded : {len(unresolved_ids)}")
    print(f"Matched to website rows: {len(unresolved_set) - len(unmatched_ids)}")
    print(f"Unmatched in website   : {len(unmatched_ids)}")
    print(f"Detail CSV             : {detail_out}")
    print(f"Summary CSV            : {summary_out}")
    print("\nBucket counts")
    print("-" * 54)
    for bucket, count in bucket_counts.most_common():
        print(f"{bucket:<24} {count:>8}")

    print("\nTop papers")
    print("-" * 54)
    for pubmed_id, payload in sorted(
        paper_summary.items(),
        key=lambda item: (-len(item[1]["unresolved_ids"]), item[0]),
    )[:20]:
        print(
            f"{pubmed_id:<12} ids={len(payload['unresolved_ids']):>4} "
            f"rows={payload['rows']:>4} buckets={dict(sorted(payload['buckets'].items()))}"
        )

    if unmatched_ids:
        print("\nSample unmatched IDs")
        print("-" * 54)
        for raw_id in sorted(unmatched_ids)[:20]:
            print(raw_id)


if __name__ == "__main__":
    main()
