from __future__ import annotations

import argparse
import csv
import difflib
import json
from pathlib import Path
import sys


def find_project_root(marker_rel: Path = Path("Data/output_data/do_terms.csv")) -> Path:
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if (base / marker_rel).exists():
            return base
    raise FileNotFoundError(f"Could not locate project root containing {marker_rel}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mainfolder.utils.disease_mapping import disease_text_variants, informative_token_key


def parse_synonyms(row: dict[str, str]) -> list[str]:
    raw_json = str(row.get("synonyms_json") or "").strip()
    if raw_json:
        try:
            vals = json.loads(raw_json)
            return [str(v).strip() for v in vals if str(v).strip()]
        except Exception:
            pass
    raw = str(row.get("synonyms") or "")
    return [part.strip() for part in raw.split(" || ") if part.strip()]


def load_do_candidates(path: Path) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            doid = str(row.get("doid") or "").strip()
            name = str(row.get("name") or "").strip()
            if not doid or not name:
                continue
            labels = [("name", name), *[("synonym", s) for s in parse_synonyms(row)]]
            seen_labels: set[str] = set()
            for source, label in labels:
                normalized_variants = disease_text_variants(label)
                for normalized in normalized_variants:
                    if normalized in seen_labels:
                        continue
                    seen_labels.add(normalized)
                    candidates.append(
                        {
                            "doid": doid,
                            "name": name,
                            "label": label,
                            "source": source,
                            "normalized": normalized,
                            "tokens": set(informative_token_key(normalized)),
                        }
                    )
    return candidates


def build_token_index(candidates: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    index: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        for token in candidate["tokens"]:
            index.setdefault(str(token), []).append(candidate)
    return index


def load_unresolved_diseases(path: Path) -> list[str]:
    diseases: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("match_type") or "").strip() != "unresolved":
                continue
            disease = str(row.get("disease") or "").strip()
            if disease:
                diseases.append(disease)
    return diseases


def score_candidate(query_variants: list[str], query_tokens: set[str], candidate: dict[str, object]) -> tuple[float, int, float, float]:
    best_seq = 0.0
    best_jaccard = 0.0
    overlap = len(query_tokens & candidate["tokens"])
    union = len(query_tokens | candidate["tokens"])
    if union:
        best_jaccard = overlap / union
    for variant in query_variants:
        ratio = difflib.SequenceMatcher(None, variant, str(candidate["normalized"])).ratio()
        if ratio > best_seq:
            best_seq = ratio
    contains_bonus = 0.05 if any(
        variant in str(candidate["normalized"]) or str(candidate["normalized"]) in variant
        for variant in query_variants
    ) else 0.0
    source_bonus = 0.03 if candidate["source"] == "name" else 0.0
    token_bonus = overlap * 0.15
    score = token_bonus + best_jaccard + best_seq + contains_bonus + source_bonus
    return score, overlap, best_jaccard, best_seq


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest likely DO terms for unresolved disease mappings.")
    parser.add_argument("--review", default="Data/output_data/disease_term_mapping_review.csv")
    parser.add_argument("--terms", default="Data/output_data/do_terms.csv")
    parser.add_argument("--out", default="Data/output_data/disease_override_suggestions.csv")
    parser.add_argument(
        "--manual-out",
        default="Data/output_data/disease_override_manual_review.csv",
        help="Compact one-row-per-disease manual review sheet.",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.95)
    args = parser.parse_args()

    review_path = PROJECT_ROOT / args.review
    terms_path = PROJECT_ROOT / args.terms
    out_path = PROJECT_ROOT / args.out
    manual_out_path = PROJECT_ROOT / args.manual_out

    unresolved = load_unresolved_diseases(review_path)
    candidates = load_do_candidates(terms_path)
    token_index = build_token_index(candidates)

    rows: list[dict[str, object]] = []
    manual_rows: list[dict[str, object]] = []
    zero_suggestion_count = 0
    for disease in unresolved:
        variants = disease_text_variants(disease)
        token_set = set(informative_token_key(disease))
        pool: dict[tuple[str, str, str], dict[str, object]] = {}
        for token in token_set:
            for candidate in token_index.get(token, []):
                key = (str(candidate["doid"]), str(candidate["normalized"]), str(candidate["source"]))
                pool[key] = candidate
        # fall back to exact substring candidates only when token indexing finds nothing
        if not pool:
            variant_set = set(variants)
            for candidate in candidates:
                normalized = str(candidate["normalized"])
                if any(v in normalized or normalized in v for v in variant_set):
                    key = (str(candidate["doid"]), str(candidate["normalized"]), str(candidate["source"]))
                    pool[key] = candidate

        ranked: dict[str, tuple[float, int, float, float, dict[str, object]]] = {}
        for candidate in pool.values():
            score, overlap, jaccard, seq_ratio = score_candidate(variants, token_set, candidate)
            if score < args.min_score:
                continue
            current = ranked.get(str(candidate["doid"]))
            payload = (score, overlap, jaccard, seq_ratio, candidate)
            if current is None or payload[:4] > current[:4]:
                ranked[str(candidate["doid"])] = payload

        ordered = sorted(ranked.values(), key=lambda item: item[:4], reverse=True)[: args.topk]
        manual_row: dict[str, object] = {
            "disease": disease,
            "query_variants": " || ".join(variants),
            "chosen_term": "",
            "chosen_name": "",
            "note": "",
        }
        if not ordered:
            zero_suggestion_count += 1
            rows.append(
                {
                    "disease": disease,
                    "rank": "",
                    "doid": "",
                    "name": "",
                    "matched_label": "",
                    "match_source": "",
                    "score": "",
                    "token_overlap": "",
                    "jaccard": "",
                    "sequence_ratio": "",
                    "query_variants": " || ".join(variants),
                }
            )
            for rank in range(1, 4):
                manual_row[f"top{rank}_doid"] = ""
                manual_row[f"top{rank}_name"] = ""
                manual_row[f"top{rank}_matched_label"] = ""
                manual_row[f"top{rank}_match_source"] = ""
                manual_row[f"top{rank}_score"] = ""
            manual_rows.append(manual_row)
            continue

        for rank, (score, overlap, jaccard, seq_ratio, candidate) in enumerate(ordered, start=1):
            rows.append(
                {
                    "disease": disease,
                    "rank": rank,
                    "doid": candidate["doid"],
                    "name": candidate["name"],
                    "matched_label": candidate["label"],
                    "match_source": candidate["source"],
                    "score": f"{score:.4f}",
                    "token_overlap": overlap,
                    "jaccard": f"{jaccard:.4f}",
                    "sequence_ratio": f"{seq_ratio:.4f}",
                    "query_variants": " || ".join(variants),
                }
            )
            if rank <= 3:
                manual_row[f"top{rank}_doid"] = candidate["doid"]
                manual_row[f"top{rank}_name"] = candidate["name"]
                manual_row[f"top{rank}_matched_label"] = candidate["label"]
                manual_row[f"top{rank}_match_source"] = candidate["source"]
                manual_row[f"top{rank}_score"] = f"{score:.4f}"

        for rank in range(len(ordered) + 1, 4):
            manual_row[f"top{rank}_doid"] = ""
            manual_row[f"top{rank}_name"] = ""
            manual_row[f"top{rank}_matched_label"] = ""
            manual_row[f"top{rank}_match_source"] = ""
            manual_row[f"top{rank}_score"] = ""
        manual_rows.append(manual_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "disease",
                "rank",
                "doid",
                "name",
                "matched_label",
                "match_source",
                "score",
                "token_overlap",
                "jaccard",
                "sequence_ratio",
                "query_variants",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    manual_out_path.parent.mkdir(parents=True, exist_ok=True)
    with manual_out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "disease",
                "query_variants",
                "top1_doid",
                "top1_name",
                "top1_matched_label",
                "top1_match_source",
                "top1_score",
                "top2_doid",
                "top2_name",
                "top2_matched_label",
                "top2_match_source",
                "top2_score",
                "top3_doid",
                "top3_name",
                "top3_matched_label",
                "top3_match_source",
                "top3_score",
                "chosen_term",
                "chosen_name",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(manual_rows)

    print(f"Unresolved diseases        : {len(unresolved)}")
    print(f"Suggestion output          : {out_path}")
    print(f"Manual review output       : {manual_out_path}")
    print(f"Diseases with no suggestion: {zero_suggestion_count}")


if __name__ == "__main__":
    main()
