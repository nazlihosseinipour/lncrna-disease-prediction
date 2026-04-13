from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys


def set_max_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


set_max_csv_field_size_limit()


def find_project_root(marker_rel: Path = Path("Data/output_data/unresolved_ids_recovery_residual.csv")) -> Path:
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if (base / marker_rel).exists():
            return base
    script_dir = Path(__file__).resolve().parent
    for base in (script_dir, *script_dir.parents):
        if (base / marker_rel).exists():
            return base
    raise FileNotFoundError(f"Could not locate project root containing {marker_rel}")


PROJECT_ROOT = find_project_root()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(fh)]


def parse_reason_labels(reason: str) -> tuple[str, tuple[str, ...]]:
    parts = [part.strip() for part in str(reason or "").split(";") if part.strip()]
    if not parts:
        return "", ()
    route = parts[0]
    labels: list[str] = []
    for part in parts[1:]:
        label = part.rsplit(":", 1)[-1].strip() if ":" in part else part
        if label and label not in labels:
            labels.append(label)
    if not labels and route:
        labels.append(route)
    return route, tuple(labels)


def print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    print("-" * len(title))
    if not counter:
        print("(none)")
        print()
        return
    width = max(len(key) for key in counter)
    for key, count in counter.most_common():
        print(f"{key:<{width}}  {count:>6}")
    print()


def count_prefixes(ids: list[str], n: int) -> Counter[str]:
    counter: Counter[str] = Counter()
    for value in ids:
        cleaned = str(value or "").strip().upper()
        if not cleaned:
            continue
        counter[cleaned[:n]] += 1
    return counter


def fasta_headers(path: Path, limit: int) -> list[str]:
    headers: list[str] = []
    if not path.exists():
        return headers
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            headers.append(line.rstrip())
            if len(headers) >= limit:
                break
    return headers


def print_lncipedia_examples(rows: list[dict[str, str]], fasta_path: Path, example_count: int) -> None:
    lnc_rows: list[dict[str, str]] = []
    for row in rows:
        route, labels = parse_reason_labels(row.get("reason", ""))
        if route == "lncipedia_local" or "lncipedia_fasta_empty" in labels:
            lnc_rows.append(row)

    print("LNCipedia residual examples")
    print("---------------------------")
    print(f"Residual LNCipedia failures: {len(lnc_rows)}")
    if not lnc_rows:
        print("(none)")
        print()
        return

    ids = [row.get("id", "") for row in lnc_rows[:example_count]]
    headers = fasta_headers(fasta_path, example_count)

    left_width = max(len("Residual ID"), *(len(x) for x in ids)) if ids else len("Residual ID")
    right_width = max(len("FASTA header"), *(len(x) for x in headers)) if headers else len("FASTA header")
    print(f"{'Residual ID':<{left_width}}  {'FASTA header':<{right_width}}")
    print(f"{'-' * left_width}  {'-' * right_width}")
    for idx in range(max(len(ids), len(headers))):
        left = ids[idx] if idx < len(ids) else ""
        right = headers[idx] if idx < len(headers) else ""
        print(f"{left:<{left_width}}  {right:<{right_width}}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze unresolved_ids_recovery_residual.csv")
    parser.add_argument(
        "--residual",
        default=str(PROJECT_ROOT / "Data/output_data/unresolved_ids_recovery_residual.csv"),
        help="Path to unresolved_ids_recovery_residual.csv",
    )
    parser.add_argument(
        "--lncipedia-fasta",
        default=str(PROJECT_ROOT / "Data/raw/lncipedia_5_2_hc.fasta"),
        help="Path to LNCipedia FASTA used for comparison examples",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="How many LNCipedia residual/header examples to print",
    )
    args = parser.parse_args(argv)

    residual_path = Path(args.residual).resolve()
    fasta_path = Path(args.lncipedia_fasta).resolve()
    rows = load_rows(residual_path)

    reason_counter: Counter[str] = Counter()
    route_counter: Counter[str] = Counter()
    ids = [row.get("id", "") for row in rows]

    for row in rows:
        route, labels = parse_reason_labels(row.get("reason", ""))
        if route:
            route_counter[route] += 1
        for label in labels:
            reason_counter[label] += 1

    print(f"Residual file: {residual_path}")
    print(f"Rows: {len(rows)}\n")

    print_counter("Count by route", route_counter)
    print_counter("Count by reason label", reason_counter)

    for prefix_len in (2, 3, 4):
        print_counter(f"Count by ID prefix (first {prefix_len} chars)", count_prefixes(ids, prefix_len))

    print_lncipedia_examples(rows, fasta_path, max(1, args.examples))


if __name__ == "__main__":
    main()
