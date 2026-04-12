from __future__ import annotations

import argparse
from pathlib import Path
import sys


def find_project_root(marker_rel: Path = Path("Data/raw/website_alldata.csv")) -> Path:
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if (base / marker_rel).exists():
            return base
    raise FileNotFoundError(f"Could not locate project root containing {marker_rel}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mainfolder.utils.unresolved_refetch import RefetchPaths, refetch_human_unresolved_only


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Refetch only the current unresolved human website IDs and update Data/output_data in place."
    )
    parser.add_argument("--timeout", type=int, default=10, help="Per-request timeout in seconds")
    parser.add_argument("--sleep_sec", type=float, default=0.05, help="Sleep between requests")
    parser.add_argument("--progress_every", type=int, default=25, help="Print progress every N IDs")
    parser.add_argument("--max_ids", type=int, help="Optional cap for a quick partial rerun")
    args = parser.parse_args(argv)

    out_dir = PROJECT_ROOT / "Data/output_data"
    paths = RefetchPaths(
        raw_website=PROJECT_ROOT / "Data/raw/website_alldata.csv",
        website_sequences=out_dir / "website_sequences.csv",
        website_full=out_dir / "website_full_matrix.csv",
        website_oop=out_dir / "website_sequences_for_oop.csv",
        fetch_report=out_dir / "sequence_fetch_report.csv",
        unresolved_ids=out_dir / "unresolved_ids.csv",
        non_human_unresolved_ids=out_dir / "non_human_unresolved_ids.csv",
        ambiguous_alternatives=out_dir / "ambiguous_symbol_alternatives.csv",
        website_missing_ids=out_dir / "website_missing_sequence_ids.txt",
    )

    summary = refetch_human_unresolved_only(
        paths,
        timeout=args.timeout,
        sleep_sec=args.sleep_sec,
        progress_every=args.progress_every,
        max_ids=args.max_ids,
    )

    print("\n=== Refetch Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
