from __future__ import annotations

import argparse
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

from mainfolder.utils.disease_override_merge import merge_manual_review_into_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge reviewed disease mappings into disease_term_overrides.csv.")
    parser.add_argument(
        "--review",
        default="Data/output_data/disease_override_manual_review_all_90_mapped.csv",
        help="Manual review CSV containing disease/chosen_term/note columns.",
    )
    parser.add_argument(
        "--overrides",
        default="Data/raw/disease_term_overrides.csv",
        help="Destination override CSV with disease,term,note columns.",
    )
    parser.add_argument(
        "--terms",
        default="Data/output_data/do_terms.csv",
        help="Active DO terms CSV used to validate chosen_term values.",
    )
    args = parser.parse_args()

    summary = merge_manual_review_into_overrides(
        PROJECT_ROOT / args.review,
        PROJECT_ROOT / args.overrides,
        do_terms_path=PROJECT_ROOT / args.terms,
    )

    print(f"review rows selected : {summary['selected_rows']}")
    print(f"overrides added      : {summary['added']}")
    print(f"overrides updated    : {summary['updated']}")
    print(f"overrides unchanged  : {summary['unchanged']}")
    print(f"total override rows  : {summary['total_overrides']}")
    print(f"output               : {PROJECT_ROOT / args.overrides}")


if __name__ == "__main__":
    main()
