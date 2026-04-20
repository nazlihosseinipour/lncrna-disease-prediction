from __future__ import annotations

from pathlib import Path
import sys


def find_project_root(marker_rel: Path = Path("Data/output_data/website_full_matrix.csv")) -> Path:
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if (base / marker_rel).exists():
            return base
    raise FileNotFoundError(f"Could not locate project root containing {marker_rel}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mainfolder.utils.compare_datasets import main


if __name__ == "__main__":
    main()
