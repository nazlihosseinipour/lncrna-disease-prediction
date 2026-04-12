from pathlib import Path
from typing import Any

import pandas as pd


def shape_from_result(data: Any, columns: list[str] | None = None) -> tuple[int, int]:
    if isinstance(data, pd.DataFrame):
        return int(data.shape[0]), int(data.shape[1])
    if hasattr(data, "shape") and getattr(data, "ndim", 0) == 2:
        return int(data.shape[0]), int(data.shape[1])

    rows = len(data) if data is not None else 0
    if columns is not None:
        cols = len(columns)
    elif rows and hasattr(data[0], "__len__"):
        cols = len(data[0])
    else:
        cols = 0
    return int(rows), int(cols)


def add_shape_record(
    records: list[dict[str, Any]],
    *,
    feature_group: str,
    feature_name: str,
    output_path: Path,
    rows: int,
    cols: int,
    source_name: str | None = None,
    method_id: int | None = None,
) -> None:
    records.append(
        {
            "feature_group": feature_group,
            "feature_name": feature_name,
            "method_id": method_id,
            "source_name": source_name or "",
            "rows": int(rows),
            "cols": int(cols),
            "output_path": str(output_path),
        }
    )


def write_shape_summary(records: list[dict[str, Any]], out_path: Path) -> Path:
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values(
            by=["source_name", "feature_group", "method_id", "feature_name"],
            na_position="last",
        ).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
