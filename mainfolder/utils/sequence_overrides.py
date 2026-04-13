from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path


OVERRIDE_COLUMNS = ["query_id", "resolved_id", "sequence", "source", "notes"]


@dataclass(frozen=True)
class SequenceOverride:
    query_id: str
    resolved_id: str = ""
    sequence: str = ""
    source: str = ""
    notes: str = ""


def ensure_sequence_override_csv(path: str | Path) -> Path:
    override_path = Path(path)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    if not override_path.exists():
        with override_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=OVERRIDE_COLUMNS)
            writer.writeheader()
    return override_path


def load_sequence_overrides(path: str | Path | None) -> dict[str, SequenceOverride]:
    if path is None:
        return {}
    override_path = ensure_sequence_override_csv(path)
    with override_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return {}
        missing = [col for col in OVERRIDE_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"{override_path} must have columns {', '.join(OVERRIDE_COLUMNS)}")

        overrides: dict[str, SequenceOverride] = {}
        for row in reader:
            query_id = str(row.get("query_id", "") or "").strip()
            if not query_id:
                continue
            overrides[query_id] = SequenceOverride(
                query_id=query_id,
                resolved_id=str(row.get("resolved_id", "") or "").strip(),
                sequence=str(row.get("sequence", "") or "").strip(),
                source=str(row.get("source", "") or "").strip(),
                notes=str(row.get("notes", "") or "").strip(),
            )
    return overrides


def sync_sequence_override_template(path: str | Path, query_ids: list[str] | tuple[str, ...] | set[str]) -> int:
    override_path = ensure_sequence_override_csv(path)
    rows: dict[str, dict[str, str]] = {}

    with override_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or OVERRIDE_COLUMNS
        if any(col not in fieldnames for col in OVERRIDE_COLUMNS):
            raise ValueError(f"{override_path} must have columns {', '.join(OVERRIDE_COLUMNS)}")
        for row in reader:
            query_id = str(row.get("query_id", "") or "").strip()
            if not query_id:
                continue
            rows[query_id] = {col: str(row.get(col, "") or "").strip() for col in OVERRIDE_COLUMNS}

    added = 0
    for query_id in sorted({str(x or "").strip() for x in query_ids if str(x or "").strip()}):
        if query_id in rows:
            continue
        rows[query_id] = {
            "query_id": query_id,
            "resolved_id": "",
            "sequence": "",
            "source": "",
            "notes": "",
        }
        added += 1

    with override_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OVERRIDE_COLUMNS)
        writer.writeheader()
        for query_id in sorted(rows):
            writer.writerow(rows[query_id])

    return added
