#!/usr/bin/env python3
"""
Merge multiple splite_plan / splite JSON files under files/ into one corpus JSON.

Output schema matches doc/方案-双路检索-BGE本地向量与远端图库.md §3.2.

Usage (from repo root):
  backend/venv/Scripts/python.exe backend/scripts/merge_splite_json.py
  backend/venv/Scripts/python.exe backend/scripts/merge_splite_json.py --input files --output files/merged_splite_corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_source(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: root must be a JSON object")

    required = ("document_id", "document_name", "chunks")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path.name}: missing keys: {missing}")

    if not isinstance(data["chunks"], list):
        raise ValueError(f"{path.name}: chunks must be a list")

    total_pages = data.get("total_pages")
    if total_pages is not None and not isinstance(total_pages, int):
        raise ValueError(f"{path.name}: total_pages must be int or omitted")

    return {
        "file": path.name,
        "document_id": data["document_id"],
        "document_name": data["document_name"],
        "total_pages": total_pages,
        "chunks": data["chunks"],
    }


def merge(input_dir: Path, output_path: Path, exclude_names: frozenset[str]) -> Dict[str, Any]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    json_paths: List[Path] = sorted(
        p for p in input_dir.glob("*.json") if p.is_file() and p.name not in exclude_names
    )
    if not json_paths:
        raise RuntimeError(f"No *.json files found under {input_dir}")

    sources: List[Dict[str, Any]] = []
    for p in json_paths:
        sources.append(_load_source(p))

    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_count": len(sources),
        "chunk_count": sum(len(s["chunks"]) for s in sources),
        "sources": sources,
    }


def main() -> int:
    root = _repo_root()
    default_input = root / "files"
    default_output = root / "files" / "merged_splite_corpus.json"
    default_exclude = frozenset({default_output.name})

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Directory containing *.json splite files (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Merged JSON output path (default: {default_output})",
    )
    args = parser.parse_args()

    input_dir = args.input if args.input.is_absolute() else root / args.input
    output_path = args.output if args.output.is_absolute() else root / args.output

    merged = merge(input_dir, output_path, exclude_names=default_exclude)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {output_path} "
        f"({merged['source_count']} documents, {merged['chunk_count']} chunks)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
