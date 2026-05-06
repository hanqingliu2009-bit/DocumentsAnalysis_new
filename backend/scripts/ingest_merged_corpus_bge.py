#!/usr/bin/env python3
"""
Ingest ``files/merged_splite_corpus.json`` into a dedicated Chroma collection using
``sentence-transformers`` + ``BAAI/bge-large-zh-v1.5`` (1024-dim).

This script sets ``EMBEDDING_BACKEND=local`` (and model / dimension) in ``os.environ``
**before** importing ``config``, so it does not depend on Ark embedding settings in ``.env``.

Usage (from repo root, with backend venv):

  backend\\venv\\Scripts\\python.exe backend\\scripts\\ingest_merged_corpus_bge.py --recreate

Optional:

  --corpus files/merged_splite_corpus.json
  --collection splite_bge_zh_v15
  --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Resolve imports: treat ``backend/`` as package root for ``config``, ``core``, etc.
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"
_DEFAULT_DIM = 1024  # bge-large-zh-v1.5 output size
_DEFAULT_COLLECTION = "splite_bge_zh_v15"


def _chunk_text(group_title: str, content: str) -> str:
    title = (group_title or "").strip()
    body = (content or "").strip()
    if title and body:
        return f"【{title}】\n{body}"
    return body or title


def _stable_chunk_id(document_id: str, chunk_id: Any) -> str:
    return f"{document_id}:{chunk_id}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        type=Path,
        default=_REPO_ROOT / "files" / "merged_splite_corpus.json",
        help="Path to merged_splite_corpus.json",
    )
    p.add_argument(
        "--collection",
        type=str,
        default=_DEFAULT_COLLECTION,
        help="Chroma collection name (use a dedicated name; do not mix dimensions).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_MODEL,
        help="sentence-transformers model id (default: BAAI/bge-large-zh-v1.5)",
    )
    p.add_argument(
        "--embedding-dimension",
        type=int,
        default=_DEFAULT_DIM,
        help="Expected embedding size (bge-large-zh-v1.5 = 1024)",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Texts per embed batch")
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the target collection if it exists, then re-ingest from scratch.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ``config`` uses paths like ``./data`` relative to process cwd; uvicorn is typically
    # started from ``backend/``. Align the ingest script so Chroma lives under ``backend/data/``.
    os.chdir(_BACKEND_DIR)

    corpus_path = (args.corpus if args.corpus.is_absolute() else _REPO_ROOT / args.corpus).resolve()

    # Must be set before ``from config import settings`` (first Settings() load).
    os.environ["EMBEDDING_BACKEND"] = "local"
    os.environ["EMBEDDING_MODEL"] = str(args.model).strip()
    os.environ["EMBEDDING_DIMENSION"] = str(int(args.embedding_dimension))

    from config import settings
    from core.document import DocumentChunk
    from storage.vector_store import EmbeddingGenerator, VectorStore

    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from tqdm import tqdm

    if not corpus_path.is_file():
        print(f"Corpus not found: {corpus_path}", file=sys.stderr)
        return 1

    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    sources = raw.get("sources")
    if not isinstance(sources, list):
        print("Invalid corpus: missing sources[]", file=sys.stderr)
        return 1

    if args.recreate:
        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(
            path=str(settings.VECTOR_DB_PATH),
            settings=chroma_settings,
        )
        try:
            client.delete_collection(name=args.collection)
            print(f"Deleted collection {args.collection!r}")
        except Exception as e:
            print(f"(no existing collection or delete skipped: {e})")

    embedder = EmbeddingGenerator(model_name=args.model.strip())
    store = VectorStore(collection_name=args.collection)

    # Sanity-check one vector vs configured dimension
    probe = embedder.embed_text("probe")
    if len(probe) != int(args.embedding_dimension):
        print(
            f"Embedding dim mismatch: got {len(probe)}, expected {args.embedding_dimension}. "
            f"Adjust --embedding-dimension or model id.",
            file=sys.stderr,
        )
        return 1

    flat_chunks: List[Dict[str, Any]] = []
    for src in sources:
        if not isinstance(src, dict):
            continue
        doc_id = str(src.get("document_id") or "")
        doc_name = str(src.get("document_name") or "")
        src_file = str(src.get("file") or "")
        for ch in src.get("chunks") or []:
            if not isinstance(ch, dict):
                continue
            flat_chunks.append(
                {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "source_file": src_file,
                    "chunk_id": ch.get("chunk_id"),
                    "group_title": str(ch.get("group_title") or ""),
                    "content": str(ch.get("content") or ""),
                }
            )

    if not flat_chunks:
        print("No chunks to ingest.", file=sys.stderr)
        return 1

    print(
        f"Ingesting {len(flat_chunks)} chunks into collection={args.collection!r} "
        f"model={args.model!r} dim={args.embedding_dimension}"
    )

    batch = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(flat_chunks), batch), desc="batches"):
        slice_ = flat_chunks[i : i + batch]
        texts = [_chunk_text(x["group_title"], x["content"]) for x in slice_]
        vectors = embedder.embed_texts(texts)

        doc_chunks: List[DocumentChunk] = []
        for row, emb, text in zip(slice_, vectors, texts):
            cid = _stable_chunk_id(row["document_id"], row["chunk_id"])
            try:
                idx = int(row["chunk_id"])
            except (TypeError, ValueError):
                idx = 0
            doc_chunks.append(
                DocumentChunk(
                    id=cid,
                    document_id=row["document_id"],
                    text=text,
                    chunk_index=idx,
                    start_char=0,
                    end_char=len(text),
                    embedding=list(emb),
                    metadata={
                        "document_name": row["document_name"][:500],
                        "source_file": row["source_file"][:500],
                        "group_title": row["group_title"][:500],
                        "split_chunk_id": str(row["chunk_id"]),
                    },
                )
            )
        store.add_chunks(doc_chunks)

    n = store.get_chunk_count()
    print(f"Done. Collection {args.collection!r} now has {n} vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
