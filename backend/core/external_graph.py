"""HTTP client for supplier-hosted graph RAG (no local embedding / Chroma retrieval)."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

import httpx

import config

logger = logging.getLogger(__name__)


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (dict, list)):
                out.append(json.dumps(item, ensure_ascii=False))
            else:
                s = str(item).strip()
                if s:
                    out.append(s)
        return out
    return [str(value).strip()] if str(value).strip() else []


def _normalize_message(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"text": raw}
        except json.JSONDecodeError:
            return {"text": raw}
    return {}


def fetch_graph_context(query: str) -> Tuple[str, List[dict], Dict[str, Any]]:
    """
    POST JSON {query, domain, search_type} to the supplier graph API.

    Returns:
        (context_text_for_llm, sources_for_api_response, raw_body_for_debug)
    """
    cfg = config.settings
    url = (cfg.EXTERNAL_GRAPH_API_URL or "").strip()
    if not url:
        raise ValueError("EXTERNAL_GRAPH_API_URL is empty")

    payload = {
        "query": query.strip(),
        "domain": cfg.EXTERNAL_GRAPH_DOMAIN,
        "search_type": cfg.EXTERNAL_GRAPH_SEARCH_TYPE,
    }
    timeout = float(cfg.EXTERNAL_GRAPH_TIMEOUT or 60.0)

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        body: Dict[str, Any] = response.json()

    if not body.get("success", True):
        msg = body.get("message", "unknown error")
        raise RuntimeError(f"Graph API reported failure: {msg}")

    message = _normalize_message(body.get("message"))
    triplets = _as_str_list(message.get("triplet"))
    summaries = _as_str_list(message.get("summary"))
    chunks = _as_str_list(message.get("chunk"))

    parts: List[str] = []
    if triplets:
        lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(triplets))
        parts.append(f"## 图数据库三元组\n{lines}")
    if summaries:
        lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(summaries))
        parts.append(f"## 摘要\n{lines}")
    if chunks:
        lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(chunks))
        parts.append(f"## 文本块\n{lines}")

    context = "\n\n".join(parts).strip()
    meta: Dict[str, Any] = {
        "triplet_count": len(triplets),
        "summary_count": len(summaries),
        "chunk_count": len(chunks),
    }

    sources: List[dict] = []
    for i, text in enumerate(triplets):
        sources.append({
            "chunk_id": f"graph-triplet-{i + 1}",
            "score": 1.0,
            "text": text[:500] + "..." if len(text) > 500 else text,
            "document_id": None,
            "document_title": "图数据库·三元组",
        })
    for i, text in enumerate(summaries):
        sources.append({
            "chunk_id": f"graph-summary-{i + 1}",
            "score": 1.0,
            "text": text[:500] + "..." if len(text) > 500 else text,
            "document_id": None,
            "document_title": "图数据库·摘要",
        })
    for i, text in enumerate(chunks):
        sources.append({
            "chunk_id": f"graph-chunk-{i + 1}",
            "score": 1.0,
            "text": text[:500] + "..." if len(text) > 500 else text,
            "document_id": None,
            "document_title": "图数据库·文本块",
        })

    return context, sources, {"response": body, "meta": meta}


def build_graph_system_prompt_no_hits() -> str:
    return (
        "你是通用助手。本轮「图数据库检索」没有返回可用的三元组、摘要或文本块；"
        "不要编造具体设备条款或内部数据。回答语言与用户问题一致。"
    )
