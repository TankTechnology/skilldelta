#!/usr/bin/env python3
"""Small, dependency-light SkillDelta router for the dsh plugin.

The support index is frozen and scoped by the caller before deployment.
Routing uniformly averages paired signed gains ``d = y_on - y_off`` of
cosine neighbors in that index. New task
embeddings use an OpenAI-compatible ``/embeddings`` endpoint; sealed replay
audits can pass ``--task-id`` with a separate query-only index so held-out
outcomes never enter the support labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_support(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = _read_json(path)
    if data.get("schema") not in (None, "skilldelta-support-v1"):
        raise ValueError(f"unsupported support schema: {data.get('schema')!r}")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("support index has no tasks")
    clean: list[dict[str, Any]] = []
    ids: set[str] = set()
    dim: int | None = None
    for row in tasks:
        if not isinstance(row, dict) or not row.get("id"):
            raise ValueError("every support task needs an id")
        vector = row.get("embedding")
        gain = row.get("d")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"task {row.get('id')} has no embedding")
        if (not isinstance(gain, (int, float)) or not math.isfinite(float(gain))
                or not -1 <= gain <= 1):
            raise ValueError(f"task {row.get('id')} has invalid d")
        task_id = str(row["id"])
        if task_id in ids:
            raise ValueError(f"duplicate support task id: {task_id}")
        ids.add(task_id)
        if dim is None:
            dim = len(vector)
        if len(vector) != dim:
            raise ValueError("support embeddings have inconsistent dimensions")
        clean.append({"id": str(row["id"]), "text": str(row.get("text", "")),
                      "embedding": [float(x) for x in vector], "d": float(gain),
                      **({"family": row["family"]} if "family" in row else {})})
    return data, clean


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("query embedding has zero or non-finite norm")
    return [x / norm for x in vector]


def _query_from_index(path: str, task_id: str) -> list[float] | None:
    """Read a sealed query-only embedding index without importing outcomes."""
    data = _read_json(path)
    if isinstance(data.get("embeddings"), dict):
        value = data["embeddings"].get(task_id)
    else:
        value = None
        for row in data.get("tasks", []):
            if isinstance(row, dict) and str(row.get("id")) == task_id:
                value = row.get("embedding")
                break
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"query index entry {task_id!r} has no embedding")
    return [float(x) for x in value]


def _embed_remote(text: str) -> list[float]:
    base = os.environ.get("SKILLDELTA_EMBEDDING_BASE_URL", "").rstrip("/")
    key = os.environ.get("SKILLDELTA_EMBEDDING_API_KEY", "")
    model = os.environ.get("SKILLDELTA_EMBEDDING_MODEL", "text-embedding-3-small")
    if not base or not key:
        raise RuntimeError("embedding service is not configured (set SKILLDELTA_EMBEDDING_BASE_URL/API_KEY)")
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    request = urllib.request.Request(
        base + "/embeddings", data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.environ.get("SKILLDELTA_EMBEDDING_TIMEOUT", "30"))) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"embedding request failed: {exc}") from exc
    try:
        vector = body["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("embedding response did not contain data[0].embedding") from exc
    if not isinstance(vector, list):
        raise RuntimeError("embedding response vector is not a list")
    return [float(x) for x in vector]


def route(args: argparse.Namespace) -> dict[str, Any]:
    if args.k < 1:
        raise ValueError("k must be positive")
    if not math.isfinite(args.threshold):
        raise ValueError("threshold must be finite")
    metadata, tasks = _load_support(args.support)
    by_id = {row["id"]: row for row in tasks}
    query_id = str(args.task_id) if args.task_id else None
    if args.query_embedding:
        query = [float(x) for x in json.loads(args.query_embedding)]
        query_source = "argument"
    elif query_id and query_id in by_id:
        query = by_id[query_id]["embedding"]
        query_source = "frozen-index"
    elif query_id and args.query_index:
        query = _query_from_index(args.query_index, query_id)
        if query is not None:
            query_source = "frozen-query-index"
        else:
            query = None
    else:
        query = None
    if query is None:
        if not args.task_text:
            raise ValueError("task-text is required for an unseen task")
        query = _embed_remote(args.task_text)
        query_source = "embedding-api"
    if len(query) != len(tasks[0]["embedding"]):
        raise ValueError("query embedding dimension does not match support index")
    query = _unit(query)
    scored = []
    for row in tasks:
        # Never let the query task contribute its own outcome to the prediction.
        if query_id is not None and row["id"] == query_id:
            continue
        vector = _unit(row["embedding"])
        scored.append((sum(a * b for a, b in zip(query, vector)), row))
    if not scored:
        raise ValueError("support index has no leave-one-out neighbors")
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    k = max(1, min(int(args.k), len(scored)))
    neighbors = [{"id": row["id"], "similarity": float(sim), "d": row["d"]}
                 for sim, row in scored[:k]]
    predicted = sum(item["d"] for item in neighbors) / k
    threshold = float(args.threshold)
    return {
        "schema": "skilldelta-route-v1",
        "task_id": query_id,
        "task_text": args.task_text or by_id.get(query_id, {}).get("text", ""),
        "query_source": query_source,
        "embedding_model": metadata.get("embedding_model", "unknown"),
        "support_count": len(scored),
        "k": k,
        "threshold": threshold,
        "predicted_gain": predicted,
        "enable_skill": predicted > threshold,
        "neighbors": neighbors,
    }


def status(args: argparse.Namespace) -> dict[str, Any]:
    metadata, tasks = _load_support(args.support)
    skill_hash = None
    if args.skill:
        digest = hashlib.sha256(pathlib.Path(args.skill).read_bytes()).hexdigest()
        skill_hash = {"path": str(pathlib.Path(args.skill).resolve()), "sha256": digest}
    return {"schema": "skilldelta-status-v1", "support": {
        "path": str(pathlib.Path(args.support).resolve()), "count": len(tasks),
        "embedding_model": metadata.get("embedding_model", "unknown"),
        "sha256": hashlib.sha256(pathlib.Path(args.support).read_bytes()).hexdigest(),
    }, "skill": skill_hash}


def record(args: argparse.Namespace) -> dict[str, Any]:
    row = {
        "schema": "skilldelta-outcome-v1",
        "time": datetime.now(timezone.utc).isoformat(),
        "task_id": str(args.task_id),
        "skill_enabled": bool(args.skill_enabled),
    }
    if args.success is not None:
        row["success"] = bool(args.success)
    if args.tokens is not None:
        row["tokens"] = int(args.tokens)
    if args.route_id:
        row["route_id"] = args.route_id
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"written": str(path.resolve()), "row": row}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--support", required=True)
    common.add_argument("--skill")
    r = sub.add_parser("route", parents=[common])
    r.add_argument("--task-text", default="")
    r.add_argument("--task-id")
    r.add_argument("--query-index", help="Optional query-only embedding index keyed by task id")
    r.add_argument("--query-embedding")
    r.add_argument("--threshold", type=float, default=0.0)
    r.add_argument("--k", type=int, default=10)
    s = sub.add_parser("status", parents=[common])
    o = sub.add_parser("record")
    o.add_argument("--output", required=True)
    o.add_argument("--task-id", required=True)
    o.add_argument("--skill-enabled", type=int, choices=[0, 1], required=True)
    o.add_argument("--success", type=int, choices=[0, 1])
    o.add_argument("--tokens", type=int)
    o.add_argument("--route-id")
    return p


def main() -> int:
    try:
        args = parser().parse_args()
        if args.action == "route":
            result = route(args)
        elif args.action == "status":
            result = status(args)
        else:
            result = record(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
