#!/usr/bin/env python3
"""Build question-only embeddings for the frozen full MedCalc benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from api_budget import record_failure, record_response, reserve_request_group


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments/data/sra_bench/medcalc_seed20260828"
OUTPUT = ROOT / "experiments/outputs/sra_medcalc_embeddings"
AUDIT_RESULTS = ROOT / "experiments/outputs/sra_medcalc/audit_results.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
            sort_keys=not compact,
        )
        if not compact:
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def audit_outcomes(path: Path, audit_ids: set[str]) -> int:
    if not path.exists():
        return 0
    return sum(
        row.get("valid") is True and str(row.get("instance_id")) in audit_ids
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    )


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    raw = usage.model_dump() if hasattr(usage, "model_dump") else vars(usage)
    prompt = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "total_tokens": int(raw.get("total_tokens") or prompt),
    }


def configure_budget(args: argparse.Namespace) -> None:
    os.environ["SKILLDELTA_API_BUDGET_STATE"] = str(args.budget_state.resolve())
    os.environ["SKILLDELTA_API_CALL_LEDGER"] = str(args.call_ledger.resolve())
    os.environ["SKILLDELTA_MAX_API_CALLS"] = str(args.max_api_calls)
    os.environ["SKILLDELTA_MAX_TOTAL_TOKENS"] = str(args.max_reported_tokens)
    os.environ["SKILLDELTA_MAX_QUOTA_DELTA"] = str(args.max_quota_delta)
    os.environ["SKILLDELTA_QUOTA_POLL_EVERY"] = str(args.quota_poll_every)
    os.environ["SKILLDELTA_QUOTA_QUERY_ATTEMPTS"] = "3"
    os.environ["SKILLDELTA_QUOTA_FAIL_CLOSED"] = "1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--audit-results", type=Path, default=AUDIT_RESULTS)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--max-api-calls", type=int, default=35)
    parser.add_argument("--max-reported-tokens", type=int, default=1_200_000)
    parser.add_argument("--max-input-characters", type=int, default=4_000_000)
    parser.add_argument("--max-quota-delta", type=int, default=15_000)
    parser.add_argument("--quota-poll-every", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--base-url-env", default="QWEN_TARGET_OPENAI_COMPATIBLE_BASE_URL")
    parser.add_argument("--api-key-env", default="QWEN_TARGET_OPENAI_COMPATIBLE_API_KEY")
    parser.add_argument("--budget-state", type=Path)
    parser.add_argument("--call-ledger", type=Path)
    args = parser.parse_args()

    data = args.data_root.resolve()
    output = args.output_dir.resolve()
    args.budget_state = (args.budget_state or output / "embedding_budget_state.json").resolve()
    args.call_ledger = (args.call_ledger or output / "embedding_api_usage.jsonl").resolve()
    manifest_path = data / "manifest.json"
    manifest = read_json(manifest_path)
    unhashed = dict(manifest)
    expected_content = unhashed.pop("manifest_content_sha256", None)
    if json_sha256(unhashed) != expected_content:
        raise SystemExit("invalid MedCalc manifest content hash")
    tasks: list[dict[str, str]] = []
    audit_ids: set[str] = set()
    for panel_name in ("support", "audit"):
        panel_path = data / f"{panel_name}.json"
        if sha256(panel_path) != manifest["panels"][panel_name]["sha256"]:
            raise SystemExit(f"{panel_name} panel SHA256 mismatch")
        for row in read_json(panel_path):
            task = {
                "instance_id": str(row["instance_id"]),
                "question": str(row["question"]),
                "skill_id": str(row["skill_annotations"][0]),
                "panel": panel_name,
            }
            tasks.append(task)
            if panel_name == "audit":
                audit_ids.add(task["instance_id"])
    if len(tasks) != 1100 or len({row["instance_id"] for row in tasks}) != 1100:
        raise SystemExit("expected all 1100 unique MedCalc tasks")
    visible = audit_outcomes(args.audit_results.resolve(), audit_ids)
    if visible:
        raise SystemExit(f"refusing to embed after observing {visible} valid audit outcomes")
    total_characters = sum(len(row["question"]) for row in tasks)
    batches = (len(tasks) + args.batch_size - 1) // args.batch_size
    if total_characters > args.max_input_characters or batches > args.max_api_calls:
        raise SystemExit("embedding plan exceeds character or call cap")

    vectors_path = output / "medcalc_question_embeddings.json"
    progress_path = output / "embedding_progress.json"
    embedding_manifest_path = output / "embedding_manifest.json"
    fingerprint = hashlib.sha256(
        json.dumps(
            [(row["instance_id"], row["question"]) for row in tasks],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    plan = {
        "mode": "live" if args.live else "dry_run",
        "model": args.model,
        "tasks": len(tasks),
        "support_tasks": 440,
        "audit_tasks": 660,
        "audit_outcomes_seen": visible,
        "input_characters": total_characters,
        "rough_tokens_chars_div_4": round(total_characters / 4),
        "batch_size": args.batch_size,
        "expected_batches": batches,
        "max_api_calls": args.max_api_calls,
        "max_reported_tokens": args.max_reported_tokens,
        "max_quota_delta": args.max_quota_delta,
        "manifest_sha256": sha256(manifest_path),
        "input_fingerprint": fingerprint,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.live:
        print("Dry run only: zero embedding API calls.")
        return

    base_url = os.environ.get(args.base_url_env, "").strip()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not base_url or not api_key:
        raise SystemExit(f"missing {args.base_url_env} or {args.api_key_env}")
    os.environ["TARGET_OPENAI_COMPATIBLE_BASE_URL"] = base_url
    os.environ["TARGET_OPENAI_COMPATIBLE_API_KEY"] = api_key
    configure_budget(args)
    progress = {
        "schema_version": 1,
        "model": args.model,
        "input_fingerprint": fingerprint,
        "manifest_sha256": sha256(manifest_path),
    }
    if progress_path.exists() and read_json(progress_path) != progress:
        raise SystemExit("embedding progress metadata mismatch")
    if not progress_path.exists():
        atomic_json(progress_path, progress)
    vectors: dict[str, list[float]] = read_json(vectors_path) if vectors_path.exists() else {}
    expected_ids = {row["instance_id"] for row in tasks}
    if set(vectors) - expected_ids:
        raise SystemExit("cached embeddings contain unexpected IDs")
    pending = [row for row in tasks if row["instance_id"] not in vectors]

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=180)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        error: Exception | None = None
        for _attempt in range(args.max_retries):
            reservation = reserve_request_group()
            try:
                response = client.embeddings.create(
                    model=args.model,
                    input=[row["question"] for row in batch],
                )
                returned = sorted(response.data, key=lambda item: int(item.index))
                if len(returned) != len(batch):
                    raise RuntimeError("embedding response length mismatch")
                for row, item in zip(batch, returned):
                    vectors[row["instance_id"]] = [float(value) for value in item.embedding]
                usage = usage_dict(response)
                record_response(
                    reservation,
                    usage,
                    stage="sra_medcalc_embedding",
                    role="embedding",
                    model=args.model,
                    returned_model=str(getattr(response, "model", "") or args.model),
                    empty=False,
                )
                atomic_json(vectors_path, vectors, compact=True)
                error = None
                print(
                    json.dumps(
                        {
                            "embedded": len(vectors),
                            "total": len(tasks),
                            "reported_tokens": usage.get("total_tokens", 0),
                        }
                    ),
                    flush=True,
                )
                break
            except Exception as exc:  # noqa: BLE001
                record_failure(
                    reservation,
                    stage="sra_medcalc_embedding",
                    role="embedding",
                    model=args.model,
                    error_type=type(exc).__name__,
                )
                error = exc
        if error is not None:
            raise error
    if set(vectors) != expected_ids or {len(value) for value in vectors.values()} != {1536}:
        raise SystemExit("embedding artifact is incomplete or wrong-dimensional")
    embedding_manifest = {
        "schema_version": 1,
        "status": "audit outcomes sealed",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": args.model,
        "dimension": 1536,
        "input": "full question text only",
        "tasks": len(tasks),
        "support_tasks": 440,
        "audit_tasks": 660,
        "audit_outcomes_read": visible,
        "manifest_sha256": sha256(manifest_path),
        "input_fingerprint": fingerprint,
        "vectors_path": str(vectors_path.relative_to(ROOT)),
        "vectors_sha256": sha256(vectors_path),
        "budget_state_sha256": sha256(args.budget_state),
    }
    atomic_json(embedding_manifest_path, embedding_manifest)
    print(json.dumps(embedding_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
