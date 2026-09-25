#!/usr/bin/env python3
"""Process-safe API response ledger and conservative per-run budget gates.

The compatible backend returns gateway-reported token usage after each
successful response.  This module persists those counts across ALFWorld's
spawned workers and prevents new request groups after either configured cap
has been reached.  It never records prompts, responses, or credentials.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class ApiBudgetExceeded(RuntimeError):
    """Raised before an API request when a configured cap is exhausted."""


def _positive_env(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return 0
    value = int(raw)
    return max(0, value)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _paths() -> tuple[Path | None, Path | None]:
    state_raw = os.environ.get("SKILLDELTA_API_BUDGET_STATE", "").strip()
    ledger_raw = os.environ.get("SKILLDELTA_API_CALL_LEDGER", "").strip()
    return (Path(state_raw) if state_raw else None, Path(ledger_raw) if ledger_raw else None)


def _query_proxy_quota_used(attempts: int = 4) -> int:
    key = (
        os.environ.get("TARGET_OPENAI_COMPATIBLE_API_KEY", "").strip()
        or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "").strip()
    )
    base_url = (
        os.environ.get("TARGET_OPENAI_COMPATIBLE_BASE_URL", "").strip()
        or os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "").strip()
    )
    if not key or not base_url:
        raise RuntimeError(
            "quota-delta cap requires compatible API key and base URL environment variables"
        )
    parsed = urlsplit(base_url)
    endpoint = f"{parsed.scheme}://{parsed.netloc}/api/usage/token/"
    request = urllib.request.Request(
        endpoint,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    payload = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    if payload is None:  # pragma: no cover - loop either succeeds or raises
        raise RuntimeError("quota query returned no payload")
    data = payload.get("data", payload)
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict) or "total_used" not in data:
        raise RuntimeError("unexpected quota response")
    return int(data["total_used"])


def _locked_json_update(path: Path, update) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        state = json.loads(raw) if raw else {}
        result = update(state)
        handle.seek(0)
        handle.truncate()
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return result


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def reserve_request_group() -> int | None:
    """Atomically reserve one top-level backend request group.

    SkillOpt may retry transport failures internally, so this is a strict cap
    on top-level model requests, not necessarily on raw HTTP attempts. Failed
    transport attempts normally have no reported token usage.
    """
    state_path, ledger_path = _paths()
    if state_path is None and ledger_path is None:
        return None

    max_calls = _positive_env("SKILLDELTA_MAX_API_CALLS")
    max_tokens = _positive_env("SKILLDELTA_MAX_TOTAL_TOKENS")
    max_quota_delta = _positive_env("SKILLDELTA_MAX_QUOTA_DELTA")
    max_public_cost = _positive_env("SKILLDELTA_MAX_PUBLIC_COST_NANODOLLARS")
    supplied_quota_baseline = _positive_env("SKILLDELTA_PROXY_QUOTA_BASELINE")
    quota_poll_every = _positive_env("SKILLDELTA_QUOTA_POLL_EVERY") or 25
    quota_query_attempts = _positive_env("SKILLDELTA_QUOTA_QUERY_ATTEMPTS") or 1
    quota_fail_closed = _truthy_env("SKILLDELTA_QUOTA_FAIL_CLOSED")
    if state_path is None:
        raise RuntimeError(
            "SKILLDELTA_API_CALL_LEDGER requires SKILLDELTA_API_BUDGET_STATE"
        )

    def update(state: dict[str, Any]) -> int | str:
        if state.get("blocked_reason"):
            return str(state["blocked_reason"])
        reserved = int(state.get("reserved_request_groups", 0))
        reported = int(state.get("reported_total_tokens", 0))
        public_cost = int(state.get("estimated_public_cost_nanodollars", 0))
        initialized_from_supplied_baseline = False
        if (
            max_quota_delta
            and "proxy_quota_baseline" not in state
            and supplied_quota_baseline
        ):
            state["proxy_quota_baseline"] = supplied_quota_baseline
            state["proxy_quota_current"] = supplied_quota_baseline
            state["proxy_quota_delta"] = 0
            state["proxy_quota_baseline_source"] = "authenticated_recent_snapshot"
            initialized_from_supplied_baseline = True
        if max_quota_delta and (
            "proxy_quota_baseline" not in state
            or (
                not initialized_from_supplied_baseline
                and reserved % quota_poll_every == 0
            )
        ):
            try:
                current_quota = _query_proxy_quota_used(quota_query_attempts)
            except Exception as exc:  # noqa: BLE001
                state["quota_query_failures"] = int(
                    state.get("quota_query_failures", 0)
                ) + 1
                state["last_quota_query_error"] = type(exc).__name__
                state["last_quota_query_error_at"] = (
                    datetime.now().astimezone().isoformat()
                )
                if "proxy_quota_baseline" not in state or quota_fail_closed:
                    state["blocked_reason"] = (
                        "quota query unavailable while fail-closed: "
                        f"{type(exc).__name__}"
                    )
                    return str(state["blocked_reason"])
            else:
                state.setdefault("proxy_quota_baseline", current_quota)
                state["proxy_quota_current"] = current_quota
                state["proxy_quota_delta"] = (
                    current_quota - int(state["proxy_quota_baseline"])
                )
                state.pop("last_quota_query_error", None)
                state.pop("last_quota_query_error_at", None)
        quota_delta = int(state.get("proxy_quota_delta", 0))
        if max_calls and reserved >= max_calls:
            state["blocked_reason"] = (
                f"API request cap reached: {reserved}/{max_calls} request groups"
            )
            return str(state["blocked_reason"])
        if max_tokens and reported >= max_tokens:
            state["blocked_reason"] = (
                f"API token cap reached: {reported}/{max_tokens} reported tokens"
            )
            return str(state["blocked_reason"])
        if max_quota_delta and quota_delta >= max_quota_delta:
            state["blocked_reason"] = (
                f"proxy quota cap reached: {quota_delta}/{max_quota_delta} units"
            )
            return str(state["blocked_reason"])
        if max_public_cost and public_cost >= max_public_cost:
            state["blocked_reason"] = (
                "public-price cost cap reached: "
                f"{public_cost}/{max_public_cost} nanodollars"
            )
            return str(state["blocked_reason"])
        reserved += 1
        state["reserved_request_groups"] = reserved
        state.setdefault("completed_responses", 0)
        state.setdefault("failed_request_groups", 0)
        state.setdefault("reported_prompt_tokens", 0)
        state.setdefault("reported_completion_tokens", 0)
        state.setdefault("reported_total_tokens", 0)
        state["max_request_groups"] = max_calls
        state["max_total_tokens"] = max_tokens
        state["max_proxy_quota_delta"] = max_quota_delta
        state["max_public_cost_nanodollars"] = max_public_cost
        state.setdefault("estimated_public_cost_nanodollars", 0)
        state["quota_poll_every_request_groups"] = quota_poll_every
        return reserved

    result = _locked_json_update(state_path, update)
    if isinstance(result, str):
        raise ApiBudgetExceeded(result)
    return int(result)


def record_response(
    reservation: int | None,
    usage: dict[str, Any] | None,
    *,
    stage: str,
    role: str,
    model: str,
    returned_model: str | None = None,
    empty: bool,
) -> None:
    state_path, ledger_path = _paths()
    if reservation is None or state_path is None:
        return
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    input_price = _positive_env("SKILLDELTA_INPUT_PRICE_NANODOLLARS_PER_TOKEN")
    output_price = _positive_env("SKILLDELTA_OUTPUT_PRICE_NANODOLLARS_PER_TOKEN")
    priced_cost = prompt * input_price + completion * output_price

    def update(state: dict[str, Any]) -> None:
        state["completed_responses"] = int(state.get("completed_responses", 0)) + 1
        state["reported_prompt_tokens"] = int(state.get("reported_prompt_tokens", 0)) + prompt
        state["reported_completion_tokens"] = int(
            state.get("reported_completion_tokens", 0)
        ) + completion
        state["reported_total_tokens"] = int(state.get("reported_total_tokens", 0)) + total
        state["estimated_public_cost_nanodollars"] = int(
            state.get("estimated_public_cost_nanodollars", 0)
        ) + priced_cost

    _locked_json_update(state_path, update)
    if ledger_path is not None:
        _append_jsonl(
            ledger_path,
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "pid": os.getpid(),
                "reservation": reservation,
                "stage": stage,
                "role": role,
                "model": model,
                "requested_model": model,
                "returned_model": returned_model,
                "status": "empty" if empty else "ok",
                "usage_source": "gateway_response",
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "estimated_public_cost_nanodollars": priced_cost,
            },
        )


def record_failure(
    reservation: int | None,
    *,
    stage: str,
    role: str,
    model: str,
    error_type: str,
) -> None:
    state_path, ledger_path = _paths()
    if reservation is None or state_path is None:
        return

    def update(state: dict[str, Any]) -> None:
        state["failed_request_groups"] = int(state.get("failed_request_groups", 0)) + 1

    _locked_json_update(state_path, update)
    if ledger_path is not None:
        _append_jsonl(
            ledger_path,
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "pid": os.getpid(),
                "reservation": reservation,
                "stage": stage,
                "role": role,
                "model": model,
                "status": "failed",
                "error_type": error_type,
            },
        )
