#!/usr/bin/env python3
"""SkillDelta eval wrapper.

Monkey-patches the SkillOpt config-loader bug at pinned commit 0389ace
(structured `env:` sections are popped by _resolve_layer_format_duplicates,
because _FLATTEN_MAP maps "env.name" -> "env" and the dedup logic pops the
dict-valued section itself). Keeps the submodule working tree clean.

Usage: see docs/data-generation.md for the external SkillOpt checkout,
local benchmark inputs, and environment-only credentials.
"""
import os
import sys


def _configure_api_budget_paths() -> None:
    """Enable durable per-response usage logging for every wrapper run."""
    if "--out_root" not in sys.argv:
        return
    idx = sys.argv.index("--out_root")
    if idx + 1 >= len(sys.argv):
        return
    out_root = os.path.abspath(sys.argv[idx + 1])
    os.environ.setdefault(
        "SKILLDELTA_API_CALL_LEDGER", os.path.join(out_root, "api_usage.jsonl")
    )
    os.environ.setdefault(
        "SKILLDELTA_API_BUDGET_STATE", os.path.join(out_root, "api_budget_state.json")
    )
    # Conservative defaults. A larger run must opt in explicitly after its
    # maximum calls/tokens and account quota have been reviewed.
    os.environ.setdefault("SKILLDELTA_MAX_API_CALLS", "500")
    os.environ.setdefault("SKILLDELTA_MAX_TOTAL_TOKENS", "1000000")


_configure_api_budget_paths()


def _align_compatible_target_model() -> None:
    """Make the wrapper's target-model environment variable authoritative.

    Upstream ``eval_only.py`` loads ``model.target`` from YAML and then calls
    ``set_target_deployment`` with that value.  Merely exporting
    ``TARGET_OPENAI_COMPATIBLE_MODEL`` therefore does not select a different
    model.  Our cross-model runners use that environment variable to select
    both the provider account and model, so translate it into the explicit
    CLI override that upstream actually honors.  Reject contradictory
    explicit values instead of silently mislabelling a run.
    """
    env_model = os.environ.get("TARGET_OPENAI_COMPATIBLE_MODEL", "").strip()
    if not env_model:
        return
    if "--target_model" in sys.argv:
        idx = sys.argv.index("--target_model")
        if idx + 1 >= len(sys.argv):
            return  # argparse will report the malformed argument
        cli_model = sys.argv[idx + 1].strip()
        if cli_model != env_model:
            raise RuntimeError(
                "Target-model mismatch: --target_model="
                f"{cli_model!r} but TARGET_OPENAI_COMPATIBLE_MODEL={env_model!r}"
            )
        return
    sys.argv.extend(["--target_model", env_model])


_align_compatible_target_model()

_SKILLOPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "SkillOpt")
sys.path.insert(0, _SKILLOPT)

import skillopt.config as sc  # noqa: E402


def _patched_resolve_layer_format_duplicates(cfg: dict) -> None:
    for dotted, flat_key in sc._FLATTEN_MAP.items():
        if sc._nested_key_present(cfg, dotted) and not isinstance(cfg.get(flat_key), dict):
            cfg.pop(flat_key, None)


sc._resolve_layer_format_duplicates = _patched_resolve_layer_format_duplicates


def _patch_azure_bridge_for_compatible():
    """SpreadsheetBench's codegen_agent imports the Azure client directly
    (bypassing the backend dispatcher). Bridge its symbols to the
    openai_compatible backend so non-Azure targets work."""
    import os

    from skillopt.model import azure_openai as az
    from skillopt.model.openai_compatible_backend import _get_client

    az.get_target_client = lambda: _get_client("target")
    az.get_reasoning_effort = lambda: None  # compatible backend does not forward it
    az._needs_responses_api = lambda deployment: False
    az.TARGET_DEPLOYMENT = os.environ.get(
        "TARGET_OPENAI_COMPATIBLE_MODEL", os.environ.get("TARGET_DEPLOYMENT", ""))

    class _NoopTracker:
        def record(self, *args, **kwargs):
            pass

    az.tracker = _NoopTracker()


_patch_azure_bridge_for_compatible()


def _patch_spreadsheet_budget_logging():
    """Log SpreadsheetBench's direct SDK calls in the durable API ledger.

    The SpreadsheetBench codegen path calls ``client.chat.completions.create``
    directly rather than going through SkillOpt's generic compatible backend.
    Without this shim those calls would be invisible to the per-run budget and
    cost accounting.  We reserve one request group for each logical rollout
    call (the upstream helper owns transport retries) and record the gateway
    usage returned by the completed response.
    """
    try:
        from api_budget import record_failure, record_response, reserve_request_group
        from skillopt.envs.spreadsheetbench import codegen_agent
    except ImportError:
        return

    original = codegen_agent._llm_call_with_retry
    if getattr(original, "_skilldelta_budget_wrapped", False):
        return

    def _usage_dict(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        def _get(name):
            if isinstance(usage, dict):
                return usage.get(name)
            return getattr(usage, name, None)
        prompt = _get("prompt_tokens") or _get("input_tokens") or 0
        completion = _get("completion_tokens") or _get("output_tokens") or 0
        total = _get("total_tokens") or (prompt + completion)
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": total}

    def wrapped(call_fn, *, retries=5, timeout=120):
        # The pinned SpreadsheetBench helper defaults to five 120-second
        # retries.  Under a shared OpenAI-compatible gateway this can leave
        # dozens of worker futures blocked for ten minutes after one transient
        # connection stall.  Keep the normal timeout, but make the retry
        # policy explicit and bounded for cross-model replay; callers can
        # still opt into the upstream defaults when needed.
        retries = max(0, int(os.environ.get("SKILLDELTA_SPREADSHEET_RETRIES", "1")))
        timeout = max(10, int(os.environ.get("SKILLDELTA_SPREADSHEET_LLM_TIMEOUT", timeout)))
        reservation = reserve_request_group()
        model = os.environ.get("TARGET_OPENAI_COMPATIBLE_MODEL", "")
        try:
            response = original(call_fn, retries=retries, timeout=timeout)
        except Exception as exc:
            record_failure(
                reservation,
                stage="spreadsheet_rollout",
                role="target",
                model=model,
                error_type=type(exc).__name__,
            )
            raise
        returned_model = getattr(response, "model", None)
        choices = getattr(response, "choices", None) or []
        empty = not choices
        if choices:
            message = getattr(choices[0], "message", None)
            empty = not bool(getattr(message, "content", "") or "")
        record_response(
            reservation,
            _usage_dict(response),
            stage="spreadsheet_rollout",
            role="target",
            model=model,
            returned_model=str(returned_model) if returned_model else None,
            empty=empty,
        )
        return response

    wrapped._skilldelta_budget_wrapped = True
    codegen_agent._llm_call_with_retry = wrapped


_patch_spreadsheet_budget_logging()


def _patch_spreadsheet_limit_bridge():
    """Expose ``env.limit`` without changing the pinned SkillOpt submodule.

    The SpreadsheetBench adapter's constructor predates the common dataloader
    limit option.  The wrapper keeps the upstream constructor untouched while
    forwarding the option after construction; a synthetic signature preserves
    eval_only's config-to-constructor filtering.
    """
    try:
        import inspect
        from skillopt.envs.spreadsheetbench import adapter as spreadsheet_adapter
    except ImportError:
        return
    cls = spreadsheet_adapter.SpreadsheetBenchAdapter
    original = cls.__init__
    if getattr(original, "_skilldelta_limit_wrapped", False):
        return

    def wrapped(self, *args, limit=0, **kwargs):
        original(self, *args, **kwargs)
        self.dataloader.limit = int(limit or 0)

    wrapped._skilldelta_limit_wrapped = True
    signature = inspect.signature(original)
    limit_parameter = inspect.Parameter(
        "limit", inspect.Parameter.KEYWORD_ONLY, default=0, annotation=int
    )
    wrapped.__signature__ = signature.replace(
        parameters=[*signature.parameters.values(), limit_parameter]
    )
    cls.__init__ = wrapped


_patch_spreadsheet_limit_bridge()


def _patch_spreadsheet_durable_codegen_resume():
    """Checkpoint SpreadsheetBench codegen results one task at a time.

    Upstream writes ``results.jsonl`` only after an entire split returns.  A
    transport interruption therefore loses the split-level checkpoint even
    though completed task artifacts are already on disk.  The canonical GLM
    replay opts into this durable path so each completed model execution is
    durable and only transport-level failures remain pending on resume.

    This patch is deliberately opt-in.  Existing benchmark runs keep the
    pinned upstream behavior unless
    ``SKILLDELTA_SPREADSHEET_DURABLE_RESUME=1`` is set.
    """
    if os.environ.get("SKILLDELTA_SPREADSHEET_DURABLE_RESUME", "0") != "1":
        return

    import json
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from skillopt.envs.spreadsheetbench import adapter as spreadsheet_adapter
    from skillopt.envs.spreadsheetbench.rollout import process_one_codegen

    cls = spreadsheet_adapter.SpreadsheetBenchAdapter
    original = cls.rollout
    if getattr(original, "_skilldelta_durable_wrapped", False):
        return

    def _append_jsonl(path: str, row: dict) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def wrapped(self, env_manager, skill_content: str, out_dir: str, **kwargs):
        if self.mode not in ("single", "multi"):
            return original(self, env_manager, skill_content, out_dir, **kwargs)
        workers = max(1, int(self.workers))

        items = list(env_manager)
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, "results.jsonl")
        attempts_path = os.path.join(out_dir, "attempts.jsonl")

        completed: dict[str, dict] = {}
        if os.path.exists(results_path):
            with open(results_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        task_id = str(row["id"])
                    except (ValueError, RecursionError, KeyError, TypeError):
                        continue
                    # Retain the first completed model execution for each task.
                    completed.setdefault(task_id, row)

        pending = [item for item in items if str(item["id"]) not in completed]
        print(
            "  [spreadsheet durable-codegen] "
            f"total={len(items)} done={len(completed)} pending={len(pending)} "
            f"workers={workers}"
        )
        started = time.time()
        transport_failures: list[str] = []

        def _run_item(item: dict) -> dict:
            return process_one_codegen(
                item,
                self.data_root,
                out_dir,
                skill_content,
                self.mode,
                self.max_turns,
                self.max_completion_tokens,
                self.exec_timeout,
                kwargs.get("use_eval_feedback", False),
                kwargs.get("diagnostic_mode", False),
                kwargs.get("diagnostic_instruction", ""),
                (kwargs.get("diagnostic_trace_context_by_id") or {}).get(
                    str(item["id"]), ""
                ),
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_item, item): item for item in pending}
            finished = 0
            for future in as_completed(futures):
                item = futures[future]
                row = future.result()
                finished += 1
                _append_jsonl(attempts_path, row)

                # Once a response reached the model-output stage, parse,
                # execution, and evaluator failures are terminal agent outcomes.
                # Setup-only release failures are also terminal.  Only an LLM
                # transport failure remains eligible for a later resume.
                retryable_transport = (
                    row.get("phase") == "llm"
                    and not row.get("llm_ok", False)
                    and str(row.get("fail_reason", "")).startswith(
                        "llm-call-failed:"
                    )
                )
                if retryable_transport:
                    transport_failures.append(str(item["id"]))
                else:
                    task_id = str(row["id"])
                    if task_id not in completed:
                        _append_jsonl(results_path, row)
                        completed[task_id] = row

                status = (
                    "TRANSPORT"
                    if retryable_transport
                    else ("PASS" if row.get("hard") else "FAIL")
                )
                print(
                    f"    {finished}/{len(pending)} "
                    f"id={str(item['id']):<10} {status} "
                    f"cases={row.get('n_pass', 0)}/{row.get('n_cases', 0)} "
                    f"dt={time.time() - started:.0f}s"
                )

        if transport_failures:
            sample = ", ".join(transport_failures[:10])
            raise RuntimeError(
                f"{len(transport_failures)} SpreadsheetBench transport failures "
                f"remain pending for resume: {sample}"
            )

        # Preserve dataset order and exactly one terminal outcome per task.
        return [completed[str(item["id"])] for item in items]

    wrapped._skilldelta_durable_wrapped = True
    cls.rollout = wrapped


_patch_spreadsheet_durable_codegen_resume()


def _patch_empty_compatible_responses():
    """Retry HTTP-200 responses whose assistant payload is actually empty.

    The 2026-08-24 LiveMath run received choices with empty ``content`` for
    323/354 calls.  The generic compatible backend treated those transactions
    as successful, and the environment consequently recorded ``agent_ok=true``
    before scoring the empty string as a wrong answer.  Preserve legitimate
    tool-call-only messages, but retry empty plain-chat responses and raise if
    the gateway keeps returning them so the environment records agent_ok=false.
    """
    import warnings

    from api_budget import record_failure, record_response, reserve_request_group
    from skillopt.model import openai_compatible_backend as _compat

    _orig = _compat._chat_messages_impl
    empty_attempts = max(1, int(os.environ.get("SKILLDELTA_EMPTY_RESPONSE_RETRIES", "3")))

    def _has_assistant_payload(result) -> bool:
        value = result[0] if isinstance(result, tuple) and result else result
        if isinstance(value, str):
            return bool(value.strip())
        content = getattr(value, "content", "")
        tool_calls = getattr(value, "tool_calls", None)
        return bool(str(content or "").strip()) or bool(tool_calls)

    def _wrapped(*args, **kwargs):
        for empty_idx in range(empty_attempts):
            stage = str(args[3] if len(args) > 3 else kwargs.get("stage", "target"))
            role = str(kwargs.get("role", "target"))
            config = _compat.OPTIMIZER_CONFIG if role == "optimizer" else _compat.TARGET_CONFIG
            model = str(kwargs.get("deployment") or config.deployment)
            reservation = reserve_request_group()
            try:
                result = _orig(*args, **kwargs)
            except Exception as exc:
                record_failure(
                    reservation,
                    stage=stage,
                    role=role,
                    model=model,
                    error_type=type(exc).__name__,
                )
                raise
            usage = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
            empty = not _has_assistant_payload(result)
            record_response(
                reservation,
                usage if isinstance(usage, dict) else {},
                stage=stage,
                role=role,
                model=model,
                empty=empty,
            )
            if not empty:
                return result
            if empty_idx + 1 < empty_attempts:
                warnings.warn(
                    "OpenAI-compatible gateway returned an empty assistant payload; "
                    f"retrying ({empty_idx + 1}/{empty_attempts - 1}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
        raise RuntimeError(
            "OpenAI-compatible gateway returned an empty assistant payload "
            f"{empty_attempts} consecutive times"
        )

    _compat._chat_messages_impl = _wrapped


_patch_empty_compatible_responses()


def _patch_dsml_answer_tags():
    """OfficeQA harness requires <answer>...</answer> tags; qwen models emit
    their native ｜DSML｜answer tags instead (2026-08-24 pilot: model computed
    the correct answer but the harness regex missed it and the turn loop
    failed with 'neither tool request nor final answer'). Rewrite DSML answer
    tags to <answer> in every target-model response. Self-gating (only
    transforms when the DSML tag is present), symmetric across arms, submodule
    untouched — a harness repair like the Azure bridge."""
    import re as _re

    import skillopt.model as _sm

    _orig = _sm.chat_target_messages
    _open = _re.compile(r"<[｜|]?DSML[｜|]?answer>", _re.IGNORECASE | _re.DOTALL)
    _close = _re.compile(r"</[｜|]?DSML[｜|]?answer>", _re.IGNORECASE | _re.DOTALL)

    def _rewrite(text: str) -> str:
        if "<answer>" in text or "DSML" not in text:
            return text
        return _close.sub("</answer>", _open.sub("<answer>", text))

    def _wrapped(*args, **kwargs):
        result = _orig(*args, **kwargs)
        if isinstance(result, str):
            return _rewrite(result)
        if isinstance(result, tuple):
            head, *rest = result
            if isinstance(head, str):
                head = _rewrite(head)
            elif hasattr(head, "content") and isinstance(head.content, str):
                head.content = _rewrite(head.content)
            return (head, *rest)
        if hasattr(result, "content") and isinstance(result.content, str):
            result.content = _rewrite(result.content)
        return result

    _sm.chat_target_messages = _wrapped


_patch_dsml_answer_tags()


def _patch_spreadsheet_formula_recalculation():
    """Recalculate model-written formulas before SpreadsheetBench scoring.

    openpyxl preserves a formula but does not populate its cached value. The
    pinned evaluator compares cached values, which can turn a correct formula
    into a false failure. Opt in for new runs with
    ``SKILLDELTA_RECALC_FORMULAS=1``; the historical protocol remains
    unchanged unless explicitly enabled.
    """
    if os.environ.get("SKILLDELTA_RECALC_FORMULAS", "0") != "1":
        return
    import shutil as _shutil
    import subprocess as _subprocess
    import tempfile as _tempfile
    from pathlib import Path as _Path
    from skillopt.envs.spreadsheetbench import rollout as _rollout

    original = _rollout.evaluate
    if getattr(original, "_skilldelta_formula_recalc", False):
        return

    def wrapped(pred_path, *args, **kwargs):
        pred = _Path(pred_path)
        if pred.exists() and pred.suffix.lower() in {".xlsx", ".xlsm"} and _shutil.which("libreoffice"):
            with _tempfile.TemporaryDirectory(prefix="skilldelta_eval_recalc_") as td:
                try:
                    _subprocess.run(
                        ["libreoffice", "--headless", "--convert-to", "xlsx", "--outdir", td, str(pred)],
                        check=True, stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL,
                    )
                    recalculated = _Path(td) / pred.name
                    if recalculated.exists():
                        _shutil.copy2(recalculated, pred)
                except Exception:
                    # Keep the original evaluator behavior if recalculation
                    # fails; the run ledger still records the model outcome.
                    pass
        return original(pred_path, *args, **kwargs)

    wrapped._skilldelta_formula_recalc = True
    _rollout.evaluate = wrapped


_patch_spreadsheet_formula_recalculation()

from scripts.eval_only import main  # noqa: E402

if __name__ == "__main__":
    main()
