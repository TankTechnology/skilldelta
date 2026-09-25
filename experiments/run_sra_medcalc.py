#!/usr/bin/env python3
"""Run budgeted paired MedCalc trajectories on frozen panels.

The implementation preserves the official SR-Agents prompt, text tool-call
protocol, and evaluator.  It repairs only the restricted tool namespace:
allowlisted ``math`` and ``datetime`` imports are removed from tool source and
their modules/classes are preloaded before execution.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

from api_budget import (
    ApiBudgetExceeded,
    record_failure,
    record_response,
    reserve_request_group,
)
from model_request import reasoning_extra_body


ROOT = Path(__file__).resolve().parent.parent
SRA_ROOT = ROOT / "experiments/data/SR-Agents"
SRA_SRC = SRA_ROOT / "src"
if str(SRA_SRC) not in sys.path:
    sys.path.insert(0, str(SRA_SRC))

from sragents.infer.engines.tool_loop import parse_tool_call  # noqa: E402
from sragents.prompts import build_prompt  # noqa: E402


SOURCE_COMMIT = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
ARMS = ("no_skill", "oracle_skill")
OUTPUT_LOCK = threading.Lock()


def allowlisted_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Permit only modules required by the frozen math/date calculators."""
    allowed = {
        "math", "datetime", "_strptime", "time", "locale", "calendar", "re",
        "_locale", "_datetime",
    }
    if name.split(".", 1)[0] not in allowed:
        raise ImportError(f"module {name!r} is not allowlisted")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "sum": sum,
    "pow": pow,
    "bool": bool,
    "True": True,
    "False": False,
    "None": None,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "isinstance": isinstance,
    "__import__": allowlisted_import,
}
DATE_IDS = {13, 68}
GESTATIONAL_ID = 69
INTEGER_IDS = {
    4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69,
}
TRIGGERS = (
    "The answer is:",
    "the answer is:",
    "Therefore, the answer is",
    "therefore, the answer is",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    raw = usage.model_dump() if hasattr(usage, "model_dump") else vars(usage)
    prompt = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
    completion = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(raw.get("total_tokens") or prompt + completion),
    }


def strip_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).lstrip()


def extract_from_trigger(text: str) -> str | None:
    position, selected = -1, ""
    for trigger in TRIGGERS:
        candidate = text.rfind(trigger)
        if candidate > position:
            position, selected = candidate, trigger
    if position < 0:
        return None
    return (
        text[position + len(selected) :]
        .split("\n")[0]
        .strip()
        .rstrip(".")
        .rstrip("/")
        .strip()
    )


def safe_number(value: str) -> float | None:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        match = re.match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", value)
        if match and float(match.group(2)) != 0:
            return float(match.group(1)) / float(match.group(2))
    return None


def evaluate_medcalc(raw_output: str, instance: dict[str, Any]) -> dict[str, Any]:
    """Pinned copy of the audited SR-Agents MedCalc evaluator."""
    text = strip_think_tags(raw_output)
    eval_data = instance["eval_data"]
    output_type = eval_data.get("output_type", "decimal")
    calculator_id = eval_data.get("calculator_id", 0)
    extracted = ""
    for line in reversed(text.strip().split("\n")):
        if line.strip().upper().startswith("ANSWER:"):
            extracted = line.strip()[len("ANSWER:") :].strip().strip("*").strip()
            break
    if not extracted:
        match = re.search(r'[Aa]nswer":\s*(.*?)\}', text)
        if match:
            candidate = match.group(1).strip().strip('"').strip("'")
            if candidate not in {
                "str(short_and_direct_answer_of_the_question)",
                "str(value which is the answer to the question)",
                "X.XX",
            }:
                extracted = candidate
    if not extracted:
        extracted = extract_from_trigger(text) or ""
    if not extracted and (output_type == "date" or calculator_id in DATE_IDS):
        match = re.search(
            r"(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})", text
        )
        if match:
            extracted = f"{int(match.group(1)):02d}/{int(match.group(2)):02d}/{match.group(3)}"
    if not extracted and calculator_id == GESTATIONAL_ID:
        match = re.search(
            r"\(?[\"']?(\d+)\s*(?:weeks?)?\s*,?\s*[\"']?(\d+)\s*(?:days?)?[\"']?\s*\)?",
            text,
        )
        if match:
            extracted = f"({match.group(1)}, {match.group(2)})"
    if not extracted:
        numbers = re.findall(r"-?\d+\.?\d*", text)
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        extracted = numbers[-1] if numbers else (lines[-1] if lines else "")

    ground_truth = str(eval_data["answer"]).strip()
    if calculator_id in DATE_IDS:
        try:
            correct = datetime.strptime(ground_truth, "%m/%d/%Y") == datetime.strptime(
                extracted.strip(), "%m/%d/%Y"
            )
        except (ValueError, TypeError):
            correct = False
        kind = "date"
    elif calculator_id == GESTATIONAL_ID:
        pattern = re.compile(
            r"\(?[\"']?(\d+)\s*(?:weeks?)?[\"']?,?\s*[\"']?(\d+)\s*(?:days?)?[\"']?\s*\)?"
        )
        truth_match, prediction_match = pattern.search(ground_truth), pattern.search(extracted)
        correct = bool(
            truth_match
            and prediction_match
            and truth_match.groups() == prediction_match.groups()
        )
        kind = "gestational_age"
    elif calculator_id in INTEGER_IDS or output_type == "integer":
        truth, prediction = safe_number(ground_truth), safe_number(extracted)
        correct = truth is not None and prediction is not None and round(prediction) == round(truth)
        kind = "integer"
    else:
        prediction = safe_number(extracted)
        lower = safe_number(str(eval_data.get("lower_limit", "")))
        upper = safe_number(str(eval_data.get("upper_limit", "")))
        correct = (
            prediction is not None
            and lower is not None
            and upper is not None
            and lower <= prediction <= upper
        )
        kind = "decimal"
    return {"correct": bool(correct), "output_type": kind, "extracted_answer": extracted}


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("valid") is True:
            done.add(
                (str(row["instance_id"]), str(row["arm"]), int(row["repetition"]))
            )
    return done


def arm_order(instance_id: str, repetition: int, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}:{instance_id}:{repetition}".encode("utf-8")
    ).digest()
    return ARMS if digest[0] % 2 == 0 else tuple(reversed(ARMS))


def _clean_tool_tree(implementation: str) -> ast.Module:
    tree = ast.parse(implementation)

    class ImportStripper(ast.NodeTransformer):
        def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
            if any(alias.name != "math" for alias in node.names):
                raise ValueError(f"non-allowlisted import: {ast.unparse(node)}")
            return None

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
            names = {alias.name for alias in node.names}
            if node.module != "datetime" or not names <= {"datetime", "timedelta"}:
                raise ValueError(f"non-allowlisted import: {ast.unparse(node)}")
            return None

    tree = ImportStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def load_tool(tool: dict[str, Any]) -> Any:
    """Compile one frozen tool in the repaired restricted namespace."""
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
        "datetime": datetime,
        "timedelta": timedelta,
    }
    tree = _clean_tool_tree(str(tool["implementation"]))
    exec(compile(tree, f"<medcalc:{tool['name']}>", "exec"), namespace)  # noqa: S102
    function = namespace.get(str(tool["name"]))
    if not callable(function):
        raise ValueError(f"tool implementation does not define {tool['name']}")
    return function


def validate_tools(skills: dict[str, dict[str, Any]]) -> dict[str, int]:
    total = 0
    for skill in skills.values():
        tools = skill.get("tools", [])
        if not tools:
            raise ValueError(f"skill has no tools: {skill['skill_id']}")
        for tool in tools:
            load_tool(tool)
            total += 1
    return {"skills": len(skills), "tools_loaded": total, "errors": 0}


def execute_tool(tool: dict[str, Any], args: dict[str, Any]) -> str:
    return str(load_tool(tool)(**args))


def verify_inputs(
    manifest_path: Path,
    panel_path: Path,
    skills_path: Path,
    panel_name: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = read_json(manifest_path)
    unhashed = dict(manifest)
    content_hash = unhashed.pop("manifest_content_sha256", None)
    if json_sha256(unhashed) != content_hash:
        raise SystemExit("manifest content SHA256 is invalid")
    if (
        manifest.get("dataset") != "medcalcbench"
        or manifest.get("source", {}).get("commit") != SOURCE_COMMIT
    ):
        raise SystemExit("runner requires the audited MedCalc manifest")
    panel_spec = manifest.get("panels", {}).get(panel_name)
    if not panel_spec or sha256(panel_path) != panel_spec.get("sha256"):
        raise SystemExit("panel does not match the frozen manifest")
    skill_spec = manifest.get("selected_skills_artifact", {})
    if sha256(skills_path) != skill_spec.get("sha256"):
        raise SystemExit("skills do not match the frozen manifest")
    panel = read_json(panel_path)
    skills = {str(row["skill_id"]): row for row in read_json(skills_path)}
    if len(skills) != 55:
        raise SystemExit("expected all 55 MedCalc skills")
    for row in panel:
        annotations = row.get("skill_annotations", [])
        if (
            row.get("dataset") != "medcalcbench"
            or len(annotations) != 1
            or str(annotations[0]) not in skills
        ):
            raise SystemExit(f"invalid MedCalc row: {row.get('instance_id')}")
    return panel, skills, manifest


def configure_budget(args: argparse.Namespace) -> None:
    os.environ["SKILLDELTA_API_BUDGET_STATE"] = str(args.budget_state.resolve())
    os.environ["SKILLDELTA_API_CALL_LEDGER"] = str(args.call_ledger.resolve())
    os.environ["SKILLDELTA_MAX_API_CALLS"] = str(args.max_calls)
    os.environ["SKILLDELTA_MAX_TOTAL_TOKENS"] = str(args.max_total_tokens)
    os.environ["SKILLDELTA_MAX_QUOTA_DELTA"] = str(args.max_quota_delta)
    os.environ["SKILLDELTA_MAX_PUBLIC_COST_NANODOLLARS"] = str(
        round(args.max_public_cost_usd * 1_000_000_000)
    )
    os.environ["SKILLDELTA_INPUT_PRICE_NANODOLLARS_PER_TOKEN"] = str(
        round(args.input_usd_per_million_tokens * 1_000)
    )
    os.environ["SKILLDELTA_OUTPUT_PRICE_NANODOLLARS_PER_TOKEN"] = str(
        round(args.output_usd_per_million_tokens * 1_000)
    )
    os.environ["SKILLDELTA_QUOTA_POLL_EVERY"] = str(args.quota_poll_every)
    os.environ["SKILLDELTA_QUOTA_QUERY_ATTEMPTS"] = "3"
    os.environ.setdefault("SKILLDELTA_QUOTA_FAIL_CLOSED", "1")
    os.environ["TARGET_OPENAI_COMPATIBLE_BASE_URL"] = args.api_base
    os.environ["TARGET_OPENAI_COMPATIBLE_API_KEY"] = args.api_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-name", choices=("support", "audit"), required=True)
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen-turbo")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--instance-ids", nargs="+")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable provider reasoning when the selected model supports it.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
        default="",
        help="Optional provider-specific reasoning effort for native-model diagnostics.",
    )
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--request-retries",
        type=int,
        default=0,
        help="Bounded retries for transient compatible-gateway failures.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--api-base-env", default="QWEN_TARGET_OPENAI_COMPATIBLE_BASE_URL")
    parser.add_argument("--api-key-env", default="QWEN_TARGET_OPENAI_COMPATIBLE_API_KEY")
    parser.add_argument("--expected-returned-model", default="qwen-turbo")
    parser.add_argument(
        "--allow-non-qwen-model",
        action="store_true",
        help="Run an explicitly named model in a separate replay output.",
    )
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--max-quota-delta", type=int, default=0)
    parser.add_argument("--max-public-cost-usd", type=float, default=0.0)
    parser.add_argument("--input-usd-per-million-tokens", type=float, default=0.05)
    parser.add_argument("--output-usd-per-million-tokens", type=float, default=0.20)
    parser.add_argument("--quota-poll-every", type=int, default=100)
    parser.add_argument("--budget-state", type=Path)
    parser.add_argument("--call-ledger", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.max_rounds <= 0 or args.max_tokens <= 0 or args.request_retries < 0:
        raise SystemExit("workers/max-rounds/max-tokens must be positive and retries nonnegative")

    manifest_path = args.manifest.resolve()
    panel_path = args.panel.resolve()
    skills_path = args.skills.resolve()
    output_path = args.output.resolve()
    panel, skills, manifest = verify_inputs(
        manifest_path, panel_path, skills_path, args.panel_name
    )
    tool_validation = validate_tools(skills)
    if args.instance_ids:
        requested = set(args.instance_ids)
        if len(requested) != len(args.instance_ids):
            raise SystemExit("--instance-ids contains duplicates")
        available = {str(row["instance_id"]) for row in panel}
        if not requested <= available:
            raise SystemExit("--instance-ids is not a subset of the frozen panel")
        panel = [row for row in panel if str(row["instance_id"]) in requested]

    requested_arms = tuple(dict.fromkeys(args.arms))
    done = completed_keys(output_path)
    pending = [
        (row, repetition, arm)
        for row in panel
        for repetition in range(args.repetitions)
        for arm in requested_arms
        if (str(row["instance_id"]), arm, repetition) not in done
    ]
    prompt_chars = {arm: 0 for arm in requested_arms}
    for row in panel:
        skill = skills[str(row["skill_annotations"][0])]
        for arm in requested_arms:
            skill_text = [str(skill["content"])] if arm == "oracle_skill" else None
            system, user = build_prompt(row, skills=skill_text)
            prompt_chars[arm] += len(system) + len(user)
    plan = {
        "mode": "execute" if args.execute else "dry_run",
        "dataset": "medcalcbench",
        "panel": args.panel_name,
        "tasks": len(panel),
        "skill_families": len({row["skill_annotations"][0] for row in panel}),
        "arms": list(requested_arms),
        "repetitions": args.repetitions,
        "trajectories": len(panel) * len(requested_arms) * args.repetitions,
        "already_completed": len(panel) * len(requested_arms) * args.repetitions - len(pending),
        "pending_trajectories": len(pending),
        "maximum_pending_model_calls": sum(
            args.max_rounds if arm == "oracle_skill" else 1
            for _row, _repetition, arm in pending
        ),
        "first_call_prompt_chars_per_repetition": prompt_chars,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "max_rounds": args.max_rounds,
        "max_tokens_per_round": args.max_tokens,
        "workers": args.workers,
        "request_retries": args.request_retries,
        "model": args.model,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "tool_validation": tool_validation,
        "max_public_cost_usd": args.max_public_cost_usd,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("Dry run only: zero model API calls.")
        return
    if not args.allow_non_qwen_model and (
        args.model != "qwen-turbo" or args.expected_returned_model != "qwen-turbo"
    ):
        raise SystemExit("the frozen study requires requested/returned qwen-turbo")
    if args.allow_non_qwen_model and args.expected_returned_model != args.model:
        raise SystemExit("non-Qwen replay requires an exact expected returned model")
    if args.max_calls <= 0 or args.max_total_tokens <= 0 or args.max_public_cost_usd <= 0:
        raise SystemExit("execution requires positive call/token/public-cost caps")
    if args.max_quota_delta <= 0:
        raise SystemExit("execution requires a positive proxy quota cap")
    if args.budget_state is None or args.call_ledger is None:
        raise SystemExit("execution requires budget-state and call-ledger paths")
    args.api_base = os.environ.get(args.api_base_env, "").strip()
    args.api_key = os.environ.get(args.api_key_env, "").strip()
    if not args.api_base or not args.api_key:
        raise SystemExit(f"missing {args.api_base_env} or {args.api_key_env}")
    configure_budget(args)

    from openai import OpenAI

    client = OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
        max_retries=0,
        timeout=args.timeout,
    )
    budget_stop = threading.Event()
    identity_stop = threading.Event()

    def call_model(
        messages: list[dict[str, str]], arm: str
    ) -> tuple[str, dict[str, int], str, str]:
        if budget_stop.is_set():
            raise ApiBudgetExceeded("global MedCalc budget stop is active")
        if identity_stop.is_set():
            raise RuntimeError("global model-identity stop is active")
        try:
            reservation = reserve_request_group()
        except ApiBudgetExceeded:
            budget_stop.set()
            raise
        request = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        extra_body = (
            None
            if args.reasoning_effort
            else reasoning_extra_body(args.model, thinking=args.thinking)
        )
        if extra_body is not None:
            request["extra_body"] = extra_body
        if args.reasoning_effort:
            request["reasoning_effort"] = args.reasoning_effort
        fingerprint = hashlib.sha256(
            json.dumps(
                request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        response = None
        for attempt in range(args.request_retries + 1):
            try:
                response = client.chat.completions.create(**request)
                break
            except Exception as exc:
                record_failure(
                    reservation,
                    stage=f"sra_medcalc_{args.panel_name}_{arm}",
                    role="target_round",
                    model=args.model,
                    error_type=type(exc).__name__,
                )
                if attempt >= args.request_retries:
                    raise
                time.sleep(min(8.0, 2.0**attempt))
                reservation = reserve_request_group()
        assert response is not None
        content = response.choices[0].message.content or ""
        usage = usage_dict(response)
        returned_model = str(getattr(response, "model", "") or "")
        record_response(
            reservation,
            usage,
            stage=f"sra_medcalc_{args.panel_name}_{arm}",
            role="target_round",
            model=args.model,
            returned_model=returned_model or None,
            empty=not bool(content.strip()),
        )
        if returned_model != args.expected_returned_model:
            identity_stop.set()
            raise RuntimeError(
                f"returned model {returned_model!r} != {args.expected_returned_model!r}"
            )
        return content, usage, returned_model, fingerprint

    def run_job(job: tuple[dict[str, Any], int, str]) -> str:
        row, repetition, arm = job
        if budget_stop.is_set() or identity_stop.is_set():
            return "skipped"
        skill_id = str(row["skill_annotations"][0])
        skill = skills[skill_id]
        skill_text = [str(skill["content"])] if arm == "oracle_skill" else None
        system, user = build_prompt(row, skills=skill_text)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        tools = skill.get("tools", []) if arm == "oracle_skill" else []
        tool_index = {str(tool["name"]): tool for tool in tools}
        generated = ""
        transcript = ""
        step_usage: list[dict[str, int]] = []
        returned_models: list[str] = []
        request_hashes: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        base_row = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "instance_id": str(row["instance_id"]),
            "dataset": "medcalcbench",
            "panel": args.panel_name,
            "skill_id": skill_id,
            "gold_skill_id": skill_id,
            "treatment_skill_id": skill_id if arm == "oracle_skill" else None,
            "arm": arm,
            "repetition": repetition,
            "requested_model": args.model,
            "source_commit": SOURCE_COMMIT,
            "manifest_content_sha256": manifest["manifest_content_sha256"],
            "temperature": args.temperature,
            "thinking": args.thinking,
            "reasoning_effort": args.reasoning_effort,
            "max_rounds": args.max_rounds,
            "max_tokens": args.max_tokens,
            "tool_executor": "allowlisted_math_datetime_v1",
        }
        try:
            for _round in range(args.max_rounds):
                content, usage, returned, fingerprint = call_model(messages, arm)
                step_usage.append(usage)
                returned_models.append(returned)
                request_hashes.append(fingerprint)
                parsed = parse_tool_call(content, tool_index) if tool_index else None
                if parsed is None:
                    generated += content
                    transcript += content
                    break
                head, tool_name, tool_args = parsed
                try:
                    result = execute_tool(tool_index[tool_name], tool_args)
                    tool_error = None
                except Exception as exc:  # noqa: BLE001
                    result = f"Error: {exc}"
                    tool_error = f"{type(exc).__name__}: {exc}"
                generated += head
                transcript += head + f"\nTOOL_RESULT: {result}\n"
                tool_calls.append(
                    {
                        "name": tool_name,
                        "arguments": tool_args,
                        "result": result,
                        "error": tool_error,
                    }
                )
                messages.append({"role": "assistant", "content": head})
                messages.append({"role": "user", "content": f"TOOL_RESULT: {result}"})
            evaluation = evaluate_medcalc(generated, row)
            valid = bool(generated.strip()) and bool(step_usage)
            usage = {
                key: sum(step.get(key, 0) for step in step_usage)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            append_jsonl(
                output_path,
                {
                    **base_row,
                    "returned_models": returned_models,
                    "request_sha256": request_hashes,
                    "raw_output": generated,
                    "transcript": transcript,
                    "usage": usage,
                    "step_usage": step_usage,
                    "model_rounds": len(step_usage),
                    "tool_calls": tool_calls,
                    "valid": valid,
                    "error": None,
                    "ground_truth": row["eval_data"]["answer"],
                    **evaluation,
                },
            )
            return "ok" if valid else "invalid"
        except ApiBudgetExceeded as exc:
            budget_stop.set()
            append_jsonl(
                output_path,
                {
                    **base_row,
                    "returned_models": returned_models,
                    "request_sha256": request_hashes,
                    "raw_output": generated,
                    "transcript": transcript,
                    "step_usage": step_usage,
                    "tool_calls": tool_calls,
                    "valid": False,
                    "correct": None,
                    "error": f"ApiBudgetExceeded: {exc}",
                },
            )
            return "budget"
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                output_path,
                {
                    **base_row,
                    "returned_models": returned_models,
                    "request_sha256": request_hashes,
                    "raw_output": generated,
                    "transcript": transcript,
                    "step_usage": step_usage,
                    "tool_calls": tool_calls,
                    "valid": False,
                    "correct": None,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return "failed"

    pending_keys = {
        (str(row["instance_id"]), arm, repetition)
        for row, repetition, arm in pending
    }
    jobs: list[tuple[dict[str, Any], int, str]] = []
    for row in panel:
        for repetition in range(args.repetitions):
            for arm in arm_order(str(row["instance_id"]), repetition, args.seed):
                if (
                    arm in requested_arms
                    and (str(row["instance_id"]), arm, repetition) in pending_keys
                ):
                    jobs.append((row, repetition, arm))

    statuses: dict[str, int] = {}
    if args.workers == 1:
        for completed, job in enumerate(jobs, 1):
            status = run_job(job)
            statuses[status] = statuses.get(status, 0) + 1
            print(json.dumps({"progress": completed, "total": len(jobs), "status": statuses}), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                status = future.result()
                statuses[status] = statuses.get(status, 0) + 1
                print(json.dumps({"progress": completed, "total": len(jobs), "status": statuses}), flush=True)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "execution_status_counts": statuses,
                "budget_stopped": budget_stop.is_set(),
                "identity_stopped": identity_stop.is_set(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if budget_stop.is_set():
        raise SystemExit(2)
    if identity_stop.is_set():
        raise SystemExit(3)


if __name__ == "__main__":
    main()
