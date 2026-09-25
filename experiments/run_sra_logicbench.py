#!/usr/bin/env python3
"""Run tightly paired no-skill/oracle-skill SRA-Bench LogicBench trials.

The runner intentionally supports only the cheapest validity battlefield.  It
uses the official SR-Agents direct prompt and LogicBench evaluator at the
audited commit, but interleaves the two treatment arms within each task.  Every
request is admitted through SkillDelta's process-safe budget gate and every
response stores per-instance token usage.

Without ``--execute`` this command is a no-cost dry run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from api_budget import (
    ApiBudgetExceeded,
    record_failure,
    record_response,
    reserve_request_group,
)
from model_request import reasoning_extra_body


ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_COMMIT = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
ARMS = ("no_skill", "oracle_skill")
SUPPORTED_ARMS = (
    *ARMS,
    "hard_negative_skill",
    "bottom_negative_skill",
    "fixed_candidate_skill",
)
THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)
TRIGGERS = (
    "The answer is:",
    "the answer is:",
    "Therefore, the answer is",
    "therefore, the answer is",
)
OUTPUT_LOCK = threading.Lock()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def strip_think_tags(text: str) -> str:
    if "<think>" not in text:
        return text
    text = THINK_CLOSED_RE.sub("", text)
    return THINK_OPEN_RE.sub("", text).lstrip()


def extract_from_trigger(raw_output: str) -> str | None:
    best_pos = -1
    best_trigger = ""
    for trigger in TRIGGERS:
        position = raw_output.rfind(trigger)
        if position > best_pos:
            best_pos = position
            best_trigger = trigger
    if best_pos == -1:
        return None
    answer = raw_output[best_pos + len(best_trigger) :].split("\n")[0].strip()
    return answer.rstrip(".").rstrip("/").strip()


def extract_bqa(text: str) -> str:
    lower = text.lower()
    matches = list(
        re.finditer(
            r"(?:the\s+)?answer\s+is[:\s]*\**\s*(yes|no|true|false)\b",
            lower,
        )
    )
    if matches:
        return "yes" if matches[-1].group(1) in ("yes", "true") else "no"
    match = re.match(r"\**(yes|no)\**[.,!\s]", lower)
    if match:
        return match.group(1)
    tail = lower[-500:]
    for pattern in (
        r"cannot\s+(?:be\s+)?(?:conclude|infer|say|determine)",
        r"not\s+necessarily\s+true",
        r"not\s+(?:possible|correct|true|valid)",
        r"cannot\s+(?:logically|necessarily|definitively)",
        r"\bno,\s",
    ):
        if re.search(pattern, tail):
            return "no"
    matches = list(re.finditer(r"\b(yes|no)\b", tail))
    if matches:
        return matches[-1].group(1)
    if "true" in tail:
        return "yes"
    if "false" in tail:
        return "no"
    return text.split("\n")[-1].strip().lower()


def extract_mcqa(text: str, question: str) -> str:
    lower = text.lower()
    matches = list(re.finditer(r"choice[_ ]?(\d+)", lower))
    if matches:
        return f"choice_{matches[-1].group(1)}"
    match = re.search(
        r"(?:answer|option)\s*(?:is)?[:\s]*\**\s*(?:choice[_ ]?)?(\d+)\b",
        lower,
    )
    if match:
        return f"choice_{match.group(1)}"
    choice_texts = []
    for index in range(1, 6):
        match = re.search(
            rf"choice_{index}:\s*(.+?)(?:\n|choice_|$)", question, re.IGNORECASE
        )
        if match:
            choice_texts.append((index, match.group(1).strip()))
    tail = lower[-500:]
    for index, choice_text in reversed(choice_texts):
        if choice_text.lower()[:40] in tail:
            return f"choice_{index}"
    answer = extract_from_trigger(text)
    if answer is None:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        answer = lines[-1] if lines else ""
    answer = answer.strip().lower()
    match = re.search(r"\b([1-5])\b", answer)
    return f"choice_{match.group(1)}" if match else answer


def evaluate_logicbench(raw_output: str, instance: dict[str, Any]) -> dict[str, Any]:
    text = strip_think_tags(raw_output).strip()
    task_type = instance["eval_data"].get("task_type", "BQA")
    extracted = (
        extract_bqa(text)
        if task_type == "BQA"
        else extract_mcqa(text, instance.get("question", ""))
    )
    ground_truth = instance["eval_data"]["answer"].strip().lower()
    return {
        "extracted_answer": extracted,
        "correct": extracted.strip().lower() == ground_truth,
        "ground_truth": ground_truth,
        "task_type": task_type,
    }


def prompt_for(instance: dict[str, Any], skill: dict[str, Any] | None) -> str:
    question = instance["question"]
    if skill is None:
        return question
    return f"Relevant Skill:\n{skill['content']}\n\n{question}"


def arm_order(instance_id: str, repetition: int, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}:{instance_id}:{repetition}".encode("utf-8")
    ).digest()
    return ARMS if digest[0] % 2 == 0 else tuple(reversed(ARMS))


def requested_arm_order(
    instance_id: str, repetition: int, seed: int, arms: tuple[str, ...]
) -> tuple[str, ...]:
    if arms == ARMS:
        return arm_order(instance_id, repetition, seed)
    return tuple(sorted(
        arms,
        key=lambda arm: hashlib.sha256(
            f"{seed}:{instance_id}:{repetition}:{arm}".encode("utf-8")
        ).digest(),
    ))


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage = row.get("usage") or {}
            # An exact-model response that exhausts its token cap without an
            # answer is still an observed agent outcome (an incorrect one),
            # not an infrastructure-missing cell.  The explicit flag covers
            # new rows; the remaining checks recover equivalent legacy rows.
            model_execution_completed = (
                row.get("model_execution_completed") is True
                or (
                    not row.get("error")
                    and row.get("returned_model")
                    and row.get("returned_model") == row.get("requested_model")
                    and int(usage.get("total_tokens") or 0) > 0
                )
            )
            if row.get("valid") is True or model_execution_completed:
                done.add(
                    (row["instance_id"], row["arm"], int(row["repetition"]))
                )
    return done


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def unpack_chat_response(response: Any, *, stream: bool) -> tuple[str, str, str, str, dict[str, int]]:
    """Normalize complete and streamed compatible-chat responses.

    Streaming changes only the HTTP transport.  It is useful for long-reasoning
    responses because compatible gateways can otherwise close an idle
    connection before returning the first response headers.
    """
    if not stream:
        choice = response.choices[0]
        message = choice.message
        return (
            str(message.content or ""),
            str(getattr(message, "reasoning_content", "") or ""),
            str(getattr(response, "model", "") or ""),
            str(getattr(choice, "finish_reason", "") or ""),
            usage_dict(response),
        )

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    returned_model = ""
    finish_reason = ""
    usage: dict[str, int] = {}
    for chunk in response:
        returned_model = str(getattr(chunk, "model", "") or returned_model)
        chunk_usage = usage_dict(chunk)
        if chunk_usage:
            usage = chunk_usage
        for choice in getattr(chunk, "choices", []) or []:
            finish_reason = str(
                getattr(choice, "finish_reason", "") or finish_reason
            )
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                content_parts.append(str(content))
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(str(reasoning))
    return (
        "".join(content_parts),
        "".join(reasoning_parts),
        returned_model,
        finish_reason,
        usage,
    )


def verify_inputs(
    panel_path: Path, skills_path: Path, manifest_path: Path, panel_name: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    if manifest.get("source", {}).get("commit") != UPSTREAM_COMMIT:
        raise SystemExit("manifest does not use the audited SR-Agents commit")
    if manifest.get("dataset") != "logicbench":
        raise SystemExit("this runner currently supports only LogicBench")
    panel_spec = manifest.get("panels", {}).get(panel_name)
    if not panel_spec:
        raise SystemExit(f"panel {panel_name!r} is not declared in the manifest")
    if sha256(panel_path) != panel_spec["sha256"]:
        raise SystemExit("panel SHA256 does not match the frozen manifest")
    skill_spec = manifest.get("selected_skills_artifact", {})
    if sha256(skills_path) != skill_spec.get("sha256"):
        raise SystemExit("selected-skills SHA256 does not match the frozen manifest")
    panel = load_json(panel_path)
    skills = {row["skill_id"]: row for row in load_json(skills_path)}
    for instance in panel:
        annotations = instance.get("skill_annotations", [])
        if instance.get("dataset") != "logicbench" or len(annotations) != 1:
            raise SystemExit(f"invalid LogicBench instance: {instance.get('instance_id')}")
        if annotations[0] not in skills:
            raise SystemExit(f"missing selected skill: {annotations[0]}")
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
    os.environ["TARGET_OPENAI_COMPATIBLE_BASE_URL"] = args.api_base
    os.environ["TARGET_OPENAI_COMPATIBLE_API_KEY"] = args.api_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--panel-name", choices=("pilot", "development", "audit"), required=True
    )
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=SUPPORTED_ARMS,
        default=list(ARMS),
        help="Treatment arms to execute; defaults to the original tight pair.",
    )
    parser.add_argument(
        "--candidate-freeze",
        type=Path,
        help=(
            "Frozen candidate assignments, required for hard_negative_skill "
            "or bottom_negative_skill."
        ),
    )
    parser.add_argument(
        "--fixed-candidate-skill-id",
        help="Candidate used by fixed_candidate_skill; intended for a frozen audit.",
    )
    parser.add_argument(
        "--gold-skill-filter",
        help="Run only frozen panel instances with this gold skill ID.",
    )
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        help="Run exactly this frozen subset; intended for a predeclared audit.",
    )
    parser.add_argument("--api-base-env", default="QWEN_TARGET_OPENAI_COMPATIBLE_BASE_URL")
    parser.add_argument("--api-key-env", default="QWEN_TARGET_OPENAI_COMPATIBLE_API_KEY")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable provider reasoning when the selected model supports it.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streamed HTTP transport without changing inference parameters.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
        default="",
        help=(
            "Optional provider-specific reasoning effort. Omitted by default "
            "to preserve the original Qwen protocol."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum concurrent requests; defaults to the original serial runner.",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=0,
        help="Bounded retries for transient compatible-gateway failures.",
    )
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--max-quota-delta", type=int, default=0)
    parser.add_argument("--max-public-cost-usd", type=float, default=0.0)
    parser.add_argument("--input-usd-per-million-tokens", type=float, default=0.0)
    parser.add_argument("--output-usd-per-million-tokens", type=float, default=0.0)
    parser.add_argument("--quota-poll-every", type=int, default=1)
    parser.add_argument("--budget-state", type=Path)
    parser.add_argument("--call-ledger", type=Path)
    parser.add_argument("--expected-returned-model", default="")
    parser.add_argument(
        "--execute", action="store_true", help="Actually call the configured endpoint."
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.request_retries < 0:
        raise SystemExit("--workers must be positive and retries nonnegative")
    if args.max_public_cost_usd < 0:
        raise SystemExit("--max-public-cost-usd cannot be negative")
    if args.max_public_cost_usd > 0 and (
        args.input_usd_per_million_tokens <= 0
        or args.output_usd_per_million_tokens <= 0
    ):
        raise SystemExit("public cost cap requires positive input/output prices")

    panel_path = args.panel.resolve()
    skills_path = args.skills.resolve()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    panel, skills, manifest = verify_inputs(
        panel_path, skills_path, manifest_path, args.panel_name
    )
    if args.gold_skill_filter:
        if args.gold_skill_filter not in skills:
            raise SystemExit("--gold-skill-filter is not in the frozen skill set")
        panel = [
            instance
            for instance in panel
            if instance["skill_annotations"][0] == args.gold_skill_filter
        ]
        if not panel:
            raise SystemExit("--gold-skill-filter selected zero panel instances")
    if args.instance_ids:
        requested_instance_ids = set(args.instance_ids)
        if len(requested_instance_ids) != len(args.instance_ids):
            raise SystemExit("--instance-ids contains duplicates")
        available_instance_ids = {
            instance["instance_id"] for instance in panel
        }
        if not requested_instance_ids.issubset(available_instance_ids):
            raise SystemExit("--instance-ids is not a subset of the selected panel")
        panel = [
            instance
            for instance in panel
            if instance["instance_id"] in requested_instance_ids
        ]
    requested_arms = tuple(dict.fromkeys(args.arms))
    if "fixed_candidate_skill" in requested_arms:
        if not args.fixed_candidate_skill_id:
            raise SystemExit(
                "fixed_candidate_skill requires --fixed-candidate-skill-id"
            )
        if args.fixed_candidate_skill_id not in skills:
            raise SystemExit("fixed candidate is not in the frozen skill set")
        if any(
            instance["skill_annotations"][0] == args.fixed_candidate_skill_id
            for instance in panel
        ):
            raise SystemExit("fixed candidate must differ from every selected gold skill")
    candidate_assignments: dict[str, dict[str, Any]] = {}
    candidate_freeze_sha256 = None
    ranked_candidate_arms = {
        "hard_negative_skill",
        "bottom_negative_skill",
    }.intersection(requested_arms)
    if ranked_candidate_arms:
        if args.candidate_freeze is None:
            raise SystemExit("ranked candidate arms require --candidate-freeze")
        candidate_path = args.candidate_freeze.resolve()
        candidate_freeze = load_json(candidate_path)
        if candidate_freeze.get("source", {}).get("commit") != UPSTREAM_COMMIT:
            raise SystemExit("candidate freeze uses an unexpected source commit")
        if (
            candidate_freeze.get("source", {}).get("manifest_content_sha256")
            != manifest.get("manifest_content_sha256")
        ):
            raise SystemExit("candidate freeze and manifest content hashes differ")
        frozen_panel = candidate_freeze.get("panels", {}).get(args.panel_name, {})
        if frozen_panel.get("panel_sha256") != sha256(panel_path):
            raise SystemExit("candidate freeze does not match the panel SHA256")
        candidate_assignments = frozen_panel.get("assignments", {})
        if not {instance["instance_id"] for instance in panel}.issubset(
            candidate_assignments
        ):
            raise SystemExit("candidate freeze assignments do not cover the panel")
        for instance in panel:
            assignment = candidate_assignments[instance["instance_id"]]
            gold = instance["skill_annotations"][0]
            negative = assignment.get("hard_negative_skill_id")
            if assignment.get("gold_skill_id") != gold:
                raise SystemExit("candidate assignment gold skill mismatch")
            if "hard_negative_skill" in ranked_candidate_arms and (
                negative == gold or negative not in skills
            ):
                raise SystemExit("invalid frozen hard-negative skill")
            bottom_negative = assignment.get("bottom_negative_skill_id")
            if "bottom_negative_skill" in ranked_candidate_arms and (
                bottom_negative == gold or bottom_negative not in skills
            ):
                raise SystemExit("invalid frozen bottom-negative skill")
        candidate_freeze_sha256 = sha256(candidate_path)
    done = completed_keys(output_path)
    pending = [
        (instance, repetition, arm)
        for instance in panel
        for repetition in range(args.repetitions)
        for arm in requested_arms
        if (instance["instance_id"], arm, repetition) not in done
    ]
    prompt_chars = {arm: 0 for arm in requested_arms}
    for instance in panel:
        gold_skill_id = instance["skill_annotations"][0]
        if "no_skill" in prompt_chars:
            prompt_chars["no_skill"] += len(prompt_for(instance, None))
        if "oracle_skill" in prompt_chars:
            prompt_chars["oracle_skill"] += len(
                prompt_for(instance, skills[gold_skill_id])
            )
        if "hard_negative_skill" in prompt_chars:
            negative_skill_id = candidate_assignments[instance["instance_id"]][
                "hard_negative_skill_id"
            ]
            prompt_chars["hard_negative_skill"] += len(
                prompt_for(instance, skills[negative_skill_id])
            )
        if "bottom_negative_skill" in prompt_chars:
            bottom_skill_id = candidate_assignments[instance["instance_id"]][
                "bottom_negative_skill_id"
            ]
            prompt_chars["bottom_negative_skill"] += len(
                prompt_for(instance, skills[bottom_skill_id])
            )
        if "fixed_candidate_skill" in prompt_chars:
            prompt_chars["fixed_candidate_skill"] += len(
                prompt_for(instance, skills[args.fixed_candidate_skill_id])
            )
    already_completed = sum(
        (instance["instance_id"], arm, repetition) in done
        for instance in panel
        for repetition in range(args.repetitions)
        for arm in requested_arms
    )
    plan = {
        "mode": "execute" if args.execute else "dry_run",
        "dataset": "logicbench",
        "panel": args.panel_name,
        "tasks": len(panel),
        "skills": sorted(skills),
        "arms": list(requested_arms),
        "repetitions": args.repetitions,
        "total_requests": len(panel) * len(requested_arms) * args.repetitions,
        "already_completed": already_completed,
        "pending_requests": len(pending),
        "prompt_chars_per_repetition": prompt_chars,
        "max_completion_tokens_if_all_pending": len(pending) * args.max_tokens,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "stream": args.stream,
        "reasoning_effort": args.reasoning_effort or None,
        "max_tokens_per_request": args.max_tokens,
        "workers": args.workers,
        "request_retries": args.request_retries,
        "model": args.model,
        "max_public_cost_usd": args.max_public_cost_usd,
        "input_usd_per_million_tokens": args.input_usd_per_million_tokens,
        "output_usd_per_million_tokens": args.output_usd_per_million_tokens,
        "gold_skill_filter": args.gold_skill_filter,
        "instance_ids_filter": args.instance_ids,
        "fixed_candidate_skill_id": args.fixed_candidate_skill_id,
        "manifest_content_sha256": manifest.get("manifest_content_sha256"),
        "candidate_freeze_sha256": candidate_freeze_sha256,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Dry run only: zero model API calls.")
        return

    if args.max_calls <= 0:
        raise SystemExit("--execute requires an explicit positive --max-calls")
    if args.budget_state is None or args.call_ledger is None:
        raise SystemExit("--execute requires --budget-state and --call-ledger")
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
    pending_keys = {
        (instance["instance_id"], arm, repetition)
        for instance, repetition, arm in pending
    }
    jobs = []
    for instance in panel:
        for repetition in range(args.repetitions):
            for arm in requested_arm_order(
                instance["instance_id"], repetition, args.seed, requested_arms
            ):
                if (instance["instance_id"], arm, repetition) in pending_keys:
                    jobs.append((instance, repetition, arm))

    budget_stop = threading.Event()
    identity_stop = threading.Event()

    def execute_job(job: tuple[dict[str, Any], int, str]) -> tuple[str, tuple[str, str, int]]:
        instance, repetition, arm = job
        key = (instance["instance_id"], arm, repetition)
        if budget_stop.is_set():
            return "skipped_after_budget", key
        if identity_stop.is_set():
            return "skipped_after_identity", key
        try:
            reservation = reserve_request_group()
        except ApiBudgetExceeded as exc:
            if not budget_stop.is_set():
                print(f"Budget stop before {key}: {exc}", file=sys.stderr, flush=True)
            budget_stop.set()
            return "budget", key

        skill_id = instance["skill_annotations"][0]
        if arm == "no_skill":
            treatment_skill_id = None
        elif arm == "oracle_skill":
            treatment_skill_id = skill_id
        elif arm == "hard_negative_skill":
            treatment_skill_id = candidate_assignments[
                instance["instance_id"]
            ]["hard_negative_skill_id"]
        elif arm == "bottom_negative_skill":
            treatment_skill_id = candidate_assignments[
                instance["instance_id"]
            ]["bottom_negative_skill_id"]
        else:
            treatment_skill_id = args.fixed_candidate_skill_id
        treatment_skill = skills[treatment_skill_id] if treatment_skill_id else None
        prompt = prompt_for(instance, treatment_skill)
        request_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": args.temperature,
                    "thinking": args.thinking,
                    "stream": args.stream,
                    "reasoning_effort": args.reasoning_effort or None,
                    "max_tokens": args.max_tokens,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        base_row: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "instance_id": instance["instance_id"],
            "dataset": "logicbench",
            "skill_id": skill_id,
            "gold_skill_id": skill_id,
            "treatment_skill_id": treatment_skill_id,
            "candidate_relevance": (
                None if arm == "no_skill" else arm == "oracle_skill"
            ),
            "arm": arm,
            "repetition": repetition,
            "requested_model": args.model,
            "source_commit": UPSTREAM_COMMIT,
            "manifest_content_sha256": manifest.get("manifest_content_sha256"),
            "candidate_freeze_sha256": candidate_freeze_sha256,
            "request_sha256": request_fingerprint,
            "temperature": args.temperature,
            "thinking": args.thinking,
            "stream": args.stream,
            "max_tokens": args.max_tokens,
        }
        try:
            request_kwargs = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            # GLM's provider exposes two alternative controls.  A native
            # reasoning-effort request must not be combined with the chat
            # template switch; the latter is used only by the strict
            # no-hidden-reasoning replay.
            extra_body = (
                None
                if args.reasoning_effort
                else reasoning_extra_body(args.model, thinking=args.thinking)
            )
            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body
            if args.reasoning_effort:
                request_kwargs["reasoning_effort"] = args.reasoning_effort
            if args.stream:
                request_kwargs["stream"] = True
                request_kwargs["stream_options"] = {"include_usage": True}
            response = None
            for attempt in range(args.request_retries + 1):
                try:
                    response = client.chat.completions.create(**request_kwargs)
                    break
                except Exception as exc:
                    record_failure(
                        reservation,
                        stage=f"sra_logicbench_{arm}",
                        role="target",
                        model=args.model,
                        error_type=type(exc).__name__,
                    )
                    if attempt >= args.request_retries:
                        raise
                    time.sleep(min(8.0, 2.0**attempt))
                    reservation = reserve_request_group()
            assert response is not None
            (
                raw_output,
                reasoning_content,
                returned_model,
                finish_reason,
                usage,
            ) = unpack_chat_response(response, stream=args.stream)
            valid = bool(raw_output.strip())
            record_response(
                reservation,
                usage,
                stage=f"sra_logicbench_{arm}",
                role="target",
                model=args.model,
                returned_model=returned_model or None,
                empty=not valid,
            )
            if args.expected_returned_model and returned_model != args.expected_returned_model:
                valid = False
                identity_error = (
                    f"returned model {returned_model!r} != expected "
                    f"{args.expected_returned_model!r}"
                )
                identity_stop.set()
            else:
                identity_error = None
            model_execution_completed = bool(
                identity_error is None
                and returned_model
                and usage.get("total_tokens", 0) > 0
            )
            evaluation = (
                evaluate_logicbench(raw_output, instance)
                if valid
                else {
                    "extracted_answer": "",
                    "correct": False if model_execution_completed else None,
                    "ground_truth": instance["eval_data"]["answer"],
                    "task_type": instance["eval_data"].get("task_type", "BQA"),
                }
            )
            append_jsonl(
                output_path,
                {
                    **base_row,
                    "returned_model": returned_model,
                    "raw_output": raw_output,
                    "finish_reason": finish_reason,
                    "reasoning_content_chars": len(reasoning_content),
                    "usage": usage,
                    "valid": valid,
                    "model_execution_completed": model_execution_completed,
                    "error": identity_error,
                    **evaluation,
                },
            )
            return "ok" if valid else "invalid", key
        except Exception as exc:  # noqa: BLE001
            record_failure(
                reservation,
                stage=f"sra_logicbench_{arm}",
                role="target",
                model=args.model,
                error_type=type(exc).__name__,
            )
            append_jsonl(
                output_path,
                {
                    **base_row,
                    "valid": False,
                    "correct": None,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return "failed", key

    status_counts: dict[str, int] = {}
    if args.workers == 1:
        outcomes = (execute_job(job) for job in jobs)
        for completed, (status, _key) in enumerate(outcomes, start=1):
            status_counts[status] = status_counts.get(status, 0) + 1
            if completed % 50 == 0 or completed == len(jobs):
                print(json.dumps({"progress": completed, "pending_at_start": len(jobs), "status": status_counts}, sort_keys=True), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(execute_job, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), start=1):
                status, _key = future.result()
                status_counts[status] = status_counts.get(status, 0) + 1
                if completed % 50 == 0 or completed == len(jobs):
                    print(json.dumps({"progress": completed, "pending_at_start": len(jobs), "status": status_counts}, sort_keys=True), flush=True)

    stopped_for_budget = budget_stop.is_set()
    stopped_for_identity = identity_stop.is_set()

    finished = completed_keys(output_path)
    completed_for_run = sum(
        (instance["instance_id"], arm, repetition) in finished
        for instance in panel
        for repetition in range(args.repetitions)
        for arm in requested_arms
    )
    print(
        json.dumps(
            {
                "completed_model_outcomes": completed_for_run,
                "expected": len(panel) * len(requested_arms) * args.repetitions,
                "budget_stopped": stopped_for_budget,
                "identity_stopped": stopped_for_identity,
                "workers": args.workers,
                "execution_status_counts": status_counts,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if stopped_for_budget:
        raise SystemExit(2)
    if stopped_for_identity:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
