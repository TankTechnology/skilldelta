#!/usr/bin/env python3
"""Generate paired SRA-Bench BigCodeBench trajectories with hard cost caps.

The script only performs model generation.  Generated code is deliberately not
executed here; ``evaluate_sra_bigcode.py`` scores it later inside bubblewrap.
Without ``--execute`` this command is a zero-cost dry run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
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
UPSTREAM_COMMIT = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
ARMS = ("no_skill", "focal_skill")
SUPPORTED_ARMS = (*ARMS, "all_gold_skills")
OUTPUT_LOCK = threading.Lock()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def prompt_for(
    instance: dict[str, Any],
    skill: dict[str, Any] | list[dict[str, Any]] | None,
) -> str:
    question = str(instance["question"])
    if skill is None:
        return question
    supplied = skill if isinstance(skill, list) else [skill]
    contents = "\n---\n".join(str(row["content"]) for row in supplied)
    return f"Relevant Skill:\n{contents}\n\n{question}"


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
            returned_model = row.get("returned_model") or row.get("returned_models")
            terminal_model_response = (
                row.get("error") is None
                and bool(returned_model)
                and any(
                    int(usage.get(name, 0) or 0) > 0
                    for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
            )
            if row.get("valid") is True or terminal_model_response:
                done.add((
                    str(row["instance_id"]),
                    str(row["arm"]),
                    int(row["repetition"]),
                ))
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


def verify_inputs(
    panel_path: Path,
    skills_path: Path,
    manifest_path: Path,
    panel_name: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    source_commit = (
        manifest.get("source_commit")
        if manifest.get("protocol") == "sra_bigcode_all_gold_confirmation_v1"
        else manifest.get("source", {}).get("commit")
    )
    if source_commit != UPSTREAM_COMMIT:
        raise SystemExit("manifest does not use the audited SR-Agents commit")
    if manifest.get("dataset") != "bigcodebench":
        raise SystemExit("this runner supports only BigCodeBench")
    panel_spec = manifest.get("panels", {}).get(panel_name)
    if not panel_spec or sha256(panel_path) != panel_spec.get("sha256"):
        raise SystemExit("panel does not match the frozen manifest")
    skill_spec = manifest.get("selected_skills_artifact", {})
    if sha256(skills_path) != skill_spec.get("sha256"):
        raise SystemExit("selected skills do not match the frozen manifest")
    panel = load_json(panel_path)
    skills = {str(row["skill_id"]): row for row in load_json(skills_path)}
    if len(panel) != int(panel_spec["tasks"]):
        raise SystemExit("panel task count does not match the manifest")
    for instance in panel:
        focal = str(instance.get("focal_skill_id", ""))
        if (
            instance.get("dataset") != "bigcodebench"
            or focal not in skills
            or focal not in instance.get("skill_annotations", [])
        ):
            raise SystemExit(f"invalid frozen instance {instance.get('instance_id')}")
    return panel, skills, manifest


def verify_audit_freeze(
    prediction_path: Path | None,
    panel: list[dict[str, Any]],
    panel_path: Path,
) -> str:
    if prediction_path is None:
        raise SystemExit("audit generation requires --audit-prediction-freeze")
    freeze = load_json(prediction_path)
    panel_ids = {str(row["instance_id"]) for row in panel}
    predictions = freeze.get("predictions", {})
    if freeze.get("audit", {}).get("outcomes_seen_at_freeze") != 0:
        raise SystemExit("audit prediction freeze is not outcome-blind")
    if freeze.get("audit", {}).get("panel_sha256") != sha256(panel_path):
        raise SystemExit("audit prediction freeze references another panel")
    if set(predictions) != panel_ids:
        raise SystemExit("audit prediction freeze does not cover the audit panel")
    return sha256(prediction_path)


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
        "--panel-name",
        choices=("support", "audit", "reserve", "confirmation", "full"),
        required=True,
    )
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--arms", nargs="+", choices=SUPPORTED_ARMS, default=list(ARMS)
    )
    parser.add_argument("--oracle-skills", type=Path)
    parser.add_argument("--oracle-probe-freeze", type=Path)
    parser.add_argument("--confirmation-prediction-freeze", type=Path)
    parser.add_argument("--audit-prediction-freeze", type=Path)
    parser.add_argument("--instance-ids", nargs="+")
    parser.add_argument("--api-base-env", default="QWEN_TARGET_OPENAI_COMPATIBLE_BASE_URL")
    parser.add_argument("--api-key-env", default="QWEN_TARGET_OPENAI_COMPATIBLE_API_KEY")
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
    parser.add_argument(
        "--allow-non-qwen-model",
        action="store_true",
        help="Run an explicitly named model in a separate replay output.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--workers", type=int, default=12)
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
    parser.add_argument("--quota-poll-every", type=int, default=25)
    parser.add_argument("--budget-state", type=Path)
    parser.add_argument("--call-ledger", type=Path)
    parser.add_argument("--expected-returned-model", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.workers <= 0 or args.repetitions <= 0 or args.request_retries < 0:
        raise SystemExit("workers/repetitions must be positive and retries nonnegative")
    if args.max_tokens not in (4096, 8192, 16384) or (
        args.max_tokens in (8192, 16384)
        and not (args.allow_non_qwen_model and args.reasoning_effort)
    ):
        raise SystemExit(
            "the frozen BigCodeBench protocol requires --max-tokens 4096; "
            "8192/16384 are reserved for explicit non-Qwen native profiles"
        )
    if args.temperature != 0:
        raise SystemExit("the frozen BigCodeBench protocol requires temperature 0")

    panel_path = args.panel.resolve()
    skills_path = args.skills.resolve()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    panel, skills, manifest = verify_inputs(
        panel_path, skills_path, manifest_path, args.panel_name
    )
    prediction_freeze_sha256 = None
    if args.panel_name == "audit":
        prediction_freeze_sha256 = verify_audit_freeze(
            args.audit_prediction_freeze, panel, panel_path
        )
    if args.panel_name == "reserve" and args.execute:
        raise SystemExit("reserve execution is not authorized by the frozen protocol")

    if args.instance_ids:
        requested = set(args.instance_ids)
        available = {str(row["instance_id"]) for row in panel}
        if len(requested) != len(args.instance_ids) or not requested <= available:
            raise SystemExit("--instance-ids must be a unique frozen-panel subset")
        panel = [row for row in panel if str(row["instance_id"]) in requested]

    requested_arms = tuple(dict.fromkeys(args.arms))
    oracle_skills: dict[str, dict[str, Any]] = {}
    oracle_probe_freeze_sha256 = None
    confirmation_prediction_freeze_sha256 = None
    if "all_gold_skills" in requested_arms:
        if args.oracle_skills is None:
            raise SystemExit("all_gold_skills requires --oracle-skills")
        if args.panel_name == "support" and args.oracle_probe_freeze is None:
            raise SystemExit(
                "support all_gold_skills requires --oracle-probe-freeze"
            )
        oracle_path = args.oracle_skills.resolve()
        if args.panel_name == "support":
            probe_freeze = load_json(args.oracle_probe_freeze.resolve())
            if probe_freeze.get("confirmatory_status") is not False:
                raise SystemExit("oracle probe freeze must be explicitly exploratory")
            if probe_freeze.get("outcomes_seen_at_freeze", {}).get(
                "all_gold_skills_cells"
            ) != 0:
                raise SystemExit("oracle probe was not frozen before its outcomes")
            if probe_freeze.get("source", {}).get(
                "support_panel_sha256"
            ) != sha256(panel_path):
                raise SystemExit("oracle probe references another support panel")
            if probe_freeze.get("oracle_skills_artifact", {}).get(
                "sha256"
            ) != sha256(oracle_path):
                raise SystemExit("oracle skills do not match the exploratory freeze")
            oracle_probe_freeze_sha256 = sha256(args.oracle_probe_freeze.resolve())
        elif args.panel_name == "confirmation":
            if args.confirmation_prediction_freeze is None:
                raise SystemExit(
                    "confirmation requires --confirmation-prediction-freeze"
                )
            prediction = load_json(args.confirmation_prediction_freeze.resolve())
            if (
                prediction.get("status")
                != "prospective confirmation predictions frozen"
                or prediction.get("audit", {}).get("outcomes_seen_at_freeze") != 0
                or prediction.get("audit", {}).get("panel_sha256")
                != sha256(panel_path)
                or set(prediction.get("predictions", {}))
                != {str(row["instance_id"]) for row in panel}
            ):
                raise SystemExit("confirmation prediction freeze is invalid")
            if manifest.get("all_gold_skills_artifact", {}).get(
                "sha256"
            ) != sha256(oracle_path):
                raise SystemExit("confirmation oracle skills mismatch")
            confirmation_prediction_freeze_sha256 = sha256(
                args.confirmation_prediction_freeze.resolve()
            )
        elif args.panel_name == "full":
            # The exhaustive benchmark run is a separate, explicitly named
            # collection protocol.  It is not part of the support/audit/
            # confirmation freezes above, so no outcome-blind prediction
            # freeze is required merely to collect the paired trajectories.
            if manifest.get("protocol") != "sra_bigcode_full_inventory_v1":
                raise SystemExit("full execution requires the full-inventory protocol")
            if manifest.get("all_gold_skills_artifact", {}).get(
                "sha256"
            ) != sha256(oracle_path):
                raise SystemExit("full oracle skills mismatch")
        else:
            raise SystemExit("all_gold_skills is not authorized for this panel")
        oracle_skills = {
            str(row["skill_id"]): row for row in load_json(oracle_path)
        }
        required = {
            str(skill_id)
            for instance in panel
            for skill_id in instance["skill_annotations"]
        }
        if not required <= set(oracle_skills):
            raise SystemExit("oracle skill artifact does not cover the panel")
    done = completed_keys(output_path)
    pending = [
        (instance, repetition, arm)
        for instance in panel
        for repetition in range(args.repetitions)
        for arm in requested_arms
        if (str(instance["instance_id"]), arm, repetition) not in done
    ]
    prompt_chars = {arm: 0 for arm in requested_arms}
    for instance in panel:
        focal = str(instance["focal_skill_id"])
        if "no_skill" in prompt_chars:
            prompt_chars["no_skill"] += len(prompt_for(instance, None))
        if "focal_skill" in prompt_chars:
            prompt_chars["focal_skill"] += len(prompt_for(instance, skills[focal]))
        if "all_gold_skills" in prompt_chars:
            prompt_chars["all_gold_skills"] += len(prompt_for(
                instance,
                [oracle_skills[str(skill_id)] for skill_id in instance["skill_annotations"]],
            ))
    already_completed = sum(
        (str(instance["instance_id"]), arm, repetition) in done
        for instance in panel
        for repetition in range(args.repetitions)
        for arm in requested_arms
    )
    plan = {
        "mode": "execute" if args.execute else "dry_run",
        "dataset": "bigcodebench",
        "panel": args.panel_name,
        "tasks": len(panel),
        "focal_skills": len({row["focal_skill_id"] for row in panel}),
        "arms": list(requested_arms),
        "repetitions": args.repetitions,
        "total_requests": len(panel) * len(requested_arms) * args.repetitions,
        "already_completed": already_completed,
        "pending_requests": len(pending),
        "prompt_chars_per_repetition": prompt_chars,
        "max_completion_tokens_if_all_pending": len(pending) * args.max_tokens,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens_per_request": args.max_tokens,
        "workers": args.workers,
        "request_retries": args.request_retries,
        "model": args.model,
        "max_public_cost_usd": args.max_public_cost_usd,
        "manifest_content_sha256": manifest.get("manifest_content_sha256"),
        "prediction_freeze_sha256": prediction_freeze_sha256,
        "oracle_probe_freeze_sha256": oracle_probe_freeze_sha256,
        "confirmation_prediction_freeze_sha256": (
            confirmation_prediction_freeze_sha256
        ),
        "evaluation_during_generation": False,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Dry run only: zero model API calls and zero code execution.")
        return

    if not args.allow_non_qwen_model and (
        args.model != "qwen-turbo" or args.expected_returned_model != "qwen-turbo"
    ):
        raise SystemExit("the frozen protocol requires requested and returned qwen-turbo")
    if args.allow_non_qwen_model and args.expected_returned_model != args.model:
        raise SystemExit("non-Qwen replay requires an exact expected returned model")
    if args.max_calls <= 0 or args.budget_state is None or args.call_ledger is None:
        raise SystemExit("execution requires explicit caps, budget state, and call ledger")
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
        (str(instance["instance_id"]), arm, repetition)
        for instance, repetition, arm in pending
    }
    jobs = []
    for instance in panel:
        for repetition in range(args.repetitions):
            for arm in requested_arm_order(
                str(instance["instance_id"]), repetition, args.seed, requested_arms
            ):
                if arm in requested_arms and (
                    str(instance["instance_id"]), arm, repetition
                ) in pending_keys:
                    jobs.append((instance, repetition, arm))

    budget_stop = threading.Event()
    identity_stop = threading.Event()

    def execute_job(job: tuple[dict[str, Any], int, str]) -> str:
        instance, repetition, arm = job
        instance_id = str(instance["instance_id"])
        focal = str(instance["focal_skill_id"])
        if budget_stop.is_set() or identity_stop.is_set():
            return "skipped"
        try:
            reservation = reserve_request_group()
        except ApiBudgetExceeded as exc:
            if not budget_stop.is_set():
                print(f"Budget stop before {instance_id}/{arm}: {exc}", file=sys.stderr)
            budget_stop.set()
            return "budget"

        treatment_skill_ids: list[str]
        if arm == "no_skill":
            treatment_skill_ids = []
            treatment = None
        elif arm == "focal_skill":
            treatment_skill_ids = [focal]
            treatment = skills[focal]
        else:
            treatment_skill_ids = [
                str(skill_id) for skill_id in instance["skill_annotations"]
            ]
            treatment = [oracle_skills[skill_id] for skill_id in treatment_skill_ids]
        prompt = prompt_for(instance, treatment)
        request_payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        extra_body = (
            None
            if args.reasoning_effort
            else reasoning_extra_body(args.model, thinking=args.thinking)
        )
        if extra_body is not None:
            request_payload["extra_body"] = extra_body
        if args.reasoning_effort:
            request_payload["reasoning_effort"] = args.reasoning_effort
        base_row: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "instance_id": instance_id,
            "dataset": "bigcodebench",
            "focal_skill_id": focal,
            "skill_annotations": instance["skill_annotations"],
            "treatment_skill_id": (
                treatment_skill_ids[0] if len(treatment_skill_ids) == 1 else None
            ),
            "treatment_skill_ids": treatment_skill_ids,
            "arm": arm,
            "repetition": repetition,
            "requested_model": args.model,
            "source_commit": UPSTREAM_COMMIT,
            "manifest_content_sha256": manifest.get("manifest_content_sha256"),
            "prediction_freeze_sha256": prediction_freeze_sha256,
            "oracle_probe_freeze_sha256": oracle_probe_freeze_sha256,
            "confirmation_prediction_freeze_sha256": (
                confirmation_prediction_freeze_sha256
            ),
            "request_sha256": hashlib.sha256(
                json.dumps(
                    request_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "temperature": args.temperature,
            "thinking": args.thinking,
            "reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
        }
        try:
            response = None
            for attempt in range(args.request_retries + 1):
                try:
                    response = client.chat.completions.create(**request_payload)
                    break
                except Exception as exc:
                    record_failure(
                        reservation,
                        stage=f"sra_bigcode_{arm}",
                        role="target",
                        model=args.model,
                        error_type=type(exc).__name__,
                    )
                    if attempt >= args.request_retries:
                        raise
                    time.sleep(min(8.0, 2.0**attempt))
                    reservation = reserve_request_group()
            assert response is not None
            raw_output = response.choices[0].message.content or ""
            returned_model = str(getattr(response, "model", "") or "")
            finish_reason = str(response.choices[0].finish_reason or "")
            usage = usage_dict(response)
            valid = bool(raw_output.strip())
            record_response(
                reservation,
                usage,
                stage=f"sra_bigcode_{arm}",
                role="target",
                model=args.model,
                returned_model=returned_model or None,
                empty=not valid,
            )
            error = None
            if returned_model != args.expected_returned_model:
                valid = False
                error = (
                    f"returned model {returned_model!r} != expected "
                    f"{args.expected_returned_model!r}"
                )
                identity_stop.set()
            append_jsonl(output_path, {
                **base_row,
                "returned_model": returned_model,
                "raw_output": raw_output,
                "finish_reason": finish_reason,
                "usage": usage,
                "valid": valid,
                "evaluated": False,
                "correct": None,
                "error": error,
            })
            return "ok" if valid else "invalid"
        except Exception as exc:  # noqa: BLE001
            append_jsonl(output_path, {
                **base_row,
                "valid": False,
                "evaluated": False,
                "correct": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
            return "failed"

    status_counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute_job, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            status_counts[status] = status_counts.get(status, 0) + 1
            if completed % 20 == 0 or completed == len(jobs):
                print(json.dumps({
                    "progress": completed,
                    "pending_at_start": len(jobs),
                    "status": status_counts,
                }, sort_keys=True), flush=True)
    print(json.dumps({"finished": status_counts}, sort_keys=True))


if __name__ == "__main__":
    main()
