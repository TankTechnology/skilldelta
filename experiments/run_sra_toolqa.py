#!/usr/bin/env python3
"""Run paired no-skill/oracle-skill trajectories on frozen ToolQA panels.

The runner uses SRA-Bench's ReAct agent, local ToolQA environment, and official
evaluator.  Every individual model step is admitted through the shared API
budget gate, while usage is also aggregated per trajectory.  Dry-run is the
default; live requests require ``--execute`` and explicit positive caps.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import string
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from api_budget import (
    ApiBudgetExceeded,
    record_failure,
    record_response,
    reserve_request_group,
)


ROOT = Path(__file__).resolve().parent.parent
SRA_ROOT = ROOT / "experiments/data/SR-Agents"
SRA_SRC = SRA_ROOT / "src"
if str(SRA_SRC) not in sys.path:
    sys.path.insert(0, str(SRA_SRC))

from sragents.infer.engines.react import ReActAgent  # noqa: E402
from sragents.toolqa import ToolEnvironment  # noqa: E402
from sragents.toolqa.fewshots import TOOLQA_EXAMPLES  # noqa: E402


def install_corpus_cache(cache_dir: Path) -> None:
    """Make the upstream retriever reuse a verified, API-free corpus index.

    The cache only replaces document encoding; query encoding still uses the
    upstream model and retrieval code.  When a cache is absent or does not
    match the frozen corpus, the original lazy implementation is retained.
    """
    try:
        import numpy as np
        from sragents.toolqa.tools import text as text_tools
    except ImportError:
        return

    cache_dir = cache_dir.resolve()
    original = text_tools.TextRetriever._ensure_index

    def cached_ensure_index(self: Any) -> None:
        if self._embeddings is not None:
            return
        cache_name = "agenda" if self.text_field == "event" else "scirex"
        cache_path = cache_dir / f"{cache_name}.npz"
        if not cache_path.is_file():
            original(self)
            return
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                expected_sha = str(data["corpus_sha256"].item())
                actual_sha = sha256(self.corpus_path)
                if expected_sha != actual_sha:
                    original(self)
                    return
                model_name = str(data["model_name"].item())
                if model_name != self.model_name:
                    original(self)
                    return
                self._texts = [str(item) for item in data["texts"].tolist()]
                self._embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            import sentence_transformers

            self._model = sentence_transformers.SentenceTransformer(self.model_name)
            print(f"  Loaded cached {cache_name} index from {cache_path}", flush=True)
        except Exception:
            # A malformed cache must never silently change benchmark behavior.
            self._embeddings = None
            original(self)

    text_tools.TextRetriever._ensure_index = cached_ensure_index


UPSTREAM_COMMIT = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
ARMS = ("no_skill", "oracle_skill")
OUTPUT_LOCK = threading.Lock()
COMPACT_FORMAT_EXAMPLE = """Question: What is 2 plus 3?
Thought 1: I should use the calculator.
Action 1: Calculate[2+3]
Observation 1: 5
Thought 2: I have the answer.
Action 2: Finish[5]"""


def evaluate_toolqa(raw_output: str, instance: dict[str, Any]) -> dict[str, Any]:
    """Pinned copy of SRA-Bench's ToolQA normalized exact-match evaluator."""
    matches = re.findall(r"Finish\[([^\]]*)\]", raw_output)
    if matches:
        extracted = matches[-1].strip()
    else:
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        extracted = lines[-1] if lines else ""

    def normalize(value: str) -> str:
        value = re.sub(r"\b(a|an|the|usd)\b", " ", value.lower())
        value = "".join(ch for ch in value if ch not in set(string.punctuation))
        return " ".join(value.split())

    ground_truth = str(instance["eval_data"]["answer"]).strip()
    if normalize(extracted) == normalize(ground_truth):
        return {"extracted_answer": extracted, "correct": True, "match_type": "exact"}
    try:
        if abs(float(extracted) - float(ground_truth)) < 1e-6:
            return {"extracted_answer": extracted, "correct": True, "match_type": "numeric"}
    except (TypeError, ValueError):
        pass
    mapped = {"true": "yes", "false": "no"}.get(extracted.lower())
    if mapped is not None and mapped == ground_truth.lower():
        return {"extracted_answer": extracted, "correct": True, "match_type": "boolean"}
    return {"extracted_answer": extracted, "correct": False, "match_type": "none"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    prompt = int(
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", 0)
        or 0
    )
    completion = int(
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", 0)
        or 0
    )
    total = int(getattr(usage, "total_tokens", 0) or prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("valid") is True:
            done.add((row["instance_id"], row["arm"], int(row["repetition"])))
    return done


def arm_order(instance_id: str, repetition: int, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{seed}:{instance_id}:{repetition}".encode("utf-8")
    ).digest()
    return ARMS if digest[0] % 2 == 0 else tuple(reversed(ARMS))


def verify_inputs(
    manifest_path: Path,
    panel_path: Path,
    skills_path: Path,
    panel_name: str,
    external_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = read_json(manifest_path)
    content_hash = manifest.get("manifest_content_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_content_sha256", None)
    if json_sha256(unhashed) != content_hash:
        raise SystemExit("manifest content SHA256 is invalid")
    if manifest.get("source", {}).get("commit") != UPSTREAM_COMMIT:
        raise SystemExit("manifest does not use the audited SR-Agents commit")
    if manifest.get("dataset") != "toolqa" or manifest.get("subset") not in {
        "table_workflows",
        "all_domains",
    }:
        raise SystemExit(
            "runner only accepts a frozen ToolQA table-workflows or all-domains manifest"
        )
    panel_spec = manifest.get("panels", {}).get(panel_name)
    if not panel_spec or sha256(panel_path) != panel_spec.get("sha256"):
        raise SystemExit("panel does not match the frozen manifest")
    skill_spec = manifest.get("selected_skills_artifact", {})
    if sha256(skills_path) != skill_spec.get("sha256"):
        raise SystemExit("selected skills do not match the frozen manifest")
    for relative, expected_size in manifest.get("external_files", {}).items():
        path = external_root / relative
        if not path.is_file() or path.stat().st_size != int(expected_size):
            raise SystemExit(f"external ToolQA file missing or wrong size: {relative}")

    panel = read_json(panel_path)
    skills = {row["skill_id"]: row for row in read_json(skills_path)}
    for row in panel:
        annotations = row.get("skill_annotations", [])
        if row.get("dataset") != "toolqa" or len(annotations) != 1:
            raise SystemExit(f"invalid ToolQA row: {row.get('instance_id')}")
        if annotations[0] not in skills:
            raise SystemExit(f"panel skill is outside the frozen subset: {annotations[0]}")
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
    os.environ["SKILLDELTA_QUOTA_FAIL_CLOSED"] = str(args.quota_fail_closed)
    os.environ["TARGET_OPENAI_COMPATIBLE_BASE_URL"] = args.api_base
    os.environ["TARGET_OPENAI_COMPATIBLE_API_KEY"] = args.api_key


class BudgetedCompletions:
    """Per-trajectory proxy around a shared OpenAI completions client."""

    def __init__(
        self,
        base: Any,
        *,
        model: str,
        arm: str,
        expected_returned_model: str,
        temperature: float,
        request_retries: int,
        budget_stop: threading.Event,
        identity_stop: threading.Event,
    ) -> None:
        self.base = base
        self.model = model
        self.arm = arm
        self.expected_returned_model = expected_returned_model
        self.temperature = temperature
        self.request_retries = request_retries
        self.budget_stop = budget_stop
        self.identity_stop = identity_stop
        self.step_usage: list[dict[str, int]] = []
        self.request_sha256: list[str] = []
        self.returned_models: list[str] = []

    def create(self, **kwargs: Any) -> Any:
        if self.budget_stop.is_set():
            raise ApiBudgetExceeded("global ToolQA budget stop is active")
        if self.identity_stop.is_set():
            raise RuntimeError("global model-identity stop is active")
        try:
            reservation = reserve_request_group()
        except ApiBudgetExceeded:
            self.budget_stop.set()
            raise

        request = dict(kwargs)
        request["temperature"] = self.temperature
        fingerprint = hashlib.sha256(
            json.dumps(
                request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        self.request_sha256.append(fingerprint)
        response = None
        for attempt in range(self.request_retries + 1):
            try:
                response = self.base.create(**request)
                break
            except Exception as exc:
                record_failure(
                    reservation,
                    stage=f"sra_toolqa_{self.arm}",
                    role="target_step",
                    model=self.model,
                    error_type=type(exc).__name__,
                )
                if attempt >= self.request_retries:
                    raise
                # The shared gateway can transiently report no available GLM
                # channel. Retries are opt-in and used only by diagnostics.
                time.sleep(min(8.0, 2.0**attempt))
                if self.budget_stop.is_set() or self.identity_stop.is_set():
                    raise
                try:
                    reservation = reserve_request_group()
                except ApiBudgetExceeded:
                    self.budget_stop.set()
                    raise
        assert response is not None

        usage = usage_dict(response)
        returned_model = str(getattr(response, "model", "") or "")
        content = response.choices[0].message.content or ""
        record_response(
            reservation,
            usage,
            stage=f"sra_toolqa_{self.arm}",
            role="target_step",
            model=self.model,
            returned_model=returned_model or None,
            empty=not bool(content.strip()),
        )
        self.step_usage.append(usage)
        self.returned_models.append(returned_model)
        if (
            self.expected_returned_model
            and returned_model != self.expected_returned_model
        ):
            self.identity_stop.set()
            raise RuntimeError(
                f"returned model {returned_model!r} != expected "
                f"{self.expected_returned_model!r}"
            )
        return response

    def aggregate_usage(self) -> dict[str, int]:
        return {
            key: sum(step.get(key, 0) for step in self.step_usage)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }


class BudgetedClient:
    def __init__(self, completions: BudgetedCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class GuardedToolEnvironment(ToolEnvironment):
    """ToolQA environment with a wall-clock guard on generated SQLite queries."""

    def __init__(self, corpus_dir: Path, sql_timeout_seconds: float) -> None:
        super().__init__(corpus_dir)
        self.sql_timeout_seconds = sql_timeout_seconds

    def _dispatch(self, action_type: str, argument: str) -> str:
        if action_type != "SQLInterpreter":
            return super()._dispatch(action_type, argument)
        deadline = time.monotonic() + self.sql_timeout_seconds
        connection = self.sql_conn
        connection.set_progress_handler(
            lambda: int(time.monotonic() >= deadline), 10_000
        )
        try:
            return super()._dispatch(action_type, argument)
        finally:
            connection.set_progress_handler(None, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--panel-name", choices=("harness", "support", "audit", "reserve", "full"), required=True
    )
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument(
        "--corpus-cache-dir",
        type=Path,
        help="Optional verified cache of the upstream Agenda/Scirex document embeddings.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen-turbo")
    parser.add_argument(
        "--allow-non-qwen-model",
        action="store_true",
        help=(
            "Allow an explicitly selected target model for a separate model-shift "
            "diagnostic; the frozen headline study remains qwen-turbo by default."
        ),
    )
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--instance-ids", nargs="+")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable the upstream model's reasoning mode (used by GLM diagnostics).",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=0,
        help="Retries for transient gateway failures (0 preserves the original protocol).",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--step-max-tokens", type=int, default=384)
    parser.add_argument("--sql-timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--prompt-profile", choices=("compact", "official"), default="compact"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--api-base-env", default="QWEN_TARGET_OPENAI_COMPATIBLE_BASE_URL")
    parser.add_argument("--api-key-env", default="QWEN_TARGET_OPENAI_COMPATIBLE_API_KEY")
    parser.add_argument("--expected-returned-model", default="")
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--max-quota-delta", type=int, default=0)
    parser.add_argument("--max-public-cost-usd", type=float, default=0.0)
    parser.add_argument("--input-usd-per-million-tokens", type=float, default=0.05)
    parser.add_argument("--output-usd-per-million-tokens", type=float, default=0.20)
    parser.add_argument("--quota-poll-every", type=int, default=25)
    parser.add_argument(
        "--quota-fail-closed",
        choices=("0", "1"),
        default="1",
        help="stop on an unavailable quota snapshot (default: 1)",
    )
    parser.add_argument("--budget-state", type=Path)
    parser.add_argument("--call-ledger", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if (
        args.workers <= 0
        or args.max_steps <= 0
        or args.step_max_tokens <= 0
        or args.sql_timeout_seconds <= 0
        or args.request_retries < 0
    ):
        raise SystemExit(
            "workers, max-steps, step-max-tokens, and sql-timeout must be positive; "
            "request-retries must be nonnegative"
        )
    manifest_path = args.manifest.resolve()
    panel_path = args.panel.resolve()
    skills_path = args.skills.resolve()
    external_root = args.external_root.resolve()
    output_path = args.output.resolve()
    panel, skills, manifest = verify_inputs(
        manifest_path, panel_path, skills_path, args.panel_name, external_root
    )
    if args.corpus_cache_dir is not None:
        install_corpus_cache(args.corpus_cache_dir)
    if args.instance_ids:
        requested = set(args.instance_ids)
        if len(requested) != len(args.instance_ids):
            raise SystemExit("--instance-ids contains duplicates")
        available = {row["instance_id"] for row in panel}
        if not requested.issubset(available):
            raise SystemExit("--instance-ids is not a subset of the panel")
        panel = [row for row in panel if row["instance_id"] in requested]

    requested_arms = tuple(dict.fromkeys(args.arms))
    done = completed_keys(output_path)
    pending = [
        (row, repetition, arm)
        for row in panel
        for repetition in range(args.repetitions)
        for arm in requested_arms
        if (row["instance_id"], arm, repetition) not in done
    ]
    examples = TOOLQA_EXAMPLES if args.prompt_profile == "official" else COMPACT_FORMAT_EXAMPLE
    first_call_chars = {arm: 0 for arm in requested_arms}
    for row in panel:
        for arm in requested_arms:
            skill_chars = (
                len(skills[row["skill_annotations"][0]]["content"])
                if arm == "oracle_skill"
                else 0
            )
            first_call_chars[arm] += len(examples) + len(row["question"]) + skill_chars

    subset = str(manifest.get("subset"))
    plan = {
        "mode": "execute" if args.execute else "dry_run",
        "dataset": "toolqa",
        "subset": subset,
        "panel": args.panel_name,
        "tasks": len(panel),
        "skills": sorted({row["skill_annotations"][0] for row in panel}),
        "arms": list(requested_arms),
        "repetitions": args.repetitions,
        "trajectories": len(panel) * len(requested_arms) * args.repetitions,
        "already_completed": len(panel) * len(requested_arms) * args.repetitions - len(pending),
        "pending_trajectories": len(pending),
        "maximum_pending_model_calls": len(pending) * args.max_steps,
        "first_call_user_chars_per_repetition": first_call_chars,
        "prompt_profile": args.prompt_profile,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "request_retries": args.request_retries,
        "max_steps": args.max_steps,
        "step_max_tokens": args.step_max_tokens,
        "sql_timeout_seconds": args.sql_timeout_seconds,
        "corpus_cache_dir": str(args.corpus_cache_dir.resolve())
        if args.corpus_cache_dir is not None
        else None,
        "workers": args.workers,
        "model": args.model,
        "allow_non_qwen_model": args.allow_non_qwen_model,
        "max_public_cost_usd": args.max_public_cost_usd,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("Dry run only: zero model API calls.")
        return

    if args.model != "qwen-turbo" and not args.allow_non_qwen_model:
        raise SystemExit("the frozen main study requires model=qwen-turbo")
    if args.max_calls <= 0 or args.max_total_tokens <= 0:
        raise SystemExit("--execute requires positive call and token caps")
    if args.max_public_cost_usd <= 0:
        raise SystemExit("--execute requires a positive public-price cost cap")
    if args.budget_state is None or args.call_ledger is None:
        raise SystemExit("--execute requires --budget-state and --call-ledger")
    args.api_base = os.environ.get(args.api_base_env, "").strip()
    args.api_key = os.environ.get(args.api_key_env, "").strip()
    if not args.api_base or not args.api_key:
        raise SystemExit(f"missing {args.api_base_env} or {args.api_key_env}")
    configure_budget(args)

    from openai import OpenAI

    base_client = OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
        max_retries=0,
        timeout=args.timeout,
    )
    tool_local = threading.local()

    def get_tools() -> ToolEnvironment:
        if not hasattr(tool_local, "environment"):
            tool_local.environment = GuardedToolEnvironment(
                external_root, args.sql_timeout_seconds
            )
        else:
            tool_local.environment.reset()
        return tool_local.environment

    budget_stop = threading.Event()
    identity_stop = threading.Event()

    def run_job(job: tuple[dict[str, Any], int, str]) -> tuple[str, tuple[str, str, int]]:
        row, repetition, arm = job
        key = (row["instance_id"], arm, repetition)
        if budget_stop.is_set():
            return "skipped_after_budget", key
        if identity_stop.is_set():
            return "skipped_after_identity", key
        skill_id = row["skill_annotations"][0]
        injected = [skills[skill_id]["content"]] if arm == "oracle_skill" else None
        completions = BudgetedCompletions(
            base_client.chat.completions,
            model=args.model,
            arm=arm,
            expected_returned_model=args.expected_returned_model,
            temperature=args.temperature,
            request_retries=args.request_retries,
            budget_stop=budget_stop,
            identity_stop=identity_stop,
        )
        client = BudgetedClient(completions)
        agent = ReActAgent(
            question=row["question"],
            tools=get_tools(),
            client=client,
            model=args.model,
            examples=examples,
            max_steps=args.max_steps,
            max_tokens=args.step_max_tokens,
            skills=injected,
            thinking=args.thinking,
        )
        base_row = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "instance_id": row["instance_id"],
            "dataset": "toolqa",
            "subset": subset,
            "panel": args.panel_name,
            "skill_id": skill_id,
            "gold_skill_id": skill_id,
            "treatment_skill_id": skill_id if arm == "oracle_skill" else None,
            "arm": arm,
            "repetition": repetition,
            "requested_model": args.model,
            "source_commit": UPSTREAM_COMMIT,
            "manifest_content_sha256": manifest["manifest_content_sha256"],
            "prompt_profile": args.prompt_profile,
            "temperature": args.temperature,
            "thinking": args.thinking,
            "request_retries": args.request_retries,
            "max_steps": args.max_steps,
            "step_max_tokens": args.step_max_tokens,
            "sql_timeout_seconds": args.sql_timeout_seconds,
            "corpus_cache_dir": (
                str(args.corpus_cache_dir.resolve())
                if args.corpus_cache_dir is not None
                else None
            ),
        }
        try:
            agent.run()
            raw_output = agent.model_scratchpad
            evaluation = evaluate_toolqa(raw_output, row)
            valid = bool(raw_output.strip()) and bool(completions.step_usage)
            append_jsonl(
                output_path,
                {
                    **base_row,
                    "returned_models": completions.returned_models,
                    "request_sha256": completions.request_sha256,
                    "raw_output": raw_output,
                    "transcript": agent.scratchpad,
                    "usage": completions.aggregate_usage(),
                    "step_usage": completions.step_usage,
                    "n_steps": agent.step_n - 1,
                    "finished": agent.finished,
                    "halted": agent.is_halted(),
                    "valid": valid,
                    "error": None,
                    "ground_truth": row["eval_data"]["answer"],
                    **evaluation,
                },
            )
            return "ok" if valid else "invalid", key
        except ApiBudgetExceeded as exc:
            budget_stop.set()
            append_jsonl(
                output_path,
                {
                    **base_row,
                    "valid": False,
                    "correct": None,
                    "usage": completions.aggregate_usage(),
                    "step_usage": completions.step_usage,
                    "raw_output": agent.model_scratchpad,
                    "transcript": agent.scratchpad,
                    "error": f"ApiBudgetExceeded: {exc}",
                },
            )
            return "budget", key
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                output_path,
                {
                    **base_row,
                    "valid": False,
                    "correct": None,
                    "usage": completions.aggregate_usage(),
                    "step_usage": completions.step_usage,
                    "raw_output": agent.model_scratchpad,
                    "transcript": agent.scratchpad,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return "failed", key

    pending_keys = {(row["instance_id"], arm, rep) for row, rep, arm in pending}
    jobs = []
    for row in panel:
        for repetition in range(args.repetitions):
            ordered = arm_order(row["instance_id"], repetition, args.seed)
            for arm in ordered:
                if arm in requested_arms and (row["instance_id"], arm, repetition) in pending_keys:
                    jobs.append((row, repetition, arm))

    statuses: dict[str, int] = {}
    if args.workers == 1:
        outcomes = (run_job(job) for job in jobs)
        for completed, (status, _) in enumerate(outcomes, 1):
            statuses[status] = statuses.get(status, 0) + 1
            print(json.dumps({"progress": completed, "total": len(jobs), "status": statuses}), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                status, _ = future.result()
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
