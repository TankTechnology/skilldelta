#!/usr/bin/env python3
"""Evaluate generated BigCodeBench code inside a bubblewrap sandbox.

The host process handles only bookkeeping.  Every answer is scored in a fresh,
networkless PID/user/mount namespace that can see a read-only Python environment,
the pinned SR-Agents evaluator, and one task payload.  The repository, home
directory, API environment, and generation ledger are not mounted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SRA_SRC = ROOT / "experiments/data/SR-Agents/src"
DEFAULT_ENV = Path(os.environ.get("SKILLDELTA_RUNTIME", sys.prefix))
DEFAULT_NLTK_DATA = Path(os.environ.get("NLTK_DATA", str(Path.home() / ".local/share/skilldelta-nltk-data")))
UPSTREAM_COMMIT = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
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


def worker() -> None:
    """Sandbox-only entry point; the input path is fixed by the mount profile."""
    payload = load_json(Path("/work/input.json"))
    instance = payload["instance"]
    raw_output = str(payload["raw_output"])
    from sragents.evaluate.common import strip_think_tags
    from sragents.evaluate.datasets.bigcodebench import _extract
    from sragents.evaluate.datasets.bigcodebench.execution import PASS, untrusted_check

    eval_data = instance["eval_data"]
    extracted = _extract(strip_think_tags(raw_output), eval_data)
    solution = (
        str(eval_data.get("code_prompt", "")) + "\n    pass\n" + extracted
        if eval_data.get("code_prompt")
        else extracted
    )
    stat, details = untrusted_check(
        solution,
        str(eval_data.get("test", "")),
        str(eval_data.get("entry_point", "")),
        max_as_limit=4 * 1024,
        max_data_limit=4 * 1024,
        max_stack_limit=10,
    )
    result = {
        "extracted_answer": extracted,
        "correct": stat == PASS,
        "result": stat,
        "details": details,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def sandbox_command(
    payload_path: Path,
    python_env: Path,
    evaluator_path: Path,
) -> list[str]:
    command = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop", "ALL",
        "--clearenv",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--dir", "/etc",
        "--ro-bind-try", "/etc/fonts", "/etc/fonts",
        "--ro-bind-try", "/etc/localtime", "/etc/localtime",
        "--ro-bind-try", "/etc/ssl", "/etc/ssl",
        "--dir", "/opt",
        "--dir", "/opt/skilldelta",
        "--dir", "/opt/sragents",
        "--ro-bind", str(python_env), "/opt/skilldelta-env",
        "--ro-bind", str(SRA_SRC), "/opt/sragents/src",
        "--ro-bind", str(evaluator_path), "/opt/skilldelta/evaluate_sra_bigcode.py",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--tmpfs", "/run",
        "--dir", "/home",
        "--dir", "/home/sandbox",
        "--dir", "/work",
        "--ro-bind", str(payload_path), "/work/input.json",
        "--chdir", "/work",
        "--setenv", "HOME", "/home/sandbox",
        "--setenv", "PATH", "/opt/skilldelta-env/bin:/usr/bin:/bin",
        "--setenv", "PYTHONPATH", "/opt/sragents/src",
        "--setenv", "PYTHONNOUSERSITE", "1",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "MPLBACKEND", "Agg",
        "--setenv", "MPLCONFIGDIR", "/tmp/matplotlib",
        "--setenv", "XDG_CACHE_HOME", "/tmp/cache",
        "--setenv", "NLTK_DATA", "/opt/nltk_data",
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "BIGCODEBENCH_TIMEOUT_PER_TASK", "30",
        "--setenv", "OMP_NUM_THREADS", "1",
        "--hostname", "skilldelta-bcb",
        "/opt/skilldelta-env/bin/python",
        "/opt/skilldelta/evaluate_sra_bigcode.py",
        "--worker",
    ]
    if DEFAULT_NLTK_DATA.exists():
        insertion = command.index("--proc")
        command[insertion:insertion] = [
            "--ro-bind", str(DEFAULT_NLTK_DATA), "/opt/nltk_data"
        ]
    return command


def child_limits() -> None:
    """Bound the whole sandbox before bubblewrap creates evaluator children."""
    # BigCodeBench itself applies a 30 GB address-space ceiling.  This stricter
    # outer guard prevents a malformed candidate from exhausting the host.
    memory = 6 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))


def sandbox_evaluate(
    instance: dict[str, Any],
    raw_output: str,
    python_env: Path,
    evaluator_path: Path,
    timeout: float,
) -> dict[str, Any]:
    payload = {"instance": instance, "raw_output": raw_output}
    with tempfile.TemporaryDirectory(prefix="skilldelta_bcb_payload_") as temporary:
        payload_path = Path(temporary) / "input.json"
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        try:
            completed = subprocess.run(
                sandbox_command(payload_path, python_env, evaluator_path),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=child_limits,
            )
        except subprocess.TimeoutExpired:
            return {
                "evaluated": True,
                "correct": False,
                "result": "outer_timeout",
                "evaluation_error": None,
            }
    if completed.returncode != 0:
        return {
            "evaluated": False,
            "correct": None,
            "result": "sandbox_error",
            "evaluation_error": (
                completed.stderr.strip()[-2000:]
                or f"bubblewrap exit {completed.returncode}"
            ),
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    result = None
    for line in reversed(lines):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "correct" in candidate:
            result = candidate
            break
    if result is None:
        return {
            "evaluated": False,
            "correct": None,
            "result": "sandbox_protocol_error",
            "evaluation_error": f"no evaluator JSON object in {len(lines)} lines",
        }
    return {
        "evaluated": True,
        "evaluation_error": None,
        "sandbox_auxiliary_stdout_lines": len(lines) - 1,
        **result,
    }


def environment_sha256(python_env: Path) -> str:
    python = python_env / "bin/python"
    completed = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    nltk_files = []
    if DEFAULT_NLTK_DATA.exists():
        for path in sorted(DEFAULT_NLTK_DATA.rglob("*")):
            if path.is_file():
                nltk_files.append((
                    str(path.relative_to(DEFAULT_NLTK_DATA)), sha256(path)
                ))
    fingerprint = {
        "python": subprocess.run(
            [str(python), "--version"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "packages": sorted(completed.stdout.splitlines()),
        "profile": "bubblewrap_unshare_all_networkless_readonly_env_v1",
        "sra_commit": UPSTREAM_COMMIT,
        "nltk_data_files": nltk_files,
    }
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_generations(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error") is not None:
                continue
            key = (str(row["instance_id"]), str(row["arm"]), int(row["repetition"]))
            usage = row.get("usage") or {}
            returned_model = row.get("returned_model") or row.get("returned_models")
            completed_model_response = (
                bool(returned_model)
                and any(
                    int(usage.get(name, 0) or 0) > 0
                    for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
            )
            if not completed_model_response or key in rows:
                continue
            rows[key] = row
    return rows


def completed_evaluations(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("evaluated") is True:
                completed.add((
                    str(row["instance_id"]),
                    str(row["arm"]),
                    int(row["repetition"]),
                ))
    return completed


def verify_inputs(
    manifest_path: Path, panel_path: Path, panel_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(manifest_path)
    panel_spec = manifest.get("panels", {}).get(panel_name, {})
    source_commit = (
        manifest.get("source_commit")
        if manifest.get("protocol") == "sra_bigcode_all_gold_confirmation_v1"
        else manifest.get("source", {}).get("commit")
    )
    if (
        manifest.get("dataset") != "bigcodebench"
        or source_commit != UPSTREAM_COMMIT
        or sha256(panel_path) != panel_spec.get("sha256")
    ):
        raise SystemExit("panel or manifest does not match the frozen study")
    panel = load_json(panel_path)
    if len(panel) != int(panel_spec.get("tasks", -1)):
        raise SystemExit("panel length mismatch")
    return manifest, panel


def main() -> None:
    if "--worker" in sys.argv:
        worker()
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--panel-name",
        choices=("support", "audit", "reserve", "confirmation", "full"),
        required=True,
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python-env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Score official reference solutions instead of model generations.",
    )
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--instance-ids", nargs="+")
    args = parser.parse_args()

    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if shutil.which("bwrap") is None:
        raise SystemExit("bubblewrap is required; refusing host code execution")
    python_env = args.python_env.resolve()
    evaluator_path = Path(__file__).resolve()
    if not (python_env / "bin/python").exists() or not SRA_SRC.exists():
        raise SystemExit("missing evaluator Python environment or pinned SRA source")
    manifest_path = args.manifest.resolve()
    panel_path = args.panel.resolve()
    manifest, panel = verify_inputs(manifest_path, panel_path, args.panel_name)
    if args.instance_ids:
        requested = set(args.instance_ids)
        available = {str(row["instance_id"]) for row in panel}
        if len(requested) != len(args.instance_ids) or not requested <= available:
            raise SystemExit("--instance-ids must be a unique panel subset")
        panel = [row for row in panel if str(row["instance_id"]) in requested]
    by_id = {str(row["instance_id"]): row for row in panel}
    env_hash = environment_sha256(python_env)

    if args.preflight:
        if args.input is not None or args.output is not None:
            raise SystemExit("--preflight does not accept --input/--output")
        output = (
            args.preflight_output.resolve()
            if args.preflight_output
            else panel_path.with_name(f"{args.panel_name}_sandbox_preflight.json")
        )
        jobs = [
            (
                str(row["instance_id"]),
                row,
                str(row["eval_data"]["code_prompt"])
                + str(row["eval_data"]["answer"]),
            )
            for row in panel
        ]
    else:
        if args.input is None or args.output is None:
            raise SystemExit("generation evaluation requires --input and --output")
        input_path = args.input.resolve()
        output = args.output.resolve()
        generations = read_generations(input_path)
        done = completed_evaluations(output)
        jobs = [
            (key, by_id[key[0]], row)
            for key, row in sorted(generations.items())
            if key[0] in by_id and key not in done
        ]
        expected = len(panel) * 2
        print(json.dumps({
            "mode": "model_generation_evaluation",
            "panel": args.panel_name,
            "panel_tasks": len(panel),
            "expected_paired_generations": expected,
            "available_valid_generations": len(generations),
            "pending_evaluations": len(jobs),
            "sandbox_profile": "bubblewrap_unshare_all_networkless_readonly_env_v1",
            "evaluator_environment_sha256": env_hash,
        }, indent=2))

    results: list[dict[str, Any]] = []

    def evaluate_job(job: Any) -> dict[str, Any]:
        if args.preflight:
            instance_id, instance, raw_output = job
            evaluated = sandbox_evaluate(
                instance, raw_output, python_env, evaluator_path, args.timeout
            )
            return {"instance_id": instance_id, **evaluated}
        key, instance, generation = job
        evaluated = sandbox_evaluate(
            instance,
            str(generation["raw_output"]),
            python_env,
            evaluator_path,
            args.timeout,
        )
        return {
            **generation,
            "generation_input_sha256": sha256(args.input.resolve()),
            "evaluator_environment_sha256": env_hash,
            **evaluated,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_job, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            if args.preflight:
                results.append(row)
            else:
                append_jsonl(output, row)
            if completed % 20 == 0 or completed == len(jobs):
                print(json.dumps({
                    "progress": completed,
                    "total": len(jobs),
                    "evaluated": sum(r.get("evaluated") is True for r in results)
                    if args.preflight else completed,
                }), flush=True)

    if args.preflight:
        results.sort(key=lambda row: row["instance_id"])
        report = {
            "schema_version": 1,
            "panel": args.panel_name,
            "panel_sha256": sha256(panel_path),
            "manifest_content_sha256": manifest.get("manifest_content_sha256"),
            "sandbox_profile": "bubblewrap_unshare_all_networkless_readonly_env_v1",
            "evaluator_environment_sha256": env_hash,
            "tasks": len(results),
            "evaluated": sum(row.get("evaluated") is True for row in results),
            "reference_pass": sum(row.get("correct") is True for row in results),
            "failures": [row for row in results if row.get("correct") is not True],
            "details": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "preflight_output": str(output),
            "tasks": report["tasks"],
            "evaluated": report["evaluated"],
            "reference_pass": report["reference_pass"],
            "failures": len(report["failures"]),
        }, indent=2))
        if report["failures"]:
            raise SystemExit("sandbox preflight failed; do not run model evaluation")


if __name__ == "__main__":
    main()
