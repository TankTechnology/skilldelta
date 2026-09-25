#!/usr/bin/env python3
"""Audit SRA-Bench reuse and freeze balanced paired-utility panels.

The first SkillDelta battlefield is LogicBench: each of its 19 gold skills is
mapped to exactly 40 distinct tasks.  This script verifies the public release,
selects skills without looking at outcomes, and writes disjoint pilot,
development, and final-audit panels plus a compact copy of only the selected
skills.

It is entirely offline and makes no model/API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRA_ROOT = ROOT / "experiments/data/SR-Agents"
DEFAULT_OUT = ROOT / "experiments/data/sra_bench/logicbench_seed20260827_v2"
EXPECTED_COMMIT = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
EXPECTED_SHA256 = {
    "corpus/corpus.json": "16ee509ae5bea8c2e17167dffecd89100a7d8dfa31256c3742426758c7169b5e",
    "instances/bigcodebench.json": "0ed01363e2c93134cf8696fea47b6d640f32f0d7ae9d7b478a05595c3f5ae788",
    "instances/champ.json": "d61346716cede953afb352e739e170b96d2bfb98824edd91b8783dc3526c7cec",
    "instances/logicbench.json": "af5055caac041ea08cee47622f5d922b47b2ba0a8e3a60b87349599eeff1bdfe",
    "instances/medcalcbench.json": "814f6b082f56cf89dd9c50be7ab87b5bcb0e41633138951e42b38d6923c3244e",
    "instances/theoremqa.json": "c969a7291e23361ba9f377e464be76093804deb628b964fb846c6eff6b28deeb",
    "instances/toolqa.json": "e41eb4d4b67c1dd8e1178c88e9e8ff15525b31803694e9459d115c274fb04f0e",
}


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


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def quantile(values: list[int], index: int) -> float:
    if len(values) == 1:
        return float(values[0])
    return float(statistics.quantiles(values, n=4, method="inclusive")[index])


def audit_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        skill_id for row in rows for skill_id in row.get("skill_annotations", [])
    )
    reuse = sorted(counts.values())
    return {
        "tasks": len(rows),
        "unique_skills": len(counts),
        "annotations": sum(reuse),
        "multi_skill_tasks": sum(
            len(row.get("skill_annotations", [])) > 1 for row in rows
        ),
        "reuse": {
            "min": min(reuse),
            "q25": quantile(reuse, 0),
            "median": float(statistics.median(reuse)),
            "q75": quantile(reuse, 2),
            "max": max(reuse),
            "mean": float(statistics.mean(reuse)),
        },
        "per_skill": dict(sorted(counts.items())),
    }


def shuffled(items: list[Any], seed_text: str) -> list[Any]:
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    result = list(items)
    random.Random(seed).shuffle(result)
    return result


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sra-root", type=Path, default=DEFAULT_SRA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", default="logicbench")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--num-skills", type=int, default=4)
    parser.add_argument("--pilot-per-skill", type=int, default=10)
    parser.add_argument("--development-per-skill", type=int, default=20)
    parser.add_argument("--audit-per-skill", type=int, default=10)
    args = parser.parse_args()

    sra_root = args.sra_root.resolve()
    bench_root = sra_root / "data/bench"
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output directory: {out}")

    commit = git_commit(sra_root)
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"SR-Agents commit mismatch: {commit} != {EXPECTED_COMMIT}")
    for relative, expected in EXPECTED_SHA256.items():
        path = bench_root / relative
        if not path.is_file():
            raise SystemExit(f"missing SRA-Bench artifact: {path}")
        observed = sha256(path)
        if observed != expected:
            raise SystemExit(f"SHA256 mismatch for {path}: {observed} != {expected}")

    corpus_path = bench_root / "corpus/corpus.json"
    corpus = load_json(corpus_path)
    corpus_by_id = {skill["skill_id"]: skill for skill in corpus}
    audit: dict[str, Any] = {}
    instance_files = sorted((bench_root / "instances").glob("*.json"))
    for path in instance_files:
        audit[path.stem] = audit_dataset(load_json(path))

    instance_path = bench_root / "instances" / f"{args.dataset}.json"
    rows: list[dict[str, Any]] = load_json(instance_path)
    if any(len(row.get("skill_annotations", [])) != 1 for row in rows):
        raise SystemExit(
            "the initial paired-panel builder requires exactly one gold skill per task"
        )
    by_skill: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_skill.setdefault(row["skill_annotations"][0], []).append(row)

    minimum = (
        args.pilot_per_skill
        + args.development_per_skill
        + args.audit_per_skill
    )
    eligible = sorted(
        skill_id for skill_id, skill_rows in by_skill.items()
        if len(skill_rows) >= minimum
    )
    selected_order = list(eligible)
    random.Random(args.seed).shuffle(selected_order)
    selected = selected_order[: args.num_skills]
    if len(selected) != args.num_skills:
        raise SystemExit(
            f"only {len(selected)} skills have at least {minimum} tasks; "
            f"requested {args.num_skills}"
        )

    pilot: list[dict[str, Any]] = []
    development: list[dict[str, Any]] = []
    final_audit: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    panel_assignments: dict[str, str] = {}
    for skill_id in selected:
        skill_rows = shuffled(
            sorted(by_skill[skill_id], key=lambda row: row["instance_id"]),
            f"{args.seed}:{skill_id}",
        )
        pilot_rows = skill_rows[: args.pilot_per_skill]
        development_start = args.pilot_per_skill
        audit_start = development_start + args.development_per_skill
        development_rows = skill_rows[development_start:audit_start]
        audit_rows = skill_rows[audit_start : audit_start + args.audit_per_skill]
        pilot.extend(pilot_rows)
        development.extend(development_rows)
        final_audit.extend(audit_rows)
        for row in pilot_rows:
            selected_ids.add(row["instance_id"])
            panel_assignments[row["instance_id"]] = "pilot"
        for row in development_rows:
            selected_ids.add(row["instance_id"])
            panel_assignments[row["instance_id"]] = "development"
        for row in audit_rows:
            selected_ids.add(row["instance_id"])
            panel_assignments[row["instance_id"]] = "audit"

    pilot.sort(key=lambda row: row["instance_id"])
    development.sort(key=lambda row: row["instance_id"])
    final_audit.sort(key=lambda row: row["instance_id"])
    selected_skills = [corpus_by_id[skill_id] for skill_id in selected]
    untouched_ids = sorted(row["instance_id"] for row in rows if row["instance_id"] not in selected_ids)

    out.mkdir(parents=True, exist_ok=False)
    panel_paths = {
        "pilot": out / "pilot.json",
        "development": out / "development.json",
        "audit": out / "audit.json",
    }
    selected_skills_path = out / "selected_skills.json"
    write_json(panel_paths["pilot"], pilot)
    write_json(panel_paths["development"], development)
    write_json(panel_paths["audit"], final_audit)
    write_json(selected_skills_path, selected_skills)
    write_json(out / "source_audit.json", audit)

    selected_skill_meta = {}
    for skill_id in selected:
        skill = corpus_by_id[skill_id]
        selected_skill_meta[skill_id] = {
            "name": skill.get("name", ""),
            "source_task_count": len(by_skill[skill_id]),
            "pilot_tasks": sum(
                row["skill_annotations"][0] == skill_id for row in pilot
            ),
            "development_tasks": sum(
                row["skill_annotations"][0] == skill_id for row in development
            ),
            "audit_tasks": sum(
                row["skill_annotations"][0] == skill_id for row in final_audit
            ),
            "content_chars": len(skill.get("content", "")),
            "content_sha256": hashlib.sha256(
                skill.get("content", "").encode("utf-8")
            ).hexdigest(),
        }

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "research_role": "paired task-conditional skill-utility battlefield",
        "leaderboard_comparable": False,
        "source": {
            "repository": "https://github.com/oneal2000/SR-Agents",
            "commit": commit,
            "instances": relative_or_absolute(instance_path, ROOT),
            "instances_sha256": sha256(instance_path),
            "corpus": relative_or_absolute(corpus_path, ROOT),
            "corpus_sha256": sha256(corpus_path),
        },
        "dataset": args.dataset,
        "selection": {
            "seed": args.seed,
            "eligible_skill_ids_sorted_then_seeded_shuffle": eligible,
            "selected_skill_ids_in_draw_order": selected,
            "num_skills": args.num_skills,
            "pilot_per_skill": args.pilot_per_skill,
            "development_per_skill": args.development_per_skill,
            "audit_per_skill": args.audit_per_skill,
            "outcomes_observed_during_selection": False,
        },
        "skills": selected_skill_meta,
        "arms": {
            "no_skill": "original task prompt only",
            "oracle_skill": "annotated gold skill content prepended using the official direct prompt",
        },
        "panels": {
            name: {
                "path": path.name,
                "tasks": len(load_json(path)),
                "paired_requests_per_repetition": 2 * len(load_json(path)),
                "sha256": sha256(path),
            }
            for name, path in panel_paths.items()
        },
        "selected_skills_artifact": {
            "path": selected_skills_path.name,
            "sha256": sha256(selected_skills_path),
        },
        "panel_assignments": panel_assignments,
        "untouched_instance_ids": untouched_ids,
        "protocol": {
            "unit": "task",
            "primary_contrast": "oracle_skill - no_skill",
            "arm_order": "deterministically randomized within task and repetition",
            "prompt_and_evaluator": f"SR-Agents {commit} LogicBench direct protocol",
            "pilot_calls_one_rep": 2 * len(pilot),
            "development_calls_one_rep": 2 * len(development),
            "audit_calls_one_rep": 2 * len(final_audit),
        },
        "manifest_content_sha256": None,
    }
    manifest["manifest_content_sha256"] = json_sha256(
        {**manifest, "manifest_content_sha256": None}
    )
    write_json(out / "manifest.json", manifest)

    print(json.dumps({
        "output": str(out),
        "dataset": args.dataset,
        "selected_skills": selected,
        "pilot_tasks": len(pilot),
        "pilot_requests_one_rep": 2 * len(pilot),
        "development_tasks": len(development),
        "development_requests_one_rep": 2 * len(development),
        "audit_tasks": len(final_audit),
        "audit_requests_one_rep": 2 * len(final_audit),
        "untouched_tasks": len(untouched_ids),
        "api_calls": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
