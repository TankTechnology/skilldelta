#!/usr/bin/env python3
"""Freeze a full-benchmark MedCalc support/audit replication.

Every one of the 55 single-skill families contributes eight support tasks and
twelve untouched audit tasks.  Selection depends only on family identity and
a deterministic seed; no model result is read to construct either panel.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SRA_ROOT = ROOT / "experiments/data/SR-Agents"
DEFAULT_INSTANCES = SRA_ROOT / "data/bench/instances/medcalcbench.json"
DEFAULT_CORPUS = SRA_ROOT / "data/bench/corpus/corpus.json"
DEFAULT_OUTPUT = ROOT / "experiments/data/sra_bench/medcalc_seed20260828"
DEFAULT_PROTOCOL = DEFAULT_OUTPUT / "protocol-freeze.json"
DEFAULT_RUN_ROOT = ROOT / "experiments/outputs/sra_medcalc"
EXPECTED_INSTANCES_SHA256 = (
    "814f6b082f56cf89dd9c50be7ab87b5bcb0e41633138951e42b38d6923c3244e"
)
SOURCE_COMMIT = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
SEED = 20260828
SUPPORT_PER_SKILL = 8
AUDIT_PER_SKILL = 12


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def result_rows(path: Path, target_ids: set[str]) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("valid") is True and str(row.get("instance_id")) in target_ids:
            count += 1
    return count


def audit_tools(skills: list[dict[str, Any]]) -> dict[str, Any]:
    imports: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    total = 0
    for skill in skills:
        tools = skill.get("tools", [])
        if not tools:
            raise SystemExit(f"MedCalc skill has no executable tool: {skill['skill_id']}")
        for tool in tools:
            total += 1
            try:
                tree = ast.parse(str(tool["implementation"]))
            except SyntaxError as exc:
                parse_errors.append(
                    {
                        "skill_id": str(skill["skill_id"]),
                        "tool": str(tool.get("name")),
                        "error": str(exc),
                    }
                )
                continue
            statements = [
                ast.unparse(node)
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            if statements:
                imports.append(
                    {
                        "skill_id": str(skill["skill_id"]),
                        "tool": str(tool["name"]),
                        "statements": statements,
                    }
                )
    if parse_errors:
        raise SystemExit(f"unparseable MedCalc tools: {parse_errors}")
    return {
        "skills_with_tools": len(skills),
        "total_tools": total,
        "syntax_parse_errors": 0,
        "tools_with_import_statements": len(imports),
        "importing_tools": imports,
        "repair": (
            "runner strips only allowlisted math/datetime imports and preloads "
            "math, datetime, and timedelta in the restricted namespace"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol-freeze", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--support-per-skill", type=int, default=SUPPORT_PER_SKILL)
    args = parser.parse_args()

    instances_path = args.instances.resolve()
    corpus_path = args.corpus.resolve()
    output = args.output_dir.resolve()
    run_root = args.run_root.resolve()
    if sha256(instances_path) != EXPECTED_INSTANCES_SHA256:
        raise SystemExit("MedCalc instances do not match the audited upstream file")
    instances = read_json(instances_path)
    if len(instances) != 1100 or any(
        row.get("dataset") != "medcalcbench"
        or len(row.get("skill_annotations", [])) != 1
        for row in instances
    ):
        raise SystemExit("unexpected MedCalc instance schema or size")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        grouped[str(row["skill_annotations"][0])].append(row)
    family_sizes = Counter({skill_id: len(rows) for skill_id, rows in grouped.items()})
    if len(grouped) != 55 or set(family_sizes.values()) != {20}:
        raise SystemExit(f"expected 55 exactly balanced 20-task families: {family_sizes}")
    if args.support_per_skill <= 0 or args.support_per_skill >= 20:
        raise SystemExit("support-per-skill must be between 1 and 19")

    corpus = read_json(corpus_path)
    corpus_by_id = {str(row["skill_id"]): row for row in corpus}
    skills = [corpus_by_id[skill_id] for skill_id in sorted(grouped)]
    if len(skills) != 55:
        raise SystemExit("missing MedCalc gold skills in the corpus")
    tool_audit = audit_tools(skills)

    support: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    membership: dict[str, dict[str, list[str]]] = {}
    for skill_id in sorted(grouped):
        rows = sorted(grouped[skill_id], key=lambda row: str(row["instance_id"]))
        family_seed = int(
            hashlib.sha256(f"{args.seed}:{skill_id}".encode()).hexdigest()[:16], 16
        )
        random.Random(family_seed).shuffle(rows)
        support_rows = rows[: args.support_per_skill]
        audit_rows = rows[args.support_per_skill :]
        support.extend(support_rows)
        audit.extend(audit_rows)
        membership[skill_id] = {
            "support": sorted(str(row["instance_id"]) for row in support_rows),
            "audit": sorted(str(row["instance_id"]) for row in audit_rows),
        }
    support.sort(key=lambda row: str(row["instance_id"]))
    audit.sort(key=lambda row: str(row["instance_id"]))
    support_ids = {str(row["instance_id"]) for row in support}
    audit_ids = {str(row["instance_id"]) for row in audit}
    if support_ids & audit_ids or support_ids | audit_ids != {
        str(row["instance_id"]) for row in instances
    }:
        raise SystemExit("support/audit partition is not disjoint and exhaustive")
    if len(support) != 440 or len(audit) != 660:
        raise SystemExit("the fixed 8/12 split must produce 440/660 tasks")

    existing_support = result_rows(run_root / "support_results.jsonl", support_ids)
    existing_audit = result_rows(run_root / "audit_results.jsonl", audit_ids)
    if existing_support or existing_audit:
        raise SystemExit(
            "refusing to rewrite the freeze after valid MedCalc outcomes exist: "
            f"support={existing_support}, audit={existing_audit}"
        )

    artifacts: dict[str, Any] = {}
    for name, rows in (("support", support), ("audit", audit)):
        path = output / f"{name}.json"
        write_json(path, rows)
        artifacts[name] = {
            "path": str(path.relative_to(ROOT)),
            "tasks": len(rows),
            "sha256": sha256(path),
            "content_sha256": json_sha256(rows),
        }
    skills_path = output / "selected_skills.json"
    write_json(skills_path, skills)

    manifest = {
        "schema_version": 1,
        "protocol": "sra_medcalc_full_benchmark_dense_sign_v1",
        "frozen_date": "2026-08-28",
        "dataset": "medcalcbench",
        "seed": args.seed,
        "source": {
            "repository": "https://github.com/oneal2000/SR-Agents",
            "commit": SOURCE_COMMIT,
            "instances_path": str(instances_path.relative_to(ROOT)),
            "instances_sha256": sha256(instances_path),
            "corpus_path": str(corpus_path.relative_to(ROOT)),
            "corpus_sha256": sha256(corpus_path),
        },
        "selection": {
            "rule": "all 55 families; deterministic 8 support / 12 audit per family",
            "skill_families": 55,
            "tasks_per_family": 20,
            "support_per_family": args.support_per_skill,
            "audit_per_family": 20 - args.support_per_skill,
            "no_family_or_task_exclusion": True,
        },
        "panels": artifacts,
        "selected_skills_artifact": {
            "path": str(skills_path.relative_to(ROOT)),
            "sha256": sha256(skills_path),
            "content_sha256": json_sha256(skills),
        },
        "tool_audit": tool_audit,
        "per_skill": membership,
        "leakage_controls": [
            "all families are included",
            "membership uses only family identity and a deterministic seed",
            "no model result is read by panel construction",
            "audit embeddings may read question text but never audit outcomes",
            "audit predictions must be hash-frozen before audit execution",
        ],
    }
    manifest["manifest_content_sha256"] = json_sha256(manifest)
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)

    protocol = {
        "schema_version": 1,
        "status": "support and audit outcomes sealed",
        "frozen_date": "2026-08-28",
        "research_questions": {
            "RQ2": "pre-execution positive-versus-nonpositive gain prediction",
            "RQ3": "success/token value of the frozen embedding gate",
        },
        "panel": {
            "benchmark": "SRA-Bench MedCalc-Bench",
            "all_tasks": 1100,
            "all_skill_families": 55,
            "support_tasks": len(support),
            "audit_tasks": len(audit),
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256(manifest_path),
            "support_sha256": artifacts["support"]["sha256"],
            "audit_sha256": artifacts["audit"]["sha256"],
            "support_outcomes_seen_at_freeze": existing_support,
            "audit_outcomes_seen_at_freeze": existing_audit,
        },
        "fixed_system": {
            "model": "qwen-turbo",
            "returned_model_must_equal": "qwen-turbo",
            "temperature": 0,
            "arms": ["no_skill", "oracle_skill"],
            "repetitions_per_arm": 1,
            "prompt_and_evaluator": "official SR-Agents MedCalc",
            "max_model_rounds_per_trajectory": 5,
            "max_tokens_per_round": 512,
            "tool_execution": "allowlisted math/datetime import repair",
        },
        "support_viability_gate": {
            "valid_paired_cells": "at least 98% before exact-cell resume; complete pairs required for freeze",
            "global_positive_prevalence": "between 0.10 and 0.90 inclusive",
            "families_containing_positive": ">=10",
            "families_containing_nonpositive": ">=10",
            "failure_action": "do not execute audit; report an always-on/off boundary regime",
        },
        "fixed_predictor": {
            "embedding_model": "text-embedding-3-small",
            "embedding_input": "full question text only",
            "normalization": "L2",
            "candidate_pool": "support tasks with the same gold skill family",
            "similarity": "cosine",
            "k": 3,
            "score": "nonnegative-cosine-weighted mean of signed paired gains",
            "decision_threshold": "strictly above global support positive prevalence",
            "audit_label_tuning": False,
        },
        "primary_metrics": {
            "RQ2": ["AUROC", "balanced accuracy", "direction accuracy"],
            "RQ3": [
                "Always-off / embedding-gate / Always-on success",
                "reported-token saving versus Always-on",
            ],
        },
        "uncertainty": [
            "10000-replicate task bootstrap",
            "10000-replicate whole-skill-family cluster bootstrap",
        ],
        "execution_caps": {
            "support": {
                "maximum_model_calls": 2640,
                "reported_tokens": 8000000,
                "proxy_quota_units": 75000,
                "public_price_usd": 0.25,
            },
            "audit": {
                "maximum_model_calls": 3960,
                "reported_tokens": 12000000,
                "proxy_quota_units": 175000,
                "public_price_usd": 0.50,
            },
        },
        "claim_boundary": (
            "A successful audit establishes cross-domain replication inside "
            "SRA-Bench, conditional on the supplied gold skill family; it does "
            "not establish retrieval or unseen-family transfer."
        ),
    }
    write_json(args.protocol_freeze.resolve(), protocol)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "manifest_sha256": sha256(manifest_path),
                "protocol_sha256": sha256(args.protocol_freeze.resolve()),
                "support_tasks": len(support),
                "audit_tasks": len(audit),
                "skills": len(skills),
                "tool_audit": tool_audit,
                "paid_api_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
