#!/usr/bin/env python3
"""Freeze the complete official ToolQA all-domain inventory.

This manifest is deliberately independent from the earlier 830-task
table-workflow study.  It contains every official ToolQA instance and every
ToolQA skill annotation under one fixed protocol so that the resulting paired
trajectories can be evaluated as a separate all-domain benchmark.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SRA_ROOT = ROOT / "experiments/data/SR-Agents"
OUT = ROOT / "experiments/data/sra_bench/toolqa_all_seed20260829"
EXPECTED_COMMIT = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
EXPECTED_INSTANCE_SHA256 = (
    "e41eb4d4b67c1dd8e1178c88e9e8ff15525b31803694e9459d115c274fb04f0e"
)
EXPECTED_CORPUS_SHA256 = (
    "16ee509ae5bea8c2e17167dffecd89100a7d8dfa31256c3742426758c7169b5e"
)
EXPECTED_EXTERNAL_ARCHIVE_SHA256 = (
    "ddfde273990adab701e4a6df88081660dc982fa60d31640a6011f7253dbddf24"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SRA_ROOT, text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"unexpected SR-Agents commit: {commit}")

    instance_path = SRA_ROOT / "data/bench/instances/toolqa.json"
    corpus_path = SRA_ROOT / "data/bench/corpus/corpus.json"
    archive_path = SRA_ROOT / "data/external/external_corpus.zip"
    if sha256(instance_path) != EXPECTED_INSTANCE_SHA256:
        raise SystemExit("ToolQA instance SHA256 mismatch")
    if sha256(corpus_path) != EXPECTED_CORPUS_SHA256:
        raise SystemExit("SRA skill corpus SHA256 mismatch")
    if not archive_path.is_file() or sha256(archive_path) != EXPECTED_EXTERNAL_ARCHIVE_SHA256:
        raise SystemExit("ToolQA external archive is missing or has the wrong SHA256")

    instances = json.loads(instance_path.read_text(encoding="utf-8"))
    if len(instances) != 1430 or len({str(row["instance_id"]) for row in instances}) != 1430:
        raise SystemExit("official ToolQA inventory is not the expected 1,430 unique tasks")
    instances = sorted(instances, key=lambda row: str(row["instance_id"]))
    skill_ids = set()
    for row in instances:
        if row.get("dataset") != "toolqa" or len(row.get("skill_annotations", [])) != 1:
            raise SystemExit(f"invalid ToolQA row: {row.get('instance_id')}")
        skill_ids.add(str(row["skill_annotations"][0]))
    if len(skill_ids) != 14 or not all(skill_id.startswith("toolqa_") for skill_id in skill_ids):
        raise SystemExit(f"unexpected ToolQA skill inventory: {sorted(skill_ids)}")

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    by_skill = {str(row["skill_id"]): row for row in corpus}
    if not skill_ids.issubset(by_skill):
        raise SystemExit(f"missing ToolQA skill descriptions: {sorted(skill_ids - set(by_skill))}")
    skills = [by_skill[skill_id] for skill_id in sorted(skill_ids)]

    external_root = SRA_ROOT / "data/external/toolqa"
    external_files: dict[str, int] = {}
    for path in sorted(external_root.rglob("*")):
        if path.is_file():
            external_files[str(path.relative_to(external_root))] = path.stat().st_size
    expected_domains = {
        "agenda/agenda_descriptions_merged.jsonl",
        "scirex/Preprocessed_Scirex.jsonl",
    }
    if not expected_domains.issubset(external_files):
        raise SystemExit("agenda/scirex ToolQA corpora are missing")
    if len(external_files) < 10:
        raise SystemExit(f"unexpectedly incomplete ToolQA external corpus: {len(external_files)} files")

    OUT.mkdir(parents=True, exist_ok=True)
    panel_path = OUT / "full.json"
    skills_path = OUT / "selected_skills.json"
    write_json(panel_path, instances)
    write_json(skills_path, skills)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "sra_toolqa_all_domain_v1",
        "frozen_date": "2026-08-29",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": "toolqa",
        "subset": "all_domains",
        "source": {
            "repository": "https://github.com/oneal2000/SR-Agents",
            "commit": EXPECTED_COMMIT,
            "instances_sha256": EXPECTED_INSTANCE_SHA256,
            "corpus_sha256": EXPECTED_CORPUS_SHA256,
            "external_archive_sha256": EXPECTED_EXTERNAL_ARCHIVE_SHA256,
        },
        "selection": {
            "official_inventory_tasks": len(instances),
            "active_scope_tasks": len(instances),
            "active_scope_skills": len(skills),
            "skill_ids": sorted(skill_ids),
            "excluded_domains": [],
            "rule": "all official ToolQA instances with their single benchmark skill annotation",
        },
        "panels": {
            "full": {
                "path": str(panel_path.relative_to(ROOT)),
                "tasks": len(instances),
                "sha256": sha256(panel_path),
                "content_sha256": json_sha256(instances),
            }
        },
        "selected_skills_artifact": {
            "path": str(skills_path.relative_to(ROOT)),
            "skills": len(skills),
            "sha256": sha256(skills_path),
            "content_sha256": json_sha256(skills),
        },
        "external_files": external_files,
        "fixed_system": {
            "model": "qwen-turbo",
            "temperature": 0.0,
            "max_steps": 8,
            "step_max_tokens": 384,
            "prompt_profile": "official",
            "arms": ["no_skill", "oracle_skill"],
            "repetitions_per_arm": 1,
            "evaluator": "pinned ToolQA normalized exact-match evaluator",
        },
        "leakage_controls": [
            "all 1,430 official task IDs are frozen before model outcomes are read",
            "the official prompt and model are identical for both arms",
            "each arm receives the same benchmark skill annotation; no skill retrieval is evaluated",
            "the prior 830-task table-workflow output is never merged into this protocol",
        ],
    }
    manifest["manifest_content_sha256"] = json_sha256(manifest)
    manifest_path = OUT / "manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "panel": str(panel_path),
                "tasks": len(instances),
                "skills": len(skills),
                "external_files": len(external_files),
                "manifest_sha256": sha256(manifest_path),
                "manifest_content_sha256": manifest["manifest_content_sha256"],
                "paid_api_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
