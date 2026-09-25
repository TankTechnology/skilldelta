#!/usr/bin/env python3
"""Freeze the exhaustive reference-valid BigCodeBench inventory.

This is deliberately separate from the earlier top-10-family confirmation
freezes.  It keeps every official task whose pinned reference passes the
evaluator, assigns a deterministic bookkeeping focal skill, and exposes the
complete annotation corpus for the all-gold arm.  No model outcomes are read
when constructing the inventory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
INSTANCES = ROOT / "experiments/data/SR-Agents/data/bench/instances/bigcodebench.json"
CORPUS = ROOT / "experiments/data/SR-Agents/data/bench/corpus/corpus.json"
OUT = ROOT / "experiments/data/sra_bench/bigcode_full_seed20260829"
SOURCE_COMMIT = "277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f"
INVALID = {
    "bigcodebench_00039": "upstream reference fails under NumPy>=2",
    "bigcodebench_00245": "upstream reference assumes obsolete scipy.stats.mode indexing",
    "bigcodebench_00634": "upstream reference assumes obsolete scipy.stats.mode shape/object support",
    "bigcodebench_00736": "upstream reference assumes obsolete scipy.stats.mode indexing",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    instances = json.loads(INSTANCES.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if len(instances) != 1140 or len({str(r["instance_id"]) for r in instances}) != 1140:
        raise SystemExit("unexpected official BigCodeBench inventory")
    corpus_by_id = {str(r["skill_id"]): r for r in corpus}
    rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for source in sorted(instances, key=lambda r: str(r["instance_id"])):
        instance_id = str(source["instance_id"])
        row = dict(source)
        annotations = [str(s) for s in row.get("skill_annotations", [])]
        if not annotations or any(s not in corpus_by_id for s in annotations):
            raise SystemExit(f"missing skill corpus entry for {instance_id}")
        # Bookkeeping only: this focal field is selected from annotations before
        # seeing outcomes and is not used by the all-gold treatment.
        row["focal_skill_id"] = annotations[0]
        if instance_id in INVALID:
            invalid_rows.append(row)
        else:
            rows.append(row)
    skills = [corpus_by_id[s] for s in sorted({
        str(s) for row in rows for s in row["skill_annotations"]
    })]
    if len(rows) != 1136 or len(skills) != 139:
        raise SystemExit(f"unexpected valid-task/skill counts: {len(rows)}, {len(skills)}")
    full_path = OUT / "full_valid.json"
    invalid_path = OUT / "reference_invalid.json"
    skills_path = OUT / "all_gold_skills.json"
    write_json(full_path, rows)
    write_json(invalid_path, invalid_rows)
    write_json(skills_path, skills)
    manifest = {
        "schema_version": 1,
        "protocol": "sra_bigcode_full_inventory_v1",
        "frozen_date": "2026-08-29",
        "dataset": "bigcodebench",
        "source_commit": SOURCE_COMMIT,
        "source": {
            "repository": "https://github.com/oneal2000/SR-Agents",
            "commit": SOURCE_COMMIT,
            "instances_path": str(INSTANCES.relative_to(ROOT)),
            "instances_sha256": sha256(INSTANCES),
            "corpus_path": str(CORPUS.relative_to(ROOT)),
            "corpus_sha256": sha256(CORPUS),
        },
        "inventory": {
            "official_tasks": len(instances),
            "reference_valid_tasks": len(rows),
            "reference_invalid_tasks": len(invalid_rows),
            "reference_invalid_ids": sorted(INVALID),
            "reference_invalid_reasons": INVALID,
            "focal_assignment": "first skill annotation, bookkeeping only",
        },
        "panels": {
            "full": {
                "path": str(full_path.relative_to(ROOT)),
                "tasks": len(rows),
                "sha256": sha256(full_path),
            }
        },
        "selected_skills_artifact": {
            "path": str(skills_path.relative_to(ROOT)),
            "skills": len(skills),
            "sha256": sha256(skills_path),
        },
        "all_gold_skills_artifact": {
            "path": str(skills_path.relative_to(ROOT)),
            "skills": len(skills),
            "sha256": sha256(skills_path),
        },
        "fixed_system": {
            "model": "qwen-turbo",
            "temperature": 0,
            "max_tokens": 4096,
            "arms": ["no_skill", "all_gold_skills"],
            "repetitions_per_arm": 1,
            "prompt": "official SR-Agents BigCodeBench direct prompt",
            "evaluation": "official pinned evaluator in network-isolated bubblewrap",
        },
        "leakage_controls": [
            "inventory and focal assignment use annotations only",
            "reference-invalid tasks are excluded before model generation",
            "all-gold content comes from the pinned SRA skill corpus",
            "the full collection protocol is distinct from earlier support/audit freezes",
        ],
    }
    manifest["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path = OUT / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "official_tasks": len(instances),
        "reference_valid_tasks": len(rows),
        "reference_invalid_tasks": len(invalid_rows),
        "all_gold_skills": len(skills),
        "paid_api_calls": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
