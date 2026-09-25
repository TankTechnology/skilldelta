"""Evaluate a locally supplied paired panel; no reference results are required."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skilldelta import Protocol, predict_gain, predict_skill_success
from skilldelta.metrics import auroc, classification, matched_rate_advantage, policy


def evaluate(rows, ids, vectors, protocol, mean_router_tokens=0.0):
    task_ids = [r["id"] for r in rows]
    if not rows or task_ids != list(ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("panel IDs must be unique and match embedding order exactly")
    families = [r["family"] for r in rows]
    y0, y1, t0, t1 = [np.asarray([r[key] for r in rows], dtype=float)
                      for key in ("y0", "y1", "tokens0", "tokens1")]
    for values in (t0, t1):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("token counts must be finite and nonnegative")
    if t1.mean() <= 0 or not np.isfinite(mean_router_tokens) or mean_router_tokens < 0:
        raise ValueError("need positive mean Use skill tokens and nonnegative router cost")
    inputs = dict(query_vectors=vectors, support_vectors=vectors,
                  query_ids=task_ids, support_ids=task_ids,
                  query_families=families, support_families=families, protocol=protocol)
    p = predict_gain(**inputs, skip_outcomes=y0, use_outcomes=y1)
    on_only = predict_skill_success(**inputs, use_outcomes=y1)
    return {"tasks": len(rows), "protocol": asdict(protocol),
            "classification": classification(y1 > y0, p["scores"], p["use_skill"]),
            "policy": policy(y0, y1, t0, t1, p["use_skill"], mean_router_tokens=mean_router_tokens),
            "matched_rate": matched_rate_advantage(y0, y1, p["use_skill"]),
            "skill_on_only_auroc": auroc(y1 > y0, on_only),
            "predictions": [{"id": task_id, "score": float(p["scores"][i]),
                             "threshold": float(p["thresholds"][i]),
                             "use_skill": bool(p["use_skill"][i])}
                            for i, task_id in enumerate(task_ids)]}


if __name__ == "__main__":
    protocols = json.loads((ROOT / "configs/protocols.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--benchmark", choices=protocols, required=True)
    parser.add_argument("--mean-router-tokens", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.panel.read_text())["rows"]
    with np.load(args.embeddings, allow_pickle=False) as data:
        result = evaluate(rows, data["ids"].tolist(), data["vectors"],
                          Protocol(**protocols[args.benchmark]), args.mean_router_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Evaluated {len(rows)} locally supplied tasks; wrote {args.output}")
