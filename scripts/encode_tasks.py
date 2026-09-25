"""Encode locally held task text with a local Sentence Transformers checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.batch_size < 1 or not args.model.is_dir():
        p.error("provide a local model directory and positive batch size")
    if args.output.suffix != ".npz" or args.output.exists():
        p.error("output must be a new .npz file")
    raw = args.input.read_bytes()
    rows = json.loads(raw)
    ids = [r["id"] for r in rows]
    questions = [r["question"] for r in rows]
    if (not ids or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)
            or any(not isinstance(q, str) or not q.strip() for q in questions)):
        p.error("provide unique nonempty string IDs and nonempty questions")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(args.model.resolve()), device=args.device,
                                local_files_only=True, trust_remote_code=False)
    vectors = model.encode(questions, batch_size=args.batch_size,
                           normalize_embeddings=True, convert_to_numpy=True)
    if (vectors.ndim != 2 or len(vectors) != len(ids) or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) == 0)):
        raise ValueError("encoder returned invalid vectors")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=np.asarray(ids), vectors=vectors)
    args.output.with_suffix(".metadata.json").write_text(json.dumps({
        "input_sha256": hashlib.sha256(raw).hexdigest(), "tasks": len(ids),
        "model_path": str(args.model.resolve()), "batch_size": args.batch_size,
        "device": args.device, "normalized": True,
    }, indent=2) + "\n")
    print(f"Encoded {len(ids)} local tasks into {args.output}")


if __name__ == "__main__":
    main()
