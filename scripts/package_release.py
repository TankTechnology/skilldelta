"""Package only the files explicitly permitted by the public release policy."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from public_release import ROOT, validate


def package(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    sources = validate(root)
    manifest_path = root / "MANIFEST.json"
    if output in sources or output == manifest_path or manifest_path.is_symlink():
        raise ValueError("output must not overwrite source or follow a manifest symlink")
    manifest = {"schema_version": 2, "distribution": "public code only",
                "files": [{"path": p.relative_to(root).as_posix(),
                           "bytes": p.stat().st_size,
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in sources]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sources + [manifest_path]:
            info = zipfile.ZipInfo("skilldelta/" + p.relative_to(root).as_posix(),
                                   date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes(), compresslevel=9)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    return {"archive": str(output), "files": len(sources) + 1,
            "bytes": output.stat().st_size, "sha256": digest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/skilldelta-public.zip")
    args = parser.parse_args()
    print(json.dumps(package(ROOT, args.output), indent=2))
