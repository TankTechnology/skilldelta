"""Check the code-only manifest and every included file's checksum."""
import hashlib
import json
from public_release import ROOT, validate


def verify():
    expected = {p.relative_to(ROOT).as_posix(): p for p in validate()}
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    items = manifest["files"]
    if len(items) != len(expected) or {i["path"] for i in items} != set(expected):
        raise ValueError("manifest and public allowlist differ")
    for item in items:
        raw = expected[item["path"]].read_bytes()
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"checksum mismatch: {item['path']}")
    print(f"Verified {len(expected)} distributed source files.")


if __name__ == "__main__":
    verify()
