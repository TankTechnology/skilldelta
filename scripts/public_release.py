"""Validate an explicit code-only distribution, without displaying secret values."""
import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {".git", ".venv", "node_modules", "data", "local_data", "raw_inputs",
                  "outputs", "results", "reference", "paper", "third_party", "dist", "build"}
TEXT_SUFFIXES = {".py", ".js", ".mjs", ".md", ".txt", ".json", ".yml", ".yaml", ".cff"}
JSON_FILES = {"configs/protocols.json", "configs/self_judge_prompt.json",
              "plugins/dsh-skilldelta/package.json", "plugins/dsh-skilldelta/package-lock.json",
              "plugins/dsh-skilldelta/examples/support.json",
              "plugins/dsh-skilldelta/examples/queries.json"}
SPECIAL = {".gitignore", "LICENSE", "Makefile", "plugins/dsh-skilldelta/LICENSE"}
PATTERNS = {
    "provider token": rb"\b(?:sk-(?:proj-|ant-api\d*-)?[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|AKIA[0-9A-Z]{16})\b",
    "GitHub token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "private key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "credential URL": rb"https?://[^\s/@:]+:[^\s/@]{8,}@",
    "literal credential": rb'''(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)["']?\s*[:=]\s*["']([^"'\r\n]{12,})["']''',
}
PLACEHOLDER = re.compile(rb"(?i)^(?:your[-_ ]|example|placeholder|dummy|test[-_ ]|fake[-_ ]|\$|\{|os\.|process\.|[*.]{3,})")


def validate(root=ROOT, *, tracked=False):
    root = Path(root).resolve()
    names = [line.strip() for line in (root / "PUBLIC_FILES.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    if len(names) != len(set(names)) or "PUBLIC_FILES.txt" not in names:
        raise ValueError("allowlist must include itself and contain no duplicates")
    paths = []
    for name in names:
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or relative.as_posix() != name or FORBIDDEN_DIRS.intersection(relative.parts)):
            raise ValueError(f"forbidden public path: {name}")
        if any(part.startswith(".env") for part in relative.parts):
            raise ValueError(f"environment file excluded: {name}")
        p = root / name
        if any((root / Path(*relative.parts[:i])).is_symlink()
               for i in range(1, len(relative.parts) + 1)):
            raise ValueError(f"symlink excluded: {name}")
        if not p.is_file() or not p.resolve().is_relative_to(root):
            raise ValueError(f"missing or external file: {name}")
        if name != "docs/assets/teaser.png":
            if name not in SPECIAL and p.suffix not in TEXT_SUFFIXES:
                raise ValueError(f"non-code format excluded: {name}")
            if p.suffix == ".json" and name not in JSON_FILES:
                raise ValueError(f"unreviewed JSON excluded: {name}")
            data = p.read_bytes()
            data.decode("utf-8")
            for label, pattern in PATTERNS.items():
                for match in re.finditer(pattern, data):
                    candidate = match.group(1) if match.lastindex else match.group()
                    if not PLACEHOLDER.search(candidate):
                        line = data[:match.start()].count(b"\n") + 1
                        raise ValueError(f"possible {label}: {name}:{line}; value omitted")
        paths.append(p)
    if tracked:
        raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
        observed = set(raw.decode().rstrip("\0").split("\0"))
        expected = set(names) | {"MANIFEST.json"}
        if observed != expected:
            raise ValueError(f"Git inventory mismatch: extra={sorted(observed-expected)}, "
                             f"missing={sorted(expected-observed)}")
    return sorted(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true")
    args = parser.parse_args()
    print(f"Validated {len(validate(tracked=args.tracked))} code-only source files.")
