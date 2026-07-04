from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan an anonymized artifact package for obvious private endpoints."
    )
    parser.add_argument("root", type=Path, help="Artifact package root to scan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists():
        raise SystemExit(f"Artifact root does not exist: {root}")

    findings: list[tuple[Path, int, str, str]] = []
    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            findings.append((path, 0, "read-error", str(exc)))
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append((path, lineno, name, line.strip()[:180]))

    if findings:
        print("Potential deanonymization risks found:")
        for path, lineno, name, snippet in findings[:80]:
            location = f"{path}:{lineno}" if lineno else str(path)
            print(f"- [{name}] {location}: {snippet}")
        if len(findings) > 80:
            print(f"... {len(findings) - 80} more findings omitted")
        raise SystemExit(1)

    print(f"No obvious private endpoints found under {root}")


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def public_ipv4_pattern() -> re.Pattern[str]:
    octet = r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    ip = rf"\b({octet})\.({octet})\.({octet})\.({octet})\b"
    private = (
        r"^(?:10\.|127\.|0\.0\.0\.0$|192\.168\.|"
        r"172\.(?:1[6-9]|2\d|3[01])\.)"
    )

    class PublicIPv4:
        def __init__(self) -> None:
            self._ip = re.compile(ip)
            self._private = re.compile(private)

        def search(self, line: str):
            for match in self._ip.finditer(line):
                value = match.group(0)
                if not self._private.search(value):
                    return match
            return None

    return PublicIPv4()  # type: ignore[return-value]


PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("public-ipv4", public_ipv4_pattern()),
    (
        "ssh-endpoint",
        re.compile(
            r"\bssh\s+-p\b|(?<!\\)\b[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|connect\.",
            re.IGNORECASE,
        ),
    ),
    (
        "password-assignment",
        re.compile(r"\b(?:pass(?:word)?|pwd|CODX_SSH_PASS)\b\s*[:=]", re.IGNORECASE),
    ),
    ("windows-user-path", re.compile(r"C:\\Users\\[^\\\s]+", re.IGNORECASE)),
]


if __name__ == "__main__":
    main()
