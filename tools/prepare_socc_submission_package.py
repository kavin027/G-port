"""Prepare a sanitized SoCC submission dry-run package.

The script creates two review-facing ZIP files:

* paper_source.zip: LaTeX source, required figures, and the compiled PDF.
* artifact.zip: source code, reviewer docs, compact diagnostics, and sanitized
  K3s evidence needed to regenerate the paper tables.

It intentionally excludes raw SSH logs, bulky exploratory directories, LaTeX
intermediates, private endpoint dumps, and Kubernetes raw resource JSON.  K3s
pod-placement text and run configs are copied through a small sanitizer so the
artifact preserves topology evidence without exposing private IPs or cloud node
names.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "submission" / "socc26_dryrun_20260701"

PAPER_FILES = [
    "paper/socc26/main.tex",
    "paper/socc26/references.bib",
    "paper/socc26/README.md",
    "paper/socc26/artifact_appendix.md",
    "paper/socc26/main.pdf",
]

PAPER_FIGURES = [
    "paper/socc26/figures/hypothesis_improvements.png",
    "paper/socc26/figures/tcp_time_to_loss.png",
    "paper/socc26/figures/runtime_realdata_improvements_a9a_w8a_rcv1.png",
    "paper/socc26/figures/controlled_alignment_p95.png",
    "paper/socc26/figures/mechanism_trace_prefix_latency.png",
]

ARTIFACT_FILES = [
    "README.md",
    "README_REVIEWER.md",
    "requirements.txt",
    ".gitignore",
    "docker/Dockerfile.network-worker",
    "run_socc_artifact_fast_path.py",
    "run_network_container_experiment.py",
    "run_direct_docker_network_experiment.py",
    "run_direct_docker_sweep.py",
    "analyze_network_container_results.py",
    "analyze_guarded_policy.py",
    "analyze_online_guard_sensitivity.py",
    "analyze_guard_prediction.py",
    "analyze_performance_fallback.py",
    "analyze_online_tail_predictor.py",
    "analyze_score_ablation.py",
    "analyze_majorrev_k8s.py",
    "analyze_k8s_stress.py",
    "collect_k8s_resource_counters.py",
    "run_majorrev_k8s_extended.py",
    "run_majorrev_k8s_stress.py",
    "run_worker_service_stress.py",
    "tools/prepare_socc_submission_package.py",
    "run_tunneled_remote_network_experiment.py",
    "run_tunneled_remote_sweep.py",
    "run_direct_remote_network_experiment.py",
    "run_direct_remote_sweep.py",
    "run_multiprocess_experiment.py",
    "run_multiprocess_sweep.py",
    "run_multiprocess_worker_scaling.py",
    "run_realdata_multiprocess_experiment.py",
    "run_realdata_multiprocess_sweep.py",
    "run_realdata_alignment_sweep.py",
    "run_realdata_sensitivity_sweep.py",
    "analyze_runtime_scaling.py",
    "analyze_realdata_results.py",
    "analyze_realdata_alignment.py",
    "analyze_realdata_sensitivity.py",
    "docs/socc_artifact_reproduction.md",
    "docs/reproducibility_tcp_runtime.md",
    "docs/two_node_tcp_experiment.md",
    "docs/direct_tcp_service_results_fresh_server.md",
    "docs/direct_multinode_port_probe.md",
    "docs/submission_checklist.md",
    "paper/socc26/artifact_appendix.md",
]

COMPACT_DIRS = [
    "guard_prediction_diagnostics",
    "tail_predictor_diagnostics",
    "score_ablation_diagnostics",
    "online_guard_sensitivity_diagnostics",
]

TOP_LEVEL_DIAGNOSTIC_PATTERNS = [
    "*.csv",
    "*.md",
    "*.tex",
]

PER_RUN_KEEP_NAMES = {
    "network_metrics.csv",
    "network_summary.csv",
    "k8s_summary.csv",
    "k8s_resource_counters.csv",
    "k8s_resource_report.md",
    "k8s_master_job.yaml",
    "k8s_workers.yaml",
    "k8s_multinode_manifest.yaml",
    "k8s_multinode_run_config.json",
    "k8s_pods_wide.txt",
}

WORKER_STRESS_KEEP_NAMES = {
    "worker_service_stress_plan.json",
    "worker_service_stress_report.md",
    "worker_service_stress_summary.csv",
    "network_metrics.csv",
    "network_summary.csv",
    "direct_docker_manifest.json",
    "direct_endpoint_summary.csv",
}

FAST_PATH_KEEP_NAMES = {
    "summary_report.md",
    "network_metrics.csv",
    "network_summary.csv",
    "direct_docker_manifest.json",
    "direct_endpoint_summary.csv",
    "direct_docker_bridge_summary.csv",
    "tcp_smoke_summary.csv",
    "guarded_policy_report.md",
    "guard_ablation_summary.csv",
    "chronological_guard_replay.csv",
    "mechanism_trace_prefix_latency.png",
}

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}

PRIVATE_IP_RE = re.compile(
    r"\b(?:"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}"
    r")\b"
)
CLOUD_NODE_RE = re.compile(r"\biz[0-9a-z]+\b", re.IGNORECASE)
WIN_USER_PATH_RE = re.compile(r"C:\\Users\\[^\\\s]+", re.IGNORECASE)


@dataclass
class PackageStats:
    copied_files: int = 0
    sanitized_files: int = 0
    missing: list[str] | None = None

    def __post_init__(self) -> None:
        if self.missing is None:
            self.missing = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory first. Refuses paths outside submission/.",
    )
    return parser.parse_args()


def assert_safe_out(out: Path) -> Path:
    resolved = out.resolve()
    submission = (ROOT / "submission").resolve()
    if resolved == ROOT or submission not in resolved.parents:
        raise SystemExit(f"Refusing output outside submission/: {resolved}")
    return resolved


def sanitize_text(text: str) -> tuple[str, bool]:
    original = text
    text = PRIVATE_IP_RE.sub("<PRIVATE_IP>", text)
    text = CLOUD_NODE_RE.sub("<K8S_NODE>", text)
    text = WIN_USER_PATH_RE.sub(r"C:\\Users\\<USER>", text)
    return text, text != original


def copy_file(
    src_rel: str | Path,
    dst_root: Path,
    stats: PackageStats,
    sanitize: bool = False,
    dst_rel: str | Path | None = None,
) -> None:
    src_rel = Path(src_rel)
    src = ROOT / src_rel
    dst = dst_root / (Path(dst_rel) if dst_rel is not None else src_rel)
    if not src.exists():
        stats.missing.append(str(src_rel).replace("\\", "/"))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sanitize and src.suffix.lower() in TEXT_SUFFIXES:
        text = src.read_text(encoding="utf-8", errors="replace")
        text, changed = sanitize_text(text)
        dst.write_text(text, encoding="utf-8", newline="")
        stats.sanitized_files += int(changed)
    else:
        shutil.copy2(src, dst)
    stats.copied_files += 1


def copy_tree(
    src_rel: str,
    dst_root: Path,
    stats: PackageStats,
    *,
    keep: callable | None = None,
    sanitize: bool = False,
) -> None:
    src_root = ROOT / src_rel
    if not src_root.exists():
        stats.missing.append(src_rel)
        return
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(ROOT)
        if keep is not None and not keep(rel, src):
            continue
        copy_file(rel, dst_root, stats, sanitize=sanitize)


def copy_paper(out: Path) -> PackageStats:
    stats = PackageStats()
    dst = out / "paper_source"
    for rel in PAPER_FILES + PAPER_FIGURES:
        src_rel = Path(rel)
        copy_file(src_rel, dst, stats, dst_rel=Path(*src_rel.parts[2:]))
    section_root = ROOT / "paper/socc26/sections"
    if not section_root.exists():
        stats.missing.append("paper/socc26/sections")
    else:
        for src in sorted(section_root.glob("*.tex")):
            src_rel = src.relative_to(ROOT)
            copy_file(src_rel, dst, stats, dst_rel=Path("sections") / src.name)
    return stats


def artifact_keep(rel: Path, src: Path) -> bool:
    name = src.name
    parts = rel.parts
    if "__pycache__" in parts or name.endswith(".pyc"):
        return False
    if src.suffix.lower() == ".log":
        return False
    return True


def copy_compact_diagnostic_dir(rel_dir: str, dst: Path, stats: PackageStats) -> None:
    copy_tree(rel_dir, dst, stats, keep=artifact_keep, sanitize=True)


def is_top_level_diag_file(rel: Path, root_name: str) -> bool:
    if len(rel.parts) != 2 or rel.parts[0] != root_name:
        return False
    return any(fnmatch.fnmatch(rel.name, pattern) for pattern in TOP_LEVEL_DIAGNOSTIC_PATTERNS)


def copy_k8s_diagnostics(src_rel: str, dst: Path, stats: PackageStats) -> None:
    root_name = Path(src_rel).name
    src_root = ROOT / src_rel
    if not src_root.exists():
        stats.missing.append(src_rel)
        return
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(ROOT)
        if is_top_level_diag_file(rel, root_name) or src.name in PER_RUN_KEEP_NAMES:
            copy_file(rel, dst, stats, sanitize=True)


def copy_worker_stress(dst: Path, stats: PackageStats) -> None:
    src_rel = "worker_service_stress_diagnostics"
    src_root = ROOT / src_rel
    if not src_root.exists():
        stats.missing.append(src_rel)
        return
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(ROOT)
        if src.name in WORKER_STRESS_KEEP_NAMES:
            copy_file(rel, dst, stats, sanitize=True)


def copy_fast_path(dst: Path, stats: PackageStats) -> None:
    src_rel = "socc_fast_path_artifact"
    src_root = ROOT / src_rel
    if not src_root.exists():
        stats.missing.append(src_rel)
        return
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(ROOT)
        if src.name in FAST_PATH_KEEP_NAMES:
            copy_file(rel, dst, stats, sanitize=True)


def copy_artifact(out: Path) -> PackageStats:
    stats = PackageStats()
    dst = out / "artifact"
    for rel in ARTIFACT_FILES:
        copy_file(rel, dst, stats)
    copy_tree("src/coded_learning_exp", dst, stats, keep=lambda _rel, src: src.suffix == ".py")
    for rel_dir in COMPACT_DIRS:
        copy_compact_diagnostic_dir(rel_dir, dst, stats)
    copy_k8s_diagnostics("majorrev_k8s_diagnostics", dst, stats)
    copy_k8s_diagnostics("majorrev_k8s_stress_diagnostics", dst, stats)
    copy_worker_stress(dst, stats)
    copy_fast_path(dst, stats)
    return stats


def zip_dir(src_dir: Path, zip_path: Path) -> str:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))
    return sha256_file(zip_path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_sensitive(paths: Iterable[Path]) -> list[str]:
    hits: list[str] = []
    for root in paths:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            rel = path.relative_to(root.parent)
            text = path.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if PRIVATE_IP_RE.search(line) or CLOUD_NODE_RE.search(line) or WIN_USER_PATH_RE.search(line):
                    hits.append(f"{rel}:{lineno}: {line[:160]}")
                    if len(hits) >= 200:
                        return hits
    return hits


def count_files(root: Path) -> tuple[int, int]:
    files = list(p for p in root.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    return len(files), total


def write_manifest(out: Path, paper_stats: PackageStats, artifact_stats: PackageStats, checksums: dict[str, str], hits: list[str]) -> None:
    paper_files, paper_bytes = count_files(out / "paper_source")
    artifact_files, artifact_bytes = count_files(out / "artifact")
    manifest = [
        "# SoCC Submission Dry-Run Package",
        "",
        "Generated by `tools/prepare_socc_submission_package.py`.",
        "",
        "## Contents",
        "",
        f"- `paper_source/`: {paper_files} files, {paper_bytes / (1024 * 1024):.2f} MiB.",
        f"- `artifact/`: {artifact_files} files, {artifact_bytes / (1024 * 1024):.2f} MiB.",
        "- `paper_source.zip`: clean LaTeX/PDF package.",
        "- `artifact.zip`: sanitized reviewer artifact package.",
        "",
        "## Copy Stats",
        "",
        f"- Paper copied files: {paper_stats.copied_files}",
        f"- Artifact copied files: {artifact_stats.copied_files}",
        f"- Sanitized text files changed: {paper_stats.sanitized_files + artifact_stats.sanitized_files}",
        "",
        "## Missing Inputs",
        "",
    ]
    missing = (paper_stats.missing or []) + (artifact_stats.missing or [])
    if missing:
        manifest.extend(f"- `{item}`" for item in missing)
    else:
        manifest.append("- None.")
    manifest.extend(["", "## Sensitive Scan", ""])
    if hits:
        manifest.append("Potential private endpoint or node-name hits remain:")
        manifest.extend(f"- `{hit}`" for hit in hits[:50])
        if len(hits) > 50:
            manifest.append(f"- ... {len(hits) - 50} additional hits omitted.")
    else:
        manifest.append("- No private-IP, cloud-node-name, or local-user-path hits in text files.")
    manifest.extend(["", "## SHA256", ""])
    for name, digest in checksums.items():
        manifest.append(f"- `{digest}  {name}`")
    (out / "PACKAGE_MANIFEST.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = assert_safe_out(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    paper_stats = copy_paper(out)
    artifact_stats = copy_artifact(out)

    checksums = {
        "paper_source.zip": zip_dir(out / "paper_source", out / "paper_source.zip"),
        "artifact.zip": zip_dir(out / "artifact", out / "artifact.zip"),
    }
    (out / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )

    hits = scan_sensitive([out / "paper_source", out / "artifact"])
    write_manifest(out, paper_stats, artifact_stats, checksums, hits)

    print(f"Wrote dry-run package to {out}")
    print(f"paper_source.zip sha256={checksums['paper_source.zip']}")
    print(f"artifact.zip sha256={checksums['artifact.zip']}")
    if hits:
        print(f"WARNING: sensitive scan found {len(hits)} possible hits; see PACKAGE_MANIFEST.md")
        return 2
    if paper_stats.missing or artifact_stats.missing:
        print("WARNING: missing inputs; see PACKAGE_MANIFEST.md")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
