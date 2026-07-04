from __future__ import annotations

import argparse
import os
import shlex
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from pathlib import PurePosixPath

import paramiko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote SSH helper for paper experiments.")
    parser.add_argument("--hosts", nargs="+", required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password-env", default="CODX_SSH_PASS")
    parser.add_argument("--timeout", type=float, default=30.0)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a shell command on all hosts.")
    run.add_argument("--shell", required=True)

    put = sub.add_parser("put", help="Upload a local file or directory tarball to all hosts.")
    put.add_argument("--local", required=True)
    put.add_argument("--remote", required=True)
    put.add_argument("--exclude", action="append", default=[])
    put.add_argument(
        "--include",
        action="append",
        default=[],
        help="Relative path or glob to include. If omitted, include the whole local tree.",
    )

    get = sub.add_parser("get", help="Download a remote file/directory tarball from the first host.")
    get.add_argument("--remote", required=True)
    get.add_argument("--local", required=True)
    return parser.parse_args()


def connect(host: str, user: str, password: str, timeout: float) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        port=22,
        username=user,
        password=password,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
    )
    return client


def run_shell(client: paramiko.SSHClient, command: str, timeout: float) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    del stdin
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out, err


def make_tarball(source: Path, excludes: list[str], includes: list[str]) -> Path:
    source = source.resolve()
    fd, name = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    tar_path = Path(name)

    def should_exclude(path: Path) -> bool:
        rel = path.relative_to(source).as_posix() if path != source else ""
        defaults = [
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "venv",
            "paper/socc26/main.pdf",
            "paper/socc26/main_reorg.pdf",
            "paper/socc26/main_revised_external.pdf",
        ]
        for pattern in defaults + excludes:
            if not pattern:
                continue
            if rel == pattern or rel.startswith(pattern.rstrip("/") + "/"):
                return True
        return False

    def iter_paths() -> list[Path]:
        if not includes:
            return list(source.rglob("*"))
        selected: set[Path] = set()
        for pattern in includes:
            matches = list(source.glob(pattern))
            if not matches:
                candidate = source / pattern
                if candidate.exists():
                    matches = [candidate]
            for match in matches:
                if match.is_dir():
                    selected.add(match)
                    selected.update(match.rglob("*"))
                elif match.exists():
                    selected.add(match)
        return sorted(selected)

    with tarfile.open(tar_path, "w:gz") as tar:
        for path in iter_paths():
            if should_exclude(path):
                continue
            arcname = Path(source.name) / path.relative_to(source)
            tar.add(path, arcname=arcname, recursive=False)
    return tar_path


def upload_tree(
    client: paramiko.SSHClient,
    local: Path,
    remote: str,
    excludes: list[str],
    includes: list[str],
) -> None:
    tar_path = make_tarball(local, excludes, includes)
    remote_tar = f"/tmp/codex_upload_{int(time.time())}.tar.gz"
    try:
        sftp = client.open_sftp()
        try:
            sftp.put(str(tar_path), remote_tar)
        finally:
            sftp.close()
        remote_posix = PurePosixPath(remote)
        target = shlex.quote(remote)
        target_new = shlex.quote(f"{remote}.new")
        cmd = (
            f"rm -rf {target}.new && mkdir -p {target_new} && "
            f"tar -xzf {shlex.quote(remote_tar)} -C {target_new} --strip-components=1 && "
            f"rm -rf {target}.old && "
            f"if [ -e {target} ]; then mv {target} {target}.old; fi && "
            f"mv {target_new} {target} && "
            f"rm -f {shlex.quote(remote_tar)}"
        )
        code, out, err = run_shell(client, cmd, 600)
        if code != 0:
            raise RuntimeError(f"remote unpack failed: {err or out}")
    finally:
        tar_path.unlink(missing_ok=True)


def download_tree(client: paramiko.SSHClient, remote: str, local: Path) -> None:
    remote_tar = f"/tmp/codex_download_{int(time.time())}.tar.gz"
    remote_q = shlex.quote(remote)
    remote_posix = PurePosixPath(remote)
    cmd = (
        f"if [ -e {remote_q} ]; then "
        f"tar -czf {remote_tar} -C {shlex.quote(str(remote_posix.parent))} "
        f"{shlex.quote(remote_posix.name)}; else exit 44; fi"
    )
    code, out, err = run_shell(client, cmd, 600)
    if code != 0:
        raise RuntimeError(f"remote pack failed: {err or out}")
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        sftp.get(remote_tar, str(local))
    finally:
        sftp.close()
        run_shell(client, f"rm -f {remote_tar}", 30)


def main() -> None:
    args = parse_args()
    password = os.environ.get(args.password_env)
    if not password:
        raise SystemExit(f"Missing password in ${args.password_env}")
    failures = 0
    hosts = args.hosts if args.cmd != "get" else args.hosts[:1]
    for host in hosts:
        print(f"=== {host} ===")
        try:
            client = connect(host, args.user, password, args.timeout)
            try:
                if args.cmd == "run":
                    code, out, err = run_shell(client, args.shell, args.timeout)
                    if out.strip():
                        print(out.rstrip())
                    if err.strip():
                        print("ERR:", err.rstrip())
                    if code != 0:
                        failures += 1
                        print(f"EXIT={code}")
                elif args.cmd == "put":
                    upload_tree(client, Path(args.local), args.remote, args.exclude, args.include)
                    print(f"uploaded {args.local} -> {args.remote}")
                elif args.cmd == "get":
                    download_tree(client, args.remote, Path(args.local))
                    print(f"downloaded {args.remote} -> {args.local}")
            finally:
                client.close()
        except Exception as exc:
            failures += 1
            print(f"ERROR: {exc!r}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
