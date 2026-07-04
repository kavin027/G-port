from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

import paramiko


DEFAULT_PROBE = r"""set -o pipefail
printf "HOSTNAME="; hostname
printf "USER="; whoami
printf "UNAME="; uname -a
printf "PYTHON3="; (python3 --version 2>/dev/null || true)
printf "PYTHON="; (python --version 2>/dev/null || true)
printf "NPROC="; nproc 2>/dev/null || true
printf "MEM="; free -h | awk '/Mem:/ {print $2 " total, " $7 " available"}'
printf "DISK="; df -h / | awk 'NR==2 {print $4 " free of " $2}'
printf "DOCKER="; (docker --version 2>/dev/null || echo missing)
printf "K3S="; (k3s --version 2>/dev/null | head -1 || echo missing)
printf "KUBECTL="; (kubectl version --client=true 2>/dev/null | head -1 || echo missing)
printf "K3S_ACTIVE="; (systemctl is-active k3s 2>/dev/null || true)
printf "K3S_AGENT_ACTIVE="; (systemctl is-active k3s-agent 2>/dev/null || true)
printf "IP4="; ip -4 -o addr show scope global | awk '{print $2 ":" $4}' | paste -sd ',' -
printf "ROUTE="; ip route | head -3 | paste -sd ';' -
"""


@dataclass
class RemoteResult:
    host: str
    status: str
    stdout: str = ""
    stderr: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small SSH helper for the coded-learning K3s cluster.")
    parser.add_argument("--hosts", nargs="+", required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--password-env", default="CODED_REMOTE_PASSWORD")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--command", default=DEFAULT_PROBE)
    parser.add_argument("--command-file", type=Path)
    parser.add_argument(
        "--upload",
        nargs=2,
        metavar=("LOCAL", "REMOTE"),
        help="Upload LOCAL to REMOTE on each host before running the command.",
    )
    parser.add_argument(
        "--download",
        nargs=2,
        metavar=("REMOTE", "LOCAL_DIR"),
        help="Download REMOTE from each host into LOCAL_DIR/<host>/ after running the command.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = os.environ.get(args.password_env)
    if not password:
        raise SystemExit(f"Environment variable {args.password_env} is not set.")
    command = args.command_file.read_text(encoding="utf-8") if args.command_file else args.command
    for result in run_many(
        hosts=args.hosts,
        user=args.user,
        port=args.port,
        password=password,
        command=command,
        timeout=args.timeout,
        upload=tuple(args.upload) if args.upload else None,
        download=tuple(args.download) if args.download else None,
    ):
        print(f"=== {result.host} [{result.status}] ===")
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print("[stderr]")
            print(result.stderr.rstrip())


def run_many(
    *,
    hosts: list[str],
    user: str,
    port: int,
    password: str,
    command: str,
    timeout: float,
    upload: tuple[str, str] | None = None,
    download: tuple[str, str] | None = None,
) -> list[RemoteResult]:
    results: list[RemoteResult] = []
    for host in hosts:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                host,
                username=user,
                password=password,
                port=port,
                timeout=timeout,
                banner_timeout=timeout,
                auth_timeout=timeout,
                look_for_keys=False,
                allow_agent=False,
            )
            if upload is not None:
                local, remote = upload
                with client.open_sftp() as sftp:
                    sftp_put_recursive(sftp, Path(local), remote)
            _, stdout, stderr = client.exec_command(command, timeout=timeout)
            out = stdout.read().decode("utf-8", "replace")
            err = stderr.read().decode("utf-8", "replace")
            exit_status = stdout.channel.recv_exit_status()
            if download is not None:
                remote, local_dir = download
                target = Path(local_dir) / safe_host_dir(host)
                target.mkdir(parents=True, exist_ok=True)
                with client.open_sftp() as sftp:
                    sftp_get_recursive(sftp, remote, target)
            status = f"exit={exit_status}"
            results.append(RemoteResult(host, status, out, err))
        except (paramiko.SSHException, socket.error, TimeoutError) as exc:
            results.append(RemoteResult(host, f"connect-error:{type(exc).__name__}", "", str(exc)))
        finally:
            client.close()
    return results


def sftp_put_recursive(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    if local.is_dir():
        mkdir_p(sftp, remote)
        for child in local.iterdir():
            sftp_put_recursive(sftp, child, remote.rstrip("/") + "/" + child.name)
        return
    parent = str(Path(remote).parent).replace("\\", "/")
    mkdir_p(sftp, parent)
    sftp.put(str(local), remote)


def sftp_get_recursive(sftp: paramiko.SFTPClient, remote: str, local: Path) -> None:
    try:
        attrs = sftp.stat(remote)
    except FileNotFoundError:
        return
    if stat_is_dir(attrs.st_mode):
        local.mkdir(parents=True, exist_ok=True)
        for item in sftp.listdir_attr(remote):
            sftp_get_recursive(
                sftp,
                remote.rstrip("/") + "/" + item.filename,
                local / item.filename,
            )
        return
    if local.exists() and local.is_dir():
        local = local / Path(remote).name
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))


def mkdir_p(sftp: paramiko.SFTPClient, remote: str) -> None:
    remote = remote.replace("\\", "/")
    if remote in {"", ".", "/"}:
        return
    parts = [part for part in remote.split("/") if part]
    current = "/" if remote.startswith("/") else ""
    for part in parts:
        current = current.rstrip("/") + "/" + part
        try:
            sftp.mkdir(current)
        except OSError:
            pass


def stat_is_dir(mode: int) -> bool:
    return (mode & 0o170000) == 0o040000


def safe_host_dir(host: str) -> str:
    return host.replace(":", "_").replace("/", "_")


if __name__ == "__main__":
    main()
