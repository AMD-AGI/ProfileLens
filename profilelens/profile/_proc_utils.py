# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Process-group helpers shared by both ProfileLens phases."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


class WorkloadTimeout(subprocess.TimeoutExpired):
    """A workload was stopped by the watchdog. ``reason`` says which limit tripped."""

    def __init__(self, cmd: Any, timeout: float, reason: str) -> None:
        super().__init__(cmd, timeout)
        self.reason = reason


def _group_cpu_seconds(pgid: int) -> float:
    """Total CPU seconds used by live processes in process group ``pgid``."""
    ticks = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            # comm may contain spaces; split after the last ") ".
            stat = Path("/proc", entry, "stat").read_text().rsplit(") ", 1)[1].split()
            if int(stat[2]) != pgid:
                continue
            ticks += int(stat[11]) + int(stat[12])  # utime + stime
        except (OSError, IndexError, ValueError):
            continue
    return ticks / os.sysconf("SC_CLK_TCK")


def _newest_mtime(paths) -> float:
    newest = 0.0
    for path in paths:
        try:
            newest = max(newest, os.stat(path).st_mtime)
        except OSError:
            continue
    return newest


def wait_with_watchdog(
    proc: Any,
    *,
    wall_timeout: float = 0.0,
    idle_timeout: float = 0.0,
    progress_paths: Any = (),
    poll_s: float = 5.0,
) -> int:
    """Wait for ``proc``, raising WorkloadTimeout on wall-clock or idle limits."""
    if wall_timeout <= 0 and idle_timeout <= 0:
        return proc.wait()

    started = time.monotonic()
    last_progress = started
    last_cpu = _group_cpu_seconds(proc.pid)
    last_mtime = _newest_mtime(progress_paths)

    while True:
        try:
            return proc.wait(timeout=poll_s)
        except subprocess.TimeoutExpired:
            pass

        now = time.monotonic()
        if wall_timeout > 0 and now - started >= wall_timeout:
            raise WorkloadTimeout(
                getattr(proc, "args", None), wall_timeout,
                f"ran for {now - started:.0f}s, exceeding the "
                f"{wall_timeout:.0f}s total limit (--workload-timeout)",
            )

        cpu, mtime = _group_cpu_seconds(proc.pid), _newest_mtime(progress_paths)
        if cpu > last_cpu + 0.01 or mtime > last_mtime:
            last_cpu, last_mtime, last_progress = cpu, mtime, now
        elif idle_timeout > 0 and now - last_progress >= idle_timeout:
            raise WorkloadTimeout(
                getattr(proc, "args", None), idle_timeout,
                f"made no progress for {now - last_progress:.0f}s -- no CPU time "
                f"and no new output from the process group. Exceeds the "
                f"{idle_timeout:.0f}s limit (--workload-idle-timeout)",
            )


def reap_process_group(
    leader_pid: int,
    console: Any = None,
    label: str = "",
    *,
    sigint_grace_s: float = 60.0,
    log_dir: str | os.PathLike[str] | None = None,
) -> None:
    """SIGINT the process group, wait for it to drain, then SIGKILL."""

    pgid = leader_pid
    try:
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return

    poll_interval_s = 0.5
    deadline = time.monotonic() + sigint_grace_s
    while time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            if console:
                console.print(
                    f"  [dim][{label}] workload process group exited cleanly "
                    f"(pgid={pgid})[/dim]"
                )
            return

    if console:
        hint = _peer_wait_hint(log_dir)
        if hint:
            console.print(hint)
        console.print(
            f"  [yellow][{label}] workload process group (pgid={pgid}) did not "
            f"exit within {sigint_grace_s:.0f}s after SIGINT; falling back to "
            f"SIGKILL — rocprofv3 finalization for any still-running subprocess "
            f"will be incomplete[/yellow]"
        )
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


_PEER_WAIT_RE = re.compile(r"will wait for all (\d+) peer processes")


def _peer_wait_hint(log_dir: str | os.PathLike[str] | None) -> str:
    if not log_dir:
        return ""
    peers = 0
    for name in ("rocprofv3_stdout.log", "rocprofv3_stderr.log"):
        path = Path(log_dir) / name
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for m in _PEER_WAIT_RE.finditer(text):
            peers = max(peers, int(m.group(1)))
    if not peers:
        return ""
    return (
        f"  [yellow]rocprofv3 is waiting for {peers} peer process(es) under the same "
        f"parent to exit — the group cannot drain until they do.[/yellow]\n"
        f"  [dim]This happens when the whole process tree is instrumented and the "
        f"workload keeps a non-GPU helper alive. Profiling the server process "
        f"directly (server mode) avoids it.[/dim]"
    )
