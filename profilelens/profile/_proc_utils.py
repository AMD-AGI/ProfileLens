# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Process-group helpers shared by both ProfileLens phases."""

from __future__ import annotations

import os
import signal
import time
from typing import Any


def reap_process_group(
    leader_pid: int,
    console: Any = None,
    label: str = "",
    *,
    sigint_grace_s: float = 60.0,
) -> None:
    """
    Two-stage shutdown to ensure rocprofv3's PMC CSVs actually get written to disk:

    1. SIGINT to the whole process group. vLLM and rocprofv3 both install
       SIGINT handlers that do orderly shutdown.

    2. Wait up to ``sigint_grace_s`` for the group to actually drain.

    3. SIGKILL fallback only if the group still has live members at the
       end of the grace window.
    """
    
    pgid = leader_pid
    try:
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        # Group is empty (leader exited AND all backgrounded children
        # already gone). Nothing to wait for.
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

    # Group still alive after the SIGINT grace window — escalate to SIGKILL.
    if console:
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
