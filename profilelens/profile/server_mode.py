# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Auto-detect and orchestrate server-style workloads under rocprofv3."""

from __future__ import annotations

import os
import re
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path


_SERVER_PATTERNS: list[tuple[str, str]] = [
    (r"^\s*vllm\s+serve\b", "vllm"),
    (r"^\s*tritonserver\b", "tritonserver"),
    (r"^\s*python3?\s+-m\s+vllm\.entrypoints\b", "python-vllm"),
]


@dataclass(frozen=True)
class ServerInfo:
    """Detection result describing a server-style workload."""

    cmd_name: str
    """Friendly server name (``vllm``, ``tritonserver``, ...)."""

    script_path: str
    """Absolute path to the user's bash script."""


def detect_server(user_command: list[str]) -> ServerInfo | None:
    """Inspect ``user_command`` for a known server-style workload."""
    if not user_command:
        return None

    script: str | None = None
    if len(user_command) >= 2 and os.path.basename(user_command[0]) in {"bash", "sh"}:
        script = user_command[1]
    elif len(user_command) >= 1 and user_command[0].endswith((".sh", ".bash")):
        script = user_command[0]

    if not script:
        return None

    p = Path(script)
    if not p.is_file():
        return None

    try:
        content = p.read_text(errors="replace")
    except OSError:
        return None

    for line in content.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        for pat, name in _SERVER_PATTERNS:
            if re.search(pat, stripped):
                return ServerInfo(cmd_name=name, script_path=str(p.resolve()))

    return None


def build_wrapper_script(
    info: ServerInfo,
    rocprofv3_cmd: list[str],
    wrapper_path: Path,
    original_command: list[str],
) -> list[str]:
    """Write a bash prelude that intercepts the server command."""
    if not rocprofv3_cmd:
        raise ValueError("rocprofv3_cmd must include the rocprofv3 binary and its arguments")

    rocp_quoted = " ".join(shlex.quote(a) for a in rocprofv3_cmd)
    user_cmd_quoted = " ".join(shlex.quote(a) for a in original_command)

    if info.cmd_name == "vllm":
        intercept_block = f"""vllm() {{
    local _vllm_bin
    _vllm_bin=$(command -v vllm 2>/dev/null)
    if [ -z "$_vllm_bin" ]; then
        echo "[profilelens] vllm not found on PATH" >&2
        return 127
    fi
    if [ "${{1-}}" = "serve" ]; then
        shift
        echo "[profilelens] intercepting 'vllm serve' with rocprofv3" >&2
        exec {rocp_quoted} -- "$_vllm_bin" serve "$@"
    else
        exec "$_vllm_bin" "$@"
    fi
}}
export -f vllm
"""
    elif info.cmd_name == "tritonserver":
        intercept_block = f"""tritonserver() {{
    local _bin
    _bin=$(command -v tritonserver 2>/dev/null)
    if [ -z "$_bin" ]; then
        echo "[profilelens] tritonserver not found on PATH" >&2
        return 127
    fi
    echo "[profilelens] intercepting tritonserver with rocprofv3" >&2
    exec {rocp_quoted} -- "$_bin" "$@"
}}
export -f tritonserver
"""
    elif info.cmd_name == "python-vllm":
        intercept_block = f"""python_vllm_intercept() {{
    echo "[profilelens] intercepting 'python -m vllm.entrypoints' with rocprofv3" >&2
    exec {rocp_quoted} -- python3 "$@"
}}
python() {{
    if [ "${{1-}}" = "-m" ] && [ "${{2-}}" = "vllm.entrypoints.openai.api_server" ]; then
        python_vllm_intercept "$@"
    else
        exec /usr/bin/env python "$@"
    fi
}}
python3() {{
    if [ "${{1-}}" = "-m" ] && [ "${{2-}}" = "vllm.entrypoints.openai.api_server" ]; then
        python_vllm_intercept "$@"
    else
        exec /usr/bin/env python3 "$@"
    fi
}}
export -f python python3
"""
    else:  # pragma: no cover — should never happen
        raise ValueError(f"unsupported server type: {info.cmd_name}")

    prelude = f"""#!/usr/bin/env bash
# profilelens server-mode wrapper — auto-generated.

set -e
set -o pipefail
{intercept_block}
exec {user_cmd_quoted}
"""

    wrapper_path.write_text(prelude)
    wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return ["bash", str(wrapper_path)]
