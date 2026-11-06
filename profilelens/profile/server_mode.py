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


_CMD_POS = r"(?:^|[(;&|`]|&&|\|\|)\s*"

_SERVER_PATTERNS: list[tuple[str, str]] = [
    (_CMD_POS + r"vllm\s+serve\b", "vllm"),
    (_CMD_POS + r"tritonserver\b", "tritonserver"),
    (_CMD_POS + r"python3?\s+-m\s+vllm\.entrypoints\b", "python-vllm"),
]


@dataclass(frozen=True)
class ServerInfo:

    cmd_name: str
    script_path: str

_MAX_DEPTH = 3

_MAX_GLOB_HITS = 32

_SCRIPT_TOKEN_RE = re.compile(r"""[\w./$~{}-]+\.(?:sh|bash)\b""")

_ASSIGN_RE = re.compile(
    r"""^\s*(?:export\s+)?([A-Za-z_]\w*)=(?:"([^"`$\n]*(?:\$\{?\w+\}?[^"`\n]*)*)"|'([^'\n]*)'|([^\s;#|&`(]+))"""
)

_SOURCE_RE = re.compile(r"""^\s*(?:source|\.)\s+("?)([^\s"';|&#]+)\1""")

# $name, ${name} and ${name:...}, including positional ${1}.
_VAR_RE = re.compile(r"\$\{(\w+)(?:[:#%/^,!][^}]*)?\}|\$(\w+)")


def _expand(token: str, variables: dict[str, str]) -> str:

    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        return variables.get(name, m.group(0))

    for _ in range(3):
        expanded = _VAR_RE.sub(sub, token)
        if expanded == token:
            break
        token = expanded
    return token


def _collect_variables(text: str, seed: dict[str, str]) -> dict[str, str]:
    variables = dict(seed)
    for line in text.splitlines():
        m = _ASSIGN_RE.match(line)
        if not m:
            continue
        value = _expand(m.group(2) or m.group(3) or m.group(4) or "", variables)
        if "$" in value:
            variables.pop(m.group(1), None)
            continue
        variables[m.group(1)] = value
    return variables


def _resolve_candidates(
    token: str, variables: dict[str, str], base_dirs: list[Path]
) -> list[Path]:
    expanded = _expand(token, variables)
    pattern = _VAR_RE.sub("*", expanded).replace("~", str(Path.home()))

    hits: list[Path] = []
    roots = [Path("/")] if pattern.startswith("/") else base_dirs
    rel = pattern.lstrip("/")
    for root in roots:
        try:
            for hit in sorted(root.glob(rel))[:_MAX_GLOB_HITS]:
                if hit.is_file() and hit not in hits:
                    hits.append(hit)
        except (OSError, ValueError):
            continue
        if hits:
            break
    return hits[:_MAX_GLOB_HITS]


def _scan_script(
    path: Path,
    base_dirs: list[Path],
    seed_vars: dict[str, str],
    visited: set[Path],
    depth: int,
) -> ServerInfo | None:
    """Look for a server launch in ``path``, then in the scripts it reaches."""
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if resolved in visited or depth > _MAX_DEPTH:
        return None
    visited.add(resolved)

    try:
        content = resolved.read_text(errors="replace")
    except OSError:
        return None

    lines = [
        stripped
        for stripped in (line.lstrip() for line in content.splitlines())
        if stripped and not stripped.startswith("#")
    ]

    for stripped in lines:
        for pat, name in _SERVER_PATTERNS:
            if re.search(pat, stripped):
                return ServerInfo(cmd_name=name, script_path=str(resolved))

    if depth == _MAX_DEPTH:
        return None

    variables = _collect_variables(content, seed_vars)

    child_bases = list(base_dirs)
    if resolved.parent not in child_bases:
        child_bases.append(resolved.parent)
    for stripped in lines:
        m = _SOURCE_RE.match(stripped)
        if not m:
            continue
        for env_path in _resolve_candidates(m.group(2), variables, child_bases):
            try:
                variables = _collect_variables(
                    env_path.read_text(errors="replace"), variables
                )
            except OSError:
                continue

    for value in variables.values():
        if value.startswith("/") and "*" not in value:
            d = Path(value)
            try:
                if d.is_dir() and d not in child_bases:
                    child_bases.append(d)
            except OSError:
                continue

    seen_tokens: set[str] = set()
    for stripped in lines:
        for token in _SCRIPT_TOKEN_RE.findall(stripped):
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            for child in _resolve_candidates(token, variables, child_bases):
                found = _scan_script(
                    child, child_bases, variables, visited, depth + 1
                )
                if found:
                    return found

    return None


def detect_server(user_command: list[str]) -> ServerInfo | None:
    """Detect a server launch by following the launcher script chain."""
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

    seed = {k: v for k, v in os.environ.items() if v and "\n" not in v}
    # Seed $1.. with the script's own arguments.
    args = user_command[2:] if os.path.basename(user_command[0]) in {"bash", "sh"} \
        else user_command[1:]
    for i, arg in enumerate(args, start=1):
        seed[str(i)] = arg

    return _scan_script(
        p,
        base_dirs=[Path.cwd(), p.parent],
        seed_vars=seed,
        visited=set(),
        depth=0,
    )


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
    else:  # pragma: no cover
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
