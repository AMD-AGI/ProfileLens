# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Discover AMD GPU properties from the installed ROCprofiler-SDK."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class GpuInfo:
    """Runtime-reported properties needed by ProfileLens metric formulas."""

    arch: str
    model: str
    cu_count: int
    xcc_count: int
    simd_per_cu: int
    max_waves_per_cu: int
    wave_size: int


def _positive_int(fields: dict[str, str], key: str) -> int:
    value = fields.get(key, "")
    try:
        parsed = int(value, 0)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _parse_agent_info(output: str) -> GpuInfo:
    """Parse one ``rocprofv3-avail list --agent`` GPU section."""
    fields: dict[str, str] = {}
    for raw_line in output.splitlines():
        key, separator, value = raw_line.partition(":")
        if not separator:
            continue
        normalized_key = key.strip().lower()
        if normalized_key == "gpu" and fields:
            break
        fields[normalized_key] = value.strip()

    arch_match = re.search(
        r"gfx[0-9a-z]+",
        " ".join((fields.get("name", ""), fields.get("gfx_target_version", ""))),
        flags=re.IGNORECASE,
    )
    arch = arch_match.group(0).lower() if arch_match else ""

    model = next(
        (
            value
            for key in ("product_name", "model_name", "name")
            if (value := fields.get(key, "")).lower()
            not in {"", "n/a", "none", "unknown", "ip discovery"}
        ),
        "",
    )
    values = {
        "arch": arch,
        "model": model,
        "cu_count": _positive_int(fields, "cu_count"),
        "xcc_count": _positive_int(fields, "num_xcc"),
        "simd_per_cu": _positive_int(fields, "simd_per_cu"),
        "max_waves_per_cu": _positive_int(fields, "max_waves_per_cu"),
        "wave_size": _positive_int(fields, "wave_front_size"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(
            "rocprofv3-avail did not report required GPU properties: "
            + ", ".join(missing)
        )

    return GpuInfo(**values)


@lru_cache(maxsize=8)
def detect_gpu(device_index: int = 0, rocprofv3_path: str = "") -> GpuInfo:
    """Discover a GPU without substituting product- or architecture-specific defaults."""
    from profilelens.profile.rocprofv3 import find_rocprofv3_avail

    avail = find_rocprofv3_avail(rocprofv3_path)
    try:
        result = subprocess.run(
            [avail, "-d", str(device_index), "list", "--agent"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not query GPU {device_index} with {avail}: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"rocprofv3-avail failed while querying GPU {device_index}: {detail}"
        )
    try:
        return _parse_agent_info(result.stdout)
    except ValueError as exc:
        raise RuntimeError(
            f"incomplete runtime hardware discovery for GPU {device_index}: {exc}"
        ) from exc
