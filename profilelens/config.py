# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configuration management for ProfileLens."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


CONFIG_DIR = Path.home() / ".profilelens"
CONFIG_FILE = CONFIG_DIR / "config.toml"

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: str) -> str:
    """Replace ${VAR} references with their environment values."""

    def _replace(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), "")

    return _ENV_VAR_RE.sub(_replace, value)


def _expand_env_recursive(obj: Any) -> Any:
    """Walk a nested dict/list and expand env vars in all string values."""
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_recursive(v) for v in obj]
    return obj


@dataclass(frozen=True)
class ProfilingConfig:
    """Settings for the profiling pipeline."""

    rocprofv3_path: str = ""
    default_top_k: int = 10
    roofline_cache_dir: str = str(Path.home() / ".profilelens" / "roofline_cache")


@dataclass(frozen=True)
class ProfileLensConfig:
    """Top-level immutable configuration."""

    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)


def _build_profiling_config(raw: dict[str, Any]) -> ProfilingConfig:
    return ProfilingConfig(
        rocprofv3_path=raw.get("rocprofv3_path", ProfilingConfig.rocprofv3_path),
        default_top_k=raw.get("default_top_k", ProfilingConfig.default_top_k),
        roofline_cache_dir=raw.get("roofline_cache_dir", ProfilingConfig.roofline_cache_dir),
    )


_cached_config: ProfileLensConfig | None = None


def load_config(path: Path | None = None) -> ProfileLensConfig:
    """Load config from *path* (defaults to ~/.profilelens/config.toml).

    Returns:
        ProfileLensConfig with env-vars expanded and defaults filled in.
    """
    global _cached_config
    if _cached_config is not None and path is None:
        return _cached_config

    target = path or CONFIG_FILE
    if target.exists():
        with open(target, "rb") as f:
            raw = tomllib.load(f)
        raw = _expand_env_recursive(raw)
    else:
        raw = {}

    config = ProfileLensConfig(
        profiling=_build_profiling_config(raw.get("profiling", {})),
    )

    if path is None:
        _cached_config = config
    return config
