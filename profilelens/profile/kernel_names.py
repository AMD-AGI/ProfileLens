# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-name normalization across the Phase 1 / Phase 2 namespace boundary."""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

# Kineto reports the kernel descriptor (.kd); rocprofv3 the kernel symbol.
_KD_SUFFIXES = (" [clone .kd]", "[clone .kd]", ".kd")


def strip_descriptor_suffix(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    for suffix in _KD_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)].rstrip()
    return name


def canonical_kernel_name(name: str) -> str:
    """Strip descriptor decoration without changing exact graph-symbol hashes."""
    name = (name or "").strip()
    if name.startswith("void "):
        name = name[5:]
    return strip_descriptor_suffix(name)


@lru_cache(maxsize=1)
def _demangler_path() -> str | None:
    """Prefer LLVM's spelling, then use the standard GNU binutils demangler."""
    tool = shutil.which("llvm-cxxfilt")
    if tool:
        return tool
    rocm_tool = Path("/opt/rocm/llvm/bin/llvm-cxxfilt")
    if rocm_tool.is_file() and os.access(rocm_tool, os.X_OK):
        return str(rocm_tool)
    return shutil.which("c++filt")


@lru_cache(maxsize=8192)
def _demangle_symbol(name: str) -> str:
    """Demangle an Itanium C++ symbol via stdin; return the input on failure."""
    if not name.startswith("_Z") or any(char in name for char in "\r\n\0"):
        return name
    tool = _demangler_path()
    if tool is None:
        return name
    try:
        result = subprocess.run(
            [tool, "--no-strip-underscore"], input=name + "\n",
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return name
    output = result.stdout.strip()
    if result.returncode or not output or "\n" in output or "\r" in output:
        return name
    return output


def boundary_kernel_name(name: str) -> str:
    """A text identity for trace/counter joins, separate from raw symbol hashes."""
    return canonical_kernel_name(_demangle_symbol(canonical_kernel_name(name)))


def kernel_name_variants(name: str) -> tuple[str, ...]:
    """Return exact, descriptor-free and demangled spellings of a kernel name."""
    original = (name or "").strip()
    if not original:
        return ()
    canonical = canonical_kernel_name(original)
    demangled = _demangle_symbol(canonical)
    demangled_canonical = canonical_kernel_name(demangled)
    return tuple(dict.fromkeys(variant for variant in (
        original, strip_descriptor_suffix(original), canonical,
        demangled, demangled_canonical,
        f"void {canonical}", f"void {demangled_canonical}",  # expose conflicts in both spellings
    ) if variant))


def add_canonical_aliases(mapping: dict) -> dict:
    """Add unambiguous name aliases to *mapping*, keeping original keys."""
    aliases: dict = {}
    ambiguous: set[str] = set()
    for key, value in list(mapping.items()):
        for alias in kernel_name_variants(key):
            if alias in mapping:
                continue
            if alias in aliases and aliases[alias] != value:
                ambiguous.add(alias)
            else:
                aliases[alias] = value
    mapping.update({key: value for key, value in aliases.items() if key not in ambiguous})
    return mapping


def lookup_kernel(mapping: dict, name: str, default=None):
    """Look up an exact key or consistent spellings in an alias-expanded mapping."""
    if not mapping or not name:
        return default
    hit = mapping.get(name)
    if hit is not None:
        return hit
    hits = [mapping[alias] for alias in kernel_name_variants(name)
            if alias in mapping and mapping[alias] is not None]
    if not hits or any(value != hits[0] for value in hits[1:]):
        return default
    return hits[0]
