# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fetch, run and cache the rocprof-compute roofline benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import types
from pathlib import Path

from rich.console import Console

ROOFLINE_REPO = "https://github.com/ROCm/rocm-systems.git"
# rocm-systems PR #6872 (gfx950 FP8 MFMA roofline fix).
ROOFLINE_REF = "c0a79552b50a959172bea2ed224ff9366f47681b"

_SPARSE_PATHS = (
    "projects/rocprofiler-compute/src/roofline",
    "projects/rocprofiler-compute/src/utils",
)
_SRC_SUBDIR = "projects/rocprofiler-compute/src"


def bench_cache_root() -> Path:
    """Cache root for fetched benchmark code under the current ProfileLens home."""
    from profilelens.config import profilelens_home

    return profilelens_home() / "roofline_bench"


def bench_dir(ref: str = ROOFLINE_REF) -> Path:
    """Cache directory for a specific pinned ref (so a new pin re-fetches clean)."""
    return bench_cache_root() / ref


def bench_src_path(ref: str = ROOFLINE_REF) -> Path:
    """Path to pass as the benchmark import root (``.../rocprofiler-compute/src``)."""
    return bench_dir(ref) / _SRC_SUBDIR


def is_fetched(ref: str = ROOFLINE_REF) -> bool:
    """True if the pinned benchmark code is already cached (idempotency guard)."""
    return (bench_dir(ref) / ".complete").is_file()


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=600,
    )


def ensure_roofline_bench(
    console: Console | None = None,
    force: bool = False,
    ref: str = ROOFLINE_REF,
) -> Path:
    """Ensure the roofline benchmark is cached; return its ``src`` path."""
    d = bench_dir(ref)
    src = bench_src_path(ref)
    if is_fetched(ref) and not force:
        return src

    if console:
        console.print(
            f"  [bold]Fetching rocprof-compute roofline benchmark[/bold] "
            f"[dim](pinned {ref[:12]}, one-time)[/dim]"
        )

    if d.exists() and force:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)

    if not (d / ".git").is_dir():
        r = _run_git(["init", "-q"], d)
        if r.returncode != 0:
            raise RuntimeError(f"git init failed: {r.stderr.strip()}")
        _run_git(["remote", "add", "origin", ROOFLINE_REPO], d)

    _run_git(["sparse-checkout", "init", "--cone"], d)
    sp = _run_git(["sparse-checkout", "set", *_SPARSE_PATHS], d)
    if sp.returncode != 0:
        raise RuntimeError(f"git sparse-checkout failed: {sp.stderr.strip()}")

    fetch = _run_git(
        ["-c", "protocol.version=2", "fetch", "--depth", "1",
         "--filter=blob:none", "origin", ref],
        d,
    )
    if fetch.returncode != 0:
        raise RuntimeError(
            "Failed to fetch the roofline benchmark from GitHub "
            f"({ROOFLINE_REPO}@{ref[:12]}).\n"
            f"  git: {fetch.stderr.strip()[-400:]}\n"
            " run `profilelens setup-roofline`"
        )
    co = _run_git(["checkout", "-q", "FETCH_HEAD"], d)
    if co.returncode != 0:
        raise RuntimeError(f"git checkout failed: {co.stderr.strip()}")

    if not src.is_dir():
        raise RuntimeError(
            f"Roofline benchmark fetched but expected path missing: {src}"
        )
    (d / ".complete").write_text(ref + "\n")
    if console:
        console.print(f"  [dim]Cached roofline benchmark at {d}[/dim]")
    return src


_ROOFLINE_RESULT_SCHEMA = "rpc-jit-gbps-v1"


def get_roofline_cache_path(cache_dir: str, gpu_model: str) -> Path:
    """Return the path where roofline.csv should be cached for a GPU model."""
    return Path(cache_dir) / gpu_model.lower() / "roofline.csv"


def _result_meta_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def _result_cache_valid(csv_path: Path) -> bool:
    """A cached roofline.csv is reusable only if its meta matches the current benchmark."""
    meta = _result_meta_path(csv_path)
    if not (csv_path.exists() and meta.is_file()):
        return False
    try:
        m = json.loads(meta.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return m.get("schema") == _ROOFLINE_RESULT_SCHEMA and m.get("ref") == ROOFLINE_REF


def run_roofline_benchmark(
    cache_dir: str,
    gpu_model: str,
    console: Console | None = None,
    force: bool = False,
    device_index: int = 0,
) -> Path:
    """Run the roofline microbenchmark on an index in the visible device list."""
    csv_path = get_roofline_cache_path(cache_dir, gpu_model)
    if _result_cache_valid(csv_path) and not force:
        if console:
            console.print(f"  [dim]Using cached roofline data: {csv_path}[/dim]")
        return csv_path

    bench_src = ensure_roofline_bench(console=console)
    workload_dir = csv_path.parent
    os.makedirs(workload_dir, exist_ok=True)

    # Run as a script so the subprocess never imports profilelens.
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--src", str(bench_src),
        "--device", str(device_index),
        "--out", str(csv_path),
    ]
    env = os.environ.copy()

    if console:
        console.print(
            f"  [bold]Running roofline microbenchmark on GPU {device_index} "
            f"(may take ~1-2 min)...[/bold]"
        )

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Roofline benchmark timed out after 1800 secs")

    if result.returncode != 0 or not csv_path.exists():
        raise RuntimeError(
            f"Roofline benchmark failed (exit {result.returncode}):\n"
            f"stderr: {(result.stderr or '')[-1200:]}\n"
            f"stdout: {(result.stdout or '')[-400:]}"
        )

    _result_meta_path(csv_path).write_text(
        json.dumps({"schema": _ROOFLINE_RESULT_SCHEMA, "ref": ROOFLINE_REF,
                    "gpu_model": gpu_model})
    )
    if console:
        console.print(f"  [dim]Roofline data saved to: {csv_path}[/dim]")
    return csv_path


def _install_vendored_yaml_shim() -> None:
    """Make ``from vendored import yaml`` resolve to system pyyaml."""
    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"pyyaml is required to run the roofline benchmark: {e}")
    mod = types.ModuleType("vendored")
    mod.yaml = yaml  # type: ignore[attr-defined]
    mod.__all__ = ["yaml"]
    sys.modules["vendored"] = mod


def _run_benchmark_main() -> int:
    """Run the fetched benchmark in this isolated process (script entry)."""
    ap = argparse.ArgumentParser(description="Run the fetched roofline benchmark.")
    ap.add_argument("--src", required=True, help="rocprofiler-compute/src path")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", required=True, help="roofline.csv output path")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"--src not found: {src}")

    self_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [
        p for p in sys.path
        if p not in ("", ".") and str(Path(p).resolve()) != self_dir
    ]
    sys.path.insert(0, str(src))
    _install_vendored_yaml_shim()

    from roofline.run_benchmark import load_bench
    from utils.hip_interface import hipSetDevice

    hipSetDevice(args.device)
    bench = load_bench(args.device, {})
    bench.unsupported_data_types = list(
        getattr(bench, "unsupported_data_types", [])
    ) + ["MALL", "L2", "L1", "LDS"]
    use_fullrate_fp8(bench)
    metrics = bench.run_benchmark(args.device)
    bench.dump_csv(metrics, str(Path(args.out)))
    return 0


def use_fullrate_fp8(bench) -> None:
    """On gfx950, select the existing full-rate FP8 kernel instead of gfx942's path."""
    if "32x32x64" not in getattr(bench, "matrix_f8f6f4_src", ""):
        return
    if bench.matrix_kernel_selector.get("F8") != "mfma_f8":
        return
    bench.matrix_kernel_selector["F8"] = "mfma_f8f6f4<FP8_E4M3>"
    bench.matrix_f8_src = bench.matrix_f8f6f4_src
    bench.matrix_ops["F8"] = bench.matrix_ops["F4"]
    print("[profilelens] FP8 roofline using mfma_f8f6f4<FP8_E4M3> (see PR #6872)")


if __name__ == "__main__":
    raise SystemExit(_run_benchmark_main())
