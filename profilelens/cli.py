# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ProfileLens CLI."""

from __future__ import annotations

import json
import math
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from profilelens import __version__
from profilelens.config import load_config
from profilelens.dtypes import QueryResult
from profilelens.trace.formatter import format_json, format_markdown, print_rich
from profilelens.trace.processor import TraceProc
from profilelens.trace.queries import get_default_queries, get_profiling_queries

app = typer.Typer(
    name="profilelens",
    help="GPU trace analysis and hardware-counter profiling with Perfetto SQL.",
    no_args_is_help=True,
)
console = Console()


class OutputFormat(str, Enum):
    rich = "rich"
    json = "json"
    markdown = "markdown"


def _validate_trace(path: str) -> TraceProc:
    """Open a trace file, exit with a clear message on failure."""
    p = Path(path)
    if not p.exists():
        console.print(f"[bold red]Error:[/] trace file not found: {path}")
        raise typer.Exit(1)
    try:
        return TraceProc(str(p))
    except Exception as exc:
        console.print(f"[bold red]Error:[/] failed to open trace: {exc}")
        raise typer.Exit(1) from exc


def _run_prebuilt(lens: TraceProc) -> list[QueryResult]:
    """Execute pre-built SQL queries and build derived tables in Python."""
    import time as _time
    from profilelens.trace.queries import (
        build_category_summary, build_cpu_op_gpu_time, build_shape_breakdowns,
    )

    visible_results: list[QueryResult] = []
    internal: dict[str, QueryResult] = {}

    for cq in get_default_queries():
        t0 = _time.time()
        qr, _ = lens.query(cq.sql, label=cq.label)
        elapsed = _time.time() - t0
        rows = len(qr.rows) if qr.rows else 0
        err = f" ERROR: {qr.error}" if qr.error else ""
        console.print(f"    [dim]{cq.label}: {rows} rows ({elapsed:.1f}s){err}[/dim]")
        if not cq.internal:
            visible_results.append(qr)
        internal[cq.label] = qr

    t0 = _time.time()
    cpu_ext = internal.get("CPU Op External IDs")
    gpu_ext = internal.get("GPU Kernel External IDs")
    dims = internal.get("CPU Op Input Dimensions")
    id_names = internal.get("CPU Op ID to Name Mapping")

    if cpu_ext and gpu_ext:
        cpu_op_gpu = build_cpu_op_gpu_time(cpu_ext, gpu_ext)
        visible_results.append(cpu_op_gpu)
        console.print(f"    [dim]CPU Op GPU Kernel Time (Python): {len(cpu_op_gpu.rows)} rows[/dim]")

        parent_chain = _build_parent_chain(cpu_ext, gpu_ext)
        if parent_chain and id_names:
            cat_summary = build_category_summary(parent_chain, id_names)
            visible_results.append(cat_summary)
            console.print(f"    [dim]Compute Category Summary: {len(cat_summary.rows)} rows[/dim]")

        if dims:
            gemm_bd, op_bd = build_shape_breakdowns(cpu_ext, gpu_ext, dims)
            visible_results.append(gemm_bd)
            visible_results.append(op_bd)
            console.print(f"    [dim]GEMM Shape Breakdown: {len(gemm_bd.rows)} rows[/dim]")
            console.print(f"    [dim]Op Shape Breakdown: {len(op_bd.rows)} rows[/dim]")

    console.print(f"    [dim]Python post-processing: {_time.time() - t0:.1f}s[/dim]")

    input_types = internal.get("CPU Op Input Types")
    if input_types and input_types.rows:
        visible_results.append(input_types)

    return visible_results


def _run_profiling_queries(lens: TraceProc) -> list[QueryResult]:
    """Run only the queries needed for the profiling pipeline (fast path)."""
    import time as _time

    results: list[QueryResult] = []
    for cq in get_profiling_queries():
        t0 = _time.time()
        qr, _ = lens.query(cq.sql, label=cq.label)
        elapsed = _time.time() - t0
        rows = len(qr.rows) if qr.rows else 0
        err = f" ERROR: {qr.error}" if qr.error else ""
        console.print(f"    [dim]{cq.label}: {rows} rows ({elapsed:.1f}s){err}[/dim]")
        results.append(qr)
    return results


def _run_merge_queries(lens: TraceProc) -> list[QueryResult]:
    """Merge multi-rank analyses."""
    from profilelens.trace.queries import (
        AGGREGATE_KERNEL_TIME,
        KERNEL_DISPATCH_DURATIONS,
    )

    out: list[QueryResult] = []
    for cq in (AGGREGATE_KERNEL_TIME, KERNEL_DISPATCH_DURATIONS):
        qr, _ = lens.query(cq.sql, label=cq.label)
        out.append(qr)
    return out


def _build_parent_chain(cpu_ext: "QueryResult", gpu_ext: "QueryResult") -> "QueryResult":
    """Build parent-chain data from the simple External id queries."""
    from profilelens.dtypes import QueryResult

    ext_to_gpu_dur: dict[int, float] = {}
    for row in (gpu_ext.rows or ()):
        eid = row["ext_id"]
        ext_to_gpu_dur[eid] = ext_to_gpu_dur.get(eid, 0) + (row["dur"] or 0)

    rows = []
    for row in (cpu_ext.rows or ()):
        eid = row["ext_id"]
        gpu_ns = ext_to_gpu_dur.get(eid, 0)
        if gpu_ns > 0:
            rows.append({
                "cpu_id": row["id"],
                "cpu_op": row["name"],
                "parent_id": row.get("parent_id"),
                "gpu_ms": round(gpu_ns / 1_000_000, 4),
            })

    return QueryResult(
        sql="(Python join)", label="CPU Op Parent Chain with GPU Kernel Time",
        rows=tuple(rows),
        columns=("cpu_id", "cpu_op", "parent_id", "gpu_ms"),
    )


from profilelens.profile._proc_utils import WorkloadTimeout as _WorkloadTimeout  # noqa: E402
from profilelens.profile._proc_utils import reap_process_group as _reap_process_group
from profilelens.profile._proc_utils import wait_with_watchdog as _wait_with_watchdog  # noqa: E402


def _aggregate_phase_audit(audit_dir: Path) -> dict[str, int]:
    """Sum per-PID audit JSONs written by the ``_worker_shim/sitecustomize.py`` hooks."""
    totals: dict[str, int] = {"processes": 0}
    if not audit_dir.is_dir():
        return totals
    for path in audit_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        totals["processes"] += 1
        for k, v in data.items():
            if k in ("pid", "ppid"):
                continue
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
    return totals


def _output_results(results: list[QueryResult], fmt: OutputFormat) -> None:
    if fmt is OutputFormat.json:
        console.print(format_json(results))
    elif fmt is OutputFormat.markdown:
        console.print(format_markdown(results))
    else:
        print_rich(results, console)


def _load_external_ranking(path: str) -> list[str]:
    """Load an ordered, de-duplicated kernel ranking from JSON or plain text."""
    p = Path(path)
    if not p.exists():
        console.print(f"[bold red]Error:[/] ranking file not found: {path}")
        raise typer.Exit(1)
    text = p.read_text()
    names: list[str] = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    names.append(item.strip())
                elif isinstance(item, dict):
                    n = item.get("kernel_name") or item.get("name")
                    if n:
                        names.append(str(n).strip())
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    seen: set[str] = set()
    ordered: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def _load_trace_context(path: str) -> dict:
    """Load external per-kernel timing (e.g. TraceLens) keyed by kernel name."""
    p = Path(path)
    if not p.exists():
        console.print(f"[bold red]Error:[/] trace-context file not found: {path}")
        raise typer.Exit(1)
    data = json.loads(p.read_text())
    items = data if isinstance(data, list) else data.get("kernels", [])
    out: dict = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        name = (it.get("kernel_name") or it.get("name") or "").strip()
        if not name:
            continue
        if it.get("total_ms") is not None:
            total_ms = float(it["total_ms"])
        elif it.get("total_us") is not None:
            total_ms = float(it["total_us"]) / 1000.0
        else:
            continue
        count = int(it.get("count") or 0)
        out[name] = {
            "total_ms": total_ms, "count": count,
            "avg_ms": total_ms / count if count > 0 else 0.0,
        }
        if it.get("e2e_pct") is not None:
            pct = float(it["e2e_pct"])
            if not math.isfinite(pct) or pct < 0:
                raise ValueError(f"Invalid external e2e_pct for {name}: {it['e2e_pct']}")
            out[name]["e2e_pct"] = pct
    return out


def _apply_external_trace_context(
    analysis_results: "list[QueryResult]", ext: dict
) -> "list[QueryResult]":
    """Replace the per-kernel aggregate result with external timing."""
    from dataclasses import replace as _replace

    if not ext:
        return analysis_results
    out: list = []
    for r in analysis_results:
        if r.label == "Aggregate Time per Kernel Name":
            rows = tuple(
                {"name": name, **v}
                for name, v in ext.items()
            )
            out.append(_replace(r, rows=rows))
        else:
            out.append(r)
    return out


def _derive_ranking_and_dtypes(
    analysis_results: list[QueryResult],
    top_k: int,
    dtype_dir: str | None = None,
    external_ranking: list[str] | None = None,
):
    """Return (kernel_names, per_kernel_dtype, observed_dtypes, dtype) for profiling."""
    from profilelens.profile.dtype_detect import (
        UNKNOWN_DTYPE,
        dtype_from_query_result,
        merge_dtype_sidecar,
        resolve_kernel_dtypes,
    )
    from profilelens.profile.kernel_context import build_kernel_dtype_from_trace
    from profilelens.profile.runner import extract_top_k_from_results

    if external_ranking:
        kernel_names = external_ranking[:top_k]
    else:
        kernel_names = extract_top_k_from_results(analysis_results, top_k)

    dtype: str | None = None
    for r in analysis_results:
        if r.label == "CPU Op Input Types" and r.rows:
            dtype = dtype_from_query_result(r.rows)
            break
    if dtype is None:
        dtype = UNKNOWN_DTYPE

    sidecar = merge_dtype_sidecar(dtype_dir) if dtype_dir else {}
    trace_dtypes = build_kernel_dtype_from_trace(analysis_results)
    per_kernel_dtype, observed_dtypes = resolve_kernel_dtypes(
        kernel_names, sidecar, trace_dtypes, global_dtype=dtype,
    )
    if UNKNOWN_DTYPE in observed_dtypes:
        dtype = UNKNOWN_DTYPE
    elif observed_dtypes:
        dtype = observed_dtypes[0]
    return kernel_names, per_kernel_dtype, observed_dtypes, dtype


def _counter_options(counter_set: str, counters: str | None, dtypes: str | None):
    """Parse measurement configuration independently of dtype attribution."""
    from profilelens.profile.dtype_detect import normalize_counter_dtype_hints

    if counter_set not in {"auto", "custom"}:
        raise typer.BadParameter("--counter-set must be auto or custom")
    names = list(dict.fromkeys(c.strip() for c in (counters or "").split(",") if c.strip()))
    if counters is not None and not names:
        raise typer.BadParameter("--counters must contain at least one counter name")
    if names:
        counter_set = "custom"
    elif counter_set == "custom":
        raise typer.BadParameter("--counter-set custom requires --counters")
    try:
        hints = normalize_counter_dtype_hints(
            [dt.strip() for dt in (dtypes or "").split(",") if dt.strip()]
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if dtypes is not None and not hints:
        raise typer.BadParameter("--dtypes must contain at least one dtype")
    return counter_set, names or None, hints


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def profile(
    ctx: typer.Context,
    kernel_names: Optional[str] = typer.Option(None, "--kernel-names", help="Comma-separated kernel names/regexes to profile."),
    from_analysis: Optional[str] = typer.Option(None, "--from-analysis", help="Path to a prior analyze JSON output."),
    trace: Optional[str] = typer.Option(None, "--trace", help="Reuse an existing kineto/torch.profiler trace for sampling/context/dtype instead of a prior analyze JSON. Pair with --rank-from to let an upstream tool (e.g. TraceLens) choose the top-K."),
    rank_from: Optional[str] = typer.Option(None, "--rank-from", help="Path to an external kernel ranking (JSON list or one-name-per-line). Selects the top-K kernels; requires --trace for sampling/dtype context."),
    trace_context_from: Optional[str] = typer.Option(None, "--trace-context-from", help="Path to an external per-kernel timing JSON ({kernel_name,total_us,count}). Overrides ProfileLens's SQL-derived E2E time-share so the report attributes timing to the upstream ranker (e.g. TraceLens)."),
    dtype_dir: Optional[str] = typer.Option(None, "--dtype-dir", help="Directory of Phase-1 shim dtype sidecars (PROFILELENS_DTYPE_DIR) for dtype-aware counter selection of non-aten kernels."),
    top_k: int = typer.Option(10, "--top-k", help="Number of top kernels from analysis."),
    counter_set: str = typer.Option("auto", "--counter-set", help="Counter preset: auto or custom."),
    counters: Optional[str] = typer.Option(None, "--counters", help="Explicit counter names, comma-separated. Selects the custom counter set; requires single-pass SDK compatibility."),
    dtypes: Optional[str] = typer.Option(None, "--dtypes", help="Comma-separated counter-coverage hints, e.g. BF16,FP8,FP4. Does not assign a dtype to unresolved kernels."),
    run_roofline: bool = typer.Option(True, "--run-roofline/--no-roofline", help="Run roofline microbenchmark."),
    refresh_roofline: bool = typer.Option(False, "--refresh-roofline", help="Force re-run of roofline benchmark."),
    sample_dispatches: int = typer.Option(10, "--sample-dispatches", help="Dispatches to sample per kernel (0 = profile all)."),
    output_dir: str = typer.Option("./profilelens_run", "--output-dir", help="Output directory."),
    output: OutputFormat = typer.Option(OutputFormat.rich, "--output", "-o", help="Output format."),
    workload_timeout: float = typer.Option(
        0.0, "--workload-timeout",
        help="Total seconds to allow the workload to run; 0 (the default) means "
             "no limit -- profiler should not cap how long your workload legitimately takes to complete.",
    ),
    workload_idle_timeout: float = typer.Option(
        1800.0, "--workload-idle-timeout",
        help="Stop the workload if it makes no progress, no CPU time and no new "
             "output for this many seconds; 0 disables. Catches deadlocks "
             "without stopping long-running work.",
    ),
) -> None:
    """Profile GPU kernels with HW counters and empirical roofline analysis."""

    counter_set, custom_ctr, counter_dtype_hints = _counter_options(counter_set, counters, dtypes)

    output_dir = os.path.abspath(output_dir)
    dtype_dir = os.path.abspath(dtype_dir) if dtype_dir else None
    rank_from = os.path.abspath(rank_from) if rank_from else None
    from profilelens.profile.runner import run_profile as _run
    from profilelens.profile.report import (
        format_profiling_json,
        format_profiling_markdown,
        load_analysis_results,
        print_profiling_rich,
    )

    user_command = ctx.args
    if not user_command:
        console.print("[bold red]Error:[/] provide a workload command after --")
        raise typer.Exit(1)

    names: list[str] = []
    dtype: str | None = None
    analysis_results: list[QueryResult] | None = None
    per_kernel_dtype: dict[str, tuple[str, str]] | None = None
    observed_dtypes: list[str] | None = None

    if trace:
        lens = _validate_trace(trace)
        console.print(f"  Analyzing supplied trace: {Path(trace).name}")
        analysis_results = _run_profiling_queries(lens)
        if trace_context_from:
            ext_tc = _load_trace_context(trace_context_from)
            analysis_results = _apply_external_trace_context(analysis_results, ext_tc)
            console.print(
                f"  E2E time-share sourced from {Path(trace_context_from).name} "
                f"({len(ext_tc)} kernels) instead of Perfetto SQL"
            )
        external = _load_external_ranking(rank_from) if rank_from else None
        names, per_kernel_dtype, observed_dtypes, dtype = _derive_ranking_and_dtypes(
            analysis_results, top_k, dtype_dir=dtype_dir, external_ranking=external,
        )
        if not names:
            console.print("[bold red]Error:[/] no kernels selected from trace/ranking")
            raise typer.Exit(1)
        src = "ranking" if external else "trace top-K"
        console.print(
            f"  Selected {len(names)} kernels from {src}; "
            f"observed dtypes: {', '.join(observed_dtypes) or dtype}"
        )
    elif from_analysis:
        analysis_results = load_analysis_results(from_analysis) or None
        names, per_kernel_dtype, observed_dtypes, dtype = _derive_ranking_and_dtypes(
            analysis_results or [], top_k, dtype_dir=dtype_dir,
        )
        if not names:
            console.print("[bold red]Error:[/] no kernels found in analysis output")
            raise typer.Exit(1)
        console.print(f"  Loaded {len(names)} kernels from analysis, dtype={dtype}")
    elif rank_from:
        names = _load_external_ranking(rank_from)[:top_k]
        if not names:
            console.print("[bold red]Error:[/] no kernels found in --rank-from file")
            raise typer.Exit(1)
        console.print(
            f"  Loaded {len(names)} kernels from ranking "
            f"[yellow](no --trace: dispatch sampling/dtype context unavailable)[/yellow]"
        )
    elif kernel_names:
        names = [n.strip() for n in kernel_names.split(",") if n.strip()]
    else:
        console.print(
            "[bold red]Error:[/] provide --trace (+ --rank-from), --from-analysis, or --kernel-names"
        )
        raise typer.Exit(1)

    console.print(f"\n[bold]ProfileLens v{__version__}[/bold]  |  profile\n")

    cfg = load_config()
    profiling_cfg = {
        "rocprofv3_path": cfg.profiling.rocprofv3_path,
        "roofline_cache_dir": cfg.profiling.roofline_cache_dir,
    }

    try:
        result = _run(
            kernel_names=names,
            user_command=user_command,
            dtype=dtype,
            counter_set_mode=counter_set,
            custom_counters=custom_ctr,
            counter_dtype_hints=counter_dtype_hints,
            run_roofline=run_roofline,
            refresh_roofline=refresh_roofline,
            output_dir=output_dir,
            top_k=top_k,
            sample_dispatches=sample_dispatches,
            analysis_results=analysis_results,
            console=console,
            profiling_config=profiling_cfg,
            per_kernel_dtype=per_kernel_dtype,
            observed_dtypes=observed_dtypes,
            workload_timeout=workload_timeout,
            workload_idle_timeout=workload_idle_timeout,
        )
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    if trace_context_from:
        from dataclasses import replace as _replace
        result = _replace(result, trace_context_source="tracelens")

    if output is OutputFormat.json:
        console.print(format_profiling_json(result, analysis_results))
    elif output is OutputFormat.markdown:
        console.print(format_profiling_markdown(result, analysis_results))
    else:
        print_profiling_rich(result, console, analysis_results)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "report.json"
    json_path.write_text(format_profiling_json(result, analysis_results))

    md_path = out_path / "REPORT.md"
    md_path.write_text(format_profiling_markdown(result, analysis_results, output_dir=output_dir))

    console.print(f"\n[bold]Outputs saved to {output_dir}/:[/bold]")
    console.print("  report.json   - structured JSON report")
    console.print("  REPORT.md     - human-readable markdown")


@app.command(name="full-analysis", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def full_analysis(
    ctx: typer.Context,
    top_k: int = typer.Option(10, "--top-k", help="Top kernels to profile."),
    counter_set: str = typer.Option("auto", "--counter-set", help="Counter preset: auto or custom."),
    counters: Optional[str] = typer.Option(None, "--counters", help="Explicit counter names, comma-separated. Selects the custom counter set; requires single-pass SDK compatibility."),
    dtypes: Optional[str] = typer.Option(None, "--dtypes", help="Comma-separated counter-coverage hints, e.g. BF16,FP8,FP4. Does not assign a dtype to unresolved kernels."),
    output_dir: str = typer.Option("./profilelens_run", "--output-dir", help="Output directory."),
    output: OutputFormat = typer.Option(OutputFormat.rich, "--output", "-o", help="Output format."),
    run_roofline: bool = typer.Option(True, "--run-roofline/--no-roofline", help="Run roofline microbenchmark."),
    refresh_roofline: bool = typer.Option(False, "--refresh-roofline", help="Force re-run of roofline benchmark."),
    sample_dispatches: int = typer.Option(10, "--sample-dispatches", help="Dispatches to sample per kernel (0 = profile all)."),
    workload_timeout: float = typer.Option(
        0.0, "--workload-timeout",
        help="Total seconds to allow the workload to run; 0 (the default) means "
             "no limit -- profiler should not cap how long your workload legitimately takes to complete.",
    ),
    workload_idle_timeout: float = typer.Option(
        1800.0, "--workload-idle-timeout",
        help="Stop the workload if it makes no progress, no CPU time and no new "
             "output for this many seconds; 0 disables. Catches deadlocks "
             "without stopping long-running work.",
    ),
) -> None:
    """One-shot: trace collection + analysis + kernel profiling + consolidated report. The workload must have kineto/torch.profiler tracing enabled internally."""
    counter_set, custom_ctr, counter_dtype_hints = _counter_options(counter_set, counters, dtypes)
    import subprocess
    import time as _time

    output_dir = os.path.abspath(output_dir)

    from profilelens.profile.runner import (
        collect_recorded_traces,
        collect_region_paths,
        extract_top_k_from_results,
        merge_rank_analyses,
        run_profile as _run,
    )
    from profilelens.profile.dtype_detect import UNKNOWN_DTYPE, dtype_from_query_result
    from profilelens.profile.report import (
        format_profiling_json,
        format_profiling_markdown,
        print_profiling_rich,
    )

    user_command = ctx.args
    if not user_command:
        console.print("[bold red]Error:[/] provide a workload command after --")
        raise typer.Exit(1)

    console.print(f"\n[bold]ProfileLens v{__version__}[/bold]  |  full-analysis\n")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Measure roofline peaks first, on an idle GPU.
    if run_roofline:
        from profilelens.profile.gpu_detect import detect_gpu
        from profilelens.profile.roofline_bench import run_roofline_benchmark

        cfg_for_roofline = load_config()
        try:
            gpu_for_roofline = detect_gpu(
                rocprofv3_path=cfg_for_roofline.profiling.rocprofv3_path
            )
            cache_dir = cfg_for_roofline.profiling.roofline_cache_dir
            console.print(
                "[bold]Roofline pre-pass:[/bold] measuring/using cached peaks before Phase 1"
            )
            run_roofline_benchmark(
                cache_dir,
                gpu_for_roofline.model,
                console=console,
                force=refresh_roofline,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            console.print(
                f"[yellow]Roofline pre-pass skipped: {exc} (will retry after Phase 1)[/yellow]"
            )

    console.print("[bold]Phase 1:[/bold] Running workload for trace collection...")
    phase1_audit_dir = Path(output_dir) / ".profilelens_audit_phase1"
    phase1_audit_dir.mkdir(exist_ok=True)
    for old in phase1_audit_dir.glob("*.json"):
        old.unlink()

    shim_dir = str(Path(__file__).resolve().parent / "profile" / "_worker_shim")
    phase1_env = os.environ.copy()
    existing_pp = phase1_env.get("PYTHONPATH", "")
    phase1_env["PYTHONPATH"] = (
        shim_dir + os.pathsep + existing_pp if existing_pp else shim_dir
    )
    phase1_env["PROFILELENS_AUDIT_DIR"] = str(phase1_audit_dir)

    phase1_dtype_dir = Path(output_dir) / ".profilelens_dtype_phase1"
    phase1_dtype_dir.mkdir(exist_ok=True)
    for old in phase1_dtype_dir.glob("*.json"):
        old.unlink()
    phase1_env["PROFILELENS_DTYPE_DIR"] = str(phase1_dtype_dir)

    t_before = _time.time()
    try:
        proc = subprocess.Popen(user_command, env=phase1_env, start_new_session=True)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/] command not found: {exc}")
        raise typer.Exit(1) from exc
    try:
        returncode = _wait_with_watchdog(
            proc,
            wall_timeout=workload_timeout,
            idle_timeout=workload_idle_timeout,
        )
    except _WorkloadTimeout as exc:
        console.print(
            f"[bold red]Error:[/] Phase 1 workload {exc.reason}; stopping it."
        )
        _reap_process_group(proc.pid, console=console, label="Phase 1 (timeout)")
        raise typer.Exit(1) from None
    finally:
        _reap_process_group(proc.pid, console=console, label="Phase 1")
    phase1_workload_s = _time.time() - t_before
    console.print(f"  Workload exited with code {returncode}")
    console.print(f"  Workload duration (with tracing): {phase1_workload_s:.1f}s")

    audit_total = _aggregate_phase_audit(phase1_audit_dir)
    triggers = (
        audit_total.get("torch_profiler_start", 0)
        + audit_total.get("torch_cuda_profiler_start", 0)
    )
    if triggers == 0:
        console.print(
            "\n[bold red]Error:[/] no torch.profiler trigger fired in "
            f"{audit_total.get('processes', 0)} workload process(es).\n"
            "  profilelens needs your workload to wrap its post-warmup hot loop in:\n"
            "    [cyan]with torch.profiler.profile(...) as p:  # or llm.start_profile() / stop_profile()[/cyan]\n"
            "  Phase 2 (PMC capture) is gated on these calls — without them,\n"
            "  rocprofv3 --selected-regions would record an empty window.\n"
            f"  Per-PID audit JSONs: {phase1_audit_dir}/"
        )
        raise typer.Exit(1)
    console.print(
        f"  [dim]Trigger audit: {triggers} torch.profiler start call(s) "
        f"across {audit_total.get('processes', 0)} process(es)[/dim]"
    )

    rank_traces = collect_recorded_traces(phase1_audit_dir)
    if not rank_traces:
        console.print("[bold red]Error:[/] workload did not export a chrome trace via torch.profiler")
        console.print("  ProfileLens records the exact path from torch.profiler's export_chrome_trace")
        console.print("  (tensorboard_trace_handler routes through it too). Ensure your workload calls")
        console.print("  prof.export_chrome_trace(...) or sets on_trace_ready=tensorboard_trace_handler(...).")
        raise typer.Exit(1)
    from profilelens.profile.runner import enrich_rank_identities
    enrich_rank_identities(rank_traces)
    rank_traces.sort(key=lambda t: (t.rank if t.rank is not None else 1 << 30, t.path.name))
    primary = rank_traces[0]
    n_ranks = len(rank_traces)
    if n_ranks > 1:
        rk = ", ".join(str(t.rank) for t in rank_traces)
        console.print(
            f"  Found {n_ranks} rank traces (ranks {rk}); primary = "
            f"rank {primary.rank}: {primary.path.name}"
        )
    else:
        console.print(f"  Found trace: {primary.path}")

    shim_pids = {p.stem for p in phase1_dtype_dir.glob("*.json")}
    rank_manifest = [
        {
            "logical_rank": rt.rank,
            "trace_rank": rt.trace_rank,
            "world_size": rt.world_size,
            "tp": rt.tp,
            "ep": rt.ep,
            "phase1_worker_pid": rt.pid,
            "trace_file": rt.path.name,
            "shim_sidecar": (str(rt.pid) in shim_pids) if rt.pid is not None else False,
        }
        for rt in rank_traces
    ]
    if n_ranks > 1:
        joined = sum(1 for m in rank_manifest if m["shim_sidecar"])
        console.print(
            f"  [dim]Rank manifest: {joined}/{n_ranks} ranks joined to a Phase-1 "
            f"shim sidecar by PID[/dim]"
        )

    console.print("\n[bold]Phase 1:[/bold] Analyzing trace (fast profiling path)...")
    lens = _validate_trace(str(primary.path))
    import time as _t1
    t_analysis_start = _t1.time()
    analysis_results = _run_profiling_queries(lens)

    rank_breakdown: dict | None = None
    if n_ranks > 1:
        console.print(f"  Ingesting {n_ranks - 1} additional rank trace(s) for merged top-K/CV...")
        extra: list[tuple[int, list[QueryResult]]] = []
        for rt in rank_traces[1:]:
            try:
                rlens = _validate_trace(str(rt.path))
                extra.append((rt.rank if rt.rank is not None else -1, _run_merge_queries(rlens)))
                console.print(f"    [dim]rank {rt.rank}: merged[/dim]")
            except Exception as exc:
                console.print(f"    [yellow]rank {rt.rank}: skipped ({exc})[/yellow]")
        analysis_results, rank_breakdown = merge_rank_analyses(analysis_results, extra)
        if rank_breakdown.get("ep_asymmetric"):
            console.print(
                "  [yellow]EP-asymmetric: ranks run different top-K kernel sets "
                "(expert-sharded). Merged top-K spans all ranks.[/yellow]"
            )

    t_analysis = _t1.time() - t_analysis_start
    console.print(f"  Analysis completed in {t_analysis:.1f}s")

    if rank_breakdown is not None:
        rank_breakdown["rank_manifest"] = rank_manifest
        rank_breakdown["join_keys"] = {
            "phase1_shim_to_trace": "pid",
            "across_phases_and_replay": "logical_rank",
            "note": "Phase-1 and Phase-2 PIDs differ (separate launches).",
        }
        ranks_json = Path(output_dir) / "phase1_ranks.json"
        ranks_json.write_text(json.dumps(rank_breakdown, indent=2))
        console.print(f"  Per-rank Phase-1 breakdown + manifest -> {ranks_json.name}")

    kernel_names = extract_top_k_from_results(analysis_results, top_k)
    if not kernel_names:
        console.print("[bold red]Error:[/] no GPU kernels found in trace")
        raise typer.Exit(1)
    console.print(f"  Top-{top_k} kernels: {', '.join(n[:30] for n in kernel_names[:3])}...")

    dtype: str | None = None
    for r in analysis_results:
        if r.label == "CPU Op Input Types" and r.rows:
            dtype = dtype_from_query_result(r.rows)
            break
    if dtype is None:
        dtype = UNKNOWN_DTYPE

    from profilelens.profile.dtype_detect import (
        merge_dtype_sidecar,
        resolve_kernel_dtypes,
    )
    from profilelens.profile.kernel_context import build_kernel_dtype_from_trace

    sidecar = merge_dtype_sidecar(phase1_dtype_dir)
    trace_dtypes = build_kernel_dtype_from_trace(analysis_results)
    per_kernel_dtype, observed_dtypes = resolve_kernel_dtypes(
        kernel_names, sidecar, trace_dtypes, global_dtype=dtype,
    )
    if UNKNOWN_DTYPE in observed_dtypes:
        dtype = UNKNOWN_DTYPE
    elif observed_dtypes:
        dtype = observed_dtypes[0]
    n_triton = sum(1 for _, s in per_kernel_dtype.values() if s == "triton_hook")
    n_trace = sum(1 for _, s in per_kernel_dtype.values() if s == "trace_corr")
    n_name = sum(1 for _, s in per_kernel_dtype.values() if s == "name_fallback")
    console.print(
        f"  Kernel dtype resolution: {', '.join(observed_dtypes) or UNKNOWN_DTYPE} "
        f"[dim](top-K sources: {n_triton} triton, {n_trace} trace, "
        f"{n_name} name-fallback)[/dim]"
    )

    analysis_json = Path(output_dir) / "analysis.json"
    analysis_json.parent.mkdir(parents=True, exist_ok=True)
    from profilelens.trace.formatter import format_json
    analysis_json.write_text(format_json(analysis_results))

    console.print(f"\n[bold]Phase 2:[/bold] Profiling {len(kernel_names)} kernels...")

    cfg = load_config()
    profiling_cfg = {
        "rocprofv3_path": cfg.profiling.rocprofv3_path,
        "roofline_cache_dir": cfg.profiling.roofline_cache_dir,
    }

    phase1_regions = collect_region_paths(phase1_audit_dir)
    eff_sample = 0 if len(phase1_regions) > 1 else sample_dispatches
    if len(phase1_regions) > 1:
        console.print(
            f"  [dim]{len(phase1_regions)} scenarios -> per-scenario reports "
            f"(profiling all dispatches in selected regions)[/dim]"
        )

    try:
        prof_result = _run(
            kernel_names=kernel_names,
            user_command=user_command,
            dtype=dtype,
            counter_set_mode=counter_set,
            custom_counters=custom_ctr,
            counter_dtype_hints=counter_dtype_hints,
            run_roofline=run_roofline,
            refresh_roofline=refresh_roofline,
            output_dir=output_dir,
            top_k=top_k,
            sample_dispatches=eff_sample,
            analysis_results=analysis_results,
            console=console,
            profiling_config=profiling_cfg,
            per_kernel_dtype=per_kernel_dtype,
            observed_dtypes=observed_dtypes,
            workload_timeout=workload_timeout,
            workload_idle_timeout=workload_idle_timeout,
        )
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    from dataclasses import replace as _replace
    prof_result = _replace(prof_result, phase1_workload_s=round(phase1_workload_s, 1))

    phase2_audit_dir = Path(output_dir) / "profile" / ".profilelens_audit"
    p2_audit = _aggregate_phase_audit(phase2_audit_dir)
    p2_starts = (
        p2_audit.get("torch_profiler_start", 0)
        + p2_audit.get("torch_cuda_profiler_start", 0)
    )
    p2_resumes = p2_audit.get("roctx_resume", 0)
    if p2_starts > 0 and p2_resumes == 0:
        console.print(
            "  [bold yellow]Warning:[/] Phase 2 hook fired but roctx never emitted.\n"
            "    --selected-regions captured nothing. Check that\n"
            f"    PROFILELENS_ROCTX_SITE points to a directory containing the\n"
            "    'roctx' Python package (libpyroctx.so). Inspect:\n"
            f"    {phase2_audit_dir}/"
        )
    elif p2_starts > 0:
        console.print(
            f"  [dim]Phase 2 trigger audit: {p2_starts} start call(s), "
            f"{p2_resumes} roctx resume(s) across "
            f"{p2_audit.get('processes', 0)} process(es)[/dim]"
        )

    console.print("\n[bold]Consolidated Report[/bold]")
    if output is OutputFormat.json:
        console.print(format_profiling_json(prof_result, analysis_results))
    elif output is OutputFormat.markdown:
        console.print(format_profiling_markdown(
            prof_result, analysis_results,
            flydsl_args=sidecar.get("flydsl_args"),
            capture_graphs=sidecar.get("capture_graphs")))
    else:
        print_profiling_rich(prof_result, console, analysis_results)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    regions = prof_result.per_region_kernels
    if n_ranks <= 1 and len(regions) > 1:
        from dataclasses import replace as _replace
        region_paths = collect_region_paths(phase1_audit_dir)
        written = []
        for rname, kps in regions.items():
            mapped = region_paths.get(rname)
            folder_name = Path(mapped).name.split(".")[0] if mapped else rname
            folder = out_path / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            sub = _replace(prof_result, kernels=kps, per_region_kernels={})
            (folder / "report.json").write_text(format_profiling_json(sub, analysis_results))
            (folder / "REPORT.md").write_text(
                format_profiling_markdown(
                    sub, analysis_results, output_dir=str(folder),
                    flydsl_args=sidecar.get("flydsl_args"),
                    capture_graphs=sidecar.get("capture_graphs"),
                )
            )
            written.append(folder_name)
        console.print(
            f"  [bold]{len(written)} independent scenarios[/bold] -> per-scenario folders: "
            + ", ".join(written)
        )
    else:
        json_path = out_path / "report.json"
        json_path.write_text(format_profiling_json(prof_result, analysis_results))
        md_path = out_path / "REPORT.md"
        md_path.write_text(format_profiling_markdown(
            prof_result, analysis_results, output_dir=output_dir,
            flydsl_args=sidecar.get("flydsl_args"),
            capture_graphs=sidecar.get("capture_graphs")))

    phase2_workload_s = prof_result.profiling_duration_s
    overhead = (phase2_workload_s / phase1_workload_s - 1) * 100 if phase1_workload_s > 0 else 0
    console.print("\n[bold]Timing:[/bold]")
    console.print(f"  Phase 1 workload (tracing):   {phase1_workload_s:.1f}s")
    console.print(f"  Phase 2 workload (profiling):  {phase2_workload_s:.1f}s")
    if overhead >= 0:
        console.print(f"  Profiling overhead:            {overhead:+.0f}%")
    else:
        console.print(f"  Profiling overhead:            {overhead:.0f}%")

    console.print(f"\n[bold]Outputs saved to {output_dir}/:[/bold]")
    console.print("  report.json   - structured JSON report")
    console.print("  REPORT.md     - human-readable markdown")
    console.print("  analysis.json - Phase 1 trace analysis")


@app.command()
def analyze(
    trace_file: str = typer.Argument(..., help="Path to the Perfetto / PyTorch trace file."),
    output: OutputFormat = typer.Option(OutputFormat.rich, "--output", "-o", help="Output format."),
) -> None:
    """Run the deterministic pre-built trace analysis queries."""
    lens = _validate_trace(trace_file)

    console.print(f"[bold]ProfileLens v{__version__}[/bold]  |  {trace_file}\n")

    with console.status("Running pre-built queries..."):
        results = _run_prebuilt(lens)
    _output_results(results, output)


@app.command()
def query(
    trace_file: Optional[str] = typer.Argument(None, help="Path to the trace file (not needed with --server)."),
    sql: Optional[str] = typer.Argument(None, help="SQL query to execute."),
    output: OutputFormat = typer.Option(OutputFormat.rich, "--output", "-o", help="Output format."),
    server: Optional[str] = typer.Option(None, "--server", help="Connect to a running trace_processor_shell at host:port instead of loading a trace file."),
) -> None:
    """Execute an arbitrary SQL query against a trace.

    With --server, only the SQL argument is needed (no trace file).
    """
    if server:
        actual_sql = sql if sql else trace_file
        if not actual_sql:
            console.print("[bold red]Error:[/] SQL query argument is required.")
            raise typer.Exit(1)
        lens = TraceProc(addr=server)
    elif trace_file and sql:
        lens = _validate_trace(trace_file)
        actual_sql = sql
    else:
        console.print("[bold red]Error:[/] provide TRACE_FILE SQL, or --server SQL.")
        raise typer.Exit(1)
    qr, err = lens.query(actual_sql, label="user_query")
    _output_results([qr], output)
    if err:
        raise typer.Exit(1)


@app.command()
def tables(
    trace_file: str = typer.Argument(..., help="Path to the trace file."),
) -> None:
    """List available Perfetto tables in a trace."""
    lens = _validate_trace(trace_file)
    with console.status("Listing tables..."):
        tbl_list, err = lens.list_tables()
    if err:
        console.print(f"[bold red]Error:[/] {err}")
        raise typer.Exit(1)
    for name in tbl_list:
        console.print(f"  {name}")


@app.command()
def schema(
    trace_file: str = typer.Argument(..., help="Path to the trace file."),
    table: str = typer.Argument(..., help="Table name."),
    output: OutputFormat = typer.Option(OutputFormat.rich, "--output", "-o", help="Output format."),
) -> None:
    """Show the column schema for a table in the trace."""
    lens = _validate_trace(trace_file)
    cols, err = lens.get_schema(table)
    if err:
        console.print(f"[bold red]Error:[/] {err}")
        raise typer.Exit(1)
    if output is OutputFormat.json:
        console.print(json.dumps(cols, indent=2))
    else:
        from rich.table import Table

        tbl = Table(title=f"Schema: {table}")
        tbl.add_column("Column")
        tbl.add_column("Type")
        for c in cols:
            tbl.add_row(c["name"], c.get("type", ""))
        console.print(tbl)


@app.command(name="setup-roofline")
def setup_roofline(
    refresh: bool = typer.Option(False, "--refresh", help="Re-fetch the pinned benchmark code even if cached."),
    measure: bool = typer.Option(False, "--measure", help="Also run the microbenchmark now to populate the per-GPU peak cache."),
    device: int = typer.Option(0, "--device", help="GPU index to benchmark when --measure is set."),
) -> None:
    """Fetch the pinned roofline benchmark and optionally measure peaks."""
    from profilelens.profile.roofline_bench import (
        ROOFLINE_REF, bench_dir, ensure_roofline_bench, is_fetched,
    )

    if is_fetched() and not refresh:
        console.print(f"[bold green]Roofline benchmark already cached[/bold green] "
                      f"[dim]({bench_dir()})[/dim]")
    else:
        try:
            ensure_roofline_bench(console=console, force=refresh)
            console.print("[bold green]Roofline benchmark fetched.[/bold green]")
        except RuntimeError as exc:
            console.print(f"[bold red]Fetch failed:[/bold red] {exc}")
            raise typer.Exit(1)
    console.print(f"  [dim]pinned ref: {ROOFLINE_REF}[/dim]")

    if measure:
        from profilelens.profile.gpu_detect import detect_gpu
        from profilelens.profile.roofline_bench import run_roofline_benchmark
        cfg = load_config()
        gpu = detect_gpu(
            device_index=device,
            rocprofv3_path=cfg.profiling.rocprofv3_path,
        )
        console.print(f"\n[bold]Measuring roofline on GPU {device} ({gpu.model})...[/bold]")
        try:
            csv = run_roofline_benchmark(
                cfg.profiling.roofline_cache_dir, gpu.model, console=console,
                force=True, device_index=device,
            )
            console.print(f"[bold green]Peaks cached:[/bold green] {csv}")
        except RuntimeError as exc:
            console.print(f"[bold red]Measurement failed:[/bold red] {exc}")
            raise typer.Exit(1)


@app.command()
def version() -> None:
    """Print the ProfileLens version."""
    console.print(f"profilelens {__version__}")


if __name__ == "__main__":
    app()
