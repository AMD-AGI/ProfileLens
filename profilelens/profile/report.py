# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consolidated report generation in Rich, JSON, and Markdown formats."""

from __future__ import annotations

import json
import logging
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from profilelens.dtypes import KernelProfile, ProfilingResult, QueryResult

logger = logging.getLogger(__name__)


def _elide(name: str, width: int) -> str:

    if len(name) <= width:
        return name
    if width <= 1:
        return name[:width]
    keep = width - 1  # one char for the ellipsis
    head = (keep + 1) // 2
    tail = keep - head
    return name[:head] + "…" + (name[-tail:] if tail else "")


def _fmt_shape_sig(sig: str) -> str:
    """Render a per-launch shape signature."""
    if not sig:
        return ""
    try:
        args = sig.split("-")
        return "x".join(
            "[]" if arg == "scalar"
            else "[" + ",".join(arg.split("x")) + "]"
            for arg in args
        )
    except Exception:
        return sig


def _bottleneck_color(b: str) -> str:
    return {
        "MEMORY": "red", "COMPUTE": "blue", "LATENCY": "yellow",
        "BALANCED": "green", "UNKNOWN": "dim",
    }.get(b, "white")


def _fmt_pct(value: float, digits: int = 1) -> str:
    """Format a percentage; render the sentinel ``-1.0`` as ``N/A``."""
    if value < 0:
        return "N/A"
    return f"{value:.{digits}f}"


def _source_suffix(source: str) -> str:
    """Render provenance only when the value is an estimate."""
    return " (estimated)" if source.endswith("estimated") else ""


def _peak_label(k) -> str:
    """Name the concrete ceiling the kernel's %-of-peak is measured against.

    Memory/BW-binding -> ``HBM-BW``; compute-binding -> ``<dtype>-peak`` (e.g.
    ``FP8-peak``). This makes the table cell self-describing ("13% of HBM-BW")
    so no legend is needed.
    """
    if k.bottleneck == "MEMORY":
        return "HBM-BW"
    if k.bottleneck == "COMPUTE":
        return f"{k.detected_dtype}-peak"
    return "HBM-BW" if k.bw_util_pct >= k.mfma_util_pct else f"{k.detected_dtype}-peak"


def _format_peaks_bullet(result) -> str:
    """One-line empirical ceilings: HBM bandwidth + per-dtype MFMA peak & ridge.

    HBM bandwidth is dtype-independent; the MFMA ceiling and ridge point are
    per-dtype, so each observed dtype is listed once.
    """
    if not result.roofline_peaks:
        return "none — utilization metrics unavailable"
    hbm = result.roofline_peaks.peak_bw_tbps.get("hbm", 0.0)
    peaks = result.peaks_by_dtype or {result.detected_dtype: result.roofline_peaks}
    mfma = ", ".join(
        f"{dt} {p.peak_compute_tflops.get('mfma', 0):.0f} TFLOPS "
        f"(ridge {p.ridge_point:.0f} FLOP/byte)"
        for dt, p in peaks.items()
    )
    return f"HBM {hbm:.2f} TB/s · MFMA ceilings — {mfma}"


def _build_trace_context(
    analysis_results: list[QueryResult] | None,
) -> dict[str, dict[str, Any]]:
    """Build kernel_name -> trace context map from Phase 1 analysis."""
    if not analysis_results:
        return {}

    ctx: dict[str, dict[str, Any]] = {}

    agg = _find_result(analysis_results, "Aggregate Time per Kernel Name")
    if agg and agg.rows:
        total_ms = sum(r.get("total_ms", 0) for r in agg.rows)
        for r in agg.rows:
            name = r.get("name", "")
            if not name:
                continue
            pct = (r.get("total_ms", 0) / total_ms * 100) if total_ms > 0 else 0
            entry = {
                "e2e_pct": round(pct, 1),
                "trace_total_ms": round(r.get("total_ms", 0), 3),
                "trace_avg_ms": round(r.get("avg_ms", 0), 4),
                "trace_count": r.get("count", 0),
            }
            ctx[name] = entry
            # Phase 1 trace names carry the ROCm ".kd"/"[clone .kd]" decoration
            # that Phase 2 rocprofv3 dispatch names lack; register an undecorated
            # alias so the kernel_name join (E2E%, trace timing) resolves either way.
            from profilelens.profile.rocprofv3 import strip_kd_decoration
            stripped = strip_kd_decoration(name)
            if stripped != name:
                ctx.setdefault(stripped, entry)

    gpu_time = _find_result(analysis_results, "CPU Op GPU Kernel Time")
    if gpu_time and gpu_time.rows:
        for r in gpu_time.rows:
            op = r.get("cpu_op", "")
            if not op:
                continue
            for kname, kctx in ctx.items():
                if "op" not in kctx:
                    kctx["op"] = op

    return ctx


def print_profiling_rich(
    result: ProfilingResult,
    console: Console,
    analysis_results: list[QueryResult] | None = None,
) -> None:
    """Print the profiling report using Rich tables."""
    trace_ctx = _build_trace_context(analysis_results)
    has_trace = bool(trace_ctx)

    console.print()
    console.print("[bold]" + "=" * 60 + "[/bold]")
    console.print("[bold]PROFILELENS PROFILING REPORT[/bold]")
    console.print("[bold]" + "=" * 60 + "[/bold]")
    console.print(f"  GPU:            {result.gpu_model} ({result.gpu_arch}) [auto-detected]")
    dtypes = ", ".join(result.observed_dtypes) if result.observed_dtypes else result.detected_dtype
    console.print(f"  Compute dtypes: {dtypes} [dim](observed; dominant {result.detected_dtype})[/dim]")
    console.print(f"  Counter Set:    {result.counter_set_name}")
    console.print(f"  Empirical peaks: {_format_peaks_bullet(result)}")
    console.print()

    from profilelens.profile.bottleneck import compute_headroom

    if has_trace:
        title = (
            f"Top-{len(result.kernels)} Kernels by E2E Time — "
            f"Ranked by Max E2E Gain"
        )
    else:
        title = "Top Kernel Summary"
    tbl = Table(title=title, show_lines=True)
    tbl.add_column("#", justify="right", style="dim", width=3)
    tbl.add_column("Kernel", max_width=35)
    if has_trace:
        tbl.add_column("E2E Time%", justify="right", width=9)
    tbl.add_column("Type", width=8)
    tbl.add_column("Dtype", width=5)
    tbl.add_column("FLOP/byte (AI)", justify="right", width=14)
    tbl.add_column("% of HW peak", justify="right", width=16)
    if has_trace:
        tbl.add_column("Max E2E Gain%", justify="right", width=13)
    tbl.add_column("Stall", width=9)

    cv_map = result.kernel_cv or {}
    has_high_cv = False

    for i, k in enumerate(result.kernels, 1):
        bn_text = Text(k.bottleneck, style=_bottleneck_color(k.bottleneck))
        tc = trace_ctx.get(k.kernel_name, {})
        e2e = tc.get("e2e_pct", 0.0)
        hr = compute_headroom(k.mfma_util_pct, k.bw_util_pct, k.bottleneck, e2e)

        # Color by how much room remains (low utilization = red = opportunity).
        if hr.pct_of_peak < 0:
            roof_text = Text("N/A", style="dim")
        else:
            rs = f"{hr.pct_of_peak:.0f}% of {_peak_label(k)}"
            if hr.pct_of_peak <= 30:
                roof_text = Text(rs, style="bold red")
            elif hr.pct_of_peak <= 60:
                roof_text = Text(rs, style="bold yellow")
            else:
                roof_text = Text(rs, style="green")

        cv_info = cv_map.get(k.kernel_name, {})
        shape = _fmt_shape_sig(k.shape_sig) if k.shape_sig else ""
        display_name = k.kernel_name + (f" {shape}" if shape else "")
        name_display = _elide(display_name, 35)
        if cv_info.get("high_cv"):
            name_display = _elide(display_name, 33) + " *"
            has_high_cv = True

        row: list = [str(i), name_display]
        if has_trace:
            row.append(f"{e2e:.1f}" if e2e else "-")
        row.append(bn_text)
        row.append(k.detected_dtype)
        row.append(f"{k.ai_hbm:.1f}")
        row.append(roof_text)
        if has_trace:
            row.append(f"{hr.priority_score * 100:.1f}%")
        row.append(k.dominant_stall)
        tbl.add_row(*row)

    console.print(tbl)
    if has_high_cv:
        console.print(
            "  [yellow]* = high dispatch variance (CV > 20%). Counter data sampled from mid dispatches.[/yellow]\n"
            '  [yellow]  For accurate analysis: profilelens profile --kernel-names "name" --sample-dispatches 0[/yellow]'
        )

    for i, k in enumerate(result.kernels, 1):
        tc = trace_ctx.get(k.kernel_name, {})
        e2e = tc.get("e2e_pct", 0.0)
        shape = _fmt_shape_sig(k.shape_sig) if k.shape_sig else ""
        heading = k.kernel_name + (f" — shape {shape}" if shape else "")
        console.print()
        console.print(f"[bold]{'=' * 60}[/bold]")
        console.print(f"[bold]KERNEL {i}: {heading}[/bold]")
        console.print(f"[bold]{'=' * 60}[/bold]")

        if tc:
            console.print("  [bold]--- Trace Context (Phase 1) ---[/bold]")
            if tc.get("op"):
                console.print(f"  Op:              {tc['op']}")
            if shape:
                console.print(f"  Shape:           {shape}, {k.detected_dtype}")
            console.print(f"  E2E Time Share:  {e2e}% of total GPU time")
            console.print(f"  Trace Total:     {tc.get('trace_total_ms', '?')} ms")
            console.print(f"  Trace Avg:       {tc.get('trace_avg_ms', '?')} ms")
            console.print(f"  Dispatch Count:  {tc.get('trace_count', '?')}")
            console.print()
        elif shape:
            console.print(f"  Shape:           {shape}, {k.detected_dtype}")
            console.print()

        hr = compute_headroom(k.mfma_util_pct, k.bw_util_pct, k.bottleneck, e2e)

        dt_peaks = (result.peaks_by_dtype or {}).get(k.detected_dtype) or result.roofline_peaks
        mfma_peak = dt_peaks.peak_compute_tflops.get("mfma", 0.0) if dt_peaks else 0.0
        hbm_peak = result.roofline_peaks.peak_bw_tbps.get("hbm", 0.0) if result.roofline_peaks else 0.0

        console.print("  [bold]--- Empirical Roofline (Phase 2: HW Counters) ---[/bold]")
        console.print(f"  AI (HBM):        {k.ai_hbm:.2f} FLOP/byte")
        if k.bw_util_pct >= 0 and hbm_peak > 0:
            console.print(
                f"  Achieved BW:     {k.achieved_bw_tbps:.2f} TB/s "
                f"({k.bw_util_pct:.0f}% of {hbm_peak:.2f} TB/s peak)"
            )
        else:
            console.print(f"  Achieved BW:     {k.achieved_bw_tbps:.2f} TB/s")
        if k.mfma_util_pct >= 0 and mfma_peak > 0:
            mfma_tflops = k.mfma_util_pct / 100.0 * mfma_peak
            console.print(
                f"  Achieved FLOPs:  {mfma_tflops:.0f} TFLOPS "
                f"({k.mfma_util_pct:.0f}% of {mfma_peak:.0f} TFLOPS {k.detected_dtype} MFMA peak)"
            )
        else:
            console.print(f"  Achieved FLOPs:  {_fmt_pct(k.mfma_util_pct)}% of MFMA peak")
        if k.valu_busy_pct >= 0:
            console.print(f"  VALU (vector):   {k.valu_busy_pct:.0f}% busy")
        else:
            console.print("  VALU (vector):   [dim]N/A (SQ_ACTIVE_INST_VALU not collected)[/dim]")
        console.print(
            f"  Occupancy:       {k.occupancy_pct:.1f}%" if k.occupancy_pct >= 0
            else "  Occupancy:       N/A"
        )
        if k.l1_hit_rate_pct >= 0:
            console.print(
                f"  L1 Hit Rate:     {k.l1_hit_rate_pct:.1f}%"
                f"{_source_suffix(k.l1_hit_rate_source)}"
            )
        else:
            console.print("  L1 Hit Rate:     [dim]N/A (counter unavailable)[/dim]")
        if k.l2_hit_rate_pct >= 0:
            console.print(
                f"  L2 Hit Rate:     {k.l2_hit_rate_pct:.1f}%"
                f"{_source_suffix(k.l2_hit_rate_source)}"
            )
        else:
            console.print("  L2 Hit Rate:     [dim]N/A (counter unavailable)[/dim]")
        if k.stall_issue_pct >= 0:
            console.print(f"  Issue Wait:      {k.stall_issue_pct:.1f}%")
        if k.stall_vmem_pct >= 0:
            console.print(f"  VMEM Stall:      {k.stall_vmem_pct:.1f}%")
        if k.stall_lds_pct >= 0:
            console.print(f"  LDS Stall:       {k.stall_lds_pct:.1f}%")
        bn_color = _bottleneck_color(k.bottleneck)
        console.print(f"  Bottleneck:      [{bn_color}]{k.bottleneck}[/{bn_color}]")
        console.print(f"  Stall Summary:   {k.dominant_stall}")
        if e2e > 0:
            console.print(f"  Max E2E Gain at roofline perf: {hr.priority_score * 100:.1f}%")

        cv_info = cv_map.get(k.kernel_name, {})
        if cv_info:
            cv_str = f"  Dispatch CV:     {cv_info['cv_pct']:.1f}% (dur range: {cv_info['min_us']:.0f}-{cv_info['max_us']:.0f} us)"
            if cv_info.get("high_cv"):
                console.print(f"  [yellow]{cv_str.strip()} *[/yellow]")
            else:
                console.print(cv_str)

        if k.gap_hints:
            console.print()
            console.print("  [bold]--- Gap Analysis ---[/bold]")
            for hint in k.gap_hints:
                console.print(f"    - {hint}")


def format_profiling_json(
    result: ProfilingResult,
    analysis_results: list[QueryResult] | None = None,
) -> str:
    """Format profiling result as JSON, merging trace context from Phase 1."""
    trace_ctx = _build_trace_context(analysis_results)

    out: dict[str, Any] = {
        "profilelens_version": _get_version(),
        "gpu": {"arch": result.gpu_arch, "model": result.gpu_model},
        "detected_dtype": result.detected_dtype,
        "observed_dtypes": list(result.observed_dtypes),
        "counter_set": result.counter_set_name,
        "trace_context_source": getattr(result, "trace_context_source", "perfetto_sql"),
        "audit": {
            "phase1_workload_s": result.phase1_workload_s,
            "phase2_workload_s": result.phase2_workload_s,
            "phase2_overhead_s": (
                result.phase2_workload_s - result.phase1_workload_s
                if result.phase1_workload_s and result.phase2_workload_s else None
            ),
            "sample_iteration_ids": list(result.sample_iteration_ids),
            "sample_dispatches_per_kernel": result.sample_dispatches_per_kernel,
            "counters_used": list(result.counters_used),
            "kernels_profiled": [k.kernel_name for k in result.kernels],
        },
    }

    if result.roofline_peaks:
        out["roofline_peaks"] = {
            "source": "empirical",
            "peak_bw_tbps": result.roofline_peaks.peak_bw_tbps,
            "peak_compute_tflops": result.roofline_peaks.peak_compute_tflops,
            "ridge_point": result.roofline_peaks.ridge_point,
        }
    if result.peaks_by_dtype:
        out["roofline_peaks_by_dtype"] = {
            dt: {
                "peak_mfma_tflops": p.peak_compute_tflops.get("mfma", 0.0),
                "ridge_point": p.ridge_point,
            }
            for dt, p in result.peaks_by_dtype.items()
        }

    kernels_json = []
    for k in result.kernels:
        entry: dict[str, Any] = {"kernel_name": k.kernel_name}
        if k.shape_sig:
            entry["shape_sig"] = k.shape_sig
            entry["shape"] = _fmt_shape_sig(k.shape_sig)
        if k.variant_label:
            entry["variant"] = k.variant_label
        entry["dtype"] = k.detected_dtype
        entry["dtype_source"] = k.dtype_source
        tc = trace_ctx.get(k.kernel_name)
        if tc:
            entry["trace_context"] = tc
        entry["hw_counters"] = {
            "mfma_flops": k.mfma_flops,
            "valu_flops": k.valu_flops,
            "hbm_bytes": k.hbm_bytes,
            "l2_bytes": k.l2_bytes,
            "l1_bytes": k.l1_bytes,
            "l2_bytes_source": k.l2_bytes_source,
            "l1_bytes_source": k.l1_bytes_source,
            "raw": k.raw_counters,
        }
        entry["derived_metrics"] = {
            "ai_hbm": k.ai_hbm,
            "ai_l2": k.ai_l2,
            "ai_l1": k.ai_l1,
            "ai_l2_source": k.l2_bytes_source,
            "ai_l1_source": k.l1_bytes_source,
            "achieved_bw_tbps": k.achieved_bw_tbps,
            "achieved_tflops": k.achieved_tflops,
            "mfma_util_pct": k.mfma_util_pct,
            "valu_busy_pct": k.valu_busy_pct,
            "bw_util_pct": k.bw_util_pct,
            "occupancy_pct": k.occupancy_pct,
            "l1_hit_rate_pct": k.l1_hit_rate_pct,
            "l2_hit_rate_pct": k.l2_hit_rate_pct,
            "l1_hit_rate_source": k.l1_hit_rate_source,
            "l2_hit_rate_source": k.l2_hit_rate_source,
        }
        entry["stalls"] = {
            "issue_pct": k.stall_issue_pct,
            "vmem_pct": k.stall_vmem_pct,
            "lds_pct": k.stall_lds_pct,
            "source": k.stall_source,
            "dominant": k.dominant_stall,
        }
        tc_entry = trace_ctx.get(k.kernel_name, {})
        e2e_val = tc_entry.get("e2e_pct", 0.0)
        from profilelens.profile.bottleneck import compute_headroom as _chr
        hr_entry = _chr(k.mfma_util_pct, k.bw_util_pct, k.bottleneck, e2e_val)
        entry["classification"] = {
            "bottleneck": k.bottleneck,
            "pct_of_peak": hr_entry.pct_of_peak,
            "headroom": hr_entry.headroom,
            "priority_score": hr_entry.priority_score,
        }
        entry["gap_analysis"] = {
            "hints": list(k.gap_hints),
            "actionable": {
                "primary_limiter": _infer_limiter(k),
                "suggested_experiments": _suggest(k),
            },
        }
        kernels_json.append(entry)

    out["kernels"] = kernels_json
    return json.dumps(out, indent=2, default=str)


def format_profiling_markdown(
    result: ProfilingResult,
    analysis_results: list[QueryResult] | None = None,
    output_dir: str | None = None,
    flydsl_args: dict[str, Any] | None = None,
    capture_graphs: dict[str, Any] | None = None,
) -> str:
    """Format profiling result as Markdown with Phase 1 trace context."""
    from profilelens.profile.dtype_detect import (
        _flydsl_args_lookup,
        _flydsl_stem,
        flydsl_stem_collisions,
        format_flydsl_arg_shapes,
        graph_gid_for_sig,
        graph_op_for_kernel,
    )
    from profilelens.profile.kernel_context import (
        KernelContext,
        build_kernel_context_map,
    )

    trace_ctx = _build_trace_context(analysis_results)
    has_trace = bool(trace_ctx)
    rich_ctx: dict[str, KernelContext] = {}
    if analysis_results:
        rich_ctx = build_kernel_context_map(analysis_results)

    # TODO: handle beyond 1:1 cases
    if flydsl_args:
        for _w in flydsl_stem_collisions(
            [k.kernel_name for k in result.kernels], flydsl_args
        ):
            logger.warning(_w)

    lines: list[str] = []
    lines.append("# ProfileLens Profiling Report\n")
    lines.append(f"- **GPU:** {result.gpu_model} ({result.gpu_arch})")
    if result.observed_dtypes and len(result.observed_dtypes) > 1:
        lines.append(
            f"- **Compute dtypes (observed):** {', '.join(result.observed_dtypes)} "
            f"&nbsp;·&nbsp; dominant **{result.detected_dtype}**"
        )
    else:
        lines.append(f"- **Compute dtype (observed):** {result.detected_dtype}")
    lines.append(f"- **Counter Set:** {result.counter_set_name}")
    lines.append(f"- **Empirical peaks (rocprof-compute):** {_format_peaks_bullet(result)}")
    lines.append("")

    # Roofline plot — render and embed if peaks + kernels are available.
    if output_dir and result.roofline_peaks and result.kernels:
        from pathlib import Path
        from profilelens.profile.roofline_plot import render_roofline_png
        png = render_roofline_png(
            result.roofline_peaks,
            result.kernels,
            Path(output_dir) / "roofline.png",
            title="Empirical Roofline (top kernels)",
            peaks_by_dtype=result.peaks_by_dtype,
        )
        if png:
            lines.append(f"![Empirical Roofline]({png.name})")
            lines.append("")

    # Profiling-pipeline audit: how long each phase took, what was sampled.
    if result.counters_used or result.kernels:
        lines.append("## Profiling pipeline audit\n")
        if result.counters_used:
            lines.append(
                f"- **PMC counters captured ({len(result.counters_used)}):** "
                + ", ".join(f"`{c}`" for c in result.counters_used)
            )
        if result.kernels:
            n_kern = len(result.kernels)
            lines.append(
                f"- **Kernels profiled:** {n_kern} unique names — "
                + ", ".join(f"`{_elide(k.kernel_name, 35)}`" for k in result.kernels[:n_kern])
            )
        lines.append("")


    from profilelens.profile.bottleneck import compute_headroom

    if has_trace:
        _e2e_from = (
            " (E2E% from TraceLens)"
            if getattr(result, "trace_context_source", "perfetto_sql") == "tracelens"
            else ""
        )
        heading = (
            f"## Top-{len(result.kernels)} Kernels by E2E Time — "
            f"Ranked by Max E2E Gain{_e2e_from}\n"
        )
    else:
        heading = "## Top Kernel Summary\n"
    lines.append(heading)
    if has_trace:
        lines.append("| # | Kernel | E2E Time% | Type | Dtype | FLOP/byte (AI) | % of HW peak | Max E2E Gain% | Stall |")
        lines.append("|---|--------|-----------|------|-------|----------------|--------------|---------------|-------|")
    else:
        lines.append("| # | Kernel | Type | Dtype | FLOP/byte (AI) | % of HW peak | Stall |")
        lines.append("|---|--------|------|-------|----------------|--------------|-------|")

    cv_map = result.kernel_cv or {}
    md_has_high_cv = False

    for i, k in enumerate(result.kernels, 1):
        tc = trace_ctx.get(k.kernel_name, {})
        base = _elide(k.kernel_name, 30 if k.shape_sig else 44)
        name = base
        cv_info = cv_map.get(k.kernel_name, {})
        if cv_info.get("high_cv"):
            name = base[:-2] + " *" if len(base) > 2 else base + " *"
            md_has_high_cv = True
        if k.shape_sig:
            name = f"{name} `{_fmt_shape_sig(k.shape_sig)}`"
        if k.variant_label:
            name = f"{name} [{k.variant_label}]"
        e2e = tc.get("e2e_pct", 0.0)
        hr = compute_headroom(k.mfma_util_pct, k.bw_util_pct, k.bottleneck, e2e)
        roof_str = (
            f"{hr.pct_of_peak:.0f}% of {_peak_label(k)}"
            if hr.pct_of_peak >= 0 else "N/A"
        )
        if has_trace:
            lines.append(
                f"| {i} | {name} | {e2e if e2e else '-'} | {k.bottleneck} | {k.detected_dtype} | "
                f"{k.ai_hbm:.1f} | {roof_str} | {hr.priority_score * 100:.1f}% | "
                f"{k.dominant_stall} |"
            )
        else:
            lines.append(
                f"| {i} | {name} | {k.bottleneck} | {k.detected_dtype} | "
                f"{k.ai_hbm:.1f} | {roof_str} | {k.dominant_stall} |"
            )
    if md_has_high_cv:
        lines.append(
            '\n*\\* = high dispatch variance (CV > 20%). Counter data sampled from mid dispatches. '
            'For accurate analysis: `profilelens profile --kernel-names "name" --sample-dispatches 0`*'
        )
    lines.append("")

    for i, k in enumerate(result.kernels, 1):
        tc = trace_ctx.get(k.kernel_name, {})
        rich = rich_ctx.get(k.kernel_name)
        disp_shape = _fmt_shape_sig(k.shape_sig) if k.shape_sig else ""
        heading = k.kernel_name + (f" — shape {disp_shape}" if disp_shape else "")
        if k.variant_label:
            heading = f"{heading} ({k.variant_label})"
        lines.append(f"## Kernel {i}: {heading}\n")

        # Phase-1 timing provenance (Perfetto SQL vs external TraceLens); op/shape/
        # dtype always come from ProfileLens's own trace analysis + shim.
        _from_tracelens = getattr(result, "trace_context_source", "perfetto_sql") == "tracelens"
        _phase1_header = (
            "**Trace (Phase 1 — timing from TraceLens):**"
            if _from_tracelens else "**Perfetto (Phase 1):**"
        )
        _e2e_src = " (TraceLens)" if _from_tracelens else ""

        if rich and (rich.module != "unknown" or rich.op != "unknown"
                     or rich.shape or rich.dispatch_count or disp_shape):
            lines.append(_phase1_header)
            _gid = (
                graph_gid_for_sig(k.shape_sig, capture_graphs, k.kernel_name)
                if capture_graphs and k.shape_sig else ""
            )
            _gop = graph_op_for_kernel(
                k.kernel_name, capture_graphs, prefer_gid=_gid,
                prefer_sig=k.shape_sig,
            ) if capture_graphs else ""
            if rich.module != "unknown":
                lines.append(f"- Module: `{rich.module}`")
            if rich.op != "unknown":
                lines.append(f"- Op: `{rich.op}`")
            elif k.dtype_source == "flydsl_hook":
                _stem = _flydsl_stem(k.kernel_name)
                if _stem:
                    lines.append(f"- Op: `flydsl::{_stem}`")
            elif _gop:
                lines.append(f"- Op: `{_gop}`")
            if disp_shape:
                lines.append(f"- Shape: {disp_shape}, {k.detected_dtype}")
            elif rich.shape:
                lines.append(f"- Shape: {rich.shape}, {k.detected_dtype}")
            elif k.dtype_source == "flydsl_hook" and flydsl_args:
                _fa = format_flydsl_arg_shapes(
                    _flydsl_args_lookup(k.kernel_name, flydsl_args)
                )
                if _fa:
                    lines.append(f"- Shape: {_fa}, {k.detected_dtype}")
            lines.append(f"- Dtype: **{k.detected_dtype}**")
            if rich.category != "unknown":
                lines.append(f"- Category: {rich.category}")
            _e2e_label = ("E2E Time Share (kernel name, all variants)"
                          if k.variant_label else "E2E Time Share")
            lines.append(f"- {_e2e_label}: {rich.e2e_pct}% of total GPU time{_e2e_src}")
            if disp_shape:
                lines.append(
                    f"- Time: {k.avg_duration_us:.1f} us avg "
                    f"({k.total_duration_us / 1000.0:.2f} ms total over "
                    f"{k.dispatch_count} dispatches, this shape)"
                )
            else:
                lines.append(f"- Time: {rich.avg_us:.1f} us avg "
                             f"({rich.total_ms:.2f} ms total over {rich.dispatch_count} dispatches)")
            lines.append("")
        elif tc:
            lines.append(
                "**Trace Context (Phase 1 — timing from TraceLens):**"
                if _from_tracelens
                else "**Trace Context (Phase 1: Perfetto Analysis):**"
            )
            if tc.get("op"):
                lines.append(f"- Op: {tc['op']}")
            _e2e_label = ("E2E Time Share (kernel name, all variants)"
                          if k.variant_label else "E2E Time Share")
            lines.append(f"- {_e2e_label}: {tc.get('e2e_pct', '?')}% of total GPU time{_e2e_src}")
            lines.append(f"- Trace Total: {tc.get('trace_total_ms', '?')} ms")
            lines.append(f"- Trace Avg: {tc.get('trace_avg_ms', '?')} ms")
            lines.append(f"- Dispatch Count: {tc.get('trace_count', '?')}")
            lines.append("")

        e2e = tc.get("e2e_pct", 0.0)
        hr = compute_headroom(k.mfma_util_pct, k.bw_util_pct, k.bottleneck, e2e)

        dt_peaks = (result.peaks_by_dtype or {}).get(k.detected_dtype) or result.roofline_peaks
        mfma_peak = dt_peaks.peak_compute_tflops.get("mfma", 0.0) if dt_peaks else 0.0
        hbm_peak = result.roofline_peaks.peak_bw_tbps.get("hbm", 0.0) if result.roofline_peaks else 0.0

        lines.append("**Empirical Roofline (Phase 2: HW Counters):**")
        lines.append(f"- Empirical AI: {k.ai_hbm:.2f} FLOP/byte (HBM)")
        if k.bw_util_pct >= 0 and hbm_peak > 0:
            lines.append(
                f"- Achieved BW: {k.achieved_bw_tbps:.2f} TB/s "
                f"({k.bw_util_pct:.0f}% of {hbm_peak:.2f} TB/s peak)"
            )
        else:
            lines.append(f"- Achieved BW: {k.achieved_bw_tbps:.2f} TB/s")
        # Compute is split across two engines: the matrix engine (MFMA FLOPs, %
        # of the dtype's MFMA peak — parallels Achieved BW) and the vector ALU
        # (busy %, cycles-based). A kernel can be 0 MFMA FLOPs yet busy on VALU —
        # that's where its compute went.
        if k.mfma_util_pct >= 0 and mfma_peak > 0:
            mfma_tflops = k.mfma_util_pct / 100.0 * mfma_peak
            lines.append(
                f"- Achieved FLOPs: {mfma_tflops:.0f} TFLOPS "
                f"({k.mfma_util_pct:.0f}% of {mfma_peak:.0f} TFLOPS {k.detected_dtype} MFMA peak)"
            )
        else:
            lines.append(f"- Achieved FLOPs: {_fmt_pct(k.mfma_util_pct, digits=0)}% of MFMA peak")
        if k.valu_busy_pct >= 0:
            lines.append(f"- VALU (vector engine): {k.valu_busy_pct:.0f}% busy")
        else:
            lines.append("- VALU (vector engine): N/A (SQ_ACTIVE_INST_VALU not collected)")
        lines.append(f"- Bottleneck: **{k.bottleneck}**")
        lines.append(f"- Stall Summary: {k.dominant_stall}")
        if k.stall_issue_pct >= 0:
            lines.append(f"- Issue Wait: {k.stall_issue_pct:.0f}%")
        if k.stall_vmem_pct >= 0:
            lines.append(f"- VMEM Stall: {k.stall_vmem_pct:.0f}%")
        if k.stall_lds_pct >= 0:
            lines.append(f"- LDS Stall: {k.stall_lds_pct:.0f}%")
        lines.append(
            f"- L1 Hit Rate: {k.l1_hit_rate_pct:.0f}%"
            f"{_source_suffix(k.l1_hit_rate_source)}"
            if k.l1_hit_rate_pct >= 0
            else "- L1 Hit Rate: N/A (counter unavailable)"
        )
        lines.append(
            f"- L2 Hit Rate: {k.l2_hit_rate_pct:.0f}%"
            f"{_source_suffix(k.l2_hit_rate_source)}"
            if k.l2_hit_rate_pct >= 0
            else "- L2 Hit Rate: N/A (counter unavailable)"
        )
        lines.append(
            f"- Occupancy: {k.occupancy_pct:.0f}%" if k.occupancy_pct >= 0
            else "- Occupancy: N/A"
        )
        if e2e > 0:
            lines.append(
                f"- Max E2E Gain at roofline perf: {hr.priority_score * 100:.1f}%"
            )
        lines.append("")
        lines.append("**Category:** " + _suggest_category(k))
        if k.gap_hints:
            lines.append("\n**Hints:**")
            for hint in k.gap_hints:
                lines.append(f"  - {hint}")
        lines.append("")

    return "\n".join(lines)


def load_analysis_results(json_path: str) -> list[QueryResult]:
    """Load analysis QueryResults from a saved JSON file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    results = []
    for entry in data:
        label = entry.get("label", "")
        sql = entry.get("sql", "")
        rows = entry.get("rows")
        columns = entry.get("columns")
        error = entry.get("error")
        results.append(QueryResult(
            sql=sql, label=label,
            rows=tuple(rows) if rows else (),
            columns=tuple(columns) if columns else (),
            error=error,
        ))
    return results


def _find_result(results: list[QueryResult], label: str) -> QueryResult | None:
    for r in results:
        if r.label == label:
            return r
    return None


def _infer_limiter(k: KernelProfile) -> str:
    if k.occupancy_pct >= 0 and k.occupancy_pct < 50:
        return "occupancy"
    if k.bw_util_pct >= 0 and k.mfma_util_pct >= 0 and k.bw_util_pct > k.mfma_util_pct:
        return "memory_bandwidth"
    if k.stall_vmem_pct > 30:
        return "vmem_latency"
    if k.stall_issue_pct > 30:
        return "instruction_issue_wait"
    if k.stall_lds_pct > 30:
        return "lds_contention"
    return "unknown"


def _suggest(k: KernelProfile) -> list[str]:
    from profilelens.profile.bottleneck import suggest_experiments
    return suggest_experiments(k.bottleneck, k.occupancy_pct, k.stall_vmem_pct)


def _suggest_category(k: KernelProfile) -> str:
    """High-level category hint derived from the bottleneck classification."""
    if k.bottleneck == "MEMORY":
        return "Memory bandwidth bound — tile/fuse to raise arithmetic intensity"
    if k.bottleneck == "COMPUTE":
        return "Compute bound — algorithmic optimization or lower-precision math"
    if k.bottleneck == "LATENCY":
        return "Latency bound — kernel tuning / synchronization reduction"
    if k.bottleneck == "BALANCED":
        return "Balanced — small wins possible; check stalls and occupancy"
    return "Unknown — re-run with PMC counters enabled"


def _get_version() -> str:
    try:
        from profilelens import __version__
        return __version__
    except Exception:
        return "unknown"
