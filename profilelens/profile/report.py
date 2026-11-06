# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consolidated report generation in Rich, JSON, and Markdown formats."""

from __future__ import annotations

import html
import json
import logging
import math
from collections import Counter
from typing import Any

from rich.console import Console
from rich.table import Table

from profilelens.dtypes import KernelProfile, ProfilingResult, QueryResult
from profilelens.profile.dtype_detect import UNKNOWN_DTYPE
from profilelens.profile.kernel_context import kernel_time_share
from profilelens.profile.kernel_names import (
    add_canonical_aliases,
    lookup_kernel,
)

logger = logging.getLogger(__name__)


def _elide(name: str, width: int) -> str:

    if len(name) <= width:
        return name
    if width <= 1:
        return name[:width]
    keep = width - 1
    head = (keep + 1) // 2
    tail = keep - head
    return name[:head] + "…" + (name[-tail:] if tail else "")


def _markdown_code(value: str) -> str:
    """Keep kernel symbols literal in Markdown tables and rendered HTML."""
    escaped = html.escape(str(value)).translate(str.maketrans(
        {char: f"&#{ord(char)};" for char in "\\`*_[]|"}
    ))
    return "<code>" + escaped + "</code>"


def _selection_label(result: ProfilingResult) -> tuple[str, str]:
    if result.counter_set_name == "custom":
        return "User-supplied counters", str(len(result.counters_used))
    if result.counter_dtype_hints:
        return "User-supplied data types", ", ".join(result.counter_dtype_hints)
    known = [dt for dt in result.observed_dtypes if dt != UNKNOWN_DTYPE]
    return "Data types", ", ".join(known) or "Unknown"


def _display_dtype(result: ProfilingResult, kernel: KernelProfile) -> str:
    if kernel.detected_dtype != UNKNOWN_DTYPE:
        return kernel.detected_dtype
    label, _value = _selection_label(result)
    return label if label.startswith("User-supplied") else "Unknown"


def _display_category(kernel: KernelProfile) -> str:
    return "Unclassified" if kernel.bottleneck == "UNKNOWN" else kernel.bottleneck.title()


def _duration(value_us: float | None) -> str:
    if value_us is None or value_us < 0 or not math.isfinite(value_us):
        return "N/A"
    return f"{value_us / 1000:.3f} ms" if value_us >= 1000 else f"{value_us:.1f} µs"


def _bytes(value: float | None) -> str:
    if value is None or value < 0 or not math.isfinite(value):
        return "N/A"
    for divisor, unit in ((1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if value >= divisor:
            return f"{value / divisor:.3f} {unit}"
    return f"{value:.0f} bytes"


def _bandwidth(value: float | None) -> str:
    if value is None or value < 0 or not math.isfinite(value):
        return "N/A"
    return f"{value:.2f} TB/s" if value >= 0.01 else f"{value * 1000:.2f} GB/s"


def _matrix_work(value: float | None) -> str:
    if value is None:
        return "N/A"
    for divisor, unit in ((1e9, "GFLOP"), (1e6, "MFLOP"), (1e3, "KFLOP")):
        if value >= divisor:
            return f"{value / divisor:.3f} {unit}"
    return f"{value:.0f} FLOP"


def _flop_rate(value: float | None) -> str:
    if value is None:
        return "N/A TFLOPS"
    if value == 0 or value >= 0.01:
        return f"{value:.2f} TFLOPS"
    if value >= 0.00001:
        return f"{value * 1000:.2f} GFLOPS"
    return f"{value * 1e6:.2f} MFLOPS"


def _percent(value: float | None, digits: int = 1) -> str:
    if value is None or value < 0 or not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}%"


def _trace_fields(result, kernel, context, rich=None, shape="", op="", shared=False):
    fields = []
    if rich is not None and rich.module != "unknown":
        fields.append(("Module", rich.module))
    if not op and rich is not None and rich.op != "unknown":
        op = rich.op
    op = op or context.get("op", "")
    if op:
        fields.append(("Op", op))
    shape = shape or (rich.shape if rich is not None else "")
    if shape:
        fields.append(("Shape", shape))
    if kernel.detected_dtype == UNKNOWN_DTYPE:
        fields.append(_selection_label(result))
    else:
        fields.append(("Data types", kernel.detected_dtype))
    if context or rich is not None:
        share = context.get("e2e_pct", rich.e2e_pct if rich is not None else -1)
        label = "GPU time share (kernel name)" if shared else "GPU time share"
        fields.append((label, _percent(share)))
        average_ms = context.get("trace_avg_ms")
        total_ms = context.get("trace_total_ms")
        count = context.get("trace_count")
        if average_ms is None and rich is not None:
            average_ms, total_ms, count = rich.avg_us / 1000, rich.total_ms, rich.dispatch_count
        if average_ms is not None:
            fields.append(("Trace timing", f"{_duration(average_ms * 1000)} avg · "
                           f"{_fmt_metric(total_ms, 3)} ms total · {count or 0} dispatches"))
    return fields


def _counter_fields(result, kernel, headroom, shared=False):
    k = kernel
    fields = [
        ("Profiled dispatch timing", f"{_duration(k.avg_duration_us)} avg · "
         f"{k.total_duration_us / 1000:.3f} ms total · {k.dispatch_count} dispatches"),
        ("Median dispatch duration", _duration(getattr(k, "median_duration_us", None))),
        ("Measured matrix work", _matrix_work(k.mfma_flops) + "/dispatch"),
    ]
    families = getattr(k, "mfma_flops_by_family", {})
    fractions = getattr(k, "mfma_fraction_by_family", {})
    if families:
        fields.append(("Matrix work by family", " · ".join(
            f"{family} {_matrix_work(operations)}" for family, operations in families.items()
        )))
    if fractions:
        fields.append(("Measured matrix mix", " · ".join(
            f"{family} {fraction * 100:.4f}%" for family, fraction in fractions.items()
        )))
    fields.append(("DRAM request traffic", _bytes(k.hbm_bytes) + "/dispatch"))
    traffic = getattr(k, "hbm_bytes_by_kind", {})
    if traffic:
        fields.append(("DRAM request breakdown", " · ".join(
            f"{kind} {_bytes(value)}" for kind, value in traffic.items()
        )))
    hbm_peak = result.roofline_peaks.peak_bw_tbps.get("hbm", 0.0) if result.roofline_peaks else 0
    bandwidth = _bandwidth(k.achieved_bw_tbps)
    if k.bw_util_pct >= 0 and hbm_peak > 0:
        bandwidth += f" ({k.bw_util_pct:.1f}% of {hbm_peak:.2f} TB/s)"
    fields.extend([
        ("Arithmetic intensity", f"{_fmt_metric(k.ai_hbm)} FLOP/byte (HBM)"),
        ("Measured MFMA throughput", _flop_rate(k.achieved_tflops)),
        ("DRAM request rate", bandwidth),
        ("Memory roof utilization", _percent(k.bw_util_pct)),
        ("Compute roof utilization", _percent(k.mfma_util_pct)),
        ("Wave-slot occupancy", _percent(k.occupancy_pct)),
        ("VALU busy", _percent(k.valu_busy_pct)),
        ("Active VALU lanes", _percent(k.valu_lane_utilization_pct)),
        ("Issue wait", _percent(k.stall_issue_pct)),
        ("LDS wait", _percent(k.stall_lds_pct)),
        ("L1 hit rate", _percent(k.l1_hit_rate_pct) + _source_suffix(k.l1_hit_rate_source)),
        ("L2 hit rate", _percent(k.l2_hit_rate_pct) + _source_suffix(k.l2_hit_rate_source)),
        ("Optimization headroom", "N/A (shared timing)" if shared else
         _percent(headroom.headroom * 100 if headroom.headroom >= 0 else None)),
    ])
    if k.stall_vmem_pct >= 0:
        fields.append(("VMEM wait", _percent(k.stall_vmem_pct)))
    return fields


def _report_hints(kernel: KernelProfile, op: str = "") -> list[str]:
    """Short counter-based investigation directions without new classifications."""
    k = kernel
    lowered = k.kernel_name.lower()
    if op == "aten::copy_" or "copybuffer" in lowered or "memcpy" in lowered:
        return [f"Review the {k.dispatch_count} small-copy dispatches.",
                "Check whether adjacent copies can be combined."]
    hints = []
    if k.stall_issue_pct >= 30:
        hints.append(f"Inspect instruction-issue waits ({k.stall_issue_pct:.1f}%).")
    if k.stall_lds_pct >= 20:
        hints.append(f"Review LDS issue waiting ({k.stall_lds_pct:.1f}%).")
    if 0 <= k.l1_hit_rate_pct < 60:
        hints.append(f"Review L1 locality ({k.l1_hit_rate_pct:.1f}% hit rate).")
    if 0 <= k.l2_hit_rate_pct < 40:
        hints.append(f"Review L2 locality ({k.l2_hit_rate_pct:.1f}% hit rate).")
    if k.bottleneck == "COMPUTE":
        hints.append("Review matrix tiling and arithmetic precision.")
    elif k.hbm_bytes is not None and k.hbm_bytes > 0 and 0 <= k.bw_util_pct < 75:
        hints.append("Review memory access and prefetch scheduling.")
    return hints[:3] or ["No automatic hint."]


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


def _fmt_metric(value: float | None, digits: int = 2) -> str:
    """Missing measurements remain N/A; measured zero remains zero."""
    return f"{value:.{digits}f}" if value is not None else "N/A"


def _source_suffix(source: str) -> str:
    """Render provenance only when the value is an estimate."""
    return " (estimated)" if source.endswith("estimated") else ""


def _peak_label(k) -> str:
    """Name the ceiling a kernel's %-of-peak is measured against."""
    if k.bottleneck == "MEMORY":
        return "HBM-BW"
    if k.bottleneck == "COMPUTE":
        return f"{k.detected_dtype}-peak"
    return "HBM-BW" if k.bw_util_pct >= k.mfma_util_pct else f"{k.detected_dtype}-peak"


def _format_peaks_bullet(result) -> str:
    """One-line summary of HBM bandwidth and per-dtype MFMA ceilings."""
    if not result.roofline_peaks:
        return "none — utilization metrics unavailable"
    hbm = result.roofline_peaks.peak_bw_tbps.get("hbm", 0.0)
    peaks = result.peaks_by_dtype or {result.detected_dtype: result.roofline_peaks}
    mfma = ", ".join(
        f"{dt} {p.peak_compute_tflops.get('mfma', 0):.0f} TFLOPS "
        f"(ridge {p.ridge_point:.0f} FLOP/byte)"
        for dt, p in peaks.items()
        if dt != UNKNOWN_DTYPE and p.peak_compute_tflops.get("mfma", 0) > 0
    )
    return f"HBM {hbm:.2f} TB/s · MFMA reference ceilings — {mfma or 'unavailable'}"


def _kernel_compute_peak(result: ProfilingResult, kernel: KernelProfile):
    if kernel.detected_dtype == UNKNOWN_DTYPE:
        return None
    peak = result.peaks_by_dtype.get(kernel.detected_dtype)
    if peak is None and result.roofline_peaks is not None:
        if result.roofline_peaks.detected_dtype == kernel.detected_dtype:
            peak = result.roofline_peaks
    return peak


def _gain_text(headroom) -> str:
    return f"{headroom.priority_score * 100:.1f}%" if headroom.priority_score >= 0 else "N/A"


def _counter_selection_note(result: ProfilingResult) -> str:
    hints = ", ".join(result.counter_dtype_hints) or "none"
    coverage = ", ".join(result.counter_dtypes) or "explicit counter list"
    unresolved = ", ".join(result.unresolved_kernels) or "none"
    return (f"Caller counter-dtype hints: {hints}. Counter coverage: {coverage}. "
            f"Unresolved kernel dtypes: {unresolved}. Counter selection is not dtype "
            "inference; selected MFMA families may cover only part of the arithmetic. "
            "UNKNOWN kernels are not assigned a datatype-specific compute roof or headroom.")


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
            pct = kernel_time_share(r, total_ms)
            ctx[name] = {
                "e2e_pct": round(pct, 1),
                "trace_total_ms": round(r.get("total_ms", 0), 3),
                "trace_avg_ms": round(r.get("avg_ms", r.get("total_ms", 0) / r["count"] if r.get("count") else 0), 4),
                "trace_count": r.get("count", 0),
            }

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
    """Print the same compact report fields used in the Markdown output."""
    from profilelens.profile.bottleneck import compute_headroom
    from profilelens.profile.kernel_context import build_kernel_context_map

    trace_ctx = add_canonical_aliases(_build_trace_context(analysis_results))
    rich_ctx = add_canonical_aliases(build_kernel_context_map(analysis_results)) if analysis_results else {}
    names = Counter(k.kernel_name for k in result.kernels)
    label, value = _selection_label(result)
    console.print("ProfileLens Kernel Report", style="bold")
    console.print(f"GPU: {result.gpu_model} ({result.gpu_arch})", markup=False)
    console.print(f"{label}: {value}", markup=False)
    console.print(f"Measured: {len(names)} kernel names · {len(result.kernels)} profile rows")
    console.print(f"Empirical peaks: {_format_peaks_bullet(result)}", markup=False)
    table = Table(title=f"Top-{len(names)} Kernels", show_lines=True)
    for heading in ("#", "Kernel", "GPU time %", "Calls", "Category", "BW util %", "Headroom %"):
        table.add_column(heading, max_width=40 if heading == "Kernel" else None)
    for index, kernel in enumerate(result.kernels, 1):
        context = lookup_kernel(trace_ctx, kernel.kernel_name, {})
        share = context.get("e2e_pct", 0)
        headroom = compute_headroom(kernel.mfma_util_pct, kernel.bw_util_pct,
                                    kernel.bottleneck, share)
        gain = "N/A (shared timing)" if names[kernel.kernel_name] > 1 else _gain_text(headroom)
        table.add_row(str(index), kernel.kernel_name, _fmt_pct(share),
                      str(context.get("trace_count", kernel.dispatch_count)),
                      _display_category(kernel), _fmt_pct(kernel.bw_util_pct), gain)
    console.print(table)
    console.print("Kernels profiled and counters used", style="bold")
    console.print(f"Counters ({len(result.counters_used)}): " + ", ".join(result.counters_used), markup=False)
    for index, kernel in enumerate(result.kernels, 1):
        context = lookup_kernel(trace_ctx, kernel.kernel_name, {})
        rich = lookup_kernel(rich_ctx, kernel.kernel_name)
        share = context.get("e2e_pct", 0)
        headroom = compute_headroom(kernel.mfma_util_pct, kernel.bw_util_pct,
                                    kernel.bottleneck, share)
        shared = names[kernel.kernel_name] > 1
        console.print()
        console.print(f"Kernel {index}: {kernel.kernel_name}", style="bold", markup=False)
        console.print("Trace analysis:", style="bold")
        for field, value in _trace_fields(result, kernel, context, rich,
                                          _fmt_shape_sig(kernel.shape_sig) if kernel.shape_sig else "",
                                          shared=shared):
            console.print(f"  - {field}: {value}", markup=False)
        console.print("Perf counter analysis:", style="bold")
        for field, value in _counter_fields(result, kernel, headroom, shared):
            console.print(f"  - {field}: {value}", markup=False)
        console.print(f"Category: {_display_category(kernel)}", markup=False)
        console.print("Kernel Optimization Hints:", style="bold")
        for hint in _report_hints(kernel, rich.op if rich else ""):
            console.print(f"  - {hint}", markup=False)


def format_profiling_json(
    result: ProfilingResult,
    analysis_results: list[QueryResult] | None = None,
) -> str:
    """Format profiling result as JSON, merging trace context from Phase 1."""
    trace_ctx = add_canonical_aliases(_build_trace_context(analysis_results))
    name_counts = Counter(k.kernel_name for k in result.kernels)

    out: dict[str, Any] = {
        "profilelens_version": _get_version(),
        "gpu": {"arch": result.gpu_arch, "model": result.gpu_model},
        "detected_dtype": result.detected_dtype,
        "observed_dtypes": [dt for dt in result.observed_dtypes if dt != UNKNOWN_DTYPE],
        "counter_set": result.counter_set_name,
        "counter_selection": {
            "caller_dtype_hints": list(result.counter_dtype_hints),
            "automatic_dtype_coverage": list(result.counter_dtypes),
            "unresolved_kernels": list(result.unresolved_kernels),
            "note": _counter_selection_note(result),
        },
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
            "kernels_profiled": list(name_counts),
            "unique_kernel_count": len(name_counts),
            "profile_row_count": len(result.kernels),
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
        tc = lookup_kernel(trace_ctx, k.kernel_name)
        if tc:
            entry["trace_context"] = {
                **tc, "scope": "kernel_name",
                "shared_across_variants": name_counts[k.kernel_name] > 1,
            }
        entry["profiled_timing"] = {
            "dispatch_count": k.dispatch_count,
            "average_us": k.avg_duration_us,
            "median_us": getattr(k, "median_duration_us", None),
            "total_us": k.total_duration_us,
            "samples_us": list(getattr(k, "duration_samples_us", ())),
            "source": "hardware_counter_dispatch_timestamps",
        }
        entry["hw_counters"] = {
            "mfma_flops": k.mfma_flops,
            "mfma_flops_by_family": getattr(k, "mfma_flops_by_family", {}),
            "mfma_fraction_by_family": getattr(k, "mfma_fraction_by_family", {}),
            "valu_flops": k.valu_flops,
            "valu_flops_note": "Unavailable from untyped instruction counts; excluded from FLOPs",
            "valu_lane_instructions": k.valu_lane_instructions,
            "valu_lane_instructions_basis": (
                "SQ_INSTS_VALU * wave_size; untyped full-wave estimate, not FLOPs"
            ),
            "mfma_counter_families": list(k.mfma_counter_families),
            "flops_basis": (
                "sum of collected floating-point MFMA families * 512; uncollected work unknown"
            ),
            "hbm_bytes": k.hbm_bytes,
            "hbm_bytes_by_kind": getattr(k, "hbm_bytes_by_kind", {}),
            "hbm_bytes_source": k.hbm_bytes_source,
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
            "valu_busy_source": "SDK VALUBusy" if "VALUBusy" in k.raw_counters else "unavailable",
            "valu_lane_utilization_pct": k.valu_lane_utilization_pct,
            "valu_lane_utilization_source": (
                "SDK VALUUtilization" if "VALUUtilization" in k.raw_counters else "unavailable"
            ),
            "bw_util_pct": k.bw_util_pct,
            "occupancy_pct": k.occupancy_pct,
            "occupancy_source": (
                "SDK OccupancyPercent" if "OccupancyPercent" in k.raw_counters else "unavailable"
            ),
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
        tc_entry = lookup_kernel(trace_ctx, k.kernel_name, {})
        e2e_val = tc_entry.get("e2e_pct", 0.0)
        from profilelens.profile.bottleneck import compute_headroom as _chr
        hr_entry = _chr(k.mfma_util_pct, k.bw_util_pct, k.bottleneck, e2e_val)
        entry["classification"] = {
            "bottleneck": k.bottleneck,
            "pct_of_peak": hr_entry.pct_of_peak,
            "headroom": hr_entry.headroom,
            "priority_score": hr_entry.priority_score if name_counts[k.kernel_name] == 1 else None,
            "priority_score_scope": "kernel_name" if name_counts[k.kernel_name] == 1
                else "unavailable_without_variant_timing",
        }
        entry["gap_analysis"] = {
            "hints": list(k.gap_hints),
            "report_hints": _report_hints(k, tc_entry.get("op", "")),
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
    """Render the full report as concise tables and per-kernel bullets."""
    from pathlib import Path

    from profilelens.profile.bottleneck import compute_headroom
    from profilelens.profile.dtype_detect import (
        _flydsl_args_lookup,
        _flydsl_stem,
        flydsl_stem_collisions,
        format_flydsl_arg_shapes,
        graph_gid_for_sig,
        graph_op_for_kernel,
    )
    from profilelens.profile.kernel_context import build_kernel_context_map
    from profilelens.profile.roofline_plot import render_roofline_png

    trace_ctx = add_canonical_aliases(_build_trace_context(analysis_results))
    rich_ctx = add_canonical_aliases(build_kernel_context_map(analysis_results)) if analysis_results else {}
    names = Counter(k.kernel_name for k in result.kernels)
    if flydsl_args:
        for warning in flydsl_stem_collisions([k.kernel_name for k in result.kernels], flydsl_args):
            logger.warning(warning)

    details = []
    for kernel in result.kernels:
        context = lookup_kernel(trace_ctx, kernel.kernel_name, {})
        rich = lookup_kernel(rich_ctx, kernel.kernel_name)
        shape = _fmt_shape_sig(kernel.shape_sig) if kernel.shape_sig else ""
        if not shape and rich is not None:
            shape = rich.shape
        op = rich.op if rich is not None and rich.op != "unknown" else ""
        if not shape and kernel.dtype_source == "flydsl_hook" and flydsl_args:
            shape = format_flydsl_arg_shapes(_flydsl_args_lookup(kernel.kernel_name, flydsl_args))
        if not op and kernel.dtype_source == "flydsl_hook":
            stem = _flydsl_stem(kernel.kernel_name)
            op = f"flydsl::{stem}" if stem else ""
        if not op and capture_graphs:
            gid = graph_gid_for_sig(kernel.shape_sig, capture_graphs, kernel.kernel_name) if kernel.shape_sig else ""
            op = graph_op_for_kernel(kernel.kernel_name, capture_graphs,
                                     prefer_gid=gid, prefer_sig=kernel.shape_sig)
        share = context.get("e2e_pct", rich.e2e_pct if rich is not None else 0)
        headroom = compute_headroom(kernel.mfma_util_pct, kernel.bw_util_pct,
                                    kernel.bottleneck, share)
        details.append((kernel, context, rich, shape, op, share, headroom))

    source_label, source_value = _selection_label(result)
    lines = ["# ProfileLens Kernel Report\n",
             f"- **GPU:** {html.escape(result.gpu_model)} ({html.escape(result.gpu_arch)})",
             f"- **{source_label}:** {html.escape(source_value)}",
             f"- **Measured:** {len(names)} kernel names · {len(result.kernels)} profile rows",
             f"- **Counter set:** {_markdown_code(result.counter_set_name)}", "",
             f"## 1. Top-{len(names)} Kernels\n",
             "| # | Kernel | Shape / variant | GPU time % | Calls | Category | Compute util % | BW util % | Headroom % | Priority % |",
             "|---|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for index, (kernel, context, _rich, shape, _op, share, headroom) in enumerate(details, 1):
        variant = shape or kernel.variant_label
        variant = _markdown_code(variant) if variant else "N/A"
        shared = names[kernel.kernel_name] > 1
        headroom_pct = ("N/A (shared timing)" if shared else
                        _fmt_pct(headroom.headroom * 100) if headroom.headroom >= 0 else "N/A")
        priority = ("N/A (shared timing)" if shared else
                    _fmt_pct(headroom.priority_score * 100, 2) if headroom.priority_score >= 0 else "N/A")
        lines.append(
            f"| {index} | {_markdown_code(_elide(kernel.kernel_name, 52))} | {variant} | "
            f"{_fmt_pct(share)} | {context.get('trace_count', kernel.dispatch_count)} | "
            f"{_display_category(kernel)} | {_fmt_pct(kernel.mfma_util_pct)} | "
            f"{_fmt_pct(kernel.bw_util_pct)} | {headroom_pct} | {priority} |"
        )

    lines.extend(["", "## 2. Empirical per-kernel roofline chart\n",
                  f"- **Measured reference peaks:** {_format_peaks_bullet(result)}",
                  f"- **{source_label}:** {html.escape(source_value)}", ""])
    if output_dir and result.roofline_peaks and result.kernels:
        image = render_roofline_png(result.roofline_peaks, result.kernels,
                                    Path(output_dir) / "roofline.png", title="Kernels",
                                    peaks_by_dtype=result.peaks_by_dtype)
        if image:
            lines.extend([f"![Kernels]({image.name})", ""])

    lines.extend(["## Kernels profiled and counters used\n",
                  f"- **PMC counters captured ({len(result.counters_used)}):** "
                  + ", ".join(_markdown_code(name) for name in result.counters_used),
                  f"- **Kernels profiled:** {len(names)} unique names · {len(result.kernels)} profile rows — "
                  + ", ".join(_markdown_code(_elide(name, 50)) for name in names), "",
                  "| # | Kernel | GPU time % | Category | Input | FLOP/byte | MFMA throughput | DRAM rate | Issue wait |",
                  "|---|---|---:|---|---|---:|---:|---:|---:|"])
    for index, (kernel, _context, _rich, _shape, _op, share, _headroom) in enumerate(details, 1):
        lines.append(
            f"| {index} | {_markdown_code(_elide(kernel.kernel_name, 52))} | {_fmt_pct(share)} | "
            f"{_display_category(kernel)} | {_display_dtype(result, kernel)} | "
            f"{_fmt_metric(kernel.ai_hbm)} | {_flop_rate(kernel.achieved_tflops)} | "
            f"{_bandwidth(kernel.achieved_bw_tbps)} | {_percent(kernel.stall_issue_pct)} |"
        )
    lines.append("")

    for index, (kernel, context, rich, shape, op, _share, headroom) in enumerate(details, 1):
        heading = _markdown_code(kernel.kernel_name)
        if shape:
            heading += " — shape " + _markdown_code(shape)
        if kernel.variant_label:
            heading += " (" + _markdown_code(kernel.variant_label) + ")"
        lines.extend([f"## Kernel {index}: {heading}\n", "**Trace analysis:**", ""])
        shared = names[kernel.kernel_name] > 1
        for label, value in _trace_fields(result, kernel, context, rich, shape, op, shared):
            rendered = _markdown_code(value) if label in {"Op", "Module", "Shape"} else html.escape(value)
            lines.append(f"- **{label}:** {rendered}")
        lines.extend(["", "**Perf counter analysis:**", ""])
        for label, value in _counter_fields(result, kernel, headroom, shared):
            lines.append(f"- **{label}:** {html.escape(value)}")
        lines.extend(["", f"**Category:** {_display_category(kernel)}", "",
                      "**Kernel Optimization Hints:**", ""])
        lines.extend("- " + html.escape(hint) for hint in _report_hints(kernel, op))
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
    if k.bottleneck == "UNKNOWN":
        return "unknown"
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
    return "Unassessed — compute dtype, counter coverage or required ceiling is unavailable"


def _get_version() -> str:
    try:
        from profilelens import __version__
        return __version__
    except Exception:
        return "unknown"
