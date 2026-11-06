# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Data types for ProfileLens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceInfo:
    """Metadata about a loaded trace file."""

    path: str
    tables: tuple[str, ...] = ()
    row_count: int = 0


@dataclass(frozen=True)
class QueryResult:
    """Result of a single SQL query against a trace."""

    sql: str
    label: str
    rows: tuple[dict[str, Any], ...] = ()
    columns: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class KernelProfile:
    """HW-counter-derived metrics for a single kernel."""

    kernel_name: str
    dispatch_count: int
    avg_duration_us: float
    total_duration_us: float
    detected_dtype: str
    mfma_flops: float | None
    valu_flops: float | None
    total_flops: float | None
    hbm_bytes: float | None
    l2_bytes: float | None
    l1_bytes: float | None
    ai_hbm: float | None
    ai_l2: float | None
    ai_l1: float | None
    achieved_bw_tbps: float | None
    achieved_tflops: float | None
    mfma_util_pct: float
    bw_util_pct: float
    occupancy_pct: float
    l1_hit_rate_pct: float
    l2_hit_rate_pct: float
    bottleneck: str
    l1_hit_rate_source: str = "unavailable"
    l2_hit_rate_source: str = "unavailable"
    l1_bytes_source: str = "unavailable"
    l2_bytes_source: str = "unavailable"
    valu_busy_pct: float = -1.0
    valu_lane_utilization_pct: float = -1.0
    valu_lane_instructions: float | None = None
    mfma_counter_families: tuple[str, ...] = ()
    hbm_bytes_source: str = "unavailable"
    stall_issue_pct: float = -1.0
    stall_vmem_pct: float = -1.0
    stall_lds_pct: float = -1.0
    stall_source: str = "unavailable"
    dominant_stall: str = "N/A"
    raw_counters: dict[str, Any] = field(default_factory=dict)
    gap_hints: tuple[str, ...] = ()
    dtype_source: str = "global"
    shape_sig: str = ""
    variant_label: str = ""
    median_duration_us: float | None = None
    duration_samples_us: tuple[float, ...] = ()
    mfma_flops_by_family: dict[str, float] = field(default_factory=dict)
    mfma_fraction_by_family: dict[str, float] = field(default_factory=dict)
    hbm_bytes_by_kind: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RooflinePeaks:
    """Empirical or theoretical peak performance for a GPU."""

    gpu_model: str
    detected_dtype: str
    peak_bw_tbps: dict[str, float] = field(default_factory=dict)
    peak_compute_tflops: dict[str, float] = field(default_factory=dict)
    ridge_point: float = 0.0


@dataclass(frozen=True)
class ProfilingResult:
    """Complete result of a profiling run."""

    gpu_arch: str
    gpu_model: str
    detected_dtype: str
    kernels: tuple[KernelProfile, ...] = ()
    roofline_peaks: RooflinePeaks | None = None
    counter_set_name: str = "auto"
    output_dir: str = ""
    profiling_duration_s: float = 0.0
    kernel_cv: dict[str, Any] = field(default_factory=dict)
    sample_iteration_ids: tuple[int, ...] = ()
    sample_dispatches_per_kernel: int = 0
    phase1_workload_s: float = 0.0
    phase2_workload_s: float = 0.0
    counters_used: tuple[str, ...] = ()
    observed_dtypes: tuple[str, ...] = ()
    peaks_by_dtype: dict[str, "RooflinePeaks"] = field(default_factory=dict)
    per_region_kernels: dict[str, tuple["KernelProfile", ...]] = field(default_factory=dict)
    trace_context_source: str = "perfetto_sql"
    counter_dtype_hints: tuple[str, ...] = ()
    counter_dtypes: tuple[str, ...] = ()
    unresolved_kernels: tuple[str, ...] = ()
