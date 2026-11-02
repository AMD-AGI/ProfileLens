# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Derive performance metrics from raw HW counters."""

from __future__ import annotations

from typing import Any

from profilelens.dtypes import KernelProfile, RooflinePeaks
from profilelens.profile.parser import AggregatedKernel


# empirical roofline peaks default
UNKNOWN_UTIL_PCT = -1.0

# Source: ROCm/rocprofiler-compute@rocm-7.2.4
# analysis_configs/<arch>/1600_vector_l1_data_cache.yaml
_CACHE_TRAFFIC_WIDTHS: dict[str, tuple[float, float, float]] = {
    "gfx942": (128.0, 128.0, 64.0),
    "gfx950": (128.0, 128.0, 64.0),
}


_COLLECTIVE_PATTERNS = (
    "cross_device",        # vllm::cross_device_reduce_*, aiter reduce_scatter_cross_device_store
    "reduce_scatter",
    "all_reduce", "allreduce",
    "all_gather", "allgather",
    "all_to_all", "alltoall", "all2all",
    "reducescatter",
    "sendrecv", "send_recv",
    "rccl", "nccl",        # ncclDevKernel_*, librccl device kernels
)


def is_collective_kernel(name: str) -> bool:
    """True if a kernel is a cross-device communication / collective kernel."""
    if not name:
        return False
    low = name.lower()
    return any(p in low for p in _COLLECTIVE_PATTERNS)


def _get(counters: dict[str, float], *keys: str) -> float:
    """Get the first matching counter value."""
    for k in keys:
        if k in counters:
            return counters[k]
    return 0.0


def compute_mfma_flops(counters: dict[str, float], dtype: str | None) -> float:
    """Compute MFMA FLOPS from counter values."""
    # ROCprofiler-SDK 7.2.4 derived_counters.xml defines each
    # SQ_INSTS_VALU_MFMA_MOPS_* unit as 512 operations.
    if dtype:
        from profilelens.profile.dtype_detect import mfma_counter_for_dtype
        key = mfma_counter_for_dtype(dtype)
        if key in counters:
            return counters[key] * 512.0

    total = 0.0
    for suffix in ("BF16", "F16", "F8", "F32", "F64", "I8"):
        total += _get(counters, f"SQ_INSTS_VALU_MFMA_MOPS_{suffix}") * 512.0
    return total


def compute_valu_flops(counters: dict[str, float], wave_size: int | None) -> float:
    """Approximate VALU FLOPS using the runtime-reported wavefront width."""
    if not wave_size or wave_size <= 0:
        return 0.0
    return _get(counters, "SQ_INSTS_VALU") * float(wave_size)


def compute_hbm_bytes(
    counters: dict[str, float], xcc_count: int | None = None
) -> float:
    """Read architecture-resolved HBM traffic from ROCprofiler-SDK metrics."""
    if "BANDWIDTH_EA" in counters and xcc_count and xcc_count > 0:
        active_cycles = _get(counters, "GRBM_GUI_ACTIVE")
        return max(counters["BANDWIDTH_EA"] * active_cycles / xcc_count, 0.0)

    # FETCH_SIZE and WRITE_SIZE are derived by the installed SDK in KiB. Their
    # underlying raw-counter formulas vary by architecture, so ProfileLens must
    # not duplicate those formulas or select them from a product-name table.
    fetched_kib = _get(counters, "FETCH_SIZE", "FetchSize")
    written_kib = _get(counters, "WRITE_SIZE", "WriteSize")
    return max((fetched_kib + written_kib) * 1024.0, 0.0)


def compute_l2_bytes(
    counters: dict[str, float], arch: str | None = None
) -> float | None:
    """Return bytes transferred into L2."""
    for key in ("L2_BYTES", "L2Bytes"):
        if key in counters:
            return counters[key]
    widths = _CACHE_TRAFFIC_WIDTHS.get((arch or "").lower())
    if widths is None:
        return None
    required = ("TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum")
    if all(key in counters for key in required):
        _, read_bytes, write_bytes = widths
        read_requests = counters["TCP_TCC_READ_REQ_sum"]
        write_requests = sum(
            _get(counters, key)
            for key in (
                "TCP_TCC_WRITE_REQ_sum",
                "TCP_TCC_ATOMIC_WITH_RET_REQ_sum",
                "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum",
            )
        )
        return max(read_requests * read_bytes + write_requests * write_bytes, 0.0)
    return None


def compute_l1_bytes(
    counters: dict[str, float], arch: str | None = None
) -> float | None:
    """Return bytes presented to L1."""
    for key in ("L1_BYTES", "L1Bytes"):
        if key in counters:
            return counters[key]
    widths = _CACHE_TRAFFIC_WIDTHS.get((arch or "").lower())
    if widths is not None and "TCP_TOTAL_CACHE_ACCESSES_sum" in counters:
        l1_access_bytes, _, _ = widths
        return max(
            counters["TCP_TOTAL_CACHE_ACCESSES_sum"] * l1_access_bytes,
            0.0,
        )
    return None


def _cache_bytes_source(
    counters: dict[str, float],
    sdk_keys: tuple[str, ...],
    raw_keys: tuple[str, ...],
    arch: str,
) -> str:
    if any(key in counters for key in sdk_keys):
        return "sdk"
    if arch.lower() in _CACHE_TRAFFIC_WIDTHS and all(
        key in counters for key in raw_keys
    ):
        return f"rocprofiler_compute_{arch.lower()}_estimated"
    return "unavailable"


def _bounded_pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return UNKNOWN_UTIL_PCT
    return min(max(numerator / denominator * 100.0, 0.0), 100.0)


def compute_l1_hit_rate(counters: dict[str, float]) -> tuple[float, str]:
    """Return L1 hit percentage and counter value."""
    required = (
        "TCP_TOTAL_CACHE_ACCESSES_sum",
        "TCP_TCC_READ_REQ_sum",
        "TCP_TCC_WRITE_REQ_sum",
        "TCP_TCC_ATOMIC_WITH_RET_REQ_sum",
    )
    if not all(key in counters for key in required):
        return UNKNOWN_UTIL_PCT, "unavailable"

    total = counters["TCP_TOTAL_CACHE_ACCESSES_sum"]
    misses = sum(counters[key] for key in required[1:])
    source = "raw_estimated"
    without_return = "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"
    if without_return in counters:
        misses += counters[without_return]
        source = "raw_exact"
    return _bounded_pct(total - misses, total), source


def compute_l2_hit_rate(counters: dict[str, float]) -> tuple[float, str]:
    """Return L2 hit percentage and SDK/raw/estimated value."""
    for key in ("L2CacheHit", "L2CacheHitRate"):
        if key in counters:
            return min(max(counters[key], 0.0), 100.0), "sdk"

    if "TCC_HIT_sum" in counters and "TCC_MISS_sum" in counters:
        hits = counters["TCC_HIT_sum"]
        misses = counters["TCC_MISS_sum"]
        return _bounded_pct(hits, hits + misses), "raw_exact"

    proxy_keys = (
        "TCP_TCC_READ_REQ_sum",
        "TCP_TCC_WRITE_REQ_sum",
        "TCP_TCC_ATOMIC_WITH_RET_REQ_sum",
        "TCC_EA0_RDREQ_sum",
        "TCC_EA0_WRREQ_sum",
    )
    if all(key in counters for key in proxy_keys):
        l2_requests = sum(counters[key] for key in proxy_keys[:3])
        if "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum" in counters:
            l2_requests += counters["TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"]
        memory_requests = counters["TCC_EA0_RDREQ_sum"] + counters["TCC_EA0_WRREQ_sum"]
        return _bounded_pct(l2_requests - memory_requests, l2_requests), "raw_estimated"

    return UNKNOWN_UTIL_PCT, "unavailable"


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _per_dispatch_roofline(
    dispatches: "tuple[tuple[dict[str, float], int], ...]",
    dtype: str | None,
    wave_size: int | None,
    xcc_count: int | None,
    arch: str,
) -> "dict[str, float | None] | None":

    import statistics

    bws: list[float] = []
    tfs: list[float] = []
    ai_hbm: list[float] = []
    ai_l2: list[float] = []
    ai_l1: list[float] = []
    for counters, dur_ns in dispatches:
        if dur_ns <= 0:
            continue
        dur_s = dur_ns / 1e9
        flops = compute_mfma_flops(counters, dtype) + compute_valu_flops(
            counters, wave_size
        )
        hbm = compute_hbm_bytes(counters, xcc_count)
        l2 = compute_l2_bytes(counters, arch)
        l1 = compute_l1_bytes(counters, arch)
        bws.append(hbm / dur_s)
        tfs.append(flops / dur_s / 1e12)
        if hbm > 0:
            ai_hbm.append(flops / hbm)
        if l2 is not None and l2 > 0:
            ai_l2.append(flops / l2)
        if l1 is not None and l1 > 0:
            ai_l1.append(flops / l1)
    if not bws:
        return None
    return {
        "achieved_bw": statistics.median(bws),
        "achieved_tflops": statistics.median(tfs),
        "ai_hbm": statistics.median(ai_hbm) if ai_hbm else 0.0,
        "ai_l2": statistics.median(ai_l2) if ai_l2 else None,
        "ai_l1": statistics.median(ai_l1) if ai_l1 else None,
    }


def compute_kernel_metrics(
    kernel: AggregatedKernel,
    roofline_peaks: RooflinePeaks | None,
    dtype: str | None,
    gpu_info: Any = None,
    dtype_source: str = "global",
) -> KernelProfile:
    """Compute all derived metrics for an aggregated kernel."""
    c = kernel.avg_counters
    has_counter_data = bool(c)
    dur_ns = kernel.avg_duration_ns
    dur_s = dur_ns / 1e9 if dur_ns > 0 else 1e-15

    wave_size = getattr(gpu_info, "wave_size", 0) if gpu_info else 0
    xcc_count = getattr(gpu_info, "xcc_count", 0) if gpu_info else 0
    arch = getattr(gpu_info, "arch", "") if gpu_info else ""
    mfma_flops = compute_mfma_flops(c, dtype)
    valu_flops = compute_valu_flops(c, wave_size)
    total_flops = mfma_flops + valu_flops

    hbm_bytes = compute_hbm_bytes(c, xcc_count)
    l2_bytes = compute_l2_bytes(c, arch)
    l1_bytes = compute_l1_bytes(c, arch)
    l2_bytes_source = _cache_bytes_source(
        c,
        ("L2_BYTES", "L2Bytes"),
        ("TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum"),
        arch,
    )
    l1_bytes_source = _cache_bytes_source(
        c,
        ("L1_BYTES", "L1Bytes"),
        ("TCP_TOTAL_CACHE_ACCESSES_sum",),
        arch,
    )

    ai_hbm = _safe_div(total_flops, hbm_bytes)
    ai_l2 = total_flops / l2_bytes if l2_bytes is not None and l2_bytes > 0 else None
    ai_l1 = total_flops / l1_bytes if l1_bytes is not None and l1_bytes > 0 else None

    achieved_bw = hbm_bytes / dur_s if dur_s > 0 else 0.0
    achieved_tflops = total_flops / dur_s / 1e12 if dur_s > 0 else 0.0
    achieved_bw_tbps = achieved_bw / 1e12


    _pd = _per_dispatch_roofline(
        getattr(kernel, "dispatches", ()), dtype, wave_size, xcc_count, arch
    )
    if _pd is not None:
        ai_hbm = _pd["ai_hbm"]
        ai_l2 = _pd["ai_l2"]
        ai_l1 = _pd["ai_l1"]
        achieved_bw = float(_pd["achieved_bw"])
        achieved_tflops = float(_pd["achieved_tflops"])
        achieved_bw_tbps = achieved_bw / 1e12

    if roofline_peaks is not None:
        peak_bw_tbps = roofline_peaks.peak_bw_tbps.get("hbm", 0.0)
        peak_mfma_tflops = roofline_peaks.peak_compute_tflops.get("mfma", 0.0)
        if peak_mfma_tflops <= 0:
            peak_mfma_tflops = roofline_peaks.peak_compute_tflops.get("valu", 0.0)
        peak_bw = peak_bw_tbps * 1e12
        peak_tflops = peak_mfma_tflops * 1e12
    else:
        peak_bw = 0.0
        peak_tflops = 0.0

    if has_counter_data and peak_tflops > 0 and dur_s > 0:
        mfma_util = _safe_div(mfma_flops / dur_s, peak_tflops) * 100.0
    else:
        mfma_util = UNKNOWN_UTIL_PCT
    if has_counter_data and hbm_bytes > 0 and peak_bw > 0:
        bw_util = _safe_div(achieved_bw, peak_bw) * 100.0
    else:
        bw_util = UNKNOWN_UTIL_PCT

    grbm_active = _get(c, "GRBM_GUI_ACTIVE")

    # Vector-ALU engine busy %, rocprof-compute's formula:
    #   100 * SQ_ACTIVE_INST_VALU / CU_NUM / GRBM_GUI_ACTIVE
    valu_busy_pct = -1.0
    valu_active = _get(c, "SQ_ACTIVE_INST_VALU")
    cu_count = getattr(gpu_info, "cu_count", 0) if gpu_info else 0
    if valu_active > 0 and cu_count and grbm_active > 0:
        valu_busy_pct = 100.0 * valu_active / cu_count / grbm_active

    # Wavefront occupancy = time-averaged concurrent waves / max concurrent waves.
    wave_cycles = _get(c, "SQ_WAVE_CYCLES")  # per-SE sum, units of 4 cycles
    max_waves_per_cu = getattr(gpu_info, "max_waves_per_cu", 0) if gpu_info else 0
    max_concurrent_waves = cu_count * max_waves_per_cu
    if wave_cycles > 0 and grbm_active > 0 and max_concurrent_waves > 0:
        occupancy = (wave_cycles * 4.0) / grbm_active / max_concurrent_waves * 100.0
    else:
        occupancy = UNKNOWN_UTIL_PCT

    l1_hit, l1_hit_source = compute_l1_hit_rate(c)
    l2_hit, l2_hit_source = compute_l2_hit_rate(c)

    stall_issue = -1.0
    stall_vmem = -1.0
    stall_lds = -1.0
    dominant_stall = "N/A"
    issue_raw = _get(c, "SQ_WAIT_INST_ANY")
    vmem_raw = _get(c, "SQ_WAIT_INST_VMEM")
    lds_raw = _get(c, "SQ_WAIT_INST_LDS")
    if "SQ_WAIT_INST_ANY" in c and wave_cycles > 0:
        stall_issue = _bounded_pct(issue_raw, wave_cycles)
    if "SQ_WAIT_INST_VMEM" in c and wave_cycles > 0:
        stall_vmem = _bounded_pct(vmem_raw, wave_cycles)
    if "SQ_WAIT_INST_LDS" in c and wave_cycles > 0:
        stall_lds = _bounded_pct(lds_raw, wave_cycles)

    if stall_vmem >= 0 and stall_lds >= 0:
        stall_source = "vmem_lds"
        if stall_vmem >= stall_lds:
            dominant_stall = f"VMEM {stall_vmem:.0f}%"
        else:
            dominant_stall = f"LDS {stall_lds:.0f}%"
    elif stall_issue >= 0:
        stall_source = "total_issue_lds" if stall_lds >= 0 else "total_issue"
        dominant_stall = f"Issue wait {stall_issue:.0f}%"
        if stall_lds >= 0:
            dominant_stall += f" (LDS {stall_lds:.0f}% subset)"
    elif stall_lds >= 0:
        stall_source = "lds_only"
        dominant_stall = f"LDS {stall_lds:.0f}% (VMEM N/A)"
    else:
        stall_source = "unavailable"

    from profilelens.profile.bottleneck import classify_bottleneck, generate_gap_hints
    if is_collective_kernel(kernel.kernel_name):
        # Cross-device communication: the single-GPU compute/HBM roofline does not apply.
        bottleneck = "COLLECTIVE"
        mfma_util = UNKNOWN_UTIL_PCT
        bw_util = UNKNOWN_UTIL_PCT
        nr = getattr(kernel, "n_ranks", 1)
        hints = [
            "Cross-device collective/communication kernel — not scored against "
            "the single-GPU roofline (XGMI/network bound). %-of-peak is N/A.",
        ]
        if nr > 1:
            per_rank = getattr(kernel, "per_rank", ())
            durs = [r.avg_duration_ns for r in per_rank if r.avg_duration_ns > 0]
            if len(durs) >= 2 and min(durs) > 0 and max(durs) / min(durs) >= 2.0:
                hints.append(
                    f"Per-rank duration imbalance {min(durs)/1000:.1f}–"
                    f"{max(durs)/1000:.1f} us across {nr} ranks -> likely "
                    "load imbalance / sync wait upstream, not the collective itself."
                )
    elif not has_counter_data:
        bottleneck = "UNKNOWN"
        hints = [
            "No HW counter data collected for this kernel — rocprofv3 likely "
            "dropped PMC samples. Re-run with rocprofv3 wrapping `vllm serve` "
            "directly to capture counters.",
        ]
    elif mfma_util >= 0 and bw_util >= 0:
        bottleneck = classify_bottleneck(mfma_util, bw_util)
        hints = generate_gap_hints(
            mfma_util, bw_util, occupancy, l1_hit, l2_hit,
            stall_vmem, stall_lds, stall_issue,
        )
    else:
        bottleneck = "UNKNOWN"
        if roofline_peaks is not None and peak_bw > 0 and hbm_bytes <= 0:
            hints = [
                "No positive HBM traffic was reported for this kernel sample — "
                "bandwidth utilization and roofline classification are N/A."
            ]
        else:
            hints = [
                "Empirical roofline peaks unavailable — re-run without --no-roofline "
                "to enable utilization metrics."
            ]

    return KernelProfile(
        kernel_name=kernel.kernel_name,
        dispatch_count=kernel.dispatch_count,
        avg_duration_us=kernel.avg_duration_ns / 1000.0,
        total_duration_us=kernel.total_duration_ns / 1000.0,
        detected_dtype=dtype or "mixed",
        mfma_flops=mfma_flops,
        valu_flops=valu_flops,
        total_flops=total_flops,
        hbm_bytes=hbm_bytes,
        l2_bytes=l2_bytes,
        l1_bytes=l1_bytes,
        ai_hbm=round(ai_hbm, 2),
        ai_l2=round(ai_l2, 2) if ai_l2 is not None else None,
        ai_l1=round(ai_l1, 2) if ai_l1 is not None else None,
        achieved_bw_tbps=round(achieved_bw_tbps, 4),
        achieved_tflops=round(achieved_tflops, 4),
        mfma_util_pct=round(mfma_util, 1),
        bw_util_pct=round(bw_util, 1),
        occupancy_pct=round(occupancy, 1),
        l1_hit_rate_pct=round(l1_hit, 1) if l1_hit >= 0 else UNKNOWN_UTIL_PCT,
        l2_hit_rate_pct=round(l2_hit, 1) if l2_hit >= 0 else UNKNOWN_UTIL_PCT,
        bottleneck=bottleneck,
        l1_hit_rate_source=l1_hit_source,
        l2_hit_rate_source=l2_hit_source,
        l1_bytes_source=l1_bytes_source,
        l2_bytes_source=l2_bytes_source,
        valu_busy_pct=round(valu_busy_pct, 1) if valu_busy_pct >= 0 else -1.0,
        stall_issue_pct=round(stall_issue, 1) if stall_issue >= 0 else -1.0,
        stall_vmem_pct=round(stall_vmem, 1) if stall_vmem >= 0 else -1.0,
        stall_lds_pct=round(stall_lds, 1) if stall_lds >= 0 else -1.0,
        stall_source=stall_source,
        dominant_stall=dominant_stall,
        raw_counters=dict(c),
        gap_hints=tuple(hints),
        dtype_source=dtype_source,
        shape_sig=kernel.shape_sig,
        variant_label=getattr(kernel, "variant_label", ""),
    )
