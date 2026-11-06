# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Derive performance metrics from raw HW counters."""

from __future__ import annotations

import math
import statistics
from typing import Any

from profilelens.dtypes import KernelProfile, RooflinePeaks
from profilelens.profile.counter_sets import DRAM_REQUEST_BYTE_COUNTERS
from profilelens.profile.dtype_detect import UNKNOWN_DTYPE
from profilelens.profile.parser import AggregatedKernel

# Sentinel for unavailable percentages.
UNKNOWN_UTIL_PCT = -1.0
MFMA_FLOAT_COUNTERS = tuple(
    f"SQ_INSTS_VALU_MFMA_MOPS_{suffix}"
    for suffix in ("BF16", "F16", "F8", "F6F4", "F32", "F64")
)

# Bytes/request from rocprofiler-compute rocm-7.2.4 (1600_vector_l1_data_cache.yaml).
_CACHE_TRAFFIC_WIDTHS: dict[str, tuple[float, float, float]] = {
    "gfx942": (128.0, 128.0, 64.0),
    "gfx950": (128.0, 128.0, 64.0),
}


_COLLECTIVE_PATTERNS = (
    "cross_device",
    "reduce_scatter",
    "all_reduce", "allreduce",
    "all_gather", "allgather",
    "all_to_all", "alltoall", "all2all",
    "reducescatter",
    "sendrecv", "send_recv",
    "rccl", "nccl",
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


def _counter_sum(counters: dict[str, float], names: tuple[str, ...]) -> float:
    values = [counters[name] for name in names]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f"Counter values must be finite and nonnegative: {names}")
    return sum(values)


def compute_mfma_flops_by_family(counters: dict[str, float]) -> dict[str, float]:
    """Operations per collected MFMA family (512 ops per MOPS unit)."""
    return {
        name.removeprefix("SQ_INSTS_VALU_MFMA_MOPS_"): _counter_sum(counters, (name,)) * 512.0
        for name in MFMA_FLOAT_COUNTERS if name in counters
    }


def compute_mfma_flops(counters: dict[str, float], dtype: str | None) -> float | None:
    """Count only collected floating-point MFMA families; missing is not zero."""
    families = compute_mfma_flops_by_family(counters)
    return sum(families.values()) if families else None


def compute_valu_lane_instructions(
    counters: dict[str, float], wave_size: int | None
) -> float | None:
    """Untyped full-wave lane-instruction estimate, not floating-point work."""
    if not wave_size or wave_size <= 0 or "SQ_INSTS_VALU" not in counters:
        return None
    return _counter_sum(counters, ("SQ_INSTS_VALU",)) * float(wave_size)


def compute_hbm_bytes_by_kind(counters: dict[str, float]) -> dict[str, float]:
    """Break down a complete SDK traffic source without inventing absent components."""
    if all(name in counters for name in DRAM_REQUEST_BYTE_COUNTERS):
        return {
            kind: 32.0 * _counter_sum(counters, (name,))
            for kind, name in zip(("read", "write", "atomic"), DRAM_REQUEST_BYTE_COUNTERS)
        }

    # The SDK reports FETCH_SIZE/WRITE_SIZE in KiB.
    read = next((name for name in ("FETCH_SIZE", "FetchSize") if name in counters), None)
    write = next((name for name in ("WRITE_SIZE", "WriteSize") if name in counters), None)
    if read is None or write is None:
        return {}
    return {
        "read": _counter_sum(counters, (read,)) * 1024.0,
        "write": _counter_sum(counters, (write,)) * 1024.0,
    }


def compute_hbm_bytes(
    counters: dict[str, float], xcc_count: int | None = None
) -> float | None:
    """Read architecture-resolved HBM traffic from ROCprofiler-SDK metrics."""
    components = compute_hbm_bytes_by_kind(counters)
    return sum(components.values()) if components else None


def memory_traffic_source(counters: dict[str, float]) -> str:
    if all(name in counters for name in DRAM_REQUEST_BYTE_COUNTERS):
        return "sdk_dram_request_32b_units"
    if compute_hbm_bytes(counters) is not None:
        return "sdk_fetch_write_kib"
    return "unavailable"


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


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
    direct = ("TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_CACHE_MISS")
    if all(name in counters for name in direct):
        total = _counter_sum(counters, (direct[0],))
        misses = _counter_sum(counters, (direct[1],))
        if total <= 0 or misses > total:
            return UNKNOWN_UTIL_PCT, "unavailable"
        return (total - misses) / total * 100.0, "raw_direct_tcp_miss_request_ratio"

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

    bws: list[float] = []
    tfs: list[float] = []
    ai_hbm: list[float] = []
    ai_l2: list[float] = []
    ai_l1: list[float] = []
    valid_duration = False
    for counters, dur_ns in dispatches:
        if dur_ns <= 0:
            continue
        valid_duration = True
        dur_s = dur_ns / 1e9
        flops = compute_mfma_flops(counters, dtype)
        hbm = compute_hbm_bytes(counters, xcc_count)
        l2 = compute_l2_bytes(counters, arch)
        l1 = compute_l1_bytes(counters, arch)
        if hbm is not None:
            bws.append(hbm / dur_s)
        if flops is not None:
            tfs.append(flops / dur_s / 1e12)
        if flops is not None and hbm is not None and hbm > 0:
            ai_hbm.append(flops / hbm)
        if flops is not None and l2 is not None and l2 > 0:
            ai_l2.append(flops / l2)
        if flops is not None and l1 is not None and l1 > 0:
            ai_l1.append(flops / l1)
    if not valid_duration:
        return None
    return {
        "achieved_bw": statistics.median(bws) if bws else None,
        "achieved_tflops": statistics.median(tfs) if tfs else None,
        "ai_hbm": statistics.median(ai_hbm) if ai_hbm else None,
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
    dur_s = dur_ns / 1e9 if dur_ns > 0 else None

    wave_size = getattr(gpu_info, "wave_size", 0) if gpu_info else 0
    xcc_count = getattr(gpu_info, "xcc_count", 0) if gpu_info else 0
    arch = getattr(gpu_info, "arch", "") if gpu_info else ""
    mfma_flops = compute_mfma_flops(c, dtype)
    mfma_families = compute_mfma_flops_by_family(c)
    mfma_fractions = ({family: operations / mfma_flops
                       for family, operations in mfma_families.items()}
                      if mfma_flops is not None and mfma_flops > 0 else {})
    duration_samples_us = tuple(
        duration_ns / 1000.0
        for _counters, duration_ns in getattr(kernel, "dispatches", ())
        if math.isfinite(duration_ns) and duration_ns > 0
    )
    valu_lane_instructions = compute_valu_lane_instructions(c, wave_size)
    # Untyped VALU counts are not FP work.
    valu_flops = None
    total_flops = mfma_flops

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

    ai_hbm = _ratio(total_flops, hbm_bytes)
    ai_l2 = _ratio(total_flops, l2_bytes)
    ai_l1 = _ratio(total_flops, l1_bytes)

    achieved_bw = _ratio(hbm_bytes, dur_s)
    achieved_tflops = _ratio(_ratio(total_flops, dur_s), 1e12)
    achieved_bw_tbps = _ratio(achieved_bw, 1e12)

    _pd = _per_dispatch_roofline(
        getattr(kernel, "dispatches", ()), dtype, wave_size, xcc_count, arch
    )
    if _pd is not None:
        ai_hbm = _pd["ai_hbm"]
        ai_l2 = _pd["ai_l2"]
        ai_l1 = _pd["ai_l1"]
        achieved_bw = _pd["achieved_bw"]
        achieved_tflops = _pd["achieved_tflops"]
        achieved_bw_tbps = _ratio(achieved_bw, 1e12)

    if roofline_peaks is not None:
        peak_bw_tbps = roofline_peaks.peak_bw_tbps.get("hbm", 0.0)
        peak_mfma_tflops = roofline_peaks.peak_compute_tflops.get("mfma", 0.0)
        peak_bw = peak_bw_tbps * 1e12
        peak_tflops = peak_mfma_tflops * 1e12
    else:
        peak_bw = 0.0
        peak_tflops = 0.0

    if dtype == UNKNOWN_DTYPE:
        peak_tflops = 0.0
    if achieved_tflops is not None and peak_tflops > 0:
        mfma_util = achieved_tflops * 1e12 / peak_tflops * 100.0
    else:
        mfma_util = UNKNOWN_UTIL_PCT
    if achieved_bw is not None and peak_bw > 0:
        bw_util = achieved_bw / peak_bw * 100.0
    else:
        bw_util = UNKNOWN_UTIL_PCT

    valu_busy_pct = c.get("VALUBusy", UNKNOWN_UTIL_PCT)
    valu_lane_utilization_pct = c.get("VALUUtilization", UNKNOWN_UTIL_PCT)
    wave_cycles = _get(c, "SQ_WAVE_CYCLES")  # per-SE sum, units of 4 cycles
    occupancy = c.get("OccupancyPercent", UNKNOWN_UTIL_PCT)

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
        # The single-GPU roofline does not apply to collectives.
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
    elif dtype == UNKNOWN_DTYPE:
        bottleneck = "UNKNOWN"
        hints = [
            "Compute dtype is unresolved. Collected MFMA counters describe selected "
            "instruction families, not necessarily all arithmetic; no datatype-specific "
            "compute utilization, bottleneck or optimization headroom is inferred.",
        ]
    elif (mfma_util >= 0 and bw_util >= 0 and mfma_flops is not None
          and mfma_flops > 0 and hbm_bytes is not None and hbm_bytes > 0):
        bottleneck = classify_bottleneck(mfma_util, bw_util)
        hints = generate_gap_hints(
            mfma_util, bw_util, occupancy, l1_hit, l2_hit,
            stall_vmem, stall_lds, stall_issue,
        )
    else:
        bottleneck = "UNKNOWN"
        if mfma_flops is None:
            hints = [
                "No floating-point MFMA-family counter was collected. Matrix "
                "FLOPs and arithmetic intensity are unavailable, not measured zeros."
            ]
        elif mfma_flops == 0:
            hints = [
                "Collected floating-point MFMA counters are zero. A matrix "
                "roofline does not classify this sample's unmeasured vector or "
                "memory work. Measured bandwidth remains independently available."
            ]
        elif hbm_bytes is None:
            hints = [
                "Complete SDK memory-traffic counters were not collected. Memory "
                "bytes, bandwidth and arithmetic intensity are unavailable."
            ]
        elif roofline_peaks is not None and peak_bw > 0 and hbm_bytes <= 0:
            hints = [
                "No positive HBM traffic was reported for this kernel sample — "
                "bandwidth utilization and roofline classification are N/A."
            ]
        else:
            hints = [
                "Empirical roofline peaks unavailable — re-run without --no-roofline "
                "to enable utilization metrics."
            ]

    if mfma_flops is not None:
        hints.append(
            "Matrix FLOPs sum only collected floating-point MFMA families and "
            "include executed tile work. Uncollected families and other arithmetic "
            "are not measured; untyped VALU lane-instruction estimates are excluded."
        )

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
        ai_hbm=ai_hbm,
        ai_l2=ai_l2,
        ai_l1=ai_l1,
        achieved_bw_tbps=achieved_bw_tbps,
        achieved_tflops=achieved_tflops,
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
        valu_lane_utilization_pct=(
            round(valu_lane_utilization_pct, 1) if valu_lane_utilization_pct >= 0 else -1.0
        ),
        valu_lane_instructions=valu_lane_instructions,
        mfma_counter_families=tuple(name for name in MFMA_FLOAT_COUNTERS if name in c),
        hbm_bytes_source=memory_traffic_source(c),
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
        median_duration_us=(statistics.median(duration_samples_us)
                            if duration_samples_us else None),
        duration_samples_us=duration_samples_us,
        mfma_flops_by_family=mfma_families,
        mfma_fraction_by_family=mfma_fractions,
        hbm_bytes_by_kind=compute_hbm_bytes_by_kind(c),
    )
