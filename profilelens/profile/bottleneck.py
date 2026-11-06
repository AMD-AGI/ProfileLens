# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bottleneck classification, headroom analysis, and optimization priority."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeadroomInfo:
    """Optimization headroom for a kernel."""

    pct_of_peak: float
    headroom: float
    e2e_pct: float
    priority_score: float


def compute_headroom(
    mfma_util: float,
    bw_util: float,
    bottleneck: str,
    e2e_pct: float,
) -> HeadroomInfo:
    """Compute optimization headroom and priority score."""
    if bottleneck == "UNKNOWN" or (mfma_util < 0 and bw_util < 0):
        return HeadroomInfo(
            pct_of_peak=-1.0,
            headroom=-1.0,
            e2e_pct=round(e2e_pct, 1),
            priority_score=-1.0,
        )

    if bottleneck == "COMPUTE":
        pct_peak = mfma_util
    elif bottleneck == "MEMORY":
        pct_peak = bw_util
    else:
        pct_peak = max(mfma_util, bw_util)

    pct_peak = min(pct_peak, 100.0)
    headroom = 1.0 - (pct_peak / 100.0)
    e2e_weight = e2e_pct / 100.0 if e2e_pct > 0 else 0.0
    priority = headroom * e2e_weight

    return HeadroomInfo(
        pct_of_peak=round(pct_peak, 1),
        headroom=round(headroom, 3),
        e2e_pct=round(e2e_pct, 1),
        priority_score=round(priority, 4),
    )


def classify_bottleneck(mfma_util: float, bw_util: float) -> str:
    """Classify kernel bottleneck from utilization percentages."""
    if mfma_util < 0 or bw_util < 0:
        return "UNKNOWN"
    if bw_util > 1.5 * mfma_util and bw_util > 30:
        return "MEMORY"
    if mfma_util > 1.5 * bw_util and mfma_util > 30:
        return "COMPUTE"
    if mfma_util < 30 and bw_util < 30:
        return "LATENCY"
    return "BALANCED"


def generate_gap_hints(
    mfma_util: float,
    bw_util: float,
    occupancy: float,
    l1_hit: float,
    l2_hit: float,
    stall_vmem: float,
    stall_lds: float,
    stall_issue: float = -1.0,
) -> list[str]:
    """Generate deterministic optimization hints from derived metrics."""
    hints: list[str] = []

    if mfma_util < 30 and bw_util < 30:
        hints.append(
            f"Both MFMA ({mfma_util:.0f}%) and BW ({bw_util:.0f}%) utilization "
            f"below 30% -> LATENCY BOUND"
        )

    if occupancy >= 0 and occupancy < 50:
        hints.append(
            f"Low occupancy {occupancy:.0f}% -> check LDS/VGPR resource usage"
        )

    if l1_hit >= 0 and l1_hit < 50:
        hints.append(
            f"L1 hit rate {l1_hit:.0f}% -> poor spatial/temporal locality"
        )

    if l2_hit >= 0 and l2_hit < 50:
        hints.append(
            f"L2 hit rate {l2_hit:.0f}% -> data doesn't fit in L2, HBM-bound"
        )

    if mfma_util > 60 and bw_util < 20:
        hints.append(
            "High MFMA but low BW -> compute bound, well-optimized for memory"
        )

    if bw_util > 60 and mfma_util < 20:
        hints.append(
            "High BW but low MFMA -> memory bound, consider fusion or recomputation"
        )

    if stall_vmem > 30:
        hints.append(
            f"VMEM stalls {stall_vmem:.0f}% -> vector memory latency dominating"
        )

    if stall_lds > 30:
        hints.append(
            f"LDS stalls {stall_lds:.0f}% -> LDS bank conflicts or capacity"
        )

    if stall_vmem < 0 and stall_issue > 30:
        hints.append(
            f"Instruction-issue wait {stall_issue:.0f}% -> inspect memory, "
            "dependency, and execution-pipeline latency; VMEM-specific data is unavailable"
        )

    return hints


def suggest_experiments(bottleneck: str, occupancy: float, stall_vmem: float) -> list[str]:
    """Suggest optimization experiments based on classification."""
    exps: list[str] = []
    if bottleneck == "MEMORY":
        exps.extend(["increase_arithmetic_intensity", "fuse_kernels", "prefetch_L2"])
    elif bottleneck == "COMPUTE":
        exps.extend(["reduce_precision", "algorithmic_optimization"])
    elif bottleneck == "LATENCY":
        if 0 <= occupancy < 50:
            exps.extend(["reduce_tile_size", "reduce_LDS_per_block", "increase_waves"])
        if stall_vmem > 30:
            exps.extend(["prefetch_global", "software_pipelining"])
        if not exps:
            exps.extend(["reduce_synchronization", "overlap_compute_memory"])
    return exps
