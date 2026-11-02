# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build single-pass counter plans from runtime ROCprofiler-SDK capabilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from profilelens.profile.dtype_detect import mfma_counters_for_dtypes


CompatibilityCheck = Callable[[tuple[str, ...]], tuple[bool, str]]


@dataclass(frozen=True)
class CounterSet:
    name: str
    counters: tuple[str, ...]
    has_stall_breakdown: bool
    has_lds_detail: bool


@dataclass(frozen=True)
class CounterGroup:
    """An all-or-nothing metric signal, ordered by profiling value."""

    name: str
    counters: tuple[str, ...]
    priority: int
    required: bool = False
    signal: str = ""
    quality: str = "exact"


# These groups describe metric intent, not product configuration. Whether each
# counter exists and whether groups fit together is decided by rocprofv3-avail
# on the actual GPU.
_BASE_GROUPS = (
    CounterGroup(
        "hbm_traffic",
        ("BANDWIDTH_EA",),
        priority=100,
        required=True,
    ),
    CounterGroup(
        "runtime_cycles",
        ("GRBM_COUNT", "GRBM_GUI_ACTIVE"),
        priority=100,
        required=True,
    ),
    CounterGroup(
        "valu_flops",
        ("SQ_INSTS_VALU",),
        priority=90,
        required=True,
    ),
    CounterGroup(
        "valu_utilization",
        ("SQ_ACTIVE_INST_VALU",),
        priority=80,
    ),
    CounterGroup(
        "occupancy",
        ("SQ_WAVE_CYCLES",),
        priority=80,
    ),
)



_DIAGNOSTIC_GROUPS = (
    CounterGroup(
        "l1_hit_rate_exact",
        (
            "TCP_TOTAL_CACHE_ACCESSES_sum",
            "TCP_TCC_READ_REQ_sum",
            "TCP_TCC_WRITE_REQ_sum",
            "TCP_TCC_ATOMIC_WITH_RET_REQ_sum",
            "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum",
        ),
        priority=65,
        signal="l1_hit_rate",
    ),
    CounterGroup(
        "l2_hit_rate_sdk",
        ("L2CacheHit",),
        priority=65,
        signal="l2_hit_rate",
    ),
    CounterGroup(
        "l2_hit_rate_raw",
        ("TCC_HIT_sum", "TCC_MISS_sum"),
        priority=64,
        signal="l2_hit_rate",
    ),
    CounterGroup(
        "l1_hit_rate_estimated",
        (
            "TCP_TOTAL_CACHE_ACCESSES_sum",
            "TCP_TCC_READ_REQ_sum",
            "TCP_TCC_WRITE_REQ_sum",
            "TCP_TCC_ATOMIC_WITH_RET_REQ_sum",
        ),
        priority=60,
        signal="l1_hit_rate",
        quality="estimated",
    ),
    CounterGroup(
        "l2_hit_rate_estimated",
        (
            "TCP_TCC_READ_REQ_sum",
            "TCP_TCC_WRITE_REQ_sum",
            "TCP_TCC_ATOMIC_WITH_RET_REQ_sum",
            "TCC_EA0_RDREQ_sum",
            "TCC_EA0_WRREQ_sum",
        ),
        priority=60,
        signal="l2_hit_rate",
        quality="estimated",
    ),
    CounterGroup(
        "issue_stalls_vmem_lds",
        ("SQ_WAIT_INST_VMEM", "SQ_WAIT_INST_LDS"),
        priority=55,
        signal="issue_stalls",
    ),
    CounterGroup(
        "issue_stalls_total_lds",
        ("SQ_WAIT_INST_ANY", "SQ_WAIT_INST_LDS"),
        priority=50,
        signal="issue_stalls",
        quality="partial",
    ),
)


def _candidate_groups(
    dtype: str | None,
    observed_dtypes: set[str] | tuple[str, ...] | list[str] | None,
) -> tuple[CounterGroup, ...]:
    requested_dtypes: Iterable[str]
    if observed_dtypes:
        requested_dtypes = observed_dtypes
    elif dtype:
        requested_dtypes = (dtype,)
    else:
        # Preserve the former unknown-dtype coverage. Without these signals a
        # direct ``profile --kernel-names`` run could report zero MFMA work
        # simply because no trace was supplied to identify the precise dtype.
        requested_dtypes = ("BF16", "FP16", "FP8")

    mfma_groups = tuple(
        CounterGroup(
            name=f"mfma_{counter.rsplit('_', 1)[-1].lower()}",
            counters=(counter,),
            priority=95,
            required=True,
        )
        for counter in mfma_counters_for_dtypes(list(requested_dtypes))
    )
    return tuple(sorted(
        (*_BASE_GROUPS, *mfma_groups, *_DIAGNOSTIC_GROUPS),
        key=lambda group: -group.priority,
    ))


def _flatten_groups(groups: Iterable[CounterGroup]) -> tuple[str, ...]:
    counters: dict[str, None] = {}
    for group in groups:
        for counter in group.counters:
            counters[counter] = None
    return tuple(counters)


def _select_compatible_groups(
    candidates: tuple[CounterGroup, ...],
    available_counters: frozenset[str],
    compatibility: CompatibilityCheck,
    log=None,
) -> tuple[CounterGroup, ...]:
    """Choose one runtime-supported candidate per metric signal."""
    selected: list[CounterGroup] = []
    selected_signals: set[str] = set()

    for group in sorted(candidates, key=lambda candidate: -candidate.priority):
        if group.signal and group.signal in selected_signals:
            continue

        missing = sorted(set(group.counters) - available_counters)
        if missing:
            detail = ", ".join(missing)
            message = (
                f"counter group '{group.name}' is unavailable; "
                f"missing counters: {detail}"
            )
            if group.required:
                raise RuntimeError(f"required {message}")
            if log:
                log(f"skipping optional {message}")
            continue

        proposed_counters = _flatten_groups((*selected, group))
        supported, detail = compatibility(proposed_counters)
        if not supported:
            reason = detail or "rocprofv3-avail reported an incompatible set"
            message = (
                f"counter group '{group.name}' cannot be collected with the "
                f"selected groups: {reason}"
            )
            if group.required:
                raise RuntimeError(f"required {message}")
            if log:
                log(f"skipping optional {message}")
            continue

        selected.append(group)
        if group.signal:
            selected_signals.add(group.signal)

    return tuple(selected)


def build_custom(counter_list: list[str]) -> CounterSet:
    """Build a counter set from user-provided counter names."""
    return CounterSet(
        name="custom",
        counters=tuple(counter_list),
        has_stall_breakdown=any("WAIT_INST" in counter for counter in counter_list),
        has_lds_detail="SQ_WAIT_INST_LDS" in counter_list,
    )


def select_counter_set(
    dtype: str | None,
    observed_dtypes: set[str] | tuple[str, ...] | list[str] | None = None,
    *,
    available_counters: frozenset[str],
    compatibility: CompatibilityCheck,
    log=None,
) -> CounterSet:
    """Build the highest-value one-pass set supported by the runtime GPU."""
    candidates = _candidate_groups(dtype, observed_dtypes)
    selected = _select_compatible_groups(
        candidates,
        available_counters,
        compatibility,
        log=log,
    )
    counters = _flatten_groups(selected)
    labels = "_".join(group.name for group in selected)
    return CounterSet(
        name=f"runtime_{labels}" if labels else "runtime_empty",
        counters=counters,
        has_stall_breakdown=any("WAIT_INST" in counter for counter in counters),
        has_lds_detail="SQ_WAIT_INST_LDS" in counters,
    )
