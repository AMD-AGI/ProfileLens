# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared measurement preparation for workload replay and external capture."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from rich.console import Console

from profilelens.dtypes import RooflinePeaks
from profilelens.profile.counter_sets import CounterSet
from profilelens.profile.dtype_detect import (
    UNKNOWN_DTYPE,
    counter_dtype_coverage,
    normalize_counter_dtype_hints,
)
from profilelens.profile.gpu_detect import GpuInfo


@dataclass(frozen=True)
class CounterPlan:
    counter_set: CounterSet
    dtype_hints: tuple[str, ...]
    dtype_coverage: tuple[str, ...]
    reference_dtypes: tuple[str, ...] = ()


def prepare_counter_plan(
    rocprofv3_path: str,
    *,
    observed_dtypes: Iterable[str] = (),
    has_unresolved: bool = True,
    counter_dtype_hints: Iterable[str] = (),
    custom_counters: Iterable[str] | None = None,
    counter_set_mode: str = "auto",
    device_index: int = 0,
    log: Callable[[str], None] | None = None,
) -> CounterPlan:
    """Select and validate one single-pass counter set; hints never assign dtypes."""
    from profilelens.profile.counter_sets import build_custom, select_counter_set
    from profilelens.profile.rocprofv3 import counter_set_supported, list_available_counters

    if counter_set_mode not in {"auto", "custom"}:
        raise ValueError("counter_set_mode must be auto or custom")
    hints = tuple(normalize_counter_dtype_hints(list(counter_dtype_hints)))
    names: tuple[str, ...] | None = None
    if custom_counters is not None:
        if isinstance(custom_counters, str):
            raise ValueError("custom_counters must be a sequence of counter names")
        values = list(custom_counters)
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("The custom counter set requires nonempty counter names")
        names = tuple(dict.fromkeys(v.strip() for v in values))
        counter_set_mode = "custom"
    elif counter_set_mode == "custom":
        raise ValueError("The custom counter set requires explicit counters")

    coverage = tuple(counter_dtype_coverage(list(observed_dtypes), has_unresolved, list(hints)))
    available = list_available_counters(rocprofv3_path, device_index=device_index)

    def compatibility(counters: tuple[str, ...]) -> tuple[bool, str]:
        return counter_set_supported(rocprofv3_path, counters, device_index=device_index)

    if names is not None:
        missing = sorted(set(names) - available)
        if missing:
            raise RuntimeError(
                "custom counters are not reported by this GPU: " + ", ".join(missing)
            )
        cset = build_custom(list(names))
        supported, detail = compatibility(cset.counters)
        if not supported:
            raise RuntimeError("custom counter set cannot be collected in one pass: " + detail)
        return CounterPlan(cset, hints, (), coverage)

    cset = select_counter_set(
        None, list(coverage), available_counters=available, compatibility=compatibility, log=log,
    )
    return CounterPlan(cset, hints, coverage, coverage)


@dataclass(frozen=True)
class PreparedRoofline:
    csv_path: str
    peaks: RooflinePeaks
    peaks_by_dtype: dict[str, RooflinePeaks]


def prepare_roofline(
    gpu: GpuInfo,
    *,
    counter_plan: CounterPlan,
    observed_dtypes: Iterable[str] = (),
    summary_dtype: str = UNKNOWN_DTYPE,
    cache_dir: str | None = None,
    refresh: bool = False,
    device_index: int = 0,
    console: Console | None = None,
) -> PreparedRoofline:
    """Use standard benchmark references without attributing hinted dtypes."""
    from profilelens.config import load_config
    from profilelens.profile.roofline import (
        build_roofline_peaks,
        build_roofline_peaks_multi,
        parse_roofline_csv,
    )
    from profilelens.profile.roofline_bench import run_roofline_benchmark

    result_path = run_roofline_benchmark(
        cache_dir or load_config().profiling.roofline_cache_dir,
        gpu.model, console=console, force=refresh, device_index=device_index,
    )
    peaks = build_roofline_peaks(
        parse_roofline_csv(result_path, summary_dtype), gpu.model, summary_dtype,
    )
    reference_dtypes = list(dict.fromkeys([
        *(dt for dt in observed_dtypes if dt != UNKNOWN_DTYPE),
        *counter_plan.dtype_hints,
        *(counter_plan.reference_dtypes or counter_plan.dtype_coverage),
    ]))
    return PreparedRoofline(
        str(result_path), peaks,
        build_roofline_peaks_multi(result_path, gpu.model, reference_dtypes),
    )
