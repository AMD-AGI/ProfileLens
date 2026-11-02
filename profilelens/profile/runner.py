# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Orchestrator for the profiling pipeline."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from profilelens.dtypes import KernelProfile, ProfilingResult, QueryResult


def _build_trace_context_map(
    analysis_results: list[QueryResult],
) -> dict[str, dict[str, Any]]:
    """Build kernel_name -> {e2e_pct, trace_total_ms, ...} from analysis."""
    ctx: dict[str, dict[str, Any]] = {}
    for r in analysis_results:
        if r.label == "Aggregate Time per Kernel Name" and r.rows:
            total_ms = sum(row.get("total_ms", 0) for row in r.rows)
            for row in r.rows:
                name = row.get("name", "")
                if name and total_ms > 0:
                    ctx[name] = {
                        "e2e_pct": round(row.get("total_ms", 0) / total_ms * 100, 1),
                        "trace_total_ms": row.get("total_ms", 0),
                        "trace_count": row.get("count", 0),
                    }
            break
    return ctx


def extract_top_k_kernels(
    analysis_json_path: str,
    top_k: int = 10,
) -> list[str]:
    """Extract top-K kernel names from a saved analyze JSON output."""
    with open(analysis_json_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        return []

    for entry in data:
        if entry.get("label") == "Aggregate Time per Kernel Name":
            rows = entry.get("rows", [])
            return [r["name"] for r in rows[:top_k] if r.get("name")]

    return []


def extract_top_k_from_results(
    results: list[QueryResult],
    top_k: int = 10,
) -> list[str]:
    """Extract top-K kernel names from in-memory analysis results."""
    for r in results:
        if r.label == "Aggregate Time per Kernel Name" and r.rows:
            return [row["name"] for row in r.rows[:top_k] if row.get("name")]
    return []


def _merge_aggregate_rows(per_rank_rows: list[tuple[Any, ...]]) -> tuple[dict, ...]:
    """Merge 'Aggregate Time per Kernel Name' rows across ranks."""
    from collections import defaultdict
    tot: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    for rows in per_rank_rows:
        for row in (rows or ()):
            name = row.get("name", "")
            if not name:
                continue
            tot[name] += row.get("total_ms", 0) or 0
            cnt[name] += row.get("count", 0) or 0
    merged = [
        {
            "name": name,
            "total_ms": round(tot[name], 4),
            "count": cnt[name],
            "avg_ms": round(tot[name] / cnt[name], 6) if cnt[name] else 0.0,
        }
        for name in tot
    ]
    merged.sort(key=lambda r: r["total_ms"], reverse=True)
    return tuple(merged)


def merge_rank_analyses(
    primary: list[QueryResult],
    extra_per_rank: list[tuple[int, list[QueryResult]]],
) -> tuple[list[QueryResult], dict[str, Any]]:
    """Fold per-rank aggregate/dispatch results into the primary analysis."""
    from profilelens.dtypes import QueryResult

    AGG = "Aggregate Time per Kernel Name"
    DISP = "Kernel Dispatch Durations"

    def _find(results, label):
        for r in results:
            if r.label == label:
                return r
        return None

    prim_agg = _find(primary, AGG)
    rank_agg_rows: list[tuple[Any, ...]] = []
    per_rank_topk: dict[int, list[str]] = {}
    if prim_agg and prim_agg.rows:
        rank_agg_rows.append(prim_agg.rows)
        per_rank_topk[0] = [r.get("name", "") for r in prim_agg.rows[:20] if r.get("name")]

    for rank, results in extra_per_rank:
        ragg = _find(results, AGG)
        if ragg and ragg.rows:
            rank_agg_rows.append(ragg.rows)
            per_rank_topk[rank] = [r.get("name", "") for r in ragg.rows[:20] if r.get("name")]

    keep_names: set[str] = set()
    for names in per_rank_topk.values():
        keep_names.update(names)

    def _filt(rows):
        return [r for r in (rows or ()) if r.get("name", "") in keep_names]

    disp_rows_all: list[dict] = []
    prim_disp = _find(primary, DISP)
    if prim_disp and prim_disp.rows:
        disp_rows_all.extend(_filt(prim_disp.rows))
    for rank, results in extra_per_rank:
        rdisp = _find(results, DISP)
        if rdisp and rdisp.rows:
            disp_rows_all.extend(_filt(rdisp.rows))

    merged_results = list(primary)

    if len(rank_agg_rows) > 1:
        merged_rows = _merge_aggregate_rows(rank_agg_rows)
        merged_agg = QueryResult(
            sql="(merged across ranks)", label=AGG,
            rows=merged_rows,
            columns=("name", "total_ms", "count", "avg_ms"),
        )
        merged_results = [merged_agg if r.label == AGG else r for r in merged_results]
        if not any(r.label == AGG for r in merged_results):
            merged_results.append(merged_agg)

    # Replace the (rank-0, all-kernel) dispatch durations with the cross-rank,
    # top-K-filtered set. 
    if prim_disp is not None and extra_per_rank and disp_rows_all:
        merged_disp = QueryResult(
            sql="(merged across ranks, top-K only)", label=DISP,
            rows=tuple(disp_rows_all),
            columns=prim_disp.columns,
        )
        merged_results = [merged_disp if r.label == DISP else r for r in merged_results]

    # EP-asymmetry: do ranks run different kernel sets?
    topk_sets = {rk: set(names) for rk, names in per_rank_topk.items()}
    asymmetric = False
    if len(topk_sets) > 1:
        ref = next(iter(topk_sets.values()))
        asymmetric = any(s != ref for s in topk_sets.values())

    breakdown = {
        "n_ranks": len(per_rank_topk),
        "ranks": sorted(per_rank_topk),
        "ep_asymmetric": asymmetric,
        "per_rank_topk": {str(k): v for k, v in sorted(per_rank_topk.items())},
    }
    return merged_results, breakdown


import re as _re

# Per-rank kineto trace pattern filenames look like:
#   dp0_pp0_tp3_dcp0_ep3_rank3.<traceid>.pt.trace.json.gz
# Capture the logical rank and (best effort) the tp/ep coordinates.
_RANK_RE = _re.compile(r"(?:^|[._])rank(\d+)(?:[._]|$)")
_TP_RE = _re.compile(r"(?:^|[._])tp(\d+)(?:[._]|$)")
_EP_RE = _re.compile(r"(?:^|[._])ep(\d+)(?:[._]|$)")


@dataclass
class RankTrace:
    """A per-rank kineto trace and its parsed parallelism coordinates."""


    path: Path
    rank: int | None = None
    tp: int | None = None
    ep: int | None = None
    trace_rank: int | None = None
    world_size: int | None = None
    pid: int | None = None


def parse_rank_meta(path: Path) -> RankTrace:
    """Parse rank/tp/ep coordinates from a trace filename (best effort)."""
    name = path.name
    def _grab(rx):
        m = rx.search(name)
        return int(m.group(1)) if m else None
    return RankTrace(path=path, rank=_grab(_RANK_RE), tp=_grab(_TP_RE), ep=_grab(_EP_RE))


_DIST_RANK_RE = _re.compile(r'"rank"\s*:\s*(\d+)')
_DIST_WORLD_RE = _re.compile(r'"world_size"\s*:\s*(\d+)')
_PID_RE = _re.compile(r'"pid"\s*:\s*(\d+)')


def parse_trace_identity(path: Path, peek_bytes: int = 4 * 1024 * 1024) -> tuple[int | None, int | None, int | None]:
    """Read (rank, world_size, worker_pid) from a kineto trace's metadata."""

    import gzip as _gzip
    from collections import Counter

    try:
        if path.name.endswith(".gz"):
            with _gzip.open(path, "rb") as f:
                head = f.read(peek_bytes)
        else:
            with open(path, "rb") as f:
                head = f.read(peek_bytes)
    except (OSError, _gzip.BadGzipFile, EOFError):
        return (None, None, None)
    text = head.decode("utf-8", "replace")

    rank = world = pid = None
    di = text.find('"distributedInfo"')
    if di != -1:
        seg = text[di:di + 400]  # first-level fields precede the nested pg_config
        m = _DIST_RANK_RE.search(seg)
        if m:
            rank = int(m.group(1))
        m = _DIST_WORLD_RE.search(seg)
        if m:
            world = int(m.group(1))

    pids = _PID_RE.findall(text)
    if pids:
        # The worker process dominates the CPU-side events; GPU device/stream
        # tracks are comparatively rare in the head.
        pid = int(Counter(pids).most_common(1)[0][0])

    return (rank, world, pid)


def enrich_rank_identities(rank_traces: list[RankTrace]) -> list[RankTrace]:
    """Fill trace_rank/world_size/pid from each trace's metadata in place."""

    for rt in rank_traces:
        trank, world, pid = parse_trace_identity(rt.path)
        rt.trace_rank = trank
        rt.world_size = world
        if rt.pid is None:
            rt.pid = pid
        if trank is not None:
            rt.rank = trank
    return rank_traces


def collect_recorded_traces(audit_dir: str | Path) -> list[RankTrace]:
    """Build the per-rank trace list from paths the worker shim recorded."""
    audit_dir = Path(audit_dir)
    if not audit_dir.is_dir():
        return []
    best: dict[tuple[str, object], tuple[RankTrace, int]] = {}
    for f in audit_dir.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        pid = data.get("pid")
        for raw in data.get("trace_paths", []) or []:
            p = Path(raw)
            if not p.is_file():
                continue
            rt = parse_rank_meta(p)  # rank/tp/ep from filename (best effort)
            rt.pid = pid
            key: tuple[str, object] = (
                ("rank", rt.rank) if rt.rank is not None else ("pid", pid)
            )
            size = p.stat().st_size
            cur = best.get(key)
            if cur is None or size > cur[1]:
                best[key] = (rt, size)
    return [rt for rt, _ in best.values()]


def collect_region_paths(audit_dir: str | Path) -> dict[str, str]:
    """Merge the shim's per-PID """
    audit_dir = Path(audit_dir)
    merged: dict[str, str] = {}
    if not audit_dir.is_dir():
        return merged
    for f in audit_dir.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        for region, path in (data.get("region_paths") or {}).items():
            merged.setdefault(region, path)
    return merged


@dataclass
class KernelCV:
    """Dispatch variance info for a kernel."""

    name: str
    count: int
    cv_pct: float
    mean_us: float
    min_us: float
    max_us: float
    high_cv: bool
    is_collective: bool = False


CV_THRESHOLD = 20.0


def _representative_ranges(
    counts: list[int],
    *,
    fractions: tuple[float, ...] = (0.12, 0.5, 0.88),
    width: int = 2,
    merge_rel: float = 0.18,
    max_ranges: int = 14,
) -> list[tuple[int, int]]:
    """Count-aware multi-range iteration selection for per-kernel coverage.

    Each window kernel contributes its own early/mid/late absolute
    indices.  The first window also covers names split across kernel IDs.
    """
    counts = [c for c in counts if c and c > 2 * width]
    if not counts:
        return []
    pts: set[int] = {1}
    for c in counts:
        for f in fractions:
            pts.add(max(1, min(c - width, int(round(f * c)))))
    ordered = sorted(pts)
    kept: list[int] = []
    for p in ordered:
        if kept and (p - kept[-1]) <= merge_rel * max(p, 1):
            continue
        kept.append(p)
    # overhead cap: drop the smallest gaps (most redundant), never the sparse
    # high end, so lifetime span is preserved.
    while len(kept) > max_ranges:
        gaps = [(kept[i + 1] - kept[i], i) for i in range(1, len(kept) - 1)]
        if not gaps:
            kept = kept[:max_ranges]
            break
        kept.pop(min(gaps)[1])
    return [(p, p + width - 1) for p in kept]


def _compute_dispatch_sample(
    analysis_results: list[QueryResult],
    kernel_names: list[str],
    sample_n: int,
    console: Console,
) -> tuple[list[int] | None, dict[str, KernelCV]]:
    """Analyze dispatch counts and variance, compute iteration IDs to sample."""
    import math
    from profilelens.profile.metrics import is_collective_kernel

    agg = None
    dispatch_dur = None
    for r in analysis_results:
        if r.label == "Aggregate Time per Kernel Name" and r.rows:
            agg = r
        if r.label == "Kernel Dispatch Durations" and r.rows:
            dispatch_dur = r

    kernel_cv_map: dict[str, KernelCV] = {}

    if not agg or not agg.rows:
        return None, kernel_cv_map

    name_to_count: dict[str, int] = {}
    for row in agg.rows:
        name = row.get("name", "")
        if name:
            name_to_count[name] = row.get("count", 0)

    if dispatch_dur and dispatch_dur.rows:
        from collections import defaultdict
        dur_by_name: dict[str, list[float]] = defaultdict(list)
        name_set = set(kernel_names)
        for row in dispatch_dur.rows:
            name = row.get("name", "")
            if name in name_set:
                dur_by_name[name].append(row.get("dur", 0) / 1000.0)

        for name in kernel_names:
            is_coll = is_collective_kernel(name)
            durs = dur_by_name.get(name, [])
            if len(durs) < 2:
                kernel_cv_map[name] = KernelCV(
                    name=name, count=len(durs), cv_pct=0.0,
                    mean_us=durs[0] if durs else 0, min_us=durs[0] if durs else 0,
                    max_us=durs[0] if durs else 0, high_cv=False,
                    is_collective=is_coll,
                )
                continue
            mean = sum(durs) / len(durs)
            variance = sum((d - mean) ** 2 for d in durs) / (len(durs) - 1)
            std = math.sqrt(variance)
            cv = (std / mean * 100) if mean > 0 else 0

            kernel_cv_map[name] = KernelCV(
                name=name, count=len(durs), cv_pct=round(cv, 1),
                mean_us=round(mean, 2), min_us=round(min(durs), 2),
                max_us=round(max(durs), 2),
                high_cv=(cv >= CV_THRESHOLD and not is_coll),
                is_collective=is_coll,
            )

    window_names = [n for n in kernel_names if not is_collective_kernel(n)]
    if not window_names:
        window_names = kernel_names
    n_collectives = len(kernel_names) - len(window_names)

    min_dispatches = float("inf")
    total_dispatches = 0
    for name in window_names:
        count = name_to_count.get(name, 0)
        if count > 0:
            min_dispatches = min(min_dispatches, count)
            total_dispatches += count

    if min_dispatches <= sample_n * 2:
        console.print(f"  [dim]Dispatch sampling skipped: smallest kernel has only "
                       f"{min_dispatches} dispatches[/dim]")
        return None, kernel_cv_map

    window_counts = [name_to_count.get(n, 0) for n in window_names]
    ranges = _representative_ranges(window_counts, width=max(2, sample_n // 5))
    if not ranges:
        return None, kernel_cv_map
    # Expand to a flat sorted iteration-id list (build_command regroups
    # contiguous runs back into rocprofv3 [lo-hi] tokens).
    iteration_ids = sorted({i for lo, hi in ranges for i in range(lo, hi + 1)})
    max_cv = max(
        (cv.cv_pct for cv in kernel_cv_map.values() if cv.cv_pct >= 0),
        default=0.0,
    )
    per_kernel = len(iteration_ids)
    console.print(
        f"  Dispatch sampling: per-kernel representative coverage — "
        f"{len(ranges)} ranges "
        f"{' '.join(f'[{lo}-{hi}]' for lo, hi in ranges)} "
        f"(~{per_kernel} samples/kernel, max kernel CV {max_cv:.1f}%)"
    )
    has_high_cv = False
    has_collective = False
    for name in kernel_names:
        count = name_to_count.get(name, 0)
        cv_info = kernel_cv_map.get(name)
        short = name[:45]
        cv_str = f"CV={cv_info.cv_pct:.1f}%" if cv_info else ""
        marker = ""
        if cv_info and cv_info.is_collective:
            marker = " [cyan](collective)[/cyan]"
            has_collective = True
        elif cv_info and cv_info.high_cv:
            marker = " [bold red]*[/bold red]"
            has_high_cv = True
        # samples actually landing in this kernel = ranges whose lo <= its count
        hit = sum(1 for i in iteration_ids if i <= count)
        console.print(f"    {short}: {count} total, {cv_str}{marker} -> ~{hit} samples")
    console.print(f"  Expected counter rows: ~{per_kernel * len(kernel_names) * 20} "
                   f"(vs ~{total_dispatches * 20} without sampling)")
    if has_high_cv:
        console.print(
            "  [yellow]* = high dispatch variance (CV > 20%). Input sizes may vary across dispatches.[/yellow]\n"
            "  [yellow]  For those kernels, consider: --sample-dispatches 0[/yellow]"
        )
    if has_collective:
        console.print(
            f"  [cyan]{n_collectives} collective/comm kernel(s) in top-K: their "
            "CV is sync-driven, not input-size-driven.[/cyan]\n"
            "  [cyan]  They do not drive the sample window and are reported "
            "separately (XGMI-bound, off-roofline).[/cyan]\n"
            "  [cyan]  For trustworthy collective counters use --sample-dispatches 0 "
            "(profile all, not a mid-window slice).[/cyan]"
        )

    return iteration_ids, kernel_cv_map


def _write_rank_breakdown(profiled, profile_output: str, console: Console) -> None:
    """Persist per-(kernel, rank) detail and print a multi-rank summary."""
    max_ranks = max((getattr(a, "n_ranks", 1) for a in profiled), default=1)
    if max_ranks <= 1:
        return  # single-rank (TP=1) — nothing extra to report.

    agents: set[str] = set()
    breakdown: dict[str, Any] = {}
    for a in profiled:
        ranks = getattr(a, "per_rank", ())
        for r in ranks:
            if r.agent_id:
                agents.add(r.agent_id)
        breakdown[a.kernel_name] = {
            "n_ranks": getattr(a, "n_ranks", 1),
            "representative_avg_us": round(a.avg_duration_ns / 1000.0, 3),
            "representative_total_us": round(a.total_duration_ns / 1000.0, 3),
            "per_rank": [
                {
                    "rank_key": r.rank_key,
                    "agent_id": r.agent_id,
                    "dispatch_count": r.dispatch_count,
                    "avg_us": round(r.avg_duration_ns / 1000.0, 3),
                    "total_us": round(r.total_duration_ns / 1000.0, 3),
                }
                for r in ranks
            ],
        }

    out = Path(profile_output) / "phase2_rank_breakdown.json"
    try:
        out.write_text(json.dumps(
            {"n_ranks": max_ranks, "n_gpu_agents": len(agents),
             "agents": sorted(agents), "kernels": breakdown},
            indent=2,
        ))
    except OSError:
        pass

    # Flag kernels whose per-rank duration is imbalanced (max/min >= 2x).
    imbalanced = []
    for name, info in breakdown.items():
        durs = [r["avg_us"] for r in info["per_rank"] if r["avg_us"] > 0]
        if len(durs) >= 2 and min(durs) > 0 and max(durs) / min(durs) >= 2.0:
            imbalanced.append((name, min(durs), max(durs)))

    console.print(
        f"  Multi-rank: aggregated {max_ranks} ranks across {len(agents)} GPU "
        f"agents -> per-rank representative (mean) shown; full breakdown in "
        f"{out.name}"
    )
    for name, lo, hi in imbalanced[:5]:
        console.print(
            f"    [yellow]rank imbalance[/yellow] {name[:45]}: "
            f"{lo:.1f}-{hi:.1f} us/dispatch across ranks"
        )


def run_profile(
    kernel_names: list[str],
    user_command: list[str],
    dtype: str | None = None,
    counter_set_mode: str = "auto",
    custom_counters: list[str] | None = None,
    run_roofline: bool = True,
    refresh_roofline: bool = False,
    output_dir: str = "./profilelens_run",
    top_k: int = 10,
    sample_dispatches: int = 10,
    analysis_results: list[QueryResult] | None = None,
    console: Console | None = None,
    profiling_config: dict[str, Any] | None = None,
    per_kernel_dtype: dict[str, tuple[str, str]] | None = None,
    observed_dtypes: list[str] | None = None,
) -> ProfilingResult:
    """Execute the full profiling pipeline."""
    from profilelens.profile.gpu_detect import detect_gpu
    from profilelens.profile.counter_sets import (
        build_custom,
        select_counter_set,
    )
    from profilelens.profile.rocprofv3 import (
        build_kernel_regex,
        counter_set_supported,
        find_rocprofv3,
        list_available_counters,
        run_profile as run_rocprofv3,
    )
    from profilelens.profile.roofline_bench import run_roofline_benchmark
    from profilelens.profile.parser import parse_and_aggregate
    from profilelens.profile.metrics import compute_kernel_metrics
    from profilelens.profile.roofline import (
        build_roofline_peaks,
        build_roofline_peaks_multi,
        parse_roofline_csv,
    )

    cfg = profiling_config or {}
    con = console or Console()
    rocprofv3_path = find_rocprofv3(cfg.get("rocprofv3_path", ""))

    con.print("[bold]Step 1:[/bold] Detecting GPU...")
    gpu = detect_gpu(rocprofv3_path=rocprofv3_path)
    con.print(f"  Detected: {gpu.model} ({gpu.arch}), {gpu.cu_count} CUs")

    con.print("[bold]Step 2:[/bold] Selecting counter set...")
    available_counters = list_available_counters(rocprofv3_path)

    def _compatibility(counters: tuple[str, ...]) -> tuple[bool, str]:
        return counter_set_supported(rocprofv3_path, counters)

    if counter_set_mode == "custom" and custom_counters:
        missing = sorted(set(custom_counters) - available_counters)
        if missing:
            raise RuntimeError(
                "custom counters are not reported by this GPU: " + ", ".join(missing)
            )
        cset = build_custom(custom_counters)
        supported, detail = _compatibility(cset.counters)
        if not supported:
            raise RuntimeError(
                "custom counter set cannot be collected in one pass: " + detail
            )
    else:
        cset = select_counter_set(
            dtype, observed_dtypes,
            available_counters=available_counters,
            compatibility=_compatibility,
            log=lambda m: con.print(f"  [yellow]{m}[/yellow]"),
        )
    if observed_dtypes:
        con.print(f"  Observed dtypes: {', '.join(observed_dtypes)}")
    con.print(f"  Counter set: {cset.name} ({len(cset.counters)} counters)")

    roofline_csv_path = None
    roofline_peaks = None
    if run_roofline:
        con.print("[bold]Step 3:[/bold] Roofline benchmark...")
        try:
            cache_dir = cfg.get(
                "roofline_cache_dir",
                str(Path.home() / ".profilelens" / "roofline_cache"),
            )
            roofline_csv_path = run_roofline_benchmark(
                cache_dir, gpu.model, console=con, force=refresh_roofline,
            )
        except RuntimeError as exc:
            con.print(f"  [yellow]Roofline benchmark skipped: {exc}[/yellow]")
    else:
        con.print("[bold]Step 3:[/bold] Roofline benchmark [dim](skipped)[/dim]")

    peaks_by_dtype: dict[str, Any] = {}
    if roofline_csv_path:
        ceilings = parse_roofline_csv(roofline_csv_path, dtype or "BF16")
        roofline_peaks = build_roofline_peaks(ceilings, gpu.model, dtype or "BF16")
        # Per-dtype peaks so each kernel is scored on its own MFMA ceiling.
        dt_list = list(observed_dtypes) if observed_dtypes else [dtype or "BF16"]
        peaks_by_dtype = build_roofline_peaks_multi(roofline_csv_path, gpu.model, dt_list)
        con.print(
            f"  HBM BW: {roofline_peaks.peak_bw_tbps.get('hbm', 0):.2f} TB/s, "
            f"MFMA: {roofline_peaks.peak_compute_tflops.get('mfma', 0):.1f} TFLOPS, "
            f"Ridge: {roofline_peaks.ridge_point} FLOP/byte"
        )
        if len(peaks_by_dtype) > 1:
            con.print(
                "  Per-dtype MFMA ceilings: "
                + ", ".join(
                    f"{dt} {p.peak_compute_tflops.get('mfma', 0):.0f} TFLOPS"
                    for dt, p in peaks_by_dtype.items()
                )
            )

    iteration_ids = None
    kernel_cv_map: dict[str, KernelCV] = {}
    if analysis_results and sample_dispatches > 0:
        con.print("[bold]Step 3b:[/bold] Dispatch analysis...")
        iteration_ids, kernel_cv_map = _compute_dispatch_sample(
            analysis_results, kernel_names, sample_dispatches, con,
        )

    con.print("[bold]Step 4:[/bold] Profiling kernels...")
    kernel_regex = build_kernel_regex(kernel_names)
    profile_output = os.path.join(output_dir, "profile")

    import time as _time
    t_profile_start = _time.time()
    success, stderr = run_rocprofv3(
        rocprofv3_path, profile_output, cset.counters, kernel_regex,
        user_command, console=con, iteration_ids=iteration_ids,
    )
    profiling_duration = _time.time() - t_profile_start
    if not success:
        con.print(f"[bold red]Profiling failed:[/bold red] {stderr[:500]}")
        raise RuntimeError(f"rocprofv3 failed: {stderr[:500]}")

    con.print(f"  Profiling completed in {profiling_duration:.1f}s")

    con.print("[bold]Step 5:[/bold] Computing metrics...")
    aggregated = parse_and_aggregate(
        profile_output, sample_iteration_ids=iteration_ids
    )

    profiled = [a for a in aggregated if a.avg_counters]
    unprofiled_count = len(aggregated) - len(profiled)
    if not profiled:
        profiled = aggregated
    con.print(
        f"  Parsed {len(profiled)} profiled kernels"
        + (f" ({unprofiled_count} without counters excluded)" if unprofiled_count else "")
    )

    _write_rank_breakdown(profiled, profile_output, con)

    if not profiled and analysis_results:
        from profilelens.profile.parser import AggregatedKernel
        for r in analysis_results:
            if r.label == "Aggregate Time per Kernel Name" and r.rows:
                for row in r.rows[:top_k]:
                    name = row.get("name", "")
                    if not name:
                        continue
                    profiled.append(AggregatedKernel(
                        kernel_name=name,
                        dispatch_count=row.get("count", 0) or 0,
                        total_duration_ns=int((row.get("total_ms", 0) or 0) * 1e6),
                        avg_duration_ns=(row.get("avg_ms", 0) or 0) * 1e6,
                        avg_counters={},
                        total_counters={},
                    ))
                break
        if profiled:
            con.print(
                f"  [yellow]No PMC CSVs from rocprofv3 — falling back to Phase 1 "
                f"trace data for top-{len(profiled)} kernels[/yellow]"
            )

    pkd = per_kernel_dtype or {}

    def _profiles_from(aggs: list) -> list[KernelProfile]:
        out: list[KernelProfile] = []
        for agg in aggs:
            kdt, ksrc = pkd.get(agg.kernel_name, (dtype or "BF16", "global"))
            kpeak = peaks_by_dtype.get(kdt, roofline_peaks)
            out.append(compute_kernel_metrics(
                agg, kpeak, kdt, gpu_info=gpu, dtype_source=ksrc,
            ))
        return out

    kernel_profiles: list[KernelProfile] = _profiles_from(profiled)

    from profilelens.profile.parser import parse_and_aggregate_regions
    per_region_kernels: dict[str, tuple[KernelProfile, ...]] = {}
    region_aggs = parse_and_aggregate_regions(
        profile_output, sample_iteration_ids=iteration_ids
    )
    if len(region_aggs) > 1:
        for rname, raggs in region_aggs.items():
            rprofiled = [a for a in raggs if a.avg_counters] or raggs
            per_region_kernels[rname] = tuple(_profiles_from(rprofiled))
        con.print(f"  Per-scenario regions detected: {', '.join(sorted(per_region_kernels))}")

    if analysis_results:
        from profilelens.profile.bottleneck import compute_headroom
        trace_ctx = _build_trace_context_map(analysis_results)
        scored: list[tuple[float, KernelProfile]] = []
        for kp in kernel_profiles:
            tc = trace_ctx.get(kp.kernel_name, {})
            e2e = tc.get("e2e_pct", 0.0)
            hr = compute_headroom(kp.mfma_util_pct, kp.bw_util_pct, kp.bottleneck, e2e)
            scored.append((hr.priority_score, kp))
        scored.sort(key=lambda x: x[0], reverse=True)
        kernel_profiles = [kp for _, kp in scored]
        con.print(f"  Re-ranked by optimization priority (headroom x e2e weight)")

    return ProfilingResult(
        gpu_arch=gpu.arch,
        gpu_model=gpu.model,
        detected_dtype=dtype or "mixed",
        kernels=tuple(kernel_profiles),
        roofline_peaks=roofline_peaks,
        observed_dtypes=tuple(observed_dtypes or ()),
        peaks_by_dtype=peaks_by_dtype,
        per_region_kernels=per_region_kernels,
        counter_set_name=cset.name,
        output_dir=output_dir,
        profiling_duration_s=round(profiling_duration, 1),
        kernel_cv={name: {"cv_pct": cv.cv_pct, "mean_us": cv.mean_us,
                          "min_us": cv.min_us, "max_us": cv.max_us,
                          "high_cv": cv.high_cv, "count": cv.count,
                          "is_collective": cv.is_collective}
                   for name, cv in kernel_cv_map.items()},
        sample_iteration_ids=tuple(iteration_ids or ()),
        sample_dispatches_per_kernel=sample_dispatches,
        phase2_workload_s=round(profiling_duration, 1),
        counters_used=tuple(cset.counters),
    )
