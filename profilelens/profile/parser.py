# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse rocprofv3 CSV output files."""

from __future__ import annotations

import csv
import hashlib
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KernelDispatch:
    """Raw data from a single kernel dispatch."""

    kernel_name: str
    start_ns: int = 0
    end_ns: int = 0
    duration_ns: int = 0
    counters: dict[str, float] = field(default_factory=dict)
    process_id: str = ""
    agent_id: str = ""

    kernel_id: str = ""
    grid_size: str = ""
    workgroup_size: str = ""
    vgpr_count: str = ""

    # Joins sampled counter rows to complete kernel-trace rows.
    dispatch_id: str = ""

    # Prevents positional attribution across independent queues.
    queue_id: str = ""


@dataclass
class RankAggregate:
    """Aggregated data for one kernel on one rank (one GPU)."""

    rank_key: str          # the Process_Id (stable per-rank discriminator)
    agent_id: str          # GPU agent string, e.g. "Agent 2"
    dispatch_count: int = 0
    total_duration_ns: int = 0
    avg_duration_ns: float = 0.0
    avg_counters: dict[str, float] = field(default_factory=dict)
    total_counters: dict[str, float] = field(default_factory=dict)
    grbm_dispatches: tuple[tuple[float, int], ...] = ()

    dispatches: tuple[tuple[dict[str, float], int], ...] = ()


@dataclass
class AggregatedKernel:
    """Aggregated data across all dispatches of a kernel."""

    kernel_name: str
    shape_sig: str = ""
    dispatch_count: int = 0
    total_duration_ns: int = 0
    avg_duration_ns: float = 0.0
    avg_counters: dict[str, float] = field(default_factory=dict)
    total_counters: dict[str, float] = field(default_factory=dict)
    # Per-dispatch (cycles_grbm, duration_ns) pairs preserved for sclk
    # derivation in metrics. Empty when GRBM_COUNT not in the PMC set.
    grbm_dispatches: tuple[tuple[float, int], ...] = ()

    dispatches: tuple[tuple[dict[str, float], int], ...] = ()
    n_ranks: int = 1
    per_rank: tuple[RankAggregate, ...] = ()
    variant_sig: str = ""
    variant_label: str = ""


@dataclass(frozen=True)
class GraphShapeMarker:
    """Capture-time operand shape re-emitted for one exact graph kernel symbol."""

    process_id: str
    kernel_key: str
    shape_sig: str
    start_ns: int
    end_ns: int


class ShapeMarker(tuple):
    """Legacy ``(stem, sig, start, end)`` marker with process ownership."""

    process_id: str

    def __new__(
        cls,
        stem: str,
        shape_sig: str,
        start_ns: int,
        end_ns: int,
        process_id: str = "",
    ) -> "ShapeMarker":
        marker = super().__new__(cls, (stem, shape_sig, start_ns, end_ns))
        marker.process_id = str(process_id or "")
        return marker

    def __reduce__(self):
        """Preserve process ownership across copy/pickle operations."""
        return (type(self), (*self, self.process_id))


class RegionMarker(tuple):
    """Profiler-region tuple with process ownership."""

    process_id: str

    def __new__(
        cls,
        name: str,
        start_ns: int,
        end_ns: int,
        process_id: str = "",
    ) -> "RegionMarker":
        marker = super().__new__(cls, (name, start_ns, end_ns))
        marker.process_id = str(process_id or "")
        return marker

    def __reduce__(self):
        """Preserve process ownership across copy/pickle operations."""
        return (type(self), (*self, self.process_id))


def _kernel_symbol_key(name: str) -> str:
    """Match the worker shim's stable key for a profiler kernel symbol."""
    name = (name or "").strip()
    if name.startswith("void "):
        name = name[5:]
    if not name:
        return ""
    return hashlib.sha1(
        name.encode("utf-8", errors="replace"), usedforsecurity=False
    ).hexdigest()[:20]


def _find_csv(output_dir: str, pattern: str) -> Path | None:
    """Find the first CSV matching a pattern in the output directory."""
    d = Path(output_dir)
    for f in sorted(d.rglob(pattern)):
        return f
    return None


def _find_all_csvs(output_dir: str, pattern: str) -> list[Path]:
    """Find all CSVs matching a pattern in the output directory."""
    d = Path(output_dir)
    return sorted(d.rglob(pattern))


def _safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val: str) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def parse_kernel_trace(output_dir: str) -> list[KernelDispatch]:
    """Parse all kernel_trace CSVs for timing data."""
    csv_paths = _find_all_csvs(output_dir, "*kernel_trace*.csv")
    if not csv_paths:
        return []

    dispatches = []
    for csv_path in csv_paths:
        # Kernel-trace CSVs may expose PID only in the filename.
        stem_pid = os.path.basename(csv_path).split("_", 1)[0]
        if not stem_pid.isdigit():
            stem_pid = ""
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Kernel_Name", row.get("Name", ""))
                if not name:
                    continue
                start = _safe_int(row.get("Start_Timestamp", row.get("BeginNs", "0")))
                end = _safe_int(row.get("End_Timestamp", row.get("EndNs", "0")))
                dur = end - start if end > start else _safe_int(row.get("DurationNs", "0"))
                dispatches.append(KernelDispatch(
                    kernel_name=name, start_ns=start, end_ns=end, duration_ns=dur,
                    process_id=(row.get("Process_Id") or row.get("Process_ID")
                                or row.get("process-id") or stem_pid),
                    agent_id=(row.get("Agent_Id") or row.get("Agent_ID") or ""),
                    kernel_id=(row.get("Kernel_Id") or row.get("Kernel_ID") or ""),
                    dispatch_id=(row.get("Dispatch_Id")
                                 or row.get("Dispatch_ID") or "").strip(),
                    queue_id=(row.get("Queue_Id")
                              or row.get("Queue_ID") or "").strip(),
                ))
    return dispatches


def parse_counter_collection(output_dir: str) -> list[KernelDispatch]:
    """Parse all counter_collection CSVs for per-dispatch HW counter values."""
    csv_paths = _find_all_csvs(output_dir, "*counter_collection*.csv")
    if not csv_paths:
        csv_paths = _find_all_csvs(output_dir, "*pmc*.csv")
    if not csv_paths:
        return []

    all_dispatches: list[KernelDispatch] = []
    seen_keys: set[tuple[str, str, str, str, str, int]] = set()

    for csv_path in csv_paths:
        stem_pid = Path(csv_path).name.split("_", 1)[0]
        if not stem_pid.isdigit():
            stem_pid = ""
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            is_long_format = "Counter_Name" in headers and "Counter_Value" in headers

            if is_long_format:
                dispatches = _parse_counter_long(reader)
            else:
                dispatches = _parse_counter_wide(reader)

            for d in dispatches:
                if not d.process_id:
                    d.process_id = stem_pid
                key = (
                    d.process_id,
                    d.agent_id,
                    d.queue_id,
                    d.kernel_name,
                    d.dispatch_id,
                    d.start_ns,
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_dispatches.append(d)

    return all_dispatches


def _parse_counter_long(reader: csv.DictReader) -> list[KernelDispatch]:
    """Parse the long/melted format: one row per (dispatch, counter)."""
    dispatch_map: dict[tuple[str, str, str, str, str, int, int], KernelDispatch] = {}

    for row in reader:
        name = row.get("Kernel_Name", row.get("Name", ""))
        if not name:
            continue
        pid = (row.get("Process_Id") or row.get("Process_ID")
               or row.get("process-id") or "")
        dispatch_id = (
            row.get("Dispatch_Id") or row.get("Dispatch_ID") or ""
        ).strip()
        correlation_id = (row.get("Correlation_Id") or "").strip()
        agent_id = (row.get("Agent_Id") or row.get("Agent_ID") or "").strip()
        queue_id = (row.get("Queue_Id") or row.get("Queue_ID") or "").strip()
        start = _safe_int(row.get("Start_Timestamp", "0"))
        end = _safe_int(row.get("End_Timestamp", "0"))
        # Correlation IDs may repeat across graph replays, so include timestamps
        # when no dispatch ordinal is available.
        key = (
            pid,
            agent_id,
            queue_id,
            name,
            dispatch_id or correlation_id,
            0 if dispatch_id else start,
            0 if dispatch_id else end,
        )

        if key not in dispatch_map:
            dur = end - start if end > start else 0
            dispatch_map[key] = KernelDispatch(
                kernel_name=name, start_ns=start, end_ns=end,
                duration_ns=dur, counters={},
                process_id=pid, agent_id=agent_id,
                kernel_id=row.get("Kernel_Id", ""),
                grid_size=row.get("Grid_Size", ""),
                workgroup_size=row.get("Workgroup_Size", ""),
                vgpr_count=row.get("VGPR_Count", ""),
                dispatch_id=dispatch_id,
                queue_id=queue_id,
            )

        counter_name = row.get("Counter_Name", "")
        counter_value = _safe_float(row.get("Counter_Value", "0"))
        if counter_name:
            dispatch_map[key].counters[counter_name] = counter_value

    return list(dispatch_map.values())


def _parse_counter_wide(reader: csv.DictReader) -> list[KernelDispatch]:
    """Parse the wide format: one row per dispatch with counter columns."""
    _skip_cols = {
        "Kernel_Name", "Name", "gpu-id", "queue-id", "process-id",
        "Dispatch_ID", "Dispatch_Id", "Correlation_Id", "Agent_Id", "Agent_ID",
        "Queue_Id", "Queue_ID", "Process_Id", "Process_ID", "Thread_Id",
        "Grid_Size", "Kernel_Id",
        "Workgroup_Size", "LDS_Block_Size", "Scratch_Size",
        "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count",
        "Start_Timestamp", "End_Timestamp",
    }
    dispatches = []
    for row in reader:
        name = row.get("Kernel_Name", row.get("Name", ""))
        if not name:
            continue
        counters: dict[str, float] = {}
        for k, v in row.items():
            if k in _skip_cols:
                continue
            fv = _safe_float(v)
            if fv != 0.0 or (isinstance(v, str) and v.strip() == "0"):
                counters[k] = fv

        start = _safe_int(row.get("Start_Timestamp", "0"))
        end = _safe_int(row.get("End_Timestamp", "0"))
        dur = end - start if end > start else 0

        dispatches.append(KernelDispatch(
            kernel_name=name, start_ns=start, end_ns=end,
            duration_ns=dur, counters=counters,
            process_id=(row.get("Process_Id") or row.get("Process_ID")
                        or row.get("process-id") or ""),
            agent_id=(row.get("Agent_Id") or row.get("Agent_ID") or ""),
            kernel_id=row.get("Kernel_Id", ""),
            grid_size=row.get("Grid_Size", ""),
            workgroup_size=row.get("Workgroup_Size", ""),
            vgpr_count=row.get("VGPR_Count", ""),
            dispatch_id=(row.get("Dispatch_Id")
                         or row.get("Dispatch_ID") or "").strip(),
            queue_id=(row.get("Queue_Id")
                      or row.get("Queue_ID") or "").strip(),
        ))
    return dispatches


# Substring that marks an MFMA matrix-FLOP counter
_MFMA_COUNTER_MARK = "MFMA_MOPS"

# Minimum fraction that must read MFMA > 0 for the matrix-FLOP reading
_MFMA_PRESENCE_MIN = 0.75


def _is_leaked_mfma(key: str, vals: list[float]) -> bool:
    """True when ``key`` is an MFMA FLOP counter whose nonzero readings are sparse"""
    if _MFMA_COUNTER_MARK not in key or not vals:
        return False
    nonzero_ratio = sum(1 for v in vals if v > 0) / len(vals)
    return nonzero_ratio < _MFMA_PRESENCE_MIN


def _aggregate_one_rank(
    rank_key: str,
    agent_id: str,
    durations: list[int],
    counter_list: list[dict[str, float]],
    grbm_pairs: list[tuple[float, int]],
    dispatch_pairs: list[tuple[dict[str, float], int]] | None = None,
) -> RankAggregate:
    """Aggregate one kernel's dispatches on a single rank using median across dispatches"""
    total_dur = sum(durations) if durations else 0
    avg_dur = total_dur / len(durations) if durations else 0.0

    all_keys: set[str] = set()
    for cd in counter_list:
        all_keys.update(cd.keys())

    total_ctrs: dict[str, float] = {}
    avg_ctrs: dict[str, float] = {}
    for k in all_keys:
        vals = [cd.get(k, 0.0) for cd in counter_list]
        total_ctrs[k] = sum(vals)
        if _is_leaked_mfma(k, vals):
            avg_ctrs[k] = 0.0
        else:
            avg_ctrs[k] = statistics.median(vals) if vals else 0.0

    n = max(len(durations), len(counter_list), 1)
    return RankAggregate(
        rank_key=rank_key,
        agent_id=agent_id,
        dispatch_count=n,
        total_duration_ns=total_dur,
        avg_duration_ns=avg_dur,
        avg_counters=avg_ctrs,
        total_counters=total_ctrs,
        grbm_dispatches=tuple(grbm_pairs),
        dispatches=tuple(dispatch_pairs or ()),
    )


def _combine_ranks(
    name: str, ranks: list[RankAggregate], shape_sig: str = "",
    variant_sig: str = "",
) -> AggregatedKernel:
    """Combine per-rank aggregates into a per-rank-representative kernel."""

    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0

    # Don't count empty counter ranks
    src = [r for r in ranks if r.avg_counters] or list(ranks)
    nr = len(src)

    dispatch_count = round(mean([r.dispatch_count for r in src]))
    total_duration_ns = int(mean([r.total_duration_ns for r in src]))
    avg_duration_ns = mean([r.avg_duration_ns for r in src])

    # Mean across ranks of each counter (avg and total flavours).
    avg_keys: set[str] = set()
    tot_keys: set[str] = set()
    for r in src:
        avg_keys.update(r.avg_counters.keys())
        tot_keys.update(r.total_counters.keys())
    avg_counters = {k: mean([r.avg_counters.get(k, 0.0) for r in src]) for k in avg_keys}
    total_counters = {k: mean([r.total_counters.get(k, 0.0) for r in src]) for k in tot_keys}

    # GRBM pairs: union across counter ranks (sclk derivation is rank-agnostic).
    grbm: list[tuple[float, int]] = []
    for r in src:
        grbm.extend(r.grbm_dispatches)


    disp: list[tuple[dict[str, float], int]] = []
    for r in src:
        disp.extend(r.dispatches)

    return AggregatedKernel(
        kernel_name=name,
        shape_sig=shape_sig,
        dispatch_count=dispatch_count,
        total_duration_ns=total_duration_ns,
        avg_duration_ns=avg_duration_ns,
        avg_counters=avg_counters,
        total_counters=total_counters,
        grbm_dispatches=tuple(grbm),
        dispatches=tuple(disp),
        n_ranks=nr,
        per_rank=tuple(src),
        variant_sig=variant_sig,
    )



_VARIANT_FIELDS: tuple[tuple[str, str], ...] = (
    ("grid", "grid_size"),
    ("wg", "workgroup_size"),
    ("vgpr", "vgpr_count"),
)


def _variant_sig(d: KernelDispatch) -> str:
    parts = [getattr(d, attr) for _short, attr in _VARIANT_FIELDS]
    return "/".join(parts) if any(parts) else ""


def _variant_label_for_group(variant_sigs: list[str]) -> dict[str, str]:
    uniq = list(dict.fromkeys(variant_sigs))  # de-dup, preserve order
    split = {s: s.split("/") for s in uniq}
    n = len(_VARIANT_FIELDS)
    # Field positions whose value is not the same across every variant.
    differing = [
        i for i in range(n)
        if len({split[s][i] if i < len(split[s]) else "" for s in uniq}) > 1
    ]
    labels: dict[str, str] = {}
    for s in uniq:
        parts = split[s]
        tags = [
            f"{_VARIANT_FIELDS[i][0]}={parts[i]}"
            for i in differing
            if i < len(parts) and parts[i]
        ]
        # Fallback: nothing varied (or fields were empty) — use the raw sig so
        # the row is never left without a disambiguator.
        labels[s] = " ".join(tags) if tags else s
    return labels


def aggregate_by_kernel(
    trace_dispatches: list[KernelDispatch],
    counter_dispatches: list[KernelDispatch],
    shape_of: "callable | None" = None,
) -> list[AggregatedKernel]:
    """Group dispatches by (kernel, shape, variant, rank), then combine ranks."""
    _sig = (lambda d: (shape_of(d) or "")) if shape_of else (lambda d: "")

    _variant_split = not trace_dispatches
    _variant = _variant_sig if _variant_split else (lambda d: "")

    # (kernel, sig, variant, rank) -> timing / counters / grbm / per-dispatch
    timing: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    counters_by: dict[tuple[str, str, str, str], list[dict[str, float]]] = defaultdict(list)
    grbm_by: dict[tuple[str, str, str, str], list[tuple[float, int]]] = defaultdict(list)
    # Per-dispatch (counters, duration_ns) pairs for population-consistent
    # roofline metrics (see metrics._per_dispatch_roofline).
    disp_by: dict[tuple[str, str, str, str], list[tuple[dict[str, float], int]]] = defaultdict(list)
    agent_of: dict[tuple[str, str, str, str], str] = {}

    for d in trace_dispatches:
        if d.duration_ns > 0:
            key = (d.kernel_name, _sig(d), _variant(d), d.process_id)
            timing[key].append(d.duration_ns)
            agent_of.setdefault(key, d.agent_id)

    trace_timed_keys = set(timing.keys())

    for d in counter_dispatches:
        key = (d.kernel_name, _sig(d), _variant(d), d.process_id)
        counters_by[key].append(d.counters)
        agent_of.setdefault(key, d.agent_id)
        if d.duration_ns > 0 and key not in trace_timed_keys:
            timing[key].append(d.duration_ns)
        if d.duration_ns > 0:
            disp_by[key].append((d.counters, int(d.duration_ns)))
        grbm = d.counters.get("GRBM_COUNT")
        if grbm is not None and grbm > 0 and d.duration_ns > 0:
            grbm_by[key].append((float(grbm), int(d.duration_ns)))

    all_keys = set(timing.keys()) | set(counters_by.keys())

    per_kernel_ranks: dict[tuple[str, str, str], list[RankAggregate]] = defaultdict(list)
    for (name, sig, variant, rank_key) in all_keys:
        ra = _aggregate_one_rank(
            rank_key=rank_key,
            agent_id=agent_of.get((name, sig, variant, rank_key), ""),
            durations=timing.get((name, sig, variant, rank_key), []),
            counter_list=counters_by.get((name, sig, variant, rank_key), []),
            grbm_pairs=grbm_by.get((name, sig, variant, rank_key), []),
            dispatch_pairs=disp_by.get((name, sig, variant, rank_key), []),
        )
        per_kernel_ranks[(name, sig, variant)].append(ra)

    results: list[AggregatedKernel] = []
    for (name, sig, variant), ranks in per_kernel_ranks.items():
        ranks.sort(key=lambda r: r.agent_id or r.rank_key)
        results.append(_combine_ranks(name, ranks, shape_sig=sig, variant_sig=variant))

    by_ns: dict[tuple[str, str], list[AggregatedKernel]] = defaultdict(list)
    for k in results:
        by_ns[(k.kernel_name, k.shape_sig)].append(k)
    for group in by_ns.values():
        variants = [k.variant_sig for k in group if k.variant_sig]
        if len(variants) > 1:
            labels = _variant_label_for_group(variants)
            for k in group:
                k.variant_label = labels.get(k.variant_sig, "")

    results.sort(key=lambda k: k.total_duration_ns, reverse=True)
    return results


def parse_shape_ranges(output_dir: str) -> list[ShapeMarker]:
    """Read per-launch shape markers."""
    ranges: list[ShapeMarker] = []
    for f in _find_all_csvs(output_dir, "*marker_api_trace.csv"):
        stem_pid = f.name.split("_", 1)[0]
        if not stem_pid.isdigit():
            stem_pid = ""
        try:
            with open(f, newline="") as fh:
                for row in csv.DictReader(fh):
                    name = (row.get("Function") or "").strip()
                    if not name.startswith("pl_shape_"):
                        continue
                    body = name[len("pl_shape_"):]
                    # Kernel names may contain "__"; split at the last delimiter.
                    stem, sep, sig = body.rpartition("__")
                    if not sep:
                        stem, sig = "", body
                    try:
                        s = int(row["Start_Timestamp"])
                        e = int(row["End_Timestamp"])
                    except (KeyError, ValueError):
                        continue
                    ranges.append(ShapeMarker(
                        stem,
                        sig,
                        s,
                        e,
                        process_id=(row.get("Process_Id") or row.get("Process_ID")
                                    or row.get("process-id") or stem_pid),
                    ))
        except OSError:
            continue
    return ranges


def parse_graph_shape_ranges(output_dir: str) -> list[GraphShapeMarker]:
    """Read exact graph-node shape markers."""
    markers: list[GraphShapeMarker] = []
    prefix = "pl_graph_shape_"
    for path in _find_all_csvs(output_dir, "*marker_api_trace.csv"):
        stem_pid = path.name.split("_", 1)[0]
        if not stem_pid.isdigit():
            stem_pid = ""
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    name = (row.get("Function") or "").strip()
                    if not name.startswith(prefix):
                        continue
                    key, sep, sig = name[len(prefix):].partition("__")
                    if not sep or not key or not sig:
                        continue
                    try:
                        start = int(row["Start_Timestamp"])
                        end = int(row["End_Timestamp"])
                    except (KeyError, ValueError):
                        continue
                    markers.append(GraphShapeMarker(
                        process_id=str(row.get("Process_Id") or row.get("Process_ID")
                                       or row.get("process-id") or stem_pid),
                        kernel_key=key,
                        shape_sig=sig,
                        start_ns=start,
                        end_ns=end,
                    ))
        except OSError:
            continue
    return markers


def _build_shape_lookup(
    shape_ranges: list[tuple[str, str, int, int]],
) -> "callable":
    """Time-window ``start_ns -> sig`` lookup."""
    ordered = sorted(shape_ranges, key=lambda r: (r[2], r[3]))

    def lookup(d: KernelDispatch) -> str:
        sig = ""
        width = None
        for _stem, s, start, end in ordered:
            if start <= d.start_ns <= end:
                w = end - start
                if width is None or w < width:
                    sig, width = s, w
        return sig

    return lookup


def _attribute_shape_sigs(
    trace_dispatches: list[KernelDispatch],
    counter_dispatches: list[KernelDispatch],
    shape_markers: list[tuple[str, str, int, int]],
    region_ranges: list[tuple[str, int, int]],
) -> dict[int, str]:
    """Attribute legacy markers through the combined ownership path."""
    return _attribute_combined_shape_sigs(
        trace_dispatches,
        counter_dispatches,
        shape_markers,
        [],
        region_ranges,
    )


def _attribute_graph_shape_sigs(
    trace_dispatches: list[KernelDispatch],
    counter_dispatches: list[KernelDispatch],
    graph_markers: list[GraphShapeMarker],
    region_ranges: list[tuple[str, int, int]],
) -> dict[int, str]:
    """Attribute exact graph markers through the combined ownership path."""
    return _attribute_combined_shape_sigs(
        trace_dispatches,
        counter_dispatches,
        [],
        graph_markers,
        region_ranges,
    )


def _attribute_combined_shape_sigs(
    trace_dispatches: list[KernelDispatch],
    counter_dispatches: list[KernelDispatch],
    shape_markers: list[tuple[str, str, int, int]],
    graph_markers: list[GraphShapeMarker],
    region_ranges: list[tuple[str, int, int]],
    sample_iteration_ids: list[int] | tuple[int, ...] | None = None,
) -> dict[int, str]:
    """Attribute legacy and exact-node markers in one launch-order stream."""
    from profilelens.profile.dtype_detect import _flydsl_stem

    sig_of: dict[int, str] = {}
    if not shape_markers and not graph_markers:
        return sig_of

    def _in_region(t: int, process_id: str = "") -> bool:
        if not region_ranges:
            return True
        for region in region_ranges:
            _name, start, end = region[:3]
            region_pid = str(
                getattr(region, "process_id", "")
                or (region[3] if len(region) > 3 else "")
                or ""
            )
            if process_id and region_pid and process_id != region_pid:
                continue
            if start <= t <= end:
                return True
        return False

    # owner id -> [(push_ns, sig)] in push order.
    seq: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
    for marker in shape_markers or ():
        # Preserve compatibility with legacy marker tuple forms.
        stem, sig, start, _end = marker[:4]
        process_id = str(
            getattr(marker, "process_id", "")
            or (marker[4] if len(marker) > 4 else "")
            or ""
        )
        if _in_region(start, process_id):
            seq[("L", process_id, stem)].append((start, sig))
    for marker in graph_markers or ():
        if _in_region(marker.start_ns, marker.process_id):
            seq[("G", marker.process_id, marker.kernel_key)].append(
                (marker.start_ns, marker.shape_sig)
            )
    if not seq:
        return sig_of

    legacy_stems = {owner[2] for owner in seq if owner[0] == "L"}

    def _owners_of(dispatch: KernelDispatch) -> list[tuple]:
        found: list[tuple] = []
        name = dispatch.kernel_name
        stem = ""
        if name in legacy_stems:
            stem = name
        else:
            normalized = _flydsl_stem(name)
            if normalized in legacy_stems:
                stem = normalized or ""
        if stem:
            process_owner = ("L", dispatch.process_id, stem)
            global_owner = ("L", "", stem)
            if process_owner in seq:
                found.append(process_owner)
            elif global_owner in seq:
                found.append(global_owner)
        key = _kernel_symbol_key(name)
        if key:
            graph_owner = ("G", dispatch.process_id, key)
            global_graph_owner = ("G", "", key)
            if graph_owner in seq:
                found.append(graph_owner)
            elif global_graph_owner in seq:
                # PID-less markers remain subject to the lane ambiguity check.
                found.append(global_graph_owner)
        return found

    # Merge marker namespaces that own the same dispatch.
    parent: dict[tuple, tuple] = {owner: owner for owner in seq}

    def _find(owner: tuple) -> tuple:
        root = owner
        while parent[root] != root:
            root = parent[root]
        while parent[owner] != root:      # path compression
            parent[owner], owner = root, parent[owner]
        return root

    def _union(a: tuple, b: tuple) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for dispatches in (counter_dispatches, trace_dispatches):
        for dispatch in dispatches:
            if not _in_region(dispatch.start_ns, dispatch.process_id):
                continue
            owners = _owners_of(dispatch)
            for other in owners[1:]:
                _union(owners[0], other)

    merged: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
    for owner, entries in seq.items():
        group = _find(owner)
        merged[group].extend(entries)
    for entries in merged.values():
        entries.sort(key=lambda e: e[0])   # CPU push order == launch order

    sampled_ids = sorted({i for i in (sample_iteration_ids or ()) if i > 0})

    def _align(dispatches: list[KernelDispatch], *, sampled: bool = False) -> None:
        """Align dispatches to markers, failing closed on count mismatch."""
        by_owner: dict[tuple, list[KernelDispatch]] = defaultdict(list)
        for dispatch in dispatches:
            owners = _owners_of(dispatch)
            if owners:
                by_owner[_find(owners[0])].append(dispatch)
        for group, sequence in merged.items():
            candidates = sorted(by_owner.get(group, []),
                                key=lambda dispatch: dispatch.start_ns)
            if not sequence:
                continue
            if region_ranges:
                profiled = [
                    d for d in candidates
                    if _in_region(d.start_ns, d.process_id)
                ]
            else:
                if sampled and sampled_ids:
                    profiled = candidates
                else:
                    # Without regions, skip leading unmarked warmups.
                    if len(candidates) < len(sequence):
                        continue
                    profiled = candidates[len(candidates) - len(sequence):]
            aligned_sequence = sequence
            if sampled and sampled_ids:
                kernel_ids = {d.kernel_id for d in profiled if d.kernel_id}
                distinct_shapes = {sig for _push, sig in sequence}
                if len(kernel_ids) > 1:
                    if len(distinct_shapes) != 1:
                        continue
                    sig = next(iter(distinct_shapes))
                    for dispatch in profiled:
                        sig_of[id(dispatch)] = sig
                    continue
                aligned_sequence = [
                    sequence[i - 1] for i in sampled_ids if i <= len(sequence)
                ]
                if len(profiled) != len(aligned_sequence):
                    continue
            elif len(profiled) != len(aligned_sequence):
                continue
            lanes = {
                (dispatch.process_id, dispatch.agent_id, dispatch.queue_id)
                for dispatch in profiled
                if (dispatch.process_id or dispatch.agent_id or dispatch.queue_id)
            }
            distinct_shapes = {sig for _push, sig in aligned_sequence}
            if len(lanes) > 1 and len(distinct_shapes) > 1:
                # Multi-shape ordering is ambiguous across execution lanes.
                continue
            for dispatch, (_push, sig) in zip(profiled, aligned_sequence):
                sig_of[id(dispatch)] = sig

    # Align the complete trace, then join sampled counters by dispatch identity.
    _align(trace_dispatches)

    sig_by_dispatch_id: dict[tuple[str, str, str, str, str], str] = {}
    ambiguous_dispatch_ids: set[tuple[str, str, str, str, str]] = set()
    sig_without_queue: dict[tuple[str, str, str], str] = {}
    ambiguous_without_queue: set[tuple[str, str, str]] = set()
    for dispatch in trace_dispatches:
        sig = sig_of.get(id(dispatch))
        dispatch_id = (dispatch.dispatch_id or "").strip()
        symbol_key = _kernel_symbol_key(dispatch.kernel_name)
        if not (sig and dispatch_id and symbol_key):
            continue
        key = (
            dispatch.process_id,
            dispatch.agent_id,
            dispatch.queue_id,
            dispatch_id,
            symbol_key,
        )
        prior = sig_by_dispatch_id.get(key)
        if prior is not None and prior != sig:
            ambiguous_dispatch_ids.add(key)
        else:
            sig_by_dispatch_id[key] = sig
        queue_agnostic_key = (dispatch.process_id, dispatch_id, symbol_key)
        prior = sig_without_queue.get(queue_agnostic_key)
        if prior is not None and prior != sig:
            ambiguous_without_queue.add(queue_agnostic_key)
        else:
            sig_without_queue[queue_agnostic_key] = sig
    for key in ambiguous_dispatch_ids:
        sig_by_dispatch_id.pop(key, None)
    for key in ambiguous_without_queue:
        sig_without_queue.pop(key, None)

    joined = set()
    for dispatch in counter_dispatches:
        dispatch_id = (dispatch.dispatch_id or "").strip()
        symbol_key = _kernel_symbol_key(dispatch.kernel_name)
        if not (dispatch_id and symbol_key):
            continue
        sig = sig_by_dispatch_id.get((
            dispatch.process_id,
            dispatch.agent_id,
            dispatch.queue_id,
            dispatch_id,
            symbol_key,
        ))
        if not sig:
            # Allow an unambiguous join when one CSV omits lane fields.
            sig = sig_without_queue.get((
                dispatch.process_id,
                dispatch_id,
                symbol_key,
            ))
        if sig:
            sig_of[id(dispatch)] = sig
            joined.add(id(dispatch))

    # Without a trace, counter rows are the launch-order source.  When dispatch
    # sampling is active, apply the same per-kernel iteration IDs to markers.
    if not trace_dispatches:
        leftover = [d for d in counter_dispatches if id(d) not in joined]
    else:
        leftover = [
            dispatch for dispatch in counter_dispatches
            if id(dispatch) not in joined and not (dispatch.dispatch_id or "").strip()
        ]
    if leftover:
        _align(leftover, sampled=bool(sampled_ids))
    return sig_of


def _drop_unattributed_when_split(
    aggs: list[AggregatedKernel],
) -> list[AggregatedKernel]:
    split_names = {a.kernel_name for a in aggs if a.shape_sig}
    return [a for a in aggs if a.shape_sig or a.kernel_name not in split_names]


def parse_and_aggregate(
    output_dir: str,
    sample_iteration_ids: list[int] | tuple[int, ...] | None = None,
) -> list[AggregatedKernel]:
    """Parse all CSV files and return aggregated kernel data."""
    trace = parse_kernel_trace(output_dir)
    counters = parse_counter_collection(output_dir)
    markers = parse_shape_ranges(output_dir)
    graph_markers = parse_graph_shape_ranges(output_dir)
    shape_of = None
    if markers or graph_markers:
        regions = parse_marker_ranges(output_dir)
        sig_of = _attribute_combined_shape_sigs(
            trace, counters, markers, graph_markers, regions,
            sample_iteration_ids=sample_iteration_ids,
        )
        shape_of = lambda d: sig_of.get(id(d), "")  # noqa: E731
    aggs = aggregate_by_kernel(trace, counters, shape_of=shape_of)
    return _drop_unattributed_when_split(aggs)


def parse_marker_ranges(output_dir: str) -> list[RegionMarker]:
    """Read the shim's per-scenario roctx ranges"""
    ranges: list[RegionMarker] = []
    for f in _find_all_csvs(output_dir, "*marker_api_trace.csv"):
        stem_pid = f.name.split("_", 1)[0]
        if not stem_pid.isdigit():
            stem_pid = ""
        try:
            with open(f, newline="") as fh:
                for row in csv.DictReader(fh):
                    name = (row.get("Function") or "").strip()
                    if not name.startswith("pl_region_"):
                        continue
                    try:
                        s = int(row["Start_Timestamp"])
                        e = int(row["End_Timestamp"])
                    except (KeyError, ValueError):
                        continue
                    ranges.append(RegionMarker(
                        name,
                        s,
                        e,
                        process_id=(row.get("Process_Id") or row.get("Process_ID")
                                    or row.get("process-id") or stem_pid),
                    ))
        except OSError:
            continue
    return ranges


def parse_and_aggregate_regions(
    output_dir: str,
    sample_iteration_ids: list[int] | tuple[int, ...] | None = None,
) -> dict[str, list[AggregatedKernel]]:
    """Aggregate kernels per scenario range (``pl_region_<N>``)"""
    ranges = parse_marker_ranges(output_dir)
    if not ranges:
        return {}
    trace = parse_kernel_trace(output_dir)
    counters = parse_counter_collection(output_dir)
    markers = parse_shape_ranges(output_dir)
    graph_markers = parse_graph_shape_ranges(output_dir)
    shape_of = None
    if markers or graph_markers:
        sig_of = _attribute_combined_shape_sigs(
            trace, counters, markers, graph_markers, ranges,
            sample_iteration_ids=sample_iteration_ids,
        )
        shape_of = lambda d: sig_of.get(id(d), "")  # noqa: E731

    def _region_of(start_ns: int, process_id: str = "") -> "str | None":
        for region in ranges:
            name, s, e = region
            region_pid = getattr(region, "process_id", "")
            if process_id and region_pid and process_id != region_pid:
                continue
            if s <= start_ns <= e:
                return name
        return None

    td_by: dict[str, list[KernelDispatch]] = defaultdict(list)
    cd_by: dict[str, list[KernelDispatch]] = defaultdict(list)
    for d in trace:
        r = _region_of(d.start_ns, d.process_id)
        if r is not None:
            td_by[r].append(d)
    for d in counters:
        r = _region_of(d.start_ns, d.process_id)
        if r is not None:
            cd_by[r].append(d)

    out: dict[str, list[AggregatedKernel]] = {}
    for rname in sorted(set(td_by) | set(cd_by)):
        aggs = aggregate_by_kernel(
            td_by.get(rname, []), cd_by.get(rname, []), shape_of=shape_of,
        )
        out[rname] = _drop_unattributed_when_split(aggs)
    return out
