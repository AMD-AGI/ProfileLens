# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build per-kernel context (Module, Op, Shape) from Phase 1 trace data."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from profilelens.dtypes import QueryResult
from profilelens.profile.kernel_names import add_canonical_aliases


def kernel_time_share(row: dict[str, Any], total_kernel_ms: float) -> float:
    """Preserve an upstream timing denominator when it supplied a percentage."""
    if row.get("e2e_pct") is not None:
        return round(float(row["e2e_pct"]), 1)
    return (
        round((row.get("total_ms", 0) or 0) / total_kernel_ms * 100.0, 1)
        if total_kernel_ms > 0 else 0.0
    )


@dataclass
class KernelContext:
    """Phase 1 trace context attributed to a single GPU kernel."""

    kernel_name: str
    op: str = "unknown"
    op_invocations: int = 0
    module: str = "unknown"
    shape: str = ""
    avg_us: float = 0.0
    total_ms: float = 0.0
    dispatch_count: int = 0
    e2e_pct: float = 0.0
    category: str = "unknown"

def _find_result(results: list[QueryResult], label: str) -> QueryResult | None:
    for r in results:
        if r.label == label:
            return r
    return None


def _format_shape(dims: dict[str, int]) -> str:
    """Format ``args.Input Dims[i][j]`` keys into ``[a,b,c]x[d,e,f]``."""
    if not dims:
        return ""
    re_dim = re.compile(r"args\.Input Dims\[(\d+)\]\[(\d+)\]")
    by_input: dict[int, dict[int, int]] = defaultdict(dict)
    for k, v in dims.items():
        m = re_dim.match(k)
        if not m:
            continue
        i, j = int(m.group(1)), int(m.group(2))
        by_input[i][j] = v

    parts: list[str] = []
    for i in sorted(by_input):
        d = by_input[i]
        if not d:
            continue
        ordered = [str(d[j]) for j in sorted(d)]
        parts.append("[" + ",".join(ordered) + "]")
    return "x".join(parts) if parts else ""


def _build_id_to_shape(dims_result: QueryResult | None) -> dict[int, dict[str, int]]:
    """Group ``CPU Op Input Dimensions`` rows by slice_id."""
    out: dict[int, dict[str, int]] = defaultdict(dict)
    if not dims_result or not dims_result.rows:
        return out
    for row in dims_result.rows:
        sid = row.get("slice_id")
        k = row.get("key")
        v = row.get("int_value")
        if sid is not None and k and v is not None:
            out[sid][k] = v
    return out


def _build_module_chain(id_names_result: QueryResult | None) -> tuple[
    dict[int, str], dict[int, int | None],
]:
    """Map cpu_op id → name and id → parent_id."""
    id_to_name: dict[int, str] = {}
    id_to_parent: dict[int, int | None] = {}
    if not id_names_result or not id_names_result.rows:
        return id_to_name, id_to_parent
    for row in id_names_result.rows:
        sid = row.get("id")
        if sid is None:
            continue
        id_to_name[sid] = row.get("name", "")
        id_to_parent[sid] = row.get("parent_id")
    return id_to_name, id_to_parent


_MODULE_RE = re.compile(r"^nn\.Module:\s*(.+)$")


def _walk_to_module(
    cpu_id: int,
    id_to_name: dict[int, str],
    id_to_parent: dict[int, int | None],
) -> str:
    """Walk parent chain and return the innermost ``nn.Module:`` ancestor."""
    seen: set[int] = set()
    cur = cpu_id
    while cur is not None and cur not in seen:
        seen.add(cur)
        name = id_to_name.get(cur, "")
        m = _MODULE_RE.match(name)
        if m:
            return m.group(1).strip()
        cur = id_to_parent.get(cur)
    return "unknown"


def build_kernel_dtype_from_trace(
    analysis_results: list[QueryResult],
) -> dict[str, str]:
    """``kernel_name -> compute dtype`` from cpu_op input-type linkage."""
    cpu_ext = _find_result(analysis_results, "CPU Op External IDs")
    gpu_ext = _find_result(analysis_results, "GPU Kernel External IDs")
    types_q = _find_result(analysis_results, "CPU Op Input Types By Id")
    if not (cpu_ext and gpu_ext and types_q
            and cpu_ext.rows and gpu_ext.rows and types_q.rows):
        return {}

    from profilelens.profile.dtype_detect import _lowest_precision_dtype

    sid_dtype: dict[int, str] = {}
    sid_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in types_q.rows:
        internal = _normalize_trace_dtype(row.get("dtype"))
        sid = row.get("slice_id")
        if sid is not None and internal:
            sid_counts[sid][internal] += 1
    for sid, counts in sid_counts.items():
        sid_dtype[sid] = _lowest_precision_dtype(counts)

    ext_to_kernels: dict[int, list[str]] = defaultdict(list)
    for row in gpu_ext.rows:
        ext_to_kernels[row["ext_id"]].append(row["name"])

    per_kernel: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in cpu_ext.rows:
        dt = sid_dtype.get(row.get("id"))
        if not dt:
            continue
        for kn in ext_to_kernels.get(row.get("ext_id"), []):
            per_kernel[kn][dt] += 1

    return {
        kn: _lowest_precision_dtype(counts)
        for kn, counts in per_kernel.items() if counts
    }


_TRACE_DTYPE_TO_INTERNAL: dict[str, str] = {
    "c10::BFloat16": "BF16", "BFloat16": "BF16",
    "c10::Half": "FP16", "Half": "FP16",
    "c10::Float8_e4m3fn": "FP8", "c10::Float8_e4m3fnuz": "FP8",
    "c10::Float8_e5m2": "FP8", "c10::Float8_e5m2fnuz": "FP8",
    "Float8_e4m3fn": "FP8", "Float8_e5m2": "FP8",
    "float": "FP32", "Float": "FP32",
    "double": "FP64", "Double": "FP64",
}


def _normalize_trace_dtype(raw: str | None) -> str | None:
    """Map a trace input-type string to an internal compute dtype, else None."""
    if not raw:
        return None
    return _TRACE_DTYPE_TO_INTERNAL.get(raw)


def build_kernel_context_map(
    analysis_results: list[QueryResult],
) -> dict[str, KernelContext]:
    """Construct a kernel_name → KernelContext mapping from Phase 1 data."""
    from profilelens.trace.categories import categorize_op

    cpu_ext = _find_result(analysis_results, "CPU Op External IDs")
    gpu_ext = _find_result(analysis_results, "GPU Kernel External IDs")
    dims_q  = _find_result(analysis_results, "CPU Op Input Dimensions")
    id_names = _find_result(analysis_results, "CPU Op ID to Name Mapping")
    agg     = _find_result(analysis_results, "Aggregate Time per Kernel Name")

    if not agg or not agg.rows:
        return {}

    total_ms = sum(r.get("total_ms", 0) or 0 for r in agg.rows)

    out: dict[str, KernelContext] = {}
    for row in agg.rows:
        name = row.get("name", "")
        if not name:
            continue
        out[name] = KernelContext(
            kernel_name=name,
            avg_us=round((row.get("avg_ms", 0) or 0) * 1000.0, 2),
            total_ms=round(row.get("total_ms", 0) or 0, 3),
            dispatch_count=row.get("count", 0) or 0,
            e2e_pct=kernel_time_share(row, total_ms),
        )

    if not (cpu_ext and gpu_ext and cpu_ext.rows and gpu_ext.rows):
        return add_canonical_aliases(out)

    kid_to_name: dict[int, str] = {}
    ext_to_kernel_names: dict[int, list[str]] = defaultdict(list)
    for row in gpu_ext.rows:
        kid_to_name[row["id"]] = row["name"]
        ext_to_kernel_names[row["ext_id"]].append(row["name"])

    sid_to_dims = _build_id_to_shape(dims_q)
    id_to_name, id_to_parent = _build_module_chain(id_names)

    per_kernel_op: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_kernel_module: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_kernel_shape: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_kernel_invocations: dict[str, int] = defaultdict(int)

    for row in cpu_ext.rows:
        eid = row.get("ext_id")
        if eid is None:
            continue
        kernels = ext_to_kernel_names.get(eid, [])
        if not kernels:
            continue
        op_name = row.get("name", "")
        cpu_id  = row.get("id")
        shape   = _format_shape(sid_to_dims.get(cpu_id, {}))
        module  = _walk_to_module(cpu_id, id_to_name, id_to_parent) if cpu_id is not None else "unknown"
        for kn in kernels:
            per_kernel_op[kn][op_name] += 1
            per_kernel_invocations[kn] += 1
            if shape:
                per_kernel_shape[kn][shape] += 1
            if module and module != "unknown":
                per_kernel_module[kn][module] += 1

    for kname, ctx in out.items():
        op_counts = per_kernel_op.get(kname, {})
        if op_counts:
            top_op, _ = max(op_counts.items(), key=lambda kv: kv[1])
            ctx.op = top_op
            ctx.op_invocations = per_kernel_invocations.get(kname, 0)
            ctx.category = categorize_op(top_op)
        shape_counts = per_kernel_shape.get(kname, {})
        if shape_counts:
            top_shape, _ = max(shape_counts.items(), key=lambda kv: kv[1])
            ctx.shape = top_shape
        module_counts = per_kernel_module.get(kname, {})
        if module_counts:
            top_module, _ = max(module_counts.items(), key=lambda kv: kv[1])
            ctx.module = top_module

    return add_canonical_aliases(out)
