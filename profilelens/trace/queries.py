# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pre-built SQL queries for Perfetto / PyTorch profiler traces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from profilelens.dtypes import QueryResult


@dataclass(frozen=True)
class PrebuiltQuery:
    """A named, pre-written SQL query."""

    label: str
    sql: str
    description: str = ""
    internal: bool = False


TOP_SLOWEST_KERNELS = PrebuiltQuery(
    label="Top 10 Slowest GPU Kernels",
    description="Individual kernel invocations ranked by wall-clock duration.",
    sql=(
        "SELECT name, dur / 1000000.0 AS dur_ms "
        "FROM slice "
        "WHERE cat = 'kernel' "
        "ORDER BY dur DESC "
        "LIMIT 10"
    ),
)

AGGREGATE_KERNEL_TIME = PrebuiltQuery(
    label="Aggregate Time per Kernel Name",
    description="Total and average time grouped by kernel name.",
    sql=(
        "SELECT name, "
        "SUM(dur) / 1000000.0 AS total_ms, "
        "COUNT(*) AS count, "
        "AVG(dur) / 1000000.0 AS avg_ms "
        "FROM slice "
        "WHERE cat = 'kernel' "
        "GROUP BY name "
        "ORDER BY total_ms DESC"
    ),
)

TOP_CPU_OP_BOTTLENECKS = PrebuiltQuery(
    label="Top 10 CPU Op Bottlenecks",
    description="Slowest individual CPU operator invocations.",
    sql=(
        "SELECT name, dur / 1000000.0 AS dur_ms "
        "FROM slice "
        "WHERE cat = 'cpu_op' "
        "ORDER BY dur DESC "
        "LIMIT 10"
    ),
)

CATEGORY_BREAKDOWN = PrebuiltQuery(
    label="Category Breakdown",
    description="Time distribution across event categories (kernel, cpu_op, python, …).",
    sql=(
        "SELECT cat, "
        "SUM(dur) / 1000000.0 AS total_ms, "
        "COUNT(*) AS count "
        "FROM slice "
        "GROUP BY cat "
        "ORDER BY total_ms DESC"
    ),
)

TRACE_TIME_SUMMARY = PrebuiltQuery(
    label="Trace Time Summary",
    description=(
        "Duration of the gpu_user_annotation slice (total trace time) "
        "and the Trace category slice (profiler overhead inclusive)."
    ),
    sql=(
        "SELECT cat, name, ROUND(dur / 1000000.0, 2) AS dur_ms "
        "FROM slice "
        "WHERE cat IN ('gpu_user_annotation', 'Trace') "
        "ORDER BY dur DESC"
    ),
)

CPU_OP_EXTID = PrebuiltQuery(
    label="CPU Op External IDs",
    description="Maps cpu_op slices to their External id, parent_id, and input dims.",
    internal=True,
    sql=(
        "SELECT s.id, s.name, s.parent_id, s.arg_set_id, a.int_value AS ext_id "
        "FROM slice s "
        "JOIN args a ON s.arg_set_id = a.arg_set_id "
        "AND a.flat_key = 'args.External id' "
        "WHERE s.cat = 'cpu_op'"
    ),
)

GPU_KERNEL_EXTID = PrebuiltQuery(
    label="GPU Kernel External IDs",
    description="Maps GPU kernels/memcpy to their External id and duration.",
    internal=True,
    sql=(
        "SELECT s.id, s.name, s.dur, s.cat, a.int_value AS ext_id "
        "FROM slice s "
        "JOIN args a ON s.arg_set_id = a.arg_set_id "
        "AND a.flat_key = 'args.External id' "
        "WHERE s.cat IN ('kernel', 'gpu_memcpy')"
    ),
)

CPU_OP_INPUT_DIMS = PrebuiltQuery(
    label="CPU Op Input Dimensions",
    description="Extracts per-input dimension values for cpu_ops (for shape grouping).",
    internal=True,
    sql=(
        "SELECT s.id AS slice_id, a.key, a.int_value "
        "FROM slice s "
        "JOIN args a ON s.arg_set_id = a.arg_set_id "
        "AND a.flat_key = 'args.Input Dims' "
        "WHERE s.cat = 'cpu_op'"
    ),
)

CPU_OP_ID_NAMES = PrebuiltQuery(
    label="CPU Op ID to Name Mapping",
    description="Maps all cpu_op slice IDs to names for parent-chain resolution.",
    internal=True,
    sql=(
        "SELECT id, name, parent_id "
        "FROM slice "
        "WHERE cat = 'cpu_op'"
    ),
)

CPU_OP_INPUT_TYPES = PrebuiltQuery(
    label="CPU Op Input Types",
    description="Distinct tensor dtypes used in cpu_ops (for profiling counter selection).",
    internal=True,
    sql=(
        "SELECT a.string_value AS dtype, COUNT(*) AS cnt "
        "FROM slice s "
        "JOIN args a ON s.arg_set_id = a.arg_set_id "
        "AND a.flat_key = 'args.Input type' "
        "WHERE s.cat = 'cpu_op' "
        "GROUP BY a.string_value "
        "ORDER BY cnt DESC"
    ),
)


CPU_OP_INPUT_TYPES_BY_ID = PrebuiltQuery(
    label="CPU Op Input Types By Id",
    description="Per-cpu_op-slice input tensor dtypes (for per-kernel dtype correlation).",
    internal=True,
    sql=(
        "SELECT s.id AS slice_id, a.string_value AS dtype "
        "FROM slice s "
        "JOIN args a ON s.arg_set_id = a.arg_set_id "
        "AND a.flat_key = 'args.Input type' "
        "WHERE s.cat = 'cpu_op'"
    ),
)


CUDA_RUNTIME_SUMMARY = PrebuiltQuery(
    label="CUDA Runtime Summary",
    description=(
        "Aggregates CUDA/HIP runtime events by name, showing total time and count. "
        "Blocking sync ops (hipDeviceSynchronize, cudaStreamSynchronize, hipMemcpyWithStream) "
        "reveal host-device synchronization overhead."
    ),
    sql=(
        "SELECT name, COUNT(*) AS cnt, "
        "ROUND(SUM(dur) / 1000000.0, 2) AS total_ms, "
        "ROUND(AVG(dur) / 1000.0, 2) AS avg_us "
        "FROM slice "
        "WHERE cat = 'cuda_runtime' "
        "GROUP BY name "
        "ORDER BY total_ms DESC"
    ),
)

PYTHON_MODULE_SUMMARY = PrebuiltQuery(
    label="Python Module Summary",
    description=(
        "Lists all nn.Module instances from python_function slices. "
        "Use to verify model architecture claims (e.g. 'model is TimeSformer', "
        "'24 JointTransformerBlocks', 'VAE Decoder is DC-AE')."
    ),
    sql=(
        "SELECT name, COUNT(*) AS invocations "
        "FROM slice "
        "WHERE cat = 'python_function' AND name LIKE 'nn.Module:%' "
        "GROUP BY name "
        "ORDER BY invocations DESC"
    ),
)

MEMCPY_BANDWIDTH = PrebuiltQuery(
    label="Memcpy Bandwidth Details",
    description=(
        "Per-memcpy event details including bytes transferred and achieved bandwidth. "
        "Use to verify claims about transfer sizes and efficiency."
    ),
    sql=(
        "SELECT s.name AS direction, "
        "ROUND(s.dur / 1000.0, 2) AS dur_us, "
        "a_bytes.int_value AS bytes, "
        "ROUND(CAST(a_bytes.int_value AS REAL) / s.dur, 2) AS achieved_bw_gbps "
        "FROM slice s "
        "LEFT JOIN args a_bytes ON s.arg_set_id = a_bytes.arg_set_id "
        "AND a_bytes.flat_key = 'args.bytes' "
        "WHERE s.cat = 'gpu_memcpy' "
        "ORDER BY s.dur DESC"
    ),
)

KERNEL_DISPATCH_DURATIONS = PrebuiltQuery(
    label="Kernel Dispatch Durations",
    description="Per-dispatch duration and timestamp for all GPU kernels (for CV analysis).",
    internal=True,
    sql=(
        "SELECT name, dur, ts "
        "FROM slice "
        "WHERE cat = 'kernel' "
        "ORDER BY ts"
    ),
)


DEFAULT_QUERIES: tuple[PrebuiltQuery, ...] = (
    TOP_SLOWEST_KERNELS,
    AGGREGATE_KERNEL_TIME,
    TOP_CPU_OP_BOTTLENECKS,
    CATEGORY_BREAKDOWN,
    TRACE_TIME_SUMMARY,
    CUDA_RUNTIME_SUMMARY,
    PYTHON_MODULE_SUMMARY,
    MEMCPY_BANDWIDTH,
    CPU_OP_EXTID,
    GPU_KERNEL_EXTID,
    CPU_OP_INPUT_DIMS,
    CPU_OP_ID_NAMES,
    CPU_OP_INPUT_TYPES,
)

PROFILING_QUERIES: tuple[PrebuiltQuery, ...] = (
    AGGREGATE_KERNEL_TIME,
    CPU_OP_INPUT_TYPES,
    CPU_OP_INPUT_TYPES_BY_ID,
    KERNEL_DISPATCH_DURATIONS,
    CPU_OP_EXTID,
    GPU_KERNEL_EXTID,
    CPU_OP_INPUT_DIMS,
    CPU_OP_ID_NAMES,
    PYTHON_MODULE_SUMMARY,
)


def get_default_queries() -> tuple[PrebuiltQuery, ...]:
    """Return the default set of pre-built SQL queries."""
    return DEFAULT_QUERIES


def get_profiling_queries() -> tuple[PrebuiltQuery, ...]:
    """Return the minimal query set for the profiling pipeline."""
    return PROFILING_QUERIES


GEMM_OPS = frozenset({
    "aten::bmm", "aten::addmm", "aten::mm", "aten::_scaled_mm", "aten::baddbmm",
})


def build_cpu_op_gpu_time(
    cpu_ext: "QueryResult", gpu_ext: "QueryResult",
) -> "QueryResult":
    """Build 'CPU Op GPU Kernel Time' from simple External id queries in Python."""
    from profilelens.dtypes import QueryResult

    ext_to_gpu: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in (gpu_ext.rows or ()):
        ext_to_gpu[row["ext_id"]].append((row["id"], row["dur"] or 0))

    op_stats: dict[str, dict] = {}
    for row in (cpu_ext.rows or ()):
        eid = row["ext_id"]
        kernels = ext_to_gpu.get(eid, [])
        if not kernels:
            continue
        name = row["name"]
        if name not in op_stats:
            op_stats[name] = {"inv_ids": set(), "gpu_ids": set(), "gpu_ns": 0}
        s = op_stats[name]
        s["inv_ids"].add(row["id"])
        for kid, dur in kernels:
            s["gpu_ids"].add(kid)
            s["gpu_ns"] += dur

    rows = []
    for name, s in sorted(op_stats.items(), key=lambda x: -x[1]["gpu_ns"]):
        rows.append({
            "cpu_op": name,
            "invocations": len(s["inv_ids"]),
            "gpu_events": len(s["gpu_ids"]),
            "total_gpu_ms": round(s["gpu_ns"] / 1_000_000, 2),
        })

    return QueryResult(
        sql="(Python join on External id)",
        label="CPU Op GPU Kernel Time",
        rows=tuple(rows),
        columns=("cpu_op", "invocations", "gpu_events", "total_gpu_ms"),
    )


def build_shape_breakdowns(
    cpu_ext: "QueryResult", gpu_ext: "QueryResult", dims: "QueryResult",
) -> tuple["QueryResult", "QueryResult"]:
    """Build GEMM Shape Breakdown and Op Shape Breakdown from simple queries."""
    from profilelens.dtypes import QueryResult
    import re

    ext_to_gpu_dur: dict[int, int] = defaultdict(int)
    ext_to_gpu_ids: dict[int, set] = defaultdict(set)
    for row in (gpu_ext.rows or ()):
        ext_to_gpu_dur[row["ext_id"]] += row["dur"] or 0
        ext_to_gpu_ids[row["ext_id"]].add(row["id"])

    slice_dims: dict[int, dict[str, int]] = defaultdict(dict)
    for row in (dims.rows or ()):
        slice_dims[row["slice_id"]][row["key"]] = row["int_value"]

    _dim_re = re.compile(r"args\.Input Dims\[(\d+)\]\[(\d+)\]")

    gemm_groups: dict[tuple, dict] = {}
    op_groups: dict[tuple, dict] = {}

    for row in (cpu_ext.rows or ()):
        eid = row["ext_id"]
        gpu_ns = ext_to_gpu_dur.get(eid, 0)
        if gpu_ns == 0:
            continue
        name = row["name"]
        sd = slice_dims.get(row["id"], {})

        if name in GEMM_OPS:
            M = sd.get("args.Input Dims[1][0]")
            K = sd.get("args.Input Dims[1][1]")
            N = sd.get("args.Input Dims[2][1]")
            key = (name, M, K, N)
            if key not in gemm_groups:
                gemm_groups[key] = {"inv_ids": set(), "gpu_ids": set(), "gpu_ns": 0}
            g = gemm_groups[key]
            g["inv_ids"].add(row["id"])
            g["gpu_ids"] |= ext_to_gpu_ids.get(eid, set())
            g["gpu_ns"] += gpu_ns
        else:
            d0 = sd.get("args.Input Dims[0][0]")
            d1 = sd.get("args.Input Dims[0][1]")
            d2 = sd.get("args.Input Dims[0][2]")
            w0 = sd.get("args.Input Dims[1][0]") if "conv" in name.lower() else None
            w1 = sd.get("args.Input Dims[1][1]") if "conv" in name.lower() else None
            w2 = sd.get("args.Input Dims[1][2]") if "conv" in name.lower() else None
            w3 = sd.get("args.Input Dims[1][3]") if "conv" in name.lower() else None
            key = (name, d0, d1, d2, w0, w1, w2, w3)
            if key not in op_groups:
                op_groups[key] = {"inv_ids": set(), "gpu_ns": 0}
            g = op_groups[key]
            g["inv_ids"].add(row["id"])
            g["gpu_ns"] += gpu_ns

    gemm_rows = []
    for (name, M, K, N), g in sorted(gemm_groups.items(), key=lambda x: -x[1]["gpu_ns"]):
        gemm_rows.append({
            "cpu_op": name, "M": M, "K": K, "N": N,
            "invocations": len(g["inv_ids"]),
            "gpu_events": len(g["gpu_ids"]),
            "total_gpu_ms": round(g["gpu_ns"] / 1_000_000, 4),
        })

    op_rows = []
    for (name, d0, d1, d2, w0, w1, w2, w3), g in sorted(op_groups.items(), key=lambda x: -x[1]["gpu_ns"]):
        row = {
            "cpu_op": name, "dim0": d0, "dim1": d1, "dim2": d2,
            "invocations": len(g["inv_ids"]),
            "total_gpu_ms": round(g["gpu_ns"] / 1_000_000, 4),
        }
        if any(v is not None for v in (w0, w1, w2, w3)):
            row["weight"] = f"({w0},{w1},{w2},{w3})"
        op_rows.append(row)

    return (
        QueryResult(
            sql="(Python shape grouping)",
            label="GEMM Shape Breakdown",
            rows=tuple(gemm_rows),
            columns=("cpu_op", "M", "K", "N", "invocations", "gpu_events", "total_gpu_ms"),
        ),
        QueryResult(
            sql="(Python shape grouping)",
            label="Op Shape Breakdown",
            rows=tuple(op_rows),
            columns=("cpu_op", "dim0", "dim1", "dim2", "invocations", "total_gpu_ms", "weight"),
        ),
    )


def build_category_summary(
    parent_chain_result: "QueryResult",
    id_names_result: "QueryResult",
) -> "QueryResult":
    """Attribute GPU time to each op's outermost categorized ancestor (TraceLens style)."""
    from profilelens.dtypes import QueryResult
    from profilelens.trace.categories import categorize_op, display_category

    if parent_chain_result.error or not parent_chain_result.rows:
        return QueryResult(
            sql="(post-processed)", label="Compute Category Summary",
            error=parent_chain_result.error or "No parent chain data",
        )

    id_to_name: dict[int, str] = {}
    id_to_parent: dict[int, int | None] = {}
    for row in (id_names_result.rows if id_names_result.rows else ()):
        id_to_name[row["id"]] = row["name"]
        id_to_parent[row["id"]] = row.get("parent_id")

    for row in parent_chain_result.rows:
        id_to_name[row["cpu_id"]] = row["cpu_op"]
        if row["cpu_id"] not in id_to_parent:
            id_to_parent[row["cpu_id"]] = row.get("parent_id")

    def _find_outermost_categorized(cpu_id: int, cpu_op: str) -> tuple[str, str]:
        best_name = cpu_op
        best_cat = categorize_op(cpu_op)

        pid = id_to_parent.get(cpu_id)
        visited: set[int] = {cpu_id}
        while pid is not None and pid not in visited:
            visited.add(pid)
            pname = id_to_name.get(pid)
            if pname is None:
                break
            pcat = categorize_op(pname)
            if pcat != "other":
                best_name = pname
                best_cat = pcat
            pid = id_to_parent.get(pid)
        return best_name, best_cat

    cat_gpu_ms: dict[str, float] = defaultdict(float)
    cat_ops: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))

    for row in parent_chain_result.rows:
        gpu_ms = row.get("gpu_ms") or 0.0
        attr_name, raw_cat = _find_outermost_categorized(row["cpu_id"], row["cpu_op"])
        _, disp = display_category(attr_name, raw_cat)
        cat_gpu_ms[disp] += gpu_ms
        outermost_id = row["cpu_id"]
        pid = id_to_parent.get(row["cpu_id"])
        visited: set[int] = {row["cpu_id"]}
        while pid is not None and pid not in visited:
            visited.add(pid)
            pname = id_to_name.get(pid)
            if pname == attr_name:
                outermost_id = pid
            pid = id_to_parent.get(pid)
        cat_ops[disp][attr_name].add(outermost_id)

    summary_rows = []
    for cat_name in sorted(cat_gpu_ms, key=lambda c: -cat_gpu_ms[c]):
        ops_dict = cat_ops[cat_name]
        ops_str = ", ".join(
            f"{op}({len(ids)})" for op, ids in
            sorted(ops_dict.items(), key=lambda x: -len(x[1]))
        )
        summary_rows.append({
            "category": cat_name,
            "total_gpu_ms": round(cat_gpu_ms[cat_name], 3),
            "op_count": sum(len(ids) for ids in ops_dict.values()),
            "unique_ops": len(ops_dict),
            "ops_included": ops_str,
        })

    elem_parts = ["UnaryElementwise", "BinaryElementwise", "Elementwise"]
    elem_ms = sum(cat_gpu_ms.get(c, 0.0) for c in elem_parts)
    if elem_ms > 0:
        elem_ops: dict[str, set[int]] = {}
        for c in elem_parts:
            for op, ids in cat_ops.get(c, {}).items():
                if op not in elem_ops:
                    elem_ops[op] = set()
                elem_ops[op] |= ids
        elem_ops_str = ", ".join(
            f"{op}({len(ids)})" for op, ids in
            sorted(elem_ops.items(), key=lambda x: -len(x[1]))
        )
        summary_rows.append({
            "category": "Elementwise (combined)",
            "total_gpu_ms": round(elem_ms, 3),
            "op_count": sum(len(ids) for ids in elem_ops.values()),
            "ops_included": elem_ops_str,
        })

    return QueryResult(
        sql="(post-processed from CPU Op Parent Chain using TraceLens categories)",
        label="Compute Category Summary",
        rows=tuple(summary_rows),
        columns=("category", "total_gpu_ms", "op_count", "ops_included"),
    )
