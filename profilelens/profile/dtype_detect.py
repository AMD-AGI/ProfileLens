# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Detect dominant compute dtype from trace analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UNKNOWN_DTYPE = "UNKNOWN"
DEFAULT_COUNTER_DTYPES = ("BF16", "FP16", "FP8")

_PYTORCH_TO_INTERNAL: dict[str, str] = {
    "BFloat16": "BF16",
    "torch.bfloat16": "BF16",
    "c10::BFloat16": "BF16",
    "Half": "FP16",
    "Float16": "FP16",
    "torch.float16": "FP16",
    "c10::Half": "FP16",
    "Float8_e4m3fn": "FP8",
    "Float8_e4m3fnuz": "FP8",
    "Float8_e5m2": "FP8",
    "Float8_e5m2fnuz": "FP8",
    "torch.float8_e4m3fn": "FP8",
    "torch.float8_e5m2": "FP8",
    "c10::Float8_e4m3fn": "FP8",
    "c10::Float8_e4m3fnuz": "FP8",
    "c10::Float8_e5m2": "FP8",
    "c10::Float8_e5m2fnuz": "FP8",
    "Float": "FP32",
    "Float32": "FP32",
    "torch.float32": "FP32",
    "c10::Float": "FP32",
    "Double": "FP64",
    "Float64": "FP64",
    "torch.float64": "FP64",
    "c10::Double": "FP64",
    "Int": "INT32",
    "Int32": "INT32",
    "Long": "INT64",
    "Int64": "INT64",
    "Byte": "INT8",
    "Int8": "INT8",
    "Char": "INT8",
}

_COMPUTE_DTYPES = {"BF16", "FP16", "FP8", "FP32", "FP64"}

_DTYPE_COUNTER_ALIAS: dict[str, str] = {
    "FP4": "F6F4", "MXFP4": "F6F4", "NVFP4": "F6F4",
    "FP6": "F6F4", "MXFP6": "F6F4", "MXFP8": "FP8",
    "F8": "FP8", "F16": "FP16", "F32": "FP32", "F64": "FP64",
    "INT4": "INT8", "NF4": "FP8",
}

_MFMA_COUNTER_FOR_DTYPE: dict[str, str] = {
    "BF16": "SQ_INSTS_VALU_MFMA_MOPS_BF16",
    "FP16": "SQ_INSTS_VALU_MFMA_MOPS_F16",
    "FP8": "SQ_INSTS_VALU_MFMA_MOPS_F8",
    "F6F4": "SQ_INSTS_VALU_MFMA_MOPS_F6F4",
    "FP32": "SQ_INSTS_VALU_MFMA_MOPS_F32",
    "FP64": "SQ_INSTS_VALU_MFMA_MOPS_F64",
    "INT8": "SQ_INSTS_VALU_MFMA_MOPS_I8",
}


def normalize_dtype(pytorch_dtype: str) -> str:
    """Map a PyTorch dtype string to an internal short name."""
    return _PYTORCH_TO_INTERNAL.get(pytorch_dtype, pytorch_dtype.upper())


def counter_dtype(dtype: str) -> str:
    """Fold a precise dtype label to the one with a readable MFMA counter."""
    dt = dtype.upper()
    return _DTYPE_COUNTER_ALIAS.get(dt, dt)


def mfma_counter_for_dtype(dtype: str) -> str | None:
    """Return the MFMA counter name for a given internal dtype."""
    return _MFMA_COUNTER_FOR_DTYPE.get(counter_dtype(dtype))


def normalize_counter_dtype_hints(dtypes: list[str] | tuple[str, ...]) -> list[str]:
    """Validate caller hints without assigning them to individual kernels."""
    normalized = list(dict.fromkeys(normalize_dtype(dtype.strip()) for dtype in dtypes))
    invalid = [dtype for dtype in normalized if mfma_counter_for_dtype(dtype) is None]
    if invalid:
        raise ValueError("Unsupported counter dtype hint(s): " + ", ".join(invalid))
    return normalized


def counter_dtype_coverage(
    observed_dtypes: list[str] | tuple[str, ...],
    unresolved: bool,
    hints: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """Choose MFMA counter families; this is not kernel dtype inference."""
    known = [dtype for dtype in observed_dtypes if mfma_counter_for_dtype(dtype)]
    requested = normalize_counter_dtype_hints(hints)
    fallback = DEFAULT_COUNTER_DTYPES if not requested and (unresolved or not known) else ()
    return list(dict.fromkeys([*known, *requested, *fallback]))


def mfma_counters_for_dtypes(dtypes: "set[str] | tuple[str, ...] | list[str]") -> list[str]:
    """Return the de-duplicated MFMA-MOPS counters covering all given dtypes."""
    out: dict[str, None] = {}
    for dt in dtypes:
        ctr = _MFMA_COUNTER_FOR_DTYPE.get(counter_dtype(dt))
        if ctr:
            out[ctr] = None
    return sorted(out.keys())


_KERNEL_NAME_DTYPE_TOKENS: tuple[tuple[str, str], ...] = (
    ("bfloat16", "BF16"), ("bf16", "BF16"), ("_bbs_", "BF16"), ("hip_bfloat16", "BF16"),
    ("float16", "FP16"), ("__half", "FP16"), ("fp16", "FP16"), ("_hhs_", "FP16"),
    ("e4m3", "FP8"), ("e5m2", "FP8"), ("float8", "FP8"), ("fp8", "FP8"),
    ("a8w8", "FP8"), ("a8w4", "FP8"), ("w8a8", "FP8"), ("_f8_", "FP8"),
    ("mxfp4", "MXFP4"), ("nvfp4", "NVFP4"), ("fp4", "FP4"), ("w4a8", "FP8"),
    ("int8", "INT8"), ("_i8_", "INT8"), ("int4", "INT4"),
    ("float32", "FP32"), ("_sss_", "FP32"), ("fp32", "FP32"),
    ("float64", "FP64"), ("fp64", "FP64"),
)


def dtype_from_kernel_name(kernel_name: str) -> str | None:
    """Infer a compute dtype from tokens in the kernel symbol."""
    lower = kernel_name.lower()
    for token, dtype in _KERNEL_NAME_DTYPE_TOKENS:
        if token in lower:
            return dtype
    return None


_TORCH_DTYPE_TO_INTERNAL: dict[str, str] = {
    "torch.bfloat16": "BF16",
    "torch.float16": "FP16", "torch.half": "FP16",
    "torch.float8_e4m3fn": "FP8", "torch.float8_e4m3fnuz": "FP8",
    "torch.float8_e5m2": "FP8", "torch.float8_e5m2fnuz": "FP8",
    "torch.float32": "FP32", "torch.float": "FP32",
    "torch.float64": "FP64", "torch.double": "FP64",
}


def torch_dtype_to_internal(dtype_str: str) -> str | None:
    """Map ``str(tensor.dtype)`` to an internal compute dtype, else ``None``."""
    return _TORCH_DTYPE_TO_INTERNAL.get(dtype_str)


def dtype_from_query_result(rows: tuple[dict[str, Any], ...] | None) -> str:
    """Pick the dominant compute dtype from CPU_OP_INPUT_TYPES query rows."""
    if not rows:
        return UNKNOWN_DTYPE

    best_dtype = UNKNOWN_DTYPE
    best_count = 0
    for row in rows:
        raw = row.get("dtype", "")
        internal = normalize_dtype(raw)
        if internal in _COMPUTE_DTYPES:
            cnt = row.get("cnt", 0)
            if cnt > best_count:
                best_count = cnt
                best_dtype = internal
    return best_dtype


_QUANT_KERNEL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("mxfp4", "MXFP4"),
    ("mxfp8", "MXFP8"),
    ("mxfp6", "MXFP6"),
    ("nvfp4", "NVFP4"),
    ("fp4", "FP4"),
    ("fp8", "FP8"),
    ("int4", "INT4"),
    ("int8", "INT8"),
    ("nf4", "NF4"),
)


def detect_quantization_from_kernels(kernel_names: list[str] | tuple[str, ...]) -> list[str]:
    """Get quantization formats from kernel names in the trace."""
    found: dict[str, None] = {}
    for raw in kernel_names:
        lower = raw.lower()
        for needle, label in _QUANT_KERNEL_PATTERNS:
            if needle in lower and label not in found:
                found[label] = None
    return list(found.keys())


def merge_dtype_sidecar(sidecar_dir: str | Path) -> dict[str, Any]:
    """Merge all per-PID dtype JSONs written by the hook sidecar."""
    merged: dict[str, Any] = {
        "triton": {}, "triton_shape_counts": {}, "capture_ops": {},
        "capture_dtypes": {}, "capture_graphs": {},
        "flydsl": {}, "flydsl_shapes": {}, "flydsl_args": {},
        "flydsl_shape_counts": {},
    }
    d = Path(sidecar_dir)
    if not d.is_dir():
        return merged

    def _accum(dst: dict[str, dict[str, int]], src: dict[str, Any]) -> None:
        for name, counts in (src or {}).items():
            bucket = dst.setdefault(name, {})
            for dt, cnt in (counts or {}).items():
                bucket[dt] = bucket.get(dt, 0) + int(cnt)

    for f in d.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        _accum(merged["triton"], data.get("triton", {}))
        _accum(merged["flydsl"], data.get("flydsl", {}))
        _accum(merged["capture_ops"], data.get("capture_ops", {}))
        for dt, cnt in (data.get("capture_dtypes", {}) or {}).items():
            merged["capture_dtypes"][dt] = merged["capture_dtypes"].get(dt, 0) + int(cnt)
        for gid, g in (data.get("capture_graphs", {}) or {}).items():
            if not isinstance(g, dict) or not g.get("ops"):
                continue
            dst = merged["capture_graphs"].setdefault(
                gid, {"ops": g.get("ops") or [], "replays": 0, "pid": data.get("pid")}
            )
            dst["replays"] += int(g.get("replays", 0) or 0)
        for stem, shapes in (data.get("flydsl_shapes", {}) or {}).items():
            if shapes and not merged["flydsl_shapes"].get(stem):
                merged["flydsl_shapes"][stem] = shapes
        _merge_flydsl_args(merged["flydsl_args"], data.get("flydsl_args", {}))
        for stem, sc in (data.get("flydsl_shape_counts", {}) or {}).items():
            dst_sc = merged["flydsl_shape_counts"].setdefault(stem, {})
            for sig, cnt in (sc or {}).items():
                dst_sc[sig] = dst_sc.get(sig, 0) + int(cnt)
        for name, sc in (data.get("triton_shape_counts", {}) or {}).items():
            dst_sc = merged["triton_shape_counts"].setdefault(name, {})
            for sig, cnt in (sc or {}).items():
                dst_sc[sig] = dst_sc.get(sig, 0) + int(cnt)
    return merged


def _merge_flydsl_args(
    dst: dict[str, dict[str, Any]], src: dict[str, Any] | None
) -> None:
    """Accumulate one PID's ``flydsl_args`` into the cross-PID ``dst``."""
    for stem, args in (src or {}).items():
        dst_stem = dst.setdefault(stem, {})
        for name, meta in (args or {}).items():
            if not isinstance(meta, dict):
                continue
            entry = dst_stem.setdefault(
                name, {"dtype": meta.get("dtype", "?"), "shapes": [], "count": 0}
            )
            entry["count"] += int(meta.get("count", 0) or 0)
            for shp in (meta.get("shapes") or []):
                if shp not in entry["shapes"]:
                    entry["shapes"].append(shp)


_PRECISION_RANK: dict[str, int] = {
    "MXFP4": -2, "FP4": -2, "NVFP4": -2, "NF4": -2,
    "INT4": -1, "FP8": 0, "MXFP8": 0, "INT8": 0,
    "FP16": 1, "BF16": 2, "FP32": 3, "FP64": 4,
}


def _argmax_dtype(counts: dict[str, int]) -> str | None:
    """Return the most frequently observed dtype."""
    best, best_n = None, -1
    for dt, n in counts.items():
        if n > best_n:
            best, best_n = dt, n
    return best


def _lowest_precision_dtype(counts: dict[str, int]) -> str | None:
    """Pick the lowest-precision dtype in a count bucket."""
    best, best_key = None, None
    for dt, n in counts.items():
        if n <= 0:
            continue
        key = (_PRECISION_RANK.get(dt, 5), -n)  # lower rank wins, then more frequent
        if best_key is None or key < best_key:
            best, best_key = dt, key
    return best


def _triton_lookup(kernel_name: str, triton: dict[str, dict[str, int]]) -> str | None:
    """Match a GPU kernel symbol to a recorded Triton launch and pick its dtype."""
    if kernel_name in triton:
        return _lowest_precision_dtype(triton[kernel_name])
    for tname, counts in triton.items():
        if not tname:
            continue
        if kernel_name.startswith(tname) or tname in kernel_name:
            return _lowest_precision_dtype(counts)
    return None


def _flydsl_stem(name: str | None) -> str | None:
    """Normalize a FlyDSL launcher or kernel symbol to its stem."""
    if not name:
        return None
    i = len(name)
    while i > 0 and name[i - 1].isdigit():
        i -= 1
    if 0 < i < len(name) and name[i - 1] == "_":
        name = name[: i - 1]
    for p in ("kn_", "launch_"):
        if name.startswith(p):
            name = name[len(p):]
            break
    return name or None


def _flydsl_lookup(kernel_name: str, flydsl: dict[str, dict[str, int]]) -> str | None:
    """Match a GPU kernel symbol to a recorded FlyDSL launch by normalized stem."""
    if not flydsl:
        return None
    stem = _flydsl_stem(kernel_name)
    if stem and stem in flydsl:
        return _lowest_precision_dtype(flydsl[stem])
    for fname, counts in flydsl.items():
        if fname and stem and (stem in fname or fname in stem):
            return _lowest_precision_dtype(counts)
    return None


def _flydsl_args_lookup(
    kernel_name: str, flydsl_args: dict[str, dict[str, Any]] | None
) -> dict[str, Any] | None:
    """Match a GPU kernel symbol to its recorded per-arg metadata by stem."""
    if not flydsl_args:
        return None
    stem = _flydsl_stem(kernel_name)
    if stem and stem in flydsl_args:
        return flydsl_args[stem]
    for fname, arg_map in flydsl_args.items():
        if fname and stem and (stem in fname or fname in stem):
            return arg_map
    return None


def _fmt_shape(shape: "list[int] | tuple[int, ...]") -> str:
    """Render one tensor shape as ``[a,b,c]`` (empty list -> ``[]``)."""
    return "[" + ",".join(str(int(d)) for d in shape) + "]"


def format_flydsl_arg_shapes(arg_map: dict[str, Any] | None) -> str:
    """Render FlyDSL per-arg tensor shapes trace-style: ``[a,b]x[c,d]x…``."""
    if not arg_map:
        return ""
    parts: list[str] = []
    for meta in arg_map.values():
        shapes = (meta or {}).get("shapes") or []
        if not shapes:
            continue
        parts.append("|".join(_fmt_shape(s) for s in shapes))
    return "x".join(parts)


def _entry_kernel_names(entry: dict[str, Any] | None) -> list[str]:
    """Exact GPU symbols claimed by one capture-inventory operation."""
    names: list[str] = []
    for node in (entry or {}).get("kernels") or []:
        name = (node or {}).get("name", "") if isinstance(node, dict) else str(node)
        if name:
            names.append(name)
    return names


def _canonical_graph_kernel_name(name: str) -> str:
    name = (name or "").strip()
    return name[5:] if name.startswith("void ") else name


def _same_graph_kernel_name(left: str, right: str) -> bool:
    return _canonical_graph_kernel_name(left) == _canonical_graph_kernel_name(right)


def _entry_exact_kernel_count(entry: dict[str, Any] | None, kernel_name: str) -> int:
    return sum(
        _same_graph_kernel_name(name, kernel_name)
        for name in _entry_kernel_names(entry)
    )


def graph_stem_dtypes(
    capture_graphs: dict[str, Any] | None,
) -> dict[str, dict[str, int]]:
    """Map captured kernel symbols or launcher stems to dtype counts."""
    out: dict[str, dict[str, int]] = {}
    for g in (capture_graphs or {}).values():
        replays = int((g or {}).get("replays", 0) or 0)
        if replays <= 0:
            continue
        for entry in (g or {}).get("ops") or []:
            stem = (entry or {}).get("stem") or ""
            exact = [_canonical_graph_kernel_name(n) for n in _entry_kernel_names(entry)]
            owners = list(dict.fromkeys(exact + ([stem] if stem else [])))
            if not owners:
                owners = [""]
            for owner in owners:
                bucket = out.setdefault(owner, {})
                for dt, cnt in ((entry or {}).get("dtypes") or {}).items():
                    bucket[dt] = bucket.get(dt, 0) + int(cnt) * replays
    return out


def _graph_lookup(
    kernel_name: str, stem_dtypes: dict[str, dict[str, int]]
) -> str | None:
    """Match a GPU kernel symbol to a graph-captured launch and pick its dtype."""
    if not stem_dtypes:
        return None
    canonical = _canonical_graph_kernel_name(kernel_name)
    if canonical in stem_dtypes:
        return _lowest_precision_dtype(stem_dtypes[canonical])
    for stem, counts in stem_dtypes.items():
        if stem and (kernel_name.startswith(stem) or stem in kernel_name):
            return _lowest_precision_dtype(counts)
    return None


def _graph_entries(capture_graphs: dict[str, Any] | None):
    """Yield every ``(gid, entry)`` in the capture inventory, in capture order."""
    for gid in sorted(capture_graphs or {}):
        for entry in ((capture_graphs or {})[gid] or {}).get("ops") or []:
            if entry:
                yield gid, entry


def _aten_op_tokens(op: str) -> list[str]:
    """Return safe symbol-match tokens for an aten operation."""
    name = (op or "").split("::", 1)[-1].lower()
    if len(name) < 4:
        return []
    toks = [name]
    squashed = name.replace("_", "")
    if squashed != name and len(squashed) >= 4:
        toks.append(squashed)
    return toks


def graph_aten_dtypes(
    capture_graphs: dict[str, Any] | None,
) -> dict[str, dict[str, int]]:
    """Map stemless captured aten operations to dtype counts."""
    out: dict[str, dict[str, int]] = {}
    for g in (capture_graphs or {}).values():
        replays = int((g or {}).get("replays", 0) or 0)
        if replays <= 0:
            continue
        for entry in (g or {}).get("ops") or []:
            if (entry or {}).get("stem"):
                continue
            dtypes = (entry or {}).get("dtypes") or {}
            for tok in _aten_op_tokens((entry or {}).get("op", "")):
                bucket = out.setdefault(tok, {})
                for dt, cnt in dtypes.items():
                    bucket[dt] = bucket.get(dt, 0) + int(cnt) * replays
    return out


def _graph_aten_lookup(
    kernel_name: str, aten_dtypes: dict[str, dict[str, int]]
) -> str | None:
    """Match a GPU symbol to a stemless (aten) capture entry by op name."""
    if not aten_dtypes:
        return None
    low = kernel_name.lower()
    squashed = low.replace("_", "")
    # Longest token first.
    for tok in sorted(aten_dtypes, key=len, reverse=True):
        if tok in low or tok in squashed:
            return _lowest_precision_dtype(aten_dtypes[tok])
    return None


def graph_gid_for_sig(
    sig: str,
    capture_graphs: dict[str, Any] | None,
    kernel_name: str = "",
) -> str:
    """Return the unique graph owning a symbol and shape signature."""
    if not sig:
        return ""
    matches = [
        (gid, entry) for gid, entry in _graph_entries(capture_graphs)
        if entry.get("sig") == sig
    ]
    if kernel_name:
        exact = [
            (gid, entry) for gid, entry in matches
            if _entry_exact_kernel_count(entry, kernel_name)
        ]
        if exact:
            matches = exact
        else:
            matches = [
                (gid, entry) for gid, entry in matches
                if entry.get("stem") and entry.get("stem") in kernel_name
            ]
    owners = {gid for gid, _entry in matches}
    return owners.pop() if len(owners) == 1 else ""


def graph_op_for_kernel(
    kernel_name: str,
    capture_graphs: dict[str, Any] | None,
    prefer_gid: str = "",
    prefer_sig: str = "",
) -> str:
    """The unambiguous capture op that owns a GPU kernel symbol."""
    entries = [
        (gid, entry) for gid, entry in _graph_entries(capture_graphs)
        if (not prefer_gid or gid == prefer_gid)
        and (not prefer_sig or entry.get("sig") == prefer_sig)
    ]
    exact_ops = {
        entry.get("op", "") for _gid, entry in entries
        if _entry_exact_kernel_count(entry, kernel_name) and entry.get("op")
    }
    if len(exact_ops) == 1:
        return exact_ops.pop()
    if len(exact_ops) > 1:
        return ""
    launcher_ops: set[str] = set()
    for _gid, entry in entries:
        stem = entry.get("stem") or ""
        if stem and (kernel_name == stem or kernel_name.startswith(stem)
                     or stem in kernel_name):
            if entry.get("op"):
                launcher_ops.add(entry["op"])
    if len(launcher_ops) == 1:
        return launcher_ops.pop()
    return ""


def flydsl_stem_collisions(
    kernel_names: "list[str] | tuple[str, ...]",
    flydsl_args: dict[str, dict[str, Any]] | None,
) -> list[str]:
    """Warn when one FlyDSL stem matches several kernel symbols."""
    if not flydsl_args:
        return []
    warnings: list[str] = []
    symbols_per_stem: dict[str, list[str]] = {}
    for name in kernel_names:
        arg_map = _flydsl_args_lookup(name, flydsl_args)
        if arg_map is None:
            continue
        stem = _flydsl_stem(name)
        key = stem if (stem and stem in flydsl_args) else next(
            (k for k in flydsl_args if k and stem and (stem in k or k in stem)), None
        )
        if key is None:
            continue
        symbols_per_stem.setdefault(key, []).append(name)
    for stem, syms in symbols_per_stem.items():
        if len(set(syms)) > 1:
            warnings.append(
                f"FlyDSL stem '{stem}' matched {len(set(syms))} trace symbols "
                f"({', '.join(sorted(set(syms)))}); per-arg shapes may be ambiguous."
            )
    return warnings


def resolve_kernel_dtypes(
    kernel_names: "list[str] | tuple[str, ...]",
    sidecar: dict[str, Any] | None,
    trace_dtypes: dict[str, str] | None = None,
    global_dtype: str | None = None,
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Resolve each kernel's dtype from kernel-specific evidence only."""
    sidecar = sidecar or {}
    triton = sidecar.get("triton", {})
    flydsl = sidecar.get("flydsl", {})
    graph_stems = graph_stem_dtypes(sidecar.get("capture_graphs"))
    graph_atens = graph_aten_dtypes(sidecar.get("capture_graphs"))
    trace_dtypes = trace_dtypes or {}

    per_kernel: dict[str, tuple[str, str]] = {}
    observed: dict[str, None] = {}

    for name in kernel_names:
        dt = _triton_lookup(name, triton)
        src = "triton_hook"
        if dt is None:
            dt = _flydsl_lookup(name, flydsl)
            if dt is not None:
                src = "flydsl_hook"
        if dt is None:
            dt = _graph_lookup(name, graph_stems)
            if dt is None:
                dt = _graph_aten_lookup(name, graph_atens)
            if dt is not None:
                src = "graph_capture"
        if dt is None and name in trace_dtypes:
            dt, src = trace_dtypes[name], "trace_corr"
        if dt is None:
            sniffed = dtype_from_kernel_name(name)
            if sniffed is not None:
                dt, src = sniffed, "name_fallback"
        if dt is None:
            dt, src = UNKNOWN_DTYPE, "unresolved"
        per_kernel[name] = (dt, src)
        observed[dt] = None

    return per_kernel, sorted(observed.keys())


def dtype_from_analysis_json(path: str | Path) -> str:
    """Extract dominant dtype from a saved analyze JSON output."""
    p = Path(path)
    if not p.exists():
        return UNKNOWN_DTYPE

    try:
        with open(p) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return UNKNOWN_DTYPE

    if isinstance(data, list):
        for entry in data:
            if entry.get("label") == "CPU Op Input Types":
                rows = entry.get("rows", [])
                return dtype_from_query_result(tuple(rows))

    return UNKNOWN_DTYPE
