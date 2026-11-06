# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare same-pass capture, then analyze immutable window data without replay."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from profilelens.dtypes import ProfilingResult, RooflinePeaks
from profilelens.profile.dtype_detect import UNKNOWN_DTYPE, normalize_dtype
from profilelens.profile.gpu_detect import GpuInfo
from profilelens.profile.kernel_names import (
    add_canonical_aliases,
    boundary_kernel_name,
    canonical_kernel_name,
    lookup_kernel,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device_identity(rocprofv3_path: str, device_index: int) -> dict[str, str]:
    """Record SDK node identity so captured dispatches use the measured GPU roof."""
    from profilelens.profile.rocprofv3 import find_rocprofv3_avail

    result = subprocess.run(
        [find_rocprofv3_avail(rocprofv3_path), "-d", str(device_index), "list", "--agent"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError("could not query capture GPU identity: " + result.stderr.strip())
    fields = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip().lower()] = value.strip()
    identity = {key: fields.get(key, "") for key in ("node_id", "gpu_id")}
    if not all(value.isdigit() for value in identity.values()):
        raise RuntimeError("SDK did not report node_id and gpu_id for capture GPU")
    return identity


def prepare_capture(
    output_dir: str | Path,
    *,
    rocprofv3_path: str = "",
    dtype: str | None = None,
    observed_dtypes: tuple[str, ...] | list[str] = (),
    counter_dtype_hints: tuple[str, ...] | list[str] = (),
    run_roofline: bool = True,
    roofline_cache_dir: str | None = None,
    counter_set_mode: str = "auto",
    custom_counters: tuple[str, ...] | list[str] | None = None,
    device_index: int = 0,
) -> dict[str, Any]:
    """Prepare a fixed counter set and worker environment for a caller-run capture."""
    from profilelens.profile.gpu_detect import detect_gpu
    from profilelens.profile.planning import prepare_counter_plan, prepare_roofline
    from profilelens.profile.rocprofv3 import find_rocprofv3, prepare_patched_rocprofv3

    if not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a nonnegative logical GPU index")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "capture_metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"capture already prepared: {metadata_path}")
    rocprof = find_rocprofv3(rocprofv3_path)
    rocm_root, _ = prepare_patched_rocprofv3(rocprof, str(out))
    if rocm_root:
        raise RuntimeError("same-pass capture requires native SDK 1.4.1 selected regions")
    gpu = detect_gpu(device_index=device_index, rocprofv3_path=rocprof)
    identity = _device_identity(rocprof, device_index)
    hints = list(
        dict.fromkeys(
            [
                *observed_dtypes,
                *counter_dtype_hints,
                *([dtype] if dtype else []),
            ]
        )
    )
    plan = prepare_counter_plan(
        rocprof,
        has_unresolved=True,
        counter_dtype_hints=hints,
        custom_counters=custom_counters,
        counter_set_mode=counter_set_mode,
        device_index=device_index,
    )
    cset = plan.counter_set
    if "GRBM_COUNT" not in cset.counters:
        raise ValueError("same-pass capture requires GRBM_COUNT to validate counted dispatches")
    prepared_roof = None
    if run_roofline:
        prepared_roof = prepare_roofline(
            gpu,
            counter_plan=plan,
            summary_dtype=UNKNOWN_DTYPE,
            cache_dir=roofline_cache_dir,
            device_index=device_index,
        )

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or not torch_spec.origin:
        raise RuntimeError("same-pass capture requires a torch installation")
    torch_lib = Path(torch_spec.origin).parent / "lib"
    kineto = torch_lib / "libtorch_cpu.so"
    if not kineto.is_file():
        raise RuntimeError(f"Kineto SDK client library not found: {kineto}")
    sdk_root = Path(rocprof).resolve().parent.parent
    audit = out / ".profilelens_audit"
    dtype_dir = out / ".profilelens_dtype"
    audit.mkdir(exist_ok=True)
    dtype_dir.mkdir(exist_ok=True)

    def prepend_paths(*paths, existing=""):
        return os.pathsep.join(
            dict.fromkeys(str(p) for p in [*paths, *existing.split(os.pathsep)] if str(p))
        )

    env = {
        "ROCP_TOOL_LIBRARIES": prepend_paths(
            kineto,
            existing=os.environ.get("ROCP_TOOL_LIBRARIES", ""),
        ),
        "LD_LIBRARY_PATH": prepend_paths(
            torch_lib,
            sdk_root / "lib/host-math/lib",
            sdk_root / "lib",
            existing=os.environ.get("LD_LIBRARY_PATH", ""),
        ),
        "PYTHONPATH": prepend_paths(
            Path(__file__).parent / "_worker_shim",
            existing=os.environ.get("PYTHONPATH", ""),
        ),
        "PROFILELENS_PHASE2": "1",
        "PROFILELENS_NATIVE_SELECTED_REGIONS": "1",
        "PROFILELENS_EXPORT_PHASE2_TRACE": "1",
        "PROFILELENS_ROCPROF_FINALIZE": "fork",
        "PROFILELENS_AUDIT_DIR": str(audit),
        "PROFILELENS_DTYPE_DIR": str(dtype_dir),
    }
    # Preserve the device namespace used for planning and roofs.
    visibility = {
        key: os.environ[key]
        for key in (
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        )
        if key in os.environ
    }
    env.update(visibility)
    peaks = prepared_roof.peaks if prepared_roof else None
    peaks_by_dtype = prepared_roof.peaks_by_dtype if prepared_roof else {}
    metadata = {
        "schema_version": 2,
        "mode": "same_pass",
        "rocprofv3_path": rocprof,
        "capture_dir": str(out),
        "gpu_info": asdict(gpu),
        "device_index": device_index,
        "device_identity": identity,
        "device_visibility": visibility,
        "dtype": None,
        "observed_dtypes": [],
        "counter_dtype_hints": list(plan.dtype_hints),
        "counter_dtypes": list(plan.dtype_coverage),
        "counter_set_name": cset.name,
        "counters": list(cset.counters),
        "roofline_peaks": asdict(peaks) if peaks else None,
        "roofline_reference_dtype": peaks.detected_dtype if peaks else None,
        "peaks_by_dtype": {key: asdict(value) for key, value in peaks_by_dtype.items()},
        "roofline_csv": prepared_roof.csv_path if prepared_roof else None,
        "worker_environment": env,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def build_capture_command(metadata: dict[str, Any], user_command: list[str]) -> list[str]:
    """Build a capture launch; the caller owns execution, windows and worker joins."""
    from profilelens.profile.rocprofv3 import build_command

    counters = metadata.get("counters")
    if metadata.get("mode") != "same_pass" or not metadata.get("capture_dir"):
        raise ValueError("use metadata returned by prepare_capture")
    if (
        not isinstance(counters, list)
        or not counters
        or any(not isinstance(name, str) or not name or name.startswith("-") for name in counters)
    ):
        raise ValueError("capture metadata has invalid counter names")
    if (
        not isinstance(user_command, (list, tuple))
        or not user_command
        or any(not isinstance(arg, str) for arg in user_command)
    ):
        raise ValueError("user_command must be a nonempty argument list")
    return build_command(
        metadata["rocprofv3_path"],
        metadata["capture_dir"],
        tuple(counters),
        "",
        user_command,
        process_sync=False,
        kernel_trace=True,
        marker_trace=True,
        selected_regions=True,
    )


def _trace_export(capture_dir: Path, trace: Path, audit_dir: str | Path | None) -> dict:
    digest = _sha256(trace)
    exports = []
    for path in Path(audit_dir or capture_dir / ".profilelens_audit").glob("*.json"):
        payload = json.loads(path.read_text())
        exports.extend(
            e
            for e in payload.get("trace_exports", [])
            if e.get("sha256") == digest and e.get("phase2") is True
        )
    matches = {(str(e["process_id"]), e["region_name"]): e for e in exports}
    if len(matches) != 1:
        raise ValueError(
            f"trace requires one successful same-pass export mapping; found {len(matches)}"
        )
    return next(iter(matches.values()))


def _fork_pid_overrides(capture: Path, pid: str, audit_dir: str | Path | None) -> dict[str, str]:
    """Map a worker's file PID to the SDK-cached ancestor PID in its CSV rows."""
    ancestry = {}
    for path in Path(audit_dir or capture / ".profilelens_audit").glob("*.json"):
        payload = json.loads(path.read_text())
        if payload.get("pid") is not None and payload.get("ppid") is not None:
            ancestry[str(payload["pid"])] = str(payload["ppid"])
    ancestors = set()
    current = pid
    while current in ancestry and ancestry[current] not in ancestors:
        current = ancestry[current]
        ancestors.add(current)
    raw_pids = set()
    for path in capture.rglob(f"{pid}_*.csv"):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                raw_pid = row.get("Process_Id") or row.get("Process_ID") or row.get("process-id")
                if raw_pid:
                    raw_pids.add(str(raw_pid))
    if len(raw_pids) != 1 or not raw_pids <= ancestors:
        raise ValueError(
            "CSV PID mismatch is not explained by the exporting worker's audited ancestry"
        )
    return {pid: next(iter(raw_pids))}


def _dispatch_identity(dispatch) -> tuple[str, str, str, str]:
    identity = (dispatch.process_id, dispatch.agent_id, dispatch.queue_id, dispatch.dispatch_id)
    if not all(identity):
        raise ValueError("same-pass capture requires process, agent, queue and dispatch IDs")
    return identity


def _unique_dispatches(dispatches, label):
    result = {}
    for dispatch in dispatches:
        key = _dispatch_identity(dispatch)
        if key in result:
            raise ValueError(f"duplicate {label} dispatch identity: {key}")
        result[key] = dispatch
    return result


def _validate_gpu_identity(capture, metadata, native) -> dict[str, Any]:
    agents = {dispatch.agent_id for dispatch in native}
    if len(agents) != 1:
        raise ValueError("a capture window must contain dispatches from exactly one GPU agent")
    agent = next(iter(agents))
    identity = metadata.get("device_identity")
    if not identity:
        if metadata.get("schema_version", 1) >= 2:
            raise ValueError("capture metadata is missing its prepared GPU identity")
        return {
            "agent_id": agent,
            "prepared_device_verified": False,
            "identity_source": "legacy metadata has no prepared device identity",
        }
    match = re.fullmatch(r"(?:Agent\s+)?(\d+)", agent)
    if not match or match.group(1) != str(identity.get("node_id")):
        raise ValueError("captured GPU agent differs from the prepared/roofline GPU")
    paths = sorted(capture.rglob("*agent_info.csv"))
    if not paths:
        raise ValueError("capture requires agent_info.csv to verify the prepared GPU identity")
    identities = set()
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if str(row.get("Node_Id")) == str(identity["node_id"]):
                    identities.add((row.get("Agent_Type"), row.get("Gpu_Id")))
    if identities != {("GPU", str(identity.get("gpu_id")))}:
        raise ValueError("captured GPU identity differs from the prepared/roofline GPU")
    return {
        "agent_id": agent,
        "prepared_device_verified": True,
        "node_id": str(identity["node_id"]),
        "gpu_id": str(identity["gpu_id"]),
    }


def _prepared_kernel_peaks(dtype, peaks_by_dtype, reference):
    # MXFP4 uses an FP4 roof, never an FP8/F6xF4 roof.
    benchmark_alias = {
        "MXFP4": "FP4",
        "NVFP4": "FP4",
        "MXFP6": "FP6",
        "MXFP8": "FP8",
        "F8": "FP8",
        "F16": "FP16",
        "F32": "FP32",
        "F64": "FP64",
    }.get(dtype, dtype)
    peaks = peaks_by_dtype.get(dtype) or peaks_by_dtype.get(benchmark_alias)
    if dtype != UNKNOWN_DTYPE and peaks is not None:
        return replace(peaks, detected_dtype=dtype)
    if reference is not None:
        return replace(reference, detected_dtype=dtype, peak_compute_tflops={}, ridge_point=0.0)
    return None


def analyze_capture_window(
    capture_dir: str | Path,
    trace_file: str | Path,
    *,
    kernel_names: list[str] | tuple[str, ...] | None = None,
    output_dir: str | Path | None = None,
    per_kernel_dtype: dict[str, tuple[str, str]] | None = None,
    audit_dir: str | Path | None = None,
) -> tuple[ProfilingResult, dict[str, Any]]:
    """Validate and analyze one completed same-pass capture window."""
    from profilelens.profile.dtype_detect import resolve_kernel_dtypes
    from profilelens.profile.kernel_context import _normalize_trace_dtype
    from profilelens.profile.metrics import compute_kernel_metrics
    from profilelens.profile.parser import (
        aggregate_window_dispatches,
        parse_counter_collection,
        parse_graph_shape_ranges,
        parse_kernel_trace,
        parse_marker_ranges,
        parse_shape_ranges,
    )

    capture = Path(capture_dir).resolve()
    trace_path = Path(trace_file).resolve()
    metadata = json.loads((capture / "capture_metadata.json").read_text())
    if metadata.get("mode") != "same_pass":
        raise ValueError("capture metadata must declare same_pass mode")
    requested = metadata.get("counters")
    if (
        not isinstance(requested, list)
        or not requested
        or any(not isinstance(name, str) or not name for name in requested)
        or len(requested) != len(set(requested))
    ):
        raise ValueError("capture metadata requires a unique nonempty counter list")
    if "GRBM_COUNT" not in requested:
        raise ValueError("same-pass capture requires GRBM_COUNT in its requested counter set")
    required_csvs = ("kernel_trace", "marker_api_trace", "counter_collection")
    missing_csvs = [kind for kind in required_csvs if not any(capture.rglob(f"*{kind}*.csv"))]
    if missing_csvs:
        raise ValueError(
            "native ROCprofiler CSV output was not flushed (missing "
            + ", ".join(missing_csvs)
            + "); check worker shutdown and finalization"
        )
    owner = _trace_export(capture, trace_path, audit_dir)
    pid, region_name = str(owner["process_id"]), owner["region_name"]
    overrides = {}
    all_regions = parse_marker_ranges(str(capture))
    if not any(r.process_id == pid and r[0] == region_name for r in all_regions):
        overrides = _fork_pid_overrides(capture, pid, audit_dir)
        all_regions = parse_marker_ranges(str(capture), process_id_overrides=overrides)
    regions = [region for region in all_regions if region.process_id == pid]
    matching = [region for region in regions if region[0] == region_name]
    if len(matching) != 1 or matching[0][2] <= matching[0][1]:
        raise ValueError("trace export does not identify one completed PMC region")
    _, start, end = matching[0]
    if any(
        region[0] != region_name and max(start, region[1]) < min(end, region[2])
        for region in regions
    ):
        raise ValueError("overlapping PMC regions make window ownership ambiguous")

    def in_window(dispatch):
        if dispatch.process_id != pid:
            return False
        overlaps = dispatch.start_ns < end and dispatch.end_ns > start
        starts_inside = start <= dispatch.start_ns < end
        if overlaps or starts_inside:
            if dispatch.start_ns < start or dispatch.end_ns > end or dispatch.duration_ns <= 0:
                raise ValueError("GPU dispatch crosses or has invalid window boundaries")
            return True
        return False

    native = [
        dispatch
        for dispatch in parse_kernel_trace(
            str(capture),
            process_id_overrides=overrides,
        )
        if in_window(dispatch)
    ]
    counters = [
        dispatch
        for dispatch in parse_counter_collection(
            str(capture),
            process_id_overrides=overrides,
            strict=True,
        )
        if in_window(dispatch)
    ]
    if not native or not counters:
        raise ValueError("window requires both actual kernel trace and PMC dispatches")
    gpu_identity = _validate_gpu_identity(capture, metadata, native)
    native_by_id = _unique_dispatches(native, "native")
    counter_by_id = _unique_dispatches(counters, "PMC")
    for key, counted in counter_by_id.items():
        traced = native_by_id.get(key)
        if traced is None:
            raise ValueError(f"PMC dispatch identity has no native trace match: {key}")
        if (
            boundary_kernel_name(counted.kernel_name) != boundary_kernel_name(traced.kernel_name)
            or counted.start_ns != traced.start_ns
            or counted.end_ns != traced.end_ns
        ):
            raise ValueError(f"PMC/native dispatch name or timestamps differ: {key}")
        if counted.counters.get("GRBM_COUNT", 0) <= 0:
            raise ValueError("every counted dispatch requires positive GRBM_COUNT")

    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt") as stream:
        payload = json.load(stream)
    gpu_events = [
        event
        for event in payload.get("traceEvents", [])
        if event.get("ph") == "X" and event.get("cat") == "kernel"
    ]
    torch_counts = Counter(boundary_kernel_name(event["name"]) for event in gpu_events)
    native_counts = Counter(boundary_kernel_name(dispatch.kernel_name) for dispatch in native)
    if not gpu_events or torch_counts != native_counts:
        raise ValueError("torch and rocprof window GPU dispatch populations differ")
    torch_durations, native_durations = defaultdict(list), defaultdict(list)
    for event in gpu_events:
        duration = float(event["dur"]) * 1000
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("torch GPU dispatch has an invalid duration")
        torch_durations[boundary_kernel_name(event["name"])].append(duration)
    for dispatch in native:
        native_durations[boundary_kernel_name(dispatch.kernel_name)].append(dispatch.duration_ns)
    for name in native_durations:
        pairs = zip(sorted(torch_durations[name]), sorted(native_durations[name]))
        if any(abs(a - b) > 2 for a, b in pairs):
            raise ValueError(f"torch and rocprof dispatch durations differ: {name}")

    if kernel_names is not None and (
        not kernel_names
        or any(
            not isinstance(name, str) or not boundary_kernel_name(name) for name in kernel_names
        )
    ):
        raise ValueError("kernel_names must contain at least one nonempty kernel name")
    wanted = {boundary_kernel_name(name) for name in kernel_names or native_counts}
    measured = {boundary_kernel_name(dispatch.kernel_name) for dispatch in counters}
    missing = wanted - measured
    if missing:
        raise ValueError(f"selected kernels lack same-window counters: {sorted(missing)}")
    selected_native = [
        dispatch for dispatch in native if boundary_kernel_name(dispatch.kernel_name) in wanted
    ]
    selected_counters = [
        dispatch for dispatch in counters if boundary_kernel_name(dispatch.kernel_name) in wanted
    ]
    for traced in selected_native:
        key = _dispatch_identity(traced)
        if key not in counter_by_id:
            raise ValueError(f"unsampled capture has incomplete selected-kernel coverage: {key}")
        missing_counters = set(requested) - counter_by_id[key].counters.keys()
        if missing_counters:
            raise ValueError(
                f"selected dispatch {key} lacks requested counters: "
                + ", ".join(sorted(missing_counters))
            )
    native_count = len(selected_native)
    selected_native = [
        replace(dispatch, kernel_name=canonical_kernel_name(dispatch.kernel_name))
        for dispatch in selected_native
    ]
    selected_counters = [
        replace(
            dispatch,
            kernel_name=canonical_kernel_name(
                native_by_id[_dispatch_identity(dispatch)].kernel_name
            ),
        )
        for dispatch in selected_counters
    ]
    aggregates = aggregate_window_dispatches(
        selected_native,
        selected_counters,
        markers=parse_shape_ranges(str(capture), process_id_overrides=overrides),
        graph_markers=parse_graph_shape_ranges(str(capture), process_id_overrides=overrides),
        regions=matching,
    )

    dtype_sidecar = capture / ".profilelens_dtype" / f"{pid}.json"
    sidecar = json.loads(dtype_sidecar.read_text()) if dtype_sidecar.is_file() else {}
    types_by_external_id = defaultdict(set)
    for event in payload.get("traceEvents", []):
        args = event.get("args", {})
        if event.get("cat") == "cpu_op" and "External id" in args:
            for raw_type in args.get("Input type", []):
                dtype = _normalize_trace_dtype(raw_type)
                if dtype:
                    types_by_external_id[str(args["External id"])].add(dtype)
    types_by_symbol = defaultdict(set)
    for event in gpu_events:
        types = types_by_external_id.get(str(event.get("args", {}).get("External id")), set())
        if len(types) == 1:
            types_by_symbol[boundary_kernel_name(event["name"])].update(types)
    trace_dtypes = {
        aggregate.kernel_name: next(iter(
            types_by_symbol[boundary_kernel_name(aggregate.kernel_name)]
        ))
        for aggregate in aggregates
        if len(types_by_symbol[boundary_kernel_name(aggregate.kernel_name)]) == 1
    }
    resolved, _ = resolve_kernel_dtypes(
        [agg.kernel_name for agg in aggregates], sidecar, trace_dtypes
    )
    explicit_dtypes = {}
    for name, evidence in (per_kernel_dtype or {}).items():
        if (
            not isinstance(evidence, (tuple, list))
            or len(evidence) != 2
            or not all(isinstance(value, str) and value for value in evidence)
        ):
            raise ValueError("per_kernel_dtype requires (dtype, evidence_source) entries")
        normalized = normalize_dtype(evidence[0])
        explicit_dtypes[name] = (normalized, evidence[1])
    add_canonical_aliases(explicit_dtypes)
    for aggregate in aggregates:
        supplied = lookup_kernel(explicit_dtypes, aggregate.kernel_name)
        if supplied is not None:
            resolved[aggregate.kernel_name] = supplied
    add_canonical_aliases(resolved)
    gpu = GpuInfo(**metadata["gpu_info"])
    reference = (
        RooflinePeaks(**metadata["roofline_peaks"]) if metadata.get("roofline_peaks") else None
    )
    peaks_by_dtype = {
        key: RooflinePeaks(**value) for key, value in metadata.get("peaks_by_dtype", {}).items()
    }
    profiles, observed_peaks = [], {}
    for aggregate in aggregates:
        dtype, source = lookup_kernel(
            resolved, aggregate.kernel_name, (UNKNOWN_DTYPE, "unresolved")
        )
        kernel_peaks = _prepared_kernel_peaks(dtype, peaks_by_dtype, reference)
        if kernel_peaks and kernel_peaks.peak_compute_tflops:
            observed_peaks[dtype] = kernel_peaks
        profiles.append(
            compute_kernel_metrics(
                aggregate,
                kernel_peaks,
                dtype,
                gpu_info=gpu,
                dtype_source=source,
            )
        )
    unresolved = tuple(
        dict.fromkeys(
            profile.kernel_name for profile in profiles if profile.detected_dtype == UNKNOWN_DTYPE
        )
    )
    if unresolved:
        warnings.warn(
            "Compute dtype unresolved for selected kernel(s): "
            + ", ".join(unresolved)
            + ". Captured counter coverage may be incomplete; configure counter dtype "
            "hints or explicit counters before a future capture.",
            UserWarning,
            stacklevel=2,
        )
    observed = tuple(
        sorted(
            {
                profile.detected_dtype
                for profile in profiles
                if profile.detected_dtype != UNKNOWN_DTYPE
            }
        )
    )
    result = ProfilingResult(
        gpu_arch=gpu.arch,
        gpu_model=gpu.model,
        detected_dtype=observed[0] if len(observed) == 1 and not unresolved else UNKNOWN_DTYPE,
        kernels=tuple(profiles),
        roofline_peaks=reference,
        peaks_by_dtype={**peaks_by_dtype, **observed_peaks},
        counter_set_name=metadata["counter_set_name"],
        counters_used=tuple(requested),
        output_dir=str(output_dir or capture),
        profiling_duration_s=(end - start) / 1e9,
        observed_dtypes=observed,
        counter_dtype_hints=tuple(metadata.get("counter_dtype_hints", ())),
        counter_dtypes=tuple(metadata.get("counter_dtypes", ())),
        unresolved_kernels=unresolved,
    )
    provenance = {
        "mode": "same_pass",
        "process_id": pid,
        "region_name": region_name,
        "file_pid_to_cached_ancestor_pid": overrides,
        "trace_path": str(trace_path),
        "trace_sha256": owner["sha256"],
        "original_export_path": owner["trace_path"],
        "start_ns": start,
        "end_ns": end,
        "torch_gpu_dispatches": len(gpu_events),
        "rocprof_gpu_dispatches": len(native),
        "counter_dispatches": len(counters),
        "selected_dispatches": native_count,
        "torch_native_population_and_durations_match": True,
        "pmc_native_identity_and_timestamps_match": True,
        "requested_counter_coverage_complete": True,
        "dispatch_population_and_durations_match": True,
        "gpu_identity": gpu_identity,
        "sample_iteration_ids": [],
        "selected_kernels": sorted(wanted),
        "capture_metadata_sha256": _sha256(capture / "capture_metadata.json"),
        "raw_csv_sha256": {
            str(path.relative_to(capture)): _sha256(path) for path in sorted(capture.rglob("*.csv"))
        },
    }
    return result, provenance
