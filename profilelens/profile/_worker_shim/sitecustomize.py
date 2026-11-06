# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ProfileLens worker shim."""

from __future__ import annotations

import atexit
import builtins
import ctypes
import hashlib
import json
import os
import sys
import threading

_FINALIZED_PID: int | None = None


def _profilelens_force_rocprof_finalize() -> None:
    """Flush native exit handlers once, preserving Python's real exit status."""
    global _FINALIZED_PID
    if _FINALIZED_PID == os.getpid():
        return
    _FINALIZED_PID = os.getpid()
    _write_audit_now()
    _write_dtype_sidecar()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=False)
        libc.__cxa_finalize.argtypes = [ctypes.c_void_p]
        libc.__cxa_finalize.restype = None
        libc.__cxa_finalize(None)
    except Exception:
        pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


class _ForkFinalizerToken:
    pass


_FORK_FINALIZER_TOKEN = _ForkFinalizerToken()


def _after_fork_finalize_hook(_token) -> None:
    """Reinstall after multiprocessing clears inherited atexit callbacks."""
    from multiprocessing import util

    atexit.register(_profilelens_force_rocprof_finalize)
    # Older fork launchers call os._exit without atexit; the PID guard dedupes.
    util.Finalize(None, _profilelens_force_rocprof_finalize, exitpriority=-100)


def _install_finalize_hook() -> None:
    mode = os.environ.get("PROFILELENS_ROCPROF_FINALIZE")
    if mode not in {"1", "fork"}:
        return
    if getattr(sys, "_profilelens_finalize_installed", False):
        return
    sys._profilelens_finalize_installed = True  # type: ignore[attr-defined]
    # Fork workers bypass libc exit handlers; the main process runs them itself.
    if mode == "1":
        atexit.register(_profilelens_force_rocprof_finalize)
    from multiprocessing import util

    util.register_after_fork(_FORK_FINALIZER_TOKEN, _after_fork_finalize_hook)


_AUDIT: dict[str, int] = {
    "torch_profiler_start": 0,
    "torch_profiler_stop": 0,
    "torch_cuda_profiler_start": 0,
    "torch_cuda_profiler_stop": 0,
    "roctx_resume": 0,
    "roctx_pause": 0,
}


_TRACE_PATHS: list[str] = []
_TRACE_EXPORTS: list[dict] = []

_GRAPH_PLAN_STATS = {"hit": 0, "miss": 0, "mismatch": 0}


def _write_audit_now() -> None:
    """Persist the current ``_AUDIT`` snapshot to ``<dir>/<pid>.json``."""
    audit_dir = os.environ.get("PROFILELENS_AUDIT_DIR")
    if not audit_dir:
        return
    try:
        path = os.path.join(audit_dir, f"{os.getpid()}.json")
        payload = {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "trace_paths": list(_TRACE_PATHS),
            "trace_exports": list(_TRACE_EXPORTS),
            "region_paths": dict(_REGION_PATHS),
            "graph_plan_stats": dict(_GRAPH_PLAN_STATS),
            **_AUDIT,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _is_phase2() -> bool:
    """True when running inside the rocprofv3-wrapped Phase 2 workload."""
    return os.environ.get("PROFILELENS_PHASE2") == "1"


def _uses_native_regions() -> bool:
    return os.environ.get("PROFILELENS_NATIVE_SELECTED_REGIONS") == "1"


_ROCTX_LIB: ctypes.CDLL | None = None
_ROCTX_LOAD_TRIED = False


def _load_roctx_lib() -> ctypes.CDLL | None:
    """Resolve ``librocprofiler-sdk-roctx.so`` via ctypes."""
    global _ROCTX_LIB, _ROCTX_LOAD_TRIED
    if _ROCTX_LOAD_TRIED:
        return _ROCTX_LIB
    _ROCTX_LOAD_TRIED = True

    candidates = []
    override = os.environ.get("PROFILELENS_ROCTX_LIB")
    if override:
        candidates.append(override)
    candidates.append("librocprofiler-sdk-roctx.so")
    candidates.append("librocprofiler-sdk-roctx.so.1")

    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        try:
            lib.roctxProfilerResume.argtypes = [ctypes.c_uint64]
            lib.roctxProfilerResume.restype = ctypes.c_int
            lib.roctxProfilerPause.argtypes = [ctypes.c_uint64]
            lib.roctxProfilerPause.restype = ctypes.c_int
        except AttributeError:
            continue
        try:
            lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
            lib.roctxRangePushA.restype = ctypes.c_int
            lib.roctxRangePop.argtypes = []
            lib.roctxRangePop.restype = ctypes.c_int
        except AttributeError:
            pass
        _ROCTX_LIB = lib
        return _ROCTX_LIB

    return None


def _emit_roctx(op: str) -> None:
    """Emit a roctx pause/resume call via ctypes."""
    lib = _load_roctx_lib()
    if lib is None:
        return
    try:
        if op == "resume":
            lib.roctxProfilerResume(ctypes.c_uint64(0))
            _AUDIT["roctx_resume"] += 1
        elif op == "pause":
            lib.roctxProfilerPause(ctypes.c_uint64(0))
            _AUDIT["roctx_pause"] += 1
    except Exception:
        pass
    _write_audit_now()


_ROCTX_SESSION = [-1]
_REGION_PATHS: dict[str, str] = {}
_PROFILER_RANGE_TLS = threading.local()


def _roctx_range_push(name: str) -> bool:
    lib = _load_roctx_lib()
    if lib is None:
        return False
    try:
        # Zero is a valid ROCTX nesting level.
        return int(lib.roctxRangePushA(name.encode())) >= 0
    except Exception:
        return False


def _roctx_range_pop() -> None:
    lib = _load_roctx_lib()
    if lib is None:
        return
    try:
        lib.roctxRangePop()
    except Exception:
        pass


def _profiler_region_push(name: str) -> bool:
    """Open one outer profiler range and coalesce nested trigger APIs."""
    depth = int(getattr(_PROFILER_RANGE_TLS, "depth", 0))
    if depth:
        _PROFILER_RANGE_TLS.depth = depth + 1
        return False
    opened = _roctx_range_push(name)
    if opened:
        _PROFILER_RANGE_TLS.depth = 1
    return opened


def _profiler_region_pop(opened: bool) -> None:
    """Leave a profiler trigger, popping ROCTX only for its outer owner."""
    depth = int(getattr(_PROFILER_RANGE_TLS, "depth", 0))
    if depth:
        _PROFILER_RANGE_TLS.depth = depth - 1
    if opened:
        _roctx_range_pop()
    if opened and depth == 1 and _uses_native_regions():
        _emit_roctx("pause")


def _iter_tensor_values(values, seen_containers=None):  # type: ignore[no-untyped-def]
    """Yield tensor-like leaves from positional, list, tuple, or dict args."""
    if seen_containers is None:
        seen_containers = set()
    for value in values:
        try:
            list(value.shape)
        except Exception:
            pass
        else:
            yield value
            continue
        if isinstance(value, dict):
            nested = value.values()
        elif isinstance(value, (list, tuple)):
            nested = value
        else:
            continue
        container_id = id(value)
        if container_id in seen_containers:
            continue
        seen_containers.add(container_id)
        try:
            yield from _iter_tensor_values(nested, seen_containers)
        finally:
            seen_containers.remove(container_id)


def _shape_signature_from_values(values) -> str:
    """Canonical per-launch shape signature from an iterable of call arguments."""
    parts: list[str] = []
    for v in _iter_tensor_values(values):
        try:
            shp = list(v.shape)
        except Exception:
            continue
        # Distinguish scalars from zero-length 1-D tensors.
        parts.append("x".join(str(int(d)) for d in shp) if shp else "scalar")
    return "-".join(parts)


def _emit_shape_marker(stem: str, sig: str) -> None:
    """Phase-2 only: emit ONE roctx marker per kernel launch."""
    if not (sig and _is_phase2()):
        return
    name = f"pl_shape_{stem}__{sig}"
    if _roctx_range_push(name):
        _roctx_range_pop()


def _kernel_symbol_key(name: str) -> str:
    """Stable, marker-safe identity for one exact GPU kernel symbol."""
    name = (name or "").strip()
    if name.startswith("void "):
        name = name[5:]
    if not name:
        return ""
    return hashlib.sha1(
        name.encode("utf-8", errors="replace"), usedforsecurity=False
    ).hexdigest()[:20]


def _emit_graph_shape_marker(kernel_key: str, sig: str) -> None:
    """Emit a replay marker keyed by an exact graph kernel symbol."""
    if not (kernel_key and sig and _is_phase2()):
        return
    if _roctx_range_push(f"pl_graph_shape_{kernel_key}__{sig}"):
        _roctx_range_pop()


def _synchronize_torch_device(device_name: "str | None") -> bool:
    """Drain one PyTorch device backend, if it exposes ``synchronize``."""
    if not device_name:
        return False
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return False
    device_module = getattr(torch_module, device_name, None)
    synchronize = getattr(device_module, "synchronize", None)
    if not callable(synchronize):
        return False
    synchronize()
    return True


def _synchronize_profiler_device(profile) -> bool:
    """Drain the profiler device before closing its ROCTX range."""
    inner = getattr(profile, "profiler", None)
    device_name = (
        getattr(profile, "use_device", None)
        or getattr(inner, "use_device", None)
    )
    return _synchronize_torch_device(device_name)


def _patch_torch_profiler_profile() -> bool:
    """Wrap ``torch.profiler.profile.start`` and ``.stop``."""
    try:
        import torch.profiler as tp  # type: ignore[import-not-found]
    except Exception:
        return False

    cls = getattr(tp, "profile", None)
    if cls is None or not isinstance(cls, type):
        return False
    if getattr(cls, "_profilelens_patched", False):
        return True

    orig_start = cls.start
    orig_stop = cls.stop
    orig_export_chrome_trace = getattr(cls, "export_chrome_trace", None)
    orig_export_stacks = getattr(cls, "export_stacks", None)
    orig_key_averages = getattr(cls, "key_averages", None)

    def _start(self, *a, **kw):  # type: ignore[no-untyped-def]
        orig_start(self, *a, **kw)
        _AUDIT["torch_profiler_start"] += 1
        _write_audit_now()
        if getattr(self, "_profilelens_native_trace_hooks", False):
            return
        if not getattr(self, "_profilelens_emitted_via_step", False):
            _emit_roctx("resume")
        _ROCTX_SESSION[0] += 1
        self._profilelens_region_name = (  # type: ignore[attr-defined]
            f"pl_region_{_ROCTX_SESSION[0]}"
        )
        if _is_phase2():
            self._profilelens_region_open = _profiler_region_push(  # type: ignore[attr-defined]
                self._profilelens_region_name  # type: ignore[attr-defined]
            )
            self._profilelens_region_active = True  # type: ignore[attr-defined]

    def _stop(self, *a, **kw):  # type: ignore[no-untyped-def]
        _AUDIT["torch_profiler_stop"] += 1
        _write_audit_now()
        if getattr(self, "_profilelens_native_trace_hooks", False):
            return orig_stop(self, *a, **kw)
        phase2 = _is_phase2()
        if not getattr(self, "_profilelens_emitted_via_step", False):
            if not phase2:
                _emit_roctx("pause")
        if not phase2:
            return orig_stop(self, *a, **kw)

        sync_error = None
        try:
            _synchronize_profiler_device(self)
        except BaseException as exc:
            sync_error = exc
        region_open = bool(getattr(self, "_profilelens_region_open", False))
        if getattr(self, "_profilelens_region_active", region_open):
            _profiler_region_pop(region_open)
            self._profilelens_region_open = False  # type: ignore[attr-defined]
            self._profilelens_region_active = False  # type: ignore[attr-defined]
        if (region_open and not _uses_native_regions()
                and not getattr(self, "_profilelens_emitted_via_step", False)):
            _emit_roctx("pause")

        try:
            result = orig_stop(self, *a, **kw)
        except BaseException as stop_error:
            if sync_error is not None:
                raise stop_error from sync_error
            raise
        if sync_error is not None:
            raise sync_error
        return result

    def _export_chrome_trace(self, path, *a, **kw):  # type: ignore[no-untyped-def]
        if _is_phase2() and os.environ.get("PROFILELENS_EXPORT_PHASE2_TRACE") != "1":
            return None
        result = orig_export_chrome_trace(self, path, *a, **kw)
        try:
            ap = os.path.abspath(path)
            _TRACE_PATHS.append(ap)
            region_name = getattr(
                self,
                "_profilelens_region_name",
                f"pl_region_{_ROCTX_SESSION[0]}",
            )
            _REGION_PATHS[region_name] = ap
            digest = hashlib.sha256()
            with open(ap, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            _TRACE_EXPORTS.append({
                "process_id": str(os.getpid()),
                "region_name": region_name,
                "trace_path": ap,
                "sha256": digest.hexdigest(),
                "phase2": _is_phase2(),
            })
            _write_audit_now()
        except Exception:
            pass
        return result

    def _export_stacks(self, *a, **kw):  # type: ignore[no-untyped-def]
        if _is_phase2():
            return None
        return orig_export_stacks(self, *a, **kw)

    def _key_averages(self, *a, **kw):  # type: ignore[no-untyped-def]
        if _is_phase2():
            class _EmptyAvgs(list):  # type: ignore[type-arg]
                def table(self, *_a, **_kw):
                    return ""
            return _EmptyAvgs()
        return orig_key_averages(self, *a, **kw)

    cls.start = _start
    cls.stop = _stop
    if orig_export_chrome_trace is not None:
        cls.export_chrome_trace = _export_chrome_trace
    if orig_export_stacks is not None:
        cls.export_stacks = _export_stacks
    if orig_key_averages is not None:
        cls.key_averages = _key_averages
    cls._profilelens_patched = True  # type: ignore[attr-defined]
    return True


def _patch_kineto_profile() -> bool:
    """Patch ``_KinetoProfile`` init, trace hooks and schedule step."""
    try:
        from torch.profiler import ProfilerAction  # type: ignore[import-not-found]
        from torch.profiler import profile as _PublicProfile  # type: ignore[import-not-found]
        from torch.profiler.profiler import _KinetoProfile  # type: ignore[import-not-found]
    except Exception:
        return False

    if getattr(_KinetoProfile, "_profilelens_kineto_patched", False):
        return True

    orig_init = _KinetoProfile.__init__
    # torch >= 2.1 defines step on the public subclass.
    step_owner = _PublicProfile if "step" in _PublicProfile.__dict__ else _KinetoProfile
    orig_step = step_owner.step
    orig_start_trace = getattr(_KinetoProfile, "start_trace", None)
    orig_stop_trace = getattr(_KinetoProfile, "stop_trace", None)
    can_hook_trace = callable(orig_start_trace) and callable(orig_stop_trace)

    def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("record_shapes") is False:
            sys.stderr.write(
                "[profilelens] forced record_shapes=True on _KinetoProfile "
                "(was False) — needed for dtype inference\n"
            )
        kwargs["record_shapes"] = True

        self._profilelens_emitted_via_step = False
        self._profilelens_native_trace_hooks = (
            can_hook_trace and _is_phase2() and _uses_native_regions()
        )
        orig_init(self, *args, **kwargs)

    def _start_trace(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = orig_start_trace(self, *args, **kwargs)
        if not getattr(self, "_profilelens_native_trace_hooks", False):
            return result
        if not getattr(self, "_profilelens_region_active", False):
            _ROCTX_SESSION[0] += 1
            self._profilelens_region_name = f"pl_region_{_ROCTX_SESSION[0]}"
            if not getattr(_PROFILER_RANGE_TLS, "depth", 0):
                _emit_roctx("resume")
            self._profilelens_region_open = _profiler_region_push(
                self._profilelens_region_name
            )
            self._profilelens_region_active = True
        return result

    def _stop_trace(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        sync_error = None
        if (getattr(self, "_profilelens_native_trace_hooks", False)
                and getattr(self, "_profilelens_region_active", False)):
            try:
                _synchronize_profiler_device(self)
            except BaseException as exc:
                sync_error = exc
            _profiler_region_pop(bool(getattr(self, "_profilelens_region_open", False)))
            self._profilelens_region_open = False
            self._profilelens_region_active = False
        try:
            result = orig_stop_trace(self, *args, **kwargs)
        except BaseException as stop_error:
            if sync_error is not None:
                raise stop_error from sync_error
            raise
        if sync_error is not None:
            raise sync_error
        return result

    RECORD_ACTIONS = (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE)

    def _patched_step(self):  # type: ignore[no-untyped-def]
        """Drive roctx push/pop on schedule action transitions."""
        if getattr(self, "_profilelens_native_trace_hooks", False):
            return orig_step(self)
        prev_action = getattr(self, "current_action", None)
        orig_step(self)
        new_action = getattr(self, "current_action", None)

        was_recording = prev_action in RECORD_ACTIONS
        is_recording = new_action in RECORD_ACTIONS
        if is_recording and not was_recording:
            _emit_roctx("resume")
            self._profilelens_emitted_via_step = True
        elif was_recording and not is_recording:
            # Phase 2 skips pause: it would close the --selected-regions window.
            if not _is_phase2():
                _emit_roctx("pause")

    _KinetoProfile.__init__ = _patched_init
    if can_hook_trace:
        _KinetoProfile.start_trace = _start_trace
        _KinetoProfile.stop_trace = _stop_trace
    step_owner.step = _patched_step
    _KinetoProfile._profilelens_kineto_patched = True  # type: ignore[attr-defined]
    return True


def _patch_torch_cuda_profiler() -> bool:
    """Wrap the older ``torch.cuda.profiler.start`` / ``stop`` APIs."""
    try:
        import torch.cuda.profiler as tcp  # type: ignore[import-not-found]
    except Exception:
        return False

    if getattr(tcp, "_profilelens_patched", False):
        return True
    if not (hasattr(tcp, "start") and hasattr(tcp, "stop")):
        return False

    orig_start = tcp.start
    orig_stop = tcp.stop
    region_stack: list[bool] = []

    def _start(*a, **kw):  # type: ignore[no-untyped-def]
        phase2 = _is_phase2()
        if not phase2:
            result = orig_start(*a, **kw)
        else:
            result = None
        _AUDIT["torch_cuda_profiler_start"] += 1
        _write_audit_now()
        _emit_roctx("resume")
        _ROCTX_SESSION[0] += 1
        if phase2:
            region_stack.append(
                _profiler_region_push(f"pl_region_{_ROCTX_SESSION[0]}")
            )
        return result

    def _stop(*a, **kw):  # type: ignore[no-untyped-def]
        _AUDIT["torch_cuda_profiler_stop"] += 1
        _write_audit_now()
        if not _is_phase2():
            _emit_roctx("pause")
            return orig_stop(*a, **kw)

        started = bool(region_stack)
        region_open = region_stack.pop() if started else False
        sync_error = None
        if started:
            try:
                _synchronize_torch_device("cuda")
            except BaseException as exc:
                sync_error = exc
        if region_open:
            _profiler_region_pop(True)
            if not _uses_native_regions():
                _emit_roctx("pause")
        elif started:
            _profiler_region_pop(False)
        if sync_error is not None:
            raise sync_error
        return None

    tcp.start = _start
    tcp.stop = _stop
    tcp._profilelens_patched = True  # type: ignore[attr-defined]
    return True


def _patch_torch_profiler_both() -> bool:
    """Apply public and Kineto profiler patches."""
    a = _patch_torch_profiler_profile()
    b = _patch_kineto_profile()
    return a and b


_TORCH_DTYPE_MAP: dict[str, str] = {
    "torch.bfloat16": "BF16",
    "torch.float16": "FP16", "torch.half": "FP16",
    "torch.float8_e4m3fn": "FP8", "torch.float8_e4m3fnuz": "FP8",
    "torch.float8_e5m2": "FP8", "torch.float8_e5m2fnuz": "FP8",
    "torch.float32": "FP32", "torch.float": "FP32",
    "torch.float64": "FP64", "torch.double": "FP64",
}

_DTYPE_OBS: dict[str, dict] = {
    "triton": {},
    "triton_shape_counts": {},
    "capture_ops": {},
    "capture_dtypes": {},
    "capture_graphs": {},
    "flydsl": {},
    "flydsl_shapes": {},
    "flydsl_args": {},
    "flydsl_shape_counts": {},
}
_DTYPE_LOCK = threading.Lock()
_DTYPE_DIRTY = [0]


def _dtype_dir() -> "str | None":
    return os.environ.get("PROFILELENS_DTYPE_DIR")


def _write_dtype_sidecar() -> None:
    """Atomically persist this PID's dtype observations."""
    d = _dtype_dir()
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{os.getpid()}.json")
        with _DTYPE_LOCK:
            payload = {
                "pid": os.getpid(),
                "triton": _DTYPE_OBS["triton"],
                "triton_shape_counts": _DTYPE_OBS["triton_shape_counts"],
                "capture_ops": _DTYPE_OBS["capture_ops"],
                "capture_dtypes": _DTYPE_OBS["capture_dtypes"],
                "capture_graphs": _DTYPE_OBS["capture_graphs"],
                "graph_plan_stats": dict(_GRAPH_PLAN_STATS),
                "flydsl": _DTYPE_OBS["flydsl"],
                "flydsl_shapes": _DTYPE_OBS["flydsl_shapes"],
                "flydsl_args": _DTYPE_OBS["flydsl_args"],
                "flydsl_shape_counts": _DTYPE_OBS["flydsl_shape_counts"],
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
            _DTYPE_DIRTY[0] = 0
    except Exception:
        pass


def _bump_dirty() -> None:
    """Flush the sidecar periodically so a hard kill still leaves data."""
    _DTYPE_DIRTY[0] += 1
    if _DTYPE_DIRTY[0] >= 512:
        _write_dtype_sidecar()


def _record_tensor_dtypes(values, bucket: dict) -> bool:
    """Record internal compute dtypes of any torch tensors in ``values``."""
    found = False
    for v in _iter_tensor_values(values):
        dt = _TORCH_DTYPE_MAP.get(str(getattr(v, "dtype", "")))
        if dt is not None:
            bucket[dt] = bucket.get(dt, 0) + 1
            found = True
    return found


_CAPTURE_TLS = threading.local()
_GRAPH_SEQ = [0]


class _HipDim3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint),
        ("y", ctypes.c_uint),
        ("z", ctypes.c_uint),
    ]


class _HipKernelNodeParams(ctypes.Structure):
    _fields_ = [
        # HIP field order differs from CUDA.
        ("blockDim", _HipDim3),
        ("extra", ctypes.POINTER(ctypes.c_void_p)),
        ("func", ctypes.c_void_p),
        ("gridDim", _HipDim3),
        ("kernelParams", ctypes.POINTER(ctypes.c_void_p)),
        ("sharedMemBytes", ctypes.c_uint),
    ]


_GRAPH_RUNTIME: "dict | None" = None
_GRAPH_RUNTIME_TRIED = False
_DEMANGLE_LIB: "ctypes.CDLL | None" = None
_DEMANGLE_TRIED = False
_GRAPH_NODE_NAME_CACHE: dict[tuple[int, int], str] = {}
_GRAPH_NODE_CACHE_KEYS: dict[str, set[tuple[int, int]]] = {}


def _clear_graph_node_name_cache(gid: "str | None") -> None:
    """Release runtime node handles after one capture is finalized."""
    if gid is None:
        return
    for cache_key in _GRAPH_NODE_CACHE_KEYS.pop(gid, ()):
        _GRAPH_NODE_NAME_CACHE.pop(cache_key, None)


def _canonical_kernel_symbol(name: str) -> str:
    """Normalize names between kineto and rocprof CSVs."""
    name = (name or "").strip()
    if name.startswith("void "):
        name = name[5:]
    name = name.rstrip()
    if not name:
        return ""
    for suffix in (" [clone .kd]", "[clone .kd]"):
        if name.endswith(suffix):
            return name[: -len(suffix)].rstrip()
    if name.endswith(".kd"):
        return name[:-3].rstrip()
    return name


def _demangle_kernel_symbol(name: str) -> str:
    """Demangle a HIP function name without importing a Python C++ package."""
    global _DEMANGLE_LIB, _DEMANGLE_TRIED
    if not name.startswith("_Z"):
        return name
    if not _DEMANGLE_TRIED:
        _DEMANGLE_TRIED = True
        try:
            lib = ctypes.CDLL("libstdc++.so.6")
            fn = getattr(lib, "__cxa_demangle")
            fn.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
            ]
            fn.restype = ctypes.c_void_p
            _DEMANGLE_LIB = lib
        except Exception:
            _DEMANGLE_LIB = None
    if _DEMANGLE_LIB is None:
        return name
    ptr = None
    try:
        status = ctypes.c_int(-1)
        ptr = _DEMANGLE_LIB.__cxa_demangle(  # type: ignore[attr-defined]
            name.encode(), None, None, ctypes.byref(status)
        )
        if ptr and status.value == 0:
            return ctypes.string_at(ptr).decode("utf-8", errors="replace")
    except Exception:
        pass
    finally:
        if ptr:
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.free.argtypes = [ctypes.c_void_p]
                libc.free(ptr)
            except Exception:
                pass
    return name


def _load_graph_runtime() -> "dict | None":
    """Load the minimal HIP/CUDA graph-query API used during capture."""
    global _GRAPH_RUNTIME, _GRAPH_RUNTIME_TRIED
    if _GRAPH_RUNTIME_TRIED:
        return _GRAPH_RUNTIME
    _GRAPH_RUNTIME_TRIED = True

    torch = sys.modules.get("torch")
    is_hip = bool(getattr(getattr(torch, "version", None), "hip", None))
    if is_hip:
        candidates = ("libamdhip64.so", "libamdhip64.so.7")
        prefix = "hip"
    else:
        candidates = ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0")
        prefix = "cuda"

    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
            capture_info = getattr(lib, f"{prefix}StreamGetCaptureInfo_v2")
            graph_nodes = getattr(lib, f"{prefix}GraphGetNodes")
            node_type = getattr(lib, f"{prefix}GraphNodeGetType")
        except (OSError, AttributeError):
            continue
        capture_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        capture_info.restype = ctypes.c_int
        graph_nodes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        graph_nodes.restype = ctypes.c_int
        node_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        node_type.restype = ctypes.c_int

        runtime: dict = {
            "lib": lib,
            "capture_info": capture_info,
            "graph_nodes": graph_nodes,
            "node_type": node_type,
            "is_hip": is_hip,
        }
        if is_hip:
            try:
                node_params = lib.hipGraphKernelNodeGetParams
                node_params.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_HipKernelNodeParams),
                ]
                node_params.restype = ctypes.c_int
                name_ref = lib.hipKernelNameRef
                name_ref.argtypes = [ctypes.c_void_p]
                name_ref.restype = ctypes.c_char_p
                name_ref_by_ptr = lib.hipKernelNameRefByPtr
                name_ref_by_ptr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                name_ref_by_ptr.restype = ctypes.c_char_p
                runtime.update({
                    "node_params": node_params,
                    "name_ref": name_ref,
                    "name_ref_by_ptr": name_ref_by_ptr,
                })
            except AttributeError:
                pass
        _GRAPH_RUNTIME = runtime
        return runtime
    return None


def _captured_kernel_nodes() -> "list[dict] | None":
    """Snapshot the active stream capture's kernel nodes and exact symbols."""
    gid = _active_capture_gid()
    if gid is None:
        return None
    runtime = _load_graph_runtime()
    torch = sys.modules.get("torch")
    if runtime is None or torch is None:
        return None
    try:
        stream = int(torch.cuda.current_stream().cuda_stream)
        status = ctypes.c_int(0)
        capture_id = ctypes.c_ulonglong(0)
        graph = ctypes.c_void_p()
        dependencies = ctypes.POINTER(ctypes.c_void_p)()
        n_dependencies = ctypes.c_size_t(0)
        rc = runtime["capture_info"](
            ctypes.c_void_p(stream), ctypes.byref(status), ctypes.byref(capture_id),
            ctypes.byref(graph), ctypes.byref(dependencies), ctypes.byref(n_dependencies),
        )
        if rc != 0 or status.value != 1 or not graph.value:
            return None

        count = ctypes.c_size_t(0)
        if runtime["graph_nodes"](graph, None, ctypes.byref(count)) != 0:
            return None
        raw_nodes = (ctypes.c_void_p * count.value)()
        if count.value and runtime["graph_nodes"](
            graph, raw_nodes, ctypes.byref(count)
        ) != 0:
            return None

        out: list[dict] = []
        for raw_node in raw_nodes[:count.value]:
            if not raw_node:
                continue
            kind = ctypes.c_int(-1)
            if runtime["node_type"](
                ctypes.c_void_p(raw_node), ctypes.byref(kind)
            ) != 0 or kind.value != 0:  # 0 == kernel node
                continue
            handle = int(raw_node)
            cache_key = (int(capture_id.value), handle)
            _GRAPH_NODE_CACHE_KEYS.setdefault(gid, set()).add(cache_key)
            cached = cache_key in _GRAPH_NODE_NAME_CACHE
            name = _GRAPH_NODE_NAME_CACHE.get(cache_key, "")
            if not cached and runtime.get("is_hip") and runtime.get("node_params"):
                params = _HipKernelNodeParams()
                if runtime["node_params"](
                    ctypes.c_void_p(raw_node), ctypes.byref(params)
                ) == 0 and params.func:
                    raw_name = runtime["name_ref_by_ptr"](
                        params.func, ctypes.c_void_p(stream)
                    )
                    if not raw_name:
                        raw_name = runtime["name_ref"](params.func)
                    if raw_name:
                        name = _demangle_kernel_symbol(
                            raw_name.decode("utf-8", errors="replace")
                        )
            if not cached:
                _GRAPH_NODE_NAME_CACHE[cache_key] = name
            out.append({
                "handle": handle,
                "name": name,
                "key": _kernel_symbol_key(_canonical_kernel_symbol(name)),
            })
        return out
    except Exception:
        return None


_GRAPH_PLAN: "dict | None" = None


def _graph_plan_dir() -> "str | None":
    return os.environ.get("PROFILELENS_GRAPH_PLAN")


def _op_fingerprint(ops: list) -> str:
    """Hash a capture's operation sequence, excluding planned kernels."""
    parts = []
    for e in ops or ():
        e = e or {}
        parts.append(
            f"{e.get('op') or ''}|{e.get('stem') or ''}|{e.get('sig') or ''}"
        )
    blob = "\n".join(parts)
    return hashlib.sha1(
        blob.encode("utf-8", errors="replace"), usedforsecurity=False
    ).hexdigest()


def _op_structure_fingerprint(ops: list) -> str:
    """Hash a capture's operation sequence without operand shapes."""
    parts = []
    for e in ops or ():
        e = e or {}
        parts.append(f"{e.get('op') or ''}|{e.get('stem') or ''}")
    blob = "\n".join(parts)
    return hashlib.sha1(
        blob.encode("utf-8", errors="replace"), usedforsecurity=False
    ).hexdigest()


def _load_graph_plan() -> dict:
    """Index every Phase-1 captured graph by op fingerprint. Loaded once."""
    global _GRAPH_PLAN
    if _GRAPH_PLAN is not None:
        return _GRAPH_PLAN
    plan: dict = {"exact": {}, "structural": {}}
    d = _graph_plan_dir()
    if d and os.path.isdir(d):
        for entry in sorted(os.listdir(d)):
            if not entry.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, entry)) as f:
                    payload = json.load(f)
            except Exception:
                continue
            for graph in (payload.get("capture_graphs") or {}).values():
                ops = graph.get("ops") or []
                if not any("kernels" in (e or {}) for e in ops):
                    continue
                fp = _op_fingerprint(ops)
                kernels = [list((e or {}).get("kernels") or ()) for e in ops]
                ordered_keys = [
                    str((node or {}).get("key") or "")
                    for per_op in kernels
                    for node in per_op
                ]
                candidate = {
                    "kernels": kernels,
                    "ordered_keys": ordered_keys,
                }
                for index, key in (
                    (plan["exact"], fp),
                    (plan["structural"], _op_structure_fingerprint(ops)),
                ):
                    entry = index.setdefault(key, {"candidates": []})
                    if not any(c["kernels"] == kernels for c in entry["candidates"]):
                        entry["candidates"].append(candidate)
    _GRAPH_PLAN = plan
    return plan


def _graph_plan_available() -> bool:
    plan = _load_graph_plan()
    return bool(plan.get("exact") or plan.get("structural"))


def _captured_kernel_key_sequence() -> "list[str] | None":
    """Return the active capture's flattened kernel-key sequence."""
    nodes = _captured_kernel_nodes()
    if nodes is None:
        return None
    return [str((node or {}).get("key") or "") for node in nodes]


def _captured_kernel_key_counts() -> "dict | None":
    """Count the active capture's kernel keys."""
    keys = _captured_kernel_key_sequence()
    if keys is None:
        return None
    counts: dict = {}
    for key in keys:
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _permutation_is_harmless(per_op_kernels) -> bool:
    """Return whether a node permutation preserves per-op ownership."""
    key_sets = {
        tuple(sorted(str((node or {}).get("key") or "") for node in per_op))
        for per_op in per_op_kernels or ()
    }
    return len(key_sets) <= 1


def _select_graph_plan_candidate(candidates: list, observed: list[str]) -> "dict | None":
    """Select one ownership plan from its observed kernel sequence."""
    exact = [c for c in candidates if observed == c.get("ordered_keys")]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    compatible = [
        c for c in candidates
        if sorted(observed) == sorted(c.get("ordered_keys") or ())
        and _permutation_is_harmless(c.get("kernels") or ())
    ]
    return compatible[0] if len(compatible) == 1 else None


def _apply_graph_plan(gid: str) -> None:
    """Validate and apply a Phase-1 kernel-ownership plan."""
    plan = _load_graph_plan()
    if not (plan.get("exact") or plan.get("structural")):
        return
    with _DTYPE_LOCK:
        graph = _DTYPE_OBS["capture_graphs"].get(gid)
        ops = list(graph.get("ops") or ()) if graph else []
    if not graph or not ops:
        return

    with _DTYPE_LOCK:
        for op_entry in graph.get("ops") or ():
            if isinstance(op_entry, dict):
                op_entry.pop("kernels", None)

    exact_entry = plan["exact"].get(_op_fingerprint(ops))
    structural_entry = plan["structural"].get(_op_structure_fingerprint(ops))

    def _eligible(entry):
        return [
            c for c in ((entry or {}).get("candidates") or ())
            if len(c.get("kernels") or ()) == len(ops)
        ]

    exact_candidates = _eligible(exact_entry)
    structural_candidates = _eligible(structural_entry)
    if not exact_candidates and not structural_candidates:
        _GRAPH_PLAN_STATS["miss"] += 1
        return

    observed = _captured_kernel_key_sequence()
    if observed is None:
        _GRAPH_PLAN_STATS["mismatch"] += 1
        return
    candidate = _select_graph_plan_candidate(exact_candidates, observed)
    if candidate is None:
        candidate = _select_graph_plan_candidate(structural_candidates, observed)
    if candidate is None:
        _GRAPH_PLAN_STATS["mismatch"] += 1
        return

    with _DTYPE_LOCK:
        for op_entry, kernels in zip(graph.get("ops") or (), candidate["kernels"]):
            op_entry["kernels"] = [
                {"name": str((n or {}).get("name") or ""),
                 "key": str((n or {}).get("key") or "")}
                for n in kernels
            ]
    _GRAPH_PLAN_STATS["hit"] += 1
    _bump_dirty()


def _claim_captured_kernel_nodes() -> "list[dict] | None":
    """Return kernel nodes added since the preceding captured operation."""
    gid = _active_capture_gid()
    if gid is None:
        return None
    if _graph_plan_available():
        return None
    nodes = _captured_kernel_nodes()
    if nodes is None:
        return None
    claim_gid = getattr(_CAPTURE_TLS, "claim_gid", None)
    if claim_gid != gid:
        _CAPTURE_TLS.claim_gid = gid
        _CAPTURE_TLS.claimed_nodes = set()
    claimed = getattr(_CAPTURE_TLS, "claimed_nodes", set())
    new_nodes = [n for n in nodes if n["handle"] not in claimed]
    claimed.update(n["handle"] for n in nodes)
    _CAPTURE_TLS.claimed_nodes = claimed
    return [{"name": n["name"], "key": n["key"]} for n in new_nodes]


def _next_graph_id() -> str:
    _GRAPH_SEQ[0] += 1
    return f"g{os.getpid()}_{_GRAPH_SEQ[0]}"


def _active_capture_gid() -> "str | None":
    stack = getattr(_CAPTURE_TLS, "stack", None)
    return stack[-1] if stack else None


def _capture_push(gid: str) -> None:
    stack = getattr(_CAPTURE_TLS, "stack", None)
    if stack is None:
        stack = []
        _CAPTURE_TLS.stack = stack
    stack.append(gid)


def _capture_pop() -> None:
    stack = getattr(_CAPTURE_TLS, "stack", None)
    if stack:
        stack.pop()


def _tensor_shapes(values) -> list:
    """Shapes of any tensor-like values, in argument order."""
    out: list = []
    for v in _iter_tensor_values(values):
        try:
            out.append([int(d) for d in v.shape])
        except Exception:
            continue
    return out


def _capture_record(
    op: str,
    stem: str,
    sig: str,
    dtypes: dict,
    shapes: list,
    kernels: "list[dict] | None" = None,
) -> None:
    """Append an op to the active capture (kernels: None = unknown, [] = none)."""
    gid = _active_capture_gid()
    if gid is None:
        return
    with _DTYPE_LOCK:
        g = _DTYPE_OBS["capture_graphs"].setdefault(gid, {"ops": [], "replays": 0})
        entry = {
            "op": op,
            "stem": stem or "",
            "sig": sig or "",
            "dtypes": dict(dtypes or {}),
            "shapes": [list(s) for s in (shapes or [])],
        }
        if kernels is not None:
            entry["kernels"] = [
                {"name": str(n.get("name") or ""), "key": str(n.get("key") or "")}
                for n in kernels
            ]
        g["ops"].append(entry)
    _bump_dirty()


def _select_replay_markers(ops: list) -> list:
    """Build exact-node markers, falling back to launcher stems."""
    markers: list = []
    for e in ops or ():
        e = e or {}
        stem = e.get("stem") or ""
        sig = e.get("sig") or ""
        if not sig:
            continue
        kernels = e.get("kernels")
        if kernels is not None:
            for node in kernels or ():
                key = (node or {}).get("key") or ""
                if key:
                    markers.append(("exact", key, sig))
                elif stem:
                    markers.append(("stem", stem, sig))
        elif stem:
            markers.append(("stem", stem, sig))
    return markers


_REPLAY_MARKER_CACHE: dict = {}


def _discard_graph_state(gid: "str | None") -> None:
    """Drop all instrumentation state for a reset or failed graph."""
    if gid is None:
        return
    with _DTYPE_LOCK:
        _DTYPE_OBS["capture_graphs"].pop(gid, None)
    _REPLAY_MARKER_CACHE.pop(gid, None)
    _clear_graph_node_name_cache(gid)


def _emit_graph_replay_markers(gid: str) -> None:
    """Re-emit the captured shape-marker sequence for one ``replay()``."""
    phase2 = _is_phase2()
    with _DTYPE_LOCK:
        g = _DTYPE_OBS["capture_graphs"].get(gid)
        if not g:
            return
        g["replays"] += 1
        if phase2:
            pairs = _REPLAY_MARKER_CACHE.get(gid)
            if pairs is None:
                try:
                    pairs = _select_replay_markers(g["ops"])
                except Exception:
                    pairs = []
                _REPLAY_MARKER_CACHE[gid] = pairs
        else:
            pairs = ()

    if not phase2:
        _bump_dirty()
        return
    for kind, owner, sig in pairs:
        if kind == "exact":
            _emit_graph_shape_marker(owner, sig)
        else:
            _emit_shape_marker(owner, sig)


def _triton_kernel_name(fn) -> "str | None":
    name = getattr(fn, "__name__", None)
    if not name:
        inner = getattr(fn, "fn", None)
        name = getattr(inner, "__name__", None)
    return name


def _record_triton(fn, args, kwargs, kernels: "list[dict] | None" = None) -> None:
    name = _triton_kernel_name(fn)
    if not name:
        return
    vals = list(args) + (list(kwargs.values()) if kwargs else [])
    sig = _shape_signature_from_values(vals)
    _emit_shape_marker(name, sig)

    seen: dict = {}
    _record_tensor_dtypes(args, seen)
    if kwargs:
        _record_tensor_dtypes(kwargs.values(), seen)

    if kernels != []:
        _capture_record(
            f"triton::{name}", name, sig, seen, _tensor_shapes(vals), kernels
        )

    if not seen and not sig:
        return
    with _DTYPE_LOCK:
        if seen:
            bucket = _DTYPE_OBS["triton"].setdefault(name, {})
            for dt, cnt in seen.items():
                bucket[dt] = bucket.get(dt, 0) + cnt
        if sig:
            scbucket = _DTYPE_OBS["triton_shape_counts"].setdefault(name, {})
            scbucket[sig] = scbucket.get(sig, 0) + 1
    _bump_dirty()


def _triton_observation_count() -> int:
    """Per-thread count used to recognize nested Triton launch wrappers."""
    return int(getattr(_CAPTURE_TLS, "triton_observations", 0))


def _observe_triton_launch(fn, args, kwargs) -> None:
    """Record one wrapper observation and make it visible to outer wrappers."""
    try:
        _record_triton(fn, args, kwargs, _claim_captured_kernel_nodes())
    finally:
        _CAPTURE_TLS.triton_observations = _triton_observation_count() + 1


def _patch_triton_jit() -> bool:
    """Wrap ``JITFunction.run`` to record per-kernel arg dtypes."""
    if not (_dtype_dir() or _is_phase2()):
        return True
    jit_mod = sys.modules.get("triton.runtime.jit")
    JITFunction = getattr(jit_mod, "JITFunction", None) if jit_mod else None
    if JITFunction is None:
        return False
    if not getattr(JITFunction, "_profilelens_dtype_patched", False):
        orig_run = JITFunction.run

        def _run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            result = orig_run(self, *args, **kwargs)
            try:
                _observe_triton_launch(self, args, kwargs)
            except Exception:
                pass
            return result

        JITFunction.run = _run
        JITFunction._profilelens_dtype_patched = True  # type: ignore[attr-defined]
    return True


def _patch_triton_autotuner() -> bool:
    """Wrap ``Autotuner.run`` (inductor's usual launch path) if already loaded."""
    if not (_dtype_dir() or _is_phase2()):
        return True
    auto_mod = sys.modules.get("triton.runtime.autotuner")
    Autotuner = getattr(auto_mod, "Autotuner", None) if auto_mod else None
    if Autotuner is None:
        return False
    if not getattr(Autotuner, "_profilelens_dtype_patched", False):
        orig_arun = Autotuner.run

        def _arun(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            observations_before = _triton_observation_count()
            result = orig_arun(self, *args, **kwargs)
            if _triton_observation_count() == observations_before:
                try:
                    _observe_triton_launch(getattr(self, "fn", self), args, kwargs)
                except Exception:
                    pass
            return result

        Autotuner.run = _arun
        Autotuner._profilelens_dtype_patched = True  # type: ignore[attr-defined]
    return True


def _flydsl_stem(name: "str | None") -> "str | None":
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


def _flydsl_fn_name(fn) -> "str | None":
    inner = getattr(fn, "func", None)
    return getattr(inner, "__name__", None) or getattr(fn, "__name__", None)


def _flydsl_named_args(fn, args, kwargs) -> list:
    """(arg_name, value) pairs via the wrapped kernel signature (fn.func)."""
    func = getattr(fn, "func", None) or fn
    try:
        import inspect
        sig = inspect.signature(func)
        bound = sig.bind(*args, **(kwargs or {}))
        bound.apply_defaults()
        out: list = []
        for pname, val in bound.arguments.items():
            kind = sig.parameters[pname].kind
            if kind == inspect.Parameter.VAR_POSITIONAL:
                out.extend((f"{pname}[{i}]", v) for i, v in enumerate(val))
            elif kind == inspect.Parameter.VAR_KEYWORD:
                out.extend((str(k), v) for k, v in val.items())
            else:
                out.append((pname, val))
        return out
    except Exception:
        return ([(f"arg{i}", v) for i, v in enumerate(args)]
                + [(str(k), v) for k, v in (kwargs or {}).items()])


def _flydsl_shape_signature(per_arg: list) -> str:
    """Canonical per-launch shape signature for same-kernel/multi-shape splitting."""
    class _S:
        __slots__ = ("shape",)
        def __init__(self, s): self.shape = s
    return _shape_signature_from_values(
        _S(shp) for _name, _label, shp in per_arg if shp is not None
    )


def _record_flydsl(
    fn, args, kwargs, kernels: "list[dict] | None" = None
) -> None:
    stem = _flydsl_stem(_flydsl_fn_name(fn))
    if not stem:
        return
    seen: dict = {}
    shapes: list = []
    per_arg: list = []
    for name, v in _flydsl_named_args(fn, args, kwargs):
        raw = str(getattr(v, "dtype", ""))
        dt = _TORCH_DTYPE_MAP.get(raw)
        try:
            shp = list(v.shape)
        except Exception:
            shp = None
        if dt is not None:
            seen[dt] = seen.get(dt, 0) + 1
            if shp is not None:
                shapes.append(shp)
        if shp is not None:
            per_arg.append((name, dt or raw.replace("torch.", "") or "?", shp))
    if not seen and not per_arg:
        return

    try:
        sig = _flydsl_shape_signature(per_arg)
    except Exception:
        sig = ""

    if kernels != []:
        _capture_record(f"flydsl::{stem}", stem, sig, seen, shapes, kernels)

    with _DTYPE_LOCK:
        bucket = _DTYPE_OBS["flydsl"].setdefault(stem, {})
        for dt, cnt in seen.items():
            bucket[dt] = bucket.get(dt, 0) + cnt
        if shapes and not _DTYPE_OBS["flydsl_shapes"].get(stem):
            _DTYPE_OBS["flydsl_shapes"][stem] = shapes
        abucket = _DTYPE_OBS["flydsl_args"].setdefault(stem, {})
        for name, label, shp in per_arg:
            e = abucket.get(name)
            if e is None:
                e = {"dtype": label, "shapes": [], "count": 0}
                abucket[name] = e
            e["count"] += 1
            if shp not in e["shapes"]:
                e["shapes"].append(shp)
        if sig:
            scbucket = _DTYPE_OBS["flydsl_shape_counts"].setdefault(stem, {})
            scbucket[sig] = scbucket.get(sig, 0) + 1
    _bump_dirty()

    _emit_shape_marker(stem, sig)


def _patch_flydsl_jit() -> bool:
    """Wrap FlyDSL ``JitFunction.__call__`` to record arg dtypes and shapes."""
    if not (_dtype_dir() or _is_phase2()):
        return True
    mod = sys.modules.get("flydsl.compiler.jit_function")
    JitFunction = getattr(mod, "JitFunction", None) if mod else None
    if JitFunction is None:
        return False
    if not getattr(JitFunction, "_profilelens_dtype_patched", False):
        orig_call = JitFunction.__call__

        def _call(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            result = orig_call(self, *args, **kwargs)
            try:
                _record_flydsl(self, args, kwargs, _claim_captured_kernel_nodes())
            except Exception:
                pass
            return result

        JitFunction.__call__ = _call
        JitFunction._profilelens_dtype_patched = True  # type: ignore[attr-defined]
    return True


def _make_capture_dtype_mode():
    """Build a TorchDispatchMode that inventories captured operations."""
    from torch.utils._python_dispatch import TorchDispatchMode  # type: ignore[import-not-found]

    class _CaptureDtypeMode(TorchDispatchMode):  # type: ignore[misc]
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            try:
                kernels = _claim_captured_kernel_nodes()
                seen: dict = {}
                _record_tensor_dtypes(args, seen)
                if kwargs:
                    _record_tensor_dtypes(kwargs.values(), seen)
                pkt = getattr(func, "_overloadpacket", func)
                opname = str(getattr(pkt, "__name__", pkt))
                vals = list(args) + (list(kwargs.values()) if kwargs else [])
                _capture_record(
                    f"aten::{opname}", "", _shape_signature_from_values(vals),
                    seen, _tensor_shapes(vals), kernels,
                )
                if seen:
                    with _DTYPE_LOCK:
                        opb = _DTYPE_OBS["capture_ops"].setdefault(opname, {})
                        for dt, cnt in seen.items():
                            opb[dt] = opb.get(dt, 0) + cnt
                            _DTYPE_OBS["capture_dtypes"][dt] = (
                                _DTYPE_OBS["capture_dtypes"].get(dt, 0) + cnt
                            )
                    _bump_dirty()
            except Exception:
                pass
            return result

    return _CaptureDtypeMode


def _patch_graph_capture() -> bool:
    """Record graph capture metadata and emit replay shape markers."""
    if not (_dtype_dir() or _is_phase2()):
        return True
    if "torch.cuda" not in sys.modules:
        return False
    torch = sys.modules.get("torch")
    cg_cls = getattr(getattr(torch, "cuda", None), "CUDAGraph", None)
    if cg_cls is None:
        return False

    if getattr(cg_cls, "_profilelens_dtype_patched", False):
        return True
    try:
        mode_cls = _make_capture_dtype_mode()
    except Exception:
        return False

    orig_begin = cg_cls.capture_begin
    orig_end = cg_cls.capture_end
    orig_replay = getattr(cg_cls, "replay", None)
    orig_reset = getattr(cg_cls, "reset", None)
    active: dict = {}

    def _begin(self, *a, **k):  # type: ignore[no-untyped-def]
        _discard_graph_state(getattr(self, "_profilelens_gid", None))
        gid = _next_graph_id()
        self._profilelens_gid = gid  # type: ignore[attr-defined]
        _capture_push(gid)
        with _DTYPE_LOCK:
            _DTYPE_OBS["capture_graphs"].setdefault(gid, {"ops": [], "replays": 0})
        try:
            m = mode_cls()
            m.__enter__()
            active[id(self)] = m
        except Exception:
            pass
        try:
            return orig_begin(self, *a, **k)
        except BaseException:
            m = active.pop(id(self), None)
            if m is not None:
                try:
                    m.__exit__(*sys.exc_info())
                except Exception:
                    pass
            _capture_pop()
            _discard_graph_state(gid)
            try:
                del self._profilelens_gid
            except Exception:
                pass
            raise

    def _end(self, *a, **k):  # type: ignore[no-untyped-def]
        gid = getattr(self, "_profilelens_gid", None)
        if gid is not None:
            try:
                _apply_graph_plan(gid)
            except Exception:
                pass
        end_exc = (None, None, None)
        try:
            res = orig_end(self, *a, **k)
        except BaseException:
            end_exc = sys.exc_info()
            if gid is not None:
                _discard_graph_state(gid)
                try:
                    del self._profilelens_gid
                except Exception:
                    pass
            raise
        finally:
            m = active.pop(id(self), None)
            if m is not None:
                try:
                    m.__exit__(*end_exc)
                except Exception:
                    pass
            _clear_graph_node_name_cache(gid)
            _capture_pop()
        return res

    def _replay(self, *a, **k):  # type: ignore[no-untyped-def]
        result = orig_replay(self, *a, **k)
        gid = getattr(self, "_profilelens_gid", None)
        if gid is not None:
            try:
                _emit_graph_replay_markers(gid)
            except Exception:
                pass
        return result

    def _reset(self, *a, **k):  # type: ignore[no-untyped-def]
        result = orig_reset(self, *a, **k)
        gid = getattr(self, "_profilelens_gid", None)
        _discard_graph_state(gid)
        try:
            del self._profilelens_gid
        except Exception:
            pass
        return result

    cg_cls.capture_begin = _begin
    cg_cls.capture_end = _end
    if orig_replay is not None:
        cg_cls.replay = _replay
    if orig_reset is not None:
        cg_cls.reset = _reset
    cg_cls._profilelens_dtype_patched = True  # type: ignore[attr-defined]
    return True


def _install_dtype_hooks() -> None:
    """Register the dtype-observation hooks."""
    if not (_dtype_dir() or _is_phase2()):
        return
    if getattr(sys, "_profilelens_dtype_installed", False):
        return
    sys._profilelens_dtype_installed = True  # type: ignore[attr-defined]
    for mod_key, patch_fn in (
        ("triton.runtime.jit", _patch_triton_jit),
        ("triton.runtime.autotuner", _patch_triton_autotuner),
        ("flydsl.compiler.jit_function", _patch_flydsl_jit),
        ("torch.cuda", _patch_graph_capture),
    ):
        if not (mod_key in sys.modules and patch_fn()):
            _PENDING_PATCHES[mod_key] = patch_fn
    if _PENDING_PATCHES:
        _install_import_hook()
    atexit.register(_write_dtype_sidecar)


_PENDING_PATCHES: dict[str, callable] = {  # type: ignore[type-arg]
    "torch.profiler": _patch_torch_profiler_both,
    "torch.cuda.profiler": _patch_torch_cuda_profiler,
}


def _install_import_hook() -> None:
    """Install the import hook for pending module patches."""
    if getattr(sys, "_profilelens_import_hook_installed", False):
        return
    sys._profilelens_import_hook_installed = True  # type: ignore[attr-defined]

    orig_import = builtins.__import__
    tls = threading.local()

    def _wrapped(name, globals=None, locals=None, fromlist=(), level=0):
        module = orig_import(name, globals, locals, fromlist, level)
        if not _PENDING_PATCHES:
            return module
        if getattr(tls, "in_hook", False):
            return module
        tls.in_hook = True
        try:
            for mod_name in list(_PENDING_PATCHES.keys()):
                if mod_name in sys.modules:
                    patch_fn = _PENDING_PATCHES.get(mod_name)
                    if patch_fn is None:
                        continue
                    try:
                        if patch_fn():
                            _PENDING_PATCHES.pop(mod_name, None)
                    except Exception:
                        _PENDING_PATCHES.pop(mod_name, None)
        finally:
            tls.in_hook = False
        return module

    builtins.__import__ = _wrapped  # type: ignore[assignment]


def _install_trigger_hook() -> None:
    audit_dir = os.environ.get("PROFILELENS_AUDIT_DIR")
    if not audit_dir:
        return
    if getattr(sys, "_profilelens_trigger_installed", False):
        return
    sys._profilelens_trigger_installed = True  # type: ignore[attr-defined]
    try:
        os.makedirs(audit_dir, exist_ok=True)
    except OSError:
        pass
    if "torch.profiler" in sys.modules and _patch_torch_profiler_both():
        _PENDING_PATCHES.pop("torch.profiler", None)
    if "torch.cuda.profiler" in sys.modules and _patch_torch_cuda_profiler():
        _PENDING_PATCHES.pop("torch.cuda.profiler", None)
    if _PENDING_PATCHES:
        _install_import_hook()
    _write_audit_now()
    atexit.register(_write_audit_now)


_install_finalize_hook()
_install_trigger_hook()
_install_dtype_hooks()
