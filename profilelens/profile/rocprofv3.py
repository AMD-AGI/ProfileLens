# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Wrapper for rocprofv3 command building and execution."""

from __future__ import annotations

import glob as _glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

# ProfileLens ships a region-relative patched librocprofiler-sdk-tool.so here rocprof-patch/7.x/.
_ROCPROF_PATCH_DIR = Path(__file__).resolve().parents[2] / "rocprof-patch"


def find_rocprofv3(config_path: str = "") -> str:
    """Locate the rocprofv3 binary."""
    if config_path and Path(config_path).is_file():
        return config_path

    found = shutil.which("rocprofv3")
    if found:
        return found

    rocm_path = Path("/opt/rocm/bin/rocprofv3")
    if rocm_path.is_file():
        return str(rocm_path)

    # In multi-ROCm setup, pick the highest rocprofv3 version.
    for cand in sorted(_glob.glob("/opt/rocm-*/bin/rocprofv3"), reverse=True):
        if Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand

    raise RuntimeError(
        "rocprofv3 not found.\n"
        "  On Ubuntu (ROCm 7.x): sudo apt install rocprofiler-sdk rocprofiler-sdk-roctx\n"
        "  Docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/install/installation.html\n"
        "  Or set profiling.rocprofv3_path in your profilelens config."
    )


def find_rocprofv3_avail(rocprofv3_path: str = "") -> str:
    """Locate the ``rocprofv3-avail`` paired with the selected profiler."""
    if rocprofv3_path:
        sibling = Path(rocprofv3_path).resolve().with_name("rocprofv3-avail")
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return str(sibling)

    found = shutil.which("rocprofv3-avail")
    if found:
        return found

    versioned = sorted(_glob.glob("/opt/rocm-*/bin/rocprofv3-avail"), reverse=True)
    for candidate in (Path("/opt/rocm/bin/rocprofv3-avail"), *map(Path, versioned)):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    raise RuntimeError(
        "rocprofv3-avail not found. Install the ROCprofiler-SDK tools from the "
        "same ROCm release as rocprofv3."
    )


def _parse_counter_info(output: str) -> frozenset[str]:
    """Extract counter names from ``rocprofv3-avail info --pmc`` output."""
    return frozenset(
        match.group(1)
        for match in re.finditer(
            r"(?m)^\s*Counter_Name\s*:\s*(\S+)\s*$",
            output,
        )
    )


def list_available_counters(
    rocprofv3_path: str,
    device_index: int = 0,
) -> frozenset[str]:
    """Return the counter inventory reported for one runtime GPU."""
    avail = find_rocprofv3_avail(rocprofv3_path)
    result = subprocess.run(
        [avail, "-d", str(device_index), "info", "--pmc"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    counters = _parse_counter_info(result.stdout) if result.returncode == 0 else frozenset()
    if not counters:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"could not read the runtime PMC inventory: {detail}")
    return counters


def counter_set_supported(
    rocprofv3_path: str,
    counters: tuple[str, ...],
    device_index: int = 0,
) -> tuple[bool, str]:
    """Ask ROCprofiler-SDK whether *counters* fit in one pass on the GPU."""
    if not counters:
        return True, ""
    avail = find_rocprofv3_avail(rocprofv3_path)
    result = subprocess.run(
        [avail, "-d", str(device_index), "pmc-check", *counters],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout.strip()
    detail = (result.stderr or output).strip()
    success_marker = "Following input counters can be collected together"
    supported = result.returncode == 0 and success_marker in output
    if not supported and not detail:
        detail = "rocprofv3-avail rejected the combined counter set"
    return supported, detail


def _read_rocprofv3_versions(rocprofv3_path: str) -> tuple[int | None, int | None, str]:
    """Parse `rocprofv3 --version` into (sdk_soversion_major, rocm_major, raw_text)."""
    try:
        out = subprocess.run(
            [rocprofv3_path, "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except Exception as e:  # noqa: BLE001 -- surfaced as a hard failure below
        raise RuntimeError(
            f"could not run `{rocprofv3_path} --version`: {e}"
        ) from e

    sdk_m = re.search(r"(?m)^\s*version:\s*(\d+)\.", out)
    rocm_m = re.search(r"(?m)^\s*rocm_version:\s*(\d+)\.", out)
    sdk_major = int(sdk_m.group(1)) if sdk_m else None
    rocm_major = int(rocm_m.group(1)) if rocm_m else None
    return sdk_major, rocm_major, out


def prepare_patched_rocprofv3(
    rocprofv3_path: str,
    work_dir: str,
    console: Console | None = None,
) -> tuple[str, str]:
    """Build a ``--rocm-root`` prefix that overlays ProfileLens's region-relative
    patched ``librocprofiler-sdk-tool.so`` over stock rocprof."""
    sdk_major, rocm_major, raw = _read_rocprofv3_versions(rocprofv3_path)
    if sdk_major != 1 or rocm_major is None or rocm_major < 7:
        raise RuntimeError(
            "ProfileLens requires a patchable rocprofv3 (rocprofiler-sdk "
            "soversion 1 and ROCm major >= 7) to collect PMC counters correctly.\n"
            f"  rocprofv3: {rocprofv3_path}\n"
            f"  detected sdk soversion: {sdk_major}, rocm major: {rocm_major}\n"
            "  The region-relative kernel-iteration fix is only available for the "
            "ROCm 7.x rocprof-sdk tool. Incompatible rocprofv3.\n"
            f"  `rocprofv3 --version` reported:\n{raw}"
        )

    patched_tool = (
        _ROCPROF_PATCH_DIR / f"{rocm_major}.x" / "librocprofiler-sdk-tool.so"
    )
    if not patched_tool.is_file():
        # The region-relative counter-collection fix that ProfileLens bundles as a
        # patched tool lib for ROCm 7.x (upstream PR ROCm/rocm-systems#7299) is
        # merged in the rocprofiler-sdk-tool shipped with ROCm 10+ (verified in the
        # SDK 1.3.5 source: `defer_counter_start` + `create_pause_resume_ctx` gate
        # the counting context on roctxProfilerResume/Pause). On those releases the
        # stock tool already does region-relative counting, so run without an
        # overlay instead of failing.
        if rocm_major >= 10:
            if console:
                console.print(
                    f"  [yellow]No bundled patched tool lib for ROCm {rocm_major}.x; "
                    f"using stock rocprofv3 (the region-relative counter fix is "
                    f"upstream in this SDK).[/yellow]"
                )
            return "", ""
        raise RuntimeError(
            f"ProfileLens has no patched tool lib for ROCm {rocm_major}.x: "
            f"{patched_tool} is missing.\n"
            "  Build one from the matching rocprofiler-sdk source with the "
            "region-relative patch (see rocprof-patch/<major>.x/README.md)."
        )

    # System ROCm lib dir: .../bin/rocprofv3 -> .../lib
    install_root = Path(rocprofv3_path).resolve().parent.parent
    sys_lib = install_root / "lib"
    sys_tool_dir = sys_lib / "rocprofiler-sdk"
    if not sys_tool_dir.is_dir():
        raise RuntimeError(
            f"could not locate the ROCm lib dir for {rocprofv3_path} "
            f"(expected {sys_tool_dir} to exist)."
        )


    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    for stale in work.glob(".rocprof_prefix*"):
        shutil.rmtree(stale, ignore_errors=True)
    prefix = work / f".rocprof_prefix_{time.strftime('%Y%m%d%H%M%S')}"
    plib = prefix / "lib"
    ptool_dir = plib / "rocprofiler-sdk"
    ptool_dir.mkdir(parents=True, exist_ok=True)

    def _relink(target, link: Path) -> None:

        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link, ignore_errors=True)
        link.symlink_to(target)

    for f in sys_lib.glob("librocprofiler-sdk*.so*"):
        _relink(f.resolve(), plib / f.name)
    for f in sys_tool_dir.glob("*"):
        _relink(f.resolve(), ptool_dir / f.name)

    # correctly symlink .so.1.1.0 with the conventional .so.1 / .so
    for f in list(ptool_dir.glob("librocprofiler-sdk-tool.so*")):
        f.unlink()
    real_tool = ptool_dir / "librocprofiler-sdk-tool.so.1.1.0"
    shutil.copy2(patched_tool, real_tool)
    _relink("librocprofiler-sdk-tool.so.1.1.0", ptool_dir / "librocprofiler-sdk-tool.so.1")
    _relink("librocprofiler-sdk-tool.so.1", ptool_dir / "librocprofiler-sdk-tool.so")

    # Mirror share/rocprofiler-sdk (counter definitions) for the relative lookup.
    metrics_path = ""
    sys_share = install_root / "share" / "rocprofiler-sdk"
    if sys_share.is_dir():
        _relink(sys_share.resolve(), prefix / "share")  # prefix/share -> .../share/rocprofiler-sdk
        metrics_path = str(sys_share.resolve())
        # prefix/lib/../share/rocprofiler-sdk must also resolve; provide that name too.
        (prefix / "share").unlink()
        (prefix / "share").mkdir(exist_ok=True)
        _relink(sys_share.resolve(), prefix / "share" / "rocprofiler-sdk")

    if console:
        console.print(
            f"  [bold green]Using region-relative patched rocprofv3[/bold green] "
            f"(ROCm {rocm_major}.x, sdk soversion {sdk_major})"
        )
        console.print(f"  [dim]--rocm-root {prefix}[/dim]")

    return str(prefix), metrics_path


def build_command(
    rocprofv3_path: str,
    output_dir: str,
    counters: tuple[str, ...],
    kernel_regex: str,
    user_command: list[str],
    output_format: str = "csv",
    iteration_ids: list[int] | None = None,
    disable_signal_handlers: bool = True,
    process_sync: bool = True,
    kernel_trace: bool = True,
    selected_regions: bool = True,
    marker_trace: bool = True,
    rocm_root: str = "",
) -> list[str]:
    """Build the full rocprofv3 command line."""
    cmd = [rocprofv3_path]
    if rocm_root:
        cmd.extend(["--rocm-root", rocm_root])
    cmd.extend(["--output-directory", output_dir])
    if kernel_trace:
        cmd.append("--kernel-trace")
    if marker_trace:
        cmd.append("--marker-trace")
    cmd.extend(["--output-format", output_format])

    if disable_signal_handlers:
        cmd.extend(["--disable-signal-handlers", "true"])

    if process_sync:
        cmd.extend(["--process-sync", "true"])

    if selected_regions:
        cmd.extend(["--selected-regions", "true"])

    if counters:
        cmd.append("--pmc")
        cmd.extend(counters)

    if kernel_regex:
        cmd.extend(["--kernel-include-regex", kernel_regex])

    if iteration_ids:

        ids = sorted(set(iteration_ids))
        ranges: list[tuple[int, int]] = []
        for i in ids:
            if ranges and i == ranges[-1][1] + 1:
                ranges[-1] = (ranges[-1][0], i)
            else:
                ranges.append((i, i))
        cmd.append("--kernel-iteration-range")
        cmd.extend(f"[{lo}-{hi}]" for lo, hi in ranges)

    cmd.append("--")
    cmd.extend(user_command)
    return cmd


def strip_kd_decoration(name: str) -> str:
    """Strip the ROCm perfetto/kineto kernel-descriptor decoration.

    On ROCm the torch/perfetto trace decorates GPU kernel names with a
    kernel-descriptor suffix (trailing ``.kd`` and/or `` [clone .kd]``) that
    rocprofv3's *dispatch* names do not carry. Normalizing to the undecorated
    form lets trace-derived names (Phase 1) and dispatch names (Phase 2) join.
    """
    import re as _re

    n = _re.sub(r"\s*\[clone\s+\.kd\]\s*$", "", name)
    n = _re.sub(r"\.kd\s*$", "", n)
    return n


def build_kernel_regex(kernel_names: list[str]) -> str:
    """Build a regex matching any of the given kernel names.

    Matching the decorated name with an anchored ``^(...)$`` selects zero kernels
    for counter collection (``--kernel-include-regex``), yielding an empty
    ``*_counter_collection.csv``. Strip the ``.kd`` decoration and allow it
    optionally so the include-regex matches the runtime dispatch name.
    """
    if not kernel_names:
        return ""
    import re as _re

    pats = [
        _re.escape(strip_kd_decoration(name)) + r"(?:\.kd)?(?:\s*\[clone\s+\.kd\])?"
        for name in kernel_names
    ]
    return "^(" + "|".join(pats) + ")$"


def run_profile(
    rocprofv3_path: str,
    output_dir: str,
    counters: tuple[str, ...],
    kernel_regex: str,
    user_command: list[str],
    console: Console | None = None,
    iteration_ids: list[int] | None = None,
) -> tuple[bool, str]:
    """Execute rocprofv3 and return (success, stderr_output)."""
    from profilelens.profile.server_mode import detect_server, build_wrapper_script

    os.makedirs(output_dir, exist_ok=True)

    # Region-relative PMC is mandatory for correct counter attribution; fail if patched rocprofv3 cannot be used
    rocm_root, metrics_path = prepare_patched_rocprofv3(
        rocprofv3_path, output_dir, console=console
    )

    server_info = detect_server(user_command)
    final_command = user_command
    server_mode = server_info is not None

    # Region-relative --selected-regions requires ProfileLens's patched tool lib.
    # On a stock (non-region-relative) rocprofv3 it records nothing even though
    # roctx markers fire, so allow opting out: profile ALL dispatches of the
    # top-K kernels (scoped by --kernel-include-regex) across the run instead.
    _selected_regions = os.environ.get("PROFILELENS_SELECTED_REGIONS", "true").lower() != "false"

    if server_mode:
        rocp_prefix = build_command(
            rocprofv3_path, output_dir, counters, kernel_regex,
            user_command=[],
            iteration_ids=iteration_ids,
            disable_signal_handlers=True,
            kernel_trace=False,
            selected_regions=_selected_regions,
            rocm_root=rocm_root,
        )
        if rocp_prefix and rocp_prefix[-1] == "--":
            rocp_prefix.pop()

        wrapper_path = Path(output_dir) / "profilelens_server_wrapper.sh"
        final_command = build_wrapper_script(
            server_info, rocp_prefix, wrapper_path, user_command,
        )
        cmd = final_command
        if console:
            console.print(
                f"  [bold green]Server-mode detected[/bold green] "
                f"({server_info.cmd_name}): rocprofv3 will wrap [italic]"
                f"{server_info.cmd_name} serve[/italic] directly, not bash."
            )
            console.print(f"  [dim]Wrapper written to: {wrapper_path}[/dim]")
    else:
        cmd = build_command(
            rocprofv3_path, output_dir, counters, kernel_regex, user_command,
            iteration_ids=iteration_ids,
            selected_regions=_selected_regions,
            rocm_root=rocm_root,
        )

    cmd_log_path = os.path.join(output_dir, "rocprofv3_cmd.log")
    with open(cmd_log_path, "w") as f:
        f.write(" ".join(repr(a) if " " in a else a for a in cmd) + "\n")

    if console:
        console.print(f"  [dim]Running: {' '.join(cmd[:6])} ... {' '.join(cmd[-3:])}[/dim]")
        console.print(f"  [dim]Full command saved to: {cmd_log_path}[/dim]")

    stdout_path = os.path.join(output_dir, "rocprofv3_stdout.log")
    stderr_path = os.path.join(output_dir, "rocprofv3_stderr.log")

    shim_dir = str(Path(__file__).resolve().parent / "_worker_shim")
    child_env = os.environ.copy()
    existing_pp = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        shim_dir + os.pathsep + existing_pp if existing_pp else shim_dir
    )
    child_env["PROFILELENS_ROCPROF_FINALIZE"] = "1"
    audit_dir = os.path.join(output_dir, ".profilelens_audit")
    os.makedirs(audit_dir, exist_ok=True)
    child_env["PROFILELENS_AUDIT_DIR"] = audit_dir
    child_env["PROFILELENS_PHASE2"] = "1"
    # Reuse Phase 1's graph-kernel inventory when available.
    if "PROFILELENS_GRAPH_PLAN" not in child_env:
        base = Path(output_dir)
        for candidate in (base / ".profilelens_dtype_phase1",
                          base.parent / ".profilelens_dtype_phase1"):
            if candidate.is_dir() and any(candidate.glob("*.json")):
                child_env["PROFILELENS_GRAPH_PLAN"] = str(candidate)
                break
    if metrics_path:
        child_env["ROCPROFILER_METRICS_PATH"] = metrics_path
    lib_dir = Path(rocm_root) / "lib"
    if lib_dir.is_dir():
        existing_ld = child_env.get("LD_LIBRARY_PATH", "")
        child_env["LD_LIBRARY_PATH"] = (
            str(lib_dir) + os.pathsep + existing_ld if existing_ld else str(lib_dir)
        )

    from profilelens.profile._proc_utils import reap_process_group

    try:
        with open(stdout_path, "w") as so, open(stderr_path, "w") as se:
            proc = subprocess.Popen(
                cmd, stdout=so, stderr=se, text=True, env=child_env,
                start_new_session=True,
            )
            try:
                returncode = proc.wait(timeout=3600)
            finally:
                reap_process_group(proc.pid, console=console, label="Phase 2")
    except subprocess.TimeoutExpired:
        reap_process_group(proc.pid, console=console, label="Phase 2 (timeout)")
        return False, "rocprofv3 timed out after 3600s"
    except FileNotFoundError:
        return False, f"rocprofv3 not found at {rocprofv3_path}"

    class _R:  # minimal shim so the existing returncode check below still works
        pass
    result = _R()
    result.returncode = returncode

    try:
        with open(stderr_path) as f:
            stderr_text = f.read()
    except OSError:
        stderr_text = ""

    if result.returncode != 0:
        try:
            with open(stdout_path) as f:
                stdout_text = f.read()
        except OSError:
            stdout_text = ""
        msg = stderr_text or stdout_text or f"Exit code {result.returncode}"
        return False, msg

    return True, stderr_text
