# ProfileLens

GPU performance analysis pipeline that combines Perfetto trace analysis with
hardware counter profiling and empirical roofline modeling.  ProfileLens
identifies the hottest kernels from a trace, resolves each kernel's compute
dtype from live execution, profiles them with `rocprofv3` in a single pass,
classifies bottlenecks against per-dtype empirical roofline ceilings, and ranks
kernels by the end-to-end time recoverable if each reached its ceiling.

The primary entry point is **`full-analysis`** — one command takes a workload to
a ranked, consolidated kernel report. 

## Install

```bash
pip install -e .
```

Requires Python 3.10+.

### System Requirements

- AMD GPU supported by the installed ROCm release
- ROCm providing `rocprofv3` and `rocprofv3-avail` (HW counters)
- The GPU is auto-detected at runtime; no manual configuration needed
- Optional: `llvm-cxxfilt` or `c++filt` to match mangled C++ kernel names

ROCprofiler-SDK 1.4.1+ is used natively. Older ROCm 7.x installs use the
bundled, temporary patched `librocprofiler-sdk-tool.so`; see
[`rocprof-patch/7.x/README.md`](rocprof-patch/7.x/README.md) for its source,
license, checksum, rebuild instructions, upstream status, and removal plan.

### Optional one-time setup

Pre-fetch the roofline benchmark so later runs can use it:

```bash
profilelens setup-roofline --measure --device 0   # fetch the benchmark, measure peaks now on specified device and cache the results
```

If skipped, `full-analysis` fetches it automatically on first use.

## Quick Start — `full-analysis`

Run the workload, analyze the trace, profile the top-K kernels, and produce a
consolidated report — all in one command:

```bash
profilelens full-analysis \
  --top-k 10 \
  --output-dir ./profilelens_run \
  -- python my_workload.py
```

The workload must have `torch.profiler` / kineto tracing enabled internally,
and must join its worker processes before exiting so profiling output is flushed.

ProfileLens will:

1. Execute the workload (with a shim that hooks Pytorch profiler, triton launches and HIP/CUDA
   graph capture) and generate the trace (Phase 1)
2. Analyze the trace to identify the top-K kernels and resolve each
   kernel's compute dtype 
3. Measure empirical GPU peaks with the fetched roofline benchmark
4. Re-run the workload under `rocprofv3` with a single-pass, multi-dtype-aware
   HW counter set scoped to the top-K kernels (Phase 2)
5. Compute per-kernel metrics, score each against its own dtype's roofline,
   classify bottlenecks, rank by recoverable end-to-end time, and generate the
   consolidated report

Useful options: `--top-k` (default 10), `--output-dir`,
`--refresh-roofline` (force a fresh roofline benchmark measurement), `--output {rich,json,markdown}`,
`--workload-timeout` (total seconds, default no limit), `--workload-idle-timeout`
(stop after this many seconds without progress, default 1800).

### Counter coverage

Kernels whose dtype cannot be resolved are reported as `UNKNOWN`; they are never
assigned another kernel's dtype or compute ceiling. Counter coverage can be steered
without changing that attribution:

```bash
# Add MFMA counter families for these dtypes
profilelens full-analysis --dtypes BF16,FP8,FP4 -- python my_workload.py

# Or request exact counters (must fit in one pass; never silently dropped)
profilelens full-analysis \
  --counters SQ_INSTS_VALU_MFMA_MOPS_BF16,SQ_INSTS_VALU_MFMA_MOPS_F8 \
  -- python my_workload.py
```

**Outputs** (saved to `--output-dir`):

| File | Description |
|------|-------------|
| `REPORT.md` | Human-readable report: kernel data, HW counters, per-dtype peaks, gap analysis |
| `report.json` | Structured JSON: kernel data, HW counters, per-dtype peaks, gap analysis |
| `analysis.json` | Phase 1 Perfetto trace |
| `roofline.png` | Roofline scatter of the top-K kernels |
| `dtype_resolution.json` | Per-kernel dtype sources, unresolved kernels and counter coverage |

### Configuration

Set `PROFILELENS_HOME` to relocate the config file and roofline caches
(default `~/.profilelens`).

### Python capture API

`profilelens.profile.capture` collects the trace and counters in a single workload
run and analyzes saved windows without replay (`prepare_capture`,
`build_capture_command`, `analyze_capture_window`). Requires ROCprofiler-SDK 1.4.1+.

---

## How it works

### 1. Dtype detection

Modern inference mixes precisions. ProfileLens resolves **each kernel's compute dtype** so it can enable the right HW counters and score the kernel against the right roofline ceiling. It does this from *execution* hooks:

1. **torch.profiler hook** — links a GPU kernel to its launching `aten` CPU op input dtype
2. **triton.runtime hook** — records the dtypes of tensor args at each Triton kernel
   launch
3. **torch.CUDAGraph hook** — records op dtypes and operand shapes during graph
   capture and restores them for replayed kernel dispatches
4. **MLIR-DSL launch hook** — records dtypes (and arg shapes) at each MLIR-based DSL
   kernel launch e.g. FlyDSL. 

### 2. Single-pass, multi-dtype-aware counters

HW counters are collected via `rocprofv3` in **one profiling pass**, scoped to the
top-K kernels. The set adapts to the observed dtypes and the installed SDK's
runtime counter inventory and single-pass compatibility result:

- one `MFMA_MOPS` counter per observed compute dtype (BF16/FP16/FP8/…),
- VALU instructions (vector-engine FLOPs/utilization),
- `SQ_WAVE_CYCLES` (time-integrated occupancy),
- VMEM/LDS issue-stall counters,
- HBM byte counters and L1/L2 cache-hit signals.

The exact counter count is selected at runtime. If all desired signals do not
fit in one pass, lower-priority optional signals are omitted and reported.

### 3. Empirical roofline

ProfileLens measures **empirical** peaks rather than using theoretical specs. So ProfileLens **fetches the upstream rocprof-compute roofline microbenchmark** and runs it against the local ROCm. It measures HBM bandwidth and a per-dtype MFMA ceiling, cached per GPU model in `~/.profilelens/roofline_cache/` See `setup-roofline` to pre-seed it.

### 4. Ranking & per-kernel metrics

Kernels are selected by top-K E2E time, then ranked by **Max E2E Gain**:

```
Max E2E Gain = (1 - %Roofline/100) x E2E Time%
%Roofline    = achieved / attainable-at-this-AI, vs the roofline ceiling
```

This deprioritizes kernels already near their ceiling even if they dominate E2E time.

Each profiled kernel's detail includes:

- **From trace** (Phase 1): E2E time share, dispatch count, op/shape, dtype.
- **From roofline & rocprof profiling** (Phase 2): 
  - **Arithmetic Intensity** (FLOP/byte).
  - **Achieved BW** (% of HBM peak). 
  - **Achieved FLOPs** (% of the kernel's dtype MFMA peak).
  - **VALU (vector engine) % busy** (vector-ALU duty cycle).
  - **Occupancy** (concurrent-wave residency), L1/L2 hit rates, VMEM/LDS issue-stall. 
- **Classification**: COMPUTE / MEMORY / LATENCY / BALANCED.
- **Gap analysis**: deterministic hints from metric thresholds.
- **Roofline chart** (`roofline.png`): kernels plotted against the HBM diagonal and
  one MFMA ceiling per observed dtype.

Metric notes:

- **FLOPs** = sum of collected floating-point MFMA families (512 ops per `MFMA_MOPS` unit).
- **Memory traffic** comes from SDK `FETCH_SIZE`/`WRITE_SIZE` or DRAM request counters.
- **Occupancy / VALU busy** use the SDK's `OccupancyPercent` / `VALUBusy`.
- Missing counters are reported as `N/A` (`null` in JSON), never as zero.

#### Report table shape

```
Top-10 Kernels by E2E Time — Ranked by Max E2E Gain

 # | Kernel | E2E Time% | Type | Dtype | FLOP/byte (AI) | % of HW peak | Max E2E Gain% | Stall
```

Measured example results are intentionally not included in the public source
tree. Generate a report from your own workload to populate these fields.

#### Pipeline

```
  Phase 1: Trace collection & analysis
  ------------------------------------
    user workload  (shim: Pytorch Profiler, Triton, CUDA-graph dtype hooks)
              |  ->  torch.profiler trace  +  per-kernel dtype facts
              v
    perfetto trace-processor  ->  top-K kernels + per-kernel dtypes + analysis.json

  Phase 2: HW counter profiling & report
  --------------------------------------
    fetched roofline benchmark                       ->  cached empirical peaks
              |
    rocprofv3 -- user workload                       (multi-dtype counters,
              |                                        --selected-regions = top-K)
              v
    *_counter_collection.csv  ->  parse + metrics + bottleneck + roofline plot
              |
              v
    report.json  +  REPORT.md
```

## Architecture

```
profilelens/
  cli.py                  # Typer CLI entry point
  config.py               # Configuration management
  dtypes.py               # Core data types (KernelProfile, RooflinePeaks, ...)
  trace/                  # Perfetto trace analysis (Phase 1)
    processor.py          #   TraceProcessor wrapper (auto-decompresses .json.gz)
    queries.py            #   Pre-built SQL queries
    formatter.py          #   JSON / Markdown / Rich output
    categories.py         #   Op categorization (gemm, attention, norm, …)
  profile/                # HW counter profiling (Phase 2)
    runner.py             #   Sets up the workload to be profiled
    _worker_shim/         #   shim: hooks torch.profiler, triton launch,
                          #     HIP/CUDA-graph capture and replay context
    gpu_detect.py         #   Discover GPU properties via rocprofv3-avail
    counter_sets.py       #   Runtime-capability-aware, multi-dtype counter plan
    dtype_detect.py       #   Per-kernel dtype resolution 
    rocprofv3.py          #   rocprofv3 PMC counter builder & runner
    roofline_bench.py     #   Fetch, run and cache the empirical rocprof-compute roofline benchmark
    parser.py             #   CSV parser for rocprofv3 output
    kernel_context.py     #   Phase 1 → per-kernel Module / Op / Shape / dtype join
    metrics.py            #   Derived metric computation
    bottleneck.py         #   Bottleneck classification
    roofline.py           #   Per-dtype roofline model (parses empirical peaks)
    roofline_plot.py      #   roofline.png chart
    report.py             #   Consolidated report formatter
```

---

## Other commands

The primary workflow is `full-analysis` above. These secondary commands support
narrower tasks.

### Standalone kernel profiling — `profile`

Profile specific kernels (by name, or from a prior `analyze` JSON) without
re-deriving the whole pipeline — useful for iterating on a known kernel:

```bash
profilelens profile --from-analysis analysis.json --top-k 10 \
  --output-dir ./profilelens_run -- python my_workload.py
```

Key options: `--kernel-names` (exact names), `--from-analysis`, `--trace`, `--top-k`,
`--counter-set auto|custom`, `--counters`, `--dtypes`, `--run-roofline/--no-roofline`,
`--refresh-roofline`, `--output {rich,json,markdown}`.

### Trace analysis — `analyze` / `query`

Run the pre-built deterministic analysis or inspect a trace with your own SQL:

```bash
profilelens analyze trace.json          # pre-built SQL queries
profilelens query / tables / schema     # raw SQL, table list, column schema
```

Use `--output rich`, `--output json`, or `--output markdown` with `analyze`
to select the output format.

### Command reference

| Command | Purpose |
|---------|---------|
| `full-analysis` | **Primary** — trace + analysis + profiling + consolidated report |
| `setup-roofline` | Fetch (and optionally `--measure`) the empirical roofline benchmark |
| `profile` | Profile specific kernels with HW counters + empirical roofline |
| `analyze` | Pre-built deterministic SQL queries over a trace |
| `query` / `tables` / `schema` | Raw SQL / table list / column schema |

## License

ProfileLens is licensed under the MIT License. See [`LICENSE.TXT`](LICENSE.TXT).
Bundled and runtime-downloaded third-party material is documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Contact

For questions, issues, or contributions, please reach out to the maintainers:

- Hari Thangirala — [@hthangirala](https://github.com/hthangirala) · hari.thangirala@amd.com

For bugs and feature requests, open a [GitHub Issue](https://github.com/AMD-AGI/ProfileLens/issues).
