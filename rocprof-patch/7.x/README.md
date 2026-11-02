# Patched `librocprofiler-sdk-tool.so` (ROCm 7.x)

ProfileLens temporarily bundles a modified ROCprofiler-SDK tool library because
the required region-relative counter-collection change has not yet been
released upstream. The upstream pull request is open:
<https://github.com/ROCm/rocm-systems/pull/7299>.

ProfileLens replays kernels under `rocprofv3` to collect PMC counters. It scopes
that replay to the `torch.profiler` window with `--selected-regions` and
`roctxProfilerResume/Pause`, then selects dispatches with
`--kernel-iteration-range`. Stock ROCprofiler-SDK 1.1.0 advances that iteration
counter from process start, independently of the selected region. The local
patch starts and stops the counter-collection context with the selected region.

## Binary provenance

| Field | Value |
| --- | --- |
| File | `librocprofiler-sdk-tool.so` |
| Upstream project | ROCprofiler-SDK in `ROCm/rocm-systems` |
| Upstream release | ROCm 7.2.4 / ROCprofiler-SDK 1.1.0 |
| Source | <https://github.com/ROCm/rocm-systems/tree/rocm-7.2.4/projects/rocprofiler-sdk> |
| Local source patch | `rocprofv3_rootcause.patch` |
| Upstream review | <https://github.com/ROCm/rocm-systems/pull/7299> (open as of 2026-07-23) |
| Format | ELF64, little-endian, Linux/GNU, x86-64 shared object |
| Size | 5,370,568 bytes |
| Binary SHA-256 | `7890c01108fc446bc68ef9b3eda39ac8af56afc0bd4d2f778358921dfdf2c251` |
| Patch SHA-256 | `781562e759361e91c7ac460bcfea5db01f28e419ff6f3d624bc572dcf34b1f74` |
| License | MIT; see `LICENSE.md` |

The binary is compatible only with the ROCm 7.x SDK ABI checked by
`profilelens/profile/rocprofv3.py`. It is not a general replacement for a
system ROCprofiler-SDK library.

## Rebuilding from corresponding source

Run these commands on a Linux x86-64 ROCm build host with CMake, Ninja, a C++
toolchain, and the ROCprofiler-SDK build prerequisites installed:

```bash
git clone --filter=blob:none --branch rocm-7.2.4 \
  https://github.com/ROCm/rocm-systems.git
cd rocm-systems
git apply /path/to/ProfileLens/rocprof-patch/7.x/rocprofv3_rootcause.patch

cmake -S projects/rocprofiler-sdk -B build/rocprofiler-sdk -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCPROFILER_BUILD_TESTS=OFF
cmake --build build/rocprofiler-sdk --target rocprofiler-sdk-tool

cp build/rocprofiler-sdk/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so.1.1.0 \
  /path/to/ProfileLens/rocprof-patch/7.x/librocprofiler-sdk-tool.so
strip --strip-unneeded \
  /path/to/ProfileLens/rocprof-patch/7.x/librocprofiler-sdk-tool.so
sha256sum /path/to/ProfileLens/rocprof-patch/7.x/librocprofiler-sdk-tool.so
```

Build paths can vary with the ROCm toolchain. A replacement binary must be
rebuilt from the exact 7.2.4 source and checked against the ABI/version guard;
record its compiler, CMake cache, build log, and new checksum in the release
records. The checksum above identifies the binary currently committed—it is
not an assertion that a differently provisioned build will be byte-identical.

## Removal plan

After the upstream change is merged and appears in the minimum supported ROCm
release, remove the binary overlay, patch, packaging inclusion, and associated
version guard. Until then, distributions must keep this binary, its
corresponding source reference, the local patch, and the upstream MIT license
together.
