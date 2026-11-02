# Third-Party Notices

This file records third-party material that ProfileLens bundles or downloads
at runtime. Python packages installed separately are declared in
`pyproject.toml` and retain their own licenses and notices.

## ROCprofiler-SDK patched tool library

ProfileLens includes a modified binary and its corresponding source patch:

- Project: ROCprofiler-SDK, part of ROCm/rocm-systems
- Upstream release: ROCm 7.2.4
- Source: <https://github.com/ROCm/rocm-systems/tree/rocm-7.2.4/projects/rocprofiler-sdk>
- Local files: `rocprof-patch/7.x/librocprofiler-sdk-tool.so` and
  `rocprof-patch/7.x/rocprofv3_rootcause.patch`
- Modification status: the local binary includes the region-relative counter
  collection patch described in `rocprof-patch/7.x/README.md`
- License: MIT

The upstream MIT license text is reproduced in
`rocprof-patch/7.x/LICENSE.md`.

## rocprofiler-compute roofline benchmark

ProfileLens downloads a pinned subset of the rocprofiler-compute source on
first use; that source is not committed in this repository.

- Project: rocprofiler-compute, part of ROCm/rocm-systems
- Source: <https://github.com/ROCm/rocm-systems/tree/c0a79552b50a959172bea2ed224ff9366f47681b/projects/rocprofiler-compute>
- Pinned commit: `c0a79552b50a959172bea2ed224ff9366f47681b`
- Downloaded paths: `projects/rocprofiler-compute/src/roofline` and
  `projects/rocprofiler-compute/src/utils`
- License: MIT

The downloaded project contains its own `LICENSE.md`. ProfileLens preserves
the upstream checkout and license in the per-user cache.
