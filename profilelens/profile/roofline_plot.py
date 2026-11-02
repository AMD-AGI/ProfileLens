# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate a roofline plot for the top-K kernels."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from profilelens.dtypes import KernelProfile, RooflinePeaks


def render_roofline_png(
    peaks: RooflinePeaks,
    kernels: Iterable[KernelProfile],
    out_path: Path,
    title: str = "Empirical Roofline (Top kernels)",
    peaks_by_dtype: "dict[str, RooflinePeaks] | None" = None,
) -> Path | None:
    """Write a roofline plot to ``out_path`` and return its path."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    hbm_tbps = peaks.peak_bw_tbps.get("hbm", 0.0)
    mfma_tflops = peaks.peak_compute_tflops.get("mfma", 0.0) or peaks.peak_compute_tflops.get("valu", 0.0)
    if hbm_tbps <= 0 or mfma_tflops <= 0:
        return None

    hbm_bps = hbm_tbps * 1e12
    mfma_fps = mfma_tflops * 1e12
    ridge = mfma_fps / hbm_bps  # FLOP/byte where the two ceilings meet

    kernels = list(kernels)
    if not kernels:
        return None

    ai_values = [max(k.ai_hbm, 1e-3) for k in kernels]
    flops_values = [max(k.achieved_tflops, 1e-6) for k in kernels]

    ai_min = min(min(ai_values) * 0.5, 0.05)
    ai_max = max(max(ai_values) * 2.0, ridge * 4.0)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Draw memory-bound diagonal (TFLOPS = AI * HBM_bw)
    import numpy as np

    ai_grid = np.geomspace(ai_min, ridge, 200)
    bw_line = ai_grid * hbm_bps / 1e12  # in TFLOPS
    ax.plot(ai_grid, bw_line, "-", color="#1f77b4", lw=2,
            label=f"HBM bw {hbm_tbps:.2f} TB/s")

    # Per-dtype MFMA ceilings when multiple compute dtypes were observed
    dtype_ceilings: list[tuple[str, float]] = []
    if peaks_by_dtype and len(peaks_by_dtype) > 1:
        seen_vals: set[int] = set()
        for dt, p in peaks_by_dtype.items():
            v = p.peak_compute_tflops.get("mfma", 0.0)
            key = int(v)
            if v > 0 and key not in seen_vals:
                seen_vals.add(key)
                dtype_ceilings.append((dt, v))
    if not dtype_ceilings:
        dtype_ceilings = [("MFMA", mfma_tflops)]

    ai_compute = np.geomspace(ridge, ai_max, 200)
    _ceil_colors = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    ceil_color_by_dtype: dict[str, str] = {}
    for j, (dt, v) in enumerate(sorted(dtype_ceilings, key=lambda x: -x[1])):
        col = _ceil_colors[j % len(_ceil_colors)]
        ceil_color_by_dtype[dt] = col
        ax.plot(ai_compute, np.full_like(ai_compute, v), "-",
                color=col, lw=2, label=f"{dt} peak {v:.0f} TFLOPS")

    ax.axvline(ridge, color="grey", ls=":", alpha=0.5,
               label=f"Ridge {ridge:.0f} FLOP/byte")

    for i, k in enumerate(kernels):
        ai = max(k.ai_hbm, 1e-3)
        tflops = max(k.achieved_tflops, 1e-6)
        n = i + 1
        ax.scatter(ai, tflops, marker="o", s=300, facecolor="white",
                   edgecolor="black", linewidth=1.4, zorder=4)
        ax.annotate(str(n), (ai, tflops), ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black", zorder=5)
        lc = ceil_color_by_dtype.get(k.detected_dtype, "black")
        label = k.kernel_name
        if len(label) > 32:
            label = label[:32] + "…"
        frac = ((np.log10(ai) - np.log10(ai_min))
                / (np.log10(ai_max) - np.log10(ai_min)))
        xytext, ha = ((-11, 7), "right") if frac > 0.6 else ((11, 7), "left")
        ax.annotate(
            f"{n}. {label} [{k.detected_dtype}]",
            (ai, tflops),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            fontsize=8,
            color=lc,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte HBM)")
    ax.set_ylabel("Achieved Throughput (TFLOPS)")
    ax.set_title(title)
    top_ceiling = max([mfma_tflops] + [v for _, v in dtype_ceilings])
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(max(min(flops_values) * 0.5, 0.001), top_ceiling * 1.5)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    return out_path
