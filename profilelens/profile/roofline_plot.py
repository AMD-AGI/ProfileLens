# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate a roofline plot for the top-K kernels."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from profilelens.dtypes import KernelProfile, RooflinePeaks


def render_roofline_png(
    peaks: RooflinePeaks,
    kernels: Iterable[KernelProfile],
    out_path: Path,
    title: str = "Kernels",
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
    mfma_tflops = peaks.peak_compute_tflops.get("mfma", 0.0)
    if hbm_tbps <= 0:
        return None

    supplied = list(kernels)
    visible = [(index, kernel) for index, kernel in enumerate(supplied, 1)
               if kernel.ai_hbm is not None and kernel.achieved_tflops is not None
               and math.isfinite(kernel.ai_hbm) and math.isfinite(kernel.achieved_tflops)
               and kernel.ai_hbm > 0 and kernel.achieved_tflops > 0]
    if not visible:
        return None
    ai_values = [kernel.ai_hbm for _index, kernel in visible]
    flops_values = [kernel.achieved_tflops for _index, kernel in visible]

    dtype_ceilings = [(dt, p.peak_compute_tflops.get("mfma", 0.0))
                     for dt, p in (peaks_by_dtype or {}).items()
                     if p.peak_compute_tflops.get("mfma", 0.0) > 0]
    if not dtype_ceilings and mfma_tflops > 0:
        dtype_ceilings = [(peaks.detected_dtype, mfma_tflops)]
    ridges = [value / hbm_tbps for _dtype, value in dtype_ceilings]
    ai_min = min(ai_values) * 0.5
    ai_max = max(max(ai_values) * 2, max(ridges, default=max(ai_values)) * 2)

    legend_rows = (len(supplied) + 1) // 2
    legend_height = 0.8 + legend_rows * 0.25
    figure_height = 6.5 + legend_height
    fig, ax = plt.subplots(figsize=(12.5, figure_height))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94,
                        bottom=(legend_height + 0.45) / figure_height)
    import numpy as np

    ai_grid = np.geomspace(ai_min, max(ridges, default=ai_max), 200)
    ax.plot(ai_grid, ai_grid * hbm_tbps, "-", color="#1f77b4", lw=2,
            label=f"HBM reference {hbm_tbps:.2f} TB/s")
    colors = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    ceil_color_by_dtype = {}
    for j, (dt, value) in enumerate(sorted(dtype_ceilings, key=lambda x: -x[1])):
        color = colors[j % len(colors)]
        ceil_color_by_dtype[dt] = color
        dtype_ridge = value / hbm_tbps
        ai_compute = np.geomspace(max(dtype_ridge, ai_min), ai_max, 200)
        ax.plot(ai_compute, np.full_like(ai_compute, value), "-", color=color, lw=2,
                label=f"{dt} reference {value:.0f} TFLOPS")
        ax.axvline(dtype_ridge, color=color, ls=":", alpha=0.25)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte HBM)")
    ax.set_ylabel("Achieved Throughput (TFLOPS)")
    ax.set_title(title)
    top_ceiling = max([max(flops_values), max(ai_values) * hbm_tbps] + [v for _, v in dtype_ceilings])
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(min(flops_values) * 0.5, top_ceiling * 1.5)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ceiling_legend = ax.legend(loc="upper left", fontsize=9)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    reserved = ceiling_legend.get_window_extent(renderer).expanded(1.03, 1.08)
    points = ax.transData.transform(list(zip(ai_values, flops_values)))
    placed = []
    scale = fig.dpi / 100.0
    for (number, kernel), point in zip(visible, points):
        candidates = []
        for radius in (26, 40, 56, 74, 96):
            for angle in (-45, 45, -135, 135, -90, 90, 0, 180):
                radians = np.deg2rad(angle)
                candidate = point + radius * scale * np.array([np.cos(radians), np.sin(radians)])
                margin = 16 * scale
                if not (bounds.x0 + margin < candidate[0] < bounds.x1 - margin
                        and bounds.y0 + margin < candidate[1] < bounds.y1 - margin):
                    continue
                if reserved.contains(*candidate):
                    continue
                overlaps = sum(np.linalg.norm(candidate - other) < 32 * scale for other in placed)
                covers_point = sum(np.linalg.norm(candidate - other) < 18 * scale for other in points)
                data_x = ax.transData.inverted().transform(candidate)[0]
                reference_values = [data_x * hbm_tbps] + [value for _dt, value in dtype_ceilings]
                covers_line = sum(abs(ax.transData.transform((data_x, value))[1] - candidate[1])
                                  < 10 * scale for value in reference_values)
                candidates.append((overlaps + covers_point + covers_line, radius, candidate))
        label_point = (min(candidates, key=lambda item: (item[0], item[1]))[2]
                       if candidates else point + np.array([20, -20]) * scale)
        placed.append(label_point)
        offset = (label_point - point) * 72.0 / fig.dpi
        ax.scatter(kernel.ai_hbm, kernel.achieved_tflops, s=22, color="black", zorder=4)
        ax.annotate(str(number), (kernel.ai_hbm, kernel.achieved_tflops),
                    xytext=offset, textcoords="offset points", ha="center", va="center",
                    fontsize=9, fontweight="bold", zorder=5,
                    bbox={"boxstyle": "circle,pad=0.28", "fc": "white", "ec": "#333333"},
                    arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8,
                                "shrinkA": 7, "shrinkB": 3})

    legend_top = (legend_height - 0.12) / figure_height
    fig.text(0.08, legend_top, "Kernel key", fontsize=10, fontweight="bold")
    visible_indices = {number for number, _kernel in visible}
    for index, kernel in enumerate(supplied, 1):
        column, row = divmod(index - 1, legend_rows)
        name = kernel.kernel_name
        if len(name) > 65:
            name = name[:43] + "…" + name[-21:]
        suffix = ""
        if index not in visible_indices:
            suffix = " · 0 compute" if kernel.mfma_flops == 0 and kernel.achieved_tflops == 0 else " · unavailable"
        fig.text(0.08 + column * 0.465, legend_top - (0.3 + row * 0.25) / figure_height,
                 f"{index:>2}. {name}{suffix}", fontsize=8, fontfamily="monospace", color="#222222")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    return out_path
