# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Roofline model: parse empirical peaks and compute ceilings."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from profilelens.dtypes import RooflinePeaks


@dataclass
class RooflineCeilings:
    """Parsed roofline benchmark data."""

    hbm_bw: float = 0.0
    l2_bw: float = 0.0
    l1_bw: float = 0.0
    lds_bw: float = 0.0
    valu_peak: float = 0.0
    mfma_peak: float = 0.0


def parse_roofline_csv(csv_path: str | Path, dtype: str = "BF16") -> RooflineCeilings:
    """Parse roofline.csv from rocprof-compute benchmark."""
    p = Path(csv_path)
    if not p.exists():
        return RooflineCeilings()

    headers: list[str] = []
    values: list[str] = []
    with open(p, newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                headers = [h.strip() for h in row[1:]]
            elif i == 1:
                values = [v.strip() for v in row[1:]]
                break

    data: dict[str, float] = {}
    for h, v in zip(headers, values):
        try:
            data[h] = float(v)
        except (ValueError, TypeError):
            pass

    dt = dtype.upper()

    mfma_key_map = {
        "BF16": "MFMABF16Flops", "FP16": "MFMAF16Flops",
        "FP32": "MFMAF32Flops", "FP64": "MFMAF64Flops",
        "FP8": "MFMAF8Flops", "INT8": "MFMAI8Ops",
    }
    valu_key_map = {
        "BF16": "BF16Flops", "FP16": "FP16Flops",
        "FP32": "FP32Flops", "FP64": "FP64Flops",
        "FP8": "FP8Flops", "INT8": "I8Ops",
    }
    ops_suffix = "Ops" if dt.startswith("I") else "Flops"
    mfma_key = mfma_key_map.get(dt, f"MFMA{dt}{ops_suffix}")
    valu_key = valu_key_map.get(dt, f"{dt}{ops_suffix}")

    return RooflineCeilings(
        hbm_bw=data.get("HBMBw", 0.0),
        l2_bw=data.get("L2Bw", 0.0),
        l1_bw=data.get("L1Bw", 0.0),
        lds_bw=data.get("LDSBw", 0.0),
        valu_peak=data.get(valu_key, 0.0),
        mfma_peak=data.get(mfma_key, 0.0),
    )


def _to_tbps(value: float) -> float:
    """Convert raw bandwidth reading to TB/s."""
    if value <= 0:
        return 0.0
    if value > 1e9:
        return value / 1e12
    if value > 1e3:
        return value / 1e3
    return value


def _to_tflops(value: float) -> float:
    """Convert raw compute reading to TFLOPS."""
    if value <= 0:
        return 0.0
    if value > 1e9:
        return value / 1e12
    if value > 1e3:
        return value / 1e3
    return value


def build_roofline_peaks(
    ceilings: RooflineCeilings,
    gpu_model: str,
    dtype: str,
) -> RooflinePeaks:
    """Convert parsed ceilings into a RooflinePeaks dataclass."""
    peak_bw = {
        "hbm": _to_tbps(ceilings.hbm_bw),
        "l2": _to_tbps(ceilings.l2_bw),
        "l1": _to_tbps(ceilings.l1_bw),
        "lds": _to_tbps(ceilings.lds_bw),
    }
    peak_compute = {
        "valu": _to_tflops(ceilings.valu_peak),
        "mfma": _to_tflops(ceilings.mfma_peak),
    }
    mfma_tflops = peak_compute["mfma"] or peak_compute["valu"]
    hbm_tbps = peak_bw["hbm"]
    ridge = mfma_tflops * 1e12 / (hbm_tbps * 1e12) if hbm_tbps > 0 else 0.0

    return RooflinePeaks(
        gpu_model=gpu_model,
        detected_dtype=dtype,
        peak_bw_tbps=peak_bw,
        peak_compute_tflops=peak_compute,
        ridge_point=round(ridge, 2),
    )


def build_roofline_peaks_multi(
    csv_path: str | Path,
    gpu_model: str,
    dtypes: "list[str] | tuple[str, ...] | set[str]",
) -> dict[str, RooflinePeaks]:
    """Build a per-dtype roofline so each kernel sits on its own MFMA ceiling."""
    from profilelens.profile.dtype_detect import counter_dtype

    out: dict[str, RooflinePeaks] = {}
    for dt in dtypes:
        roof_dt = counter_dtype(dt)  # MXFP4 -> FP8, INT4 -> INT8, ...
        ceilings = parse_roofline_csv(csv_path, roof_dt)
        out[dt] = build_roofline_peaks(ceilings, gpu_model, dt)
    return out
