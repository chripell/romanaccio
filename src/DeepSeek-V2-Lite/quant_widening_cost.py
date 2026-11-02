#!/usr/bin/env python3
"""Measures what a widened (merge-max) quant table costs in weight requantization error.

Sharing one loadable across experts requires one shared table, and a shared table must be at
least as wide as every expert it covers (`make_quant_table.py --merge-max`). Widening is not
free: `int8_step == threshold / 127`, so a threshold twice as large quantizes every weight of
that tensor with twice the step. This script quantifies that, per expert and per projection,
under symmetric absmax INT8 -- the exact scheme the vendor table encodes (FINDINGS §30).

It reports NMSE of quantize-then-dequantize against the fp32 weight, which is the part of the
error this script can compute exactly. It does NOT model EsAAC's per-channel weight packing or
the accumulation/requantization chain, so treat the numbers as the weight-side lower bound on
the accuracy cost, not as end-to-end error.

Usage:
    python quant_widening_cost.py DIR_WITH_ONNX --own-glob 'L*E*_table.json' \
        --global-table global_a.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

PROJECTIONS = ("W_gate", "W_up", "W_down")


def nmse(ref: np.ndarray, approx: np.ndarray) -> float:
    denom = float(np.sum(ref.astype(np.float64) ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum((ref.astype(np.float64) - approx.astype(np.float64)) ** 2) / denom)


def quant_dequant(w: np.ndarray, threshold: float) -> np.ndarray:
    step = threshold / 127.0
    codes = np.clip(np.round(w.astype(np.float64) / step), -127, 127)
    return (codes * step).astype(np.float32)


def weights_of(onnx_path: Path) -> dict:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path, help="directory of exported expert ONNX files")
    p.add_argument("--global-table", type=Path, required=True)
    p.add_argument("--pattern", default="L*E*.onnx")
    args = p.parse_args()

    glob_table = json.loads(args.global_table.read_text())["blob_info"]
    files = sorted(args.directory.glob(args.pattern))
    if not files:
        print(f"no {args.pattern} in {args.directory}")
        return 1

    print(f"{'expert':<8} " + " ".join(f"{n:>22}" for n in PROJECTIONS))
    print(f"{'':<8} " + " ".join(f"{'own NMSE / global NMSE':>22}" for _ in PROJECTIONS))
    totals = {n: [] for n in PROJECTIONS}
    for f in files:
        tag = f.name[: -len(".onnx")]
        own_path = f.with_name(f"{tag}_table.json")
        if not own_path.is_file():
            print(f"{tag:<8} skipped: no {own_path.name}")
            continue
        own = json.loads(own_path.read_text())["blob_info"]
        w = weights_of(f)
        cells = []
        for name in PROJECTIONS:
            if name not in w:
                cells.append(f"{'missing':>22}")
                continue
            arr = w[name].reshape(w[name].shape[0], -1)
            e_own = nmse(arr, quant_dequant(arr, own[name]["threshold"]))
            e_glob = nmse(arr, quant_dequant(arr, glob_table[name]["threshold"]))
            totals[name].append((e_own, e_glob))
            cells.append(f"{e_own:.3e} / {e_glob:.3e}")
        print(f"{tag:<8} " + " ".join(f"{c:>22}" for c in cells))

    print("\nper-projection summary (mean over experts, and worst case):")
    for name in PROJECTIONS:
        if not totals[name]:
            continue
        own = np.array([a for a, _ in totals[name]])
        glob = np.array([b for _, b in totals[name]])
        ratio = glob / np.where(own > 0, own, np.nan)
        print(f"  {name:<7} own {own.mean():.3e} (max {own.max():.3e})   "
              f"global {glob.mean():.3e} (max {glob.max():.3e})   "
              f"error x{np.nanmean(ratio):.2f} mean, x{np.nanmax(ratio):.2f} worst")
    print("\nNMSE scales with the square of the threshold widening, so a tensor whose threshold")
    print("was doubled by merge-max shows 4x the weight error. Anything above ~1e-2 NMSE on a")
    print("projection is worth an end-to-end accuracy check before trusting the shared table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
