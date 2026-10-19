"""NVDLA / EIC7700X SDP Lookup Table (LUT) Generator.

Generates the 688-byte `struct _lut_dev` hardware descriptor containing
piecewise linear/exponential interpolation tables for non-linear activations:
  - Sigmoid:  sigma(x) = 1 / (1 + exp(-x))
  - SiLU:     silu(x)  = x * sigma(x) (Swish)
  - Tanh:     tanh(x)  = (exp(2x) - 1) / (exp(2x) + 1)

Hardware Layout (DWARF-verified from EsSimulator):
  +0:   uint32_t precision          (0x0002 for FP16)
  +4:   uint32_t lut_cfg            (configuration flags)
  +8:   uint32_t lut_info           (routing and table select)
  +12:  float    le_start           (linear-exp table domain start, e.g. -8.0)
  +16:  float    le_end             (linear-exp table domain end, e.g. +8.0)
  +20:  float    lo_start           (linear-only table domain start, e.g. -8.0)
  +24:  float    lo_end             (linear-only table domain end, e.g. +8.0)
  +28:  uint32_t le_slope_scale     (fixed-point slope scale)
  +32:  uint32_t lo_slope_scale
  +36:  uint32_t le_slope_shift     (bit shift for slope interpolation)
  +40:  uint32_t lo_slope_shift
  +44:  int16_t  linear_exp_table[65]   (130 bytes: 65 piecewise sample points)
  +174: int16_t  linear_only_table[257] (514 bytes: 257 piecewise sample points)
  Total: 44 + 130 + 514 = 688 bytes.
"""

from enum import Enum, auto
import struct
from typing import Callable, Tuple, Union
import numpy as np


class ActivationType(Enum):
    SIGMOID = auto()
    SILU = auto()       # x * sigmoid(x)
    TANH = auto()
    EXP = auto()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -16.0, 16.0)))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


ACTIVATION_FUNCTIONS = {
    ActivationType.SIGMOID: _sigmoid,
    ActivationType.SILU: _silu,
    ActivationType.TANH: _tanh,
}


def generate_sdp_lut(
    act_type: ActivationType = ActivationType.SIGMOID,
    domain: Tuple[float, float] = (-8.0, 8.0)
) -> bytes:
    """Generates the 688-byte binary blob for an SDP piecewise LUT."""
    fn: Callable[[np.ndarray], np.ndarray] = ACTIVATION_FUNCTIONS.get(act_type, _sigmoid)
    x_min, x_max = domain

    # Header parameters
    precision = 2               # FP16 precision
    lut_cfg = 0x0031            # Standard linear interpolation mode
    lut_info = 0                # Route to SDP output
    le_start = float(x_min)
    le_end = float(x_max)
    lo_start = float(x_min)
    lo_end = float(x_max)

    # Fixed-point slope scaling factors (1.0 in FP16/fixed point)
    le_slope_scale = 0x3f80
    lo_slope_scale = 0x3f80
    le_slope_shift = 0
    lo_slope_shift = 0

    header = struct.pack(
        "<IIIffffIIII",
        precision,
        lut_cfg,
        lut_info,
        le_start,
        le_end,
        lo_start,
        lo_end,
        le_slope_scale,
        lo_slope_scale,
        le_slope_shift,
        lo_slope_shift
    )
    assert len(header) == 44, f"Header size mismatch: {len(header)} != 44"

    # 1. Sample 65 points for linear_exp_table across [x_min, x_max]
    le_x = np.linspace(x_min, x_max, 65, dtype=np.float32)
    le_y = fn(le_x).astype(np.float16)
    le_table_bytes = le_y.tobytes()
    assert len(le_table_bytes) == 130, f"LE table size mismatch: {len(le_table_bytes)} != 130"

    # 2. Sample 257 points for linear_only_table across [x_min, x_max]
    lo_x = np.linspace(x_min, x_max, 257, dtype=np.float32)
    lo_y = fn(lo_x).astype(np.float16)
    lo_table_bytes = lo_y.tobytes()
    assert len(lo_table_bytes) == 514, f"LO table size mismatch: {len(lo_table_bytes)} != 514"

    full_lut = header + le_table_bytes + lo_table_bytes
    assert len(full_lut) == 688, f"Full LUT size mismatch: {len(full_lut)} != 688"
    return full_lut
