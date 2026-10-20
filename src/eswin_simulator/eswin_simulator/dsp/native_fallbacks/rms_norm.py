"""
Native Python/NumPy bit-accurate implementation of ONNX RMSNorm.
Matches Cadence Vision Q7 DSP kernel behavior across all channel widths.
"""

from typing import Optional, Union
import numpy as np


def rms_norm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5,
    out_dtype: Optional[Union[str, np.dtype]] = None,
    out_scale: Optional[float] = None,
) -> np.ndarray:
    """Computes Root Mean Square Layer Normalization (RMSNorm).

    Mathematical formulation:
        variance = mean(x^2, axis=-1, keepdims=True)
        rms = sqrt(variance + eps)
        normed = (x / rms) * weight

    If out_scale is provided (INT8 quantized mode):
        out = clip(round(normed / out_scale), -128, 127).astype(np.int8)

    Args:
        x: Input activations of arbitrary shape (..., C).
        weight: Normalization scale vector (C,).
        eps: Epsilon for numerical stability (typically 1e-5 or 1e-6).
        out_dtype: Target data type for output ('float16', 'float32', 'int8').
                   If None, preserves input dtype.
        out_scale: Optional quantization scale factor for INT8 outputs.

    Returns:
        Normalized tensor matching x's shape with requested out_dtype.
    """
    x_f32 = x.astype(np.float32, copy=False)
    w_f32 = weight.astype(np.float32, copy=False)

    channel_dim = x_f32.shape[-1]
    if w_f32.shape[-1] != channel_dim:
        raise ValueError(f"Weight dimension ({w_f32.shape[-1]}) does not match input channels ({channel_dim})")

    # Compute variance across hidden channels
    variance = np.mean(np.square(x_f32), axis=-1, keepdims=True)
    rms = np.sqrt(variance + eps)

    normed = (x_f32 / rms) * w_f32

    # INT8 Quantization mode
    if out_scale is not None or (out_dtype is not None and np.dtype(out_dtype) == np.int8):
        scale = out_scale if out_scale is not None else 1.0
        quantized = np.round(normed / scale)
        return np.clip(quantized, -128, 127).astype(np.int8)

    target_dtype = np.dtype(out_dtype) if out_dtype is not None else x.dtype
    return normed.astype(target_dtype)
