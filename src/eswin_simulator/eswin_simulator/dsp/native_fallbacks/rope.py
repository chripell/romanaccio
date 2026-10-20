"""
Native Python/NumPy bit-accurate implementation of Rotary Position Embedding (RoPE).
Supports Hugging Face / SmolLM / Llama style and interleaved pair formats.
"""

from typing import Optional, Tuple, Union
import numpy as np


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotates half the hidden dimensions of the input tensor (Hugging Face style)."""
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    return np.concatenate((-x2, x1), axis=-1)


def rotate_interleaved(x: np.ndarray) -> np.ndarray:
    """Rotates adjacent dimension pairs [-x1, x0, -x3, x2, ...] (DSP hardware style)."""
    x_rot = np.empty_like(x)
    x_rot[..., 0::2] = -x[..., 1::2]
    x_rot[..., 1::2] = x[..., 0::2]
    return x_rot


def compute_rope_frequencies(
    head_dim: int,
    max_position: int,
    theta_base: float = 10000.0,
    interleaved: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precomputes cosine and sine frequency tables for RoPE.

    Args:
        head_dim: Dimensionality of each attention head (must be even).
        max_position: Maximum sequence length.
        theta_base: Base frequency for rotary embeddings (default 10000.0).
        interleaved: True for adjacent pairs, False for half-split concatenation.

    Returns:
        (cos, sin) tensors with shape (max_position, head_dim).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    inv_freq = 1.0 / (theta_base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(max_position, dtype=np.float32)
    angles = np.outer(positions, inv_freq)  # (max_position, head_dim // 2)

    if interleaved:
        # Repeat adjacent elements: (max_position, head_dim)
        emb = np.repeat(angles, 2, axis=-1)
    else:
        # Hugging Face style: concatenate halves
        emb = np.concatenate((angles, angles), axis=-1)

    return np.cos(emb), np.sin(emb)


def apply_rope(
    x: np.ndarray,
    positions: Optional[np.ndarray] = None,
    cos: Optional[np.ndarray] = None,
    sin: Optional[np.ndarray] = None,
    theta_base: float = 10000.0,
    mode: str = "half",
) -> np.ndarray:
    """Applies Rotary Position Embedding to a Query or Key tensor.

    Args:
        x: Input activations with shape (..., SeqLen, HeadDim).
        positions: Optional 1D or 2D integer sequence positions.
        cos: Precomputed cosine frequency tensor.
        sin: Precomputed sine frequency tensor.
        theta_base: Base frequency (default 10000.0).
        mode: "half" (Hugging Face / Llama style) or "interleaved" (DSP pair).

    Returns:
        Rotated tensor matching x's shape and dtype.
    """
    orig_dtype = x.dtype
    x_f32 = x.astype(np.float32, copy=False)
    seq_len = x.shape[-2]
    head_dim = x.shape[-1]

    if cos is None or sin is None:
        if positions is None:
            positions = np.arange(seq_len, dtype=np.int32)
        max_pos = int(np.max(positions)) + 1
        full_cos, full_sin = compute_rope_frequencies(
            head_dim=head_dim,
            max_position=max_pos,
            theta_base=theta_base,
            interleaved=(mode == "interleaved"),
        )
        cos = full_cos[positions]
        sin = full_sin[positions]

    # Ensure broadcastable shape with x
    while cos.ndim < x_f32.ndim:
        cos = np.expand_dims(cos, axis=0)
        sin = np.expand_dims(sin, axis=0)

    if mode == "interleaved":
        x_rot = rotate_interleaved(x_f32)
    else:
        x_rot = rotate_half(x_f32)

    out = (x_f32 * cos) + (x_rot * sin)
    return out.astype(orig_dtype)
