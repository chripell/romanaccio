"""
Native Python/NumPy bit-accurate implementations of ESWIN Cadence Vision Q7 DSP kernels.
"""

from eswin_simulator.dsp.native_fallbacks.rms_norm import rms_norm
from eswin_simulator.dsp.native_fallbacks.softmax import softmax
from eswin_simulator.dsp.native_fallbacks.rope import (
    apply_rope,
    compute_rope_frequencies,
    rotate_half,
    rotate_interleaved,
)
from eswin_simulator.dsp.native_fallbacks.topk import topk

__all__ = [
    "rms_norm",
    "softmax",
    "apply_rope",
    "compute_rope_frequencies",
    "rotate_half",
    "rotate_interleaved",
    "topk",
]
