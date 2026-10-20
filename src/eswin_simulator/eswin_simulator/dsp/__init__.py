"""
ESWIN EIC7700X Cadence Vision Q7 DSP Submodel Execution Engine.
Provides C ABI bindings to vendor shared libraries and bit-accurate 64-bit native fallbacks.
"""

from eswin_simulator.dsp.dsp_executor import (
    DSPExecutor,
    ESDSPBuffer,
    ESDSPBufferGroup,
)
from eswin_simulator.dsp.native_fallbacks import (
    rms_norm,
    softmax,
    apply_rope,
    compute_rope_frequencies,
    topk,
)

__all__ = [
    "DSPExecutor",
    "ESDSPBuffer",
    "ESDSPBufferGroup",
    "rms_norm",
    "softmax",
    "apply_rope",
    "compute_rope_frequencies",
    "topk",
]
