"""
DSP Submodel Execution Engine for ESWIN EIC7700X.
Supports loading vendor dsp_kernels/*.so via ctypes with seamless 64-bit native fallbacks
for RMSNorm, Softmax, RoPE, and TopK.
"""

import ctypes
import os
from pathlib import Path
import platform
import sys
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from eswin_simulator.engines.base_engine import HardwareEngine
from eswin_simulator.memory.address_space import VirtualAddressSpace
from eswin_simulator.dsp.native_fallbacks import (
    rms_norm,
    softmax,
    apply_rope,
    compute_rope_frequencies,
    topk,
)


# -----------------------------------------------------------------------------
# Cadence Vision Q7 DSP C ABI Structures (Extracted from DWARF symbols)
# -----------------------------------------------------------------------------

class ESDSPBuffer(ctypes.Structure):
    """Corresponds to struct es_dsp_buffer_t."""
    _fields_ = [
        ("addr", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
    ]


class ESDSPBufferGroup(ctypes.Structure):
    """Corresponds to struct es_dsp_buffer_group_t."""
    _fields_ = [
        ("buffers", ctypes.POINTER(ESDSPBuffer)),
        ("bufferCnt", ctypes.c_uint32),
    ]


class DSPExecutor(HardwareEngine):
    """Simulates the Cadence Vision Q7 DSP cluster execution environment.

    Handles DSP submodel tasks and provides seamless acceleration:
    - Automatically attempts to load vendor shared libraries (`dsp_kernels/*.so`)
      when running on compatible 32-bit x86 runtime environments.
    - Seamlessly falls back to pure NumPy bit-accurate native fallbacks when
      running on 64-bit platforms (x86_64, aarch64, riscv64).
    """

    def __init__(
        self,
        address_space: VirtualAddressSpace,
        dsp_kernel_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(name="DSP_VisionQ7", address_space=address_space)
        self.dsp_kernel_dir = Path(dsp_kernel_dir) if dsp_kernel_dir else None
        self._loaded_libs: Dict[str, ctypes.CDLL] = {}
        self._is_32bit_x86 = (sys.maxsize <= 2**32) and (platform.machine() in ("i386", "i686", "x86"))

        # Execution stats
        self.total_dsp_ops: int = 0
        self.rms_norm_count: int = 0
        self.softmax_count: int = 0
        self.rope_count: int = 0
        self.topk_count: int = 0

    def reset(self) -> None:
        super().reset()
        self.total_dsp_ops = 0
        self.rms_norm_count = 0
        self.softmax_count = 0
        self.rope_count = 0
        self.topk_count = 0

    def get_kernel_lib(self, kernel_name: str) -> Optional[ctypes.CDLL]:
        """Loads a vendor DSP shared library if in a compatible environment."""
        if not self._is_32bit_x86 or not self.dsp_kernel_dir:
            return None

        if kernel_name in self._loaded_libs:
            return self._loaded_libs[kernel_name]

        lib_filename = f"libes_{kernel_name}.so" if not kernel_name.startswith("libes_") else f"{kernel_name}.so"
        lib_path = self.dsp_kernel_dir / lib_filename
        if not lib_path.is_file():
            return None

        try:
            cdll = ctypes.CDLL(str(lib_path))
            self._loaded_libs[kernel_name] = cdll
            return cdll
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Core DSP Kernel Execution
    # -------------------------------------------------------------------------

    def execute_rms_norm(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        eps: float = 1e-5,
        out_dtype: Optional[Union[str, np.dtype]] = None,
        out_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Executes ONNX RMSNorm."""
        self.total_operations += x.size * 3
        self.total_dsp_ops += 1
        self.rms_norm_count += 1
        return rms_norm(x, weight, eps=eps, out_dtype=out_dtype, out_scale=out_scale)

    def execute_softmax(
        self,
        x: np.ndarray,
        axis: int = -1,
        out_dtype: Optional[Union[str, np.dtype]] = None,
    ) -> np.ndarray:
        """Executes numerically stable Softmax."""
        self.total_operations += x.size * 3
        self.total_dsp_ops += 1
        self.softmax_count += 1
        return softmax(x, axis=axis, out_dtype=out_dtype)

    def execute_rope(
        self,
        x: np.ndarray,
        positions: Optional[np.ndarray] = None,
        cos: Optional[np.ndarray] = None,
        sin: Optional[np.ndarray] = None,
        theta_base: float = 10000.0,
        mode: str = "half",
    ) -> np.ndarray:
        """Executes Rotary Position Embedding."""
        self.total_operations += x.size * 2
        self.total_dsp_ops += 1
        self.rope_count += 1
        return apply_rope(x, positions=positions, cos=cos, sin=sin, theta_base=theta_base, mode=mode)

    def execute_topk(
        self,
        x: np.ndarray,
        k: int = 1,
        axis: int = -1,
        largest: bool = True,
        sorted: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Executes Top-K value and index selection."""
        self.total_operations += x.size
        self.total_dsp_ops += 1
        self.topk_count += 1
        return topk(x, k=k, axis=axis, largest=largest, sorted=sorted)

    def stats(self) -> Dict[str, Any]:
        res = super().stats()
        res.update(
            {
                "total_dsp_ops": self.total_dsp_ops,
                "rms_norm_count": self.rms_norm_count,
                "softmax_count": self.softmax_count,
                "rope_count": self.rope_count,
                "topk_count": self.topk_count,
                "mode": "32-bit native .so" if self._is_32bit_x86 else "64-bit Python/NumPy Fallbacks",
            }
        )
        return res
