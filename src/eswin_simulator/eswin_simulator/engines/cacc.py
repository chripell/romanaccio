"""
Convolution Accumulator (CACC) Engine simulation for ESWIN EIC7700X.
Performs 32-bit partial sum accumulation, precision conversion, scaling,
and hardware saturating clipping for FP16 and INT8 execution pipelines.
"""

from typing import Dict, Any, Optional, Union
import numpy as np

from eswin_simulator.engines.base_engine import HardwareEngine
from eswin_simulator.memory.address_space import VirtualAddressSpace


class CACCEngine(HardwareEngine):
    """Simulates the NPU Convolution Accumulator (CACC) hardware unit.

    In the NVDLA/ESWIN NPU architecture, CACC resides downstream of CMAC:
    - In FP16 mode: accumulates 32-bit floating point products and rounds
      to IEEE 754 half-precision float16.
    - In INT8 mode: accumulates 32-bit signed integer products (int32_t),
      adds optional channel/tensor bias, applies quantization scaling factors,
      and strictly clips/saturates into signed 8-bit integers [-128, 127].
    """

    def __init__(self, address_space: VirtualAddressSpace):
        super().__init__(name="CACC", address_space=address_space)
        self.total_accumulations: int = 0
        self.total_saturations: int = 0

    def reset(self) -> None:
        super().reset()
        self.total_accumulations = 0
        self.total_saturations = 0

    def accumulate(
        self,
        partial_sum: np.ndarray,
        buffer: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Accumulates partial sums across multi-pass channels or spatial strips.

        Args:
            partial_sum: Incoming partial product sum array.
            buffer: Optional existing accumulator buffer. If None, initializes
                    a new 32-bit accumulator buffer matching partial_sum.

        Returns:
            Updated 32-bit accumulator array.
        """
        self.total_operations += partial_sum.size
        self.total_accumulations += 1

        if buffer is None:
            if np.issubdtype(partial_sum.dtype, np.floating):
                return partial_sum.astype(np.float32, copy=True)
            else:
                return partial_sum.astype(np.int32, copy=True)

        buffer += partial_sum
        return buffer

    def finalize_fp16(
        self,
        accumulator: np.ndarray,
        bias: Optional[np.ndarray] = None,
        to_fp16: bool = True,
    ) -> np.ndarray:
        """Finalizes an FP16 pipeline convolution accumulation.

        Args:
            accumulator: 32-bit floating-point accumulator buffer.
            bias: Optional bias tensor to add prior to FP16 conversion.
            to_fp16: If True, casts to np.float16; otherwise keeps np.float32.

        Returns:
            Finalized output tensor.
        """
        out = accumulator.astype(np.float32, copy=False)
        if bias is not None:
            # Reshape bias for broadcasting across spatial dimensions if needed
            bias_arr = np.asarray(bias, dtype=np.float32)
            if out.ndim == 4 and bias_arr.ndim == 1:
                bias_arr = bias_arr.reshape(1, -1, 1, 1)
            elif out.ndim == 2 and bias_arr.ndim == 1:
                bias_arr = bias_arr.reshape(1, -1)
            out = out + bias_arr

        if to_fp16:
            return out.astype(np.float16)
        return out

    def finalize_int8(
        self,
        accumulator: np.ndarray,
        scale: Union[float, np.ndarray] = 1.0,
        bias: Optional[np.ndarray] = None,
        clip_min: int = -128,
        clip_max: int = 127,
    ) -> np.ndarray:
        """Finalizes an INT8 pipeline convolution accumulation with saturating clipping.

        Mathematical operation:
            y = clip(round((acc + bias) * scale), clip_min, clip_max)

        Args:
            accumulator: 32-bit signed integer accumulator array.
            scale: Quantization multiplier/scale (scalar or per-channel array).
            bias: Optional 32-bit signed integer bias array.
            clip_min: Minimum saturating value (default -128 for signed INT8).
            clip_max: Maximum saturating value (default 127 for signed INT8).

        Returns:
            8-bit signed integer (int8) output tensor.
        """
        acc = accumulator.astype(np.float32, copy=False)
        if bias is not None:
            bias_arr = np.asarray(bias, dtype=np.float32)
            if acc.ndim == 4 and bias_arr.ndim == 1:
                bias_arr = bias_arr.reshape(1, -1, 1, 1)
            elif acc.ndim == 2 and bias_arr.ndim == 1:
                bias_arr = bias_arr.reshape(1, -1)
            acc = acc + bias_arr

        scale_arr = np.asarray(scale, dtype=np.float32)
        if acc.ndim == 4 and scale_arr.ndim == 1:
            scale_arr = scale_arr.reshape(1, -1, 1, 1)
        elif acc.ndim == 2 and scale_arr.ndim == 1:
            scale_arr = scale_arr.reshape(1, -1)

        scaled = np.round(acc * scale_arr)

        # Count saturated elements
        saturated_mask = (scaled < clip_min) | (scaled > clip_max)
        self.total_saturations += int(np.sum(saturated_mask))

        clipped = np.clip(scaled, clip_min, clip_max)
        return clipped.astype(np.int8)

    def stats(self) -> Dict[str, Any]:
        res = super().stats()
        res.update(
            {
                "total_accumulations": self.total_accumulations,
                "total_saturations": self.total_saturations,
            }
        )
        return res
