"""
Single Data Point (SDP) Engine simulation for ESWIN EIC7700X.
Executes pointwise ALU operations (bias add, batchnorm, eltwise add/sub/mul),
fixed-function non-linearities (ReLU, LeakyReLU, PReLU, Clip),
piecewise-linear 256/512-entry LUT activations (SiLU/Swish, Sigmoid, Tanh, GELU),
and precision conversions (FP16 <-> INT8).
"""

from dataclasses import dataclass
import struct
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from eswin_simulator.engines.base_engine import HardwareEngine
from eswin_simulator.memory.address_space import VirtualAddressSpace


@dataclass
class SDPLookupTable:
    """Represents an SDP 688-byte piecewise linear lookup table (struct _lut_dev).

    Attributes:
        precision: Data type (2 = FP16, 4 = INT8).
        lut_cfg: Hardware configuration flags (e.g. 0x0031).
        lut_info: Routing and table selection metadata.
        le_start: Linear-exponential domain minimum.
        le_end: Linear-exponential domain maximum.
        lo_start: Linear-only domain minimum (default -8.0 for SiLU/Sigmoid).
        lo_end: Linear-only domain maximum (default +8.0 for SiLU/Sigmoid).
        le_slope_scale: Fixed-point slope multiplier for LE interpolation.
        lo_slope_scale: Fixed-point slope multiplier for LO interpolation.
        le_slope_shift: Bit shift for LE slope interpolation.
        lo_slope_shift: Bit shift for LO slope interpolation.
        linear_exp_table: 65 piecewise FP16 sample points (130 bytes).
        linear_only_table: 257 piecewise FP16 sample points (514 bytes).
    """

    precision: int
    lut_cfg: int
    lut_info: int
    le_start: float
    le_end: float
    lo_start: float
    lo_end: float
    le_slope_scale: int
    lo_slope_scale: int
    le_slope_shift: int
    lo_slope_shift: int
    linear_exp_table: np.ndarray  # 65 elements (FP16)
    linear_only_table: np.ndarray  # 257 elements (FP16)

    @classmethod
    def from_bytes(cls, raw_bytes: Union[bytes, bytearray, memoryview]) -> "SDPLookupTable":
        """Parses a 688-byte binary blob into an SDPLookupTable descriptor."""
        if len(raw_bytes) < 688:
            raise ValueError(f"SDP LUT blob size ({len(raw_bytes)}B) is less than required 688 bytes.")

        header = struct.unpack("<IIIffffIIII", raw_bytes[:44])
        le_table = np.frombuffer(raw_bytes[44 : 44 + 130], dtype=np.float16).copy()
        lo_table = np.frombuffer(raw_bytes[174 : 174 + 514], dtype=np.float16).copy()

        le_start = header[3]
        le_end = header[4]
        lo_start = header[5]
        lo_end = header[6]

        # In NVDLA LUTs, symmetric domains [-R, +R] often specify only the negative start -R
        # with end stored as 0.0 or <= start. If end <= start or (lo_end == 0.0 and lo_start < 0.0),
        # the true domain end is abs(start).
        if lo_end <= lo_start or (lo_end == 0.0 and lo_start < 0.0):
            lo_end = abs(lo_start)
        if le_end <= le_start or (le_end == 0.0 and le_start < 0.0):
            le_end = abs(le_start)

        return cls(
            precision=header[0],
            lut_cfg=header[1],
            lut_info=header[2],
            le_start=le_start,
            le_end=le_end,
            lo_start=lo_start,
            lo_end=lo_end,
            le_slope_scale=header[7],
            lo_slope_scale=header[8],
            le_slope_shift=header[9],
            lo_slope_shift=header[10],
            linear_exp_table=le_table,
            linear_only_table=lo_table,
        )

    def to_bytes(self) -> bytes:
        """Serializes the SDPLookupTable into a 688-byte binary blob."""
        header = struct.pack(
            "<IIIffffIIII",
            self.precision,
            self.lut_cfg,
            self.lut_info,
            self.le_start,
            self.le_end,
            self.lo_start,
            self.lo_end,
            self.le_slope_scale,
            self.lo_slope_scale,
            self.le_slope_shift,
            self.lo_slope_shift,
        )
        le_bytes = self.linear_exp_table.astype(np.float16).tobytes()
        lo_bytes = self.linear_only_table.astype(np.float16).tobytes()
        assert len(header) == 44
        assert len(le_bytes) == 130
        assert len(lo_bytes) == 514
        return header + le_bytes + lo_bytes

    @classmethod
    def create_silu_lut(cls) -> "SDPLookupTable":
        """Synthesizes a standard SiLU (x / (1 + exp(-x))) 688-byte LUT."""
        x_le = np.linspace(-1.0, 1.0, 65, dtype=np.float32)
        x_lo = np.linspace(-8.0, 8.0, 257, dtype=np.float32)

        def silu_fn(v):
            return v / (1.0 + np.exp(-v))

        le_samples = silu_fn(x_le).astype(np.float16)
        lo_samples = silu_fn(x_lo).astype(np.float16)

        return cls(
            precision=2,  # FP16
            lut_cfg=0x0031,
            lut_info=0x00FCFB00,
            le_start=-1.0,
            le_end=1.0,
            lo_start=-8.0,
            lo_end=8.0,
            le_slope_scale=0,
            lo_slope_scale=0,
            le_slope_shift=0,
            lo_slope_shift=0,
            linear_exp_table=le_samples,
            linear_only_table=lo_samples,
        )

    @classmethod
    def create_sigmoid_lut(cls) -> "SDPLookupTable":
        """Synthesizes a standard Sigmoid (1 / (1 + exp(-x))) 688-byte LUT."""
        x_le = np.linspace(-1.0, 1.0, 65, dtype=np.float32)
        x_lo = np.linspace(-8.0, 8.0, 257, dtype=np.float32)

        def sigmoid_fn(v):
            return 1.0 / (1.0 + np.exp(-v))

        le_samples = sigmoid_fn(x_le).astype(np.float16)
        lo_samples = sigmoid_fn(x_lo).astype(np.float16)

        return cls(
            precision=2,
            lut_cfg=0x0031,
            lut_info=0x00FCFB00,
            le_start=-1.0,
            le_end=1.0,
            lo_start=-8.0,
            lo_end=8.0,
            le_slope_scale=0,
            lo_slope_scale=0,
            le_slope_shift=0,
            lo_slope_shift=0,
            linear_exp_table=le_samples,
            linear_only_table=lo_samples,
        )


class SDPEngine(HardwareEngine):
    """Simulates the NPU Single Data Point (SDP) hardware execution engine.

    The SDP engine executes post-convolution operations or standalone offline activations:
    - Bias addition & batch normalization scaling
    - Fixed-function non-linear activations (ReLU, LeakyReLU, PReLU, Clip)
    - Non-linear activations via 256/512-entry piecewise linear LUTs (SiLU, Sigmoid, Tanh, GELU)
    - Pointwise ALU elementwise operations (Add, Sub, Mul, Min, Max)
    - Precision conversions (FP16 <-> INT8 with scaling and quantization zero-points)
    """

    def __init__(self, address_space: VirtualAddressSpace):
        super().__init__(name="SDP", address_space=address_space)
        self.total_activations: int = 0
        self.total_eltwise: int = 0
        self.total_conversions: int = 0

    def reset(self) -> None:
        super().reset()
        self.total_activations = 0
        self.total_eltwise = 0
        self.total_conversions = 0

    # -------------------------------------------------------------------------
    # 1. Bias Addition & Normalization
    # -------------------------------------------------------------------------

    def bias_add(self, x: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Adds channel or elementwise bias to tensor x with broadcasting."""
        self.total_operations += x.size
        b = np.asarray(bias)
        if x.ndim == 4 and b.ndim == 1:
            b = b.reshape(1, -1, 1, 1)
        elif x.ndim == 3 and b.ndim == 1:
            b = b.reshape(1, 1, -1)
        elif x.ndim == 2 and b.ndim == 1:
            b = b.reshape(1, -1)
        return x + b

    def batch_norm(
        self,
        x: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        gamma: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Applies Batch Normalization: y = ((x - mean) / sqrt(var + eps)) * gamma + beta."""
        self.total_operations += x.size * 2
        inv_std = 1.0 / np.sqrt(var + eps)
        scale = inv_std if gamma is None else inv_std * gamma
        shift = -mean * scale if beta is None else beta - mean * scale

        if x.ndim == 4 and scale.ndim == 1:
            scale = scale.reshape(1, -1, 1, 1)
            shift = shift.reshape(1, -1, 1, 1)
        elif x.ndim == 2 and scale.ndim == 1:
            scale = scale.reshape(1, -1)
            shift = shift.reshape(1, -1)

        return x * scale + shift

    def scale(
        self,
        x: np.ndarray,
        scale_val: Union[float, np.ndarray],
        bias_val: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """Applies linear scaling: y = x * scale + bias."""
        self.total_operations += x.size
        res = x * scale_val
        if bias_val is not None:
            res = self.bias_add(res, np.asarray(bias_val))
        return res

    # -------------------------------------------------------------------------
    # 2. Fixed-Function Activations
    # -------------------------------------------------------------------------

    def relu(self, x: np.ndarray) -> np.ndarray:
        """Computes ReLU activation: max(0, x)."""
        self.total_operations += x.size
        self.total_activations += 1
        return np.maximum(0, x)

    def leaky_relu(self, x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
        """Computes LeakyReLU activation: max(x, x * negative_slope)."""
        self.total_operations += x.size
        self.total_activations += 1
        return np.where(x >= 0, x, x * negative_slope)

    def prelu(self, x: np.ndarray, slope: np.ndarray) -> np.ndarray:
        """Computes Parametric ReLU (PReLU) with per-channel learnable slope."""
        self.total_operations += x.size
        self.total_activations += 1
        s = np.asarray(slope)
        if x.ndim == 4 and s.ndim == 1:
            s = s.reshape(1, -1, 1, 1)
        elif x.ndim == 2 and s.ndim == 1:
            s = s.reshape(1, -1)
        return np.where(x >= 0, x, x * s)

    def clip(self, x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Clips tensor values within [min_val, max_val]."""
        self.total_operations += x.size
        self.total_activations += 1
        return np.clip(x, min_val, max_val)

    # -------------------------------------------------------------------------
    # 3. Lookup Table (LUT) Piecewise Linear Interpolation
    # -------------------------------------------------------------------------

    def interpolate_lut(
        self,
        x: np.ndarray,
        lut: Union[bytes, SDPLookupTable],
        table_type: str = "LO",
        activation_type: Optional[str] = None,
    ) -> np.ndarray:
        """Evaluates hardware piecewise linear interpolation against an SDP LUT table.

        Args:
            x: Input activation tensor.
            lut: 688-byte raw buffer or SDPLookupTable object.
            table_type: "LO" (Linear-Only, 257 points) or "LE" (Linear-Exp, 65 points).
            activation_type: Optional hint ('silu', 'sigmoid', 'tanh') for domain saturation rules.

        Returns:
            Interpolated output tensor matching x's shape.
        """
        self.total_operations += x.size * 3
        self.total_activations += 1

        lut_desc = SDPLookupTable.from_bytes(lut) if isinstance(lut, (bytes, bytearray, memoryview)) else lut

        if table_type.upper() == "LE":
            table = lut_desc.linear_exp_table.astype(np.float32)
            start = lut_desc.le_start
            end = lut_desc.le_end
            num_intervals = 64
        else:  # "LO" default
            table = lut_desc.linear_only_table.astype(np.float32)
            start = lut_desc.lo_start
            end = lut_desc.lo_end
            num_intervals = 256

        x_f32 = x.astype(np.float32, copy=False)
        step = (end - start) / float(num_intervals)

        # Clip values to domain for table indexing
        x_clamped = np.clip(x_f32, start, end)
        idx_float = (x_clamped - start) / step
        idx0 = np.floor(idx_float).astype(np.int32)
        idx0 = np.clip(idx0, 0, num_intervals - 1)
        idx1 = idx0 + 1

        alpha = idx_float - idx0

        # Piecewise linear interpolation
        y_interp = (1.0 - alpha) * table[idx0] + alpha * table[idx1]

        # Domain boundary handling (extrapolation / saturation)
        # Check if table represents SiLU or Sigmoid based on end sample
        is_silu = (table[-1] > 2.0) or (activation_type and activation_type.lower() == "silu")
        is_sigmoid = (0.9 < table[-1] <= 1.1) or (activation_type and activation_type.lower() == "sigmoid")

        if is_silu:
            # SiLU: for x > end, f(x) -> x; for x < start, f(x) -> 0.0
            out = np.where(x_f32 > end, x_f32, y_interp)
            out = np.where(x_f32 < start, 0.0, out)
        elif is_sigmoid:
            # Sigmoid: for x > end, f(x) -> 1.0; for x < start, f(x) -> 0.0
            out = np.where(x_f32 > end, 1.0, y_interp)
            out = np.where(x_f32 < start, 0.0, out)
        else:
            # General LUT: clamp to boundaries
            out = np.where(x_f32 > end, table[-1], y_interp)
            out = np.where(x_f32 < start, table[0], out)

        return out.astype(x.dtype)

    def silu(
        self,
        x: np.ndarray,
        lut: Optional[Union[bytes, SDPLookupTable]] = None,
        use_lut: bool = False,
    ) -> np.ndarray:
        """Computes SiLU / Swish activation: x * sigmoid(x)."""
        if use_lut:
            if lut is None:
                lut = SDPLookupTable.create_silu_lut()
            return self.interpolate_lut(x, lut, table_type="LO", activation_type="silu")

        self.total_operations += x.size * 2
        self.total_activations += 1
        x_f32 = x.astype(np.float32, copy=False)
        res = x_f32 / (1.0 + np.exp(-x_f32))
        return res.astype(x.dtype)

    def sigmoid(
        self,
        x: np.ndarray,
        lut: Optional[Union[bytes, SDPLookupTable]] = None,
        use_lut: bool = False,
    ) -> np.ndarray:
        """Computes Sigmoid activation: 1 / (1 + exp(-x))."""
        if use_lut:
            if lut is None:
                lut = SDPLookupTable.create_sigmoid_lut()
            return self.interpolate_lut(x, lut, table_type="LO", activation_type="sigmoid")

        self.total_operations += x.size * 2
        self.total_activations += 1
        x_f32 = x.astype(np.float32, copy=False)
        res = 1.0 / (1.0 + np.exp(-x_f32))
        return res.astype(x.dtype)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Computes hyperbolic tangent activation."""
        self.total_operations += x.size
        self.total_activations += 1
        return np.tanh(x)

    def gelu(self, x: np.ndarray, approximate: bool = False) -> np.ndarray:
        """Computes GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))."""
        self.total_operations += x.size * 3
        self.total_activations += 1
        x_f32 = x.astype(np.float32, copy=False)
        if approximate:
            # Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            inner = np.sqrt(2.0 / np.pi) * (x_f32 + 0.044715 * np.power(x_f32, 3))
            res = 0.5 * x_f32 * (1.0 + np.tanh(inner))
        else:
            import math
            # Using erf
            from scipy.special import erf  # type: ignore
            res = 0.5 * x_f32 * (1.0 + erf(x_f32 / math.sqrt(2.0)))
        return res.astype(x.dtype)

    # -------------------------------------------------------------------------
    # 4. Elementwise ALU Operations (Eltwise)
    # -------------------------------------------------------------------------

    def eltwise_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pointwise tensor addition: a + b."""
        self.total_operations += a.size
        self.total_eltwise += 1
        return a + b

    def eltwise_sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pointwise tensor subtraction: a - b."""
        self.total_operations += a.size
        self.total_eltwise += 1
        return a - b

    def eltwise_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pointwise tensor multiplication (Hadamard product): a * b."""
        self.total_operations += a.size
        self.total_eltwise += 1
        return a * b

    def eltwise_max(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pointwise tensor maximum: max(a, b)."""
        self.total_operations += a.size
        self.total_eltwise += 1
        return np.maximum(a, b)

    def eltwise_min(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pointwise tensor minimum: min(a, b)."""
        self.total_operations += a.size
        self.total_eltwise += 1
        return np.minimum(a, b)

    # -------------------------------------------------------------------------
    # 5. Precision Conversions
    # -------------------------------------------------------------------------

    def convert_precision(
        self,
        x: np.ndarray,
        src_dtype: Union[str, np.dtype],
        dst_dtype: Union[str, np.dtype],
        scale: float = 1.0,
        zero_point: int = 0,
        clip_min: int = -128,
        clip_max: int = 127,
    ) -> np.ndarray:
        """Converts tensor between floating-point and quantized integer formats.

        Supported conversions:
        - FP16 -> INT8: clip(round(x / scale) + zero_point, clip_min, clip_max)
        - INT8 -> FP16: (x - zero_point) * scale
        - FP32 -> FP16, FP16 -> FP32
        """
        self.total_operations += x.size
        self.total_conversions += 1

        src = np.dtype(src_dtype)
        dst = np.dtype(dst_dtype)

        if src == dst:
            return x.copy()

        # Float -> INT8 Quantization
        if np.issubdtype(src, np.floating) and np.issubdtype(dst, np.signedinteger):
            x_f32 = x.astype(np.float32, copy=False)
            quantized = np.round(x_f32 / scale) + zero_point
            clipped = np.clip(quantized, clip_min, clip_max)
            return clipped.astype(dst)

        # INT8 -> Float Dequantization
        elif np.issubdtype(src, np.signedinteger) and np.issubdtype(dst, np.floating):
            x_f32 = x.astype(np.float32, copy=False)
            dequantized = (x_f32 - zero_point) * scale
            return dequantized.astype(dst)

        # Float <-> Float casting
        else:
            return x.astype(dst)

    def stats(self) -> Dict[str, Any]:
        res = super().stats()
        res.update(
            {
                "total_activations": self.total_activations,
                "total_eltwise": self.total_eltwise,
                "total_conversions": self.total_conversions,
            }
        )
        return res
