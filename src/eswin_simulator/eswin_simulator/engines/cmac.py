"""
Convolution MAC Array (CMAC) Engine simulation for ESWIN EIC7700X.
Executes 2D convolutions, grouped convolutions, and 1x1 pointwise Conv (GEMM)
for FP16 and INT8 precisions with NVDLA weight tiling and unpacking.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from eswin_simulator.engines.base_engine import HardwareEngine
from eswin_simulator.memory.address_space import VirtualAddressSpace


def unpack_nvdla_weights_conv2d(
    raw_data: Union[bytes, np.ndarray],
    c_out: int,
    c_in: int,
    kh: int,
    kw: int,
    dtype: np.dtype = np.float16,
    cout_sub: int = 8,
    cin_align: int = 4,
) -> np.ndarray:
    """Unpacks NVDLA memory-tiled weights for 2D convolution from .model blobs.

    In the NVDLA weight format for spatial convolutions (KH, KW > 1):
    Weights are laid out in blocks of `cout_sub` (typically 8) output channels,
    with input channels aligned to `cin_align` (typically 4 for input, 16/32/64 internal).

    Memory layout:
        (c_out_groups, c_in_aligned, kh, kw, cout_sub)

    Args:
        raw_data: Raw byte buffer or 1D array of weights.
        c_out: Number of true output channels.
        c_in: Number of true input channels.
        kh: Kernel height.
        kw: Kernel width.
        dtype: Element data type (np.float16 or np.int8).
        cout_sub: Sub-channel atom size for output (default 8).
        cin_align: Channel alignment for input channels (default 4).

    Returns:
        Canonical 4D weight tensor with shape (c_out, c_in, kh, kw).
    """
    if isinstance(raw_data, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(raw_data, dtype=dtype)
    else:
        arr = np.asarray(raw_data, dtype=dtype)

    c_out_groups = int(np.ceil(c_out / cout_sub))
    c_in_aligned = int(np.ceil(c_in / cin_align) * cin_align)

    expected_elements = c_out_groups * c_in_aligned * kh * kw * cout_sub
    if arr.size < expected_elements:
        raise ValueError(
            f"Weight buffer size ({arr.size}) is smaller than required "
            f"tiled size ({expected_elements}) for conv2d [C_out={c_out}, C_in={c_in}, K=({kh},{kw})]"
        )

    tiled = arr[:expected_elements].reshape(c_out_groups, c_in_aligned, kh, kw, cout_sub)
    # Permute to (c_out_groups, cout_sub, c_in_aligned, kh, kw)
    unpacked = tiled.transpose(0, 4, 1, 2, 3).reshape(c_out_groups * cout_sub, c_in_aligned, kh, kw)
    return np.ascontiguousarray(unpacked[:c_out, :c_in, :, :])


def unpack_nvdla_weights_1x1(
    raw_data: Union[bytes, np.ndarray],
    c_out: int,
    c_in: int,
    dtype: np.dtype = np.float16,
    cout_sub: int = 8,
    cin_sub: int = 2,
) -> np.ndarray:
    """Unpacks NVDLA memory-tiled weights for 1x1 pointwise convolution / GEMM.

    In the NVDLA weight format for 1x1 convolutions:
    Weights are laid out in atomic sub-tiles of (cout_sub=8, cin_sub=2) to match
    32-byte memory burst transactions.

    Memory layout:
        (c_out // cout_sub, c_in // cin_sub, cout_sub, cin_sub)

    Args:
        raw_data: Raw byte buffer or 1D array of weights.
        c_out: Number of true output channels / rows.
        c_in: Number of true input channels / columns.
        dtype: Element data type (np.float16 or np.int8).
        cout_sub: Sub-channel atom size for output (default 8).
        cin_sub: Sub-channel atom size for input pairs (default 2).

    Returns:
        Canonical 2D weight matrix with shape (c_out, c_in).
    """
    if isinstance(raw_data, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(raw_data, dtype=dtype)
    else:
        arr = np.asarray(raw_data, dtype=dtype)

    total_elements = arr.size
    c_in_padded = total_elements // c_out if c_out > 0 else c_in
    if c_in_padded < c_in:
        c_in_padded = c_in

    expected_elements = (c_out // cout_sub) * (c_in_padded // cin_sub) * cout_sub * cin_sub
    if arr.size < expected_elements:
        raise ValueError(
            f"Weight buffer size ({arr.size}) is smaller than required "
            f"tiled size ({expected_elements}) for 1x1 conv [C_out={c_out}, C_in={c_in}, C_in_padded={c_in_padded}]"
        )

    tiled = arr[:expected_elements].reshape(c_out // cout_sub, c_in_padded // cin_sub, cout_sub, cin_sub)
    # Permute to (c_out // cout_sub, cout_sub, c_in_padded // cin_sub, cin_sub)
    unpacked = tiled.transpose(0, 2, 1, 3).reshape(c_out, c_in_padded)
    return np.ascontiguousarray(unpacked[:c_out, :c_in])


def unpack_nvdla_weights(
    raw_data: Union[bytes, np.ndarray],
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float16,
) -> np.ndarray:
    """Universal dispatcher for unpacking NVDLA weights from .model blobs.

    Args:
        raw_data: Raw byte buffer or array.
        shape: Canonical target shape:
               - (C_out, C_in) for 2D GEMM / 1x1 conv
               - (C_out, C_in, 1, 1) for 4D 1x1 conv
               - (C_out, C_in, KH, KW) for 4D spatial conv
        dtype: Data type (np.float16 or np.int8).

    Returns:
        Unpacked canonical weight array matching requested shape.
    """
    if len(shape) == 2:
        c_out, c_in = shape
        return unpack_nvdla_weights_1x1(raw_data, c_out=c_out, c_in=c_in, dtype=dtype)
    elif len(shape) == 4:
        c_out, c_in, kh, kw = shape
        if kh == 1 and kw == 1:
            w2d = unpack_nvdla_weights_1x1(raw_data, c_out=c_out, c_in=c_in, dtype=dtype)
            return w2d.reshape(c_out, c_in, 1, 1)
        else:
            return unpack_nvdla_weights_conv2d(raw_data, c_out=c_out, c_in=c_in, kh=kh, kw=kw, dtype=dtype)
    else:
        raise ValueError(f"Unsupported weight shape: {shape}")


class CMACEngine(HardwareEngine):
    """Simulates the NPU Convolution MAC Array (CMAC) hardware unit.

    Performs 2D spatial convolution and 1x1 pointwise convolution (GEMM).
    - In FP16 mode: multiplication is performed using half-precision IEEE 754,
      accumulated into a 32-bit floating point (float32) accumulator.
    - In INT8 mode: multiplication is performed using 8-bit signed integers,
      accumulated into a 32-bit signed integer (int32) accumulator.
    """

    def __init__(
        self,
        address_space: VirtualAddressSpace,
        default_precision: str = "FP16",
    ):
        super().__init__(name="CMAC", address_space=address_space)
        self.default_precision = default_precision.upper()
        self.total_macs: int = 0
        self.total_convolutions: int = 0
        self.total_gemms: int = 0

    def reset(self) -> None:
        super().reset()
        self.total_macs = 0
        self.total_convolutions = 0
        self.total_gemms = 0

    def conv2d(
        self,
        x: np.ndarray,
        w: np.ndarray,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        precision: Optional[str] = None,
    ) -> np.ndarray:
        """Executes a 2D spatial convolution.

        Args:
            x: Input activation tensor with shape (N, C_in, H, W).
            w: Weight tensor with shape (C_out, C_in // groups, KH, KW).
            stride: Stride (int or (stride_h, stride_w)).
            padding: Padding (int, (pad_h, pad_w), or (pad_top, pad_bottom, pad_left, pad_right)).
            dilation: Dilation (int or (dil_h, dil_w)).
            groups: Number of channel groups (default 1, depthwise if groups == C_in == C_out).
            precision: 'FP16' or 'INT8'. If None, uses default_precision.

        Returns:
            32-bit accumulator array (float32 for FP16 mode, int32 for INT8 mode)
            with shape (N, C_out, H_out, W_out).
        """
        prec = (precision or self.default_precision).upper()

        if x.ndim != 4:
            raise ValueError(f"Input tensor x must be 4D (N, C, H, W), got shape {x.shape}")
        if w.ndim != 4:
            raise ValueError(f"Weight tensor w must be 4D (C_out, C_in_g, KH, KW), got shape {w.shape}")

        N, C_in, H, W = x.shape
        C_out, C_in_g, KH, KW = w.shape

        if C_in % groups != 0 or C_out % groups != 0:
            raise ValueError(f"Channels (C_in={C_in}, C_out={C_out}) not divisible by groups={groups}")
        if C_in_g != C_in // groups:
            raise ValueError(f"Weight channel dim {C_in_g} does not match C_in // groups ({C_in // groups})")

        sh, sw = (stride, stride) if isinstance(stride, int) else stride
        dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation

        if isinstance(padding, int):
            pad_top = pad_bottom = pad_left = pad_right = padding
        elif len(padding) == 2:
            pad_top = pad_bottom = padding[0]
            pad_left = pad_right = padding[1]
        elif len(padding) == 4:
            pad_top, pad_bottom, pad_left, pad_right = padding
        else:
            raise ValueError(f"Invalid padding specification: {padding}")

        # Compute effective kernel dimensions considering dilation
        eff_kh = 1 + (KH - 1) * dh
        eff_kw = 1 + (KW - 1) * dw

        H_out = (H + pad_top + pad_bottom - eff_kh) // sh + 1
        W_out = (W + pad_left + pad_right - eff_kw) // sw + 1

        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"Invalid output dimensions ({H_out}, {W_out}) for input ({H}, {W}), "
                f"kernel ({KH}, {KW}), padding ({pad_top}, {pad_bottom}, {pad_left}, {pad_right}), "
                f"stride ({sh}, {sw})"
            )

        # Pad input if needed
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            pad_val = 0
            x_padded = np.pad(
                x,
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=pad_val,
            )
        else:
            x_padded = x

        # Precision handling
        if prec == "INT8":
            x_calc = x_padded.astype(np.int32, copy=False)
            w_calc = w.astype(np.int32, copy=False)
            accum_dtype = np.int32
        else:  # FP16 / default
            x_calc = x_padded.astype(np.float32, copy=False)
            w_calc = w.astype(np.float32, copy=False)
            accum_dtype = np.float32

        # Fast path for 1x1 conv with stride 1, no dilation, no padding
        if eff_kh == 1 and eff_kw == 1 and sh == 1 and sw == 1 and groups == 1:
            # (N, C_in, H, W) -> (N, H, W, C_in)
            x_perm = x_calc.transpose(0, 2, 3, 1)
            # w: (C_out, C_in)
            w_mat = w_calc.reshape(C_out, C_in)
            out = np.matmul(x_perm, w_mat.T).transpose(0, 3, 1, 2)
            mac_count = N * C_out * C_in * H_out * W_out
        else:
            # Vectorized sliding window view
            windows = np.lib.stride_tricks.sliding_window_view(x_calc, (eff_kh, eff_kw), axis=(2, 3))
            # Subsample by stride and dilation
            windows_strided = windows[:, :, ::sh, ::sw, ::dh, ::dw]

            if groups == 1:
                # windows: (N, C_in, H_out, W_out, KH, KW)
                # w: (C_out, C_in, KH, KW)
                out = np.einsum("nchwkl,ockl->nohw", windows_strided, w_calc, dtype=accum_dtype)
            else:
                C_in_g = C_in // groups
                C_out_g = C_out // groups
                win_g = windows_strided.reshape(N, groups, C_in_g, H_out, W_out, KH, KW)
                w_g = w_calc.reshape(groups, C_out_g, C_in_g, KH, KW)
                out_g = np.einsum("ngchwkl,gockl->ngohw", win_g, w_g, dtype=accum_dtype)
                out = out_g.reshape(N, C_out, H_out, W_out)

            mac_count = N * C_out * (C_in // groups) * KH * KW * H_out * W_out

        self.total_macs += mac_count
        self.total_operations += mac_count
        self.total_convolutions += 1

        return out

    def gemm_1x1(
        self,
        x: np.ndarray,
        w: np.ndarray,
        precision: Optional[str] = None,
    ) -> np.ndarray:
        """Executes a 1x1 Pointwise Convolution / GEMM matrix multiplication.

        Computes:
            y = x @ W^T

        Supports inputs in 2D (N, C_in), 3D (B, S, C_in), or 4D (N, C_in, 1, 1).

        Args:
            x: Input activations.
            w: 2D weight matrix (C_out, C_in) or 4D weight tensor (C_out, C_in, 1, 1).
            precision: 'FP16' or 'INT8'.

        Returns:
            Accumulator array (float32 for FP16, int32 for INT8) with shape matching x's batch dims.
        """
        prec = (precision or self.default_precision).upper()

        orig_shape = x.shape
        orig_ndim = x.ndim

        # Flatten weights to 2D
        if w.ndim == 4:
            w_2d = w.reshape(w.shape[0], w.shape[1])
        elif w.ndim == 2:
            w_2d = w
        else:
            raise ValueError(f"Weight tensor w must be 2D or 4D, got shape {w.shape}")

        C_out, C_in = w_2d.shape

        if orig_ndim == 4:
            N, C_in_act, H, W = orig_shape
            if C_in_act != C_in:
                raise ValueError(f"Channel mismatch: input C_in={C_in_act}, weight C_in={C_in}")
            x_2d = x.transpose(0, 2, 3, 1).reshape(-1, C_in)
        elif orig_ndim == 3:
            B, S, C_in_act = orig_shape
            if C_in_act != C_in:
                raise ValueError(f"Channel mismatch: input C_in={C_in_act}, weight C_in={C_in}")
            x_2d = x.reshape(-1, C_in)
        elif orig_ndim == 2:
            N, C_in_act = orig_shape
            if C_in_act != C_in:
                raise ValueError(f"Channel mismatch: input C_in={C_in_act}, weight C_in={C_in}")
            x_2d = x
        elif orig_ndim == 1:
            if x.shape[0] != C_in:
                raise ValueError(f"Channel mismatch: input len={x.shape[0]}, weight C_in={C_in}")
            x_2d = x.reshape(1, C_in)
        else:
            raise ValueError(f"Unsupported input ndim: {orig_ndim}")

        if prec == "INT8":
            x_calc = x_2d.astype(np.int32, copy=False)
            w_calc = w_2d.astype(np.int32, copy=False)
            out_2d = np.matmul(x_calc, w_calc.T)
        else:
            x_calc = x_2d.astype(np.float32, copy=False)
            w_calc = w_2d.astype(np.float32, copy=False)
            out_2d = np.matmul(x_calc, w_calc.T)

        mac_count = x_2d.shape[0] * C_out * C_in
        self.total_macs += mac_count
        self.total_operations += mac_count
        self.total_gemms += 1

        # Reshape back to match input representation
        if orig_ndim == 4:
            N, _, H, W = orig_shape
            out = out_2d.reshape(N, H, W, C_out).transpose(0, 3, 1, 2)
            return np.ascontiguousarray(out)
        elif orig_ndim == 3:
            B, S, _ = orig_shape
            return out_2d.reshape(B, S, C_out)
        elif orig_ndim == 2:
            return out_2d
        else:
            return out_2d.reshape(C_out)

    def stats(self) -> Dict[str, Any]:
        res = super().stats()
        res.update(
            {
                "total_macs": self.total_macs,
                "total_convolutions": self.total_convolutions,
                "total_gemms": self.total_gemms,
                "precision": self.default_precision,
            }
        )
        return res
