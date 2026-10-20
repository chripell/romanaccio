"""
Planar Data Processor (PDP) Simulation Engine for ESWIN EIC7700X.
Simulates 2D Max Pooling, Average Pooling, and Global Spatial Pooling
across FP16, FP32, and INT8 precisions.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from eswin_simulator.engines.base_engine import HardwareEngine
from eswin_simulator.memory.address_space import VirtualAddressSpace


class PDPEngine(HardwareEngine):
    """Planar Data Processor (PDP) simulating hardware pooling operations.

    In the NVDLA/ESWIN architecture, PDP performs 2D pooling (Max, Average, Min)
    on planar feature maps directly from SPRAM or DRAM with hardware line-buffers.
    """

    def __init__(self, address_space: VirtualAddressSpace):
        super().__init__(name="PDP_PlanarEngine", address_space=address_space)
        self.total_pools: int = 0
        self.max_pool_count: int = 0
        self.avg_pool_count: int = 0

    def reset(self) -> None:
        super().reset()
        self.total_pools = 0
        self.max_pool_count = 0
        self.avg_pool_count = 0

    @staticmethod
    def _normalize_2d_param(param: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(param, (int, np.integer)):
            return int(param), int(param)
        if isinstance(param, (tuple, list)) and len(param) == 2:
            return int(param[0]), int(param[1])
        raise ValueError(f"Expected int or tuple of 2 ints, got: {param}")

    @staticmethod
    def _normalize_padding(
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int]:
        """Returns (pad_top, pad_bottom, pad_left, pad_right)."""
        if isinstance(padding, (int, np.integer)):
            p = int(padding)
            return p, p, p, p
        if isinstance(padding, (tuple, list)):
            if len(padding) == 2:
                ph, pw = int(padding[0]), int(padding[1])
                return ph, ph, pw, pw
            if len(padding) == 4:
                return int(padding[0]), int(padding[1]), int(padding[2]), int(padding[3])
        raise ValueError(f"Expected int or 2/4-tuple padding, got: {padding}")

    def max_pool2d(
        self,
        x: np.ndarray,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = 0,
        ceil_mode: bool = False,
    ) -> np.ndarray:
        """Executes 2D Max Pooling on tensor x (NCHW or CHW)."""
        kh, kw = self._normalize_2d_param(kernel_size)
        sh, sw = self._normalize_2d_param(stride if stride is not None else kernel_size)
        pt, pb, pl, pr = self._normalize_padding(padding)

        orig_shape = x.shape
        orig_dtype = x.dtype
        x_4d = x if x.ndim == 4 else x.reshape(-1, *x.shape[-3:])
        if x_4d.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor for max_pool2d, got shape {orig_shape}")

        n, c, h, w = x_4d.shape

        # Compute output spatial dimensions
        if ceil_mode:
            h_out = int(np.ceil((h + pt + pb - kh) / sh)) + 1
            w_out = int(np.ceil((w + pl + pr - kw) / sw)) + 1
        else:
            h_out = (h + pt + pb - kh) // sh + 1
            w_out = (w + pl + pr - kw) // sw + 1

        if h_out <= 0 or w_out <= 0:
            raise ValueError(f"Invalid pooling output dimensions ({h_out}, {w_out}) for input ({h}, {w})")

        # Apply spatial padding with negative infinity / minimum value
        if pt > 0 or pb > 0 or pl > 0 or pr > 0:
            min_val = np.iinfo(orig_dtype).min if np.issubdtype(orig_dtype, np.integer) else -np.inf
            padded = np.full((n, c, h + pt + pb, w + pl + pr), fill_value=min_val, dtype=orig_dtype)
            padded[:, :, pt : pt + h, pl : pl + w] = x_4d
        else:
            padded = x_4d

        # Sliding window extraction
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(2, 3))
        # windows shape: (n, c, H_win, W_win, kh, kw)
        strided_windows = windows[:, :, : h_out * sh : sh, : w_out * sw : sw, :, :]
        out = np.max(strided_windows, axis=(-2, -1))

        self.total_pools += 1
        self.max_pool_count += 1
        self.total_operations += out.size * kh * kw

        if len(orig_shape) == 3:
            return out[0]
        return out

    def avg_pool2d(
        self,
        x: np.ndarray,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> np.ndarray:
        """Executes 2D Average Pooling on tensor x (NCHW or CHW)."""
        kh, kw = self._normalize_2d_param(kernel_size)
        sh, sw = self._normalize_2d_param(stride if stride is not None else kernel_size)
        pt, pb, pl, pr = self._normalize_padding(padding)

        orig_shape = x.shape
        orig_dtype = x.dtype
        is_int8 = np.issubdtype(orig_dtype, np.integer)

        x_4d = x if x.ndim == 4 else x.reshape(-1, *x.shape[-3:])
        if x_4d.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor for avg_pool2d, got shape {orig_shape}")

        n, c, h, w = x_4d.shape

        if ceil_mode:
            h_out = int(np.ceil((h + pt + pb - kh) / sh)) + 1
            w_out = int(np.ceil((w + pl + pr - kw) / sw)) + 1
        else:
            h_out = (h + pt + pb - kh) // sh + 1
            w_out = (w + pl + pr - kw) // sw + 1

        if h_out <= 0 or w_out <= 0:
            raise ValueError(f"Invalid pooling output dimensions ({h_out}, {w_out}) for input ({h}, {w})")

        # Floating point conversion for accumulation
        x_calc = x_4d.astype(np.float32)

        if pt > 0 or pb > 0 or pl > 0 or pr > 0:
            padded = np.zeros((n, c, h + pt + pb, w + pl + pr), dtype=np.float32)
            padded[:, :, pt : pt + h, pl : pl + w] = x_calc
            if not count_include_pad:
                mask = np.zeros((1, 1, h + pt + pb, w + pl + pr), dtype=np.float32)
                mask[:, :, pt : pt + h, pl : pl + w] = 1.0
            else:
                mask = None
        else:
            padded = x_calc
            mask = None

        windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(2, 3))
        strided_windows = windows[:, :, : h_out * sh : sh, : w_out * sw : sw, :, :]

        if mask is not None:
            mask_windows = np.lib.stride_tricks.sliding_window_view(mask, (kh, kw), axis=(2, 3))
            strided_mask = mask_windows[:, :, : h_out * sh : sh, : w_out * sw : sw, :, :]
            counts = np.sum(strided_mask, axis=(-2, -1))
            counts = np.maximum(counts, 1.0)
            sums = np.sum(strided_windows, axis=(-2, -1))
            out_calc = sums / counts
        else:
            out_calc = np.mean(strided_windows, axis=(-2, -1))

        if is_int8:
            # Round and saturate to [-128, 127]
            out = np.clip(np.round(out_calc), -128, 127).astype(orig_dtype)
        else:
            out = out_calc.astype(orig_dtype)

        self.total_pools += 1
        self.avg_pool_count += 1
        self.total_operations += out.size * kh * kw

        if len(orig_shape) == 3:
            return out[0]
        return out

    def global_avg_pool2d(self, x: np.ndarray) -> np.ndarray:
        """Executes Global Spatial Average Pooling across H and W axes."""
        orig_shape = x.shape
        orig_dtype = x.dtype
        is_int8 = np.issubdtype(orig_dtype, np.integer)

        x_4d = x if x.ndim == 4 else x.reshape(-1, *x.shape[-3:])
        x_calc = x_4d.astype(np.float32)
        out_calc = np.mean(x_calc, axis=(2, 3), keepdims=True)

        if is_int8:
            out = np.clip(np.round(out_calc), -128, 127).astype(orig_dtype)
        else:
            out = out_calc.astype(orig_dtype)

        self.total_pools += 1
        self.avg_pool_count += 1
        self.total_operations += x_4d.size

        if len(orig_shape) == 3:
            return out[0]
        return out

    def global_max_pool2d(self, x: np.ndarray) -> np.ndarray:
        """Executes Global Spatial Max Pooling across H and W axes."""
        orig_shape = x.shape
        x_4d = x if x.ndim == 4 else x.reshape(-1, *x.shape[-3:])
        out = np.max(x_4d, axis=(2, 3), keepdims=True)

        self.total_pools += 1
        self.max_pool_count += 1
        self.total_operations += x_4d.size

        if len(orig_shape) == 3:
            return out[0]
        return out

    def stats(self) -> Dict[str, Any]:
        res = super().stats()
        res.update(
            {
                "total_pools": self.total_pools,
                "max_pool_count": self.max_pool_count,
                "avg_pool_count": self.avg_pool_count,
            }
        )
        return res
