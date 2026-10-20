"""
Rubik Data Transformation Engine for ESWIN EIC7700X.
Simulates tensor restructuring hardware operations:
Reshape, Transpose/Permute, Slice, Concat, Squeeze, Flatten, and Depth/Space transformations.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from eswin_simulator.engines.base_engine import HardwareEngine
from eswin_simulator.memory.address_space import VirtualAddressSpace


class RubikEngine(HardwareEngine):
    """Rubik Engine simulating tensor reordering and data transformation hardware.

    In the NVDLA/ESWIN architecture, the Rubik hardware engine handles arbitrary
    multi-dimensional data layouts, channel reshuffling, axis transpositions,
    tensor slicing/merging, and planar $\\leftrightarrow$ interleaved conversions
    directly between DRAM and on-chip SRAM without host CPU intervention.
    """

    def __init__(self, address_space: VirtualAddressSpace):
        super().__init__(name="Rubik_TransformEngine", address_space=address_space)
        self.total_reorders: int = 0
        self.bytes_reordered: int = 0
        self.reshape_count: int = 0
        self.transpose_count: int = 0
        self.slice_count: int = 0
        self.concat_count: int = 0

    def reset(self) -> None:
        super().reset()
        self.total_reorders = 0
        self.bytes_reordered = 0
        self.reshape_count = 0
        self.transpose_count = 0
        self.slice_count = 0
        self.concat_count = 0

    def reshape(self, x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Reshapes tensor x into target shape preserving C-order element sequence."""
        out = np.ascontiguousarray(x.reshape(shape))
        self.total_reorders += 1
        self.reshape_count += 1
        self.bytes_reordered += x.nbytes
        self.total_operations += x.size
        return out

    def transpose(self, x: np.ndarray, perm: Tuple[int, ...]) -> np.ndarray:
        """Permutes dimensions of tensor x according to perm and returns contiguous array."""
        out = np.ascontiguousarray(np.transpose(x, axes=perm))
        self.total_reorders += 1
        self.transpose_count += 1
        self.bytes_reordered += x.nbytes
        self.total_operations += x.size
        return out

    def slice(
        self,
        x: np.ndarray,
        starts: Union[int, List[int], Tuple[int, ...]],
        ends: Union[int, List[int], Tuple[int, ...]],
        axes: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        steps: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
    ) -> np.ndarray:
        """Extracts sub-tensor slices along specified axes."""
        if isinstance(starts, (int, np.integer)):
            starts = [int(starts)]
        if isinstance(ends, (int, np.integer)):
            ends = [int(ends)]

        if axes is None:
            axes = list(range(len(starts)))
        elif isinstance(axes, (int, np.integer)):
            axes = [int(axes)]

        if steps is None:
            steps = [1] * len(starts)
        elif isinstance(steps, (int, np.integer)):
            steps = [int(steps)]

        indices: List[Any] = [slice(None)] * x.ndim
        for start, end, axis, step in zip(starts, ends, axes, steps):
            norm_axis = axis if axis >= 0 else axis + x.ndim
            indices[norm_axis] = slice(start, end, step)

        out = np.ascontiguousarray(x[tuple(indices)])
        self.total_reorders += 1
        self.slice_count += 1
        self.bytes_reordered += out.nbytes
        self.total_operations += out.size
        return out

    def concat(self, tensors: List[np.ndarray], axis: int = 0) -> np.ndarray:
        """Concatenates a list of tensors along the given axis."""
        if not tensors:
            raise ValueError("concat requires at least one tensor")

        out = np.ascontiguousarray(np.concatenate(tensors, axis=axis))
        self.total_reorders += 1
        self.concat_count += 1
        self.bytes_reordered += out.nbytes
        self.total_operations += out.size
        return out

    def split(
        self,
        x: np.ndarray,
        split_size_or_sections: Union[int, List[int]],
        axis: int = 0,
    ) -> List[np.ndarray]:
        """Splits tensor x into sub-tensors along specified axis."""
        norm_axis = axis if axis >= 0 else axis + x.ndim
        dim_size = x.shape[norm_axis]

        if isinstance(split_size_or_sections, (int, np.integer)):
            chunk_size = int(split_size_or_sections)
            num_chunks = int(np.ceil(dim_size / chunk_size))
            split_points = [i * chunk_size for i in range(1, num_chunks)]
        else:
            split_points = np.cumsum(split_size_or_sections)[:-1]

        splits = np.split(x, split_points, axis=norm_axis)
        out = [np.ascontiguousarray(s) for s in splits]

        self.total_reorders += 1
        self.slice_count += len(out)
        self.bytes_reordered += x.nbytes
        self.total_operations += x.size
        return out

    def squeeze(
        self,
        x: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> np.ndarray:
        """Removes single-dimensional entries from tensor shape."""
        out = np.ascontiguousarray(np.squeeze(x, axis=axis))
        self.total_reorders += 1
        self.bytes_reordered += x.nbytes
        self.total_operations += x.size
        return out

    def unsqueeze(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Inserts a new axis of size 1 at specified position."""
        out = np.ascontiguousarray(np.expand_dims(x, axis=axis))
        self.total_reorders += 1
        self.bytes_reordered += x.nbytes
        self.total_operations += x.size
        return out

    def flatten(self, x: np.ndarray, start_dim: int = 0, end_dim: int = -1) -> np.ndarray:
        """Flattens a contiguous range of dimensions into a single dimension."""
        norm_start = start_dim if start_dim >= 0 else start_dim + x.ndim
        norm_end = end_dim if end_dim >= 0 else end_dim + x.ndim

        new_shape = (
            x.shape[:norm_start]
            + (int(np.prod(x.shape[norm_start : norm_end + 1])),)
            + x.shape[norm_end + 1 :]
        )
        return self.reshape(x, new_shape)

    def space_to_depth(self, x: np.ndarray, block_size: int) -> np.ndarray:
        """Reorganizes spatial blocks into depth (NCHW format).

        Transforms (N, C, H * B, W * B) -> (N, C * B^2, H, W).
        """
        if x.ndim != 4:
            raise ValueError(f"space_to_depth requires 4D tensor, got shape {x.shape}")
        n, c, h, w = x.shape
        if h % block_size != 0 or w % block_size != 0:
            raise ValueError(f"H ({h}) and W ({w}) must be divisible by block_size ({block_size})")

        b = block_size
        h_out = h // b
        w_out = w // b

        reshaped = x.reshape(n, c, h_out, b, w_out, b)
        transposed = np.transpose(reshaped, (0, 3, 5, 1, 2, 4))
        out = np.ascontiguousarray(transposed.reshape(n, c * b * b, h_out, w_out))

        self.total_reorders += 1
        self.bytes_reordered += x.nbytes
        self.total_operations += x.size
        return out

    def depth_to_space(self, x: np.ndarray, block_size: int) -> np.ndarray:
        """Reorganizes depth blocks into spatial dimensions (NCHW format).

        Transforms (N, C * B^2, H, W) -> (N, C, H * B, W * B).
        """
        if x.ndim != 4:
            raise ValueError(f"depth_to_space requires 4D tensor, got shape {x.shape}")
        n, c, h, w = x.shape
        b = block_size
        if c % (b * b) != 0:
            raise ValueError(f"Channels ({c}) must be divisible by block_size^2 ({b * b})")

        c_out = c // (b * b)
        reshaped = x.reshape(n, b, b, c_out, h, w)
        transposed = np.transpose(reshaped, (0, 3, 4, 1, 5, 2))
        out = np.ascontiguousarray(transposed.reshape(n, c_out, h * b, w * b))

        self.total_reorders += 1
        self.bytes_reordered += x.nbytes
        self.total_operations += x.size
        return out

    def stats(self) -> Dict[str, Any]:
        res = super().stats()
        res.update(
            {
                "total_reorders": self.total_reorders,
                "bytes_reordered": self.bytes_reordered,
                "reshape_count": self.reshape_count,
                "transpose_count": self.transpose_count,
                "slice_count": self.slice_count,
                "concat_count": self.concat_count,
            }
        )
        return res
