"""
Zero-copy NumPy array views over simulated memory buffers.
"""

from typing import Tuple, Optional, Union
import numpy as np


def create_tensor_view(
    buffer: Union[bytearray, memoryview, bytes],
    offset: int,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    strides: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Creates a NumPy ndarray view over a region of a memory buffer.

    Args:
        buffer: The underlying memory buffer.
        offset: Byte offset into the buffer.
        shape: Desired tensor shape (e.g. (N, C, H, W)).
        dtype: NumPy data type.
        strides: Optional byte strides. If None, default C-contiguous strides are used.

    Returns:
        np.ndarray view referencing the buffer memory directly.
    """
    total_elements = 1
    for dim in shape:
        total_elements *= max(1, dim)

    total_bytes = total_elements * dtype.itemsize

    if offset + total_bytes > len(buffer):
        raise ValueError(
            f"Buffer view out of bounds: offset {offset} + {total_bytes} bytes "
            f"exceeds buffer capacity of {len(buffer)} bytes."
        )

    # Use memoryview to create zero-copy slice
    mview = memoryview(buffer)[offset : offset + total_bytes]
    arr = np.frombuffer(mview, dtype=dtype)

    if strides is not None:
        # Validate and apply custom strides if compatible
        try:
            arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            return arr
        except Exception:
            pass

    return arr.reshape(shape)
