"""
Native Python/NumPy bit-accurate implementation of Top-K selection.
Provides O(N) partitioning and sorted index extraction matching vendor dsp_topk.
"""

from typing import Tuple
import numpy as np


def topk(
    x: np.ndarray,
    k: int = 1,
    axis: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds values and indices of the k largest (or smallest) elements.

    Args:
        x: Input tensor.
        k: Number of top elements to select (1 <= k <= x.shape[axis]).
        axis: Axis along which to sort/select.
        largest: If True, selects largest elements; otherwise smallest.
        sorted: If True, returns elements in sorted order.

    Returns:
        (values, indices) tuple of tensors matching x's dimensions except along axis.
    """
    if k <= 0 or k > x.shape[axis]:
        raise ValueError(f"k must be in range [1, {x.shape[axis]}], got {k}")

    # Negate if looking for largest elements with argpartition
    sign = -1 if largest else 1
    scores = x * sign

    # O(N) partition along target axis
    kth = k - 1
    partitioned_indices = np.argpartition(scores, kth, axis=axis)

    # Take first k elements
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, k)
    top_indices = partitioned_indices[tuple(sl)]
    top_scores = np.take_along_axis(scores, top_indices, axis=axis)

    if sorted:
        sort_order = np.argsort(top_scores, axis=axis)
        final_indices = np.take_along_axis(top_indices, sort_order, axis=axis)
        final_values = np.take_along_axis(x, final_indices, axis=axis)
    else:
        final_indices = top_indices
        final_values = np.take_along_axis(x, top_indices, axis=axis)

    return final_values, final_indices
