"""
Native Python/NumPy bit-accurate implementation of Softmax.
Features online max-subtraction for numerical stability across arbitrary axes.
"""

from typing import Optional, Union
import numpy as np


def softmax(
    x: np.ndarray,
    axis: int = -1,
    out_dtype: Optional[Union[str, np.dtype]] = None,
) -> np.ndarray:
    """Computes numerically stable Softmax activation along specified axis.

    Mathematical formulation:
        m = max(x, axis=axis, keepdims=True)
        e = exp(x - m)
        y = e / sum(e, axis=axis, keepdims=True)

    Args:
        x: Input logits array.
        axis: Axis along which softmax is computed (default -1).
        out_dtype: Desired output data type (default matches input dtype).

    Returns:
        Probability distribution tensor summing to 1.0 along axis.
    """
    x_f32 = x.astype(np.float32, copy=False)

    # Online max subtraction to prevent exp overflow
    m = np.max(x_f32, axis=axis, keepdims=True)
    exp_shifted = np.exp(x_f32 - m)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)

    out = exp_shifted / sum_exp

    target_dtype = np.dtype(out_dtype) if out_dtype is not None else x.dtype
    return out.astype(target_dtype)
