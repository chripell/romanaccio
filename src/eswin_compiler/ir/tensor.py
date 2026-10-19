"""Tensor data structures for ESWIN IR (EIR)."""

from dataclasses import dataclass, field
import numpy as np
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .node import EIRNode


# Byte width map for supported EIR datatypes
DTYPE_BYTE_SIZES = {
    "float32": 4,
    "float16": 2,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "int64": 8,
    "uint64": 8,
}


def normalize_to_4d(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """Converts 1D, 2D, 3D, or 4D shape tuples into canonical NPU 4D format [N, C, H, W].
    
    The EIC7700X NPU hardware requires all tensor descriptors to specify 4 dimensions:
    - 1D: [C] -> [1, C, 1, 1]
    - 2D: [N, C] or [M, K] -> [N, C, 1, 1]
    - 3D: [N, C, W] -> [N, C, 1, W]
    - 4D: [N, C, H, W] -> unchanged
    """
    if len(shape) == 1:
        return (1, shape[0], 1, 1)
    elif len(shape) == 2:
        return (shape[0], shape[1], 1, 1)
    elif len(shape) == 3:
        return (shape[0], shape[1], 1, shape[2])
    elif len(shape) == 4:
        return (shape[0], shape[1], shape[2], shape[3])
    elif len(shape) == 0:
        return (1, 1, 1, 1)
    else:
        # Collapse higher dimensions into batch if > 4D
        prod_lead = int(np.prod(shape[:-3]))
        return (prod_lead, shape[-3], shape[-2], shape[-1])


@dataclass
class EIRTensor:
    """Represents a data tensor in the EIR computational graph.
    
    Attributes:
        name: Unique tensor name in the graph.
        shape: 4D tensor dimensions (N, C, H, W).
        dtype: Data type string ("float16", "float32", "int8", etc.).
        data: Optional numpy array if this is a constant/weight tensor.
        memory_domain: Target memory residency ("SYSTEM" for DRAM, "SRAM" for SPRAM).
        alignment: Required memory alignment in bytes (default: 4096).
    """
    name: str
    shape: Tuple[int, int, int, int]
    dtype: str = "float16"
    data: Optional[np.ndarray] = None
    memory_domain: str = "SYSTEM"
    alignment: int = 4096

    # Graph linkages
    producer: Optional['EIRNode'] = None
    consumers: List['EIRNode'] = field(default_factory=list)

    def __post_init__(self):
        # Enforce canonical 4D shape
        if len(self.shape) != 4:
            self.shape = normalize_to_4d(self.shape)

    @property
    def is_constant(self) -> bool:
        """Returns True if this tensor holds static constant weights or biases."""
        return self.data is not None

    @property
    def num_elements(self) -> int:
        """Returns the total number of elements in the tensor."""
        return int(np.prod(self.shape))

    @property
    def size_bytes(self) -> int:
        """Returns total size in bytes based on shape and datatype."""
        item_size = DTYPE_BYTE_SIZES.get(self.dtype, 2)
        return self.num_elements * item_size

    @property
    def strides(self) -> Tuple[int, int, int, int]:
        """Calculates standard pitch-linear contiguous byte strides [stride_0..stride_3]."""
        item_size = DTYPE_BYTE_SIZES.get(self.dtype, 2)
        n, c, h, w = self.shape
        stride_3 = item_size                 # W stride
        stride_2 = w * stride_3              # H stride
        stride_1 = h * stride_2              # C stride
        stride_0 = c * stride_1              # N stride
        return (stride_0, stride_1, stride_2, stride_3)

    def __repr__(self) -> str:
        const_tag = " (const)" if self.is_constant else ""
        return f"<EIRTensor '{self.name}' shape={list(self.shape)} dtype={self.dtype}{const_tag}>"
