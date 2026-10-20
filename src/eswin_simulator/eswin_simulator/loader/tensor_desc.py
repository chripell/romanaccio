"""
Tensor descriptor data structure and metadata parsing.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


DATA_FORMAT_NAMES = {
    0: "UNKNOWN",
    1: "NCHW",
    2: "NHWC",
}

DATA_TYPE_NAMES = {
    0: "UNKNOWN",
    1: "FLOAT",
    2: "HALF",
    3: "INT16",
    4: "INT8",
}

DATA_TYPE_NUMPY = {
    0: np.dtype("uint8"),
    1: np.dtype("float32"),
    2: np.dtype("float16"),
    3: np.dtype("int16"),
    4: np.dtype("int8"),
}

DATA_TYPE_SIZES = {
    0: 1,
    1: 4,
    2: 2,
    3: 2,
    4: 1,
}

DATA_CATEGORY_NAMES = {
    0: "IMAGE",
    1: "WEIGHT",
    2: "FEATURE",
    3: "PLANAR",
    4: "BIAS",
}


@dataclass
class TensorDescriptor:
    """Represents a tensor descriptor parsed from an ESWIN .model loadable."""

    name: str
    id: int
    mem_id: int
    size: int
    offset: int
    data_format_id: int
    data_type_id: int
    data_category_id: int
    pixel_format_id: int
    pixel_mapping_id: int
    n: int
    c: int
    h: int
    w: int
    strides: Tuple[int, ...]
    is_input: bool = False
    is_output: bool = False
    memory_domain: str = "SYSTEM"

    @property
    def data_format(self) -> str:
        """Returns human-readable data format (e.g. NCHW, NHWC)."""
        return DATA_FORMAT_NAMES.get(self.data_format_id, f"UNKNOWN({self.data_format_id})")

    @property
    def data_type(self) -> str:
        """Returns human-readable data type (e.g. HALF, INT8)."""
        return DATA_TYPE_NAMES.get(self.data_type_id, f"UNKNOWN({self.data_type_id})")

    @property
    def data_category(self) -> str:
        """Returns human-readable data category (e.g. FEATURE, WEIGHT)."""
        return DATA_CATEGORY_NAMES.get(self.data_category_id, f"UNKNOWN({self.data_category_id})")

    @property
    def dtype(self) -> np.dtype:
        """Returns equivalent NumPy data type."""
        return DATA_TYPE_NUMPY.get(self.data_type_id, np.dtype("uint8"))

    @property
    def itemsize(self) -> int:
        """Bytes per element."""
        return DATA_TYPE_SIZES.get(self.data_type_id, 1)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """4D Tensor shape (N, C, H, W)."""
        return (self.n, self.c, self.h, self.w)

    @property
    def num_elements(self) -> int:
        """Total number of elements in tensor."""
        return max(1, self.n) * max(1, self.c) * max(1, self.h) * max(1, self.w)

    def summary(self) -> str:
        """Returns one-line summary string of tensor."""
        direction = "INPUT" if self.is_input else ("OUTPUT" if self.is_output else "INTERNAL")
        return (
            f"Tensor(name='{self.name}', {direction}, shape={self.shape}, "
            f"type={self.data_type}, format={self.data_format}, size={self.size}B, mem_id={self.mem_id})"
        )
