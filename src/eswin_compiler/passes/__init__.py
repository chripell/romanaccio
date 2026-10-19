"""EIR Transformation and Optimization Passes."""

from .base import GraphPass
from .gemm_to_conv1x1 import GemmToConv1x1Pass

__all__ = [
    "GraphPass",
    "GemmToConv1x1Pass",
]
