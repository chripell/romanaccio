"""EIR Transformation and Optimization Passes."""

from .base import GraphPass
from .dsp_partitioner import DSPPartitionerPass, DSP_KERNEL_CATALOG
from .dual_die_splitter import DualDieSplitterPass
from .fusion_sdp import SDPFusionPass
from .gemm_to_conv1x1 import GemmToConv1x1Pass
from .sram_tiler import SRAMTilerPass

__all__ = [
    "GraphPass",
    "DSPPartitionerPass",
    "DSP_KERNEL_CATALOG",
    "DualDieSplitterPass",
    "GemmToConv1x1Pass",
    "SDPFusionPass",
    "SRAMTilerPass",
]
