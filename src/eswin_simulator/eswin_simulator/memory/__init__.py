"""
Memory management and virtual address space for ESWIN simulator.
"""

from eswin_simulator.memory.address_space import (
    VirtualAddressSpace,
    AllocatedBuffer,
    SPRAM_CAPACITY_BYTES,
    DRAM_BASE_ADDRESS,
    SPRAM_BASE_ADDRESS,
)
from eswin_simulator.memory.buffer_view import create_tensor_view

__all__ = [
    "VirtualAddressSpace",
    "AllocatedBuffer",
    "SPRAM_CAPACITY_BYTES",
    "DRAM_BASE_ADDRESS",
    "SPRAM_BASE_ADDRESS",
    "create_tensor_view",
]
