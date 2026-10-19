"""Memory allocation and SPRAM (4MB on-chip SRAM) residency planning pass."""

from typing import List
from .base import GraphPass
try:
    from ..ir.graph import EIRGraph
    from ..ir.tensor import EIRTensor
except (ImportError, ValueError):
    from ir.graph import EIRGraph
    from ir.tensor import EIRTensor


# Default SPRAM capacity per die on EIC7700X
DEFAULT_SPRAM_CAPACITY = 4 * 1024 * 1024  # 4 MiB (4,194,304 bytes)


class SRAMTilerPass(GraphPass):
    """Plans on-chip 4MB SPRAM residency for high-throughput weights and activations.
    
    Tensors that fit within the on-chip SPRAM budget are assigned `memory_domain = 'SRAM'`.
    Tensors exceeding capacity remain in `SYSTEM` (DRAM) memory.
    """

    def __init__(self, sram_capacity: int = DEFAULT_SPRAM_CAPACITY):
        self.sram_capacity = sram_capacity

    def run(self, graph: EIRGraph) -> bool:
        modified = False
        remaining_sram = self.sram_capacity

        # Prioritize constant weights first, then intermediate buffers
        candidate_tensors: List[EIRTensor] = []

        # Sort tensors: weights first, then largest activations
        for t in graph.tensors.values():
            if t in graph.inputs or t in graph.outputs:
                continue  # External I/O stays in host-accessible SYSTEM DRAM
            candidate_tensors.append(t)

        candidate_tensors.sort(key=lambda t: (not t.is_constant, -t.size_bytes))

        for t in candidate_tensors:
            if t.size_bytes <= remaining_sram:
                if t.memory_domain != "SRAM":
                    t.memory_domain = "SRAM"
                    modified = True
                remaining_sram -= t.size_bytes
            else:
                if t.memory_domain != "SYSTEM":
                    t.memory_domain = "SYSTEM"
                    modified = True

        return modified
