"""
Abstract base class for all ESWIN EIC7700X hardware simulation engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from eswin_simulator.memory.address_space import VirtualAddressSpace


class HardwareEngine(ABC):
    """Abstract base class representing an on-chip hardware accelerator sub-engine."""

    def __init__(self, name: str, address_space: VirtualAddressSpace):
        self.name = name
        self.vas = address_space
        self.total_operations: int = 0
        self.total_bytes_transferred: int = 0

    @abstractmethod
    def reset(self) -> None:
        """Resets engine state and counters."""
        self.total_operations = 0
        self.total_bytes_transferred = 0

    def stats(self) -> Dict[str, Any]:
        """Returns engine execution statistics."""
        return {
            "engine": self.name,
            "total_operations": self.total_operations,
            "total_bytes_transferred": self.total_bytes_transferred,
        }
