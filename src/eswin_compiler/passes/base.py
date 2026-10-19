"""Abstract base class for EIR transformation and optimization passes."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir.graph import EIRGraph


class GraphPass(ABC):
    """Abstract interface for optimization passes that transform an EIRGraph."""

    @abstractmethod
    def run(self, graph: 'EIRGraph') -> bool:
        """Executes the pass on the target graph.
        
        Returns:
            bool: True if the graph was modified, False otherwise.
        """
        pass
