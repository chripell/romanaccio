"""Abstract base class for model importers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union
try:
    from ..ir.graph import EIRGraph
except (ImportError, ValueError):
    from ir.graph import EIRGraph



class ModelImporter(ABC):
    """Abstract interface for importing neural network models into EIR."""

    @abstractmethod
    def import_model(self, source: Union[str, Path, Any]) -> EIRGraph:
        """Parses the input model artifact and returns an EIRGraph instance."""
        pass
