"""
Loader module for ESWIN .model FlatBuffers loadables.
"""

from eswin_simulator.loader.loadable_reader import LoadableModel, MemoryRegion, AddressEntry, TaskEntry, BlobEntry, RelocEntry, SubmitEntry
from eswin_simulator.loader.tensor_desc import TensorDescriptor

__all__ = [
    "LoadableModel",
    "MemoryRegion",
    "AddressEntry",
    "TaskEntry",
    "BlobEntry",
    "RelocEntry",
    "SubmitEntry",
    "TensorDescriptor",
]
