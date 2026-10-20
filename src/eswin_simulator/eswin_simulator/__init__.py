"""
eswin_simulator - Open-Source Cross-Platform Functional Simulator for ESWIN EIC7700X NPU/DSP Loadables.
"""

from eswin_simulator.loader.loadable_reader import LoadableModel
from eswin_simulator.loader.tensor_desc import TensorDescriptor
from eswin_simulator.memory.address_space import VirtualAddressSpace

__version__ = "0.1.0"
__all__ = ["LoadableModel", "TensorDescriptor", "VirtualAddressSpace"]
