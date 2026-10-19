"""
eswin_runtime - Python bindings for Open ESWIN NPU Runtime
"""

from .runtime import (
    RuntimeContext,
    Model,
    Session,
    DmaBuffer,
    SamplerParams,
    ModelInfo,
    PerfStats,
    DeviceMode,
    CacheMode,
    version,
)

__version__ = version()
__all__ = [
    "RuntimeContext",
    "Model",
    "Session",
    "DmaBuffer",
    "SamplerParams",
    "ModelInfo",
    "PerfStats",
    "DeviceMode",
    "CacheMode",
    "version",
    "__version__",
]
