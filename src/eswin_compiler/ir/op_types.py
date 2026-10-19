"""Operator enumerations and hardware engine targets for ESWIN IR (EIR)."""

from enum import Enum, auto


class EngineType(Enum):
    """Target execution engine for EIR nodes."""
    UNKNOWN = auto()
    NPU = auto()       # NVDLA hardware engine (CMAC, SDP, PDP, RUBIK)
    DSP = auto()       # Cadence Vision Q7 DSP vector core
    CPU = auto()       # Host CPU fallback


class OpType(Enum):
    """Canonical operator types supported in EIR computational graphs."""
    UNKNOWN = auto()
    # Dataflow boundaries
    INPUT = auto()

    OUTPUT = auto()
    CONSTANT = auto()

    # Core Tensor Contractions (mapped to NPU CMAC)
    CONV = auto()
    CONV_TRANSPOSE = auto()
    MATMUL = auto()
    GEMM = auto()

    # Element-wise Arithmetic (mapped to NPU SDP or DSP)
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()

    # Activations
    RELU = auto()
    LEAKY_RELU = auto()
    SIGMOID = auto()
    SILU = auto()        # SiLU / Swish: x * sigmoid(x)
    GELU = auto()
    TANH = auto()
    CLIP = auto()

    # Normalization Layers (typically mapped to DSP kernels)
    RMSNORM = auto()
    LAYERNORM = auto()
    BATCHNORM = auto()

    # Softmax & Attention (mapped to DSP or HAE)
    SOFTMAX = auto()
    ROPE = auto()               # Rotary Position Embedding
    KV_CACHE_UPDATE = auto()    # In-place KV cache store

    # Fused & Composite Operators
    SWIGLU = auto()             # Fused silu(gate) * up
    FUSED_SDP = auto()          # Hardware-fused BiasAdd + Scale + Activation

    # Tensor Rearrangement & Memory Operations (mapped to RUBIK or zero-copy)
    RESHAPE = auto()
    TRANSPOSE = auto()
    CONCAT = auto()
    SPLIT = auto()
    SLICE = auto()
    SQUEEZE = auto()
    UNSQUEEZE = auto()
