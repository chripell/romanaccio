"""Operation node representation for ESWIN IR (EIR)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .op_types import EngineType, OpType
from .tensor import EIRTensor


@dataclass
class EIRNode:
    """Represents an operation in the computational DAG.
    
    Attributes:
        name: Unique node name.
        op_type: Canonical EIR OpType.
        inputs: List of input EIRTensor instances.
        outputs: List of output EIRTensor instances.
        attributes: Dictionary of operator-specific attributes (strides, pads, epsilon, etc.).
        engine: Assigned execution engine (NPU, DSP, CPU).
        dsp_kernel_name: Target DSP shared library name if engine == EngineType.DSP.
    """
    name: str
    op_type: OpType
    inputs: List[EIRTensor] = field(default_factory=list)
    outputs: List[EIRTensor] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    engine: EngineType = EngineType.UNKNOWN
    dsp_kernel_name: Optional[str] = None

    def __post_init__(self):
        # Establish linkages
        for t in self.inputs:
            if self not in t.consumers:
                t.consumers.append(self)
        for t in self.outputs:
            t.producer = self

    def add_input(self, tensor: EIRTensor) -> None:
        """Adds an input tensor and updates consumer references."""
        self.inputs.append(tensor)
        if self not in tensor.consumers:
            tensor.consumers.append(self)

    def add_output(self, tensor: EIRTensor) -> None:
        """Adds an output tensor and sets its producer reference."""
        self.outputs.append(tensor)
        tensor.producer = self

    def replace_input(self, old_tensor: EIRTensor, new_tensor: EIRTensor) -> None:
        """Replaces all occurrences of old_tensor with new_tensor."""
        for i, t in enumerate(self.inputs):
            if t.name == old_tensor.name:
                self.inputs[i] = new_tensor
        if self in old_tensor.consumers and all(inp.name != old_tensor.name for inp in self.inputs):
            old_tensor.consumers.remove(self)
        if self not in new_tensor.consumers:
            new_tensor.consumers.append(self)

    def __repr__(self) -> str:
        in_names = [t.name for t in self.inputs]
        out_names = [t.name for t in self.outputs]
        return f"<EIRNode '{self.name}' op={self.op_type.name} inputs={in_names} outputs={out_names} engine={self.engine.name}>"
