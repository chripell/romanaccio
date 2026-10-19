"""ONNX Model Importer for ESWIN IR (EIR).

Converts standard ONNX model graphs into EIR computational graphs,
normalizing shapes to canonical 4D [N, C, H, W] and assigning initial hardware engines.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

from .base import ModelImporter
try:
    from ..ir.graph import EIRGraph
    from ..ir.node import EIRNode
    from ..ir.op_types import EngineType, OpType
    from ..ir.tensor import EIRTensor, normalize_to_4d
except (ImportError, ValueError):
    from ir.graph import EIRGraph
    from ir.node import EIRNode
    from ir.op_types import EngineType, OpType
    from ir.tensor import EIRTensor, normalize_to_4d


# ONNX data type mapping to EIR type string
ONNX_DTYPE_MAP = {
    TensorProto.FLOAT: "float32",
    TensorProto.FLOAT16: "float16",
    TensorProto.INT8: "int8",
    TensorProto.UINT8: "uint8",
    TensorProto.INT16: "int16",
    TensorProto.UINT16: "uint16",
    TensorProto.INT32: "int32",
    TensorProto.UINT32: "uint32",
    TensorProto.INT64: "int64",
    TensorProto.UINT64: "uint64",
}

# Operator type mapping to EIR OpType and default execution engine
OP_TYPE_MAP: Dict[str, Tuple[OpType, EngineType]] = {
    "Conv": (OpType.CONV, EngineType.NPU),
    "ConvTranspose": (OpType.CONV_TRANSPOSE, EngineType.NPU),
    "Gemm": (OpType.GEMM, EngineType.NPU),
    "MatMul": (OpType.MATMUL, EngineType.NPU),
    "Add": (OpType.ADD, EngineType.NPU),
    "Sub": (OpType.SUB, EngineType.NPU),
    "Mul": (OpType.MUL, EngineType.NPU),
    "Div": (OpType.DIV, EngineType.NPU),
    "Relu": (OpType.RELU, EngineType.NPU),
    "LeakyRelu": (OpType.LEAKY_RELU, EngineType.NPU),
    "Sigmoid": (OpType.SIGMOID, EngineType.NPU),
    "Clip": (OpType.CLIP, EngineType.NPU),
    "Softmax": (OpType.SOFTMAX, EngineType.DSP),
    "RMSNorm": (OpType.RMSNORM, EngineType.DSP),
    "SimplifiedLayerNormalization": (OpType.RMSNORM, EngineType.DSP),
    "LayerNormalization": (OpType.LAYERNORM, EngineType.DSP),
    "BatchNormalization": (OpType.BATCHNORM, EngineType.NPU),
    "Reshape": (OpType.RESHAPE, EngineType.NPU),
    "Transpose": (OpType.TRANSPOSE, EngineType.NPU),
    "Concat": (OpType.CONCAT, EngineType.NPU),
    "Split": (OpType.SPLIT, EngineType.NPU),
    "Slice": (OpType.SLICE, EngineType.NPU),
    "Squeeze": (OpType.SQUEEZE, EngineType.NPU),
    "Unsqueeze": (OpType.UNSQUEEZE, EngineType.NPU),
}


class ONNXImporter(ModelImporter):
    """Ingests ONNX protobuf models and lowers them into EIR."""

    def import_model(self, source: Union[str, Path, onnx.ModelProto]) -> EIRGraph:
        """Parses an ONNX model file or ModelProto and constructs an EIRGraph."""
        if isinstance(source, (str, Path)):
            model_proto = onnx.load(str(source))
        elif isinstance(source, onnx.ModelProto):
            model_proto = source
        else:
            raise TypeError(f"Expected file path or onnx.ModelProto, got {type(source)}")

        onnx_graph = model_proto.graph
        graph_name = onnx_graph.name or "onnx_model"
        eir_graph = EIRGraph(name=graph_name)

        # 1. Parse Initializers (Constant Weights & Biases)
        init_tensors: Dict[str, EIRTensor] = {}
        for init in onnx_graph.initializer:
            np_arr = numpy_helper.to_array(init)
            shape_4d = normalize_to_4d(np_arr.shape)
            dtype_str = ONNX_DTYPE_MAP.get(init.data_type, "float16")
            t = EIRTensor(
                name=init.name,
                shape=shape_4d,
                dtype=dtype_str,
                data=np_arr
            )
            init_tensors[init.name] = t
            eir_graph.add_tensor(t)

        # 2. Parse Value Infos (Tensor Shapes and Types)
        shape_map: Dict[str, Tuple[int, int, int, int]] = {}
        dtype_map: Dict[str, str] = {}

        for vi in list(onnx_graph.input) + list(onnx_graph.output) + list(onnx_graph.value_info):
            name = vi.name
            t_type = vi.type.tensor_type
            dtype_str = ONNX_DTYPE_MAP.get(t_type.elem_type, "float16")
            dtype_map[name] = dtype_str

            dims = []
            if t_type.HasField("shape"):
                for d in t_type.shape.dim:
                    if d.HasField("dim_value"):
                        dims.append(d.dim_value)
                    elif d.HasField("dim_param"):
                        # Dynamic batch or sequence length: default to 1 for static NPU compilation
                        dims.append(1)
                    else:
                        dims.append(1)
            shape_map[name] = normalize_to_4d(tuple(dims) if dims else (1, 1, 1, 1))

        # 3. Register Graph Inputs (excluding initializers)
        for inp in onnx_graph.input:
            if inp.name in init_tensors:
                continue
            shape = shape_map.get(inp.name, (1, 1, 1, 1))
            dtype = dtype_map.get(inp.name, "float16")
            t = EIRTensor(name=inp.name, shape=shape, dtype=dtype)
            eir_graph.add_input(t)

        # 4. Construct Nodes
        for idx, node in enumerate(onnx_graph.node):
            node_name = node.name or f"{node.op_type}_{idx}"
            op_info = OP_TYPE_MAP.get(node.op_type, (OpType.UNKNOWN, EngineType.CPU))
            eir_op_type, default_engine = op_info

            # Extract attributes
            attrs = {}
            for attr in node.attribute:
                val = onnx.helper.get_attribute_value(attr)
                attrs[attr.name] = val

            # Resolve input tensors
            node_inputs = []
            for inp_name in node.input:
                if not inp_name:
                    continue  # Optional input omitted in ONNX
                if inp_name in eir_graph.tensors:
                    node_inputs.append(eir_graph.tensors[inp_name])
                else:
                    shape = shape_map.get(inp_name, (1, 1, 1, 1))
                    dtype = dtype_map.get(inp_name, "float16")
                    t = EIRTensor(name=inp_name, shape=shape, dtype=dtype)
                    eir_graph.add_tensor(t)
                    node_inputs.append(t)

            # Resolve output tensors
            node_outputs = []
            for out_name in node.output:
                if out_name in eir_graph.tensors:
                    node_outputs.append(eir_graph.tensors[out_name])
                else:
                    shape = shape_map.get(out_name, (1, 1, 1, 1))
                    dtype = dtype_map.get(out_name, "float16")
                    t = EIRTensor(name=out_name, shape=shape, dtype=dtype)
                    eir_graph.add_tensor(t)
                    node_outputs.append(t)

            eir_node = EIRNode(
                name=node_name,
                op_type=eir_op_type,
                inputs=node_inputs,
                outputs=node_outputs,
                attributes=attrs,
                engine=default_engine
            )
            eir_graph.add_node(eir_node)

        # 5. Register Graph Outputs
        for out in onnx_graph.output:
            if out.name in eir_graph.tensors:
                eir_graph.add_output(eir_graph.tensors[out.name])
            else:
                shape = shape_map.get(out.name, (1, 1, 1, 1))
                dtype = dtype_map.get(out.name, "float16")
                t = EIRTensor(name=out.name, shape=shape, dtype=dtype)
                eir_graph.add_output(t)

        return eir_graph
