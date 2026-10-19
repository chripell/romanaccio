"""Optimization pass fusing Bias Addition and Non-linear Activations into the SDP engine."""

from typing import List, Optional

from .base import GraphPass
try:
    from ..ir.graph import EIRGraph
    from ..ir.node import EIRNode
    from ..ir.op_types import EngineType, OpType
    from ..ir.tensor import EIRTensor
except (ImportError, ValueError):
    from ir.graph import EIRGraph
    from ir.node import EIRNode
    from ir.op_types import EngineType, OpType
    from ir.tensor import EIRTensor


FUSIBLE_ACTIVATIONS = {
    OpType.RELU: "relu",
    OpType.SIGMOID: "sigmoid",
    OpType.SILU: "silu",
    OpType.CLIP: "clip",
}


class SDPFusionPass(GraphPass):
    """Fuses BiasAdd and Activation nodes into the upstream CONV / SDP hardware pipeline."""

    def run(self, graph: EIRGraph) -> bool:
        modified = False

        # 1. Fuse Conv + BiasAdd
        for node in list(graph.nodes):
            if node.op_type == OpType.CONV:
                if self._fuse_bias(graph, node):
                    modified = True

        # 2. Fuse Conv + Activation
        for node in list(graph.nodes):
            if node.op_type == OpType.CONV:
                if self._fuse_activation(graph, node):
                    modified = True

        if modified:
            graph.validate()

        return modified

    def _fuse_bias(self, graph: EIRGraph, conv_node: EIRNode) -> bool:
        """Fuses downstream ADD node as a bias input to the CONV node."""
        if len(conv_node.outputs) != 1:
            return False

        conv_out = conv_node.outputs[0]
        # Check if the sole consumer of conv_out is an ADD node
        if len(conv_out.consumers) != 1:
            return False

        consumer = conv_out.consumers[0]
        if consumer.op_type != OpType.ADD:
            return False

        # Identify which input to ADD is the constant bias
        bias_tensor: Optional[EIRTensor] = None
        for inp in consumer.inputs:
            if inp.name != conv_out.name and inp.is_constant:
                bias_tensor = inp
                break

        if bias_tensor is None:
            return False

        # Attach bias to CONV node
        conv_node.add_input(bias_tensor)
        conv_node.attributes["has_bias"] = True

        # Reroute output of ADD to be output of CONV
        add_out = consumer.outputs[0]
        conv_node.outputs = [add_out]
        add_out.producer = conv_node

        # Remove ADD node from graph
        graph.nodes.remove(consumer)
        if conv_out.name in graph.tensors:
            del graph.tensors[conv_out.name]

        return True

    def _fuse_activation(self, graph: EIRGraph, conv_node: EIRNode) -> bool:
        """Fuses downstream activation (ReLU, Sigmoid, SiLU, Clip) into the CONV SDP engine."""
        if len(conv_node.outputs) != 1:
            return False

        conv_out = conv_node.outputs[0]
        if len(conv_out.consumers) != 1:
            return False

        consumer = conv_out.consumers[0]
        act_name = FUSIBLE_ACTIVATIONS.get(consumer.op_type)
        if act_name is None:
            return False

        # Record fused activation in CONV node attributes
        conv_node.attributes["activation"] = act_name
        conv_node.attributes["fused_sdp"] = True
        if act_name == "clip":
            conv_node.attributes["clip_min"] = consumer.attributes.get("min", 0.0)
            conv_node.attributes["clip_max"] = consumer.attributes.get("max", 6.0)

        # Reroute activation output to CONV
        act_out = consumer.outputs[0]
        conv_node.outputs = [act_out]
        act_out.producer = conv_node

        # Remove activation node from graph
        graph.nodes.remove(consumer)
        if conv_out.name in graph.tensors:
            del graph.tensors[conv_out.name]

        return True
