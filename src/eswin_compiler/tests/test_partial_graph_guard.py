"""Guard against silent single-op truncation (eswin_compiler Milestone 7 item 6).

The emitters consume exactly one CONV node, but `compile_safetensors_llm` hands
`_compile_single_layer` genuine multi-node graphs. Before the guard, a SwiGLU or attention
block produced a valid loadable containing one convolution and a success log naming it after
the block -- nothing failed, which is why it survived.

These tests lock in the measured node counts (so a change to the synthesiser is visible),
the rejection, the explicit opt-in, and -- importantly -- that single-operation graphs are
NOT caught by the guard.
"""

import logging
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compiler import CompilerConfig, EswinCompiler  # noqa: E402
from frontend.safetensors_importer import SafeTensorsImporter  # noqa: E402
from ir.graph import EIRGraph  # noqa: E402
from ir.node import EIRNode  # noqa: E402
from ir.op_types import EngineType, OpType  # noqa: E402
from ir.tensor import EIRTensor  # noqa: E402
from passes.gemm_to_conv1x1 import GemmToConv1x1Pass  # noqa: E402

# Post-lowering shapes measured on 2026-10-27. If the synthesiser changes, these fail and
# the guard's message must be re-read, not just these numbers.
MLP_COUNTS = {OpType.CONV: 3, OpType.MUL: 1}
ATTN_COUNTS = {OpType.CONV: 5, OpType.RMSNORM: 1, OpType.ROPE: 1, OpType.SOFTMAX: 1}


def lowered(graph):
    GemmToConv1x1Pass().run(graph)
    return graph


def counts(graph):
    return Counter(n.op_type for n in graph.nodes)


def mlp_graph():
    importer = SafeTensorsImporter()
    return lowered(importer.synthesize_mlp_block(
        layer_idx=0, hidden_size=576, intermediate_size=1536))


def attn_graph():
    importer = SafeTensorsImporter()
    return lowered(importer.synthesize_attention_block(
        layer_idx=0, hidden_size=576, num_heads=9, num_kv_heads=3, head_dim=64))


def compiler(**kw):
    return EswinCompiler(config=CompilerConfig(**kw))


class TestMeasuredNodeCounts(unittest.TestCase):
    """The evidence the guard is based on, asserted rather than commented."""

    def test_mlp_block_is_multi_op(self):
        c = counts(mlp_graph())
        self.assertEqual(c.get(OpType.CONV), 3,
                         f"SwiGLU MLP expected 3 CONV, got {dict(c)}")
        self.assertEqual(c.get(OpType.MUL), 1)

    def test_attention_block_is_multi_op(self):
        c = counts(attn_graph())
        self.assertEqual(c.get(OpType.CONV), 5,
                         f"attention expected 5 CONV, got {dict(c)}")
        for op in (OpType.RMSNORM, OpType.ROPE, OpType.SOFTMAX):
            self.assertEqual(c.get(op), 1, f"missing DSP-side op {op}")


class TestGuardRejectsPartialGraphs(unittest.TestCase):

    def test_mlp_rejected_by_default(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                compiler()._compile_single_layer(mlp_graph(), Path(tmp) / "m.model")
            msg = str(ctx.exception)
            self.assertIn("one operation per loadable", msg)
            # Must name what is lost, or the message is not actionable.
            self.assertIn("would be dropped", msg)
            self.assertIn("--allow-partial-graph", msg)

    def test_attention_rejected_by_default(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                compiler()._compile_single_layer(attn_graph(), Path(tmp) / "a.model")

    def test_no_file_written_on_rejection(self):
        """A rejected graph must not leave a partial artifact behind to be trusted later."""
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "mlp.model"
            with self.assertRaises(ValueError):
                compiler()._compile_single_layer(mlp_graph(), out)
            self.assertFalse(out.exists(), "guard fired but a loadable was written")


class TestOptInStillWorks(unittest.TestCase):
    """The escape hatch must work, and must be loud about what it produced."""

    def test_allow_partial_graph_emits_and_warns(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "mlp.model"
            with self.assertLogs("eswin_compiler", level=logging.WARNING) as captured:
                compiler(allow_partial_graph=True)._compile_single_layer(mlp_graph(), out)
            self.assertTrue(out.is_file(), "opt-in path should still emit")
            joined = " ".join(captured.output)
            self.assertIn("PARTIAL", joined)
            self.assertIn("dropped", joined)


class TestSingleOpGraphNotAffected(unittest.TestCase):
    """No false positives: the existing one-op paths must keep working untouched."""

    def test_single_conv_graph_compiles(self):
        """A genuine one-operation graph (what the LM-head path produces) is unaffected."""
        g = EIRGraph(name="one_op_graph")
        x = EIRTensor(name="input", shape=(1, 576, 1, 1), dtype="float16")
        g.add_input(x)
        w = EIRTensor(
            name="W_0",
            shape=(1536, 576, 1, 1),
            dtype="float16",
            data=np.zeros((1536, 576, 1, 1), dtype=np.float16),
        )
        g.add_tensor(w)
        y = EIRTensor(name="output", shape=(1, 1536, 1, 1), dtype="float16")
        g.add_output(y)
        g.add_node(EIRNode(
            name="gemm_0",
            op_type=OpType.CONV,
            inputs=[x, w],
            outputs=[y],
            attributes={"kernel_shape": [1, 1], "strides": [1, 1]},
            engine=EngineType.NPU,
        ))
        self.assertEqual(len(g.nodes), 1)
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "one.model"
            compiler()._compile_single_layer(g, out)     # must not raise
            self.assertTrue(out.is_file())

    def test_dynamic_weights_path_rejects_multi_op_too(self):
        """--dynamic-weights must not smuggle a partial graph past the guard."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                compiler(dynamic_weights=True)._compile_single_layer(
                    mlp_graph(), Path(tmp) / "dyn.model")


class TestCliFlag(unittest.TestCase):

    def test_flag_advertised(self):
        import subprocess
        res = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "compiler.py"),
             "--help"], capture_output=True, text=True, timeout=240)
        self.assertEqual(res.returncode, 0, res.stderr[-400:])
        self.assertIn("--allow-partial-graph", res.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
