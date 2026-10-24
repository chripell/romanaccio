"""Tests for the generalised Class B weight converter (SwiGLU expert-FFN archetype).

Runs against the real vendor-compiled SwiGLU block when it is available, since that is the
loadable whose structure the converter was written against:

    deep/artifacts/test_swiglu_mlp_npu_b1.model

Set ESWIN_SWIGLU_MODEL to point elsewhere. When the file is absent the vendor-model tests
skip, and the synthetic tests (built by our own gemm_builder) still run.

The key structural fact these tests pin: EsAAC packs a whole FFN's gate/up/down weights into
ONE Class A memory region (mem_id=1, 5,308,416 bytes = 3 x 1,769,472, nine blobs), so an
expert FFN needs one runtime-bound tensor and one bind_id - not three.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.dynamic_weight_binding import (  # noqa: E402
    BoundWeight,
    extract_weight_blob,
    find_weight_regions,
    is_weight_blob,
    make_weights_runtime_bound,
    next_bind_id,
)
from backend.flatbuffer_emitter import ModelReader, ModelWriter  # noqa: E402
from backend.gemm_builder import NVDLAGEMMBuilder  # noqa: E402

SWIGLU_MODEL = Path(os.environ.get(
    "ESWIN_SWIGLU_MODEL",
    "/home/chri/aixigua/deep/artifacts/test_swiglu_mlp_npu_b1.model",
))
HAVE_SWIGLU = SWIGLU_MODEL.is_file()

# Observed geometry of the vendor SwiGLU block.
SWIGLU_WEIGHT_MEM_ID = 1
SWIGLU_WEIGHT_SIZE = 5308416        # 3 x 1,769,472 (gate + up + down)
SWIGLU_WEIGHT_BLOBS = 9
SWIGLU_TASK_BLOBS = 11
SWIGLU_TOTAL_BLOBS = 20


def synthetic_gemm():
    rng = np.random.default_rng(11)
    w = (rng.standard_normal((1536, 576)) * 0.05).astype(np.float32)
    return NVDLAGEMMBuilder.build_linear_3pass(w, "input", "output", "synthetic")


class TestBlobClassification(unittest.TestCase):
    """The heuristic must never classify a task stream or intermediate as a weight."""

    def test_weight_names_recognised(self):
        for n in ("WsplitM_0_0", "w_gatesplitM_1_1", "w_upsplitM_0_3",
                  "w_downsplitM_2_8", "_block_mlp_gate_proj_MatMul_transpose_weightsplitM_0_10"):
            self.assertTrue(is_weight_blob(n), f"should be a weight: {n}")

    def test_task_and_intermediate_names_rejected(self):
        for n in ("kmd-task-surf_list", "umd-task-op_list", "kmd-task-dep_graph",
                  "custom_name_0_cvtPrecisionsplitM_0", "conv_gatesplitM_0.sram.edma",
                  "custom_name_0_cvtPrecisiontiling_M_Concat", "mul_silu",
                  "_block_self_attn_q_proj_MatMul_biasaddsplitM_0_BIAS"):
            self.assertFalse(is_weight_blob(n), f"must NOT be treated as a weight: {n}")

    def test_task_streams_are_never_droppable(self):
        """Guard the invariant that matters most: losing a command stream bricks the model."""
        self.assertFalse(is_weight_blob("kmd-task-op_list"))
        self.assertFalse(is_weight_blob("umd-task-surf_list"))


@unittest.skipUnless(HAVE_SWIGLU, f"vendor SwiGLU model not present at {SWIGLU_MODEL}")
class TestVendorSwiGLU(unittest.TestCase):

    def setUp(self):
        self.model = ModelReader.read_file(str(SWIGLU_MODEL))

    def test_all_ffn_weights_share_one_region(self):
        """gate + up + down are packed into a single Class A region, not three."""
        regions = find_weight_regions(self.model)
        self.assertEqual(len(regions), 1, "expected exactly one weight region")
        r = regions[0]
        self.assertEqual(r.id, SWIGLU_WEIGHT_MEM_ID)
        self.assertEqual(r.size, SWIGLU_WEIGHT_SIZE)
        self.assertEqual(len(r.contents), SWIGLU_WEIGHT_BLOBS)
        self.assertEqual(r.domain, 0, "weights live in DRAM, not SPRAM")
        names = " ".join(r.contents)
        for proj in ("w_gatesplitM", "w_upsplitM", "w_downsplitM"):
            self.assertIn(proj, names, f"{proj} should be in the same region")

    def test_bind_id_allocator_skips_the_activation_slot(self):
        """Activation input already holds bind_id 0, so the weight must take 1."""
        self.assertEqual(next_bind_id(self.model), 1)

    def test_conversion_produces_class_b(self):
        bound = make_weights_runtime_bound(self.model)
        self.assertEqual(len(bound), 1)
        b = bound[0]
        self.assertIsInstance(b, BoundWeight)
        self.assertEqual((b.mem_id, b.bind_id, b.size), (1, 1, SWIGLU_WEIGHT_SIZE))
        self.assertEqual(len(b.dropped_blobs), SWIGLU_WEIGHT_BLOBS)

        e = next(x for x in self.model.memory_list if x.id == 1)
        self.assertEqual(e.contents, [])
        self.assertEqual(e.offsets, [])
        self.assertEqual(e.flags, 5, "Class B input flags")
        self.assertEqual(e.bind_id, 1)
        self.assertEqual(e.tensor_desc_id, b.tensor_desc_id)

    def test_only_weight_blobs_are_dropped(self):
        make_weights_runtime_bound(self.model)
        names = [b.name for b in self.model.blobs]
        self.assertEqual(len(self.model.blobs), SWIGLU_TOTAL_BLOBS - SWIGLU_WEIGHT_BLOBS)
        self.assertFalse(any("splitM" in n and n.startswith(("w_", "W")) for n in names),
                         "a weight blob survived")
        task_streams = [n for n in names if "task" in n]
        self.assertEqual(len(task_streams), SWIGLU_TASK_BLOBS,
                         "a KMD/UMD command stream was dropped - that would brick the model")

    def test_weight_descriptor_added_and_activations_untouched(self):
        make_weights_runtime_bound(self.model)
        cats = sorted(t.data_category for t in self.model.tensor_descs)
        self.assertEqual(cats, [1, 2, 2], "expected one WEIGHT and two FEATURE descriptors")
        wd = next(t for t in self.model.tensor_descs if t.data_category == 1)
        self.assertEqual(wd.mem_id, SWIGLU_WEIGHT_MEM_ID)
        self.assertEqual(wd.size, SWIGLU_WEIGHT_SIZE)

    def test_no_weight_relocs_and_activation_relocs_kept(self):
        before = len(self.model.relocs)
        make_weights_runtime_bound(self.model)
        self.assertEqual(len(self.model.relocs), before, "conversion must not add relocs")
        targets = {r.address_id for r in self.model.relocs}
        self.assertNotIn(SWIGLU_WEIGHT_MEM_ID, targets)
        self.assertIn(4, targets, "input activation reloc lost")
        self.assertIn(5, targets, "output activation reloc lost")

    def test_weight_bytes_are_extractable_for_bitmatch(self):
        """The submit-time buffer must be reconstructible from the original blobs."""
        region = find_weight_regions(self.model)[0]
        data = extract_weight_blob(self.model, region)
        self.assertEqual(len(data), SWIGLU_WEIGHT_SIZE)
        self.assertNotEqual(data, b"\x00" * SWIGLU_WEIGHT_SIZE, "extracted all zeros")
        # Each of the nine blobs must appear at its declared offset.
        by_name = {b.name: bytes(b.data or b"") for b in self.model.blobs}
        for off, name in zip(region.offsets, region.contents):
            self.assertEqual(data[off:off + len(by_name[name])], by_name[name],
                             f"blob {name} not at declared offset {off}")

    def test_roundtrip_and_size_reduction(self):
        make_weights_runtime_bound(self.model)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "swiglu_dynamic.model"
            ModelWriter.write_file(self.model, out)
            rt = ModelReader.read_file(out)
            e = next(x for x in rt.memory_list if x.id == SWIGLU_WEIGHT_MEM_ID)
            self.assertEqual(e.contents, [])
            self.assertEqual(e.flags, 5)
            self.assertEqual(e.bind_id, 1)
            self.assertTrue(any(t.data_category == 1 for t in rt.tensor_descs),
                            "WEIGHT descriptor lost in round-trip")
            original = SWIGLU_MODEL.stat().st_size
            self.assertLess(out.stat().st_size, original // 10,
                            "dynamic model should be far smaller than the static one")


class TestSyntheticGemm(unittest.TestCase):
    """The converter must also work on our own emitter's output, not just vendor models."""

    def test_single_weight_region_converted(self):
        m = synthetic_gemm()
        regions = find_weight_regions(m)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].size, 1769472)

        bound = make_weights_runtime_bound(m)
        self.assertEqual(len(bound), 1)
        self.assertEqual(bound[0].bind_id, 1)
        e = next(x for x in m.memory_list if x.id == bound[0].mem_id)
        self.assertEqual((e.contents, e.flags, e.bind_id), ([], 5, 1))
        self.assertTrue(any(t.data_category == 1 for t in m.tensor_descs))

    def test_idempotent_second_call_is_a_noop(self):
        """A converted region is already Class B, so re-running must not double-bind."""
        m = synthetic_gemm()
        make_weights_runtime_bound(m)
        self.assertEqual(make_weights_runtime_bound(m), [])
        self.assertEqual(sum(1 for t in m.tensor_descs if t.data_category == 1), 1)

    def test_explicit_mem_ids_override_the_heuristic(self):
        m = synthetic_gemm()
        self.assertEqual(find_weight_regions(m, mem_ids=[99]), [])
        bound = make_weights_runtime_bound(m, mem_ids=[99])
        self.assertEqual(bound, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
