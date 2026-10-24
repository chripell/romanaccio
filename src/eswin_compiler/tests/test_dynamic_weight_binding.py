"""Tests for runtime-bound (Class B) weight emission — eswin_compiler Milestone 7.

The contract under test is the one validated on hardware (FINDINGS.md §27): a loadable whose
weight memory entry is Class B reads its weights from a buffer supplied at submit time, and
is byte-identical to the same loadable with the weights baked into blobs. These tests pin the
*emitted structures* so a refactor cannot silently break that contract off-board; the board
bit-match itself is run by `src/eswin_runtime/tools/probe_weight_bitmatch.cpp`.

Field values asserted here are not arbitrary — each was confirmed against a real vendor
loadable and then proven to execute correctly on `roma`:
  flags 3 -> 5, contents/offsets -> [], bind_id 0 -> 1, tensor_desc_id 0 -> 2,
  a data_category=1 (WEIGHT) tensor descriptor, the WsplitM_* blobs dropped,
  and NO relocs for the weight.
"""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.flatbuffer_emitter import ModelReader, ModelWriter  # noqa: E402
from backend.gemm_builder import NVDLAGEMMBuilder  # noqa: E402

C_OUT, C_IN = 1536, 576
TOTAL_W_SIZE = 1769472       # 3 slices x 589824
WEIGHT_MEM_ID = 1
WEIGHT_BIND_ID = 1
WEIGHT_TENSOR_DESC_ID = 2
DATA_CATEGORY_WEIGHT = 1


def build(dynamic: bool):
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((C_OUT, C_IN)) * 0.05).astype(np.float32)
    return NVDLAGEMMBuilder.build_linear_3pass(
        w, "input", "output", "test_model", dynamic_weight=dynamic
    )


def weight_mem(model):
    return next(e for e in model.memory_list if e.id == WEIGHT_MEM_ID)


class TestDynamicWeightEmission(unittest.TestCase):

    def test_static_path_unchanged(self):
        """The default must stay Class A, or every existing loadable changes."""
        m = build(dynamic=False)
        wm = weight_mem(m)
        self.assertEqual(wm.flags, 3, "static weight entry should keep flags=3")
        self.assertEqual(len(wm.contents), 3, "static weight entry should reference 3 blobs")
        self.assertEqual(wm.bind_id, 0, "static weight entry should not claim a bind slot")
        self.assertEqual(wm.tensor_desc_id, 0)
        self.assertTrue(any("WsplitM" in b.name for b in m.blobs),
                        "static model must carry weight blobs")
        self.assertFalse(any(t.data_category == DATA_CATEGORY_WEIGHT for t in m.tensor_descs),
                         "static model must not expose a WEIGHT input descriptor")
        self.assertEqual(len(m.tensor_descs), 2, "static model has activation in + out only")

    def test_dynamic_weight_is_class_b(self):
        """Class B means: no blob data, flags=5, a bind slot, and a tensor descriptor."""
        m = build(dynamic=True)
        wm = weight_mem(m)
        self.assertEqual(wm.contents, [], "dynamic weight must carry no blob contents")
        self.assertEqual(wm.offsets, [], "dynamic weight must carry no blob offsets")
        self.assertEqual(wm.flags, 5, "dynamic weight must use the Class B input flags")
        self.assertEqual(wm.bind_id, WEIGHT_BIND_ID,
                         "weight must occupy input slot 1 (activation keeps slot 0)")
        self.assertEqual(wm.tensor_desc_id, WEIGHT_TENSOR_DESC_ID)
        self.assertEqual(wm.size, TOTAL_W_SIZE,
                         "size must still be declared so the caller knows how much to bind")

        self.assertFalse(any("WsplitM" in b.name for b in m.blobs),
                         "dynamic model must not embed weight blobs")

        wd = [t for t in m.tensor_descs if t.data_category == DATA_CATEGORY_WEIGHT]
        self.assertEqual(len(wd), 1, "exactly one WEIGHT descriptor expected")
        self.assertEqual(wd[0].mem_id, WEIGHT_MEM_ID)
        self.assertEqual(wd[0].size, TOTAL_W_SIZE)
        self.assertEqual((wd[0].c, wd[0].w), (C_OUT, C_IN), "descriptor shape mismatch")

    def test_dynamic_weight_adds_no_relocs(self):
        """Measured on hardware: extra weight relocs change nothing, so emit none."""
        static = build(dynamic=False)
        dynamic = build(dynamic=True)
        self.assertEqual(len(dynamic.relocs), len(static.relocs),
                         "dynamic weights must not add relocs")
        self.assertFalse(any(r.address_id == WEIGHT_MEM_ID for r in dynamic.relocs),
                         "no reloc should target the weight address")
        # The activation I/O relocs must survive untouched.
        self.assertTrue(any(r.address_id == 4 for r in dynamic.relocs), "input reloc lost")
        self.assertTrue(any(r.address_id == 5 for r in dynamic.relocs), "output reloc lost")

    def test_dynamic_model_is_much_smaller(self):
        """Dropping the weight blobs is the whole point: 1716 loadables become 27."""
        s = len(ModelWriter.serialize(build(dynamic=False)))
        d = len(ModelWriter.serialize(build(dynamic=True)))
        self.assertLess(d, s // 10,
                        f"dynamic model should be far smaller ({d} vs {s} bytes)")

    def test_dynamic_survives_serialization_roundtrip(self):
        """ModelWriter must not drop the empty contents or the bind_id."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dyn.model"
            ModelWriter.write_file(build(dynamic=True), path)
            rt = ModelReader.read_file(path)

        wm = weight_mem(rt)
        self.assertEqual(wm.contents, [])
        self.assertEqual(wm.flags, 5)
        self.assertEqual(wm.bind_id, WEIGHT_BIND_ID)
        self.assertEqual(wm.tensor_desc_id, WEIGHT_TENSOR_DESC_ID)
        self.assertTrue(any(t.data_category == DATA_CATEGORY_WEIGHT for t in rt.tensor_descs),
                        "WEIGHT descriptor lost in round-trip")
        self.assertFalse(any("WsplitM" in b.name for b in rt.blobs))

    def test_both_paths_share_geometry(self):
        """Only the weight *binding* may differ, not the graph."""
        s, d = build(dynamic=False), build(dynamic=True)
        self.assertEqual(len(s.tasks), len(d.tasks))
        self.assertEqual(len(s.memory_list), len(d.memory_list))
        self.assertEqual([m.size for m in s.memory_list], [m.size for m in d.memory_list])
        self.assertEqual(len(s.address_list), len(d.address_list))
        # The weight address is still referenced by the task, so the surface chain is intact.
        self.assertIn(WEIGHT_MEM_ID, s.tasks[0].address_list)
        self.assertIn(WEIGHT_MEM_ID, d.tasks[0].address_list)


class TestCommandLineFlag(unittest.TestCase):

    def test_flag_is_registered(self):
        """`--dynamic-weights` must appear in the CLI help."""
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(
            [sys.executable, str(root / "compiler.py"), "--help"],
            capture_output=True, text=True, cwd=str(root), timeout=120,
        )
        self.assertEqual(out.returncode, 0, f"--help failed: {out.stderr[-500:]}")
        self.assertIn("--dynamic-weights", out.stdout)
        self.assertIn("runtime-bound", out.stdout)

    def test_config_defaults_to_false(self):
        """Opt-in only: the default must not change existing output."""
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        from compiler import CompilerConfig
        self.assertFalse(CompilerConfig().dynamic_weights)
        self.assertTrue(CompilerConfig(dynamic_weights=True).dynamic_weights)


if __name__ == "__main__":
    unittest.main(verbosity=2)
