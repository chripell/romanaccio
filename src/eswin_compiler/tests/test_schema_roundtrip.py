"""Tests for FlatBuffers schema round-trip serialization and deserialization.

Verifies that vendor .model files can be parsed into high-level dataclasses
and re-serialized into valid .model binaries with 100% semantic parity.
"""

from pathlib import Path
import sys
import unittest

# Setup python paths
_TESTS_DIR = Path(__file__).parent.resolve()
_COMPILER_DIR = _TESTS_DIR.parent
_SCHEMA_GEN_DIR = _COMPILER_DIR / "schema" / "generated"

for p in (str(_COMPILER_DIR), str(_SCHEMA_GEN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.flatbuffer_emitter import (
    ModelReader,
    ModelWriter,
    LoadableModel,
    ModelVersion,
)


class TestSchemaRoundTrip(unittest.TestCase):
    """Test suite for .model FlatBuffers deserialization and serialization."""

    @classmethod
    def setUpClass(cls):
        # Locate artifacts directory
        cls.deep_dir = Path("/home/chri/aixigua/deep")
        cls.artifacts_dir = cls.deep_dir / "artifacts"
        cls.conv_model_path = cls.artifacts_dir / "test_conv_npu_b1.model"
        cls.swiglu_model_path = cls.artifacts_dir / "test_swiglu_mlp_npu_b1.model"
        cls.smollm_model_path = cls.deep_dir / "models" / "smollm" / "smollm_lm_head_d0_npu_b1.model"

    def _assert_models_equal(self, m1: LoadableModel, m2: LoadableModel, name: str = ""):
        """Helper to assert deep semantic equivalence between two LoadableModels."""
        # 1. Version
        self.assertEqual(m1.version.major, m2.version.major, f"{name}: version.major mismatch")
        self.assertEqual(m1.version.minor, m2.version.minor, f"{name}: version.minor mismatch")
        self.assertEqual(m1.version.sub_minor, m2.version.sub_minor, f"{name}: version.sub_minor mismatch")

        # 2. Tasks
        self.assertEqual(len(m1.tasks), len(m2.tasks), f"{name}: task count mismatch")
        for i, (t1, t2) in enumerate(zip(m1.tasks, m2.tasks)):
            self.assertEqual(t1.id, t2.id, f"{name}: task[{i}].id mismatch")
            self.assertEqual(t1.kmd_id, t2.kmd_id, f"{name}: task[{i}].kmd_id mismatch")
            self.assertEqual(t1.interface, t2.interface, f"{name}: task[{i}].interface mismatch")
            self.assertEqual(t1.address_list, t2.address_list, f"{name}: task[{i}].address_list mismatch")

        # 3. Memory List
        self.assertEqual(len(m1.memory_list), len(m2.memory_list), f"{name}: memory_list count mismatch")
        for i, (mem1, mem2) in enumerate(zip(m1.memory_list, m2.memory_list)):
            self.assertEqual(mem1.id, mem2.id, f"{name}: mem[{i}].id mismatch")
            self.assertEqual(mem1.domain, mem2.domain, f"{name}: mem[{i}].domain mismatch")
            self.assertEqual(mem1.flags, mem2.flags, f"{name}: mem[{i}].flags mismatch")
            self.assertEqual(mem1.size, mem2.size, f"{name}: mem[{i}].size mismatch")
            self.assertEqual(mem1.alignment, mem2.alignment, f"{name}: mem[{i}].alignment mismatch")
            self.assertEqual(mem1.contents, mem2.contents, f"{name}: mem[{i}].contents mismatch")
            self.assertEqual(mem1.offsets, mem2.offsets, f"{name}: mem[{i}].offsets mismatch")
            self.assertEqual(mem1.bind_id, mem2.bind_id, f"{name}: mem[{i}].bind_id mismatch")
            self.assertEqual(mem1.tensor_desc_id, mem2.tensor_desc_id, f"{name}: mem[{i}].tensor_desc_id mismatch")

        # 4. Address List
        self.assertEqual(len(m1.address_list), len(m2.address_list), f"{name}: address_list count mismatch")
        for i, (a1, a2) in enumerate(zip(m1.address_list, m2.address_list)):
            self.assertEqual(a1.id, a2.id, f"{name}: addr[{i}].id mismatch")
            self.assertEqual(a1.mem_id, a2.mem_id, f"{name}: addr[{i}].mem_id mismatch")
            self.assertEqual(a1.offset, a2.offset, f"{name}: addr[{i}].offset mismatch")
            self.assertEqual(a1.size, a2.size, f"{name}: addr[{i}].size mismatch")

        # 5. Event List
        self.assertEqual(len(m1.event_list), len(m2.event_list), f"{name}: event_list count mismatch")
        for i, (e1, e2) in enumerate(zip(m1.event_list, m2.event_list)):
            self.assertEqual(e1.id, e2.id, f"{name}: event[{i}].id mismatch")
            self.assertEqual(e1.type, e2.type, f"{name}: event[{i}].type mismatch")
            self.assertEqual(e1.target, e2.target, f"{name}: event[{i}].target mismatch")
            self.assertEqual(e1.val, e2.val, f"{name}: event[{i}].val mismatch")
            self.assertEqual(e1.op, e2.op, f"{name}: event[{i}].op mismatch")

        # 6. Blobs
        self.assertEqual(len(m1.blobs), len(m2.blobs), f"{name}: blobs count mismatch")
        for i, (b1, b2) in enumerate(zip(m1.blobs, m2.blobs)):
            self.assertEqual(b1.name, b2.name, f"{name}: blob[{i}].name mismatch")
            self.assertEqual(b1.size, b2.size, f"{name}: blob[{i}].size mismatch")
            self.assertEqual(b1.interface, b2.interface, f"{name}: blob[{i}].interface mismatch")
            self.assertEqual(b1.sub_interface, b2.sub_interface, f"{name}: blob[{i}].sub_interface mismatch")
            self.assertEqual(b1.version.major, b2.version.major, f"{name}: blob[{i}].version.major mismatch")
            self.assertEqual(b1.version.minor, b2.version.minor, f"{name}: blob[{i}].version.minor mismatch")
            self.assertEqual(len(b1.data), len(b2.data), f"{name}: blob[{i}].data length mismatch")
            self.assertEqual(b1.data, b2.data, f"{name}: blob[{i}].data content mismatch")

        # 7. Tensor Descriptors
        self.assertEqual(len(m1.tensor_descs), len(m2.tensor_descs), f"{name}: tensor_descs count mismatch")
        for i, (t1, t2) in enumerate(zip(m1.tensor_descs, m2.tensor_descs)):
            self.assertEqual(t1.name, t2.name, f"{name}: tensor[{i}].name mismatch")
            self.assertEqual(t1.id, t2.id, f"{name}: tensor[{i}].id mismatch")
            self.assertEqual(t1.mem_id, t2.mem_id, f"{name}: tensor[{i}].mem_id mismatch")
            self.assertEqual(t1.size, t2.size, f"{name}: tensor[{i}].size mismatch")
            self.assertEqual(t1.offset, t2.offset, f"{name}: tensor[{i}].offset mismatch")
            self.assertEqual(t1.data_format, t2.data_format, f"{name}: tensor[{i}].data_format mismatch")
            self.assertEqual(t1.data_type, t2.data_type, f"{name}: tensor[{i}].data_type mismatch")
            self.assertEqual(t1.data_category, t2.data_category, f"{name}: tensor[{i}].data_category mismatch")
            self.assertEqual(t1.pixel_format, t2.pixel_format, f"{name}: tensor[{i}].pixel_format mismatch")
            self.assertEqual(t1.pixel_mapping, t2.pixel_mapping, f"{name}: tensor[{i}].pixel_mapping mismatch")
            self.assertEqual((t1.n, t1.c, t1.h, t1.w), (t2.n, t2.c, t2.h, t2.w), f"{name}: tensor[{i}] dims mismatch")
            self.assertEqual((t1.stride_0, t1.stride_1, t1.stride_2), (t2.stride_0, t2.stride_1, t2.stride_2), f"{name}: tensor[{i}] strides mismatch")

        # 8. Relocs
        self.assertEqual(len(m1.relocs), len(m2.relocs), f"{name}: relocs count mismatch")
        for i, (r1, r2) in enumerate(zip(m1.relocs, m2.relocs)):
            self.assertEqual(r1.address_id, r2.address_id, f"{name}: reloc[{i}].address_id mismatch")
            self.assertEqual(r1.write_id, r2.write_id, f"{name}: reloc[{i}].write_id mismatch")
            self.assertEqual(r1.offset, r2.offset, f"{name}: reloc[{i}].offset mismatch")
            self.assertEqual(r1.interface, r2.interface, f"{name}: reloc[{i}].interface mismatch")
            self.assertEqual(r1.sub_interface, r2.sub_interface, f"{name}: reloc[{i}].sub_interface mismatch")
            self.assertEqual(r1.reloc_type, r2.reloc_type, f"{name}: reloc[{i}].reloc_type mismatch")

        # 9. Submits
        self.assertEqual(len(m1.submits), len(m2.submits), f"{name}: submits count mismatch")
        for i, (s1, s2) in enumerate(zip(m1.submits, m2.submits)):
            self.assertEqual(s1.id, s2.id, f"{name}: submit[{i}].id mismatch")
            self.assertEqual(s1.task_id, s2.task_id, f"{name}: submit[{i}].task_id mismatch")

    def test_conv_model_roundtrip(self):
        """Test parsing and re-serializing test_conv_npu_b1.model."""
        self.assertTrue(self.conv_model_path.exists(), f"Missing {self.conv_model_path}")

        # 1. Read original model
        orig_model = ModelReader.read_file(self.conv_model_path)
        self.assertEqual(str(orig_model.version), "0.7.0")
        self.assertEqual(len(orig_model.tasks), 2)
        self.assertEqual(len(orig_model.blobs), 12)
        self.assertEqual(len(orig_model.tensor_descs), 2)
        self.assertEqual(orig_model.tensor_descs[0].name, "input")
        self.assertEqual(orig_model.tensor_descs[1].name, "output")

        # 2. Serialize to bytes
        serialized_bytes = ModelWriter.serialize(orig_model)
        self.assertEqual(serialized_bytes[4:8], b"NVDA", "Serialized bytes missing NVDA magic")

        # 3. Read serialized bytes
        reparsed_model = ModelReader.read_bytes(serialized_bytes)

        # 4. Verify 100% semantic parity
        self._assert_models_equal(orig_model, reparsed_model, "test_conv_npu_b1")

    def test_swiglu_mlp_roundtrip(self):
        """Test parsing and re-serializing test_swiglu_mlp_npu_b1.model."""
        self.assertTrue(self.swiglu_model_path.exists(), f"Missing {self.swiglu_model_path}")

        orig_model = ModelReader.read_file(self.swiglu_model_path)
        self.assertEqual(str(orig_model.version), "0.7.0")
        self.assertEqual(len(orig_model.tasks), 2)
        self.assertEqual(len(orig_model.blobs), 20)
        self.assertEqual(len(orig_model.relocs), 14)

        serialized_bytes = ModelWriter.serialize(orig_model)
        self.assertEqual(serialized_bytes[4:8], b"NVDA")

        reparsed_model = ModelReader.read_bytes(serialized_bytes)
        self._assert_models_equal(orig_model, reparsed_model, "test_swiglu_mlp_npu_b1")

    def test_smollm_head_roundtrip(self):
        """Test parsing and re-serializing smollm_lm_head_d0_npu_b1.model (29MB)."""
        if not self.smollm_model_path.exists():
            self.skipTest(f"SmolLM head model not present at {self.smollm_model_path}")

        orig_model = ModelReader.read_file(self.smollm_model_path)
        self.assertEqual(str(orig_model.version), "0.7.0")
        self.assertGreater(len(orig_model.blobs), 0)

        serialized_bytes = ModelWriter.serialize(orig_model)
        self.assertEqual(serialized_bytes[4:8], b"NVDA")

        reparsed_model = ModelReader.read_bytes(serialized_bytes)
        self._assert_models_equal(orig_model, reparsed_model, "smollm_lm_head_d0_npu_b1")


if __name__ == "__main__":
    unittest.main()
