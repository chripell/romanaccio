"""Tests for ``compiler.py --convert-weights`` (Milestone 7 CLI entry point).

Runs the real CLI as a subprocess because the point under test *is* the CLI: argument
plumbing, the ".model needs --convert-weights" guard, and that the emitted pair
(loadable + reference weight blob) is what the hardware bit-match consumes.

The vendor SwiGLU loadable is the reference input; set ESWIN_SWIGLU_MODEL to relocate it.
Without that file the CLI tests skip, but the guard and help tests still run because they
need no loadable.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import os

COMPILER = Path(__file__).resolve().parents[1] / "compiler.py"
VENV_PY = Path(sys.executable)

SWIGLU_MODEL = Path(os.environ.get(
    "ESWIN_SWIGLU_MODEL",
    "/home/chri/aixigua/deep/artifacts/test_swiglu_mlp_npu_b1.model",
))
HAVE_SWIGLU = SWIGLU_MODEL.is_file()

# Verified geometry of that model's single weight region.
WEIGHT_SIZE = 5308416


def run_cli(*args: str, timeout: int = 240):
    return subprocess.run(
        [str(VENV_PY), str(COMPILER), *args],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(COMPILER.parent),
    )


class TestFlagRegistered(unittest.TestCase):

    def test_flags_documented_in_help(self):
        out = run_cli("--help")
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        self.assertIn("--convert-weights", out.stdout)
        self.assertIn("--weight-mem-id", out.stdout)

    def test_model_suffix_mentioned_in_format_error(self):
        """The 'unrecognized format' text must list .model, or users get stuck."""
        src = Path(__file__).resolve().parents[1].joinpath("compiler.py").read_text()
        self.assertIn(".onnx, .safetensors, .model or directory", src)


class TestModelWithoutFlagIsRejected(unittest.TestCase):

    def test_plain_model_input_fails_with_guidance(self):
        """A loadable is output, not input: it must not be silently accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake.model"
            fake.write_bytes(b"\x00" * 16)
            res = run_cli("-m", str(fake), "-o", str(Path(tmp) / "out"))
            self.assertNotEqual(res.returncode, 0,
                                "a .model input without --convert-weights must fail")
            combined = res.stdout + res.stderr
            self.assertIn("--convert-weights", combined,
                          "the error should say how to proceed")
            self.assertFalse((Path(tmp) / "out").exists() and
                             list((Path(tmp) / "out").glob("*.model")),
                             "nothing should be emitted on the rejected path")


@unittest.skipUnless(HAVE_SWIGLU, f"vendor SwiGLU model not present at {SWIGLU_MODEL}")
class TestConvertWeightsRuns(unittest.TestCase):

    def test_emits_classb_loadable_and_reference_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            res = run_cli("-m", str(SWIGLU_MODEL), "-o", str(out), "--convert-weights")
            self.assertEqual(res.returncode, 0, (res.stdout + res.stderr)[-600:])

            model = out / f"{SWIGLU_MODEL.stem}_classb.model"
            weights = out / f"{SWIGLU_MODEL.stem}_weights_mem1.bin"
            self.assertTrue(model.is_file(), "converted loadable missing")
            self.assertTrue(weights.is_file(), "reference weight blob missing")
            self.assertEqual(weights.stat().st_size, WEIGHT_SIZE)
            self.assertLess(model.stat().st_size, SWIGLU_MODEL.stat().st_size // 10,
                            "converted loadable should have lost its weight blobs")

    def test_converted_loadable_is_class_b_when_read_back(self):
        sys.path.insert(0, str(COMPILER.parent))
        from backend.flatbuffer_emitter import ModelReader

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            self.assertEqual(run_cli("-m", str(SWIGLU_MODEL), "-o", str(out),
                                     "--convert-weights").returncode, 0)
            m = ModelReader.read_file(str(out / f"{SWIGLU_MODEL.stem}_classb.model"))

        region = next(e for e in m.memory_list if e.id == 1)
        self.assertEqual(region.contents, [])
        self.assertEqual(region.offsets, [])
        self.assertEqual(region.flags, 5)
        self.assertEqual(region.bind_id, 1)
        self.assertEqual(region.tensor_desc_id, 2)
        self.assertEqual(region.size, WEIGHT_SIZE)
        # The reference blob is what makes the bit-match possible; the model must not
        # still carry the weights it was converted away from.
        self.assertFalse(any("WsplitM" in b.name or "splitM" in b.name and
                             b.name.startswith("w_") for b in m.blobs))
        wd = [t for t in m.tensor_descs if t.data_category == 1]
        self.assertEqual(len(wd), 1)
        self.assertEqual(wd[0].size, WEIGHT_SIZE)

    def test_explicit_mem_id_selects_the_same_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            auto = Path(tmp) / "auto"
            expl = Path(tmp) / "expl"
            self.assertEqual(run_cli("-m", str(SWIGLU_MODEL), "-o", str(auto),
                                     "--convert-weights").returncode, 0)
            self.assertEqual(run_cli("-m", str(SWIGLU_MODEL), "-o", str(expl),
                                     "--convert-weights", "--weight-mem-id", "1").returncode, 0)
            self.assertEqual((auto / f"{SWIGLU_MODEL.stem}_weights_mem1.bin").read_bytes(),
                             (expl / f"{SWIGLU_MODEL.stem}_weights_mem1.bin").read_bytes(),
                             "explicit selection must match auto-detection")

    def test_unknown_mem_id_fails_loudly(self):
        """A wrong id must not quietly produce an unconverted copy."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            res = run_cli("-m", str(SWIGLU_MODEL), "-o", str(out),
                          "--convert-weights", "--weight-mem-id", "99")
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("No Class A weight region", res.stdout + res.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
