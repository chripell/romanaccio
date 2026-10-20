"""
Unit tests for LoadableModel parser and tensor descriptors (Milestone 1).
"""

import pytest
from pathlib import Path

from eswin_simulator.loader.loadable_reader import LoadableModel, MemoryRegion, BlobEntry
from eswin_simulator.loader.tensor_desc import TensorDescriptor


ARTIFACTS_DIR = Path("/home/chri/aixigua/deep/artifacts")
CONV_MODEL = ARTIFACTS_DIR / "test_conv_npu_b1.model"
SWIGLU_MODEL = ARTIFACTS_DIR / "test_swiglu_mlp_npu_b1.model"
LINEAR_MODEL = ARTIFACTS_DIR / "test_linear_1x1conv_npu_b1.model"


def test_invalid_magic_raises_error(tmp_path: Path):
    """Assert that a file without 'NVDA' magic raises ValueError."""
    fake_model = tmp_path / "fake.model"
    fake_model.write_bytes(b"\x00\x00\x00\x00FAKE\x00\x00")
    with pytest.raises(ValueError, match="Invalid file identifier"):
        LoadableModel.from_file(fake_model)


def test_file_too_small_raises_error(tmp_path: Path):
    """Assert that a truncated file raises ValueError."""
    tiny_file = tmp_path / "tiny.model"
    tiny_file.write_bytes(b"NV")
    with pytest.raises(ValueError, match="File too small"):
        LoadableModel.from_file(tiny_file)


@pytest.mark.skipif(not CONV_MODEL.is_file(), reason="Artifact model not found")
def test_parse_conv_model():
    """Assert parsed structure of test_conv_npu_b1.model."""
    model = LoadableModel.from_file(CONV_MODEL)

    # Version check
    assert model.version == (0, 7, 0)

    # Tensor checks
    assert len(model.input_tensors) == 1
    assert len(model.output_tensors) == 1

    inp = model.input_tensors[0]
    assert inp.name == "input"
    assert inp.shape == (1, 3, 32, 32)
    assert inp.data_type == "HALF"
    assert inp.data_format == "NCHW"
    assert inp.size == 6144
    assert inp.is_input is True
    assert inp.is_output is False

    out = model.output_tensors[0]
    assert out.name == "output"
    assert out.shape == (1, 16, 30, 30)
    assert out.data_type == "HALF"
    assert out.data_format == "NCHW"
    assert out.size == 28800
    assert out.is_input is False
    assert out.is_output is True

    # Memory regions checks
    assert len(model.memory_regions) == 18
    # SPRAM region
    spram_reg = [r for r in model.memory_regions.values() if r.domain == "SRAM"]
    assert len(spram_reg) == 1
    assert spram_reg[0].size == 4 * 1024 * 1024

    # Blobs check
    assert len(model.blobs) == 12
    w0 = model.get_blob("W_0")
    assert w0 is not None
    assert len(w0) == 1152

    # Tasks check
    assert len(model.tasks) == 2
    assert model.tasks[0].interface == "UMD"
    assert model.tasks[1].interface == "KMD"


@pytest.mark.skipif(not SWIGLU_MODEL.is_file(), reason="Artifact model not found")
def test_parse_swiglu_mlp_model():
    """Assert parsed structure of test_swiglu_mlp_npu_b1.model."""
    model = LoadableModel.from_file(SWIGLU_MODEL)

    assert model.version == (0, 7, 0)
    assert len(model.input_tensors) == 1
    assert len(model.output_tensors) == 1

    inp = model.input_tensors[0]
    assert inp.shape == (1, 576, 1, 1)
    assert inp.data_type == "HALF"
    assert inp.size == 1152

    out = model.output_tensors[0]
    assert out.shape == (1, 576, 1, 1)
    assert out.data_type == "HALF"
    assert out.size == 1152

    # Verify weight split blobs exist
    assert "w_gatesplitM_0_0" in model.blobs
    assert "w_upsplitM_0_3" in model.blobs
    assert "w_downsplitM_0_6" in model.blobs


@pytest.mark.skipif(not LINEAR_MODEL.is_file(), reason="Artifact model not found")
def test_parse_linear_model():
    """Assert parsed structure of test_linear_1x1conv_npu_b1.model."""
    model = LoadableModel.from_file(LINEAR_MODEL)

    inp = model.input_tensors[0]
    assert inp.shape == (1, 576, 1, 1)

    out = model.output_tensors[0]
    assert out.shape == (1, 1536, 1, 1)
    assert out.size == 3072
