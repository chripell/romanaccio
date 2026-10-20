"""
Unit tests for VirtualAddressSpace and memory buffers (Milestone 1).
"""

import pytest
from pathlib import Path
import numpy as np

from eswin_simulator.loader.loadable_reader import LoadableModel
from eswin_simulator.memory.address_space import (
    VirtualAddressSpace,
    SPRAM_CAPACITY_BYTES,
    DRAM_BASE_ADDRESS,
)


ARTIFACTS_DIR = Path("/home/chri/aixigua/deep/artifacts")
CONV_MODEL = ARTIFACTS_DIR / "test_conv_npu_b1.model"


@pytest.mark.skipif(not CONV_MODEL.is_file(), reason="Artifact model not found")
def test_address_space_allocation():
    """Verify memory allocation across DRAM and SPRAM."""
    model = LoadableModel.from_file(CONV_MODEL)
    vas = VirtualAddressSpace(model)

    stats = vas.stats()
    assert stats["spram_capacity_bytes"] == SPRAM_CAPACITY_BYTES
    assert stats["spram_used_bytes"] == 4 * 1024 * 1024
    assert stats["dram_allocated_bytes"] > 0
    assert len(vas.regions) == 18
    assert len(vas.address_table) == 18


@pytest.mark.skipif(not CONV_MODEL.is_file(), reason="Artifact model not found")
def test_tensor_bind_and_read():
    """Verify binding and reading input/output tensors bit-for-bit."""
    model = LoadableModel.from_file(CONV_MODEL)
    vas = VirtualAddressSpace(model)

    inp = model.input_tensors[0]
    out = model.output_tensors[0]

    # Generate test deterministic pattern
    test_data = np.arange(inp.num_elements, dtype=inp.dtype).reshape(inp.shape)
    vas.bind_input(inp, test_data)

    read_back = vas.read_tensor(inp)
    np.testing.assert_array_equal(test_data, read_back)

    # Test reading output buffer initially zero
    out_arr = vas.read_tensor(out)
    assert out_arr.shape == out.shape
    assert np.all(out_arr == 0)


@pytest.mark.skipif(not CONV_MODEL.is_file(), reason="Artifact model not found")
def test_static_blob_loaded_into_region():
    """Verify that static blobs are populated into memory regions upon initialization."""
    model = LoadableModel.from_file(CONV_MODEL)
    vas = VirtualAddressSpace(model)

    # W_0 is placed in memory region 1 at offset 0
    w0_blob = model.get_blob("W_0")
    assert w0_blob is not None

    reg1_buf = vas.get_region_buffer(1)
    loaded_slice = bytes(reg1_buf.data[0 : len(w0_blob)])
    assert loaded_slice == w0_blob


def test_buffer_overflow_detection():
    """Assert that writing beyond buffer bounds raises IndexError."""
    model = LoadableModel.from_file(CONV_MODEL)
    vas = VirtualAddressSpace(model)

    inp = model.input_tensors[0]
    overflow_bytes = bytes(inp.size + 100)
    with pytest.raises(ValueError, match="exceeds tensor"):
        vas.bind_input(inp, overflow_bytes)
