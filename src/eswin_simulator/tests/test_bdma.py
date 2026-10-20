"""
Unit tests for Memory Relocations, BDMA Engine, and Tensor File I/O (Milestone 2).
"""

from pathlib import Path
import struct
import numpy as np
import pytest

from eswin_simulator.loader.loadable_reader import LoadableModel
from eswin_simulator.memory.address_space import (
    VirtualAddressSpace,
    SPRAM_BASE_ADDRESS,
    DRAM_BASE_ADDRESS,
)
from eswin_simulator.engines.bdma import BDMAEngine, BDMATransferDesc


ARTIFACTS_DIR = Path("/home/chri/aixigua/deep/artifacts")
CONV_MODEL = ARTIFACTS_DIR / "test_conv_npu_b1.model"


@pytest.mark.skipif(not CONV_MODEL.is_file(), reason="Artifact model not found")
def test_relocation_resolution():
    """Verify that dynamic relocations patch 64-bit IOVA addresses bit-exactly into target buffers."""
    model = LoadableModel.from_file(CONV_MODEL)
    vas = VirtualAddressSpace(model)

    assert vas.relocations_applied == 4

    # Verify input tensor (address_id=4) was patched into region 15
    inp_vaddr = vas.get_virtual_address(4)
    inp_lo_expected = inp_vaddr & 0xFFFF_FFFF
    inp_hi_expected = (inp_vaddr >> 32) & 0xFFFF_FFFF

    reg15 = vas.get_region_buffer(15)
    lo_bytes = reg15.read(1238, 4)
    hi_bytes = reg15.read(1242, 4)

    lo_val = struct.unpack("<I", lo_bytes)[0]
    hi_val = struct.unpack("<I", hi_bytes)[0]

    assert lo_val == inp_lo_expected, f"Relocation lo mismatch: got 0x{lo_val:x}, expected 0x{inp_lo_expected:x}"
    assert hi_val == inp_hi_expected, f"Relocation hi mismatch: got 0x{hi_val:x}, expected 0x{inp_hi_expected:x}"

    # Verify output tensor (address_id=5) was patched into region 15
    out_vaddr = vas.get_virtual_address(5)
    out_lo_expected = out_vaddr & 0xFFFF_FFFF
    out_hi_expected = (out_vaddr >> 32) & 0xFFFF_FFFF

    out_lo_val = struct.unpack("<I", reg15.read(2326, 4))[0]
    out_hi_val = struct.unpack("<I", reg15.read(2330, 4))[0]

    assert out_lo_val == out_lo_expected
    assert out_hi_val == out_hi_expected


@pytest.mark.skipif(not CONV_MODEL.is_file(), reason="Artifact model not found")
def test_bdma_1d_linear_transfer():
    """Verify contiguous 1D memory copy between DRAM and SPRAM."""
    model = LoadableModel.from_file(CONV_MODEL)
    vas = VirtualAddressSpace(model)
    bdma = BDMAEngine(vas)

    inp = model.input_tensors[0]
    # Fill input in DRAM with recognizable data
    test_pattern = np.arange(inp.num_elements, dtype=inp.dtype).tobytes()
    vas.bind_input(inp, test_pattern)

    src_dram_vaddr = vas.get_virtual_address(inp.mem_id) + inp.offset
    dst_spram_vaddr = SPRAM_BASE_ADDRESS + 0x1000

    # Transfer DRAM -> SPRAM
    transferred = bdma.dram_to_spram(src_dram_vaddr, dst_spram_vaddr, len(test_pattern))
    assert transferred == len(test_pattern)

    # Read from SPRAM and verify bit-for-bit equality
    spram_data = vas.read_vaddr_bytes(dst_spram_vaddr, len(test_pattern))
    assert spram_data == test_pattern

    # Transfer SPRAM -> another location in DRAM (region 2, size 32KB)
    reg2 = vas.get_region_buffer(2)
    dst_dram_vaddr = reg2.base_vaddr
    bdma.spram_to_dram(dst_spram_vaddr, dst_dram_vaddr, len(test_pattern))

    dram_data = vas.read_vaddr_bytes(dst_dram_vaddr, len(test_pattern))
    assert dram_data == test_pattern


@pytest.mark.skipif(not CONV_MODEL.is_file(), reason="Artifact model not found")
def test_bdma_2d_strided_transfer():
    """Verify 2D strided memory copy (e.g. sub-matrix extraction with pitch padding)."""
    model = LoadableModel.from_file(CONV_MODEL)
    vas = VirtualAddressSpace(model)
    bdma = BDMAEngine(vas)

    # Source matrix: 4 rows, pitch=32 bytes, active data=16 bytes per row
    src_buf = vas.get_region_buffer(0)
    src_vaddr = src_buf.base_vaddr

    # Populate source with distinct rows
    for r in range(4):
        row_data = bytes([r * 10 + c for c in range(16)])
        padding = bytes(16)
        src_buf.write(r * 32, row_data + padding)

    # Destination buffer: contiguous 4 rows x 16 bytes = 64 bytes in SPRAM
    dst_vaddr = SPRAM_BASE_ADDRESS + 0x2000

    desc = BDMATransferDesc(
        src_addr=src_vaddr,
        dst_addr=dst_vaddr,
        line_size=16,
        line_repeat=4,
        src_line_stride=32,  # Skip 16 bytes of padding between source rows
        dst_line_stride=16,  # Pack tightly in destination
    )

    copied = bdma.transfer(desc)
    assert copied == 64

    # Verify destination is tightly packed without padding
    dst_bytes = vas.read_vaddr_bytes(dst_vaddr, 64)
    expected_bytes = b"".join(bytes([r * 10 + c for c in range(16)]) for r in range(4))
    assert dst_bytes == expected_bytes


@pytest.mark.skipif(not CONV_MODEL.is_file(), reason="Artifact model not found")
def test_bdma_3d_strided_transfer():
    """Verify 3D strided volume copy (surfaces, lines, elements)."""
    model = LoadableModel.from_file(CONV_MODEL)
    vas = VirtualAddressSpace(model)
    bdma = BDMAEngine(vas)

    # 3 surfaces, 2 lines per surface, 8 bytes per line
    # Source strides: line_stride=16, surf_stride=64
    src_buf = vas.get_region_buffer(0)
    src_vaddr = src_buf.base_vaddr

    expected_payload = bytearray()
    for s in range(3):
        for l in range(2):
            chunk = bytes([s * 50 + l * 10 + i for i in range(8)])
            src_buf.write(s * 64 + l * 16, chunk)
            expected_payload.extend(chunk)

    dst_vaddr = SPRAM_BASE_ADDRESS + 0x4000
    # Copy to destination packed tightly
    desc = BDMATransferDesc(
        src_addr=src_vaddr,
        dst_addr=dst_vaddr,
        line_size=8,
        line_repeat=2,
        surf_repeat=3,
        src_line_stride=16,
        dst_line_stride=8,
        src_surf_stride=64,
        dst_surf_stride=16,
    )

    copied = bdma.transfer(desc)
    assert copied == 3 * 2 * 8

    dst_bytes = vas.read_vaddr_bytes(dst_vaddr, copied)
    assert dst_bytes == bytes(expected_payload)


@pytest.mark.skipif(not CONV_MODEL.is_file(), reason="Artifact model not found")
def test_tensor_file_io(tmp_path: Path):
    """Verify loading input binary files and dumping output binary files."""
    model = LoadableModel.from_file(CONV_MODEL)
    vas = VirtualAddressSpace(model)

    inp = model.input_tensors[0]
    out = model.output_tensors[0]

    # Create dummy input .bin file
    test_input_file = tmp_path / "test_input.bin"
    rng_data = np.random.randn(*inp.shape).astype(inp.dtype)
    test_input_file.write_bytes(rng_data.tobytes())

    # Load file into virtual memory
    bytes_loaded = vas.load_input_file(inp, test_input_file)
    assert bytes_loaded == inp.size

    # Verify tensor readback matches
    read_back = vas.read_tensor(inp)
    np.testing.assert_array_equal(rng_data, read_back)

    # Simulate generating output by copying input into output region
    out_buf = vas.get_region_buffer(out.mem_id)
    out_payload = bytes(out.size)
    out_buf.write(out.offset, out_payload)

    # Dump output file to disk
    test_output_file = tmp_path / "test_output.bin"
    bytes_dumped = vas.dump_output_file(out, test_output_file)
    assert bytes_dumped == out.size
    assert test_output_file.is_file()
    assert len(test_output_file.read_bytes()) == out.size
