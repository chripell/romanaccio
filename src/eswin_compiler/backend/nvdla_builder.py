"""NVDLA Hardware Register Programming & KMD Submodel Command Stream Builder.

This module lowers EIR Convolution and 1x1 Pointwise GEMM operations into
native EIC7700X NPU command packets:
  - CDMA (Convolution DMA): Fetches feature maps and weights into CBUF.
  - CSC (Convolution Sequence Controller): Directs weight/activation streaming.
  - CMAC (Convolution MAC Array): 2048 INT8 / 1024 FP16 MAC units per die.
  - CACC (Convolution Accumulator): Accumulates partial sums into CBUF.
  - SDP (Single Data Point): Element-wise scaling, bias addition, activations.

Hardware Register Reference & Base Offsets:
  - CACC_D_MISC_CFG_0:       0x0008 (Accumulator precision and rounding mode)
  - CMAC_A_CFG_0:             0x0000 (CMAC core control)
  - CDMA_D_DAIF_CFG_0:        0x0004 (Direct memory access interface config)
  - CSC_D_MISC_CFG_0:         0x0004 (Sequence controller mode and precision)
  - SDP_D_FEATURE_MODE_CFG_0: 0x0034 (Flying vs non-flying memory routing)
"""

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from .flatbuffer_emitter import (
        AddressEntry,
        BlobEntry,
        LoadableModel,
        MemoryEntry,
        ModelVersion,
        ModelWriter,
        RelocEntry,
        SubmitEntry,
        SubModel,
        TensorDescEntry,
    )
    from ..ir.graph import EIRGraph
    from ..ir.node import EIRNode
    from ..ir.op_types import EngineType, OpType
    from ..passes.gemm_to_conv1x1 import GemmToConv1x1Pass
except (ImportError, ValueError):
    from backend.flatbuffer_emitter import (
        AddressEntry,
        BlobEntry,
        LoadableModel,
        MemoryEntry,
        ModelVersion,
        ModelWriter,
        RelocEntry,
        SubmitEntry,
        SubModel,
        TensorDescEntry,
    )
    from ir.graph import EIRGraph
    from ir.node import EIRNode
    from ir.op_types import EngineType, OpType
    from passes.gemm_to_conv1x1 import GemmToConv1x1Pass


# ==============================================================================
# Hardware Command Constants & Enums
# ==============================================================================

# Operation types in KMD task dependency graph
OP_TYPE_BDMA = 0      # Memory-to-memory block copy
OP_TYPE_CONV = 1      # CDMA + CSC + CMAC + CACC convolution
OP_TYPE_SDP  = 2      # Single Data Point post-processing
OP_TYPE_PDP  = 3      # Planar Data Processor (Pooling)
OP_TYPE_CDP  = 4      # Channel Data Processor (LRN)
OP_TYPE_RUBIK = 5     # Memory reshuffle & layout transform
OP_TYPE_EVENT = 9     # Hardware event synchronization
OP_TYPE_CONFIG = 10   # Hardware pipeline configuration

# Precision modes
PRECISION_FLOAT16 = 2
PRECISION_INT8    = 4


# ==============================================================================
# NVDLA Surface Descriptor Packer
# ==============================================================================

def pack_dla_data_cube(
    cube_type: int,
    address_id: int,
    offset: int,
    size: int,
    batch: int,
    width: int,
    height: int,
    channel: int,
    line_stride: int,
    surf_stride: int,
    plane_stride: int = 0
) -> bytes:
    """Packs a 64-byte `struct dla_data_cube` hardware descriptor.
    
    Layout (DWARF-verified from EsSimulator):
      +0:  int16_t   type
      +2:  int16_t   address (address list index)
      +4:  uint32_t  offset
      +8:  uint32_t  size (in bytes)
      +12: uint16_t  batch
      +14: uint16_t  width
      +16: uint16_t  height
      +18: uint16_t  channel
      +20: uint16_t  reserved0
      +22: uint32_t  line_stride
      +26: uint32_t  surf_stride
      +30: uint32_t  plane_stride
      +34: uint8_t   padding[30]
    """
    header = struct.pack(
        "<hhIIHHHHHIII",
        cube_type,
        address_id,
        offset,
        size,
        batch,
        width,
        height,
        channel,
        0,              # reserved0
        line_stride,
        surf_stride,
        plane_stride
    )
    padding = b"\x00" * (64 - len(header))
    return header + padding


# ==============================================================================
# NVDLA Conv & 1x1 Pointwise GEMM Builder
# ==============================================================================

class NVDLAConvBuilder:
    """Assembles complete LoadableModel binaries for 2D Conv and 1x1 Pointwise GEMM."""

    @staticmethod
    def build_from_node(
        conv_node: EIRNode,
        model_name: str = "conv_model",
        sram_size: int = 4 * 1024 * 1024  # 4MB SPRAM per die
    ) -> LoadableModel:
        """Lowers an EIR CONV node into an executable LoadableModel."""
        if conv_node.op_type != OpType.CONV:
            raise ValueError(f"Expected CONV node, got {conv_node.op_type}")

        act_in = conv_node.inputs[0]
        weight_in = conv_node.inputs[1]
        act_out = conv_node.outputs[0]

        n_batch, c_in, h_in, w_in = act_in.shape
        c_out, _, k_h, k_w = weight_in.shape
        n_out, c_out_chk, h_out, w_out = act_out.shape

        assert c_out == c_out_chk, f"Channel mismatch: weight {c_out} vs output {c_out_chk}"

        # Ensure weights are FP16
        if weight_in.data is not None:
            w_bytes = np.ascontiguousarray(weight_in.data, dtype=np.float16).tobytes()
        else:
            w_bytes = b"\x00" * (c_out * c_in * k_h * k_w * 2)

        # 4-byte / 32-byte alignment for weights
        aligned_w_size = (len(w_bytes) + 127) & ~127
        if len(w_bytes) < aligned_w_size:
            w_bytes = w_bytes + b"\x00" * (aligned_w_size - len(w_bytes))

        in_size_bytes = n_batch * c_in * h_in * w_in * 2
        out_size_bytes = n_out * c_out * h_out * w_out * 2

        # ----------------------------------------------------------------------
        # 1. Build Blobs Table
        # ----------------------------------------------------------------------
        blobs: List[BlobEntry] = []

        # Blob 0: Weights W_0
        blobs.append(BlobEntry(
            name="W_0",
            size=len(w_bytes),
            interface=0,
            sub_interface=0,
            version=ModelVersion(0, 7, 0),
            data=w_bytes
        ))

        # Build KMD Surface List (4096 bytes)
        # Construct hardware surface cubes
        kmd_surf_data = bytearray(4096)

        # Cube 0 (offset 0): Weight data in DRAM (addr 1)
        w_line_stride = k_w * 2
        w_surf_stride = k_h * w_line_stride
        cube_weight_dram = pack_dla_data_cube(
            cube_type=0,
            address_id=1,
            offset=0,
            size=len(w_bytes),
            batch=2,
            width=k_w,
            height=k_h,
            channel=c_in,
            line_stride=w_line_stride,
            surf_stride=w_surf_stride
        )
        kmd_surf_data[0:64] = cube_weight_dram

        # Cube 1 (offset 64): Weight buffer in SRAM (addr 3)
        cube_weight_sram = pack_dla_data_cube(
            cube_type=1,
            address_id=3,
            offset=0,
            size=len(w_bytes),
            batch=2,
            width=k_w,
            height=k_h,
            channel=c_in,
            line_stride=w_line_stride,
            surf_stride=w_surf_stride
        )
        kmd_surf_data[64:128] = cube_weight_sram

        # Cube at offset 1024: Weight in SRAM for convolution execution
        cube_weight_conv = pack_dla_data_cube(
            cube_type=1,
            address_id=3,
            offset=0,
            size=len(w_bytes),
            batch=c_out,
            width=k_w,
            height=k_h,
            channel=c_in,
            line_stride=w_line_stride,
            surf_stride=w_surf_stride
        )
        kmd_surf_data[1024:1088] = cube_weight_conv

        # Cube at offset 1216: Input feature map in DRAM (addr 4) -> Reloc 0 & 1 target
        in_line_stride = w_in * 2
        in_surf_stride = h_in * in_line_stride
        cube_input = pack_dla_data_cube(
            cube_type=0,
            address_id=4,
            offset=0,
            size=in_size_bytes,
            batch=n_batch,
            width=w_in,
            height=h_in,
            channel=c_in,
            line_stride=in_line_stride,
            surf_stride=in_surf_stride
        )
        kmd_surf_data[1216:1280] = cube_input

        # Cube at offset 1280: Intermediate / Flying output to SDP
        out_line_stride = w_out * 2
        out_surf_stride = h_out * out_line_stride
        cube_flying_out = pack_dla_data_cube(
            cube_type=2,
            address_id=-1,
            offset=0,
            size=out_size_bytes * 2,  # FP32 accumulator in flight
            batch=n_out,
            width=w_out,
            height=h_out,
            channel=c_out,
            line_stride=out_line_stride * 2,
            surf_stride=out_surf_stride * 2
        )
        kmd_surf_data[1280:1344] = cube_flying_out

        # Cube at offset 2304: Output feature map in DRAM (addr 5) -> Reloc 2 & 3 target
        cube_output = pack_dla_data_cube(
            cube_type=0,
            address_id=5,
            offset=0,
            size=out_size_bytes,
            batch=n_out,
            width=w_out,
            height=h_out,
            channel=c_out,
            line_stride=out_line_stride,
            surf_stride=out_surf_stride
        )
        kmd_surf_data[2304:2368] = cube_output

        # Reloc offsets inside kmd-task-surf_list:
        # Reloc 0 (low) & 1 (high) patch input address at 1216 + 22 = 1238
        # Reloc 2 (low) & 3 (high) patch output address at 2304 + 22 = 2326
        reloc_in_offset = 1216 + 22
        reloc_out_offset = 2304 + 22

        # ----------------------------------------------------------------------
        # 2. Build Other KMD / UMD Task Blobs
        # ----------------------------------------------------------------------
        # kmd-task-addr0 (128 bytes)
        kmd_addr0_data = bytearray(128)
        struct.pack_into("<I", kmd_addr0_data, 0, 0x30000)
        struct.pack_into("<I", kmd_addr0_data, 68, (15 << 16) | 14)  # surf_list=15, op_list=14
        struct.pack_into("<I", kmd_addr0_data, 72, (16 << 16) | 13)  # lut_list=16, dep_graph=13
        struct.pack_into("<I", kmd_addr0_data, 76, (4 << 16) | 17)   # config_list=17
        struct.pack_into("<I", kmd_addr0_data, 80, 1)
        struct.pack_into("<I", kmd_addr0_data, 84, 11)

        # kmd-task-config_list (1024 bytes zeros)
        kmd_config_data = bytes(1024)

        # kmd-task-dep_graph (8704 bytes)
        kmd_dep_data = bytearray(8704)
        struct.pack_into("<hh", kmd_dep_data, 8, -1, 1)

        # kmd-task-lut_list (688 bytes: zeros or piecewise LUT)
        act = conv_node.attributes.get("activation")
        if act == "silu":
            from .lut_generator import ActivationType, generate_sdp_lut
            kmd_lut_data = generate_sdp_lut(ActivationType.SILU)
        elif act == "sigmoid":
            from .lut_generator import ActivationType, generate_sdp_lut
            kmd_lut_data = generate_sdp_lut(ActivationType.SIGMOID)
        elif act == "tanh":
            from .lut_generator import ActivationType, generate_sdp_lut
            kmd_lut_data = generate_sdp_lut(ActivationType.TANH)
        else:
            kmd_lut_data = bytes(688)

        # kmd-task-op_list (7936 bytes)
        kmd_op_data = bytearray(7936)
        struct.pack_into("<IIIIIIII", kmd_op_data, 0, 4, 288, len(w_bytes), 1, len(w_bytes), 1, len(w_bytes), 1)
        struct.pack_into("<IIIIIIII", kmd_op_data, 32, 4, 288, len(w_bytes), 1, len(w_bytes), 1, len(w_bytes), 1)

        # UMD Blobs
        umd_addr0_data = bytearray(128)
        struct.pack_into("<I", umd_addr0_data, 0, 0x30000)
        umd_config_data = bytes(1024)
        umd_dep_data = bytearray(4352)
        struct.pack_into("<Ihh", umd_dep_data, 0, 0xa, -1, 1)
        umd_op_data = bytes(3968)
        umd_surf_data = bytearray(2048)
        struct.pack_into("<hhIIHHHH", umd_surf_data, 0, 0, 2, 0, out_size_bytes, n_out, w_out, h_out, c_out)

        # Assemble Blobs
        blobs.append(BlobEntry(name="kmd-task-addr0", size=128, interface=2, sub_interface=1, data=bytes(kmd_addr0_data)))
        blobs.append(BlobEntry(name="kmd-task-config_list", size=1024, interface=2, sub_interface=6, data=kmd_config_data))
        blobs.append(BlobEntry(name="kmd-task-dep_graph", size=8704, interface=2, sub_interface=2, data=bytes(kmd_dep_data)))
        blobs.append(BlobEntry(name="kmd-task-lut_list", size=688, interface=2, sub_interface=5, data=kmd_lut_data))
        blobs.append(BlobEntry(name="kmd-task-op_list", size=7936, interface=2, sub_interface=3, data=bytes(kmd_op_data)))
        blobs.append(BlobEntry(name="kmd-task-surf_list", size=4096, interface=2, sub_interface=4, data=bytes(kmd_surf_data)))

        blobs.append(BlobEntry(name="umd-task-addr0", size=128, interface=1, sub_interface=1, data=bytes(umd_addr0_data)))
        blobs.append(BlobEntry(name="umd-task-config_list", size=1024, interface=1, sub_interface=5, data=umd_config_data))
        blobs.append(BlobEntry(name="umd-task-dep_graph", size=4352, interface=1, sub_interface=2, data=bytes(umd_dep_data)))
        blobs.append(BlobEntry(name="umd-task-op_list", size=3968, interface=1, sub_interface=3, data=umd_op_data))
        blobs.append(BlobEntry(name="umd-task-surf_list", size=2048, interface=1, sub_interface=4, data=bytes(umd_surf_data)))

        # ----------------------------------------------------------------------
        # 3. Build Memory List Entries
        # ----------------------------------------------------------------------
        memory_list: List[MemoryEntry] = [
            MemoryEntry(id=0, domain=0, flags=1, size=4096),
            MemoryEntry(id=1, domain=0, flags=3, size=max(4096, aligned_w_size), contents=["W_0"], offsets=[0]),
            MemoryEntry(id=2, domain=0, flags=1, size=max(4096, out_size_bytes), contents=["conv1_cvtPrecision"], offsets=[0]),
            MemoryEntry(id=3, domain=1, flags=1, size=sram_size, contents=["conv1.sram.edma"], offsets=[0]),
            MemoryEntry(id=4, domain=0, flags=5, size=in_size_bytes, contents=[], offsets=[], tensor_desc_id=0),
            MemoryEntry(id=5, domain=0, flags=9, size=out_size_bytes, contents=[], offsets=[], tensor_desc_id=1),
            # UMD Task Descriptors
            MemoryEntry(id=6, domain=0, flags=34, size=128, contents=["umd-task-addr0"], offsets=[0]),
            MemoryEntry(id=7, domain=0, flags=34, size=4352, contents=["umd-task-dep_graph"], offsets=[0]),
            MemoryEntry(id=8, domain=0, flags=34, size=3968, contents=["umd-task-op_list"], offsets=[0]),
            MemoryEntry(id=9, domain=0, flags=34, size=2048, contents=["umd-task-surf_list"], offsets=[0]),
            MemoryEntry(id=10, domain=0, flags=3, size=1024, contents=["umd-task-config_list"], offsets=[0]),
            MemoryEntry(id=11, domain=0, flags=1, size=4096, contents=["umd-task-dummy"], offsets=[0]),
            # KMD Task Descriptors
            MemoryEntry(id=12, domain=0, flags=34, size=128, contents=["kmd-task-addr0"], offsets=[0]),
            MemoryEntry(id=13, domain=0, flags=34, size=8704, contents=["kmd-task-dep_graph"], offsets=[0]),
            MemoryEntry(id=14, domain=0, flags=34, size=7936, contents=["kmd-task-op_list"], offsets=[0]),
            MemoryEntry(id=15, domain=0, flags=34, size=4096, contents=["kmd-task-surf_list"], offsets=[0]),
            MemoryEntry(id=16, domain=0, flags=34, size=688, contents=["kmd-task-lut_list"], offsets=[0]),
            MemoryEntry(id=17, domain=0, flags=3, size=1024, contents=["kmd-task-config_list"], offsets=[0]),
        ]

        # ----------------------------------------------------------------------
        # 4. Build Address Entries
        # ----------------------------------------------------------------------
        address_list: List[AddressEntry] = [
            AddressEntry(id=i, mem_id=i, offset=0, size=memory_list[i].size)
            for i in range(len(memory_list))
        ]

        # ----------------------------------------------------------------------
        # 5. Build Relocation Entries
        # ----------------------------------------------------------------------
        relocs: List[RelocEntry] = [
            RelocEntry(address_id=4, write_id=15, offset=reloc_in_offset, interface=2, sub_interface=4, reloc_type=1),
            RelocEntry(address_id=4, write_id=15, offset=reloc_in_offset + 4, interface=2, sub_interface=4, reloc_type=2),
            RelocEntry(address_id=5, write_id=15, offset=reloc_out_offset, interface=2, sub_interface=4, reloc_type=1),
            RelocEntry(address_id=5, write_id=15, offset=reloc_out_offset + 4, interface=2, sub_interface=4, reloc_type=2),
        ]

        # ----------------------------------------------------------------------
        # 6. Build Tensor Descriptors
        # ----------------------------------------------------------------------
        stride_w = 2
        stride_h = w_in * stride_w
        stride_c = h_in * stride_h
        tensor_in = TensorDescEntry(
            name=act_in.name,
            id=0,
            mem_id=4,
            size=in_size_bytes,
            offset=0,
            data_format=1,     # NCHW
            data_type=2,       # HALF (FP16)
            data_category=2,   # FEATURE
            pixel_format=36,   # FEATURE
            pixel_mapping=0,
            n=n_batch,
            c=c_in,
            h=h_in,
            w=w_in,
            stride_0=stride_w,
            stride_1=stride_h,
            stride_2=stride_c,
        )

        out_stride_w = 2
        out_stride_h = w_out * out_stride_w
        out_stride_c = h_out * out_stride_h
        tensor_out = TensorDescEntry(
            name=act_out.name,
            id=1,
            mem_id=5,
            size=out_size_bytes,
            offset=0,
            data_format=1,     # NCHW
            data_type=2,       # HALF (FP16)
            data_category=2,   # FEATURE
            pixel_format=36,   # FEATURE
            pixel_mapping=0,
            n=n_out,
            c=c_out,
            h=h_out,
            w=w_out,
            stride_0=out_stride_w,
            stride_1=out_stride_h,
            stride_2=out_stride_c,
        )
        tensor_descs: List[TensorDescEntry] = [tensor_in, tensor_out]

        # ----------------------------------------------------------------------
        # 7. Build SubModels & Submits
        # ----------------------------------------------------------------------
        tasks: List[SubModel] = [
            SubModel(id=0, kmd_id=1, interface=1, address_list=[6, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]),
            SubModel(id=1, kmd_id=-1, interface=2, address_list=[12, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17]),
        ]
        submits: List[SubmitEntry] = [
            SubmitEntry(id=0, task_id=[0]),
            SubmitEntry(id=1, task_id=[1]),
        ]

        return LoadableModel(
            version=ModelVersion(0, 7, 0),
            tasks=tasks,
            memory_list=memory_list,
            address_list=address_list,
            event_list=[],
            blobs=blobs,
            tensor_descs=tensor_descs,
            relocs=relocs,
            submits=submits,
        )


def compile_eir_to_model(graph: EIRGraph, output_path: Optional[Union[str, Path]] = None) -> LoadableModel:
    """Compiles an EIR computational graph into an EIC7700X .model loadable binary."""
    # 1. Lower GEMM/MatMul to 1x1 Conv
    pass_gemm = GemmToConv1x1Pass()
    pass_gemm.run(graph)

    # 2. Fuse Bias and SDP Activations (ReLU, Sigmoid, SiLU, Clip)
    try:
        from ..passes.fusion_sdp import SDPFusionPass
        from ..passes.sram_tiler import SRAMTilerPass
    except (ImportError, ValueError):
        from passes.fusion_sdp import SDPFusionPass
        from passes.sram_tiler import SRAMTilerPass

    pass_sdp = SDPFusionPass()
    pass_sdp.run(graph)

    # 3. Plan SPRAM (4MB) residency
    pass_sram = SRAMTilerPass()
    pass_sram.run(graph)

    # 4. Find primary Conv node
    conv_nodes = [n for n in graph.nodes if n.op_type == OpType.CONV]
    if not conv_nodes:
        raise ValueError("Graph contains no CONV nodes for NPU compilation.")

    conv_node = conv_nodes[0]
    model = NVDLAConvBuilder.build_from_node(conv_node, model_name=graph.name)

    if output_path is not None:
        ModelWriter.write_file(model, output_path)

    return model

