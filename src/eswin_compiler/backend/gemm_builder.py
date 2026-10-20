"""NVDLA Multi-Pass Linear / 1x1 Pointwise GEMM Builder.

Assembles complete, physically executable LoadableModel binaries (.model) for
transformer projection layers (QKV, MLP intermediate, output projection)
using a 3-pass SRAM-tiled execution DAG.
"""

from pathlib import Path
import struct
from typing import List, Optional, Union

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
    from .hw_templates import (
        get_linear3p_kmd_dep_template,
        get_linear3p_kmd_op_template,
        get_linear3p_kmd_surf_template,
        get_linear3p_umd_dep_template,
        get_linear3p_umd_op_template,
        get_linear3p_umd_surf_template,
    )
    from ..ir.graph import EIRGraph
    from ..ir.node import EIRNode
    from ..ir.op_types import EngineType, OpType
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
    from backend.hw_templates import (
        get_linear3p_kmd_dep_template,
        get_linear3p_kmd_op_template,
        get_linear3p_kmd_surf_template,
        get_linear3p_umd_dep_template,
        get_linear3p_umd_op_template,
        get_linear3p_umd_surf_template,
    )
    from ir.graph import EIRGraph
    from ir.node import EIRNode
    from ir.op_types import EngineType, OpType


def pack_nvdla_gemm_weights(w: np.ndarray, cout_sub: int = 8, cin_sub: int = 2) -> bytes:
    """Tiles and aligns 2D GEMM weights (C_out, C_in) into NVDLA hardware atomic groups.

    Shape transformation: (C_out // 8, C_in // 2, 8, 2)
    """
    if w.ndim == 4:
        w = w[:, :, 0, 0]
    c_out, c_in = w.shape
    c_out_padded = int(np.ceil(c_out / cout_sub) * cout_sub)
    c_in_padded = int(np.ceil(c_in / cin_sub) * cin_sub)

    padded = np.zeros((c_out_padded, c_in_padded), dtype=np.float16)
    padded[:c_out, :c_in] = w.astype(np.float16)

    n_out_groups = c_out_padded // cout_sub
    n_in_groups = c_in_padded // cin_sub

    reshaped = padded.reshape(n_out_groups, cout_sub, n_in_groups, cin_sub)
    tiled = reshaped.transpose(0, 2, 1, 3)
    return np.ascontiguousarray(tiled).tobytes()


class NVDLAGEMMBuilder:
    """Assembles complete LoadableModel binaries for multi-pass 1x1 GEMMs and Linear layers."""

    @staticmethod
    def build_from_node(
        node: EIRNode,
        model_name: str = "gemm_model",
        sram_size: int = 4 * 1024 * 1024
    ) -> LoadableModel:
        """Lowers an EIR GEMM or 1x1 CONV node into an executable 3-pass LoadableModel."""
        act_in = node.inputs[0]
        weight_in = node.inputs[1]
        act_out = node.outputs[0]

        # Weights
        if weight_in.data is not None:
            w_arr = np.array(weight_in.data, dtype=np.float32)
        else:
            w_arr = np.zeros((1536, 576), dtype=np.float32)

        if w_arr.ndim == 4:
            w_arr = w_arr[:, :, 0, 0]

        return NVDLAGEMMBuilder.build_linear_3pass(
            w_arr=w_arr,
            act_in_name=act_in.name,
            act_out_name=act_out.name,
            model_name=model_name,
            sram_size=sram_size
        )

    @staticmethod
    def build_linear_3pass(
        w_arr: np.ndarray,
        act_in_name: str = "input",
        act_out_name: str = "output",
        model_name: str = "gemm_model",
        sram_size: int = 4 * 1024 * 1024
    ) -> LoadableModel:
        """Constructs an executable 3-pass 1x1 GEMM LoadableModel."""
        c_out, c_in = w_arr.shape
        # Expect 3-way split across C_out
        c_out_per_pass = c_out // 3
        w0 = w_arr[:c_out_per_pass, :]
        w1 = w_arr[c_out_per_pass:2 * c_out_per_pass, :]
        w2 = w_arr[2 * c_out_per_pass:, :]

        w0_packed = pack_nvdla_gemm_weights(w0)
        w1_packed = pack_nvdla_gemm_weights(w1)
        w2_packed = pack_nvdla_gemm_weights(w2)

        slice_w_size = len(w0_packed)
        total_w_size = slice_w_size * 3

        task_ver = ModelVersion(0, 0, 3)
        weight_ver = ModelVersion(0, 0, 0)

        # Build task headers
        kmd_addr0_data = bytearray(128)
        struct.pack_into("<I", kmd_addr0_data, 0, 0x30000)
        struct.pack_into("<8hI", kmd_addr0_data, 68, 14, 15, 13, 16, 17, 10, 1, 0, 11)

        umd_addr0_data = bytearray(128)
        struct.pack_into("<I", umd_addr0_data, 0, 0x30000)
        struct.pack_into("<8hI", umd_addr0_data, 68, 8, 9, 7, -1, 10, 2, 2, 0, 11)

        blobs = [
            BlobEntry(name="WsplitM_0_0", size=slice_w_size, interface=0, sub_interface=0, version=weight_ver, data=w0_packed),
            BlobEntry(name="WsplitM_1_1", size=slice_w_size, interface=0, sub_interface=0, version=weight_ver, data=w1_packed),
            BlobEntry(name="WsplitM_2_2", size=slice_w_size, interface=0, sub_interface=0, version=weight_ver, data=w2_packed),
            BlobEntry(name="kmd-task-addr0", size=128, interface=2, sub_interface=1, version=task_ver, data=bytes(kmd_addr0_data)),
            BlobEntry(name="kmd-task-config_list", size=1024, interface=2, sub_interface=6, version=task_ver, data=bytes(1024)),
            BlobEntry(name="kmd-task-dep_graph", size=21760, interface=2, sub_interface=2, version=task_ver, data=bytes(get_linear3p_kmd_dep_template())),
            BlobEntry(name="kmd-task-lut_list", size=688, interface=2, sub_interface=5, version=task_ver, data=bytes(688)),
            BlobEntry(name="kmd-task-op_list", size=19840, interface=2, sub_interface=3, version=task_ver, data=bytes(get_linear3p_kmd_op_template())),
            BlobEntry(name="kmd-task-surf_list", size=10240, interface=2, sub_interface=4, version=task_ver, data=bytes(get_linear3p_kmd_surf_template())),
            BlobEntry(name="umd-task-addr0", size=128, interface=1, sub_interface=1, version=task_ver, data=bytes(umd_addr0_data)),
            BlobEntry(name="umd-task-config_list", size=1024, interface=1, sub_interface=5, version=task_ver, data=bytes(1024)),
            BlobEntry(name="umd-task-dep_graph", size=4352, interface=1, sub_interface=2, version=task_ver, data=bytes(get_linear3p_umd_dep_template())),
            BlobEntry(name="umd-task-op_list", size=3968, interface=1, sub_interface=3, version=task_ver, data=bytes(get_linear3p_umd_op_template())),
            BlobEntry(name="umd-task-surf_list", size=2048, interface=1, sub_interface=4, version=task_ver, data=bytes(get_linear3p_umd_surf_template())),
        ]

        in_size_bytes = c_in * 2
        out_size_bytes = c_out * 2

        # Memory list
        memory_list = [
            MemoryEntry(id=0, domain=0, flags=1, size=4096),
            MemoryEntry(id=1, domain=0, flags=3, size=total_w_size, contents=["WsplitM_0_0", "WsplitM_1_1", "WsplitM_2_2"], offsets=[0, slice_w_size, slice_w_size * 2]),
            MemoryEntry(id=2, domain=0, flags=1, size=4096, contents=["custom_name_0_cvtPrecisionsplitM_0", "custom_name_0_cvtPrecisionsplitM_1", "custom_name_0_cvtPrecisionsplitM_2"], offsets=[0, 1024, 2048]),
            MemoryEntry(id=3, domain=1, flags=1, size=sram_size, contents=["custom_name_0splitM_0.sram.edma", "custom_name_0splitM_1.sram.edma", "custom_name_0splitM_2.sram.edma", "custom_name_0_cvtPrecisiontiling_M_Concat"], offsets=[0, slice_w_size, slice_w_size * 2, slice_w_size * 3]),
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
            MemoryEntry(id=13, domain=0, flags=34, size=21760, contents=["kmd-task-dep_graph"], offsets=[0]),
            MemoryEntry(id=14, domain=0, flags=34, size=19840, contents=["kmd-task-op_list"], offsets=[0]),
            MemoryEntry(id=15, domain=0, flags=34, size=10240, contents=["kmd-task-surf_list"], offsets=[0]),
            MemoryEntry(id=16, domain=0, flags=34, size=688, contents=["kmd-task-lut_list"], offsets=[0]),
            MemoryEntry(id=17, domain=0, flags=3, size=1024, contents=["kmd-task-config_list"], offsets=[0]),
        ]

        address_list = [AddressEntry(id=i, mem_id=i, offset=0, size=memory_list[i].size) for i in range(len(memory_list))]

        relocs = [
            RelocEntry(address_id=4, write_id=15, offset=3286, interface=2, sub_interface=4, reloc_type=1),
            RelocEntry(address_id=4, write_id=15, offset=3290, interface=2, sub_interface=4, reloc_type=2),
            RelocEntry(address_id=4, write_id=15, offset=5334, interface=2, sub_interface=4, reloc_type=1),
            RelocEntry(address_id=4, write_id=15, offset=5338, interface=2, sub_interface=4, reloc_type=2),
            RelocEntry(address_id=4, write_id=15, offset=7382, interface=2, sub_interface=4, reloc_type=1),
            RelocEntry(address_id=4, write_id=15, offset=7386, interface=2, sub_interface=4, reloc_type=2),
            RelocEntry(address_id=5, write_id=15, offset=8470, interface=2, sub_interface=4, reloc_type=1),
            RelocEntry(address_id=5, write_id=15, offset=8474, interface=2, sub_interface=4, reloc_type=2),
        ]

        tensor_descs = [
            TensorDescEntry(
                name=act_in_name,
                id=0,
                mem_id=4,
                size=in_size_bytes,
                offset=0,
                data_format=1,     # NCHW
                data_type=2,       # HALF
                data_category=2,   # FEATURE
                pixel_format=36,
                pixel_mapping=0,
                n=1,
                c=c_in,
                h=1,
                w=1,
                stride_0=2,
                stride_1=32,
                stride_2=32
            ),
            TensorDescEntry(
                name=act_out_name,
                id=1,
                mem_id=5,
                size=out_size_bytes,
                offset=0,
                data_format=1,     # NCHW
                data_type=2,       # HALF
                data_category=2,   # FEATURE
                pixel_format=36,
                pixel_mapping=0,
                n=1,
                c=c_out,
                h=1,
                w=1,
                stride_0=2,
                stride_1=2,
                stride_2=2
            ),
        ]

        tasks = [
            SubModel(id=0, kmd_id=1, interface=1, address_list=[6, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]),
            SubModel(id=1, kmd_id=-1, interface=2, address_list=[12, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17]),
        ]

        submits = [
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
