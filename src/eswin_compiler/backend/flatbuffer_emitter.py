"""FlatBuffers Loadable (.model) Reader and Writer for ESWIN EIC7700X NPU.

This module provides high-level Python dataclasses representing the internal
structure of an EIC7700X compiled NPU model (.model file), along with `ModelReader`
to parse binary loadables and `ModelWriter` to serialize them into valid FlatBuffers
packages accepted by the NPU runtime (libnn_runtime.so) and EsSimulator.
"""

from dataclasses import dataclass, field
from enum import IntEnum
import os
from pathlib import Path
import sys
from typing import List, Optional, Union

# Ensure schema/generated is accessible
_CURRENT_DIR = Path(__file__).parent.resolve()
_SCHEMA_GEN_DIR = _CURRENT_DIR.parent / "schema" / "generated"
if str(_SCHEMA_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_SCHEMA_GEN_DIR))

import flatbuffers
from eswin.loadable import (
    AddressListEntry,
    Blob,
    DataCategory,
    DataFormat,
    DataType,
    EventListEntry,
    EventOp,
    EventType,
    Interface,
    Loadable,
    MemoryDomain,
    MemoryFlags,
    MemoryListEntry,
    PixelFormat,
    PixelMapping,
    RelocListEntry,
    SubmitListEntry,
    SubModelListEntry,
    TensorDescListEntry,
    Version,
)


# ==============================================================================
# High-Level Dataclasses
# ==============================================================================

@dataclass
class ModelVersion:
    """Loadable specification version (standard vendor version is 0.7.0)."""
    major: int = 0
    minor: int = 7
    sub_minor: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.sub_minor}"


@dataclass
class SubModel:
    """Sub-model / task descriptor within the loadable.
    
    A typical model contains:
    - Task 0: UMD (User Mode Driver) task graph (interface = 1, kmd_id = 1)
    - Task 1: KMD (Kernel Mode Driver) hardware execution task (interface = 2, kmd_id = -1)
    """
    id: int = 0
    kmd_id: int = -1
    interface: int = 1  # 1: UMD, 2: KMD
    address_list: List[int] = field(default_factory=list)


@dataclass
class MemoryEntry:
    """Memory allocation descriptor for weights, activations, and descriptors.
    
    Attributes:
        id: Unique identifier for this memory allocation.
        domain: 0 for SYSTEM (DRAM), 1 for SRAM.
        flags: Bitmask specifying allocation intent:
               1: ALLOC, 2: SET (pre-loaded), 4: INPUT, 8: OUTPUT, 16: CACHE, 32: NO_CACHE.
        size: Allocation size in bytes.
        alignment: Required alignment in bytes (typically 4096).
        contents: List of blob symbol names stored in this memory region.
        offsets: Relative byte offsets for each blob in `contents`.
        bind_id: Symbolic bind identifier.
        tensor_desc_id: Index into tensor_desc_list for I/O buffers.
    """
    id: int = 0
    domain: int = 0
    flags: int = 0
    size: int = 0
    alignment: int = 4096
    contents: List[str] = field(default_factory=list)
    offsets: List[int] = field(default_factory=list)
    bind_id: int = 0
    tensor_desc_id: int = 0


@dataclass
class AddressEntry:
    """Symbolic address entry mapping a virtual buffer to a memory entry."""
    id: int = 0
    mem_id: int = 0
    offset: int = 0
    size: int = 0


@dataclass
class EventEntry:
    """Synchronization primitive (wait / signal event)."""
    id: int = 0
    type: int = 0
    target: int = 0
    val: int = 0
    op: int = 0


@dataclass
class BlobEntry:
    """Binary data payload (weights, command lists, descriptors, LUTs).
    
    Attributes:
        name: Unique symbol name (e.g., 'W_0', 'kmd-task-op_list').
        size: Size in bytes.
        interface: 0: NONE/Weight, 1: UMD, 2: KMD.
        sub_interface: Sub-task or descriptor stream index.
        version: Blob format version.
        data: Raw payload bytes.
    """
    name: str = ""
    size: int = 0
    interface: int = 0
    sub_interface: int = 0
    version: ModelVersion = field(default_factory=ModelVersion)
    data: bytes = b""


@dataclass
class TensorDescEntry:
    """Tensor descriptor for network input and output interfaces.
    
    Attributes:
        name: User-facing name (e.g. 'input', 'output').
        id: Unique descriptor index.
        mem_id: Target memory allocation ID.
        size: Total tensor size in bytes.
        offset: Byte offset inside memory allocation.
        data_format: 1: NCHW, 2: NHWC.
        data_type: 1: FLOAT, 2: HALF (float16), 3: INT16, 4: INT8.
        data_category: 0: IMAGE, 1: WEIGHT, 2: FEATURE, 3: PLANAR, 4: BIAS.
        pixel_format: 36: FEATURE (standard for tensors).
        pixel_mapping: 0: PITCH_LINEAR.
        n, c, h, w: Tensor dimension shapes.
        stride_0..7: Memory strides in bytes.
    """
    name: str = ""
    id: int = 0
    mem_id: int = 0
    size: int = 0
    offset: int = 0
    data_format: int = 1
    data_type: int = 2
    data_category: int = 2
    pixel_format: int = 36
    pixel_mapping: int = 0
    n: int = 1
    c: int = 1
    h: int = 1
    w: int = 1
    stride_0: int = 0
    stride_1: int = 0
    stride_2: int = 0
    stride_3: int = 0
    stride_4: int = 0
    stride_5: int = 0
    stride_6: int = 0
    stride_7: int = 0


@dataclass
class RelocEntry:
    """Relocation entry for patching physical IOVA addresses into command packets."""
    address_id: int = 0
    write_id: int = 0
    offset: int = 0
    interface: int = 2
    sub_interface: int = 4
    reloc_type: int = 1


@dataclass
class SubmitEntry:
    """Execution submission entry linking tasks to scheduler queues."""
    id: int = 0
    task_id: List[int] = field(default_factory=list)


@dataclass
class LoadableModel:
    """Complete in-memory representation of an EIC7700X .model file."""
    version: ModelVersion = field(default_factory=ModelVersion)
    tasks: List[SubModel] = field(default_factory=list)
    memory_list: List[MemoryEntry] = field(default_factory=list)
    address_list: List[AddressEntry] = field(default_factory=list)
    event_list: List[EventEntry] = field(default_factory=list)
    blobs: List[BlobEntry] = field(default_factory=list)
    tensor_descs: List[TensorDescEntry] = field(default_factory=list)
    relocs: List[RelocEntry] = field(default_factory=list)
    submits: List[SubmitEntry] = field(default_factory=list)


# ==============================================================================
# ModelReader
# ==============================================================================

class ModelReader:
    """Deserializer for ESWIN EIC7700X .model files."""

    @staticmethod
    def read_file(file_path: Union[str, Path]) -> LoadableModel:
        """Reads a .model file from disk and parses it into a LoadableModel."""
        with open(file_path, "rb") as f:
            data = f.read()
        return ModelReader.read_bytes(data)

    @staticmethod
    def read_bytes(buf: bytes) -> LoadableModel:
        """Parses a FlatBuffers byte buffer into a LoadableModel."""
        if len(buf) < 8:
            raise ValueError(f"Buffer too short to be a valid .model file ({len(buf)} bytes)")

        # Verify NVDA file identifier
        if not Loadable.Loadable.LoadableBufferHasIdentifier(buf, 0):
            magic = buf[4:8]
            raise ValueError(f"Invalid file identifier: expected b'NVDA', got {magic!r}")

        root = Loadable.Loadable.GetRootAsLoadable(buf, 0)
        model = LoadableModel()

        # 1. Version
        v = root.Version()
        if v is not None:
            model.version = ModelVersion(
                major=v.Major(),
                minor=v.Minor(),
                sub_minor=v.SubMinor()
            )

        # 2. Task List (SubModels)
        for i in range(root.TaskListLength()):
            item = root.TaskList(i)
            if item is None:
                continue
            addrs = [item.AddressList(j) for j in range(item.AddressListLength())]
            model.tasks.append(SubModel(
                id=item.Id(),
                kmd_id=item.KmdId(),
                interface=item.Interface(),
                address_list=addrs
            ))

        # 3. Memory List
        for i in range(root.MemoryListLength()):
            item = root.MemoryList(i)
            if item is None:
                continue
            contents = [
                item.Contents(j).decode("utf-8") if isinstance(item.Contents(j), bytes) else item.Contents(j)
                for j in range(item.ContentsLength())
            ]
            offsets = [item.Offsets(j) for j in range(item.OffsetsLength())]
            model.memory_list.append(MemoryEntry(
                id=item.Id(),
                domain=item.Domain(),
                flags=item.Flags(),
                size=item.Size(),
                alignment=item.Alignment(),
                contents=contents,
                offsets=offsets,
                bind_id=item.BindId(),
                tensor_desc_id=item.TensorDescId()
            ))

        # 4. Address List
        for i in range(root.AddressListLength()):
            item = root.AddressList(i)
            if item is None:
                continue
            model.address_list.append(AddressEntry(
                id=item.Id(),
                mem_id=item.MemId(),
                offset=item.Offset(),
                size=item.Size()
            ))

        # 5. Event List
        for i in range(root.EventListLength()):
            item = root.EventList(i)
            if item is None:
                continue
            model.event_list.append(EventEntry(
                id=item.Id(),
                type=item.Type(),
                target=item.Target(),
                val=item.Val(),
                op=item.Op()
            ))

        # 6. Blobs
        for i in range(root.BlobsLength()):
            item = root.Blobs(i)
            if item is None:
                continue
            b_name = item.Name()
            name_str = b_name.decode("utf-8") if isinstance(b_name, bytes) else (b_name or "")
            b_ver_obj = item.Version()
            b_ver = ModelVersion(b_ver_obj.Major(), b_ver_obj.Minor(), b_ver_obj.SubMinor()) if b_ver_obj else ModelVersion()
            
            # Efficiently extract byte payload
            if hasattr(item, "DataAsNumpy") and not item.DataIsNone():
                blob_data = bytes(item.DataAsNumpy())
            else:
                blob_data = bytes(item.Data(j) for j in range(item.DataLength()))

            model.blobs.append(BlobEntry(
                name=name_str,
                size=item.Size(),
                interface=item.Interface(),
                sub_interface=item.SubInterface(),
                version=b_ver,
                data=blob_data
            ))

        # 7. Tensor Descriptors
        for i in range(root.TensorDescListLength()):
            item = root.TensorDescList(i)
            if item is None:
                continue
            t_name = item.Name()
            name_str = t_name.decode("utf-8") if isinstance(t_name, bytes) else (t_name or "")
            model.tensor_descs.append(TensorDescEntry(
                name=name_str,
                id=item.Id(),
                mem_id=item.MemId(),
                size=item.Size(),
                offset=item.Offset(),
                data_format=item.DataFormat(),
                data_type=item.DataType(),
                data_category=item.DataCategory(),
                pixel_format=item.PixelFormat(),
                pixel_mapping=item.PixelMapping(),
                n=item.N(),
                c=item.C(),
                h=item.H(),
                w=item.W(),
                stride_0=item.Stride0(),
                stride_1=item.Stride1(),
                stride_2=item.Stride2(),
                stride_3=item.Stride3(),
                stride_4=item.Stride4(),
                stride_5=item.Stride5(),
                stride_6=item.Stride6(),
                stride_7=item.Stride7(),
            ))

        # 8. Reloc List
        for i in range(root.RelocListLength()):
            item = root.RelocList(i)
            if item is None:
                continue
            model.relocs.append(RelocEntry(
                address_id=item.AddressId(),
                write_id=item.WriteId(),
                offset=item.Offset(),
                interface=item.Interface(),
                sub_interface=item.SubInterface(),
                reloc_type=item.RelocType()
            ))

        # 9. Submit List
        for i in range(root.SubmitListLength()):
            item = root.SubmitList(i)
            if item is None:
                continue
            tids = [item.TaskId(j) for j in range(item.TaskIdLength())]
            model.submits.append(SubmitEntry(
                id=item.Id(),
                task_id=tids
            ))

        return model


# ==============================================================================
# ModelWriter
# ==============================================================================

class ModelWriter:
    """Serializer for constructing FlatBuffers .model packages."""

    @staticmethod
    def write_file(model: LoadableModel, file_path: Union[str, Path]) -> None:
        """Serializes model and writes it to file_path."""
        data = ModelWriter.serialize(model)
        with open(file_path, "wb") as f:
            f.write(data)

    @staticmethod
    def serialize(model: LoadableModel) -> bytes:
        """Serializes LoadableModel into a standard FlatBuffers .model byte string."""
        builder = flatbuffers.Builder(initialSize=1024 * 1024)

        # ----------------------------------------------------------------------
        # 1. Serialize Blobs (strings, byte vectors, then tables)
        # ----------------------------------------------------------------------
        blob_offsets = []
        for b in model.blobs:
            name_offset = builder.CreateString(b.name)
            data_offset = builder.CreateByteVector(b.data)

            Blob.Start(builder)
            Blob.AddName(builder, name_offset)
            Blob.AddSize(builder, b.size if b.size != 0 else len(b.data))
            Blob.AddInterface(builder, b.interface)
            Blob.AddSubInterface(builder, b.sub_interface)
            v_offset = Version.CreateVersion(builder, b.version.major, b.version.minor, b.version.sub_minor)
            Blob.AddVersion(builder, v_offset)
            Blob.AddData(builder, data_offset)
            blob_offsets.append(Blob.End(builder))

        # ----------------------------------------------------------------------
        # 2. Serialize Memory Entries
        # ----------------------------------------------------------------------
        mem_offsets = []
        for mem in model.memory_list:
            content_str_offsets = [builder.CreateString(c) for c in mem.contents]
            contents_vec = 0
            if content_str_offsets:
                MemoryListEntry.StartContentsVector(builder, len(content_str_offsets))
                for c_off in reversed(content_str_offsets):
                    builder.PrependUOffsetTRelative(c_off)
                contents_vec = builder.EndVector()

            offsets_vec = 0
            if mem.offsets:
                MemoryListEntry.StartOffsetsVector(builder, len(mem.offsets))
                for off in reversed(mem.offsets):
                    builder.PrependUint64(off)
                offsets_vec = builder.EndVector()

            MemoryListEntry.Start(builder)
            MemoryListEntry.AddId(builder, mem.id)
            MemoryListEntry.AddDomain(builder, mem.domain)
            MemoryListEntry.AddFlags(builder, mem.flags)
            MemoryListEntry.AddSize(builder, mem.size)
            MemoryListEntry.AddAlignment(builder, mem.alignment)
            if contents_vec != 0:
                MemoryListEntry.AddContents(builder, contents_vec)
            if offsets_vec != 0:
                MemoryListEntry.AddOffsets(builder, offsets_vec)
            MemoryListEntry.AddBindId(builder, mem.bind_id)
            MemoryListEntry.AddTensorDescId(builder, mem.tensor_desc_id)
            mem_offsets.append(MemoryListEntry.End(builder))

        # ----------------------------------------------------------------------
        # 3. Serialize Address Entries
        # ----------------------------------------------------------------------
        addr_offsets = []
        for addr in model.address_list:
            AddressListEntry.Start(builder)
            AddressListEntry.AddId(builder, addr.id)
            AddressListEntry.AddMemId(builder, addr.mem_id)
            AddressListEntry.AddOffset(builder, addr.offset)
            AddressListEntry.AddSize(builder, addr.size)
            addr_offsets.append(AddressListEntry.End(builder))

        # ----------------------------------------------------------------------
        # 4. Serialize Event Entries
        # ----------------------------------------------------------------------
        ev_offsets = []
        for ev in model.event_list:
            EventListEntry.Start(builder)
            EventListEntry.AddId(builder, ev.id)
            EventListEntry.AddType(builder, ev.type)
            EventListEntry.AddTarget(builder, ev.target)
            EventListEntry.AddVal(builder, ev.val)
            EventListEntry.AddOp(builder, ev.op)
            ev_offsets.append(EventListEntry.End(builder))

        # ----------------------------------------------------------------------
        # 5. Serialize Task Entries (SubModels)
        # ----------------------------------------------------------------------
        task_offsets = []
        for task in model.tasks:
            addr_vec = 0
            if task.address_list:
                SubModelListEntry.StartAddressListVector(builder, len(task.address_list))
                for a in reversed(task.address_list):
                    builder.PrependUint16(a)
                addr_vec = builder.EndVector()

            SubModelListEntry.Start(builder)
            SubModelListEntry.AddId(builder, task.id)
            SubModelListEntry.AddKmdId(builder, task.kmd_id)
            SubModelListEntry.AddInterface(builder, task.interface)
            if addr_vec != 0:
                SubModelListEntry.AddAddressList(builder, addr_vec)
            task_offsets.append(SubModelListEntry.End(builder))

        # ----------------------------------------------------------------------
        # 6. Serialize Tensor Descriptors
        # ----------------------------------------------------------------------
        td_offsets = []
        for td in model.tensor_descs:
            name_offset = builder.CreateString(td.name)

            TensorDescListEntry.Start(builder)
            TensorDescListEntry.AddName(builder, name_offset)
            TensorDescListEntry.AddId(builder, td.id)
            TensorDescListEntry.AddMemId(builder, td.mem_id)
            TensorDescListEntry.AddSize(builder, td.size)
            TensorDescListEntry.AddOffset(builder, td.offset)
            TensorDescListEntry.AddDataFormat(builder, td.data_format)
            TensorDescListEntry.AddDataType(builder, td.data_type)
            TensorDescListEntry.AddDataCategory(builder, td.data_category)
            TensorDescListEntry.AddPixelFormat(builder, td.pixel_format)
            TensorDescListEntry.AddPixelMapping(builder, td.pixel_mapping)
            TensorDescListEntry.AddN(builder, td.n)
            TensorDescListEntry.AddC(builder, td.c)
            TensorDescListEntry.AddH(builder, td.h)
            TensorDescListEntry.AddW(builder, td.w)
            TensorDescListEntry.AddStride0(builder, td.stride_0)
            TensorDescListEntry.AddStride1(builder, td.stride_1)
            TensorDescListEntry.AddStride2(builder, td.stride_2)
            TensorDescListEntry.AddStride3(builder, td.stride_3)
            TensorDescListEntry.AddStride4(builder, td.stride_4)
            TensorDescListEntry.AddStride5(builder, td.stride_5)
            TensorDescListEntry.AddStride6(builder, td.stride_6)
            TensorDescListEntry.AddStride7(builder, td.stride_7)
            td_offsets.append(TensorDescListEntry.End(builder))

        # ----------------------------------------------------------------------
        # 7. Serialize Reloc Entries
        # ----------------------------------------------------------------------
        reloc_offsets = []
        for r in model.relocs:
            RelocListEntry.Start(builder)
            RelocListEntry.AddAddressId(builder, r.address_id)
            RelocListEntry.AddWriteId(builder, r.write_id)
            RelocListEntry.AddOffset(builder, r.offset)
            RelocListEntry.AddInterface(builder, r.interface)
            RelocListEntry.AddSubInterface(builder, r.sub_interface)
            RelocListEntry.AddRelocType(builder, r.reloc_type)
            reloc_offsets.append(RelocListEntry.End(builder))

        # ----------------------------------------------------------------------
        # 8. Serialize Submit Entries
        # ----------------------------------------------------------------------
        submit_offsets = []
        for s in model.submits:
            task_id_vec = 0
            if s.task_id:
                SubmitListEntry.StartTaskIdVector(builder, len(s.task_id))
                for tid in reversed(s.task_id):
                    builder.PrependUint16(tid)
                task_id_vec = builder.EndVector()

            SubmitListEntry.Start(builder)
            SubmitListEntry.AddId(builder, s.id)
            if task_id_vec != 0:
                SubmitListEntry.AddTaskId(builder, task_id_vec)
            submit_offsets.append(SubmitListEntry.End(builder))

        # ----------------------------------------------------------------------
        # 9. Build Root Vector Offsets
        # ----------------------------------------------------------------------
        task_list_vec = 0
        if task_offsets:
            Loadable.StartTaskListVector(builder, len(task_offsets))
            for toff in reversed(task_offsets):
                builder.PrependUOffsetTRelative(toff)
            task_list_vec = builder.EndVector()

        mem_list_vec = 0
        if mem_offsets:
            Loadable.StartMemoryListVector(builder, len(mem_offsets))
            for moff in reversed(mem_offsets):
                builder.PrependUOffsetTRelative(moff)
            mem_list_vec = builder.EndVector()

        addr_list_vec = 0
        if addr_offsets:
            Loadable.StartAddressListVector(builder, len(addr_offsets))
            for aoff in reversed(addr_offsets):
                builder.PrependUOffsetTRelative(aoff)
            addr_list_vec = builder.EndVector()

        ev_list_vec = 0
        if ev_offsets:
            Loadable.StartEventListVector(builder, len(ev_offsets))
            for eoff in reversed(ev_offsets):
                builder.PrependUOffsetTRelative(eoff)
            ev_list_vec = builder.EndVector()

        blobs_vec = 0
        if blob_offsets:
            Loadable.StartBlobsVector(builder, len(blob_offsets))
            for boff in reversed(blob_offsets):
                builder.PrependUOffsetTRelative(boff)
            blobs_vec = builder.EndVector()

        td_list_vec = 0
        if td_offsets:
            Loadable.StartTensorDescListVector(builder, len(td_offsets))
            for tdoff in reversed(td_offsets):
                builder.PrependUOffsetTRelative(tdoff)
            td_list_vec = builder.EndVector()

        reloc_list_vec = 0
        if reloc_offsets:
            Loadable.StartRelocListVector(builder, len(reloc_offsets))
            for roff in reversed(reloc_offsets):
                builder.PrependUOffsetTRelative(roff)
            reloc_list_vec = builder.EndVector()

        submit_list_vec = 0
        if submit_offsets:
            Loadable.StartSubmitListVector(builder, len(submit_offsets))
            for soff in reversed(submit_offsets):
                builder.PrependUOffsetTRelative(soff)
            submit_list_vec = builder.EndVector()

        # ----------------------------------------------------------------------
        # 10. Assemble Root Loadable Table
        # ----------------------------------------------------------------------
        Loadable.Start(builder)

        # Inline Version Struct
        v_offset = Version.CreateVersion(
            builder,
            model.version.major,
            model.version.minor,
            model.version.sub_minor
        )
        Loadable.AddVersion(builder, v_offset)

        if task_list_vec != 0:
            Loadable.AddTaskList(builder, task_list_vec)
        if mem_list_vec != 0:
            Loadable.AddMemoryList(builder, mem_list_vec)
        if addr_list_vec != 0:
            Loadable.AddAddressList(builder, addr_list_vec)
        if ev_list_vec != 0:
            Loadable.AddEventList(builder, ev_list_vec)
        if blobs_vec != 0:
            Loadable.AddBlobs(builder, blobs_vec)
        if td_list_vec != 0:
            Loadable.AddTensorDescList(builder, td_list_vec)
        if reloc_list_vec != 0:
            Loadable.AddRelocList(builder, reloc_list_vec)
        if submit_list_vec != 0:
            Loadable.AddSubmitList(builder, submit_list_vec)

        root = Loadable.End(builder)

        # Finish with file identifier 'NVDA'
        builder.Finish(root, file_identifier=b"NVDA")
        return bytes(builder.Output())
