"""
FlatBuffers .model loadable parser and metadata reader.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import sys

# Ensure schema path is in sys.path
SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"
if str(SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEMA_DIR))

from eswin.loadable.Loadable import Loadable
from eswin_simulator.loader.tensor_desc import TensorDescriptor


MAGIC_NVDA = b"NVDA"

MEMORY_DOMAIN_NAMES = {
    0: "SYSTEM",
    1: "SRAM",
}

INTERFACE_NAMES = {
    0: "NONE",
    1: "UMD",
    2: "KMD",
}

# Bitmask flags from loadable.fbs
FLAG_ALLOC = 1
FLAG_SET = 2
FLAG_INPUT = 4
FLAG_OUTPUT = 8
FLAG_CACHE = 16
FLAG_NO_CACHE = 32


@dataclass
class MemoryRegion:
    """Represents a virtual/hardware memory allocation descriptor."""

    id: int
    domain_id: int
    flags: int
    size: int
    alignment: int
    contents: List[str] = field(default_factory=list)
    offsets: List[int] = field(default_factory=list)
    bind_id: int = 0
    tensor_desc_id: int = 0

    @property
    def domain(self) -> str:
        return MEMORY_DOMAIN_NAMES.get(self.domain_id, f"UNKNOWN({self.domain_id})")

    @property
    def is_alloc(self) -> bool:
        return bool(self.flags & FLAG_ALLOC)

    @property
    def is_input(self) -> bool:
        return bool(self.flags & FLAG_INPUT)

    @property
    def is_output(self) -> bool:
        return bool(self.flags & FLAG_OUTPUT)

    @property
    def is_cached(self) -> bool:
        return bool(self.flags & FLAG_CACHE)


@dataclass
class AddressEntry:
    """Represents an address entry mapping an address ID to a memory region and offset."""

    id: int
    mem_id: int
    offset: int
    size: int


@dataclass
class TaskEntry:
    """Represents a submodel task partition."""

    id: int
    kmd_id: int
    interface_id: int
    address_list: List[int] = field(default_factory=list)

    @property
    def interface(self) -> str:
        return INTERFACE_NAMES.get(self.interface_id, f"UNKNOWN({self.interface_id})")


@dataclass
class BlobEntry:
    """Represents an embedded binary blob (weights, command streams, lookup tables)."""

    name: str
    size: int
    interface_id: int
    data: bytes

    @property
    def interface(self) -> str:
        return INTERFACE_NAMES.get(self.interface_id, f"UNKNOWN({self.interface_id})")


@dataclass
class RelocEntry:
    """Represents an IOVA address relocation patching record."""

    address_id: int
    write_id: int
    offset: int
    interface_id: int
    sub_interface: int
    reloc_type: int


@dataclass
class SubmitEntry:
    """Represents an execution queue submission task set."""

    id: int
    task_ids: List[int] = field(default_factory=list)


class LoadableModel:
    """Parses and manages an ESWIN .model FlatBuffers container."""

    def __init__(self, raw_bytes: bytes, file_path: Optional[str] = None):
        self._raw_bytes = raw_bytes
        self.file_path = file_path
        self._validate_header()
        self._parse()

    def _validate_header(self) -> None:
        if len(self._raw_bytes) < 8:
            raise ValueError(f"File too small ({len(self._raw_bytes)} bytes) to be an ESWIN .model loadable.")
        magic = self._raw_bytes[4:8]
        if magic != MAGIC_NVDA:
            raise ValueError(f"Invalid file identifier {magic!r}. Expected {MAGIC_NVDA!r} ('NVDA').")

    def _parse(self) -> None:
        self._root = Loadable.GetRootAsLoadable(self._raw_bytes, 0)

        # Version
        ver = self._root.Version()
        if ver is not None:
            self.version = (ver.Major(), ver.Minor(), ver.SubMinor())
        else:
            self.version = (0, 0, 0)

        # Memory List
        self.memory_regions: Dict[int, MemoryRegion] = {}
        for i in range(self._root.MemoryListLength()):
            m = self._root.MemoryList(i)
            contents = [m.Contents(j).decode("utf-8") for j in range(m.ContentsLength())]
            offsets = [m.Offsets(j) for j in range(m.OffsetsLength())]
            reg = MemoryRegion(
                id=m.Id(),
                domain_id=m.Domain(),
                flags=m.Flags(),
                size=m.Size(),
                alignment=m.Alignment(),
                contents=contents,
                offsets=offsets,
                bind_id=m.BindId(),
                tensor_desc_id=m.TensorDescId(),
            )
            self.memory_regions[reg.id] = reg

        # Address List
        self.address_entries: Dict[int, AddressEntry] = {}
        for i in range(self._root.AddressListLength()):
            a = self._root.AddressList(i)
            self.address_entries[a.Id()] = AddressEntry(
                id=a.Id(),
                mem_id=a.MemId(),
                offset=a.Offset(),
                size=a.Size(),
            )

        # Tensor Descriptors
        self.tensors: List[TensorDescriptor] = []
        for i in range(self._root.TensorDescListLength()):
            td = self._root.TensorDescList(i)
            name = td.Name().decode("utf-8") if td.Name() else f"tensor_{td.Id()}"
            strides = (
                td.Stride0(),
                td.Stride1(),
                td.Stride2(),
                td.Stride3(),
                td.Stride4(),
                td.Stride5(),
                td.Stride6(),
                td.Stride7(),
            )
            mem_reg = self.memory_regions.get(td.MemId())
            is_input = mem_reg.is_input if mem_reg else False
            is_output = mem_reg.is_output if mem_reg else False
            domain_name = mem_reg.domain if mem_reg else "SYSTEM"

            desc = TensorDescriptor(
                name=name,
                id=td.Id(),
                mem_id=td.MemId(),
                size=td.Size(),
                offset=td.Offset(),
                data_format_id=td.DataFormat(),
                data_type_id=td.DataType(),
                data_category_id=td.DataCategory(),
                pixel_format_id=td.PixelFormat(),
                pixel_mapping_id=td.PixelMapping(),
                n=td.N(),
                c=td.C(),
                h=td.H(),
                w=td.W(),
                strides=strides,
                is_input=is_input,
                is_output=is_output,
                memory_domain=domain_name,
            )
            self.tensors.append(desc)

        # Blobs
        self.blobs: Dict[str, BlobEntry] = {}
        for i in range(self._root.BlobsLength()):
            b = self._root.Blobs(i)
            bname = b.Name().decode("utf-8") if b.Name() else f"blob_{i}"
            # Extract raw binary slice
            data_arr = b.DataAsNumpy()
            raw_data = data_arr.tobytes() if data_arr is not None else b""
            self.blobs[bname] = BlobEntry(
                name=bname,
                size=b.Size(),
                interface_id=b.Interface(),
                data=raw_data,
            )

        # Tasks / Submodels
        self.tasks: List[TaskEntry] = []
        for i in range(self._root.TaskListLength()):
            t = self._root.TaskList(i)
            addr_list = [t.AddressList(j) for j in range(t.AddressListLength())]
            self.tasks.append(
                TaskEntry(
                    id=t.Id(),
                    kmd_id=t.KmdId(),
                    interface_id=t.Interface(),
                    address_list=addr_list,
                )
            )

        # Relocations
        self.relocations: List[RelocEntry] = []
        for i in range(self._root.RelocListLength()):
            r = self._root.RelocList(i)
            self.relocations.append(
                RelocEntry(
                    address_id=r.AddressId(),
                    write_id=r.WriteId(),
                    offset=r.Offset(),
                    interface_id=r.Interface(),
                    sub_interface=r.SubInterface(),
                    reloc_type=r.RelocType(),
                )
            )

        # Submits
        self.submits: List[SubmitEntry] = []
        for i in range(self._root.SubmitListLength()):
            s = self._root.SubmitList(i)
            task_ids = [s.TaskId(j) for j in range(s.TaskIdLength())]
            self.submits.append(SubmitEntry(id=s.Id(), task_ids=task_ids))

    @property
    def input_tensors(self) -> List[TensorDescriptor]:
        """Returns all input tensor descriptors."""
        return [t for t in self.tensors if t.is_input]

    @property
    def output_tensors(self) -> List[TensorDescriptor]:
        """Returns all output tensor descriptors."""
        return [t for t in self.tensors if t.is_output]

    def get_blob(self, name: str) -> Optional[bytes]:
        """Returns raw bytes of an embedded blob, or None if not found."""
        entry = self.blobs.get(name)
        return entry.data if entry else None

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "LoadableModel":
        """Loads and parses a .model file from disk."""
        p = Path(path)
        with open(p, "rb") as f:
            data = f.read()
        return cls(data, file_path=str(p))

    def summary(self) -> Dict[str, Any]:
        """Returns a structured dictionary summary of the loadable model."""
        return {
            "file": self.file_path,
            "version": f"{self.version[0]}.{self.version[1]}.{self.version[2]}",
            "inputs": [
                {
                    "name": t.name,
                    "shape": t.shape,
                    "type": t.data_type,
                    "format": t.data_format,
                    "size": t.size,
                    "domain": t.memory_domain,
                }
                for t in self.input_tensors
            ],
            "outputs": [
                {
                    "name": t.name,
                    "shape": t.shape,
                    "type": t.data_type,
                    "format": t.data_format,
                    "size": t.size,
                    "domain": t.memory_domain,
                }
                for t in self.output_tensors
            ],
            "memory_regions_count": len(self.memory_regions),
            "address_entries_count": len(self.address_entries),
            "blobs_count": len(self.blobs),
            "tasks_count": len(self.tasks),
            "relocations_count": len(self.relocations),
        }
