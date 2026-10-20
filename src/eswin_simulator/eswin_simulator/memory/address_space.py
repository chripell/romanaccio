"""
Simulated virtual address space for DRAM (SYSTEM) and on-chip 4 MiB SPRAM (SRAM).
Includes dynamic relocation resolver and tensor binary file I/O.
"""

from pathlib import Path
import struct
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from eswin_simulator.loader.loadable_reader import LoadableModel, MemoryRegion, AddressEntry
from eswin_simulator.loader.tensor_desc import TensorDescriptor
from eswin_simulator.memory.buffer_view import create_tensor_view


SPRAM_CAPACITY_BYTES = 4 * 1024 * 1024  # 4 MiB on-chip static RAM
DRAM_BASE_ADDRESS = 0x8000_0000_0000    # Simulated 64-bit IOVA base for DRAM
SPRAM_BASE_ADDRESS = 0x1000_0000_0000   # Simulated base address for SPRAM


class AllocatedBuffer:
    """Represents a concrete allocated memory buffer in simulated space."""

    def __init__(self, region: MemoryRegion, base_vaddr: int):
        self.region = region
        self.base_vaddr = base_vaddr
        self.size = region.size
        # Allocate zero-initialized memory buffer
        self.data = bytearray(region.size)

    def write(self, offset: int, data: bytes) -> None:
        """Writes raw bytes into buffer at offset."""
        end = offset + len(data)
        if end > len(self.data):
            raise IndexError(
                f"Buffer overflow in region {self.region.id}: write of {len(data)}B at offset {offset} "
                f"exceeds region size {len(self.data)}B."
            )
        self.data[offset:end] = data

    def read(self, offset: int, size: int) -> bytes:
        """Reads raw bytes from buffer at offset."""
        end = offset + size
        if end > len(self.data):
            raise IndexError(
                f"Buffer overread in region {self.region.id}: read of {size}B at offset {offset} "
                f"exceeds region size {len(self.data)}B."
            )
        return bytes(self.data[offset:end])


class VirtualAddressSpace:
    """Simulates 64-bit DRAM and on-chip 4 MiB SPRAM for ESWIN EIC7700X."""

    def __init__(self, loadable: Optional[LoadableModel] = None):
        # 4 MiB on-chip SPRAM shared buffer
        self.spram_buffer = bytearray(SPRAM_CAPACITY_BYTES)
        self.spram_bytes_used = 0

        # Memory regions dictionary: mem_id -> AllocatedBuffer
        self.regions: Dict[int, AllocatedBuffer] = {}
        # Address resolution dictionary: address_id -> AddressEntry
        self.address_table: Dict[int, AddressEntry] = {}
        # Fast lookup list for 64-bit virtual addresses: [(base_vaddr, end_vaddr, AllocatedBuffer)]
        self._vaddr_map: List[Tuple[int, int, AllocatedBuffer]] = []

        self.relocations_applied = 0
        self.loadable = loadable
        if loadable is not None:
            self.initialize_from_loadable(loadable)

    def initialize_from_loadable(self, loadable: LoadableModel) -> None:
        """Allocates all memory regions, resolves static contents, and applies relocations."""
        self.loadable = loadable
        self.address_table = dict(loadable.address_entries)
        self._vaddr_map.clear()
        self.spram_bytes_used = 0

        current_dram_addr = DRAM_BASE_ADDRESS

        # 1. Allocate buffers for all memory regions
        for mem_id, reg in loadable.memory_regions.items():
            align = max(1, reg.alignment)

            if reg.domain_id == 1:  # SRAM (On-chip SPRAM)
                # Align allocation inside SPRAM
                rem = self.spram_bytes_used % align
                if rem != 0:
                    self.spram_bytes_used += (align - rem)
                vaddr = SPRAM_BASE_ADDRESS + self.spram_bytes_used

                buf = AllocatedBuffer(reg, vaddr)
                self.regions[mem_id] = buf
                self.spram_bytes_used += reg.size

                if self.spram_bytes_used > SPRAM_CAPACITY_BYTES:
                    raise MemoryError(
                        f"Simulated SPRAM overflow: {self.spram_bytes_used} bytes requested, "
                        f"exceeds 4 MiB ({SPRAM_CAPACITY_BYTES} bytes)."
                    )
                self._vaddr_map.append((vaddr, vaddr + reg.size, buf))
            else:  # SYSTEM (DRAM)
                # Align DRAM address
                rem = current_dram_addr % align
                if rem != 0:
                    current_dram_addr += (align - rem)
                vaddr = current_dram_addr
                buf = AllocatedBuffer(reg, vaddr)
                self.regions[mem_id] = buf
                current_dram_addr += reg.size
                self._vaddr_map.append((vaddr, vaddr + reg.size, buf))

        # 2. Populate static blob contents (weights, registers, LUTs)
        for mem_id, reg in loadable.memory_regions.items():
            buf = self.regions[mem_id]
            for blob_name, offset in zip(reg.contents, reg.offsets):
                blob_bytes = loadable.get_blob(blob_name)
                if blob_bytes:
                    buf.write(offset, blob_bytes)

        # 3. Resolve and apply dynamic relocations
        self.apply_relocations()

    def apply_relocations(self) -> int:
        """Parses RelocEntry records and patches 64-bit IOVA addresses into target buffers."""
        if not self.loadable or not self.loadable.relocations:
            return 0

        count = 0
        for reloc in self.loadable.relocations:
            # 1. Source virtual address
            src_vaddr = self.get_virtual_address(reloc.address_id)

            # 2. Destination buffer: write_id corresponds to target memory region
            target_mem_id = reloc.write_id
            if target_mem_id not in self.regions:
                # If write_id is an address ID, resolve to mem_id
                if target_mem_id in self.address_table:
                    target_mem_id = self.address_table[target_mem_id].mem_id

            target_buf = self.regions.get(target_mem_id)
            if target_buf is None:
                continue

            # 3. Patch address based on reloc_type
            if reloc.reloc_type == 1:
                # Type 1: Lower 32 bits of 64-bit address (little-endian uint32)
                val_u32 = src_vaddr & 0xFFFF_FFFF
                patch_bytes = struct.pack("<I", val_u32)
                target_buf.write(reloc.offset, patch_bytes)
                count += 1
            elif reloc.reloc_type == 2:
                # Type 2: Upper 32 bits of 64-bit address (little-endian uint32)
                val_u32 = (src_vaddr >> 32) & 0xFFFF_FFFF
                patch_bytes = struct.pack("<I", val_u32)
                target_buf.write(reloc.offset, patch_bytes)
                count += 1

        self.relocations_applied = count
        return count

    def get_region_buffer(self, mem_id: int) -> AllocatedBuffer:
        """Returns the AllocatedBuffer for a given memory region ID."""
        if mem_id not in self.regions:
            raise KeyError(f"Memory region ID {mem_id} not found in virtual address space.")
        return self.regions[mem_id]

    def resolve_address(self, address_id: int) -> Tuple[AllocatedBuffer, int, int]:
        """Resolves an address ID into (AllocatedBuffer, offset, size)."""
        if address_id not in self.address_table:
            raise KeyError(f"Address ID {address_id} not found in address table.")
        entry = self.address_table[address_id]
        buf = self.get_region_buffer(entry.mem_id)
        return buf, entry.offset, entry.size

    def get_virtual_address(self, address_id: int) -> int:
        """Returns the simulated 64-bit IOVA virtual address for an address ID."""
        buf, offset, _ = self.resolve_address(address_id)
        return buf.base_vaddr + offset

    def _find_buffer_for_vaddr(self, vaddr: int, size: int) -> Tuple[AllocatedBuffer, int]:
        """Maps a 64-bit virtual address to (AllocatedBuffer, offset)."""
        # Fast path check
        for base, end, buf in self._vaddr_map:
            if base <= vaddr < end:
                offset = vaddr - base
                if offset + size > buf.size:
                    raise IndexError(
                        f"Virtual memory access out of bounds: vaddr 0x{vaddr:x} + {size}B "
                        f"exceeds region boundary 0x{end:x}."
                    )
                return buf, offset

        raise KeyError(f"Virtual address 0x{vaddr:x} does not map to any allocated memory region.")

    def read_vaddr_bytes(self, vaddr: int, size: int) -> bytes:
        """Reads raw bytes from a 64-bit virtual address."""
        buf, offset = self._find_buffer_for_vaddr(vaddr, size)
        return buf.read(offset, size)

    def write_vaddr_bytes(self, vaddr: int, data: bytes) -> None:
        """Writes raw bytes to a 64-bit virtual address."""
        buf, offset = self._find_buffer_for_vaddr(vaddr, len(data))
        buf.write(offset, data)

    def bind_input(self, tensor: TensorDescriptor, data: Union[np.ndarray, bytes, bytearray, memoryview]) -> None:
        """Binds input data to the specified tensor's memory location."""
        buf = self.get_region_buffer(tensor.mem_id)
        if isinstance(data, np.ndarray):
            raw_bytes = data.astype(tensor.dtype).tobytes()
        elif isinstance(data, (bytes, bytearray, memoryview)):
            raw_bytes = bytes(data)
        else:
            raise TypeError(f"Unsupported data type for bind_input: {type(data)}")

        if len(raw_bytes) > tensor.size:
            raise ValueError(
                f"Input data size ({len(raw_bytes)}B) exceeds tensor '{tensor.name}' capacity ({tensor.size}B)."
            )
        buf.write(tensor.offset, raw_bytes)

    def read_tensor(self, tensor: TensorDescriptor) -> np.ndarray:
        """Reads and reconstructs a NumPy ndarray from the tensor's memory location."""
        buf = self.get_region_buffer(tensor.mem_id)
        return create_tensor_view(
            buffer=buf.data,
            offset=tensor.offset,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )

    def write_bytes(self, address_id: int, offset: int, data: bytes) -> None:
        """Writes raw bytes to an address ID with relative offset."""
        buf, base_off, max_size = self.resolve_address(address_id)
        if offset + len(data) > max_size:
            raise IndexError(f"Write exceeds allocated address {address_id} capacity ({max_size}B).")
        buf.write(base_off + offset, data)

    def read_bytes(self, address_id: int, offset: int, size: int) -> bytes:
        """Reads raw bytes from an address ID with relative offset."""
        buf, base_off, max_size = self.resolve_address(address_id)
        if offset + size > max_size:
            raise IndexError(f"Read exceeds allocated address {address_id} capacity ({max_size}B).")
        return buf.read(base_off + offset, size)

    # -------------------------------------------------------------------------
    # Tensor File I/O Operations (Milestone 2)
    # -------------------------------------------------------------------------

    def load_input_file(self, tensor: TensorDescriptor, file_path: Union[str, Path]) -> int:
        """Loads a raw binary (.bin) file directly into the input tensor's memory.

        Returns:
            Number of bytes loaded.
        """
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Input binary file not found: {file_path}")

        raw_bytes = p.read_bytes()
        self.bind_input(tensor, raw_bytes)
        return len(raw_bytes)

    def dump_output_file(self, tensor: TensorDescriptor, file_path: Union[str, Path]) -> int:
        """Dumps tensor data from simulated virtual memory into a raw binary (.bin) file.

        Returns:
            Number of bytes written.
        """
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        buf = self.get_region_buffer(tensor.mem_id)
        raw_bytes = buf.read(tensor.offset, tensor.size)
        p.write_bytes(raw_bytes)
        return len(raw_bytes)

    def load_inputs(self, input_paths: Union[List[Union[str, Path]], Dict[str, Union[str, Path]]]) -> Dict[str, int]:
        """Binds multiple input files to input tensors by index or by name."""
        if not self.loadable:
            raise RuntimeError("No loadable model associated with address space.")

        loaded = {}
        inputs = self.loadable.input_tensors

        if isinstance(input_paths, dict):
            # Keyed by tensor name
            for t in inputs:
                if t.name in input_paths:
                    bytes_loaded = self.load_input_file(t, input_paths[t.name])
                    loaded[t.name] = bytes_loaded
        elif isinstance(input_paths, list):
            # Positional list of paths
            for idx, path in enumerate(input_paths):
                if idx < len(inputs):
                    t = inputs[idx]
                    bytes_loaded = self.load_input_file(t, path)
                    loaded[t.name] = bytes_loaded

        return loaded

    def dump_outputs(self, output_dir: Union[str, Path], prefix: str = "") -> Dict[str, Path]:
        """Dumps all output tensors into the specified directory as .bin files."""
        if not self.loadable:
            raise RuntimeError("No loadable model associated with address space.")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dumped = {}

        for idx, t in enumerate(self.loadable.output_tensors):
            fname = f"{prefix}{t.name}.bin" if not prefix else f"{prefix}_{t.name}.bin"
            out_path = out_dir / fname
            self.dump_output_file(t, out_path)
            dumped[t.name] = out_path

        return dumped

    def stats(self) -> Dict[str, Any]:
        """Returns memory utilization statistics."""
        dram_total = sum(b.size for b in self.regions.values() if b.region.domain_id == 0)
        return {
            "dram_allocated_bytes": dram_total,
            "dram_allocated_mb": round(dram_total / (1024 * 1024), 2),
            "spram_used_bytes": self.spram_bytes_used,
            "spram_capacity_bytes": SPRAM_CAPACITY_BYTES,
            "spram_used_percent": round((self.spram_bytes_used / SPRAM_CAPACITY_BYTES) * 100, 2),
            "regions_count": len(self.regions),
            "addresses_count": len(self.address_table),
            "relocations_applied": self.relocations_applied,
        }
