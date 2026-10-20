"""
Bridge DMA (BDMA) Engine simulation for ESWIN EIC7700X.
Executes high-throughput multi-dimensional strided memory transfers (DRAM <-> SPRAM).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from eswin_simulator.engines.base_engine import HardwareEngine
from eswin_simulator.loader.tensor_desc import TensorDescriptor
from eswin_simulator.memory.address_space import (
    VirtualAddressSpace,
    SPRAM_BASE_ADDRESS,
    SPRAM_CAPACITY_BYTES,
)


@dataclass
class BDMATransferDesc:
    """Descriptor configuring a multi-dimensional BDMA strided transfer.

    Attributes:
        src_addr: 64-bit source base virtual address.
        dst_addr: 64-bit destination base virtual address.
        line_size: Number of contiguous bytes transferred per line.
        line_repeat: Number of lines per surface (dimension H).
        src_line_stride: Byte distance between consecutive source lines.
        dst_line_stride: Byte distance between consecutive destination lines.
        surf_repeat: Number of surfaces (dimension C / Batch).
        src_surf_stride: Byte distance between consecutive source surfaces.
        dst_surf_stride: Byte distance between consecutive destination surfaces.
    """

    src_addr: int
    dst_addr: int
    line_size: int
    line_repeat: int = 1
    src_line_stride: Optional[int] = None
    dst_line_stride: Optional[int] = None
    surf_repeat: int = 1
    src_surf_stride: Optional[int] = None
    dst_surf_stride: Optional[int] = None

    def __post_init__(self) -> None:
        if self.line_size <= 0:
            raise ValueError(f"BDMA line_size must be >= 1, got {self.line_size}")
        if self.line_repeat <= 0:
            raise ValueError(f"BDMA line_repeat must be >= 1, got {self.line_repeat}")
        if self.surf_repeat <= 0:
            raise ValueError(f"BDMA surf_repeat must be >= 1, got {self.surf_repeat}")

        # Set default strides to tightly packed contiguous layouts if omitted
        if self.src_line_stride is None:
            self.src_line_stride = self.line_size
        if self.dst_line_stride is None:
            self.dst_line_stride = self.line_size
        if self.src_surf_stride is None:
            self.src_surf_stride = self.line_repeat * self.src_line_stride
        if self.dst_surf_stride is None:
            self.dst_surf_stride = self.line_repeat * self.dst_line_stride

    @property
    def total_bytes(self) -> int:
        """Total payload bytes transferred."""
        return self.surf_repeat * self.line_repeat * self.line_size


class BDMAEngine(HardwareEngine):
    """Bridge DMA (BDMA) simulation engine for DRAM <-> SPRAM strided transfers."""

    def __init__(self, address_space: VirtualAddressSpace):
        super().__init__(name="BDMA", address_space=address_space)

    def reset(self) -> None:
        super().reset()

    def transfer(self, desc: BDMATransferDesc) -> int:
        """Executes a 1D, 2D, or 3D strided BDMA transfer.

        Mathematical model:
        For surface s in 0 .. (surf_repeat - 1):
            For line l in 0 .. (line_repeat - 1):
                src_off = s * src_surf_stride + l * src_line_stride
                dst_off = s * dst_surf_stride + l * dst_line_stride
                dst[dst_addr + dst_off : dst_addr + dst_off + line_size] =
                    src[src_addr + src_off : src_addr + src_off + line_size]

        Returns:
            Total payload bytes copied.
        """
        line_sz = desc.line_size
        bytes_copied = 0

        # Fast contiguous optimization: single linear transfer if strides are tightly packed
        is_contiguous_src = (desc.src_line_stride == line_sz) and (
            desc.src_surf_stride == desc.line_repeat * desc.src_line_stride
        )
        is_contiguous_dst = (desc.dst_line_stride == line_sz) and (
            desc.dst_surf_stride == desc.line_repeat * desc.dst_line_stride
        )

        if is_contiguous_src and is_contiguous_dst:
            total_sz = desc.total_bytes
            data = self.vas.read_vaddr_bytes(desc.src_addr, total_sz)
            self.vas.write_vaddr_bytes(desc.dst_addr, data)
            bytes_copied = total_sz
        else:
            # Multi-dimensional strided copy
            for s in range(desc.surf_repeat):
                src_s_base = desc.src_addr + s * desc.src_surf_stride
                dst_s_base = desc.dst_addr + s * desc.dst_surf_stride

                for l in range(desc.line_repeat):
                    src_line_addr = src_s_base + l * desc.src_line_stride
                    dst_line_addr = dst_s_base + l * desc.dst_line_stride

                    line_chunk = self.vas.read_vaddr_bytes(src_line_addr, line_sz)
                    self.vas.write_vaddr_bytes(dst_line_addr, line_chunk)
                    bytes_copied += line_sz

        self.total_operations += 1
        self.total_bytes_transferred += bytes_copied
        return bytes_copied

    def copy_linear(self, src_addr: int, dst_addr: int, size: int) -> int:
        """Convenience method for a 1D linear memory copy."""
        desc = BDMATransferDesc(
            src_addr=src_addr,
            dst_addr=dst_addr,
            line_size=size,
            line_repeat=1,
            surf_repeat=1,
        )
        return self.transfer(desc)

    def copy_2d(
        self,
        src_addr: int,
        dst_addr: int,
        width_bytes: int,
        height_lines: int,
        src_stride: int,
        dst_stride: int,
    ) -> int:
        """Convenience method for a 2D strided matrix copy."""
        desc = BDMATransferDesc(
            src_addr=src_addr,
            dst_addr=dst_addr,
            line_size=width_bytes,
            line_repeat=height_lines,
            src_line_stride=src_stride,
            dst_line_stride=dst_stride,
            surf_repeat=1,
        )
        return self.transfer(desc)

    def copy_3d(
        self,
        src_addr: int,
        dst_addr: int,
        line_size: int,
        line_repeat: int,
        surf_repeat: int,
        src_line_stride: int,
        dst_line_stride: int,
        src_surf_stride: int,
        dst_surf_stride: int,
    ) -> int:
        """Convenience method for a 3D strided volume copy."""
        desc = BDMATransferDesc(
            src_addr=src_addr,
            dst_addr=dst_addr,
            line_size=line_size,
            line_repeat=line_repeat,
            surf_repeat=surf_repeat,
            src_line_stride=src_line_stride,
            dst_line_stride=dst_line_stride,
            src_surf_stride=src_surf_stride,
            dst_surf_stride=dst_surf_stride,
        )
        return self.transfer(desc)

    def dram_to_spram(self, src_dram_addr: int, dst_spram_addr: int, size: int) -> int:
        """Transfers data from DRAM to on-chip SPRAM."""
        return self.copy_linear(src_dram_addr, dst_spram_addr, size)

    def spram_to_dram(self, src_spram_addr: int, dst_dram_addr: int, size: int) -> int:
        """Transfers data from on-chip SPRAM to DRAM."""
        return self.copy_linear(src_spram_addr, dst_dram_addr, size)

    def copy_tensor_to_spram(self, tensor: TensorDescriptor, spram_offset: int = 0) -> int:
        """Transfers an entire tensor from its allocated DRAM region into SPRAM."""
        src_buf = self.vas.get_region_buffer(tensor.mem_id)
        src_vaddr = src_buf.base_vaddr + tensor.offset
        dst_vaddr = SPRAM_BASE_ADDRESS + spram_offset
        return self.copy_linear(src_vaddr, dst_vaddr, tensor.size)

    def copy_tensor_from_spram(self, tensor: TensorDescriptor, spram_offset: int = 0) -> int:
        """Transfers an entire tensor from SPRAM into its allocated DRAM region."""
        src_vaddr = SPRAM_BASE_ADDRESS + spram_offset
        dst_buf = self.vas.get_region_buffer(tensor.mem_id)
        dst_vaddr = dst_buf.base_vaddr + tensor.offset
        return self.copy_linear(src_vaddr, dst_vaddr, tensor.size)
