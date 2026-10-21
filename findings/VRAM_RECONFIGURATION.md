# NPU Shared VRAM (MMZ) Architecture & Reconfiguration Guide

This document explains the hardware and device-tree memory layout for the dual-die ESWIN EIC7700X / EIC7702 SoC (board: **Framework / DeepComputing FML13V03**) and provides step-by-step instructions for reallocating system RAM to NPU shared VRAM (MMZ).

---

## 1. Physical Memory & NUMA Topography

The FML13V03 board features a dual-die multi-chip module (MCM):
- **Die 0 (NUMA Node 0)**: 16 GB Physical LPDDR5
- **Die 1 (NUMA Node 1)**: 16 GB Physical LPDDR5
- **Total Physical Memory**: 32 GB across both dies.

### Current Factory Allocation:
| Region | NUMA Node | Physical Base Address | Size | Mapping / Driver |
| :--- | :--- | :--- | :--- | :--- |
| **Linux OS & Userland** | Node 0 | `0x0000000080000000` | ~4.5 GB | Standard Linux `ZONE_NORMAL` |
| **Linux OS & Userland** | Node 1 | `0x0000002080000000` | ~4.5 GB | Standard Linux `ZONE_NORMAL` |
| **NPU MMZ 0 (`mmz_nid_0_part_0`)** | Node 0 | `0x00000001c0000000` | **11,264 MiB (11 GB)** | ESWIN MMZ `no-map` non-reusable |
| **NPU MMZ 1 (`mmz_nid_1_part_0`)** | Node 1 | `0x0000002140000000` | **11,264 MiB (11 GB)** | ESWIN MMZ `no-map` non-reusable |
| **Linux CMA per NUMA** | Nodes 0 & 1 | Dynamic | 512 MiB per node | `cma_pernuma=0x20000000` |

Total NPU Shared VRAM across both dies: **22 GB (11 GB Die 0 + 11 GB Die 1)**.
Total Linux OS RAM across both dies: **~9.6 GB usable** (`free -m`).

---

## 2. Boot & Device Tree Chain

The kernel command line and boot configuration on board `roma`:
1. **Bootloader**: U-Boot EFI stub invokes GRUB (`/boot/grub/grub.cfg`).
2. **Device Tree Loading**: GRUB explicitly loads the Device Tree Blob from `/boot/dtb-6.6.92-eic7x-2025.07`:
   ```grub
   linux   /boot/vmlinuz-6.6.92-eic7x-2025.07 root=UUID=... ro quiet splash console=tty0 console=ttys0,115200n8
   initrd  /boot/initrd.img-6.6.92-eic7x-2025.07
   devicetree /boot/dtb-6.6.92-eic7x-2025.07
   ```
   Note that `/boot/dtb-6.6.92-eic7x-2025.07` is a symlink pointing to:
   `/boot/dtbs/6.6.92-eic7x-2025.07/eswin/eic7702-deepcomputing-fml13v03.dtb`.

3. **Kernel Reservation**: During early boot, the kernel parses `/reserved-memory`:
   ```
   [    0.000000] Reserved memory: created mmz_nid_0_part_0@1,c0000000 eswin reserve memory at 0x00000001c0000000, size 11264 MiB
   [    0.000000] OF: reserved mem: 0x00000001c0000000..0x000000047fffffff (11534336 KiB) nomap non-reusable mmz_nid_0_part_0@1,c0000000
   [    0.000000] Reserved memory: created mmz_nid_1_part_0@21,40000000 eswin reserve memory at 0x0000002140000000, size 11264 MiB
   [    0.000000] OF: reserved mem: 0x0000002140000000..0x00000023ffffffff (11534336 KiB) nomap non-reusable mmz_nid_1_part_0@21,40000000
   ```
4. **Driver Attachment**: The `mmz_vb` driver queries the DTB and mounts `/dev/eswin_mmz` memory pools:
   ```
   [    9.169320] __add_eswin_rsvmem_heap for mmz_nid_0_part_0 successfully!
   [    9.176388] __add_eswin_rsvmem_heap for mmz_nid_1_part_0 successfully!
   ```

---

## 3. How to Adjust MMZ Shared VRAM

If you wish to increase NPU VRAM beyond 11 GB (e.g. to 13 GB per die, leaving ~3 GB per die / 6 GB total for headless Linux), follow these steps:

### Step 1: Backup Existing DTB
```bash
sudo cp /boot/dtbs/6.6.92-eic7x-2025.07/eswin/eic7702-deepcomputing-fml13v03.dtb \
        /boot/dtbs/6.6.92-eic7x-2025.07/eswin/eic7702-deepcomputing-fml13v03.dtb.bak
```

### Step 2: Decompile DTB to DTS Source
```bash
sudo dtc -I dtb -O dts -o /boot/fml13v03.dts \
  /boot/dtbs/6.6.92-eic7x-2025.07/eswin/eic7702-deepcomputing-fml13v03.dtb
```

### Step 3: Edit Reserved Memory Nodes
Open `/boot/fml13v03.dts` and locate the `reserved-memory` section:
```dts
mmz_nid_0_part_0 {
    compatible = "eswin-reserve-memory";
    reg = <0x01 0xc0000000 0x02 0xc0000000>;  /* base=0x1c0000000, size=11 GB */
    no-map;
};

mmz_nid_1_part_0 {
    compatible = "eswin-reserve-memory";
    reg = <0x21 0x40000000 0x02 0xc0000000>;  /* base=0x2140000000, size=11 GB */
    no-map;
};
```

> [!IMPORTANT]
> The `reg` cells follow `<address_high address_low size_high size_low>`.
> - Current 11 GB size: `0x02 0xc0000000` ($11 \times 1024^3 = 11,811,160,064 = \text{0x2C0000000}$).
> - To allocate **12.5 GB**: `size = 0x03 0x20000000`. Adjust base address so that `base + size` does not exceed physical die memory limit (`0x0480000000` on Die 0, `0x2400000000` on Die 1).
> - Keep at least 2.5 GB – 3.0 GB per NUMA node for Linux page tables, kernel slab, and user processes.

### Step 4: Recompile and Install DTB
```bash
sudo dtc -I dts -O dtb -o /boot/dtbs/6.6.92-eic7x-2025.07/eswin/eic7702-deepcomputing-fml13v03.dtb /boot/fml13v03.dts
```

### Step 5: Reboot and Verify
```bash
sudo reboot
# After reboot:
sudo dmesg | grep -i mmz
free -m
```
The driver will automatically detect the expanded heap and expose the new memory capacity to `libes_runtime.so` / NPU worker pool.
