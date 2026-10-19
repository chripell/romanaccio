# EIC7700X Contiguous DMA Memory & Error Resolution Handbook

## 1. Physical Memory Architecture & MMZ Topography

The ESWIN EIC7700X SoC utilizes a dual-die NUMA hardware architecture featuring two identical dies (Die 0 and Die 1), each equipped with a dedicated NPU core, four Andes DSP cores, a Hardware Acceleration Engine (HAE), and dedicated high-bandwidth memory zones.

```
       +------------------------------------+------------------------------------+
       |               DIE 0                |               DIE 1                |
       +------------------------------------+------------------------------------+
Nodes  | NUMA Node 0 (Cores 0-3)            | NUMA Node 1 (Cores 4-7)            |
NPU    | /dev/npu0                          | /dev/npu1                          |
DSPs   | /dev/es-dsp0 .. /dev/es-dsp3       | /dev/es-dsp4 .. /dev/es-dsp7       |
HAE    | /dev/es_hae                        | /dev/es_hae1                       |
SPRAM  | llc_spram0 (4 MB fast SRAM)        | llc_spram1 (4 MB fast SRAM)        |
MMZ    | mmz_nid_0_part_0 (11,264 MB DMA)   | mmz_nid_1_part_0 (11,264 MB DMA)   |
Heap   | /dev/dma_heap/mmz_nid_0_part_0     | /dev/dma_heap/mmz_nid_1_part_0     |
       +------------------------------------+------------------------------------+
```

### Contiguous Memory Allocation (MMZ)
The NPU fixed-function hardware and SMMU cannot directly execute from arbitrary virtual or fragmented physical userspace memory. All tensor inputs, outputs, KV-caches, and activation ping-pong buffers must reside within physically contiguous memory allocated from the kernel Media Memory Zone (`mmz`).

- **Allocator Library**: `libes_memory.so` (dynamically loaded via `dlopen`).
  > **Crucial Discovery**: `ES_SYS_MemAlloc` is located inside `/lib/libes_memory.so`, **not** `/lib/libes_sys.so.1`. Attempting to resolve `ES_SYS_MemAlloc` from `libes_sys.so` causes `dlsym` to return `nullptr`.
- **Heap Interface**: Internally communicates via Linux DMA-BUF heap `/dev/dma_heap/mmz_nid_0_part_0` or `/dev/dma_heap/mmz_nid_1_part_0`.
- **Memory File Descriptor (`memFd`)**: A 64-bit integer handle referencing the allocated DMA-BUF in kernel space.

---

## 2. DMA Buffer Lifecycle & SMMU Preparation

To ensure zero-copy execution and cache coherency, buffers follow an explicit 8-stage lifecycle managed through RAII in `eswin::DmaBuffer` and the C11 handle `es_dma_buf_t`:

```
   1. Allocate (ES_SYS_MemAlloc)
        │
   2. Map Userspace Virtual Address (ES_SYS_Mmap)
        │
   3. Host CPU Writes Tensor Data
        │
   4. Flush CPU Data Cache (ES_SYS_MemFlushCache) [if cached]
        │
   5. SMMU Prepare (ES_NPU_Prepare -> ioctl 0xc0046e0f)
        │
   6. Hardware Task Submission (ES_NPU_Submit / ES_NPU_SubmitFlexibleTask)
        │
   7. SMMU Unprepare (ES_NPU_Unprepare)
        │
   8. Unmap & Free (ES_SYS_Munmap + ES_SYS_MemFree)
```

### Thread Affinity & SMMU Mapping Rules
1. **Device Context Initialization**:
   - Before any SMMU operations (`ES_NPU_Prepare`) or model execution can occur, `ES_NPU_GetNumDevices(&count)` must be called once during runtime startup. This call initializes internal device manager descriptors and thread-local state.
2. **Current Device Context Requirement**:
   - `ES_NPU_Prepare(&dev_buf)` queries the thread-local runtime context (`esCtxLocal` or `esCtxDefault`).
   - If `ES_NPU_SetDevice(deviceId)` has not been called on the calling thread, `ES_NPU_Prepare` immediately aborts with error `0xA00F604A` (`ES_NPU_ERROR_CONTEXT_NOT_FOUND`).
   - `DmaBuffer::prepare(uint16_t device_id)` automatically calls `set_device(device_id)` prior to invoking the SMMU mapping routine.

### Device Node Permissions (`/dev/npu*`, `/dev/mmz_vb`)
- In default vendor firmware, `/etc/udev/rules.d/86-ai.rules` and `20-memory.rules` specified `GROUP="eswin", OWNER="eswin"` with mode `0660`. Because the `eswin` user and group do not exist in standard user accounts, the kernel udev engine fell back to `root:root` with mode `0660`.
- Consequently, unprivileged non-root users (such as `chri`) attempting to run inference or map DMA buffers encountered `openat("/dev/npu0") = -1 EACCES (Permission denied)` and `open("/dev/mmz_vb") failed!`, causing `ES_NPU_Prepare` to fail with error `0xFFFFFFFF`.
- To allow standard non-root execution without requiring `sudo`, configure `/etc/udev/rules.d/86-ai.rules` and `/etc/udev/rules.d/20-memory.rules` with mode `0666`:
  ```bash
  # /etc/udev/rules.d/86-ai.rules
  KERNEL=="es-dsp[0-9]*", MODE="0666"
  KERNEL=="npu[0-9]*", MODE="0666"
  KERNEL=="es_hae*", MODE="0666"
  KERNEL=="mmz_nid_*", MODE="0666"
  KERNEL=="eswin_ipc_*", MODE="0666"

  # /etc/udev/rules.d/20-memory.rules
  KERNEL=="linux,cma", MODE="0666"
  KERNEL=="llc_spram[0-1]", MODE="0666"
  KERNEL=="mmz_nid_[0-1]_part_[0-1]", MODE="0666"
  KERNEL=="system", MODE="0666"
  KERNEL=="mmz_vb", MODE="0666"
  KERNEL=="malloc_dmabuf", MODE="0666"
  KERNEL=="perf_count", MODE="0666"
  ```
  Apply with: `sudo udevadm control --reload-rules && sudo udevadm trigger`.

---

## 3. Error Code Resolution Handbook

The ESWIN software stack returns 32-bit packed status codes formatted as:
`0xA00[Module][ErrorID]`

```
  31         24 23       16 15        13 12                      0
 +-------------+-----------+------------+-------------------------+
 | Prefix: 0xA0| Module ID | Reserved   | Error Code (13 bits)    |
 +-------------+-----------+------------+-------------------------+
```

### Module IDs
- `0x0E` (14): System & Memory Management Subsystem (`libes_sys.so`, `libes_memory.so`).
- `0x0F` (15): NPU Runtime Subsystem (`libedla_runtime.so`).

### NPU Runtime Errors (`Module 0x0F`)

| Error Code | Decimal ID | Symbolic Name | Root Cause | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| `0xA00F6001` | 1 | `ES_NPU_ERROR_INVALID_PARAM` | Invalid argument or null pointer passed to API. | Check pointers, offsets, and tensor descriptors. |
| `0xA00F6002` | 2 | `ES_NPU_ERROR_NO_MEMORY` | MMZ or host memory allocation exhausted. | Free unused DMA buffers or allocate across alternate NUMA node. |
| `0xA00F6048` | 72 | `ES_NPU_ERROR_MODEL_NOT_PRESENT` | Model descriptor lookup failed; device manager table not initialized. | Ensure `ES_NPU_GetNumDevices` is called before loading models. |
| `0xA00F604A` | 74 | `ES_NPU_ERROR_CONTEXT_NOT_FOUND` | No active device context associated with calling thread. | Invoke `ES_NPU_SetDevice(deviceId)` before `ES_NPU_Prepare`. |
| `0xA00F6050` | 80 | `ES_NPU_ERROR_DEVICE_NOT_READY` | Device hardware engine is unresponsive or in reset. | Check `sudo dmesg` for hardware timeouts; reboot if needed. |
| `0xA00F6064` | 100 | `ES_NPU_ERROR_TASK_TIMEOUT` | Hardware execution exceeded deadline. | Inspect model complexity, reduce batch size, or check DSP sync. |
| `0xFFFFFFFF` | 8191 | `ES_NPU_ERROR_IOCTL_FAILED` | Kernel ioctl returned `-1` (e.g. `EACCES` or `EINVAL`). | Check device permissions (`/dev/npu0`) or buffer alignment. |

### System & Memory Errors (`Module 0x0E`)

| Error Code | Decimal ID | Symbolic Name | Root Cause | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| `0xA00E6001` | 1 | `ES_SYS_ERROR_INVALID_PARAM` | Size 0 or invalid cache mode passed to allocator. | Ensure buffer size is positive and page-aligned. |
| `0xA00E6002` | 2 | `ES_SYS_ERROR_ALLOC_FAILED` | Kernel MMZ carveout out of contiguous space. | Free stale buffers or request smaller chunk size. |
| `0xA00E6005` | 5 | `ES_SYS_ERROR_MMAP_FAILED` | Userspace `mmap` failed on DMA-BUF file descriptor. | Check file descriptor validity and process memory limits (`ulimit -v`). |
