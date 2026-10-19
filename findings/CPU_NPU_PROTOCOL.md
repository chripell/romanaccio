# ESWIN EIC7700X CPU <-> NPU / DSP Hardware Protocol Specification

## 1. Device Nodes & Character Interfaces

Communication between userspace and the hardware accelerator cluster is arbitrated through character devices and DMA-BUF heaps:

| Device Node | Subsystem | Permissions | Purpose |
| :--- | :--- | :--- | :--- |
| `/dev/npu0` | Die 0 NPU Core | `crw-rw---- root:root` | Command submission, model loading, SRAM allocation, SMMU mapping for Die 0. |
| `/dev/npu1` | Die 1 NPU Core | `crw-rw---- root:root` | Command submission, model loading, SRAM allocation, SMMU mapping for Die 1. |
| `/dev/dma_heap/mmz_nid_0_part_0` | Die 0 DMA Heap | `crw-rw-r-- root:root` | 11 GB physically contiguous CMA/MMZ carveout for Die 0 tensors. |
| `/dev/dma_heap/mmz_nid_1_part_0` | Die 1 DMA Heap | `crw-rw-r-- root:root` | 11 GB physically contiguous CMA/MMZ carveout for Die 1 tensors. |
| `/dev/eswin_ipc_misc_dev0` | Inter-Die Mailbox | `crw-rw---- root:root` | Hardware mailbox for cross-die peer synchronization (`_peer` models). |
| `/dev/eswin_ipc_misc_dev1` | Inter-Die Mailbox | `crw-rw---- root:root` | Die 1 peer mailbox endpoint. |
| `/dev/es-dsp0` .. `/dev/es-dsp7` | Cadence DSP Cores| `crw-rw---- root:root` | Low-level DSP task queue and shared memory dispatch. |
| `/dev/es_hae`, `/dev/es_hae1` | Hardware Accel Eng | `crw-rw---- root:root` | Fixed-function 2D image color conversion and resize engine. |

---

## 2. Kernel Driver ioctl Protocol (`eic7700_npu.ko`)

By disassembling `npu_dev_ioctl` in `eic7700_npu.ko` and call sites in `libedla_runtime.so.1.0`, we established the complete kernel ioctl dispatch table.
All NPU ioctls use magic type `'n'` (`0x6E`):

```c
#define NPU_IOCTL_MAGIC 'n'

#define NPU_IOCTL_SUBMIT_MODEL    _IOWR('n', 0x03, struct npu_model_desc)     // 0xc0046e03
#define NPU_IOCTL_UNLOAD_MODEL    _IOWR('n', 0x04, struct npu_model_desc)     // 0xc0046e04
#define NPU_IOCTL_SUBMIT_TASK     _IOWR('n', 0x05, struct npu_task_desc)      // 0xc0046e05
#define NPU_IOCTL_PROCESS_REPORT  _IOR ('n', 0x08, struct npu_report_desc)    // 0x80046e08
#define NPU_IOCTL_SUBMIT_OP_TASK  _IOWR('n', 0x09, struct npu_op_task_desc)   // 0xc0046e09
#define NPU_IOCTL_GET_SRAM        _IOR ('n', 0x0A, struct npu_sram_desc)      // 0x80046e0a
#define NPU_IOCTL_HANDLE_PERF     _IOR ('n', 0x0B, struct npu_perf_desc)      // 0x80046e0b
#define NPU_IOCTL_GET_PERF_DATA   _IOR ('n', 0x0C, struct npu_perf_data_desc) // 0x80046e0c
#define NPU_IOCTL_MUTEX_LOCK      _IOR ('n', 0x0D, uint32_t)                  // 0x80046e0d
#define NPU_IOCTL_MUTEX_UNLOCK    _IOWR('n', 0x0E, uint32_t)                  // 0xc0046e0e
#define NPU_IOCTL_PREPARE_DMA     _IOWR('n', 0x0F, struct ES_DEV_BUF)         // 0xc0046e0f
#define NPU_IOCTL_UNPREPARE_DMA   _IOWR('n', 0x10, struct ES_DEV_BUF)         // 0xc0046e10
#define NPU_IOCTL_MUTEX_TRYLOCK   _IOWR('n', 0x11, uint32_t)                  // 0xc0046e11
```

### 2.1 Memory Mapping Protocol (`NPU_IOCTL_PREPARE_DMA`)
Before any tensor buffer can be accessed by the NPU:
1. Userspace calls `ES_SYS_MemAlloc` which allocates from `/dev/dma_heap/mmz_nid_X_part_0` and receives a standard Linux DMA-BUF file descriptor `memFd`.
2. Userspace invokes `ES_NPU_Prepare(ES_DEV_BUF_S *devBuf)`.
3. `libedla_runtime` executes:
   ```c
   ioctl(npu_fd, 0xc0046e0f, devBuf); // NPU_IOCTL_PREPARE_DMA
   ```
4. **Kernel Side Action**:
   - Imports `dma_buf` from `devBuf->memFd`.
   - Attaches NPU device to DMA-BUF (`dma_buf_attach`).
   - Maps DMA-BUF to the NPU's SMMU page table (`dma_buf_map_attachment`).
   - Obtains the NPU Device Virtual Address (IOVA).
   - Writes the IOVA back into the driver's internal address lookup table.
5. In reverse, `ES_NPU_Unprepare` issues `0xc0046e10` (`NPU_IOCTL_UNPREPARE_DMA`) to unmap from SMMU and release DMA-BUF attachments.

---

## 3. Task Submission & Hardware Execution Pipeline

When `ES_NPU_Submit` is called:

```
[Userspace Application]
        |
        | ES_NPU_Submit(NPU_TASK_S*)
        v
[libedla_runtime.so]
        |
        | 1. Walk reloc_list, patch register descriptors with IOVA
        | 2. Pack task payload: input/output buffer handles, model ID
        | 3. ioctl(npu_fd, NPU_IOCTL_SUBMIT_TASK, &task_desc)
        v
[Kernel Mode Driver: eic7700_npu.ko]
        |
        | 1. Validate IOVA mappings and security permissions
        | 2. Push command packet into SiFive E31 Ring Buffer
        v
[Embedded SiFive E31 Controller]
        |
        | 1. Fetches command descriptor from Ring Buffer
        | 2. Programs NVDLA hardware registers:
        |      - CDMA: weight & input surface base registers
        |      - CSC: convolution scheduling
        |      - CMAC: systolic array activation
        |      - CACC / SDP: bias and activation LUT
        | 3. Starts hardware execution
        v
[NPU Hardware Accelerator]
        |
        | Computes matrix operations directly against DRAM / SPRAM
        | Fires hardware completion interrupt upon completion
        v
[Kernel Interrupt Handler]
        |
        | Wakes sleeping ioctl thread via wait_queue
        v
[libedla_runtime.so]
        |
        | Returns ES_SUCCESS to Application
```

---

## 4. Cross-Die Peer Synchronization (`_peer` Models)

The dual-die EIC7702 architecture supports hardware-synchronized distributed inference across Die 0 and Die 1:

1. **Model Compilation**:
   When compiled with dual-die targeting, `EsAAC` generates two paired loadables: `block_X_npu_b1_d0_peer.model` and `block_X_npu_b1_d1_peer.model`.
2. **Mailbox Handshake**:
   - Die 0 opens `/dev/eswin_ipc_misc_dev0`.
   - Die 1 opens `/dev/eswin_ipc_misc_dev1`.
   - During `ES_NPU_LoadModelFromFile`, the DSP firmware on Die 0 issues a hardware handshake command across the inter-die AXI mailbox to verify that Die 1 is alive and ready.
   - **Crucial Diagnostic Finding**: If a `_peer` model is loaded on Die 0 in isolation without simultaneously running the Die 1 counterpart, the DSP mailbox handshake **hangs indefinitely** waiting for the peer acknowledgment.
3. **Execution Barrier**:
   Between transformer layers or during tensor-parallel all-reduce operations, the submodel event list triggers an inter-die interrupt. Both dies pause until the cross-die barrier is cleared.
