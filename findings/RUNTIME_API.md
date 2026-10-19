# ESWIN EIC7700X NPU Userspace Runtime API Specification

## 1. Runtime Libraries & Link Dependencies

The user-space NPU runtime comprises three key shared libraries located in `/lib` or `/usr/lib`:

| Library Name | Demangled Interface | Key Responsibilities |
| :--- | :--- | :--- |
| **`libedla_runtime.so.1.0`** | `eswin::priv::*`, `ES_NPU_*` | Primary NPU runtime API: Loadable parsing, context/stream management, tensor descriptors, task submission, synchronization. |
| **`libes_memory.so`** | `ES_SYS_MemAlloc`, `ES_SYS_MemFree` | Contiguous DMA memory management via `/dev/dma_heap/mmz_nid_*` and `/dev/malloc_dmabuf`. |
| **`libes_sys.so.1`** | System & platform initialization | Platform diagnostics, clock configuration, chip version queries. |
| **`libes_ak.so`** | `ES_AK_DSP_*` | AcceleratorKit high-level accelerated primitives (WarpAffine, Resize, NMS). |

> [!IMPORTANT]
> A common development pitfall: `ES_SYS_MemAlloc` and `ES_SYS_MemFree` are defined in **`libes_memory.so`**, **NOT** in `libes_sys.so`. Applications must link `-les_memory -ledla_runtime`.

---

## 2. Execution Model & Core Data Structures

The NPU execution hierarchy follows standard heterogeneous compute abstractions:
$$\text{Device} \longrightarrow \text{Context} \longrightarrow \text{Stream} \longrightarrow \text{Task}$$

```
+-------------------------------------------------------------------------+
|                              NPU Device                                 |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |                            Context                                |  |
|  |                                                                   |  |
|  |  +---------------------------+     +---------------------------+  |  |
|  |  |         Stream 0          |     |         Stream 1          |  |  |
|  |  |                           |     |                           |  |  |
|  |  |  [Task A] -> [Task B] ... |     |  [Task C] -> [Task D] ... |  |  |
|  |  +---------------------------+     +---------------------------+  |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

### 2.1 Core Data Structures (`es_npu_types.h`)

#### `NPU_DIMS4_S`
```c
typedef struct {
    ES_S32 n;  // Batch size
    ES_S32 c;  // Number of channels
    ES_S32 h;  // Height
    ES_S32 w;  // Width
} NPU_DIMS4_S;
```

#### `NPU_TENSOR_S`
```c
typedef struct {
    ES_CHAR name[81];                   // Tensor name matching ONNX graph
    ES_U64 bufferSize;                  // Aligned buffer size in bytes
    NPU_DIMS4_S dims;                   // [N, C, H, W]
    ES_U8 dataFormat;                   // 1: NCHW, 2: NHWC
    ES_U8 dataType;                     // 1: FP32, 2: FP16, 3: INT16, 4: INT8
    ES_U8 dataCategory;                 // 0: Feature Map, 1: Weight, 2: Bias
    ES_U8 pixelFormat;
    ES_U8 pixelMapping;
    ES_U32 stride[8];                   // Stride along each dimension
} NPU_TENSOR_S;
```

#### `ES_DEV_BUF_S`
```c
typedef struct ES_DEV_BUF {
    ES_U64 memFd;                       // dmabuf file descriptor from ES_SYS_MemAlloc
    ES_U64 offset;                      // Offset within buffer (usually 0)
    ES_U64 size;                        // Byte size of tensor
    ES_U64 reserve;                     // Reserved (0)
} ES_DEV_BUF_S;                         // Exactly 32 bytes
```

#### `NPU_TASK_S`
```c
typedef ES_S32 (*NPU_TaskCallback)(void *arg);

typedef struct {
    ES_U32 taskId;                      // System-assigned task ID upon submit
    ES_U32 modelId;                     // Model ID from ES_NPU_LoadModelFromFile
    ES_U8 inputFdNum;                   // Number of input buffers
    ES_U8 outputFdNum;                  // Number of output buffers
    ES_DEV_BUF_S inputFd[10];           // Array of input DMA buffer descriptors
    ES_DEV_BUF_S outputFd[10];          // Array of output DMA buffer descriptors
    NPU_TaskCallback callback;          // Asynchronous completion callback
    ES_VOID *callbackArg;               // Callback user data
    NPU_TASK_STATE_E state;             // Task lifecycle state
    ES_U8 sdkPrivate[64];               // Reserved SDK internal state
} NPU_TASK_S;
```

---

## 3. Runtime Lifecycle & Function Reference

### 3.1 Device & Context Management
```c
// Queries the number of available physical NPU devices (returns 2 on dual-die).
// MUST be called before ES_NPU_SetDevice!
ES_S32 ES_NPU_GetNumDevices(ES_U32 *count);

// Binds the calling thread/process to device ID (0 or 1).
ES_S32 ES_NPU_SetDevice(ES_U32 devId);

// Releases device binding.
ES_S32 ES_NPU_ReleaseDevice(ES_U32 devId);

// Creates an execution context for the currently bound device.
ES_S32 ES_NPU_CreateContext(npu_context *ctx);
ES_S32 ES_NPU_DestroyContext(npu_context ctx);

// Creates a command stream within a context.
ES_S32 ES_NPU_CreateStream(npu_context ctx, npu_stream *stream);
ES_S32 ES_NPU_DestroyStream(npu_stream stream);

// Blocks until all tasks submitted to the stream have completed.
ES_S32 ES_NPU_SynchronizeStream(npu_stream stream);
```

### 3.2 Contiguous Memory Allocation
```c
// Allocates contiguous physical memory from the MMZ DMA heap.
// Type 0: uncached, Type 1: cached.
ES_S32 ES_SYS_MemAlloc(ES_U64 *phyAddr, void **virtAddr, ES_U32 size, ES_U32 memType);

// Frees memory allocated by ES_SYS_MemAlloc.
ES_S32 ES_SYS_MemFree(ES_U64 phyAddr, void *virtAddr);

// Flushes CPU caches for cached memory buffers before NPU submission.
ES_S32 ES_SYS_MemFlushCache(ES_U64 phyAddr, void *virtAddr, ES_U32 size);
```

### 3.3 Model Loading & Tensor Introspection
```c
// Loads an offline .model binary loadable into memory and registers it.
// modelId is assigned by the runtime upon success.
ES_S32 ES_NPU_LoadModelFromFile(const ES_CHAR *modelPath, ES_U32 *modelId);

// Unloads the model and frees associated internal descriptors.
ES_S32 ES_NPU_UnloadModel(ES_U32 modelId);

// Queries input and output tensor counts.
ES_S32 ES_NPU_GetNumInputTensors(ES_U32 modelId, ES_U32 *numInputs);
ES_S32 ES_NPU_GetNumOutputTensors(ES_U32 modelId, ES_U32 *numOutputs);

// Retrieves tensor metadata (dimensions, format, type, bufferSize).
ES_S32 ES_NPU_GetInputTensorDesc(ES_U32 modelId, ES_U32 index, NPU_TENSOR_S *tensorDesc);
ES_S32 ES_NPU_GetOutputTensorDesc(ES_U32 modelId, ES_U32 index, NPU_TENSOR_S *tensorDesc);
```

### 3.4 Buffer Binding & Execution
```c
// Maps DMA buffer descriptor to NPU SMMU page tables (ioctl NPU_IOCTL_PREPARE_DMA).
ES_S32 ES_NPU_Prepare(ES_DEV_BUF_S *devBuf);

// Unmaps DMA buffer from SMMU page tables (ioctl NPU_IOCTL_UNPREPARE_DMA).
ES_S32 ES_NPU_Unprepare(ES_DEV_BUF_S *devBuf);

// Synchronously submits an array of tasks to a stream and waits for completion.
ES_S32 ES_NPU_Submit(NPU_TASK_S *tasks, ES_U32 taskNum, npu_stream stream);

// Asynchronously submits an array of tasks; completes via callback or SynchronizeStream.
ES_S32 ES_NPU_SubmitAsync(NPU_TASK_S *tasks, ES_U32 taskNum, npu_stream stream);
```

---

## 4. Known Driver Pitfalls & Multi-Die Constraints

### 4.1 Process-Bound Device State (`ES_NPU_ERROR_DEV_MULTI_SET`)
- **Observed Behavior**: The vendor `libedla_runtime.so` stores device context in global/static process variables rather than per-thread storage.
- If a process calls `ES_NPU_SetDevice(0)` and then another thread calls `ES_NPU_SetDevice(1)`, the runtime returns error code **`0xa00f6046` (`ES_NPU_ERROR_DEV_MULTI_SET`)**.
- Attempting to submit to both devices from different threads within the same process leads to silent kernel driver deadlocks.
- **Architectural Solution**:
  - Run **separate processes** pinned to NUMA node 0 and NUMA node 1 (`numactl -N 0` and `numactl -N 1`), communicating via Unix domain sockets or shared memory.
  - Or use vendor-compiled `_peer` models with firmware-level hardware mailbox synchronization.

### 4.2 Device File Permissions
- `/dev/npu0` and `/dev/npu1` have default permissions `0660 root:root`.
- Non-root processes fail during `ES_NPU_LoadModelFromFile` with error code `0xa00f6048` (`ES_NPU_ERROR_DEV_OPEN_FAILED`).
- Solution: Add a udev rule `/etc/udev/rules.d/99-npu.rules`:
  ```udev
  KERNEL=="npu*", MODE="0666"
  KERNEL=="eswin_ipc_misc_dev*", MODE="0666"
  KERNEL=="es-dsp*", MODE="0666"
  ```

---

## 5. Complete Error Code Reference (`es_npu_error.h`)

| Error Macro | Value | Cause & Recovery Action |
| :--- | :--- | :--- |
| `ES_SUCCESS` | `0x00000000` | Successful execution. |
| `ES_NPU_ERROR_BAD_PARAM` | `0xa00f6001` | Invalid parameter or NULL pointer passed to API. |
| `ES_NPU_ERROR_NO_MEMORY` | `0xa00f6004` | Out of MMZ DMA memory or SPRAM allocation failure. |
| `ES_NPU_ERROR_NOT_INIT` | `0xa00f6005` | `ES_NPU_GetNumDevices` or `ES_NPU_SetDevice` was not called first. |
| `ES_NPU_ERROR_DEV_NOT_FOUND` | `0xa00f6040` | Device ID exceeds number of physical devices. |
| `ES_NPU_ERROR_DEV_OPEN_FAILED`| `0xa00f6048` | Failed to open `/dev/npuX` (permissions or device missing). |
| `ES_NPU_ERROR_DEV_MULTI_SET` | `0xa00f6046` | Attempted to set multiple devices in the same process. |
| `ES_NPU_ERROR_MODEL_NOT_FOUND`| `0xa00f6010` | Invalid `modelId` or model file does not exist on disk. |
| `ES_NPU_ERROR_SUBMIT_TASK_FAILED`| `0xa00f6024`| Hardware submission failed (invalid descriptor / SMMU fault). |
| `ES_NPU_ERROR_TIME_OUT` | `0xa00f6025` | Hardware task timeout (NPU hung or waiting on peer die). |
