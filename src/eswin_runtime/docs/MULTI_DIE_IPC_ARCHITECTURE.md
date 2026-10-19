# EIC7700X Multi-Die Process Isolation & NUMA Worker Architecture

## 1. Overview & Problem Statement

The **ESWIN EIC7700X** (SiFive P550 8-core RISC-V SoC, FML13V03) features a dual-die chiplet design containing two independent NPU acceleration engines:
- **Die 0**: NPU Core 0 (`/dev/npu0`), DSP Cores 0-3 (`/dev/es-dsp0-3`), NUMA Node 0 (Cores 0-3), MMZ Zone `mmz_nid_0_part_0`.
- **Die 1**: NPU Core 1 (`/dev/npu1`), DSP Cores 4-7 (`/dev/es-dsp4-7`), NUMA Node 1 (Cores 4-7), MMZ Zone `mmz_nid_1_part_0`.

### The Single-Process Bottleneck
The proprietary vendor NPU user-space library (`libedla_runtime.so`) was primarily designed for single-device operation. In multi-device setups, disassembling `libedla_runtime.so` reveals critical architectural constraints:
1. **Thread-Local Context & Global Engine State**: The NPU runtime uses thread-local pointers (`_ZN5eswin12esCtxDefaultE`) and process-global task queues (`HWProcessorMgr` and `TaskScheduler`). When two threads in the same process call `ES_NPU_SetDevice(0)` and `ES_NPU_SetDevice(1)` simultaneously, internal state tracking, mutex locks, and condition variable notifications (`condition_variable::wait` inside `submitSync`) experience severe contention, deadlock risks, and driver frame teardown faults (`destroy_frame.part.0` kernel Oops).
2. **NUMA Memory Locality Violations**: Allocating buffers on Die 0's memory controller while submitting from Die 1 causes cross-interconnect DDR traffic, latency spikes, and potential bus saturation.

To achieve clean, deterministic, full-throughput dual-die concurrency, `eswin_runtime` implements **Multi-Die Process Isolation with NUMA-Pinned Worker Daemons**.

---

## 2. Architecture Diagram

```
+-----------------------------------------------------------------------------------------+
|                                    MASTER APPLICATION                                    |
|                                                                                         |
|   +------------------------------------+       +------------------------------------+   |
|   |         DmaBuffer (Die 0)          |       |         DmaBuffer (Die 1)          |   |
|   |      (mmz_nid_0_part_0 pool)       |       |      (mmz_nid_1_part_0 pool)       |   |
|   +-----------------+------------------+       +-----------------+------------------+   |
|                     |                                            |                      |
|                     v                                            v                      |
|   +---------------------------------------------------------------------------------+   |
|   |                         WorkerPool Controller (C++17)                           |   |
|   |                                                                                 |   |
|   |    Channel 0 (AF_UNIX socketpair)              Channel 1 (AF_UNIX socketpair)   |   |
+---+-------------------+------------------------------------------+------------------+---+
                        |                                          |
                        | IPC (SOCK_SEQPACKET)                     | IPC (SOCK_SEQPACKET)
                        v                                          v
+------------------------------------+   +------------------------------------+
|       Worker Daemon (Die 0)        |   |       Worker Daemon (Die 1)        |
|                                    |   |                                    |
|  PID: Child 0                      |   |  PID: Child 1                      |
|  NUMA Affinity: Node 0 (Cores 0-3) |   |  NUMA Affinity: Node 1 (Cores 4-7) |
|  NPU Device: 0 (/dev/npu0)         |   |  NPU Device: 1 (/dev/npu1)         |
|  Memory Pool: mmz_nid_0_part_0     |   |  Memory Pool: mmz_nid_1_part_0     |
|                                    |   |                                    |
|  +------------------------------+  |   |  +------------------------------+  |
|  | Imported rcvd_fds via        |  |   |  | Imported rcvd_fds via        |  |
|  | ES_SYS_RcvMemFd              |  |   |  | ES_SYS_RcvMemFd              |  |
|  +------------------------------+  |   |  +------------------------------+  |
|  | ES_NPU_Prepare(input/output) |  |   |  | ES_NPU_Prepare(input/output) |  |
|  +------------------------------+  |   |  +------------------------------+  |
|  | ES_NPU_Submit(NpuTask*)      |  |   |  | ES_NPU_Submit(NpuTask*)      |  |
|  +------------------------------+  |   |  +------------------------------+  |
+------------------------------------+   +------------------------------------+
```

---

## 3. Inter-Process Communication (IPC) Protocol

Communication between the master application and the worker daemons uses bi-directional Unix domain sockets created via:
```c
socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, fds);
```

### Why `SOCK_SEQPACKET`?
- **Preserves Message Boundaries**: Unlike `SOCK_STREAM`, every `send()` maps to exactly one `recv()`, eliminating partial-read framing overhead.
- **Connection-Oriented & Reliable**: Guaranteed delivery with zero protocol serialization overhead.
- **Low Latency**: Kernel socket pair buffers on local host execute round-trip IPC handshakes in < 15 microseconds.

### Binary Message Schema
Messages are strictly packed C structures (`#pragma pack(push, 8)`):

```cpp
enum class IpcCmd : uint32_t {
    Ping            = 0,
    LoadModel       = 1,
    UnloadModel     = 2,
    PrepareBuffer   = 3,
    UnprepareBuffer = 4,
    SubmitTask      = 5,
    Shutdown        = 6
};

struct IpcBufferDesc {
    uint64_t mem_fd;    // Master process memFd handle
    uint64_t offset;    // Offset into contiguous physical allocation
    uint64_t size;      // Allocation size in bytes
    uint64_t reserved;  // Alignment padding
};

struct IpcRequest {
    IpcCmd        cmd;
    uint32_t      seq_id;
    uint32_t      model_id;
    uint8_t       input_count;
    uint8_t       output_count;
    char          model_path[256];
    IpcBufferDesc inputs[10];
    IpcBufferDesc outputs[10];
};

struct IpcResponse {
    IpcStatus status;
    uint32_t  seq_id;
    uint32_t  model_id;
    uint8_t   num_inputs;
    uint8_t   num_outputs;
    uint64_t  input_sizes[10];
    uint64_t  output_sizes[10];
    int32_t   driver_error;
    uint64_t  elapsed_ns;
};
```

---

## 4. Cross-Process DMA Memory Sharing (`ES_SYS_RcvMemFd`)

### The `memFd` Layout
In ESWIN's system memory management library (`libes_memory.so`), the 64-bit `memFd` returned by `ES_SYS_MemAlloc` encodes:
```
63                             32 31                             0
+-------------------------------+--------------------------------+
|          Owner PID            |      Kernel File Descriptor    |
|       (static_cast<pid_t>)    |        (int32_t dma_buf_fd)    |
+-------------------------------+--------------------------------+
```

### Cross-Process Handle Import
When a buffer is allocated in the master process (`PID_M`) and passed to worker process (`PID_W`):
1. `PID_W` detects that `(snd_fd >> 32) != getpid()`.
2. `PID_W` invokes:
   ```cpp
   uint64_t rcvd_fd = 0;
   int ret = ES_SYS_RcvMemFd(snd_fd, &rcvd_fd);
   ```
3. Internally, `libes_memory.so` performs:
   - `pidfd_open(snd_pid, 0)`
   - `pidfd_getfd(pidfd, snd_raw_fd, 0)` to duplicate the file descriptor into the worker process's fd table.
   - Reconstructs `rcvd_fd = (uint64_t(getpid()) << 32) | rcvd_raw_fd`.

### Security & Non-Root Permissions (`PR_SET_PTRACER`)
By default, the Linux kernel security module (Yama / ptrace restrictions) prevents an unprivileged process from calling `pidfd_getfd` on another process unless the target is ptraced.
To allow seamless unprivileged operation as user `chri`, the master process sets:
```c
prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
```
prior to spawning workers, allowing worker daemons to duplicate DMA handles via `pidfd_getfd` without requiring `sudo` or root privileges.

---

## 5. Hardware Task Structure Contract (`NPU_TASK_S`)

Disassembly and DWARF inspection of `libedla_runtime.so` confirmed the exact 744-byte struct layout expected by `ES_NPU_Submit`:

```cpp
#pragma pack(push, 8)
struct DevBuf {
    uint64_t mem_fd;    // Translated local memFd
    uint64_t offset;    // Offset in bytes
    uint64_t size;      // Buffer size in bytes
    uint64_t reserved;  // Alignment padding
};

struct NpuTask {
    uint32_t taskId;            // [0..3]   Assigned by SDK
    uint32_t modelId;           // [4..7]   Loaded model handle
    uint8_t  inputFdNum;        // [8]      Number of input tensors (<= 10)
    uint8_t  outputFdNum;       // [9]      Number of output tensors (<= 10)
    uint8_t  reserved0[6];      // [10..15] Alignment padding to 16
    DevBuf   inputFd[10];       // [16..335] Array of 10 DevBuf descriptors (320 bytes)
    DevBuf   outputFd[10];      // [336..655] Array of 10 DevBuf descriptors (320 bytes)
    void    *callback;          // [656..663] Optional completion callback
    void    *callbackArg;       // [664..671] Callback user argument
    uint32_t state;             // [672..675] NPU_TASK_STATE_E
    uint8_t  sdkPrivate[64];    // [676..739] Internal SDK scheduler state
    uint8_t  reserved1[4];      // [740..743] Tail padding to 744 bytes
};
#pragma pack(pop)

static_assert(sizeof(NpuTask) == 744, "NpuTask must exactly match vendor NPU_TASK_S");
```

### Critical Driver Requirements:
1. Every input and output buffer descriptor must be explicitly registered with the NPU hardware via `ES_NPU_Prepare(&dev_buf)` before submission.
2. The `inputFd` array must reside strictly at byte offset 16; misplacement causes the kernel driver ioctl to receive corrupted window sizes (`io_tensor_size != win_arg->tensor_size`).
3. Each worker process invokes `ES_NPU_Prepare()` within its own context so that the local NPU MMU page tables are populated.

---

## 6. CPU & NUMA Affinity Strategy

On the EIC7700X, inter-die memory access traverses the inter-chiplet bus (D2D link), which incurs latency penalties. Each worker daemon pins its execution thread and memory policies upon startup:

```cpp
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
if (m_die_id == 0) {
    // NUMA Node 0: Cores 0, 1, 2, 3
    for (int c = 0; c < 4; ++c) CPU_SET(c, &cpuset);
} else {
    // NUMA Node 1: Cores 4, 5, 6, 7
    for (int c = 4; c < 8; ++c) CPU_SET(c, &cpuset);
}
sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
```

Furthermore, DMA buffers are allocated from the local zone:
- Die 0: `mmz_nid_0_part_0`
- Die 1: `mmz_nid_1_part_0`

---

## 7. Performance & Concurrency Results

Validation benchmark results on board `roma` (YOLOv3 model, 1 input tensor, 3 output tensors):

| Metric | Single Die (Die 0 only) | Concurrent Dual Die (Die 0 + Die 1) | Speedup / Scaling |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | 17.03 ms / inference | 17.03 ms / inference | **0% contention penalty** |
| **Throughput (FPS)** | ~58.7 inferences/sec | **117.41 inferences/sec** | **2.00x linear scaling** |
| **NUMA Isolation** | Cores 0-3 / Node 0 | Cores 0-3 (Die 0) + Cores 4-7 (Die 1) | **Deterministic separation** |
| **IPC Roundtrip Latency** | N/A | < 15 microseconds | **Negligible (< 0.1% overhead)** |
