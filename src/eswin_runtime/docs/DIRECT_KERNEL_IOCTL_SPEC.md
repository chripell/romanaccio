# Direct Kernel ioctl Specification (Milestone 11, Layer 0')

Goal: reach the NPU without `libedla_runtime.so.1.0` or `libes_memory.so`. Allocation
goes through the standard Linux **dma-heap** uAPI, mapping through `mmap`, cache
maintenance through **DMA_BUF_IOCTL_SYNC**, and SMMU binding through `eic7700_npu.ko`.

`readelf -d libes_runtime.so` confirms no vendor library in `DT_NEEDED` — only
`libstdc++`, `libm`, `libgomp`, `libgcc_s`, `libc`. The backend never calls `dlopen`.

Build with the switch on:

```bash
make ENABLE_DIRECT_KERNEL_MODE=1                     # adds -DES_ENABLE_DIRECT_KERNEL_MODE=1
cmake -B build -DENABLE_DIRECT_KERNEL_MODE=ON
```

## Device nodes

| Node | Role |
|---|---|
| `/dev/npu0`, `/dev/npu1` | NPU character devices, `0660 root:root` |
| `/dev/dma_heap/mmz_nid_0_part_0` | Die 0 MMZ carveout (11 GiB default) |
| `/dev/dma_heap/mmz_nid_1_part_0` | Die 1 MMZ carveout |

The `0660 root:root` mode means `open()` fails `EACCES` without `sudo` or a udev rule;
the backend says so explicitly rather than reporting a generic failure.

## ioctl catalog

All NPU commands use magic `'n'` (`0x6E`). Values below were recovered by disassembling
`npu_dev_ioctl` (see `findings/CPU_NPU_PROTOCOL.md`) and are pinned by `static_assert` in
`direct_kernel_backend.hpp`, so the table is testable on any host:

| Command | Encoding | dir | nr |
|---|---|---|---|
| `SUBMIT_MODEL` | `0xc0046e03` | RW | 0x03 |
| `UNLOAD_MODEL` | `0xc0046e04` | RW | 0x04 |
| `SUBMIT_TASK` | `0xc0046e05` | RW | 0x05 |
| `PROCESS_REPORT` | `0x80046e08` | R | 0x08 |
| `SUBMIT_OP_TASK` | `0xc0046e09` | RW | 0x09 |
| `GET_SRAM` | `0x80046e0a` | R | 0x0a |
| `HANDLE_PERF` | `0x80046e0b` | R | 0x0b |
| `GET_PERF_DATA` | `0x80046e0c` | R | 0x0c |
| `MUTEX_LOCK` | `0x80046e0d` | R | 0x0d |
| `MUTEX_UNLOCK` | `0xc0046e0e` | RW | 0x0e |
| `PREPARE_DMA` | `0xc0046e0f` | RW | 0x0f |
| `UNPREPARE_DMA` | `0xc0046e10` | RW | 0x10 |
| `MUTEX_TRYLOCK` | `0xc0046e11` | RW | 0x11 |

Standard Linux, derived rather than recovered:

- `DMA_HEAP_IOCTL_ALLOC` = `0xc0184800` = `_IOWR('H', 0, struct dma_heap_allocation_data)`,
  24-byte request `{u64 len; u32 fd; u32 fd_flags; u64 heap_flags}`.
- `DMA_BUF_IOCTL_SYNC` = `_IOW('H', 0, struct dma_buf_sync)`, 8-byte `{u64 flags}`;
  `START|WRITE` before CPU writes, `END|WRITE` to publish them.

## The payload-size discrepancy (open question)

**Every** NPU command encodes a size field of **4 bytes**, including `PREPARE_DMA`, which
the catalog annotates as `_IOWR('n', 0x0F, struct ES_DEV_BUF)`. `ES_DEV_BUF` is 32 bytes
(`mem_fd`, `offset`, `size`, `reserved`), which would encode as `0xc0206e0f`. The kernel
dispatches on the full 32-bit command word, so `_IOWR('n', 0x0F, DevBuf)` would be
rejected as an unknown ioctl.

Size 4 is consistent with the mutex commands, which the same catalog types as `uint32_t`.
The working hypothesis is that the kernel takes a 4-byte handle — most likely the DMA-BUF
fd — and the `ES_DEV_BUF` annotation describes the *userspace* struct the vendor library
fills in, not the ioctl payload. `prepare_dma()` therefore passes
`static_cast<uint32_t>(buf.fd)`.

This is **not verified**. To settle it on `roma`:

```bash
make CROSS_COMPILE=riscv64-linux-gnu-
scp build/test_direct_kernel_mode build/libes_runtime.so roma:/tmp/m11/
ssh roma "cd /tmp/m11 && sudo LD_LIBRARY_PATH=/tmp/m11 ./test_direct_kernel_mode"
```

The suite prints `PREPARE_DMA(fd as u32) accepted` or
`[OPEN] PREPARE_DMA returned -1 (…)` with the errno. If it is `EINVAL`, the payload is
the 32-byte struct after all (and the size field is simply ignored by this driver); if it
is `EBADF`/`ENODEV`, the fd must first be imported into the NPU driver's own dma-buf
attachment table. Either answer is a real result worth recording in `FINDINGS.md`.

Deliberately **reported rather than asserted** in the test: the allocation, mapping,
round-trip and `DMA_BUF_IOCTL_SYNC` steps *are* asserted on hardware, because they are
standard Linux uAPI and must work. `PREPARE_DMA` is not, because asserting a guess would
either fake a pass or fail the suite for a reason that is not a code defect.

## What is not implemented

- **`SUBMIT_MODEL`** — needs `struct npu_model_desc`, which has not been recovered. Without
  it there is no `modelId`, so `SUBMIT_TASK` has nothing to reference. This is the blocker
  for a fully blob-free inference path and the reason Milestone 11 is partial.
- **Firmware mailbox / interrupt handling** — the vendor library's completion path
  (`PROCESS_REPORT`, wait queues) is not reproduced.
- **No integration** — `DmaBuffer` and `WorkerPool` still go through `EswinDriver`'s
  `dlopen` path. The direct backend is standalone and additive; switching the allocator
  over is a separate, board-verified change.

## Verified state (2026-10-24)

- Host: all 13 encodings asserted, node naming checked, graceful-failure paths checked
  (`open()` and `alloc_dma()` fail cleanly and name the missing node), no `DT_NEEDED` on
  any vendor library.
- RISC-V: clean `riscv64-linux-gnu-` cross-compile of the library and the test.
- Board: **not run** — SSH authentication to `roma` failed at the time
  (`Permission denied (publickey,password)`), so dma-heap allocation, the mapping
  round-trip and the `PREPARE_DMA` hypothesis are all still unverified on silicon.
