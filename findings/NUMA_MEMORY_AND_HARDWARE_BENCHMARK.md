# NUMA Memory Architecture, Bandwidth Benchmarks & Shared NPU Memory Configuration

## 1. Executive Summary

This report delivers a deep empirical and architectural analysis of the memory subsystem on the **DC-ROMA RISC-V Mainboard II (FML13V03)**, powered by the dual-die **ESWIN EIC7702 SoC**.

Key topics covered:
1. **Physical NUMA Memory Benchmark**: Rigorous measurement of single-core, multi-core, and cross-die read, write, copy, triad bandwidth, and pointer-chasing latency on physical silicon.
2. **MMZ DMA Buffer Throughput**: Characterization of CPU $\leftrightarrow$ MMZ DMA buffer transfers in `NoCache` vs `Cached` modes.
3. **Hardware Shared Memory Expansion Guide**: Mathematical mapping and step-by-step U-Boot reconfiguration commands to expand NPU MMZ memory from **22 GiB up to 27 GiB or 28 GiB**, leaving 4–5 GiB for headless Linux.
4. **Memory Bandwidth Bottleneck & MoE Architectures**: Quantitative analysis of the autoregressive decoding memory wall, proving why **Mixture-of-Experts (MoE)** architectures are essential for achieving 10+ tokens/s on this SoC.

---

## 2. Hardware Memory Architecture & NoC Bandwidth Ceilings

The EIC7702X SoC on the FML13V03 mainboard integrates two identical EIC7700X dies (Die 0 and Die 1) interconnected via a coherent TileLink/AXI Die-to-Die (D2D) Chiplink bus:

```
┌────────────────────────────────────────────── Die 0 ──────────────────────────────────────────────┐
│  [Cores 0-3 @ 1.8 GHz] ── [4 MB L3] ──(mcput_snoc_mp: 256-bit @ 800 MHz = 25.6 GB/s max)──┐       │
│                                                                                           ▼       │
│  [NPU 0 (19.95 TOPS)]  ── [4 MB LLC] ──(llc_noc p0 @ 1.188 GHz)──► [LPDDR5 Ctrl 0+1 (64-bit)]    │
│  [DSP 0-3 @ 1.0 GHz]   ── [HAE 0]    ──(mnoc/rnoc p3/p4)─────────► [6400 MT/s = 51.2 GB/s/die]   │
│                                                                     Base: 0x00_8000_0000 (16 GiB) │
│                                                                     MMZ:  0x01_C000_0000 (11 GiB) │
└─────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                  │ D2D Interconnect Bus
                                                  │ (Cap: ~17 GB/s pin / ~11.4 GB/s nominal)
┌─────────────────────────────────────────────────┴─────────────────────────────────────────────────┐
│  [Cores 4-7 @ 1.8 GHz] ── [4 MB L3] ──(mcput_snoc_mp: 256-bit @ 800 MHz = 25.6 GB/s max)──┐       │
│                                                                                           ▼       │
│  [NPU 1 (19.95 TOPS)]  ── [4 MB LLC] ──(llc_noc p0 @ 1.188 GHz)──► [LPDDR5 Ctrl 0+1 (64-bit)]    │
│  [DSP 4-7 @ 1.0 GHz]   ── [HAE 1]    ──(mnoc/rnoc p3/p4)─────────► [6400 MT/s = 51.2 GB/s/die]   │
│                                                                     Base: 0x20_0000_0000 (16 GiB) │
│                                                                     MMZ:  0x21_4000_0000 (11 GiB) │
└────────────────────────────────────────────── Die 1 ──────────────────────────────────────────────┘
```

### 2.1. Theoretical LPDDR5 Peak vs. CPU L3 Outer Port Ceiling (Why 4 CPU Cores Cannot Saturate 51.2 GB/s)

From the boot log (`DDR type:LPDDR5_D2D_X16;Size:16GB,Data Rate:6400MT/s`) and **EIC7700X TRM Part 1 (§1.3 p. 73, §3.2 p. 136, §3.4.1.16 p. 286, §3.7 p. 391–392)**:

1. **Raw LPDDR5 Peak Bandwidth**:
   - Each die has two 32-bit LPDDR5 controllers (`ddrc0` at `0x52300000` and `ddrc1` at `0x52380000`, each operating two 16-bit channels) = **64-bit bus per die @ 6400 MT/s**:
     $$\text{Peak LPDDR5 Bandwidth per Die} = \frac{64\text{ bits}}{8} \times 6400\text{ MT/s} = \mathbf{51.2\text{ GB/s per die}} \quad (\mathbf{102.4\text{ GB/s dual-die}})$$
2. **Why the 4-Core CPU Cluster (MCPU) Caps Below ~23 GB/s per Die**:
   - Each DDR controller (`ddrt0`, `ddrt1`) exposes 5 AXI ports (`p0..p4`) clocked at `clk_ddr_aclk = 800 MHz`.
   - **Port `p0` (`lnoc_ddrt0_p0` & `lnoc_ddrt1_p0`)** is a dedicated 1.188 GHz highway (`llc_noc`) wired **exclusively to the NPU LLC**, allowing the NPU + E31 DSP complex to stream weights at ~45+ GB/s per die.
   - By contrast, all 4 SiFive P550 cores (`1.8 GHz`) share a 4-bank 4 MiB L3 cache whose outer interface is a **single 256-bit (32-byte) TileLink-C port (`mcput_snoc_mp`)** feeding into `sys_noc` (`800 MHz`):
     $$\text{MCPU L3 Outer Port Ceiling} = 32\text{ Bytes} \times 800\text{ MHz} = \mathbf{25.6\text{ GB/s per die}} \quad (\mathbf{51.2\text{ GB/s dual-die}})$$
   - By hardware design, the 4-core CPU cluster is bottlenecked by its single 256-bit `mcput_snoc_mp` port (`~22.6–23.2 GB/s` measured Read, or **90.5% of the 25.6 GB/s port ceiling**), ensuring the CPU can never starve the NPU (`llc_noc`) or multimedia engines (`mnoc`) of the remaining ~28+ GB/s per die of LPDDR5 bandwidth.
3. **P550 L2 TLB 4 KiB Limitation vs. L1 DTLB 2 MiB HugePages (TRM §3.4.1.5 p. 192)**:
   - The P550's 512-entry unified L2 TLB **only supports 4 KiB pages** ($512 \times 4\text{ KiB} = \mathbf{2\text{ MiB}}$ coverage per core).
   - **2 MiB Mega Pages (HugePages)** are cached exclusively in the **32-entry L1 DTLB** ($32 \times 2\text{ MiB} = \mathbf{64\text{ MiB}}$ coverage).
   - Because `/sys/kernel/mm/transparent_hugepage/enabled` defaults to `[madvise]` on `roma`, large buffers allocated without `madvise(..., MADV_HUGEPAGE)` (and `/dev/es_mem` MMZ buffers mapped via 4 KiB `remap_pfn_range` PTEs) suffer an L2 TLB miss every 4 KiB, triggering 4-level Sv48 page-table walks in DRAM. Enabling `MADV_HUGEPAGE` cuts local memory latency from **205 ns $\to$ 180 ns** and increases 8-core Read bandwidth from **40.5 GB/s $\to$ 43.5 GB/s**.
4. **Write-Allocate / Read-For-Ownership (RFO) on RV64GC Write-Back Caches**:
   - Because P550 implements `RV64GC` (without `Zicboz` cache-block zeroing), storing to a cache line in `Write`, `Copy`, or `Triad` first issues a 64-byte **Read-For-Ownership (RFO)** from DRAM before writing back the dirty line.
   - Consequently, actual physical DRAM pin traffic (`Pin*`) is **$2\times$ nominal for `Write`** ($16\text{ B/elem}$), **$1.5\times$ nominal for `Copy`** ($24\text{ B/elem}$), and **$\frac{4}{3}\times$ nominal for `Triad`** ($32\text{ B/elem}$).

---

## 3. Physical Silicon NUMA Benchmark Results (v2 Audit)

The upgraded benchmark suite (`numa_bench.cpp` v2, compiled with `-O3 -march=rv64gc -fopenmp`) verifies physical page-frame numbers (`PFN`) via `/proc/self/pagemap` and `ES_SYS_RetrieveMemNodeByFd`, uses an 8-way unrolled per-thread reduction for `Read` (fixing the v1 serial `fadd.d` latency bottleneck), and fixes the v1 8-core OpenMP loop partitioning bug.

All tests operate on **256 MiB per array** (**768 MiB working set per die** for `Triad`), far exceeding the 4 MiB L3 cache.

### 3.1. Physical Page-Frame (`PFN`) Locality Audit (`/proc/self/pagemap`)

Per **TRM Table 3-39 (p. 220)**, physical addresses `0x00_8000_0000..0x04_7FFF_FFFF` are hardwired in silicon to Die 0 LPDDR5, and `0x20_0000_0000..0x23_FFFF_FFFF` are hardwired to Die 1 LPDDR5:

| Allocator Mode & Target Node | Die 0 Pages (%) | Die 1 Pages (%) | Verified Hardware Physical Address Span | Page Granularity |
| :--- | :---: | :---: | :---: | :---: |
| `Linux numa_alloc_onnode(Node 0)` | **100.0%** | 0.0% | `0x01_0542_0000 .. 0x01_AEFC_5000` | 4 KiB PTE |
| `Linux numa_alloc_onnode(Node 1)` | 0.0% | **100.0%** | `0x20_3DB0_F000 .. 0x20_9EFF_5000` | 4 KiB PTE |
| `Strict mmap + mbind(MPOL_BIND) + THP (Node 0)` | **100.0%** | 0.0% | `0x01_35A8_5000 .. 0x01_8046_9000` | **2 MiB PMD (HugePage)** |
| `Strict mmap + mbind(MPOL_BIND) + THP (Node 1)` | 0.0% | **100.0%** | `0x20_3E40_0000 .. 0x20_9EFF_0000` | **2 MiB PMD (HugePage)** |
| `MMZ Cached (mmz_nid_0_part_0)` | **100.0%** | 0.0% | `0x01_C000_0000 .. 0x01_D000_0000` | 4 KiB `VM_PFNMAP` |
| `MMZ Cached (mmz_nid_1_part_0)` | 0.0% | **100.0%** | `0x21_4000_0000 .. 0x21_5000_0000` | 4 KiB `VM_PFNMAP` |

### 3.2. Single-Core Local vs. Remote Performance (`Strict mmap + mbind + 2 MiB HugePage` vs. `MMZ 4K`)

| Benchmark Case | Latency (2M THP) | Latency (4K MMZ) | Read (GB/s) | Write (GB/s) | Copy (GB/s) | Triad (GB/s) | PinTriad (GB/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Core 0 $\to$ Die 0 (Local)** | **180.42 ns** | 204.48 ns | **11.33** | **7.95** | **11.09** | **6.46** | **8.61** |
| **Core 0 $\to$ Die 1 (Remote)** | **651.25 ns** | 670.59 ns | 1.79 | 1.02 | 1.79 | 1.79 | 2.38 |
| **Core 4 $\to$ Die 1 (Local)** | **184.54 ns** | 205.65 ns | **11.14** | **7.60** | **10.91** | **6.34** | **8.45** |
| **Core 4 $\to$ Die 0 (Remote)** | **662.48 ns** | 679.77 ns | 1.76 | 1.01 | 1.74 | 1.74 | 2.32 |

#### Analysis:
1. **Single-Core Bandwidth**: A single P550 core at 1.8 GHz achieves **11.33 GB/s pure Read** and **11.09 GB/s nominal Copy** (**16.64 GB/s physical pin traffic** with RFO), utilizing **65% of the entire die's 25.6 GB/s L3 outer port** from one core alone.
2. **2 MiB HugePage Latency Benefit**: Eliminating 4 KiB L2 TLB misses reduces random pointer-chasing latency across a 128 MiB working set by **~24 ns** (**180.4 ns** vs. **204.5 ns**).
3. **Cross-Die Penalty**: Remote cross-die access increases latency by **$3.6\times$** (`180 ns` $\to$ `651 ns`) and reduces single-core Read/Copy bandwidth by **$6.3\times$** (`11.33 GB/s` $\to$ `1.79 GB/s`) and Write bandwidth by **$7.8\times$** (`7.95 GB/s` $\to$ `1.02 GB/s`).

### 3.3. Multi-Core Die Saturation (4 Cores per Die — Allocator Comparison)

| Configuration (4 Threads Pinned to Die) | Read (GB/s) | Write (GB/s) | Copy (GB/s) | Triad (GB/s) | PinCopy (GB/s) | PinTriad (GB/s) | % of 25.6 GB/s L3 Port |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Die 0 $\to$ Node 0 (`Strict 2M HugePage`)** | **22.56** | 10.05 | **14.16** | **13.02** | **21.24** | **17.36** | **88.1%** |
| **Die 1 $\to$ Node 1 (`Strict 2M HugePage`)** | **23.17** | 10.63 | **14.07** | **12.52** | **21.10** | **16.70** | **90.5%** |
| **Die 0 $\to$ Node 0 (`Linux 4K numa_alloc`)** | 21.87 | 10.19 | 14.05 | 13.32 | 21.07 | 17.76 | 85.4% |
| **Die 0 $\to$ Die 0 (`MMZ Cached 4K PTE`)** | 20.88 | **11.71** | 12.78 | 10.10 | 19.16 | 13.47 | 81.6% |
| **Die 1 $\to$ Die 1 (`MMZ Cached 4K PTE`)** | 20.56 | 11.13 | 12.50 | 10.63 | 18.74 | 14.17 | 80.3% |
| **Die 0 $\to$ Node 1 (`Remote 2M HugePage`)** | 5.93 | 4.05 | 5.99 | 5.69 | 8.98 | 7.59 | 23.2% |
| **Die 1 $\to$ Node 0 (`Remote 2M HugePage`)** | 5.84 | 4.00 | 5.88 | 5.61 | 8.82 | 7.48 | 22.8% |

#### Analysis:
- **Saturation of the 25.6 GB/s MCPU L3 Outer Port (`mcput_snoc_mp`)**:
  - 4 cores on a single die reach **22.56–23.17 GB/s Read** and **21.10–21.24 GB/s PinCopy**, saturating **88%–90.5%** of the MCPU L3 outer port's theoretical 25.6 GB/s ceiling (`32 B @ 800 MHz`).
  - Notice that both pure `Read` (`22.56 GB/s`, no RFO) and `PinCopy` (`21.24 GB/s`, 2 reads + 1 writeback = 24 B/elem) hit the **exact same ~21–23 GB/s hardware wall**, confirming that the bottleneck is the single 256-bit `mcput_snoc_mp` port between L3 and `sys_noc`.

### 3.4. Full-System 8-Core Concurrent Bandwidth & D2D Interconnect Cap

With all 8 cores active simultaneously (Cores 0–3 processing a full 768 MiB working set on Die 0 while Cores 4–7 process an independent 768 MiB working set on Die 1):

| Concurrent 8-Core Workload | Read (GB/s) | Copy (GB/s) | PinCopy (GB/s) | Triad (GB/s) | PinTriad (GB/s) | % of 51.2 GB/s Dual-L3 Port |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Concurrent Local (`Strict 2M HugePage`)** | **43.50 GB/s** | **25.76 GB/s** | **38.64 GB/s** | **24.01 GB/s** | **32.01 GB/s** | **85.0%** |
| **Concurrent Local (`MMZ Cached 4K PTE`)** | **40.51 GB/s** | 23.02 GB/s | 34.53 GB/s | 21.68 GB/s | 28.91 GB/s | 79.1% |
| **Concurrent Cross-Die (`Die0->1 + Die1->0`)** | 11.41 GB/s | 11.37 GB/s | 17.06 GB/s | 10.94 GB/s | 14.59 GB/s | 22.3% |

### 3.5. Linux Allocator Node-Locality Hazards on `roma` & How to Guarantee Zero Cross-Die Traffic

Inspecting `numactl --hardware` on `roma` during model execution reveals a critical operational hazard:
```text
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 | node 0 size: 4789 MB | node 0 free:   65 MB  <-- Node 0 exhausted by page cache!
node 1 cpus: 4 5 6 7 | node 1 size: 4805 MB | node 1 free: 3680 MB
```
1. **Why `malloc` / `std::vector` Silently Spills Across Dies**:
   - Because each die reserves 11 GiB for MMZ, Linux only manages ~4.8 GiB per node. The kernel, systemd services, and VFS file page cache (from reading `.model` / `.bin` files on Node 0) consume almost all of Node 0's free memory (`node 0 free: 65 MB`).
   - Standard heap allocations (`std::vector`, `malloc`) faulted on the main thread (or non-strict allocations when Node 0 is under page-cache pressure) either land on a single node (forcing the other die's 4 threads to read across the 651 ns D2D link at $3.8\times$ lower bandwidth) or trigger kernel fallback (`Fallback order for Node 0: 0 1`) onto Node 1.
   - Furthermore, `/proc/sys/kernel/numa_balancing` is enabled (`1`) by default, causing the kernel to periodically unmap PTEs and migrate pages across nodes when accessed by threads on both dies.
2. **How to Guarantee 100% Node-Locality for Large CPU Buffers (KV Caches, Scratch Workspaces)**:
   - **Rule 1 (Evict Unneeded Weight File Page Cache)**: After `fread()`-ing multi-hundred-MB weight blobs (`weights_die0.bin`, `weights_die1.bin`) into MMZ `DmaBuffer`s, call `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` before closing the file descriptor so the 1.5+ GiB duplicate VFS page cache does not exhaust Node 0 RAM.
   - **Rule 2 (Strict Per-Die `mmap` + `mbind(MPOL_BIND)` + `MADV_HUGEPAGE`)**: Allocate per-die CPU buffers (e.g., GQA KV cache heads `kv_h = 0, 1` on Node 0 for Cores 0–3, and `kv_h = 2, 3` on Node 1 for Cores 4–7) using anonymous `mmap`, bind each half strictly to its owning node with `mbind(..., MPOL_BIND, &nodemask, maxnode, MPOL_MF_STRICT | MPOL_MF_MOVE)`, and call `madvise(..., MADV_HUGEPAGE)` so the CPU accesses node-local **2 MiB HugePages** at the full **43.5 GB/s** dual-die bandwidth.
   - **Rule 3 (Strict Static Thread-to-Die Ownership)**: Never use `schedule(dynamic)` across KV heads located on different dies. Always map threads `tid = 0..3` (Cores 0–3, Die 0) exclusively to `kv_h = 0, 1` (Node 0 memory) and threads `tid = 4..7` (Cores 4–7, Die 1) exclusively to `kv_h = 2, 3` (Node 1 memory).

---

## 4. MMZ DMA Buffer Throughput (CPU $\leftrightarrow$ MMZ)

The NPU and DSP accelerators execute out of physically contiguous MMZ carveout memory (`/dev/malloc_dmabuf`). Host CPU threads frequently read or write these buffers (e.g. inserting prompt embeddings, reading output logits, copying KV states).

Throughput was measured using `ES_SYS_MemAlloc` across both memory modes:

| Region | Memory Mode | Write Throughput | Read Throughput | Characteristics / Risks |
| :--- | :---: | :---: | :---: | :--- |
| `mmz_nid_0_part_0` | `NoCache` | 0.18 GB/s | **0.01 GB/s (10 MB/s)** | Uncached single-beat reads; severe bus stall |
| `mmz_nid_0_part_0` | `Cached` | **0.84 GB/s** | **1.74 GB/s** | Cache-line bursts enabled; requires cache flush |
| `mmz_nid_1_part_0` | `NoCache` | 0.61 GB/s | 0.06 GB/s | Uncached single-beat reads |
| `mmz_nid_1_part_0` | `Cached` | **2.89 GB/s** | **1.73 GB/s** | High throughput; requires cache invalidation |

### Critical Finding on `NoCache` Mode:
- **CPU reads from `NoCache` MMZ memory are catastrophic (10–60 MB/s)**. In `NoCache` mode, the RISC-V CPU issues uncacheable, non-burst single 64-bit bus transactions across the AXI fabric. Reading a 64 MB tensor in `NoCache` mode stalls the CPU for **6+ seconds**.
- **Software Recommendation**:
  - Any buffer where the host CPU must read tensor outputs (e.g., LM head logits) **MUST** use `ES_MEM_ALLOC_ATTR_E::Cached` mode and issue cache invalidation before reading.
  - Large prompt embeddings written by the CPU should also use `Cached` mode followed by `flush_cache` prior to NPU launch.

---

## 5. Hardware Configuration: Expanding NPU Shared Memory (Above 11 GiB)

### 5.1. Physical Layout on the 32 GiB Board
On the FML13V03 32 GiB board, physical memory is mapped as:
- **Die 0**: `0x00_8000_0000` to `0x04_8000_0000` ($16 \text{ GiB}$)
- **Die 1**: `0x20_0000_0000` to `0x24_0000_0000` ($16 \text{ GiB}$)

The Linux kernel and MMZ carveout share each die's 16 GiB space:
- Memory from the die base up to `start_address` belongs to the **Linux OS / Kernel**.
- Memory from `start_address` up to the end of the die is reserved by the driver for **NPU/DSP MMZ**.

The MMZ boundary rule is:
$$\text{start\_address} + \text{size} = \text{0x04\_8000\_0000} \quad (\text{Die 0})$$
$$\text{start\_address} + \text{size} = \text{0x24\_0000\_0000} \quad (\text{Die 1})$$

### 5.2. Memory Allocation Tradeoff Configurations

Because this machine is used primarily for headless AI inference (no desktop environment or web browser), Linux system RAM can be safely reduced to 4–5 GiB total (~2.0–2.5 GiB per NUMA node):

| Target Configuration | MMZ Per Die | Total NPU VRAM | Linux RAM Per Node | Total Linux RAM | Feasible Models |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Factory Default** | 4.0 GiB | 8.0 GiB | 12.0 GiB | 24.0 GiB | YOLO, small CNNs |
| **Current Setup** | 11.0 GiB | 22.0 GiB | ~4.8 GiB | ~9.6 GiB | DeepSeek-7B (1K), SmolLM-135M |
| **Proposed Config A** | **13.5 GiB** | **27.0 GiB** | **2.5 GiB** | **5.0 GiB** | **Qwen2.5-14B (INT8) / 7B (32K ctx)** |
| **Proposed Config B** | **14.0 GiB** | **28.0 GiB** | **2.0 GiB** | **4.0 GiB** | **Maximal NPU VRAM** |

---

### 5.3. Step-by-Step U-Boot Reconfiguration

MMZ carveouts are passed from U-Boot to Linux via Device Tree nodes modified at boot time using the U-Boot `fdt mmz` command.

#### Reconfiguration for Config A: 27 GiB Total NPU VRAM (13.5 GiB / die, 5 GiB Linux)
- MMZ Size: $13.5 \times 1024^3 = 14,495,514,624\text{ bytes} = \text{0x360000000}$.
- Die 0: `start = 0x480000000 - 0x360000000 = 0x120000000`.
- Die 1: `start = 0x2400000000 - 0x360000000 = 0x20a0000000`.

To apply via U-Boot prompt:
```text
setenv fdt0_cmd "fdt mmz mmz_nid_0_part_0 0x120000000 0x360000000"
setenv fdt1_cmd "fdt mmz mmz_nid_1_part_0 0x20a0000000 0x360000000"
setenv bootcmd "run fdt0_cmd; run fdt1_cmd; ${bootcmd}"
saveenv
reset
```

#### Reconfiguration for Config B: 28 GiB Total NPU VRAM (14.0 GiB / die, 4 GiB Linux)
- MMZ Size: $14.0 \times 1024^3 = 15,032,385,536\text{ bytes} = \text{0x380000000}$.
- Die 0: `start = 0x480000000 - 0x380000000 = 0x100000000`.
- Die 1: `start = 0x2400000000 - 0x380000000 = 0x2080000000`.

To apply via U-Boot prompt:
```text
setenv fdt0_cmd "fdt mmz mmz_nid_0_part_0 0x100000000 0x380000000"
setenv fdt1_cmd "fdt mmz mmz_nid_1_part_0 0x2080000000 0x380000000"
setenv bootcmd "run fdt0_cmd; run fdt1_cmd; ${bootcmd}"
saveenv
reset
```

#### Post-Boot Verification on Linux:
```bash
# Verify Linux system memory:
free -h

# Verify MMZ carveouts reported by kernel:
dmesg | grep -i mmz

# Verify node balance:
numactl --hardware
```

---

## 6. Memory Bandwidth Constraints: Dense LLMs vs MoE Architectures

### 6.1. The Autoregressive Memory Wall
In autoregressive LLM decoding (generating one token at a time), arithmetic intensity is severely limited.

For a dense model with $P$ parameters:
- Every token generation step requires streaming **all $P$ weights from DRAM once**:
  $$\text{DRAM Bytes Streamed Per Token} \approx P \times S_{\text{dtype}}$$
  where $S_{\text{dtype}} = 1\text{ byte}$ for INT8, $2\text{ bytes}$ for FP16.
- The FLOPs performed per token are:
  $$\text{FLOPs Per Token} \approx 2 \times P$$
- The operational intensity is therefore:
  $$I = \frac{\text{FLOPs}}{\text{Bytes Transferred}} = \frac{2 \times P}{P \times S_{\text{dtype}}} = \frac{2}{S_{\text{dtype}}}$$
  For INT8 quantization, $I = 2 \text{ FLOPs / byte}$.

Because the hardware provides:
- **Compute Capacity**: $2 \times 19.95\text{ TOPS} = \mathbf{39.9\text{ TOPS}}$ (INT8).
- **Dual-Die Memory Bandwidth**: $\sim \mathbf{28\text{ GB/s}}$ (practical sustained read bandwidth across both dies).

The ridge point of the Roofline model is:
$$I_{\text{ridge}} = \frac{39.9 \times 10^{12}\text{ FLOP/s}}{28.0 \times 10^9\text{ Byte/s}} \approx \mathbf{1,425\text{ FLOPs/byte}}$$

Since autoregressive decoding has an intensity of only **2 FLOPs/byte**, it sits **$700\times$ below the compute roofline**. The hardware is **100% memory-bandwidth bound**. The NPUs spend over 99% of their clock cycles waiting for LPDDR5 memory fetches.

### 6.2. Throughput Comparison: Dense vs Mixture-of-Experts (MoE)

Maximum theoretical token generation speed is bounded by memory read bandwidth:
$$\text{Tokens / Second} \le \frac{\text{Memory Read Bandwidth (GB/s)}}{\text{Active Parameters Streamed (GB/token)}}$$

| Model Architecture | Quantization | Total Params | Active Params Per Token | Memory Streamed Per Token | Max Speed (Single Die @ 14 GB/s) | Max Speed (Dual Die @ 28 GB/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SmolLM-135M** (Dense) | INT8 | 135 M | 135 M | 0.14 GB | **100+ tok/s** | **150+ tok/s** |
| **Qwen2.5-1.5B** (Dense) | INT8 | 1.54 B | 1.54 B | 1.54 GB | 9.1 tok/s | **18.2 tok/s** |
| **Qwen2.5-7B** (Dense) | INT8 | 7.61 B | 7.61 B | 7.61 GB | 1.8 tok/s | **3.7 tok/s** |
| **Qwen2.5-14B** (Dense) | INT8 | 14.7 B | 14.7 B | 14.7 GB | 0.9 tok/s | **1.9 tok/s** |
| **DeepSeek-V2-Lite** (MoE) | INT8 | 15.7 B | **2.40 B** | **2.40 GB** | **5.8 tok/s** | **11.7 tok/s** |
| **Qwen2-57B-A14B** (MoE)* | INT8 | 57.0 B | **14.0 B** | **14.0 GB** | 1.0 tok/s | **2.0 tok/s** |
| **MiniCPM-MoE-8x2B** (MoE) | INT8 | 13.6 B | **4.10 B** | **4.10 GB** | **3.4 tok/s** | **6.8 tok/s** |

*\*Note: Qwen2-57B requires 57 GB weights, exceeding the board's 32 GB RAM, listed for architectural comparison.*

### 6.3. Why MoE is the Optimal Architectural Match for EIC7702
1. **High Model Capacity with Low Memory Bandwidth Consumption**:
   - A dense 7B model requires streaming 7.6 GB of weights for every single generated token, limiting generation speed to ~3.7 tokens/s.
   - An MoE architecture of comparable intelligence (such as DeepSeek-V2-Lite, 16B total) activates only 2 out of 64 routed experts per token. Only **2.4 GB of weights** are read per token!
   - Result: DeepSeek-V2-Lite runs at **~11.7 tokens/s** ($3.2\times$ faster than dense 7B) while retaining the reasoning knowledge of a 16B model.
2. **NUMA-Friendly Expert Partitioning**:
   - The dual-die architecture allows splitting experts cleanly:
     - Experts 0–31 placed in Die 0 MMZ (`mmz_nid_0_part_0`).
     - Experts 32–63 placed in Die 1 MMZ (`mmz_nid_1_part_0`).
   - Host CPU routing threads evaluate the router gate and dispatch tokens to the specific die hosting the activated experts, avoiding cross-die memory transfers.

---

## 7. Conclusions, Recommended Execution Rules & Real-World `lfm3_v2` Verification

1. **Memory Allocation**:
   - For workloads targeting 7B–14B models, reconfigure U-Boot to **Config A (27 GiB NPU MMZ, 5 GiB Linux)**. This provides 13.5 GiB per die, sufficient for 14B INT8 models or 7B models with 32K context windows.
2. **Buffer Access Policy**:
   - CPU reads from MMZ must **never use `NoCache` mode** (`10.2 MB/s` bottleneck). Always allocate with `ES_MEM_ALLOC_ATTR_E::Cached` when CPU readback is needed.
3. **Strict Per-Die NUMA Binding with 2 MiB HugePages (`mmap` + `mbind(MPOL_BIND)` + `MADV_HUGEPAGE`)**:
   - Standard `std::vector` / `malloc` uses `MPOL_DEFAULT` (First-Touch), which silently spills Node 0 pages onto Node 1 whenever Node 0's Linux zone fills up with VFS file page cache (`weights_die0.bin` + `weights_die1.bin` consume `1.75 GiB` in VFS cache if not evicted).
   - Always call `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` immediately after streaming model weight files into NPU MMZ carveouts. On `roma`, this restored Node 0 free Linux RAM from **`65 MB` to `2,933 MB`**.
   - For large CPU-side KV caches (`NumaKvBuffer<T>` in `lfm3_v2.cpp`), allocate anonymous virtual memory with `mmap`, apply `madvise(..., MADV_HUGEPAGE)` (so the P550 32-entry L1 DTLB covers 64 MiB per core and bypasses the 4 KiB-only 1024-entry L2 TLB), and strictly bind KV head slices `kh = 0, 1` to **Node 0** (accessed by cores 0–3) and `kh = 2, 3` to **Node 1** (accessed by cores 4–7) via `mbind(..., MPOL_BIND, ..., MPOL_MF_STRICT | MPOL_MF_MOVE)`.

### 7.1. Measured Impact on `lfm3_v2` (LFM3-1.7B Dual-Die Hybrid Engine on `roma`)

| Metric | Before (`std::vector` + VFS Page Cache Spill) | After (`NumaKvBuffer<T>` 2M THP + `POSIX_FADV_DONTNEED` + INT8 SWAR Prefill) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Node 0 Free Linux RAM (during run)** | `65 MB` (98% exhausted, spilling to Die 1) | **`2,933 MB`** (`100%` local Node 0/1 allocation) | **`+2,868 MB` freed** |
| **`--bench-attn` 512 ctx (6 Global Layers)** | `1.39 ms/tok` | **`1.25 ms/tok`** | **`-10.1%` latency** |
| **`--bench-attn` 4096 ctx (6 Global Layers)** | `7.36 ms/tok` | **`7.30 ms/tok`** | **`-0.8%` latency** |
| **`--bench-attn` 32768 ctx (6 Global Layers)** | `66.16 ms/tok` (`59.1 GB/s`) | **`65.11 ms/tok` (`60.1 GB/s` eff.)** | **`-1.6%` latency** |
| **4K C-Code Prefill (`bench_c_code_4k.c`, 3,781 tok)** | `86,204.2 ms` (`43.86 tok/s`, `22.8 ms/tok`) | **`59,168.8 ms` (`63.90 tok/s`, `15.6 ms/tok`, peak `74.4 tok/s`)** | **`+45.7%` (`1.46×` faster, `-27.0 s`)** |
| **Autoregressive Decode (short context)** | `8.81 tok/s` (`113.5 ms/tok`) | **`8.85 tok/s` (`113.0 ms/tok`)** | **`+0.5%`** |
| **Autoregressive Decode (at 3,909 tok context)** | `7.79 tok/s` (`128.4 ms/tok`) | **`7.84 tok/s` (`128.6 ms/tok`)** | **`+0.6%`** |
| **Golden Gate Cosine (`min-layer` / `128K logits`)** | `0.98892771` / `0.99594256` (4/4 exact) | **`0.98892771` / `0.99594256` (4/4 exact)** | **Bit-for-bit identical** |
