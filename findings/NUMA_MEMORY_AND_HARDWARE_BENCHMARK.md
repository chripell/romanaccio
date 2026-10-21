# NUMA Memory Architecture, Bandwidth Benchmarks & Shared NPU Memory Configuration

## 1. Executive Summary

This report delivers a deep empirical and architectural analysis of the memory subsystem on the **DC-ROMA RISC-V Mainboard II (FML13V03)**, powered by the dual-die **ESWIN EIC7702 SoC**.

Key topics covered:
1. **Physical NUMA Memory Benchmark**: Rigorous measurement of single-core, multi-core, and cross-die read, write, copy, triad bandwidth, and pointer-chasing latency on physical silicon.
2. **MMZ DMA Buffer Throughput**: Characterization of CPU $\leftrightarrow$ MMZ DMA buffer transfers in `NoCache` vs `Cached` modes.
3. **Hardware Shared Memory Expansion Guide**: Mathematical mapping and step-by-step U-Boot reconfiguration commands to expand NPU MMZ memory from **22 GiB up to 27 GiB or 28 GiB**, leaving 4–5 GiB for headless Linux.
4. **Memory Bandwidth Bottleneck & MoE Architectures**: Quantitative analysis of the autoregressive decoding memory wall, proving why **Mixture-of-Experts (MoE)** architectures are essential for achieving 10+ tokens/s on this SoC.

---

## 2. Hardware Memory Architecture Overview

The EIC7702 SoC on the FML13V03 mainboard integrates two identical dies (Die 0 and Die 1) interconnected via a high-speed proprietary Die-to-Die (D2D) bus:

```
┌──────────────────────────────────────── Die 0 ────────────────────────────────────────┐
│  [Cores 0-3 @ 1.4 GHz] ── [4 MB L3 Cache] ── [LPDDR5 Controller 0 (128-bit, 16 GiB)]  │
│  [NPU 0 (19.95 TOPS)]  ── [4 MB SPRAM 0]  ── Base: 0x00_8000_0000                     │
│  [DSP 0-3 @ 1.0 GHz]   ── [HAE 0]         ── MMZ Carveout: mmz_nid_0_part_0           │
└───────────────────────────────────────────┬───────────────────────────────────────────┘
                                            │ D2D Interconnect Bus
                                            │ (Cap: ~22 GB/s bidirectional)
┌───────────────────────────────────────────┴───────────────────────────────────────────┐
│  [Cores 4-7 @ 1.4 GHz] ── [4 MB L3 Cache] ── [LPDDR5 Controller 1 (128-bit, 16 GiB)]  │
│  [NPU 1 (19.95 TOPS)]  ── [4 MB SPRAM 1]  ── Base: 0x20_0000_0000                     │
│  [DSP 4-7 @ 1.0 GHz]   ── [HAE 1]         ── MMZ Carveout: mmz_nid_1_part_0           │
└──────────────────────────────────────── Die 1 ────────────────────────────────────────┘
```

### Key Hardware Characteristics:
- **Total Physical Memory**: 32 GiB LPDDR5 divided equally:
  - **Die 0 / Node 0**: 16 GiB physical address space spanning `0x00_8000_0000` to `0x04_8000_0000`.
  - **Die 1 / Node 1**: 16 GiB physical address space spanning `0x20_0000_0000` to `0x24_0000_0000`.
- **NUMA Topography**:
  - NUMA Node 0: Cores 0, 1, 2, 3 (Die 0).
  - NUMA Node 1: Cores 4, 5, 6, 7 (Die 1).
- **Die-to-Die (D2D) Bus**: Connects the two dies. Local memory accesses hit the local LPDDR5 controller directly; cross-die accesses must traverse the D2D interconnect, incurring higher latency and bus contention.

---

## 3. Physical Silicon NUMA Benchmark Results

A dedicated benchmark suite (`numa_bench.cpp`) was compiled with `-O3 -fopenmp` using `libnuma-dev` (`numa_alloc_onnode`) and executed directly on the physical `roma` hardware (Linux 6.6.92-eic7x, 8x RISC-V cores @ 1.4 GHz). 

All tests operated on **256 MiB streaming buffers**, far exceeding the 4 MiB on-chip L3 cache to guarantee true LPDDR5 DRAM access.

### 3.1. Single-Core Local vs Remote Performance

| Benchmark Case | Latency (ns) | Read (GB/s) | Write (GB/s) | Copy (GB/s) | Triad (GB/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Core 0 $\to$ Node 0 (Local)** | **206.57 ns** | **3.52** | **7.86** | **9.45** | **6.26** |
| **Core 0 $\to$ Node 1 (Remote)** | **665.91 ns** | 1.81 | 1.02 | 1.73 | 1.76 |
| **Core 4 $\to$ Node 1 (Local)** | **207.75 ns** | **3.43** | **7.27** | **9.25** | **6.94** |
| **Core 4 $\to$ Node 0 (Remote)** | **674.36 ns** | 1.76 | 1.01 | 1.68 | 1.69 |

#### Analysis:
1. **The Remote Latency Penalty**:
   - Local pointer-chasing latency is **~207 ns**.
   - Cross-die remote latency is **~670 ns** — a **$3.25\times$ latency penalty**.
2. **The Remote Bandwidth Penalty**:
   - Remote writes degrade by **$7.7\times$** (7.86 GB/s $\to$ 1.02 GB/s) due to D2D write-buffer serialization and remote cache-coherency handshakes.
   - Remote copy throughput degrades by **$5.5\times$** (9.45 GB/s $\to$ 1.73 GB/s).
   - Remote streaming reads drop by **$1.95\times$** (3.52 GB/s $\to$ 1.81 GB/s).

### 3.2. Multi-Core Die Saturation (4 Cores per Die)

To evaluate memory controller saturation, 4 threads were pinned to all 4 cores of a die:

| Benchmark Case | Read (GB/s) | Write (GB/s) | Copy (GB/s) | Triad (GB/s) |
| :--- | :---: | :---: | :---: | :---: |
| **Die 0 (Cores 0–3) $\to$ Node 0 (Local)** | **13.93** | **10.56** | **13.73** | **12.20** |
| **Die 0 (Cores 0–3) $\to$ Node 1 (Remote)** | 5.88 | 4.03 | 5.85 | 5.65 |
| **Die 1 (Cores 4–7) $\to$ Node 1 (Local)** | **13.75** | **9.63** | **14.00** | **11.19** |
| **Die 1 (Cores 4–7) $\to$ Node 0 (Remote)** | 5.55 | 3.97 | 5.74 | 5.61 |

#### Analysis:
- A single die reaches its memory controller ceiling at **~14.0 GB/s Read** and **~10.5 GB/s Write**.
- 4 remote cores saturating the D2D bus achieve only **~5.8 GB/s Read** and **~4.0 GB/s Write** — less than half the local throughput.

### 3.3. Full-System 8-Core Concurrent Bandwidth & D2D Interconnect Cap

To test aggregate throughput and D2D interconnect saturation, all 8 cores were exercised concurrently:

| Concurrent Workload | Configuration | Aggregate Triad Bandwidth |
| :--- | :--- | :---: |
| **Concurrent Independent Local** | Die 0 $\to$ Node 0 (4 cores) **+** Die 1 $\to$ Node 1 (4 cores) | **47.93 GB/s** |
| **Concurrent Cross-Die Interconnect** | Die 0 $\to$ Node 1 (4 cores) **+** Die 1 $\to$ Node 0 (4 cores) | **21.85 GB/s** |

#### Crucial Architectural Takeaways:
1. **Total System Memory Bandwidth is 48 GB/s**: When both dies access their respective local memory channels concurrently, the system delivers an aggregate **47.93 GB/s** of memory bandwidth.
2. **D2D Interconnect Ceiling is ~22 GB/s**: When cross-die traffic is forced simultaneously in both directions, the D2D bus caps bidirectional aggregate throughput at **21.85 GB/s**.
3. **Software Implication**: Processes orchestrating Die 0 (NPU 0) must pin CPU threads to Cores 0–3 and allocate memory on NUMA Node 0. Processes orchestrating Die 1 (NPU 1) must pin to Cores 4–7 and allocate on NUMA Node 1. Crossing NUMA nodes cuts bandwidth in half and triples latency.

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

## 7. Conclusions & Recommended Execution Rules

1. **Memory Allocation**:
   - For workloads targeting 7B–14B models, reconfigure U-Boot to **Config A (27 GiB NPU MMZ, 5 GiB Linux)**. This provides 13.5 GiB per die, sufficient for 14B INT8 models or 7B models with 32K context windows.
2. **Buffer Access Policy**:
   - CPU reads from MMZ must **never use `NoCache` mode** (10 MB/s bottleneck). Always allocate with `ES_MEM_ALLOC_ATTR_E::Cached` when CPU readback is needed.
3. **NUMA Thread Binding**:
   - Die 0 runtime workers must be strictly bound to CPU cores 0–3 with `numa_alloc_onnode(..., 0)`.
   - Die 1 runtime workers must be strictly bound to CPU cores 4–7 with `numa_alloc_onnode(..., 1)`.
   - Never allow cross-die memory allocation, which degrades bandwidth by up to $7.7\times$ and triples access latency.
