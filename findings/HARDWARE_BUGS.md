# ESWIN EIC7700X / EIC7702 (`FML13V03` / `roma`) Hardware, Kernel, Driver & Compiler Errata

Comprehensive reference catalog of silicon defects, kernel driver bugs, userspace DMA/NPU library hazards, and `EsAAC` compiler quirks discovered while bringing up `LFM3-1.7B` and `LFM2.5-8B-A1B` on the dual-die **ESWIN EIC7700X / EIC7702** (`DeepComputing FML13V03`).

---

## Errata Summary Matrix

| ID | Component / Layer | Severity | Core Defect / Trigger | Verified Mitigation / Workaround | Primary Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HW-01** | Die 0 NPU Silicon (`51c00000.eswin-npu`) | **CRITICAL** | Intermittent dynamic timing margin collapse at `1.50 GHz @ 1050 mV` (`llc=1188 MHz`) on `W16A16` `down_proj` under 8/8-slot load, striking **both** PEC column #0 (`t % 8 in {0, 1}`) and PEC column #2 (`t % 8 in {4, 5}`) across multiple randomized/toggle seeds | Bypass slots `{4, 5}` (`kT8SafeSlots = {0,1,2,3,6,7}` in `lfm2_moe.cpp`: `0/10,000` deterministic soak + `0/51,200` 16-seed randomized/toggle sweep across `safe6`), or cap Die 0 at `1.04 GHz @ 900 mV` (`0/2,000` at 8/8) | [`FINDINGS.md` §52.8](../FINDINGS.md) |
| **HW-02** | `libes_memory.so` (`ES_SYS_MemFlushCache`) | **HIGH** | Buffers `> 1 MiB` flush **only** target `nid` CPU caches (`SYS_MemFlushCacheAllByNid`); smaller buffers flush whole allocation (`DMA_BUF_IOCTL_SYNC`) | Pin filling thread to Core `0` (Die 0) / Core `4` (Die 1) before `flush_cache()`; use compact geometry surfaces (`-58.7 ms/tok`) | [`FINDINGS.md` §44, §47.1, §52.8](../FINDINGS.md) |
| **HW-03** | Linux Kernel (`eswin_rsvmem_heap_mmap`) | **CRITICAL** | 32-bit signed truncation (`sext.w a3, s2` at `0xffffffff80d00f86`) corrupts `mmap()` lengths `>= 2.0 GiB` | Split MMZ allocations into `< 2.0 GiB` chunks (`m_w_blob[2][3]`), abort on MMZ failure (no OS RAM fallback), `POSIX_FADV_DONTNEED` on weight reads | [`FINDINGS.md` §52.1](../FINDINGS.md) |
| **HW-04** | Kernel Driver (`eic7700_npu.ko`) | **CRITICAL** | Hard 62-model/die ceiling; missing `ERR_PTR(-EINVAL)` check causes `destroy_frame.part.0+0x68/0x140` kernel Oops + multi-thread `submitSync` stalls | Route D Class B shared archetypes (`12–16 / 62` loadables/die) + isolated per-die `WorkerPool` processes | [`FINDINGS.md` §33.3–§33.4](../FINDINGS.md) |
| **HW-05** | `libes_npu.so` & `eic7700_npu.ko` | **HIGH** | Repeated `ES_NPU_SetDevice` (`setDevice:202`) floods `systemd-journald` (`> 250k` lines); `SCHED_FIFO` starves `sshd`; `~0.35 ms` cross-die `ioctl` lock | Call `set_device` once at worker init, keep `SCHED_OTHER` + `CPU_SET`, avoid cross-die locks on small subgraphs (Hybrid `TP=2`) | [`FINDINGS.md` §50.1, §52.5](../FINDINGS.md) |
| **HW-06** | NVDLA Hardware Surface Layout | **MEDIUM** | Partial chunks read uninitialized tail bytes inside declared descriptor; INT8 `FEATURE` uses 32-ch atoms `[C/32, T, 32]`; `T=64` SDP planar transpose doubles `QuantM` to `256 B` | Zero-fill & flush DMA buffers on reset (`reset_dma_surfaces()`); align `mlp_t1` `CvtPrecision_QuantM` up to `256 B` | [`FINDINGS.md` §45.3, §47.2–§47.3](../FINDINGS.md) |
| **HW-07** | `EsAAC` Proprietary Compiler | **MEDIUM** | Silent exit `0` on `has no graph`; SIGSEGV on `per_tensor` or `/Conv.data_type="int16"`; `w2` `int16` `32,767` ceiling saturation; `mlp_t64` 4-pass M-loop bug | Parse logs & count outputs; keep `/Conv="int8"` + `output_data_type=["int16"]`; tune `truncate`; use `mlp_t8` `kT8SafeSlots` | [`FINDINGS.md` §33.2, §50.2, §52.5–§52.6](../FINDINGS.md) |

---

## HW-01 — Die 0 NPU `1.50 GHz` Dynamic Timing Margin Hazard (`W16A16` `down_proj`, PEC Columns #0 & #2: `t % 8 in {0, 1, 4, 5}`)

- **Severity**: **CRITICAL** (silent intermittent numerical corruption at full stock `1.50 GHz` clock).
- **Affected Hardware**: `roma` Die 0 NPU (`51c00000.eswin-npu`). Identical Die 1 NPU (`71c00000.eswin-npu`) exhibited **zero** errors (`0 / 2,000` and `0 / 356` experts in Stage 1B) on bit-identical weights and inputs.
- **Operating Environment Observed**:
  - `/proc/device-tree/opp-table@npu` & devfreq (`51c00000.eswin-npu` / `71c00000.eswin-npu`, `simple_ondemand` governor, stock `cur_freq = max_freq = 1500000000`):
    - `520 MHz @ 800 mV`
    - `750 MHz @ 800 mV`
    - `1.04 GHz @ 900 mV` (`llc clk rate: 800000000` per `dmesg`)
    - `1.50 GHz @ 1050 mV` (`llc clk rate: 1188000000` per `dmesg`)
  - Regulators (read-only): `/sys/class/regulator/regulator.5` (`NPU_SVCC`, Die 0) and `/sys/class/regulator/regulator.6` (`NPU_SVCC2`, Die 1), both `uV=1050000` (`min=700000`, `max=1120000`) at `1.50 GHz`.
  - Operating board temperatures across Stage 1–2 sustained loads: `tz0 = 47.2–52.8°C`, `tz1 = 41.8–49.0°C` (no thermal throttling).
  - `/sys/module/eic7700_npu/parameters/*`: `frame_timeout=1000000`, `loop_buf_cnt=1500`, `loop_buf_enable=0`, `npu_debug_control=1`, `npu_dump_data_enable=1`.

### Initial Failure Signature (`3,818`-Token `LFM2.5` Determinism Probe)
During 4K-token (`3,818` tokens) prefix-cache determinism testing of `LFM2.5-8B-A1B` (`lfm2_moe`), two consecutive cold prefill passes diverged intermittently at full `1.50 GHz` (`chunk=2, pos=128, L12, stage=EXP_D0, die=0, e=12`). Bit-level inspection of the `lfm2_expert_mlp_t8_classb.model` INT16 output (`[C=2048, T=8]`) showed:
- Errors occur with **~1%–2.75% probability** on Die 0 `L12 e=12` `mlp_t8` (`ops 48..67`, the `W16A16` `down_proj` split-M blocks), while earlier `mlp_t64` diagnostics on Die 0 also hit slots `t=[12..13]` and `t=[28..29]` (`t % 8 in {4, 5}`, PEC column #2).
- Every recorded `L12 e=12` mismatch struck **one** of three 256-channel `splitM` blocks:
  - `c=[0..255]`: `n_diff=179` in `t=[4..5]`
  - `c=[258..511]`: `n_diff=192`, `cnt_t=[0,0,0,0,96,96,0,0]`, `cnt_cg=4:52 5:50 6:47 7:43` (`diffs_t03 = 0`)
  - `c=[1024..1279]`: `n_diff=206`, `cnt_t=[0,0,0,0,103,103,0,0]`, `cnt_cg=16:46 17:48 18:55 19:57` (`diffs_t03 = 0`)

### Controlled Isolation Matrix (`L12 e=12`, `N = 2,000` Iterations Each)

1. **Coherency & Archetype Isolation (`LFM2_STRESS_EXP` Conditions A–E, `5,000×2` at `1.50 GHz`)**:
   - `CondA` (`mlp_t8`, `CacheMode::Cached`, Dual-Die): `die0=4` (`id=12`, `t=[4..5]`, `reflush_diff=n_diff`, `resubmit_diff=0`), `die1=0`.
   - `CondB` (`mlp_t8`, `CacheMode::NoCache`, Dual-Die): `die0=1` (`id=12`, `c=[258..511]`, `t=[4..5]`, `cur=6839 vs 8224`, `resubmit_diff=0`), `die1=0` — **definitively proves the fault is NOT CPU cache coherency**.
   - `CondC` (`mlp_t8`, `CacheMode::NoCache`, Die 0 serial): `0`.
   - `CondD` (`mlp_t1`, `CacheMode::Cached`, Dual-Die): `0 / 0`.
   - `CondE` (`conv_in_t64`, `CacheMode::Cached`, Die 0, `N=1,000`): `0`.

2. **Focused `L12 e=12` Root-Cause Matrix (`N = 2,000`: `1.50 GHz` vs. `1.04 GHz` Devfreq Cap)**:

| Test ID | Condition (`startup_diffs=0`, `d1_w_copy_diffs=0`) | Result at `1.50 GHz` | Result at `1.04 GHz` | What It Proved |
| :--- | :--- | :---: | :---: | :--- |
| `T1a` | Die 0 serial repeating `e=12` in-place | **`mis=21`** (`1407.3 ms`) | **`mis=0`** (`1553.4 ms`) | Single-die `1.50 GHz` timing failure even without Die 1 concurrency |
| `T1b` | Die 0 serial alternating `e=10` / `e=12` | `mis10=0`, **`mis12=42`** (`2794.1 ms`) | **`mis10=0, mis12=0`** (`3046.6 ms`) | Data/pattern-sensitive electrical margin on `e=12` |
| `T1c` | Dual-die alternating (`Die 0: e=10,12`, `Die 1: e=11,13`) | **`mis_target=55`**, `t03=0`, `t47=10918` | **`mis_target=0`**, `t47=0` | Higher toggle/switching stress increases `1.50 GHz` failure rate (`2.75%`) |
| `T2` | Dual-die with `e=12` copied to fresh Node-0 `DmaBuffer` | **`mis_target=21`**, `t03=0`, `t47=4015` | **`mis_target=0`**, `t47=0` | Rules out `m_w_blob` DRAM bank/location corruption on Die 0 |
| `T3` | Dual-die executing **identical** `e=12` copy on **Die 1** | **`mis_target=0`**, `t47=0` (`3853.5 ms`) | **`mis_target=0`**, `t47=0` | Die 1 PEC column #2 has sufficient margin at `1.50 GHz`; Die 0 does not |
| `T4` | Dual-die `b_cnt=4` (`t=4..7` zeroed) on Die 0 | **`mis_target=0`**, `t47=0` (`3543.0 ms`) | **`mis_target=0`**, `t47=0` | Keeping `t=4..7` idle/zero eliminates Die 0 errors even at `1.50 GHz` |
| `T5` | Dual-die `mlp_t1` (`W=1`) on Die 0 | **`mis_target=0`** (`3096.5 ms`) | **`mis_target=0`** | `T=1` single-column decode avoids 8-slot switching load |

### Full-SoC Milestone 18 / Phase G Silicon Sweep (2026-11-21)
Using the standalone cross-compiled C++17 `stress_hw01_probe` (dual `64 MiB` staging buffers, Core `0`/`4` cache-flush pinning), we swept **all** `LFM3-1.7B` archetypes and **all 712** `LFM2.5-8B-A1B` experts + **48** `INT16` `T=64` operators across Die 0 and Die 1:

1. **Stage 1A — `LFM3-1.7B` All-Archetype Sweep @ `1.50 GHz`**:
   - `TP=2` `mlp_t64` all 24 layers (Die 0 & Die 1, `N=100`, 48 rows, `33.6 s`), `TP=2` `mlp_t64` cross-die (`L00/L06/L12/L22`, `N=1,000`, `44.3 s`), `TP=2` `mlp_t8` cross-die (`L00/L06/L12/L22`, `N=1,000`, `34.1 s`), `PP=2` `mlp_t64` cross-die (`L00/L06/L11` full 7680-width `63.9 MB`, `N=500`, `20.5 s`), plus `conv_in_t64` (`12.1 s`), `proj2048_t64` (`17.4 s`), and `attn_qkv_t64` (`10.1 s`, `N=300` each): **`0` failures everywhere** (`ref_diffs=0`).
2. **Stage 1B — `LFM2.5-8B-A1B` All 48 `INT16` `T=64` Ops & All 712 Experts @ `1.50 GHz` (`slots=all`)**:
   - All `48` `INT16` `T=64` operators (`18` `conv_in` `20.1 s`, `24` `proj2048` `18.5 s`, `6` `attn_qkv` `6.0 s`, `N=150` each): **`0` failures**.
   - All **`712` experts** (`356` Die 0 even + `356` Die 1 odd, `N=100` each, `320.9 s`):
     - **Die 1**: **`0 / 356` failing experts**.
     - **Die 0**: **`3 / 356` failing experts**, proving `HW-01` ALSO strikes **PEC column #0 (`t=[0..1]`)**, not only PEC column #2 (`t=[4..5]`):
       - `lfm2_l09_exp16`: `mis_act=1/100`, **`mod8_act=[106,107,0,0,0,0,0,0]` (`t=[0..1]`, PEC col #0!)**, `c=[0..255]`, `first=(c=0,t=1,ref=11494,cur=6797,resub=11494)`
       - `lfm2_l11_exp08`: `mis_act=2/100`, `mod8_act=[0,0,0,0,202,210,0,0]` (`t=[4..5]`, PEC col #2), `c=[1281..1535]`, `first=(c=1281,t=4,ref=-7480,cur=-7881,resub=-7480)`
       - `lfm2_l20_exp18`: `mis_act=4/100`, **`mod8_act=[684,676,0,0,0,0,0,0]` (`t=[0..1]`, PEC col #0!)**, `c=[0..255]`, `first=(c=0,t=1,ref=14148,cur=12822,resub=14148)`
3. **Stage 1C / 1D — Deep Thermal Soak on Top-3 Die-0 Experts (`lfm2_l12_exp12`, `lfm2_l20_exp18`, `lfm2_l11_exp08`, alternating dual-die pairs, `tz0` up to `52.8°C`)**:
   - `slots=all` @ `1.50 GHz` (`N=2,000`, `91.7 s`):
     - `lfm2_l20_exp18_pair`: **`mis_act=61/2,000` (`3.05%`)**, `diffs_act=20,740`, **`mod8_act=[10263,10477,0,0,0,0,0,0]` (`t=[0..1]`, PEC col #0!)**, `c=[0..1278]`, `resub_mis=0`.
     - `lfm2_l11_exp08_pair`: **`mis_act=6/2,000` (`0.30%`)**, `diffs_act=1,236`, `mod8_act=[0,0,0,0,606,630,0,0]` (`t=[4..5]`, PEC col #2), `c=[1281..1535]`, `resub_mis=0`.
     - `lfm2_l12_exp12_pair` scored `0/2,000` on `stress_hw01_probe`'s deterministic activation vector (vs `55/2,000` on `lfm2_moe`'s seed — confirming strong activation-pattern dependence); all Die 1 pairs scored `0/2,000`.
   - **`slots=safe6` (`{0,1,2,3,6,7}` active, `{4,5}` zeroed) @ `1.50 GHz` (`N=10,000` alternating pairs = `60,000` Die 0 submits + `60,000` Die 1 submits, `450.5 s`)**: **`0 / 10,000` active & `0 / 10,000` padded errors** across all 12 rows (`lfm2_l20_exp18`'s `t=[0..1]` failure rate dropped from `61/2,000` to **`0/10,000`** even with identical non-zero INT8 vectors in slots `{0, 1}`!).
   - **`slots=safe4` (`{0..3}` active) @ `1.50 GHz` (`N=5,000`, `226.5 s`)**: **`0 / 5,000`** active & padded errors across all 12 rows.
   - **`slots=all` @ `1.04 GHz` (`900 mV`, `llc=800 MHz`, `N=2,000`, `92.8 s`)**: **`0 / 2,000`** across all 12 rows (`restored_f0=1500000000`, `restored_f1=1500000000` verified).
4. **Refined Physical Root-Cause Interpretation**: `HW-01` is **NOT** a single-column defect isolated to PEC column #2. Rather, it behaves as an **array-wide instantaneous dynamic IR-drop / Fmax timing droop on Die 0 at `1.50 GHz @ 1050 mV` (`llc=1188 MHz`)** when all `8/8` token slots switch simultaneously during `W16A16` `down_proj`, tripping multiple critical paths (`t=[0..1]` PEC col #0 and `t=[4..5]` PEC col #2). Cutting simultaneous active slot occupancy to `<= 6/8` (`75%` via `kT8SafeSlots`) or lowering OPP to `1.04 GHz @ 900 mV` restores margin across all columns.

### Verified Mitigation
To preserve full `1.50 GHz` performance without downclocking the SoC (`1.04 GHz` cuts throughput by ~30%), [`src/LFM2.5-8B-A1B/lfm2_moe.cpp`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/lfm2_moe.cpp) maps up to **6 gathered tokens** per `mlp_t8` dispatch onto **`kT8SafeSlots[6] = {0, 1, 2, 3, 6, 7}`** in `pack_gathered_int8_t8()` and `run_die_experts()`, explicitly **zero-padding slots `{4, 5}`** while stepping `base += 6` and unpacking `unpack_t = (b_cnt <= 4) ? b_cnt : 8`.
- Verified on `roma` at full `1.50 GHz` (`cur_freq=1500000000`):
  - **16-Seed Randomized & Worst-Case Toggle `HW-01` Sweep (`stress_hw01_probe --seeds 16`, `200` iters/seed, `s0`=det, `s1`=`±127` toggle checkerboard, `s2..s15`=`xorshift64*` non-zero INT8 across `lfm2_l20_exp18`, `lfm2_l11_exp08`, `lfm2_l12_exp12`, `lfm2_l09_exp16` + Die 1 pairs)**:
    - **`slots=all` (`8/8`)**: Die 0 failed **`91` times** across seeds `s0, s11, s12, s14, s15` (`l20_exp18_pair`: `33` `s0` on `t=[0..1]`; `l20_exp18_pair_alt`: `29` `s15` on `t=[0..1]`; `l11_exp08_pair`: `9` `s0` on `t=[4..5]`; `l12_exp12_pair_alt`: `2` `s12,s14` on both `t=[0..1]` & `t=[4..5]`; `l09_exp16_pair`: `18` `s11,s15` on both `t=[0..1]` & `t=[4..5]`). All Die 1 pairs = `0`.
    - **`slots=safe6` (`{0,1,2,3,6,7}`, zero-padding `{4,5}`)**: **ALL 16 rows scored `mis_act=0, mis_pad=0` (`0 / 51,200` submits)** across the exact same 16 seeds, confirming `kT8SafeSlots` eliminates both PEC col #0 (`t=[0..1]`) and PEC col #2 (`t=[4..5]`) errors without changing `kT8SafeSlots` or downclocking to `1.04 GHz`.
    - **`LFM3-1.7B` `TP=2` (`mlp_t64` & `mlp_t8` `crossdie`, `L00/L06/L12/L22`, 16 seeds × 100 iters)**: **`0 / 25,600`** errors (`Die 0`: `5,618.6–5,783.2 ms` `t64`, `2,139.2–2,184.7 ms` `t8`; `Die 1`: `6,212.6–6,377.3 ms` `t64`, `2,468.6–2,564.1 ms` `t8`).
  - **2× `3,818`-token `--verify-prefix-cache` (Stage 2)**: **5/5 PASS** both runs, **`max_abs=0.000000`**, **`cos=1.00000000`**, `4/4` greedy match.
  - **`--verify-golden golden_ref.bin`**: **`0.97924192`** min-layer cosine, **`0.99837211`** 128K logits cosine, `4/4` greedy match, **`11.00–11.38 tok/s`** decode.

### Open Silicon Investigation Items (Follow-Up Status)
1. **Multi-seed randomized activation sweep on `kT8SafeSlots` — [CLOSED: `0 / 51,200`]**: Verified across `16` randomized + worst-case `±127` toggle seeds (see above).
2. **Multi-seed randomized soak on `LFM3-1.7B` `mlp_t64` / `mlp_t8` — [CLOSED: `0 / 25,600`]**: Verified immune across `16` randomized/toggle seeds (see above).
3. **Die 0 vs. Die 1 `~1.87×` wall-time asymmetry in `stress_hw01_probe` — [CLOSED: NUMA PINNING BUG IN PROBE]**: Caused because `load_job_to_die(1, ...)` left the spawning main thread pinned on `Core 4` (NUMA Node 1), forcing Die 0 worker threads onto NUMA Node 1 (`29.9 s` vs `15.9 s`). Adding `pin_to_core(owning_core[d])` inside `run_single()` restored Die 0 (`Core 0`) speed ahead of Die 1 (`Core 4`): `5,384.8–6,067.9 ms` vs `6,714.9–7,322.2 ms`.
4. **PMIC / OPP voltage margin — [OPEN]**: `NPU_SVCC` and `NPU_SVCC2` both operate at `1050 mV` (`max=1120 mV`) at `1.50 GHz`; determine whether Die 0 vs Die 1 Fmax spread is pure silicon process variation or board power-rail droop.

---

## HW-02 — `libes_memory.so` (`ES_SYS_MemFlushCache`): NUMA-Local Cache Flush Only & Whole-Buffer Flush Penalty

- **Root Cause (Disassembly of `libes_memory.so`)**:
  - `ES_SYS_MemFlushCache(fd)` accepts only a file descriptor (no offset or length).
  - For buffers `<= 1 MiB`, it invokes `DMA_BUF_IOCTL_SYNC` (`0x40086200`), performing complete range coherency.
  - For buffers **`> 1 MiB`** (including all weight chunks and large staging buffers), it calls **`SYS_MemFlushCacheAllByNid(nid)`**, flushing CPU L1/L2 caches **only on the cores belonging to `nid`**, leaving dirty cache lines resident if written from the remote NUMA node.
  - In addition, flushing full `2 MiB` driver-aligned output buckets on every subgraph cost `~58.7 ms/token` (`FINDINGS.md` §44, §47.1).
- **Verified Mitigations**:
  1. In [`lfm2_moe.cpp`](file:///home/chri/aixigua/romanaccio/src/LFM2.5-8B-A1B/lfm2_moe.cpp) (`Lfm3Engine::init()`), strictly pin the calling thread via `sched_setaffinity()` to `Core 0` (Die 0) / `Core 4` (Die 1) while `memcpy`-ing and calling `flush_cache()` on each die's weight chunks, then restore `pin_openmp_threads_numa()`.
  2. Size DMA output buffers to exact write-extent geometry (`32 KiB` for `T=1`, `256/768 KiB` for `T=64`), reclaiming **`58.7 ms/token`** in `LFM3-1.7B` (`FINDINGS.md` §47.1).

---

## HW-03 — Linux Kernel `eswin_rsvmem_heap_mmap`: 32-Bit Signed `vma_size` Truncation (`sext.w` at `0xffffffff80d00f86`)

- **Root Cause (Disassembly of `/boot/vmlinux-6.6.92-win2030`)**:
  ```asm
  ffffffff80d00f66: ld      s2, 8(s0)          # vma->vm_end
  ffffffff80d00f68: ld      a0, 0(s0)          # vma->vm_start
  ffffffff80d00f6a: sub     s2, s2, a0         # s2 = vma_size (64-bit size_t)
  ...
  ffffffff80d00f86: sext.w  a3, s2             # BUG: 32-bit sign extension of vma_size!
  ffffffff80d00f9c: jal     remap_pfn_range
  ```
  - `2.0 GiB <= size < 4.0 GiB`: sign-extends bit 31 into bits `32..63`, failing `remap_pfn_range` with `-EINVAL`.
  - `size == 4.0 GiB`: truncates lower 32 bits to `0` (`-EINVAL`).
  - `size == 5.166 GiB`: silently maps only `1.166 GiB` (`0x4AA60000`), leaving the upper `4.0 GiB` unmapped.
  - Combined with `DmaBuffer`'s legacy `posix_memalign` fallback and Linux VFS page cache (`10.33 GiB` across `.bin` reads), this previously exhausted the `8.98 GiB` Linux OS RAM and froze SSH via OOM/`kswapd` (`FINDINGS.md` §52.1).
- **Verified Mitigations**:
  1. Abort immediately in [`DmaBuffer::DmaBuffer`](file:///home/chri/aixigua/romanaccio/src/eswin_runtime/src/backend/dma_buffer.cpp) on any failed MMZ allocation or `mmap` `>= 256 MiB` (no OS RAM fallback).
  2. Partition each die's `5.166 GiB` weights into **3 `< 2.0 GiB` MMZ chunks** (`m_w_blob[2][3]`) with `w_slice(die, offset, size)`.
  3. Evict Linux page cache every `64 MiB` via `POSIX_FADV_DONTNEED` strictly while loading `weights_die{0,1}.bin` (never on `.kvb` prefix cache files), preserving **`8.78 GiB` free OS RAM**.

---

## HW-04 — Kernel Driver `eic7700_npu.ko`: 62-Model/Die Limit & Missing `ERR_PTR` Check (`destroy_frame.part.0`)

- **Root Cause (`FINDINGS.md` §33.3–§33.4)**:
  - Each die supports a hard ceiling of **62 resident `.model` loadables**. Loading a 63rd fails, and the subsequent unload/submit dereferences an `ERR_PTR(-EINVAL)` (`ffffffffffffffea`) stored after `io_tensor_to_io_addr:143` failure inside `destroy_frame.part.0+0x68/0x140`, triggering a fatal kernel Oops that renders the NPU die unusable until reboot.
  - Multi-threaded submissions from a single process to both dies trigger `submitSync` / condition-variable stalls and `destroy_frame` races.
- **Verified Mitigations**:
  - **Route D Class B Archetypes**: Share compiled `.model` topologies across layers and all 712 MoE experts via DMA weight-offset slicing (`DevBuf.offset`), holding total resident models at **12–16 / 62 per die** (gate `< 55`).
  - **Isolated `WorkerPool` Processes**: Isolate Die 0 and Die 1 NPU contexts across forked worker processes (`src/eswin_runtime/src/backend/worker_pool.cpp`).

---

## HW-05 — `libes_npu.so` & `eic7700_npu.ko`: `setDevice:202` Syslog Flood, `SCHED_FIFO` Starvation & `~0.35 ms` Cross-Die `ioctl` Lock

- **Symptoms & Root Causes (`FINDINGS.md` §50.1, §52.5)**:
  1. Calling `ES_NPU_SetDevice` (`driver.set_device`) inside the IPC dispatch loop on an already-bound worker thread emits `[ERR][NPU][setDevice:202] send msg failed` via syslog on every call (`~850` calls/sec -> `> 250,000` journal lines), saturating `systemd-journald` and hanging SSH during long `-n 4096` generations.
  2. Running spin-yielding `m_tp_thread` and worker threads at `SCHED_FIFO` priority 1 across busy OpenMP cores starves `sshd` and network softirqs.
  3. Concurrent `ES_NPU_Submit` / `ES_SYS_MemFlushCache` ioctls across `/dev/npu0` and `/dev/npu1` serialize behind a **`~0.35 ms` kernel lock** (`~0.70 ms` floor per dual-die pair), making small-tensor `TP=2` (`conv_in`, `attn_qkv`, `proj2048`) slower than `PP=2` (`+13.9 ms/tok` penalty).
- **Verified Mitigations**:
  - Call `driver.set_device(m_die_id)` strictly **once** at worker initialization (`ZERO setDevice errors` in `journalctl`).
  - Keep worker/`m_tp_thread` on `SCHED_OTHER` with explicit NUMA core affinity (`CPU_SET`).
  - Use **Hybrid `TP=2`** (keep small ops single-die, parallelize only large `MLP` and `LM Head` layers) and **single-IPC batched dispatches** (`SubmitExpertBatch`, `SubmitLmHeadBatch`).

---

## HW-06 — NVDLA Hardware Surface Semantics & `T=64` SDP Planar-Transpose Layout

- **Findings (`FINDINGS.md` §45.3, §47.2–§47.3)**:
  1. **Stale Tail Reads**: When a DMA descriptor exceeds valid tensor bounds (`T_valid < T`), NVDLA reads uninitialized buffer bytes between tensor end and descriptor end, breaking determinism unless buffers are zero-primed (`reset_dma_surfaces()` on reset/restore).
  2. **32-Channel INT8 Atom**: NVDLA `T > 1` INT8 `FEATURE` surfaces pack 32-byte atoms as `[C/32, T, 32]` rather than FP16's `[C/16, T, 16]`.
  3. **`T=64` SDP Planar Transpose**: `mlp_t64` folds `[c=64, w=64]` `w2` slices into `[c=128, w=32]`, requiring **256-byte** (`2 × 64` INT16) `CvtPrecision_QuantM` tables (`align_mlp_t1_to_t64` in `LFM3`).

---

## HW-07 — `EsAAC` Compiler Quirks

- **Summary of Verified Compiler Workarounds (`FINDINGS.md` §33.2, §50.2, §52.5–§52.6)**:
  - **Silent Zero Exit on Failure**: `EsAAC` exits `0` even on truncated/invalid graphs (`Model <x>.onnx has no graph`); scripts must verify output counts and grep logs.
  - **`per_tensor` & `/Conv INT16` Crashes**: `EsAAC` SIGSEGVs (exit 139) if `quantized_method = "per_tensor"` or `/Conv.data_type = "int16"`; use `"per_channel"` and set `model_info.output_data_type = ["int16"]` while keeping `/Conv.data_type = "int8"`.
  - **`CvtPrecision_QuantM` `±32,767` Saturation**: Wide-range calibration layers force large `truncate` values that saturate small-magnitude layers/channels in `w2` or the INT8 LM head; solved via calibrated `truncate` shifts (`sh=25` `mlp_tp2`, `sh2=27` `expert_mlp`) and `INT16` `dyntok` operators/LM head (`Ops out=int16, ExpertMLP out=int16, LMHead=int16`).
  - **`lfm2_expert_mlp_t64` 4-Pass M-Loop + Column #2**: Batched `T=64` expert execution triggers both a 4-pass M-loop failure and `HW-01` (`t % 8 in {4, 5}`); avoid using `expert_mlp_t64` on NPU, using `mlp_t8` `kT8SafeSlots` instead.
