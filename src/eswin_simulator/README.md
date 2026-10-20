# Open EsSimulator (`eswin_simulator`)

**eswin_simulator** is an open-source, 64-bit cross-platform functional simulator for the ESWIN EIC7700X / EIC7702 NPU and DSP loadable container (`.model` FlatBuffers).

It replaces the proprietary, 32-bit x86-only vendor `EsSimulator` binary with a clean, extensible, platform-independent execution environment.

---

## 1. Features & Milestone 1 Deliverables

### Milestone 1: Model Loader & Virtual Address Space Inspector
- **Universal `.model` Reader**: Full FlatBuffers (`NVDA` identifier) parsing of loadables across all model topologies (vision CNNs, transformer blocks, split dual-die LM heads).
- **Virtual Memory Architecture**: Simulates 64-bit DRAM address space alongside the on-chip 4 MiB SPRAM (`SRAM`), enforcing page alignments and buffer boundaries.
- **Static Blob Ingestion**: Automatically resolves and populates embedded weight blobs (`W_0`, tiled splits), hardware command packets (`kmd-task-op_list`, `kmd-task-surf_list`), and LUT tables into simulated physical memory.
- **Model Inspector CLI (`eswin-sim`)**:
  - Detailed terminal tables of input/output tensor shapes, data formats (`NCHW`/`NHWC`), and precisions (`HALF`/`INT8`/`FLOAT`).
  - Memory utilization breakdown (DRAM vs on-chip SPRAM footprint).
  - Submodel task topology (UMD vs KMD tasks) and relocation tables.
  - Machine-readable `--json` output format for toolchain automation and continuous integration.

---

## 2. Quick Start & CLI Usage

### Installation
From the repository root (or within a virtual environment):
```bash
cd src/eswin_simulator
pip install -e .
```

### Inspecting a Model (`--info`)
```bash
# Basic inspection of input/output tensors, memory, and submodels
eswin-sim -m path/to/model.model --info

# Detailed inspection including memory regions and relocation tables
eswin-sim -m path/to/model.model --info -v

# Programmatic JSON inspection
eswin-sim -m path/to/model.model --json
```

### Example Output
```
================================================================================
  ESWIN EIC7700X Open NPU/DSP Functional Simulator (eswin-sim)
  Milestone 1: Model Loader & Virtual Address Space Inspector
================================================================================

[Model Information]
  • File Path         : test_conv_npu_b1.model
  • Container Magic   : NVDA [OK]
  • Loadable Version  : 0.7.0
  • File Size         : 37.67 KiB
  • Tasks / Submodels : 2
  • Blobs Embedded    : 12

[Input Tensors (1)]
  Name                     Shape              Type     Format   Size         Domain  
  ------------------------------------------------------------------------------
  input                    (1, 3, 32, 32)     HALF     NCHW     6.00 KiB     SYSTEM  

[Output Tensors (1)]
  Name                     Shape              Type     Format   Size         Domain  
  ------------------------------------------------------------------------------
  output                   (1, 16, 30, 30)    HALF     NCHW     28.12 KiB    SYSTEM  

[Simulated Memory Utilization]
  • DRAM Regions (SYSTEM) : 0.11 MiB (114096 bytes across 17 regions)
  • SPRAM (On-Chip SRAM)  : 4.00 MiB / 4.00 MiB (100.0%)
  • Total Address IDs     : 18
  • Relocation Entries    : 4
```

---

## 3. Python API

```python
from eswin_simulator.loader import LoadableModel
from eswin_simulator.memory import VirtualAddressSpace
import numpy as np

# 1. Load .model container
model = LoadableModel.from_file("test_conv_npu_b1.model")
print(f"Loaded {model.file_path}, version {model.version}")

# 2. Inspect input/output tensors
for t in model.input_tensors:
    print("Input:", t.name, t.shape, t.data_type, t.size)

for t in model.output_tensors:
    print("Output:", t.name, t.shape, t.data_type, t.size)

# 3. Initialize simulated virtual memory
vas = VirtualAddressSpace(model)
print("Memory Stats:", vas.stats())

# 4. Bind input tensor data
input_tensor = model.input_tensors[0]
dummy_data = np.random.randn(*input_tensor.shape).astype(input_tensor.dtype)
vas.bind_input(input_tensor, dummy_data)

# 5. Read back tensor from virtual memory
reconstructed = vas.read_tensor(input_tensor)
assert np.allclose(dummy_data, reconstructed)
```

---

## 4. Running the Test Suite

Run unit tests via `pytest`:
```bash
pytest -v
```

All 9 unit tests cover:
- Binary magic validation (`NVDA`) and error detection on corrupted containers.
- 4D Tensor descriptor parsing (shape, strides, data format, data type, element sizes).
- Memory allocation across 64-bit DRAM and the 4 MiB on-chip SPRAM.
- Static weight blob population and verification.
- Bit-exact tensor binding, buffer readback, and overflow protection.
