"""Convert a loadable's blob-backed weight regions into runtime-bound (Class B) tensors.

eswin_compiler Milestone 7. This generalises the single-weight `dynamic_weight` path in
`gemm_builder.py` to *any* compiled loadable, including the SwiGLU expert-FFN archetype that
DeepSeek-V2-Lite needs, and it works on vendor-compiled models as well as our own.

## What Class A and Class B mean

Every `memory_list` entry is one of two kinds (see FINDINGS.md §25):

* **Class A** — `contents`/`offsets` populated. The loadable carries the bytes; the driver
  allocates and copies them at load time. Weights normally live here.
* **Class B** — `contents: []`, plus a `bind_id` and a `tensor_desc_id`. The loadable carries
  only a size; the caller supplies a `DevBuf` per submit and the driver resolves
  `address_list -> mem_id -> bind_id -> DevBuf`. Activations live here.

Making a weight Class B means one loadable can serve many weight sets, selected by
`DevBuf.offset` into one resident blob. That is the MoE expert-selection primitive: 27
loadables instead of 1716. Verified byte-identical to the baked-in blob on `roma`
(FINDINGS.md §26, §27).

## Why this is one region, not three

Inspecting the vendor SwiGLU block (`test_swiglu_mlp_npu_b1.model`) shows a single Class A
weight region holding **all nine** projection slices:

    mem_id=1 flags=3 size=5,308,416 contents=[
        w_gatesplitM_0_0, w_gatesplitM_1_1, w_gatesplitM_2_2,     # 3 x 589,824
        w_upsplitM_0_3,   w_upsplitM_1_4,   w_upsplitM_2_5,       # 3 x 589,824
        w_downsplitM_0_6, w_downsplitM_1_7, w_downsplitM_2_8 ]    # 786,432 x2 + 196,608

5,308,416 = 3 x 1,769,472, i.e. gate + up + down each contribute one third. EsAAC packs a
whole FFN's weights into one contiguous region with per-blob offsets, so an expert FFN needs
**one** runtime-bound tensor and **one** `bind_id`, not three. For MoE that is convenient: a
single per-die blob can hold every expert's gate/up/down, and selecting an expert is one
offset. The allocator below still handles the multi-region case, because nothing guarantees
future templates pack the same way.

## Relocs

No relocs are added for a Class B weight. That was measured, not assumed: a variant carrying
six extra relocs at weight-cube +22/+26 produced byte-identical output to one without them.
The activation I/O relocs are left untouched.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

try:
    from .flatbuffer_emitter import LoadableModel, MemoryEntry, TensorDescEntry
except ImportError:  # pragma: no cover - script/flat-layout fallback
    from backend.flatbuffer_emitter import LoadableModel, MemoryEntry, TensorDescEntry

# Class B input/output conventions observed in both vendor and gemm_builder loadables.
CLASS_B_INPUT_FLAGS = 5
CLASS_B_OUTPUT_FLAGS = 9
DATA_CATEGORY_WEIGHT = 1
DATA_CATEGORY_FEATURE = 2
DATA_TYPE_HALF = 2
DATA_FORMAT_NCHW = 1
PIXEL_FORMAT_FEATURE = 36

# Blob names that are command streams or intermediates, never weights. Task descriptors are
# the ones that must never be converted: unbinding them would leave the NPU with no program.
_TASK_MARKERS = ("-task-",)
_INTERMEDIATE_MARKERS = (
    "cvtPrecision", ".sram.edma", "tiling_", "Concat", "mul_", "biasadd", "_BIAS",
)


@dataclass
class BoundWeight:
    """One weight region converted to Class B."""
    mem_id: int
    bind_id: int
    tensor_desc_id: int
    size: int
    dropped_blobs: List[str]


def is_weight_blob(name: str) -> bool:
    """True when a blob name looks like a weight rather than a task stream or intermediate."""
    if any(mk in name for mk in _TASK_MARKERS):
        return False
    if any(mk in name for mk in _INTERMEDIATE_MARKERS):
        return False
    # Observed weight spellings: WsplitM_*, w_gatesplitM_*, w_upsplitM_*, w_downsplitM_*,
    # *_transpose_weight*, *_weightsplitM_*.
    low = name.lower()
    return "weight" in low or "splitm" in low


def find_weight_regions(model: LoadableModel,
                        mem_ids: Optional[Sequence[int]] = None) -> List[MemoryEntry]:
    """Class A, DRAM-resident memory entries that hold weight blobs.

    Pass `mem_ids` to select explicitly; the auto-detection is a name heuristic and should
    be verified against the emitted model rather than trusted blindly.
    """
    out = []
    for e in model.memory_list:
        if mem_ids is not None and e.id not in mem_ids:
            continue
        if not e.contents:
            continue                      # already Class B
        if e.domain != 0:
            continue                      # SPRAM staging, not a weight source
        if mem_ids is None and not all(is_weight_blob(c) for c in e.contents):
            continue
        out.append(e)
    return out


def next_bind_id(model: LoadableModel) -> int:
    """First free input bind slot.

    `bind_id` indexes the task's `inputFd[]` array, and inputs and outputs are numbered
    independently, so only existing Class B *input* entries (flags 5) consume slots.
    """
    used = {e.bind_id for e in model.memory_list
            if not e.contents and e.flags == CLASS_B_INPUT_FLAGS}
    candidate = 0
    while candidate in used:
        candidate += 1
    return candidate


def make_weights_runtime_bound(
    model: LoadableModel,
    mem_ids: Optional[Sequence[int]] = None,
    name_prefix: str = "weight",
) -> List[BoundWeight]:
    """Converts weight regions in @p model from Class A to Class B, in place.

    For each region: drops its blobs, clears `contents`/`offsets`, sets the Class B input
    flags, assigns the next free input `bind_id`, and appends a `data_category=WEIGHT`
    tensor descriptor sized to the whole region. Returns one `BoundWeight` per region so the
    caller can build the submit-time `DevBuf` list.

    The caller must then supply each region per submit as
    `inputFd[bound.bind_id] = {mem_fd, offset, bound.size}`.
    """
    regions = find_weight_regions(model, mem_ids)
    if not regions:
        return []

    next_desc_id = max((t.id for t in model.tensor_descs), default=-1) + 1
    blob_by_name: Dict[str, int] = {b.name: i for i, b in enumerate(model.blobs)}

    bound: List[BoundWeight] = []
    drop_indices: set = set()

    for region in regions:
        bind_id = next_bind_id(model)
        desc_id = next_desc_id
        next_desc_id += 1

        # Remember the blob indices before clearing contents.
        dropped = list(region.contents)
        for name in dropped:
            if name in blob_by_name:
                drop_indices.add(blob_by_name[name])

        region.contents = []
        region.offsets = []
        region.flags = CLASS_B_INPUT_FLAGS
        region.bind_id = bind_id
        region.tensor_desc_id = desc_id

        # Mark the slot taken so the next region gets a different one.
        model.tensor_descs.append(TensorDescEntry(
            name=f"{name_prefix}{len(bound)}" if len(bound) else name_prefix,
            id=desc_id,
            mem_id=region.id,
            size=region.size,
            offset=0,
            data_format=DATA_FORMAT_NCHW,
            data_type=DATA_TYPE_HALF,
            data_category=DATA_CATEGORY_WEIGHT,
            pixel_format=PIXEL_FORMAT_FEATURE,
            pixel_mapping=0,
            n=1, c=1, h=1, w=1,
            stride_0=2, stride_1=2, stride_2=2,
        ))

        bound.append(BoundWeight(mem_id=region.id, bind_id=bind_id,
                                 tensor_desc_id=desc_id, size=region.size,
                                 dropped_blobs=dropped))

    if drop_indices:
        model.blobs = [b for i, b in enumerate(model.blobs) if i not in drop_indices]

    return bound


def extract_weight_blob(model: LoadableModel, region: MemoryEntry) -> bytes:
    """Concatenates a Class A region's blobs in declared offset order.

    Used to build the submit-time buffer that must reproduce the baked-in weights exactly —
    the reference data for the bit-match test.
    """
    buf = bytearray(region.size)
    by_name = {b.name: b for b in model.blobs}
    for off, name in sorted(zip(region.offsets, region.contents)):
        blob = by_name.get(name)
        if blob is None:
            raise KeyError(f"blob '{name}' referenced by mem_id={region.id} is missing")
        data = bytes(blob.data or b"")
        buf[off:off + len(data)] = data
    return bytes(buf)
