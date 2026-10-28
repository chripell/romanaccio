"""Weight-region detection: the classifier must handle INT8 loadables.

An INT8 EsAAC loadable does not keep weights alone in their region. Measured on a
DeepSeek-V2-Lite expert FFN compiled with --quant-stats (one 8,908,800-byte region, 32
blobs): the `W_*splitM_*` weights are interleaved with `CvtPrecision_QuantM*` blobs that
carry that expert's quantization scales. Requiring every blob in a region to look like a
weight -- the original rule -- rejected the region outright, so no INT8 loadable could be
converted at all.

The rule is now "at least one weight-like blob and no command stream". These tests pin that,
using a synthetic model shaped like the measured INT8 one, so they do not depend on a 9 MB
fixture that is not in the repository.
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.dynamic_weight_binding import (  # noqa: E402
    find_weight_regions,
    is_task_blob,
    is_weight_blob,
)
from backend.flatbuffer_emitter import MemoryEntry  # noqa: E402


class _FakeModel:
    """find_weight_regions only reads .memory_list, so a stub keeps the test honest."""

    def __init__(self, memory_list):
        self.memory_list = memory_list


def int8_like_model():
    """Mirrors the measured INT8 expert-FFN memory layout."""
    return _FakeModel([
        # Class B scratch / activation entries: no contents.
        MemoryEntry(id=0, domain=0, flags=1, size=4096),
        MemoryEntry(id=4, domain=0, flags=5, size=2048, tensor_desc_id=0),
        MemoryEntry(id=5, domain=0, flags=9, size=2048, tensor_desc_id=1),
        # The weight region: weights INTERLEAVED with per-slice quant blobs.
        MemoryEntry(id=1, domain=0, flags=3, size=8908800, contents=[
            "W_gatesplitM_0_0", "CvtPrecision_QuantM1",
            "W_gatesplitM_1_2", "CvtPrecision_QuantM3",
            "W_upsplitM_0_4", "CvtPrecision_QuantM5",
            "W_downsplitM_0_6", "CvtPrecision_QuantM7",
        ], offsets=list(range(0, 8 * 1000, 1000))),
        # Activations/intermediates in DRAM: no weight-like names.
        MemoryEntry(id=2, domain=0, flags=1, size=4096,
                    contents=["silu", "gate_projcvtPrecisionsplitM_0"], offsets=[0, 2048]),
        # SPRAM staging: domain 1, must never be treated as a weight source.
        MemoryEntry(id=3, domain=1, flags=1, size=4194304,
                    contents=["gate_projsplitM_0.sram.edma"], offsets=[0]),
        # Command streams: dropping these would brick the loadable.
        MemoryEntry(id=13, domain=0, flags=34, size=110976,
                    contents=["kmd-task-dep_graph"], offsets=[0]),
        MemoryEntry(id=17, domain=0, flags=3, size=1024,
                    contents=["kmd-task-config_list"], offsets=[0]),
    ])


class TestTaskBlobClassification(unittest.TestCase):

    def test_command_streams_are_task_blobs(self):
        for n in ("kmd-task-dep_graph", "umd-task-surf_list", "kmd-task-config_list"):
            self.assertTrue(is_task_blob(n), n)
            self.assertFalse(is_weight_blob(n), n)

    def test_weights_and_quant_blobs_are_not_task_blobs(self):
        for n in ("W_gatesplitM_0_0", "CvtPrecision_QuantM1"):
            self.assertFalse(is_task_blob(n), n)


class TestInt8RegionDetection(unittest.TestCase):

    def test_mixed_weight_and_quant_region_is_detected(self):
        """The regression: an INT8 region is half weights, half quant blobs."""
        regions = find_weight_regions(int8_like_model())
        self.assertEqual([r.id for r in regions], [1],
                         "exactly the interleaved weight region must be selected")
        self.assertEqual(regions[0].size, 8908800)

    def test_non_weight_regions_are_never_selected(self):
        regions = find_weight_regions(int8_like_model())
        ids = {r.id for r in regions}
        for bad in (0, 2, 3, 4, 5, 13, 17):
            self.assertNotIn(bad, ids, f"mem_id={bad} must not be treated as weights")

    def test_region_containing_a_command_stream_is_rejected(self):
        """Even if a weight name sneaks in, a task stream disqualifies the region."""
        m = _FakeModel([MemoryEntry(id=1, domain=0, flags=3, size=2048,
                                    contents=["W_splitM_0", "kmd-task-op_list"],
                                    offsets=[0, 1024])])
        self.assertEqual(find_weight_regions(m), [])

    def test_explicit_mem_ids_bypass_the_heuristic(self):
        m = int8_like_model()
        self.assertEqual([r.id for r in find_weight_regions(m, mem_ids=[13])], [13],
                         "explicit selection must override the name heuristic")
        self.assertEqual(find_weight_regions(m, mem_ids=[99]), [])

    def test_all_weight_region_still_detected(self):
        """The fp16 case (every blob a weight) must not regress."""
        m = _FakeModel([MemoryEntry(id=1, domain=0, flags=3, size=1769472,
                                    contents=["WsplitM_0_0", "WsplitM_1_1", "WsplitM_2_2"],
                                    offsets=[0, 589824, 1179648])])
        self.assertEqual([r.id for r in find_weight_regions(m)], [1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
