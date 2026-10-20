"""
Unit tests for Differential Comparator (Milestone 6).
Validates Cosine Similarity, SNR, NMSE, percentiles, and error histograms.
"""

import pytest
import numpy as np

from eswin_simulator.evaluation import TensorComparator, ComparisonReport


class TestTensorComparator:
    def test_bit_exact_comparison(self):
        a = np.random.randn(10, 20).astype(np.float32)
        rep = TensorComparator.compare(a, a, tolerance=1e-5)

        assert rep.passed
        assert rep.cosine_similarity == 1.0
        assert rep.max_abs_diff == 0.0
        assert rep.snr_db == float("inf")
        assert rep.nmse == 0.0
        assert rep.percentiles["p100"] == 0.0
        assert rep.histogram["error < 1e-5"] == 200

    def test_near_exact_comparison(self):
        np.random.seed(42)
        ref = np.random.randn(100, 50).astype(np.float32)
        # Small perturbation < 1e-4
        sim = ref + np.random.uniform(-5e-5, 5e-5, size=ref.shape).astype(np.float32)

        rep = TensorComparator.compare(sim, ref, tolerance=1e-4, cos_threshold=0.9999)
        assert rep.passed
        assert rep.cosine_similarity > 0.99999
        assert rep.max_abs_diff < 1e-4
        assert rep.snr_db > 60.0

    def test_tolerance_failure(self):
        ref = np.ones((10, 10), dtype=np.float32)
        sim = np.full((10, 10), 2.0, dtype=np.float32)

        rep = TensorComparator.compare(sim, ref, tolerance=0.1)
        assert not rep.passed
        assert rep.max_abs_diff == 1.0

    def test_report_summary_and_json(self):
        a = np.ones((5, 5), dtype=np.float32)
        b = np.ones((5, 5), dtype=np.float32)
        rep = TensorComparator.compare(a, b)

        summary_text = rep.summary()
        assert "[PASSED]" in summary_text
        assert "Cosine Similarity" in summary_text

        json_text = rep.to_json()
        assert '"passed": true' in json_text
        assert '"cosine_similarity": 1.0' in json_text
