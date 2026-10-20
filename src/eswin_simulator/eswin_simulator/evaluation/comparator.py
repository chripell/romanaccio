"""
Differential Tensor Comparator and Numerical Metrics Engine for ESWIN Simulator.
Computes Cosine Similarity, Signal-to-Noise Ratio (SNR), NMSE, Absolute/Relative
error metrics, percentiles, and error histograms.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


@dataclass
class ComparisonReport:
    """Structured report detailing numerical verification between simulated and reference tensors."""

    simulated_shape: Tuple[int, ...]
    reference_shape: Tuple[int, ...]
    dtype: str
    total_elements: int
    cosine_similarity: float
    snr_db: float
    nmse: float
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    mean_rel_diff: float
    percentiles: Dict[str, float] = field(default_factory=dict)
    histogram: Dict[str, int] = field(default_factory=dict)
    passed: bool = True
    tolerance: float = 1e-4
    cos_threshold: float = 0.999

    def summary(self) -> str:
        status_str = "[PASSED]" if self.passed else "[FAILED]"
        snr_str = f"{self.snr_db:.2f} dB" if not np.isinf(self.snr_db) else "+inf dB (bit-exact)"

        lines = [
            "=" * 70,
            f"  Differential Tensor Evaluation Report  {status_str:>29}",
            "=" * 70,
            f"  • Tensor Shape      : {self.simulated_shape} ({self.total_elements:,} elements, dtype={self.dtype})",
            f"  • Cosine Similarity : {self.cosine_similarity:.8f} (threshold: >={self.cos_threshold})",
            f"  • Signal-to-Noise   : {snr_str}",
            f"  • Normalized MSE    : {self.nmse:.8e}",
            "-" * 70,
            "  [Absolute & Relative Differences]",
            f"  • Max Absolute Diff : {self.max_abs_diff:.8e} (tolerance: <= {self.tolerance})",
            f"  • Mean Absolute Diff: {self.mean_abs_diff:.8e}",
            f"  • Max Relative Diff : {self.max_rel_diff:.8e}",
            f"  • Mean Relative Diff: {self.mean_rel_diff:.8e}",
            "-" * 70,
            "  [Error Percentile Distribution]",
            f"  • p50 (median)      : {self.percentiles.get('p50', 0.0):.8e}",
            f"  • p90               : {self.percentiles.get('p90', 0.0):.8e}",
            f"  • p99               : {self.percentiles.get('p99', 0.0):.8e}",
            f"  • p100 (worst-case) : {self.percentiles.get('p100', 0.0):.8e}",
        ]

        if self.histogram:
            lines.extend(
                [
                    "-" * 70,
                    "  [Error Histogram]",
                ]
            )
            for bucket, count in self.histogram.items():
                pct = (count / max(self.total_elements, 1)) * 100.0
                bar = "#" * int(pct / 2.5)
                lines.append(f"  • {bucket:<18} : {count:>8} ({pct:5.2f}%)  {bar}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulated_shape": list(self.simulated_shape),
            "reference_shape": list(self.reference_shape),
            "dtype": self.dtype,
            "total_elements": self.total_elements,
            "cosine_similarity": float(self.cosine_similarity),
            "snr_db": float(self.snr_db) if not np.isinf(self.snr_db) else "inf",
            "nmse": float(self.nmse),
            "max_abs_diff": float(self.max_abs_diff),
            "mean_abs_diff": float(self.mean_abs_diff),
            "max_rel_diff": float(self.max_rel_diff),
            "mean_rel_diff": float(self.mean_rel_diff),
            "percentiles": {k: float(v) for k, v in self.percentiles.items()},
            "histogram": self.histogram,
            "passed": self.passed,
            "tolerance": float(self.tolerance),
            "cos_threshold": float(self.cos_threshold),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class TensorComparator:
    """Computes rigorous differential parity metrics between tensor arrays."""

    @classmethod
    def compare(
        cls,
        simulated: np.ndarray,
        reference: np.ndarray,
        tolerance: float = 1e-4,
        cos_threshold: float = 0.999,
        eps: float = 1e-12,
    ) -> ComparisonReport:
        """Compares simulated tensor against reference tensor."""
        if simulated.shape != reference.shape:
            # Flatten if size matches
            if simulated.size == reference.size:
                simulated = simulated.reshape(reference.shape)
            else:
                raise ValueError(
                    f"Tensor shapes do not match: simulated {simulated.shape} vs reference {reference.shape}"
                )

        u = simulated.astype(np.float64, copy=False).flatten()
        v = reference.astype(np.float64, copy=False).flatten()
        total_elements = u.size

        # Cosine similarity
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u < eps and norm_v < eps:
            cos_sim = 1.0
        elif norm_u < eps or norm_v < eps:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(u, v) / (norm_u * norm_v))

        # Absolute and relative differences
        diff = np.abs(u - v)
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))

        denom = np.abs(v) + eps
        rel_diff = diff / denom
        max_rel = float(np.max(rel_diff))
        mean_rel = float(np.mean(rel_diff))

        # SNR (Signal to Noise Ratio in dB)
        noise_energy = np.sum(diff ** 2)
        signal_energy = np.sum(v ** 2)
        if noise_energy < eps:
            snr_db = float("inf")
            nmse = 0.0
        else:
            snr_db = float(10.0 * np.log10((signal_energy + eps) / (noise_energy + eps)))
            nmse = float(noise_energy / (signal_energy + eps))

        # Percentiles
        p50 = float(np.percentile(diff, 50))
        p90 = float(np.percentile(diff, 90))
        p95 = float(np.percentile(diff, 95))
        p99 = float(np.percentile(diff, 99))
        p100 = max_abs

        percentiles = {
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "p100": p100,
        }

        # Error histogram
        histogram = {
            "error < 1e-5": int(np.sum(diff < 1e-5)),
            "1e-5 <= error < 1e-4": int(np.sum((diff >= 1e-5) & (diff < 1e-4))),
            "1e-4 <= error < 1e-3": int(np.sum((diff >= 1e-4) & (diff < 1e-3))),
            "1e-3 <= error < 1e-2": int(np.sum((diff >= 1e-3) & (diff < 1e-2))),
            "1e-2 <= error < 1e-1": int(np.sum((diff >= 1e-2) & (diff < 1e-1))),
            "error >= 1e-1": int(np.sum(diff >= 1e-1)),
        }

        passed = bool((cos_sim >= cos_threshold) and (max_abs <= tolerance or mean_rel <= tolerance))

        return ComparisonReport(
            simulated_shape=tuple(simulated.shape),
            reference_shape=tuple(reference.shape),
            dtype=str(simulated.dtype),
            total_elements=total_elements,
            cosine_similarity=cos_sim,
            snr_db=snr_db,
            nmse=nmse,
            max_abs_diff=max_abs,
            mean_abs_diff=mean_abs,
            max_rel_diff=max_rel,
            mean_rel_diff=mean_rel,
            percentiles=percentiles,
            histogram=histogram,
            passed=passed,
            tolerance=tolerance,
            cos_threshold=cos_threshold,
        )

    @classmethod
    def compare_files(
        cls,
        sim_path: Union[str, Path],
        ref_path: Union[str, Path],
        shape: Tuple[int, ...],
        dtype: Union[str, np.dtype],
        tolerance: float = 1e-4,
        cos_threshold: float = 0.999,
    ) -> ComparisonReport:
        """Reads raw binary files and computes differential comparison."""
        dt = np.dtype(dtype)
        sim_data = np.fromfile(str(sim_path), dtype=dt).reshape(shape)
        ref_data = np.fromfile(str(ref_path), dtype=dt).reshape(shape)
        return cls.compare(sim_data, ref_data, tolerance=tolerance, cos_threshold=cos_threshold)
