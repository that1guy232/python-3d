from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.rendering.frame_comparison import (  # noqa: E402
    FrameComparisonThresholds,
    amplified_rgb_difference,
    compare_rgb_frames,
)


class FrameComparisonTests(unittest.TestCase):
    def test_exact_frames_pass_default_cutover_policy(self) -> None:
        frame = np.full((4, 6, 3), 80, dtype=np.uint8)
        result = compare_rgb_frames(frame, frame.copy())

        self.assertTrue(result.passed)
        self.assertEqual(result.stable_pixel_ratio, 1.0)
        self.assertEqual(result.max_absolute_error, 0)

    def test_reference_drift_is_excluded_from_backend_metrics(self) -> None:
        legacy_before = np.zeros((10, 10, 3), dtype=np.uint8)
        legacy_after = legacy_before.copy()
        packet = legacy_before.copy()
        legacy_after[0, 0] = 100
        packet[0, 0] = 200

        result = compare_rgb_frames(
            legacy_before,
            packet,
            drift_reference=legacy_after,
            thresholds=FrameComparisonThresholds(min_stable_pixel_ratio=0.99),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.stable_pixel_ratio, 0.99)
        self.assertEqual(result.max_absolute_error, 0)

    def test_material_stable_difference_fails_and_is_visualized(self) -> None:
        reference = np.zeros((2, 2, 3), dtype=np.uint8)
        candidate = reference.copy()
        candidate[1, 1] = (20, 10, 5)

        result = compare_rgb_frames(reference, candidate)
        difference = amplified_rgb_difference(reference, candidate, scale=4)

        self.assertFalse(result.passed)
        self.assertEqual(result.changed_pixel_ratio, 0.25)
        self.assertEqual(tuple(difference[1, 1]), (80, 40, 20))

    def test_localized_severe_difference_fails_maximum_error_policy(self) -> None:
        reference = np.zeros((100, 100, 3), dtype=np.uint8)
        candidate = reference.copy()
        candidate[50, 50] = (17, 0, 0)

        result = compare_rgb_frames(reference, candidate)

        self.assertLess(result.changed_pixel_ratio, 0.002)
        self.assertEqual(result.max_absolute_error, 17)
        self.assertFalse(result.passed)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame shapes differ"):
            compare_rgb_frames(
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.zeros((3, 2, 3), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
