from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world.ui.pause_panel import PauseMenuPanel


class PauseMenuPanelTests(unittest.TestCase):
    def test_slider_track_clamps_ratio_and_computes_geometry(self) -> None:
        rect = (10.0, 20.0, 100.0, 40.0)

        self.assertEqual(
            PauseMenuPanel.slider_track(rect, padding=10.0, ratio=0.25),
            (20.0, 48.0, 80.0, 5, 40.0),
        )
        self.assertEqual(
            PauseMenuPanel.slider_track(rect, padding=10.0, ratio=2.0),
            (20.0, 48.0, 80.0, 5, 100.0),
        )
        self.assertEqual(
            PauseMenuPanel.slider_track(rect, padding=10.0, ratio=-1.0),
            (20.0, 48.0, 80.0, 5, 20.0),
        )


if __name__ == "__main__":
    unittest.main()
