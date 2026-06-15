from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world.ui.interactions import WorldUIInteractions


class FakeBattleOverlay:
    def __init__(self) -> None:
        self.calls = []

    def handle_mouse_down(self, pos) -> bool:
        self.calls.append(("down", pos))
        return True

    def handle_mouse_motion(self, pos) -> bool:
        self.calls.append(("motion", pos))
        return False

    def handle_mouse_up(self, pos) -> bool:
        self.calls.append(("up", pos))
        return True


class FakeMenu:
    def __init__(self) -> None:
        self.calls = []

    def compute_buttons(self, *, width, height):
        self.calls.append(("compute", width, height))
        return [{"rect": (0, 0, width, height)}]

    def handle_click(self, pos) -> None:
        self.calls.append(("click", pos))

    def handle_motion(self, pos) -> None:
        self.calls.append(("motion", pos))

    def handle_release(self, pos) -> None:
        self.calls.append(("release", pos))


class WorldUIInteractionsTests(unittest.TestCase):
    def test_battle_input_forwards_to_overlay(self) -> None:
        overlay = FakeBattleOverlay()
        interactions = WorldUIInteractions(SimpleNamespace(battle_overlay=overlay))

        self.assertTrue(interactions.handle_battle_click((1, 2)))
        self.assertFalse(interactions.handle_battle_motion((3, 4)))
        self.assertTrue(interactions.handle_battle_release((5, 6)))

        self.assertEqual(
            overlay.calls,
            [("down", (1, 2)), ("motion", (3, 4)), ("up", (5, 6))],
        )

    def test_pause_input_uses_active_menu(self) -> None:
        pause_menu = FakeMenu()
        setting_menu = FakeMenu()
        scene = SimpleNamespace(
            pause_menu=pause_menu,
            setting_menu=setting_menu,
            showing_settings_menu=False,
        )
        interactions = WorldUIInteractions(scene)

        self.assertEqual(
            interactions.compute_pause_buttons(width=10, height=20),
            [{"rect": (0, 0, 10, 20)}],
        )
        interactions.handle_pause_click((1, 1))
        scene.showing_settings_menu = True
        interactions.handle_pause_motion((2, 2))
        interactions.handle_pause_release((3, 3))

        self.assertEqual(pause_menu.calls, [("compute", 10, 20), ("click", (1, 1))])
        self.assertEqual(setting_menu.calls, [("motion", (2, 2)), ("release", (3, 3))])

    def test_inventory_click_closes_only_on_close_rect(self) -> None:
        scene = SimpleNamespace(
            inventory_open=True,
            paused=True,
            showing_settings_menu=True,
        )
        interactions = WorldUIInteractions(scene)

        self.assertFalse(interactions.handle_inventory_click((0, 0)))
        self.assertTrue(scene.inventory_open)

        x, y, w, h = interactions.inventory_close_rect()
        self.assertTrue(interactions.handle_inventory_click((x + w * 0.5, y + h * 0.5)))
        self.assertFalse(scene.inventory_open)
        self.assertFalse(scene.paused)
        self.assertFalse(scene.showing_settings_menu)


if __name__ == "__main__":
    unittest.main()
