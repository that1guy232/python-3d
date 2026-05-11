"""Entry point kept minimal by delegating to Engine.

This refactor introduces a lightweight engine & scene abstraction so that
additional world assets (static or dynamic) can be registered cleanly without
inflating this file again. The previous monolithic loop now lives in
`engine.py`.
"""

from engine.core.engine import Engine  # noqa: E402 (local import order)
from game.main_menu import MainMenuScene  # noqa: E402 (local import order)


def make_initial_scene() -> MainMenuScene:
    """Build the first scene shown at launch."""
    return MainMenuScene()


def main():  # small wrapper for clarity / debuggers
    """Start the engine with the main menu as the first active scene."""
    Engine(initial_scene_factory=make_initial_scene).run()


if __name__ == "__main__":
    main()
