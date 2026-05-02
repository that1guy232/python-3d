"""Entry point kept minimal by delegating to Engine.

This refactor introduces a lightweight engine & scene abstraction so that
additional world assets (static or dynamic) can be registered cleanly without
inflating this file again. The previous monolithic loop now lives in
`engine.py`.
"""

from core.engine import Engine  # noqa: E402 (local import order)
from core.loading_scene import LoadingScene  # noqa: E402 (local import order)
from world.worldscene import WorldScene  # noqa: E402 (local import order)


def make_initial_scene() -> LoadingScene:
    world = WorldScene(defer_setup=True)
    return LoadingScene(
        world,
        title="Loading World",
        initial_status="Preparing world",
    )


def main():  # small wrapper for clarity / debuggers
    Engine(initial_scene_factory=make_initial_scene).run()


if __name__ == "__main__":
    main()
