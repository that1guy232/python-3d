"""Entry point kept minimal by delegating to Engine.

This refactor introduces a lightweight engine & scene abstraction so that
additional world assets (static or dynamic) can be registered cleanly without
inflating this file again. The previous monolithic loop now lives in
`engine.py`.
"""

from core.engine import Engine  # noqa: E402 (local import order)


def main():  # small wrapper for clarity / debuggers
    Engine().run()


if __name__ == "__main__":
    main()
