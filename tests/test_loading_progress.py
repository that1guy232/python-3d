from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world import world_builder


class LoadingProgressTests(unittest.TestCase):
    def test_world_object_step_count_is_derived_from_step_specs(self) -> None:
        scene = SimpleNamespace()
        args = (scene, 200, 25.0, 12.5, 25, 0, 1000, 2000, 750)

        specs = world_builder.create_world_object_step_specs(*args)
        count = world_builder.create_world_object_step_count(*args)

        self.assertEqual(count, len(specs))
        self.assertEqual(
            [spec.label for spec in specs],
            [
                "Creating buildings",
                "Generating ground mesh",
                "Building structures",
                "Building showcase polygons",
                "Creating roads",
                "Spawning trees",
                "Spawning goblins",
                "Spawning grass",
                "Spawning rocks",
                "Building fences",
                "Adding ground details",
            ],
        )


if __name__ == "__main__":
    unittest.main()
