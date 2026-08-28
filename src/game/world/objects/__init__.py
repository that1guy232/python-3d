"""World objects subpackage.

This module re-exports commonly used in-world object classes so callers can
import them from a single location::

    from game.world.objects import Chest, Door, Grass, Road, Rock, Torch, Tree, WallTile, Window

The actual implementations remain in sibling modules to minimize changes to
the rest of the codebase.
"""

from .ground_tile import GroundTile
from .wall_tile import WallTile
from .road import Road
from .torch import Torch
from .door import Door
from .chest import Chest
from .window import Window
from .tree import Tree
from .grass import Grass
from .rock import Rock

__all__ = [
    "GroundTile",
    "WallTile",
    "Road",
    "Torch",
    "Door",
    "Chest",
    "Window",
    "Tree",
    "Grass",
    "Rock",
]
