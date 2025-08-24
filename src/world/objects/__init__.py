"""World objects subpackage.

This module re-exports commonly used in-world object classes so callers can
import them from a single location::

    from world.objects import WallTile, Road

The actual implementations remain in the sibling modules (``wall_tile.py``
and ``road.py``) to minimize changes to the rest of the codebase.
"""

from .wall_tile import WallTile
from .road import Road

__all__ = ["WallTile", "Road"]
