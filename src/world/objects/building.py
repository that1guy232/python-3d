from __future__ import annotations

from typing import Callable, List, Optional
import math
import random

from pygame.math import Vector3

from .wall_tile import WallTile


class Building:

    def __init__(
        self,
        position: Vector3 | tuple[float, float] | None = None,
        *,
        target_height: float | None = None,
    ) -> None:
        self.position = position or Vector3(0.0, 0.0, 0.0)
        self.center = Vector3(self.position.x, self.position.y, self.position.z)

        self.shapes: list = []
        self._bbox: Optional[tuple[float, float, float, float]] = None
        self.target_height = target_height
            
    def create_perimeter_walls(
        self,
        *,
        wall_height: float = 3.0,
        wall_thickness: float = 0.5,
        texture: Optional[int] = None,
        uv_repeat: tuple[float, float] = (1.0, 1.0),
        base_y: Optional[float] = None,
        width: Optional[float] = None,
        depth: Optional[float] = None,
        max_tile_width: float = 100.0,  # controls subdivision; default == no subdivision
    ) -> List[WallTile]:
        """Create a 4-wall rectangular (box) perimeter centered on the building.

        Same semantics as your original function, but each side is subdivided
        into tiles whose width does not exceed max_tile_width.
        """
        cx, cz = float(self.position.x), float(self.position.z)

        if base_y is None:
            if self.target_height is not None:
                base_y = float(self.target_height)
            else:
                base_y = float(self.position.y)
        else:
            base_y = float(base_y)

        DEFAULT_SIDE = 100.0
        outer_x = float(width) if width is not None else DEFAULT_SIDE
        outer_z = float(depth) if depth is not None else DEFAULT_SIDE

        half_x = outer_x / 2.0
        half_z = outer_z / 2.0

        walls: List[WallTile] = []
        # We'll keep per-side tile lists so we can remove the shortest side later.
        sides_tiles: list[list[WallTile]] = []
        side_spans: list[float] = []

        normals = [
            (0.0, 1.0),   # North (positive Z)
            (1.0, 0.0),   # East (positive X)
            (0.0, -1.0),  # South (negative Z)
            (-1.0, 0.0),  # West (negative X)
        ]

        for nx, nz in normals:
            # Default centers
            center_x = cx
            center_z = cz

            # thickness half (perpendicular half-extent)
            half_width = wall_thickness / 2.0

            if abs(nz) > 0.0:
                # North/South: wall plane perpendicular to Z, span along X
                center_z = cz + nz * (half_z - half_width)
                span_half = half_x
                span_axis = "x"
            else:
                # East/West: wall plane perpendicular to X, span along Z
                center_x = cx + nx * (half_x - half_width)
                span_half = half_z
                span_axis = "z"

            full_span = span_half * 2.0
            # compute subdivision count
            num_tiles = max(1, int(math.ceil(full_span / float(max_tile_width))))
            tile_width = full_span / num_tiles
            tile_half = tile_width / 2.0

            theta = math.atan2(nz, nx)
            eps = max(1e-5, 0.01 * min(1.0, wall_thickness))

            # nudge the whole wall slightly inward to avoid z-fighting (same as original)
            center_x -= nx * eps
            center_z -= nz * eps

            # span start (outer edge along span axis)
            if span_axis == "x":
                span_start = center_x - span_half
            else:
                span_start = center_z - span_half

            this_side_tiles: list[WallTile] = []
            for i in range(num_tiles):
                center_along = span_start + (i + 0.5) * tile_width
                if span_axis == "x":
                    tx = center_along
                    tz = center_z
                    w = half_width          # half-thickness (X half is tile width? rotation handles it)
                    d = tile_half           # half-span along X mapped to depth param
                else:
                    tx = center_x
                    tz = center_along
                    w = half_width
                    d = tile_half

                tile = WallTile(
                    position=Vector3(tx, base_y + wall_height * 0.5, tz),
                    width=w,
                    height=wall_height * 0.5,
                    depth=d,
                    texture=texture,
                    uv_repeat=uv_repeat,
                    thickness=wall_thickness,
                )
                tile.rotation = Vector3(0.0, theta, 0.0)

                this_side_tiles.append(tile)
                walls.append(tile)
                self.attach_shapes([tile])

            sides_tiles.append(this_side_tiles)
            side_spans.append(full_span)

        # find shortest side (same behavior as your original code which removed the single shortest wall)
        shortest_id = None
        shortest_span = float("inf")
        for idx, span in enumerate(side_spans):
            if span < shortest_span:
                shortest_span = span
                shortest_id = idx

        # remove tiles that belong to the shortest side (if any)
        if shortest_id is not None:
            for t in sides_tiles[shortest_id]:
                # remove from walls list if present
                try:
                    walls.remove(t)
                except ValueError:
                    pass
            # Note: attached shapes remain attached (same as original behavior).
            # If you want to also detach them from `self.shapes` you'd need to do so here.

        return walls
    def attach_shapes(self, shapes: list) -> None:
        """Attach shape objects (Object3D-derived) to this building and
        update the cached bounding-box. Shapes should have a working
        `get_world_vertices()` method.
        """
        if not shapes:
            return
        # extend list and recompute bbox
        self.shapes.extend(shapes)
        self._update_bounding_box()

    def _update_bounding_box(self) -> None:
        """Compute an axis-aligned bounding box (min_x, max_x, min_z, max_z)
        from the world-space vertices of attached shapes. If no shapes are
        attached the bbox remains None (caller should fall back to radius).
        """
        if not self.shapes:
            self._bbox = None
            return

        min_x = float("inf")
        max_x = float("-inf")
        min_z = float("inf")
        max_z = float("-inf")

        for s in self.shapes:
            verts = []
            try:
                verts = s.get_world_vertices() or []
            except Exception:
                # ignore shapes that don't implement get_world_vertices
                continue
            for v in verts:
                if v.x < min_x:
                    min_x = v.x
                if v.x > max_x:
                    max_x = v.x
                if v.z < min_z:
                    min_z = v.z
                if v.z > max_z:
                    max_z = v.z

        if min_x == float("inf"):
            # no valid vertices found
            self._bbox = None
            return

        self._bbox = (min_x, max_x, min_z, max_z)

    def get_bounding_box(self) -> Optional[tuple[float, float, float, float]]:
        """Get the cached axis-aligned bounding box (min_x, max_x, min_z, max_z)
        of attached shapes, or None if no shapes are attached.
        """
        return self._bbox

    def contains_point(self, x: float, z: float, margin: float = 0.0) -> bool:
        """Return True if the X/Z point lies within the building footprint.

        This is used by the world spawner's `avoid_areas` checks. The method
        prefers the computed axis-aligned bounding box of attached shapes. If
        no bbox is available it falls back to a default square footprint used
        by `create_perimeter_walls`.
        """
        # Use bounding box if available
        if self._bbox is not None:
            min_x, max_x, min_z, max_z = self._bbox
            return (min_x - margin) <= x <= (max_x + margin) and (
                min_z - margin
            ) <= z <= (max_z + margin)

        print("Using default footprint")
        # Fallback to default footprint (same default as create_perimeter_walls)
        DEFAULT_SIDE = 100.0
        half_x = DEFAULT_SIDE / 2.0
        half_z = DEFAULT_SIDE / 2.0
        if abs(x - self.position.x) <= (half_x + margin) and abs(
            z - self.position.z
        ) <= (half_z + margin):
            return True

        return False

    def get_floor_level(self) -> float:
        """Return the world-space floor/base Y for this building.

        Priority:
        - If `target_height` is set, return that.
        - Else, if attached shapes provide vertices, return the minimum
          Y value found among their world vertices.
        - Fallback to 0.0 if nothing else is available.
        """
        if self.target_height is not None:
            return float(self.target_height)

        min_y = float("inf")
        for s in self.shapes:
            try:
                verts = s.get_world_vertices() or []
            except Exception:
                continue
            for v in verts:
                if v.y < min_y:
                    min_y = v.y

        if min_y == float("inf"):
            return 0.0
        return float(min_y)

    @property
    def height(self) -> float:
        """Return the building height in world units.

        Priority:
        - If an explicit override was set via the `height` setter, use that.
        - Else if attached shapes provide vertices, return (max_y - min_y).
        - Fallback to a sensible default (10.0).
        """
        # honor explicit override if present
        if getattr(self, "_height_override", None) is not None:
            return float(self._height_override)

        min_y = float("inf")
        max_y = float("-inf")
        for s in self.shapes:
            try:
                verts = s.get_world_vertices() or []
            except Exception:
                continue
            for v in verts:
                if v.y < min_y:
                    min_y = v.y
                if v.y > max_y:
                    max_y = v.y

        if min_y == float("inf") or max_y == float("-inf") or max_y <= min_y:
            # no usable geometry; return a practical default
            return 10.0

        return float(max_y - min_y)

    @height.setter
    def height(self, value: float) -> None:
        """Allow callers to explicitly set/override the building height."""
        self._height_override = float(value)

    def get_corners(self) -> tuple[list[Vector3], float]:
        """Return the four outer-corner world positions and the floor Y.

        Returns a tuple (corners, floor_y) where `corners` is a list of four
        `pygame.math.Vector3` points in the order:
          (min_x, min_z), (max_x, min_z), (max_x, max_z), (min_x, max_z)
        with Y set to the computed floor level. If no shapes/bbox are present
        this will fall back to the same default side length used by
        `create_perimeter_walls` (100.0 units).
        """
        # Determine bounding rectangle in X/Z
        if self._bbox is not None:
            min_x, max_x, min_z, max_z = self._bbox
        else:
            DEFAULT_SIDE = 100.0
            half_x = DEFAULT_SIDE / 2.0
            half_z = DEFAULT_SIDE / 2.0
            min_x = self.position.x - half_x
            max_x = self.position.x + half_x
            min_z = self.position.z - half_z
            max_z = self.position.z + half_z

        floor_y = self.get_floor_level()

        corners = [
            Vector3(min_x, floor_y, min_z),
            Vector3(max_x, floor_y, min_z),
            Vector3(max_x, floor_y, max_z),
            Vector3(min_x, floor_y, max_z),
        ]

        return corners, float(floor_y)
