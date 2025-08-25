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
    ) -> List[WallTile]:
        """Create a 4-wall rectangular (box) perimeter centered on the building.

        Walls are positioned so their outer faces form the exact rectangle defined by width/depth.
        Properly handles corner overlaps and z-fighting.

        Parameters
        - wall_height: full world-space height of each wall (meters)
        - wall_thickness: thickness of walls (meters)
        - texture / uv_repeat: forwarded to each `WallTile`
        - base_y: world-space base Y for walls
        - width, depth: optional full side lengths along X and Z (outer dimensions)

        Returns a list of 4 `WallTile` instances (N, E, S, W) forming a closed box without overlaps.
        """
        cx, cz = float(self.position.x), float(self.position.z)
        # Determine base Y for walls. Priority:
        # 1) explicit `base_y` argument
        # 2) `self.target_height` if set
        # 3) building position Y (makes placement intuitive when using
        #    world coordinates or when the scene positions the building)
        if base_y is None:
            if self.target_height is not None:
                base_y = float(self.target_height)
            else:
                base_y = float(self.position.y)
        else:
            base_y = float(base_y)

        # Determine outer footprint dimensions (use defaults if not provided)
        DEFAULT_SIDE = 100.0
        outer_x = float(width) if width is not None else DEFAULT_SIDE
        outer_z = float(depth) if depth is not None else DEFAULT_SIDE

        half_x = outer_x / 2.0
        half_z = outer_z / 2.0

        walls: List[WallTile] = []
        # Normals: (nx, nz) for outward direction
        normals = [
            (0.0, 1.0),  # North (positive Z)
            (1.0, 0.0),  # East (positive X)
            (0.0, -1.0),  # South (negative Z)
            (-1.0, 0.0),  # West (negative X)
        ]

        for nx, nz in normals:
            center_x = cx
            center_z = cz

            if abs(nz) > 0.0:  # North/South walls (span along X axis)
                # Position: move Z inward by half thickness
                center_z = cz + nz * (half_z - wall_thickness / 2.0)
                # CORRECTED:
                #   width = thickness/2 (normal direction)
                #   depth = full width/2 (span direction)
                half_width = wall_thickness / 2.0
                half_depth = half_x
            else:  # East/West walls (span along Z axis)
                # Position: move X inward by half thickness
                center_x = cx + nx * (half_x - wall_thickness / 2.0)
                # CORRECTED:
                #   width = thickness/2 (normal direction)
                #   depth = full depth/2 (span direction)
                half_width = wall_thickness / 2.0
                half_depth = half_z  # NOT reduced by thickness!

            theta = math.atan2(nz, nx)  # Corrected rotation order

            eps = max(1e-5, 0.01 * min(1.0, wall_thickness))
            center_x -= nx * eps
            center_z -= nz * eps

            tile = WallTile(
                position=Vector3(center_x, base_y + wall_height * 0.5, center_z),
                width=half_width,  # Now correctly set
                height=wall_height * 0.5,
                depth=half_depth,  # Now correctly set
                texture=texture,
                uv_repeat=uv_repeat,
                thickness=wall_thickness,
            )
            tile.rotation = Vector3(0.0, theta, 0.0)

            walls.append(tile)
            self.attach_shapes(
                [tile]
            )  # we still attach all shapes for the bounding box

        shortest_id = None
        shortest_span = float("inf")
        for idx, wall in enumerate(walls):
            span = max(float(wall.width), float(wall.depth))
            if span < shortest_span:
                shortest_span = span
                shortest_id = idx

        if shortest_id is not None:
            # remove the single shortest wall
            walls.pop(shortest_id)

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
