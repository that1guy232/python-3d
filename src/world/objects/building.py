from __future__ import annotations

from typing import List, Optional
import math

from pygame.math import Vector3

from .wall_tile import WallTile


class Building:
    DEFAULT_SIDE = 100.0

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
        self._height_override: Optional[float] = None

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
        max_tile_width: float = 100.0,
    ) -> List[WallTile]:
        """Create a 4-wall rectangular perimeter with the shortest side removed."""
        dimensions = self._get_dimensions(width, depth)
        walls = self._create_all_walls(
            dimensions, wall_height, wall_thickness, texture, uv_repeat, base_y, max_tile_width
        )
        return self._remove_shortest_side(walls)

    def _get_dimensions(self, width: Optional[float], depth: Optional[float]) -> tuple[float, float]:
        """Get building dimensions, using defaults if not specified."""
        return (
            float(width) if width is not None else self.DEFAULT_SIDE,
            float(depth) if depth is not None else self.DEFAULT_SIDE
        )

    def _create_all_walls(
        self, dimensions: tuple[float, float], wall_height: float, wall_thickness: float,
        texture: Optional[int], uv_repeat: tuple[float, float], base_y: Optional[float],
        max_tile_width: float
    ) -> List[List[WallTile]]:
        """Create walls for all four sides."""
        outer_x, outer_z = dimensions
        base_y = self._get_base_y(base_y)
        
        normals = [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]
        walls = []
        
        for nx, nz in normals:
            side_walls = self._create_wall_side(
                nx, nz, outer_x, outer_z, wall_height, wall_thickness,
                texture, uv_repeat, base_y, max_tile_width
            )
            walls.append(side_walls)
            
        return walls

    def _get_base_y(self, base_y: Optional[float]) -> float:
        """Get the base Y coordinate for walls."""
        if base_y is not None:
            return float(base_y)
        if self.target_height is not None:
            return float(self.target_height)
        return float(self.position.y)

    def _create_wall_side(
        self, nx: float, nz: float, outer_x: float, outer_z: float,
        wall_height: float, wall_thickness: float, texture: Optional[int],
        uv_repeat: tuple[float, float], base_y: float, max_tile_width: float
    ) -> List[WallTile]:
        """Create tiles for one wall side."""
        cx, cz = float(self.position.x), float(self.position.z)
        half_x, half_z = outer_x / 2.0, outer_z / 2.0
        half_width = wall_thickness / 2.0
        
        # Calculate wall center and span
        if abs(nz) > 0.0:  # North/South wall
            center_x, center_z = cx, cz + nz * (half_z - half_width)
            span_half, span_axis = half_x, "x"
        else:  # East/West wall
            center_x, center_z = cx + nx * (half_x - half_width), cz
            span_half, span_axis = half_z, "z"

        # Anti-z-fighting nudge
        eps = max(1e-5, 0.01 * min(1.0, wall_thickness))
        center_x -= nx * eps
        center_z -= nz * eps

        # Create tiles
        full_span = span_half * 2.0
        num_tiles = max(1, int(math.ceil(full_span / max_tile_width)))
        tile_width = full_span / num_tiles
        tile_half = tile_width / 2.0
        theta = math.atan2(nz, nx)

        tiles = []
        span_start = (center_x if span_axis == "x" else center_z) - span_half
        
        for i in range(num_tiles):
            center_along = span_start + (i + 0.5) * tile_width
            
            if span_axis == "x":
                tx, tz, w, d = center_along, center_z, half_width, tile_half
            else:
                tx, tz, w, d = center_x, center_along, half_width, tile_half

            tile = WallTile(
                position=Vector3(tx, base_y + wall_height * 0.5, tz),
                width=w, height=wall_height * 0.5, depth=d,
                texture=texture, uv_repeat=uv_repeat, thickness=wall_thickness,
            )
            tile.rotation = Vector3(0.0, theta, 0.0)
            tiles.append(tile)
            
        self.attach_shapes(tiles)
        return tiles

    def _remove_shortest_side(self, all_walls: List[List[WallTile]]) -> List[WallTile]:
        """Remove the shortest wall side and return remaining walls."""
        if not all_walls:
            return []
            
        # Find shortest side
        spans = [len(side_walls) for side_walls in all_walls]  # Use tile count as proxy for span
        shortest_idx = spans.index(min(spans))
        
        # Collect all walls except shortest side
        walls = []
        for i, side_walls in enumerate(all_walls):
            if i != shortest_idx:
                walls.extend(side_walls)
                
        return walls

    def attach_shapes(self, shapes: list) -> None:
        """Attach shapes and update bounding box."""
        if shapes:
            self.shapes.extend(shapes)
            self._update_bounding_box()

    def _update_bounding_box(self) -> None:
        """Compute bounding box from attached shapes."""
        if not self.shapes:
            self._bbox = None
            return

        bounds = [float("inf"), float("-inf"), float("inf"), float("-inf")]  # min_x, max_x, min_z, max_z
        
        for shape in self.shapes:
            try:
                verts = shape.get_world_vertices() or []
                for v in verts:
                    bounds[0] = min(bounds[0], v.x)  # min_x
                    bounds[1] = max(bounds[1], v.x)  # max_x
                    bounds[2] = min(bounds[2], v.z)  # min_z
                    bounds[3] = max(bounds[3], v.z)  # max_z
            except Exception:
                continue

        self._bbox = tuple(bounds) if bounds[0] != float("inf") else None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get bounding box, computing default if needed."""
        if self._bbox is not None:
            return self._bbox
            
        # Default bounds
        half = self.DEFAULT_SIDE / 2.0
        return (
            self.position.x - half, self.position.x + half,
            self.position.z - half, self.position.z + half
        )

    def get_bounding_box(self) -> Optional[tuple[float, float, float, float]]:
        """Get cached bounding box or None."""
        return self._bbox

    def contains_point(self, x: float, z: float, margin: float = 0.0) -> bool:
        """Check if point is within building footprint."""
        min_x, max_x, min_z, max_z = self.bounds
        return (min_x - margin <= x <= max_x + margin and 
                min_z - margin <= z <= max_z + margin)

    def get_floor_level(self) -> float:
        """Get building floor level."""
        if self.target_height is not None:
            return float(self.target_height)

        min_y = float("inf")
        for shape in self.shapes:
            try:
                verts = shape.get_world_vertices() or []
                for v in verts:
                    min_y = min(min_y, v.y)
            except Exception:
                continue

        return 0.0 if min_y == float("inf") else float(min_y)

    @property
    def height(self) -> float:
        """Get building height."""
        if self._height_override is not None:
            return float(self._height_override)

        min_y, max_y = float("inf"), float("-inf")
        for shape in self.shapes:
            try:
                verts = shape.get_world_vertices() or []
                for v in verts:
                    min_y, max_y = min(min_y, v.y), max(max_y, v.y)
            except Exception:
                continue

        if min_y == float("inf") or max_y == float("-inf") or max_y <= min_y:
            return 10.0
        return float(max_y - min_y)

    @height.setter
    def height(self, value: float) -> None:
        """Set building height override."""
        self._height_override = float(value)

    def get_corners(self) -> tuple[List[Vector3], float]:
        """Get building corner positions and floor level."""
        min_x, max_x, min_z, max_z = self.bounds
        floor_y = self.get_floor_level()
        
        corners = [
            Vector3(min_x, floor_y, min_z),
            Vector3(max_x, floor_y, min_z), 
            Vector3(max_x, floor_y, max_z),
            Vector3(min_x, floor_y, max_z),
        ]
        
        return corners, float(floor_y)