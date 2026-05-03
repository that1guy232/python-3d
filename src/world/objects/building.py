from __future__ import annotations

from typing import List, Optional
import math

from pygame.math import Vector3

from .wall_tile import WallTile
from engine.rendering.lighting import INDOOR_LIGHT_FACTOR, INDOOR_NORMAL


class Building:
    DEFAULT_SIDE = 100.0
    WALL_NORMALS = ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0))

    def __init__(
        self,
        position: Vector3 | tuple[float, float] | tuple[float, float, float] | None = None,
        *,
        target_height: float | None = None,
    ) -> None:
        self.position = self._coerce_position(position)
        self.center = Vector3(self.position.x, self.position.y, self.position.z)
        self.shapes: list = []
        self._bbox: Optional[tuple[float, float, float, float]] = None
        self.target_height = target_height
        self._height_override: Optional[float] = None

    def _coerce_position(
        self,
        position: Vector3 | tuple[float, float] | tuple[float, float, float] | None,
    ) -> Vector3:
        """Accept Vector3 or tuple positions."""
        if position is None:
            return Vector3(0.0, 0.0, 0.0)
        if isinstance(position, Vector3):
            return position
        if len(position) == 2:
            return Vector3(float(position[0]), 0.0, float(position[1]))
        return Vector3(float(position[0]), float(position[1]), float(position[2]))

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
        doorway: bool = True,
        doorway_side: str = "south",
        doorway_width: Optional[float] = None,
        doorway_height: Optional[float] = None,
        roof: bool = True,
        roof_thickness: float = 3.0,
        roof_overhang: float = 5.0,
        roof_texture: Optional[int] = None,
    ) -> List[WallTile]:
        """Create a rectangular building shell with a doorway and roof."""
        dimensions = self._get_dimensions(width, depth)
        base_y_value = self._get_base_y(base_y)
        walls = self._create_all_walls(
            dimensions=dimensions,
            wall_height=wall_height,
            wall_thickness=wall_thickness,
            texture=texture,
            uv_repeat=uv_repeat,
            base_y=base_y_value,
            max_tile_width=max_tile_width,
            doorway=doorway,
            doorway_side=doorway_side,
            doorway_width=doorway_width,
            doorway_height=doorway_height,
        )

        if roof:
            walls.extend(
                self._create_roof(
                    dimensions=dimensions,
                    wall_height=wall_height,
                    roof_thickness=roof_thickness,
                    roof_overhang=roof_overhang,
                    texture=roof_texture if roof_texture is not None else texture,
                    uv_repeat=uv_repeat,
                    base_y=base_y_value,
                )
            )

        return walls

    def _get_dimensions(self, width: Optional[float], depth: Optional[float]) -> tuple[float, float]:
        """Get building dimensions, using defaults if not specified."""
        return (
            float(width) if width is not None else self.DEFAULT_SIDE,
            float(depth) if depth is not None else self.DEFAULT_SIDE
        )

    def _create_all_walls(
        self, dimensions: tuple[float, float], wall_height: float, wall_thickness: float,
        texture: Optional[int], uv_repeat: tuple[float, float], base_y: float,
        max_tile_width: float, doorway: bool, doorway_side: str,
        doorway_width: Optional[float], doorway_height: Optional[float],
    ) -> List[WallTile]:
        """Create walls for all four sides."""
        outer_x, outer_z = dimensions
        door_normal = self._normal_from_side(doorway_side)
        walls: list[WallTile] = []
        
        for nx, nz in self.WALL_NORMALS:
            if doorway and self._normals_match((nx, nz), door_normal):
                walls.extend(
                    self._create_doorway_wall_side(
                        nx, nz, outer_x, outer_z, wall_height, wall_thickness,
                        texture, uv_repeat, base_y, max_tile_width,
                        doorway_width, doorway_height,
                    )
                )
            else:
                walls.extend(
                    self._create_wall_side(
                        nx, nz, outer_x, outer_z, wall_height, wall_thickness,
                        texture, uv_repeat, base_y, max_tile_width
                    )
                )
            
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
        _, _, span_half, _ = self._get_side_geometry(
            nx, nz, outer_x, outer_z, wall_thickness
        )
        return self._create_wall_segment(
            nx, nz, outer_x, outer_z, wall_thickness,
            texture, uv_repeat, base_y, max_tile_width,
            -span_half, span_half, 0.0, wall_height,
        )

    def _create_doorway_wall_side(
        self, nx: float, nz: float, outer_x: float, outer_z: float,
        wall_height: float, wall_thickness: float, texture: Optional[int],
        uv_repeat: tuple[float, float], base_y: float, max_tile_width: float,
        doorway_width: Optional[float], doorway_height: Optional[float],
    ) -> List[WallTile]:
        """Create one wall side with a centered doorway opening."""
        _, _, span_half, _ = self._get_side_geometry(
            nx, nz, outer_x, outer_z, wall_thickness
        )
        full_span = span_half * 2.0
        door_width = self._resolve_doorway_width(full_span, doorway_width)
        door_height = self._resolve_doorway_height(wall_height, doorway_height)
        door_half = door_width * 0.5

        walls: list[WallTile] = []
        for span_min, span_max, y_min, y_max in (
            (-span_half, -door_half, 0.0, wall_height),
            (door_half, span_half, 0.0, wall_height),
            (-door_half, door_half, door_height, wall_height),
        ):
            walls.extend(
                self._create_wall_segment(
                    nx, nz, outer_x, outer_z, wall_thickness,
                    texture, uv_repeat, base_y, max_tile_width,
                    span_min, span_max, y_min, y_max,
                )
            )

        return walls

    def _create_wall_segment(
        self, nx: float, nz: float, outer_x: float, outer_z: float,
        wall_thickness: float, texture: Optional[int],
        uv_repeat: tuple[float, float], base_y: float, max_tile_width: float,
        span_min: float, span_max: float, y_min: float, y_max: float,
    ) -> List[WallTile]:
        """Create tiles for a horizontal/vertical slice of one wall side."""
        if span_max <= span_min or y_max <= y_min:
            return []

        center_x, center_z, _, span_axis = self._get_side_geometry(
            nx, nz, outer_x, outer_z, wall_thickness
        )

        full_span = span_max - span_min
        num_tiles = max(1, int(math.ceil(full_span / max_tile_width)))
        tile_width = full_span / num_tiles
        tile_half = tile_width / 2.0
        half_width = wall_thickness / 2.0
        segment_height = y_max - y_min
        segment_center_y = base_y + y_min + segment_height * 0.5
        theta = math.atan2(nz, nx)

        tiles = []
        span_start = (center_x if span_axis == "x" else center_z) + span_min
        
        for i in range(num_tiles):
            center_along = span_start + (i + 0.5) * tile_width
            
            if span_axis == "x":
                tx, tz, w, d = center_along, center_z, half_width, tile_half
            else:
                tx, tz, w, d = center_x, center_along, half_width, tile_half

            tile = WallTile(
                position=Vector3(tx, segment_center_y, tz),
                width=w, height=segment_height * 0.5, depth=d,
                texture=texture, uv_repeat=uv_repeat, thickness=wall_thickness,
            )
            tile.rotation = Vector3(0.0, theta, 0.0)
            tile.indoor_face_indices = (1,)
            tile.indoor_light_factor = INDOOR_LIGHT_FACTOR
            tile.indoor_normal_override = INDOOR_NORMAL
            tiles.append(tile)
            
        self.attach_shapes(tiles)
        return tiles

    def _create_roof(
        self,
        *,
        dimensions: tuple[float, float],
        wall_height: float,
        roof_thickness: float,
        roof_overhang: float,
        texture: Optional[int],
        uv_repeat: tuple[float, float],
        base_y: float,
    ) -> List[WallTile]:
        """Create a thin horizontal roof slab."""
        outer_x, outer_z = dimensions
        thickness = max(0.1, float(roof_thickness))
        overhang = max(0.0, float(roof_overhang))

        roof = WallTile(
            position=Vector3(
                float(self.position.x),
                base_y + float(wall_height) + thickness * 0.5,
                float(self.position.z),
            ),
            width=thickness * 0.5,
            height=(outer_x + overhang * 2.0) * 0.5,
            depth=(outer_z + overhang * 2.0) * 0.5,
            texture=texture,
            uv_repeat=uv_repeat,
            thickness=thickness,
        )
        roof.rotation = Vector3(0.0, 0.0, math.pi * 0.5)
        roof.indoor_face_indices = (1,)
        roof.indoor_light_factor = INDOOR_LIGHT_FACTOR
        roof.indoor_normal_override = INDOOR_NORMAL
        self.attach_shapes([roof])
        return [roof]

    def _get_side_geometry(
        self, nx: float, nz: float, outer_x: float, outer_z: float, wall_thickness: float
    ) -> tuple[float, float, float, str]:
        """Return side center and span axis for a wall normal."""
        cx, cz = float(self.position.x), float(self.position.z)
        half_x, half_z = outer_x / 2.0, outer_z / 2.0
        half_width = wall_thickness / 2.0

        if abs(nz) > 0.0:
            center_x, center_z = cx, cz + nz * (half_z - half_width)
            span_half, span_axis = half_x, "x"
        else:
            center_x, center_z = cx + nx * (half_x - half_width), cz
            span_half, span_axis = half_z, "z"

        eps = max(1e-5, 0.01 * min(1.0, wall_thickness))
        center_x -= nx * eps
        center_z -= nz * eps
        return center_x, center_z, span_half, span_axis

    def _resolve_doorway_width(self, span: float, doorway_width: Optional[float]) -> float:
        """Clamp doorway width so both side jambs remain visible."""
        if doorway_width is None:
            doorway_width = min(80.0, max(28.0, span * 0.22))

        max_width = max(8.0, span - 12.0)
        return max(8.0, min(float(doorway_width), max_width))

    def _resolve_doorway_height(
        self, wall_height: float, doorway_height: Optional[float]
    ) -> float:
        """Clamp doorway height below the wall top so a lintel can be drawn."""
        if doorway_height is None:
            doorway_height = min(32.0, float(wall_height) * 0.72)

        max_height = max(4.0, float(wall_height) - 4.0)
        return max(4.0, min(float(doorway_height), max_height))

    def _normal_from_side(self, side: str) -> tuple[float, float]:
        """Map a cardinal side name to an XZ normal."""
        side_key = side.lower()
        if side_key in {"north", "front_z+", "z+"}:
            return (0.0, 1.0)
        if side_key in {"east", "right", "x+"}:
            return (1.0, 0.0)
        if side_key in {"south", "front", "front_z-", "z-"}:
            return (0.0, -1.0)
        if side_key in {"west", "left", "x-"}:
            return (-1.0, 0.0)
        return (0.0, -1.0)

    def _normals_match(
        self, first: tuple[float, float], second: tuple[float, float]
    ) -> bool:
        """Compare wall normals with a little tolerance."""
        return abs(first[0] - second[0]) < 1e-6 and abs(first[1] - second[1]) < 1e-6

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
