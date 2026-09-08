"""Building footprint helper that turns specs into wall and roof pieces."""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional
import math

from pygame.math import Vector3

from engine.textures.texture_utils import get_texture_size
from engine.geometry.wall import WallTile
from engine.rendering.lighting.model import INDOOR_LIGHT_FACTOR, INDOOR_NORMAL


class _GabledRoofGeometry(NamedTuple):
    """Shared dimensions for the render-only pitched roof and its trim."""

    ridge_axis: str
    run: float
    ridge_half_length: float
    rise: float
    pitch: float
    slope_half_length: float


class _BuildingPalette(NamedTuple):
    """Flat-shaded colors shared by one procedural cottage variant."""

    wall: tuple[float, float, float]
    timber: tuple[float, float, float]
    roof: tuple[float, float, float]
    stone: tuple[float, float, float]


_BUILDING_PALETTES = (
    _BuildingPalette(
        wall=(0.78, 0.61, 0.34),
        timber=(0.30, 0.17, 0.08),
        roof=(0.58, 0.24, 0.13),
        stone=(0.47, 0.49, 0.42),
    ),
    _BuildingPalette(
        wall=(0.55, 0.67, 0.43),
        timber=(0.25, 0.18, 0.09),
        roof=(0.20, 0.34, 0.32),
        stone=(0.44, 0.48, 0.43),
    ),
    _BuildingPalette(
        wall=(0.76, 0.69, 0.49),
        timber=(0.34, 0.20, 0.10),
        roof=(0.29, 0.39, 0.46),
        stone=(0.50, 0.51, 0.46),
    ),
    _BuildingPalette(
        wall=(0.68, 0.45, 0.29),
        timber=(0.27, 0.16, 0.08),
        roof=(0.26, 0.36, 0.23),
        stone=(0.46, 0.45, 0.39),
    ),
)


class Building:
    """A rectangular building authoring object with cached footprint bounds."""

    DEFAULT_SIDE = 100.0
    WALL_NORMALS = ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0))
    ROOF_RISE_RATIO = 0.36
    ROOF_MIN_RISE = 28.0
    ROOF_MAX_RISE = 68.0
    GABLE_STRIP_WIDTH = 8.0

    def __init__(
        self,
        position: (
            Vector3 | tuple[float, float] | tuple[float, float, float] | None
        ) = None,
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
        windows: Optional[list[dict]] = None,
        roof: bool = True,
        roof_thickness: float = 3.0,
        roof_overhang: float = 5.0,
        roof_texture: Optional[int] = None,
        terrain_height_at: Optional[Callable[[float, float], float]] = None,
        terrain_embed_depth: float = 4.0,
        terrain_sample_spacing: float = 24.0,
        wall_color: Optional[tuple[float, float, float]] = None,
        roof_color: Optional[tuple[float, float, float]] = None,
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
            windows=windows,
            terrain_height_at=terrain_height_at,
            terrain_embed_depth=terrain_embed_depth,
            terrain_sample_spacing=terrain_sample_spacing,
        )
        if wall_color is not None:
            self._paint_tiles(walls, wall_color)

        if roof:
            roof_pieces = self._create_roof(
                dimensions=dimensions,
                wall_height=wall_height,
                roof_thickness=roof_thickness,
                roof_overhang=roof_overhang,
                texture=roof_texture if roof_texture is not None else texture,
                uv_repeat=uv_repeat,
                base_y=base_y_value,
            )
            if roof_color is not None:
                self._paint_tiles(roof_pieces, roof_color)
            walls.extend(roof_pieces)

        return walls

    def create_interior_walls(
        self,
        partitions: list[dict] | None,
        *,
        wall_height: float = 3.0,
        wall_thickness: float = 0.5,
        texture: Optional[int] = None,
        uv_repeat: tuple[float, float] = (1.0, 1.0),
        base_y: Optional[float] = None,
        max_tile_width: float = 100.0,
        wall_color: Optional[tuple[float, float, float]] = None,
    ) -> List[WallTile]:
        """Create internal partition walls from local building-space specs."""
        base_y_value = self._get_base_y(base_y)
        walls: list[WallTile] = []

        for partition in partitions or ():
            if not isinstance(partition, dict):
                continue
            try:
                axis = str(partition.get("axis", "")).lower()
                coord = float(partition["coord"])
                span_min = float(partition["span_min"])
                span_max = float(partition["span_max"])
            except (KeyError, TypeError, ValueError):
                continue

            if axis not in {"x", "z"}:
                continue
            if span_max < span_min:
                span_min, span_max = span_max, span_min
            if span_max <= span_min:
                continue

            side = str(
                partition.get(
                    "side",
                    "east" if axis == "x" else "north",
                )
            ).lower()
            nx, nz = self._normal_from_side(side)
            openings = self._clip_span_openings(
                list(partition.get("openings", ()) or ()),
                span_min=span_min,
                span_max=span_max,
                wall_height=wall_height,
            )
            walls.extend(
                self._create_interior_wall_with_openings(
                    axis=axis,
                    coord=coord,
                    span_min=span_min,
                    span_max=span_max,
                    nx=nx,
                    nz=nz,
                    wall_height=wall_height,
                    wall_thickness=wall_thickness,
                    texture=texture,
                    uv_repeat=uv_repeat,
                    base_y=base_y_value,
                    max_tile_width=max_tile_width,
                    openings=openings,
                )
            )

        if wall_color is not None:
            self._paint_tiles(walls, wall_color)
        return walls

    def create_architectural_details(
        self,
        *,
        wall_height: float,
        wall_thickness: float,
        width: float,
        depth: float,
        wall_texture: Optional[int],
        roof_texture: Optional[int],
        doorway_side: str = "south",
        doorway_width: float = 32.0,
        doorway_height: float = 48.0,
        windows: Optional[list[dict]] = None,
        base_y: Optional[float] = None,
        roof_thickness: float = 4.0,
        roof_overhang: float = 6.0,
        detail_variant: int = 0,
        chimney: bool = True,
    ) -> List[WallTile]:
        """Create static, render-only details around the collision shell.

        The regular horizontal roof remains the interior ceiling and collision
        surface.  These pieces provide the visible pitched roof and facade
        depth, and callers deliberately keep them out of the collision index.
        """
        dimensions = self._get_dimensions(width, depth)
        outer_x, outer_z = dimensions
        if outer_x <= 1.0 or outer_z <= 1.0 or float(wall_height) <= 1.0:
            return []

        base_y_value = self._get_base_y(base_y)
        palette = self.style_palette(detail_variant)
        roof_geometry = self._gabled_roof_geometry(
            dimensions,
            roof_overhang=roof_overhang,
        )
        details: list[WallTile] = []
        details.extend(
            self._create_gabled_roof_panels(
                geometry=roof_geometry,
                wall_height=wall_height,
                roof_thickness=roof_thickness,
                texture=roof_texture,
                base_y=base_y_value,
                fallback_color=palette.roof,
            )
        )
        details.extend(
            self._create_gable_fill(
                dimensions=dimensions,
                geometry=roof_geometry,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
                texture=wall_texture,
                base_y=base_y_value,
                fallback_color=palette.wall,
            )
        )
        details.extend(
            self._create_facade_details(
                dimensions=dimensions,
                geometry=roof_geometry,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
                wall_texture=wall_texture,
                roof_texture=roof_texture,
                doorway_side=doorway_side,
                doorway_width=doorway_width,
                doorway_height=doorway_height,
                windows=windows,
                base_y=base_y_value,
                roof_thickness=roof_thickness,
                detail_variant=detail_variant,
                chimney=chimney,
                palette=palette,
            )
        )

        for detail in details:
            detail.architectural_detail = True
        self.attach_shapes(details)
        return details

    def _get_dimensions(
        self, width: Optional[float], depth: Optional[float]
    ) -> tuple[float, float]:
        """Get building dimensions, using defaults if not specified."""
        return (
            float(width) if width is not None else self.DEFAULT_SIDE,
            float(depth) if depth is not None else self.DEFAULT_SIDE,
        )

    def _create_all_walls(
        self,
        dimensions: tuple[float, float],
        wall_height: float,
        wall_thickness: float,
        texture: Optional[int],
        uv_repeat: tuple[float, float],
        base_y: float,
        max_tile_width: float,
        doorway: bool,
        doorway_side: str,
        doorway_width: Optional[float],
        doorway_height: Optional[float],
        windows: Optional[list[dict]],
        terrain_height_at: Optional[Callable[[float, float], float]],
        terrain_embed_depth: float,
        terrain_sample_spacing: float,
    ) -> List[WallTile]:
        """Create walls for all four sides."""
        outer_x, outer_z = dimensions
        door_normal = self._normal_from_side(doorway_side)
        walls: list[WallTile] = []

        for nx, nz in self.WALL_NORMALS:
            openings: list[dict[str, float]] = []
            if doorway and self._normals_match((nx, nz), door_normal):
                _, _, span_half, _ = self._get_side_geometry(
                    nx, nz, outer_x, outer_z, wall_thickness
                )
                full_span = span_half * 2.0
                door_width = self._resolve_doorway_width(full_span, doorway_width)
                door_height = self._resolve_doorway_height(wall_height, doorway_height)
                openings.append(
                    {
                        "offset": 0.0,
                        "width": door_width,
                        "y_min": 0.0,
                        "y_max": door_height,
                    }
                )

            openings.extend(
                self._window_openings_for_side(
                    nx,
                    nz,
                    windows=windows,
                    wall_height=wall_height,
                )
            )

            if openings:
                walls.extend(
                    self._create_wall_side_with_openings(
                        nx,
                        nz,
                        outer_x,
                        outer_z,
                        wall_height,
                        wall_thickness,
                        texture,
                        uv_repeat,
                        base_y,
                        max_tile_width,
                        openings,
                        terrain_height_at,
                        terrain_embed_depth,
                        terrain_sample_spacing,
                    )
                )
                continue

            walls.extend(
                self._create_wall_side(
                    nx,
                    nz,
                    outer_x,
                    outer_z,
                    wall_height,
                    wall_thickness,
                    texture,
                    uv_repeat,
                    base_y,
                    max_tile_width,
                    terrain_height_at,
                    terrain_embed_depth,
                    terrain_sample_spacing,
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
        self,
        nx: float,
        nz: float,
        outer_x: float,
        outer_z: float,
        wall_height: float,
        wall_thickness: float,
        texture: Optional[int],
        uv_repeat: tuple[float, float],
        base_y: float,
        max_tile_width: float,
        terrain_height_at: Optional[Callable[[float, float], float]],
        terrain_embed_depth: float,
        terrain_sample_spacing: float,
    ) -> List[WallTile]:
        """Create tiles for one wall side."""
        _, _, span_half, _ = self._get_side_geometry(
            nx, nz, outer_x, outer_z, wall_thickness
        )
        return self._create_wall_segment(
            nx,
            nz,
            outer_x,
            outer_z,
            wall_thickness,
            texture,
            uv_repeat,
            base_y,
            max_tile_width,
            -span_half,
            span_half,
            0.0,
            wall_height,
            terrain_height_at=terrain_height_at,
            terrain_embed_depth=terrain_embed_depth,
            terrain_sample_spacing=terrain_sample_spacing,
        )

    def _window_openings_for_side(
        self,
        nx: float,
        nz: float,
        *,
        windows: Optional[list[dict]],
        wall_height: float,
    ) -> list[dict[str, float]]:
        openings: list[dict[str, float]] = []
        for window in windows or ():
            if not isinstance(window, dict):
                continue
            side = str(window.get("side", "")).lower()
            if not side:
                continue
            if not self._normals_match((nx, nz), self._normal_from_side(side)):
                continue

            try:
                width = max(1.0, float(window.get("width", 0.0)))
                height = max(1.0, float(window.get("height", 0.0)))
                sill_height = max(0.0, float(window.get("sill_height", 0.0)))
                offset = float(window.get("offset", 0.0))
            except (TypeError, ValueError):
                continue

            y_min = max(0.0, min(float(wall_height), sill_height))
            y_max = max(0.0, min(float(wall_height), sill_height + height))
            if y_max <= y_min + 1e-6:
                continue

            openings.append(
                {
                    "offset": offset,
                    "width": width,
                    "y_min": y_min,
                    "y_max": y_max,
                }
            )
        return openings

    def _create_wall_side_with_openings(
        self,
        nx: float,
        nz: float,
        outer_x: float,
        outer_z: float,
        wall_height: float,
        wall_thickness: float,
        texture: Optional[int],
        uv_repeat: tuple[float, float],
        base_y: float,
        max_tile_width: float,
        openings: list[dict[str, float]],
        terrain_height_at: Optional[Callable[[float, float], float]],
        terrain_embed_depth: float,
        terrain_sample_spacing: float,
    ) -> List[WallTile]:
        """Create one wall side with rectangular openings cut out."""
        _, _, span_half, _ = self._get_side_geometry(
            nx, nz, outer_x, outer_z, wall_thickness
        )
        clipped_openings = self._clip_wall_openings(
            openings,
            span_half=span_half,
            wall_height=wall_height,
        )
        if not clipped_openings:
            return self._create_wall_side(
                nx,
                nz,
                outer_x,
                outer_z,
                wall_height,
                wall_thickness,
                texture,
                uv_repeat,
                base_y,
                max_tile_width,
                terrain_height_at,
                terrain_embed_depth,
                terrain_sample_spacing,
            )

        span_breaks = [-span_half, span_half]
        y_breaks = [0.0, float(wall_height)]
        for left, right, y_min, y_max in clipped_openings:
            span_breaks.extend((left, right))
            y_breaks.extend((y_min, y_max))

        span_breaks = sorted(set(round(value, 6) for value in span_breaks))
        y_breaks = sorted(set(round(value, 6) for value in y_breaks))
        walls: list[WallTile] = []
        for span_idx in range(len(span_breaks) - 1):
            span_min = span_breaks[span_idx]
            span_max = span_breaks[span_idx + 1]
            if span_max <= span_min:
                continue
            span_mid = (span_min + span_max) * 0.5
            for y_idx in range(len(y_breaks) - 1):
                y_min = y_breaks[y_idx]
                y_max = y_breaks[y_idx + 1]
                if y_max <= y_min:
                    continue
                y_mid = (y_min + y_max) * 0.5
                if any(
                    left <= span_mid <= right and bottom <= y_mid <= top
                    for left, right, bottom, top in clipped_openings
                ):
                    continue

                walls.extend(
                    self._create_wall_segment(
                        nx,
                        nz,
                        outer_x,
                        outer_z,
                        wall_thickness,
                        texture,
                        uv_repeat,
                        base_y,
                        max_tile_width,
                        span_min,
                        span_max,
                        y_min,
                        y_max,
                        terrain_height_at=terrain_height_at,
                        terrain_embed_depth=terrain_embed_depth,
                        terrain_sample_spacing=terrain_sample_spacing,
                    )
                )

        return walls

    def _clip_wall_openings(
        self,
        openings: list[dict[str, float]],
        *,
        span_half: float,
        wall_height: float,
    ) -> list[tuple[float, float, float, float]]:
        clipped: list[tuple[float, float, float, float]] = []
        min_span = -float(span_half)
        max_span = float(span_half)
        max_y = float(wall_height)
        for opening in openings:
            try:
                half_width = max(0.5, float(opening["width"]) * 0.5)
                offset = float(opening.get("offset", 0.0))
                left = max(min_span, offset - half_width)
                right = min(max_span, offset + half_width)
                y_min = max(0.0, min(max_y, float(opening["y_min"])))
                y_max = max(0.0, min(max_y, float(opening["y_max"])))
            except (KeyError, TypeError, ValueError):
                continue

            if right <= left + 1e-6 or y_max <= y_min + 1e-6:
                continue
            clipped.append((left, right, y_min, y_max))
        return clipped

    def _clip_span_openings(
        self,
        openings: list[dict],
        *,
        span_min: float,
        span_max: float,
        wall_height: float,
    ) -> list[tuple[float, float, float, float]]:
        clipped: list[tuple[float, float, float, float]] = []
        min_span = float(span_min)
        max_span = float(span_max)
        max_y = float(wall_height)
        for opening in openings:
            try:
                half_width = max(0.5, float(opening["width"]) * 0.5)
                offset = float(opening.get("offset", 0.0))
                left = max(min_span, offset - half_width)
                right = min(max_span, offset + half_width)
                y_min = max(0.0, min(max_y, float(opening["y_min"])))
                y_max = max(0.0, min(max_y, float(opening["y_max"])))
            except (KeyError, TypeError, ValueError):
                continue

            if right <= left + 1e-6 or y_max <= y_min + 1e-6:
                continue
            clipped.append((left, right, y_min, y_max))
        return clipped

    def _create_interior_wall_with_openings(
        self,
        *,
        axis: str,
        coord: float,
        span_min: float,
        span_max: float,
        nx: float,
        nz: float,
        wall_height: float,
        wall_thickness: float,
        texture: Optional[int],
        uv_repeat: tuple[float, float],
        base_y: float,
        max_tile_width: float,
        openings: list[tuple[float, float, float, float]],
    ) -> List[WallTile]:
        span_breaks = [float(span_min), float(span_max)]
        y_breaks = [0.0, float(wall_height)]
        for left, right, y_min, y_max in openings:
            span_breaks.extend((left, right))
            y_breaks.extend((y_min, y_max))

        span_breaks = sorted(set(round(value, 6) for value in span_breaks))
        y_breaks = sorted(set(round(value, 6) for value in y_breaks))
        walls: list[WallTile] = []
        for span_idx in range(len(span_breaks) - 1):
            segment_min = span_breaks[span_idx]
            segment_max = span_breaks[span_idx + 1]
            if segment_max <= segment_min:
                continue
            span_mid = (segment_min + segment_max) * 0.5
            for y_idx in range(len(y_breaks) - 1):
                y_min = y_breaks[y_idx]
                y_max = y_breaks[y_idx + 1]
                if y_max <= y_min:
                    continue
                y_mid = (y_min + y_max) * 0.5
                if any(
                    left <= span_mid <= right and bottom <= y_mid <= top
                    for left, right, bottom, top in openings
                ):
                    continue

                walls.extend(
                    self._create_interior_wall_segment(
                        axis=axis,
                        coord=coord,
                        span_min=segment_min,
                        span_max=segment_max,
                        nx=nx,
                        nz=nz,
                        wall_thickness=wall_thickness,
                        texture=texture,
                        uv_repeat=uv_repeat,
                        base_y=base_y,
                        max_tile_width=max_tile_width,
                        y_min=y_min,
                        y_max=y_max,
                    )
                )

        return walls

    def _create_interior_wall_segment(
        self,
        *,
        axis: str,
        coord: float,
        span_min: float,
        span_max: float,
        nx: float,
        nz: float,
        wall_thickness: float,
        texture: Optional[int],
        uv_repeat: tuple[float, float],
        base_y: float,
        max_tile_width: float,
        y_min: float,
        y_max: float,
    ) -> List[WallTile]:
        if span_max <= span_min or y_max <= y_min:
            return []

        full_span = span_max - span_min
        num_tiles = max(1, int(math.ceil(full_span / max_tile_width)))
        tile_width = full_span / num_tiles
        tile_half = tile_width / 2.0
        half_width = wall_thickness / 2.0
        segment_height = y_max - y_min
        theta = math.atan2(nz, nx)
        tiles = []

        for i in range(num_tiles):
            center_along = span_min + (i + 0.5) * tile_width
            if axis == "x":
                tx = float(self.position.x) + float(coord)
                tz = float(self.position.z) + center_along
            else:
                tx = float(self.position.x) + center_along
                tz = float(self.position.z) + float(coord)

            tile = WallTile(
                position=Vector3(tx, base_y + y_min + segment_height * 0.5, tz),
                width=half_width,
                height=segment_height * 0.5,
                depth=tile_half,
                texture=texture,
                uv_repeat=uv_repeat,
                thickness=wall_thickness,
            )
            tile.rotation = Vector3(0.0, theta, 0.0)
            tile.indoor_face_indices = tuple(range(len(tile.faces)))
            tile.indoor_light_factor = INDOOR_LIGHT_FACTOR
            tile.indoor_normal_override = INDOOR_NORMAL
            tiles.append(tile)

        self.attach_shapes(tiles)
        return tiles

    def _create_doorway_wall_side(
        self,
        nx: float,
        nz: float,
        outer_x: float,
        outer_z: float,
        wall_height: float,
        wall_thickness: float,
        texture: Optional[int],
        uv_repeat: tuple[float, float],
        base_y: float,
        max_tile_width: float,
        doorway_width: Optional[float],
        doorway_height: Optional[float],
        terrain_height_at: Optional[Callable[[float, float], float]],
        terrain_embed_depth: float,
        terrain_sample_spacing: float,
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
                    nx,
                    nz,
                    outer_x,
                    outer_z,
                    wall_thickness,
                    texture,
                    uv_repeat,
                    base_y,
                    max_tile_width,
                    span_min,
                    span_max,
                    y_min,
                    y_max,
                    terrain_height_at=terrain_height_at,
                    terrain_embed_depth=terrain_embed_depth,
                    terrain_sample_spacing=terrain_sample_spacing,
                )
            )

        return walls

    def _create_wall_segment(
        self,
        nx: float,
        nz: float,
        outer_x: float,
        outer_z: float,
        wall_thickness: float,
        texture: Optional[int],
        uv_repeat: tuple[float, float],
        base_y: float,
        max_tile_width: float,
        span_min: float,
        span_max: float,
        y_min: float,
        y_max: float,
        *,
        terrain_height_at: Optional[Callable[[float, float], float]] = None,
        terrain_embed_depth: float = 4.0,
        terrain_sample_spacing: float = 24.0,
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

            bottom_y, top_y = self._wall_segment_vertical_bounds(
                base_y=base_y,
                y_min=y_min,
                y_max=y_max,
                span_axis=span_axis,
                center_x=center_x,
                center_z=center_z,
                center_along_start=center_along - tile_half,
                center_along_end=center_along + tile_half,
                nx=nx,
                nz=nz,
                half_width=half_width,
                terrain_height_at=terrain_height_at,
                terrain_embed_depth=terrain_embed_depth,
                terrain_sample_spacing=terrain_sample_spacing,
            )
            tile_height = max(0.0, (top_y - bottom_y) * 0.5)
            if tile_height <= 0.0:
                continue

            tile = WallTile(
                position=Vector3(tx, (bottom_y + top_y) * 0.5, tz),
                width=w,
                height=tile_height,
                depth=d,
                texture=texture,
                uv_repeat=uv_repeat,
                thickness=wall_thickness,
            )
            tile.rotation = Vector3(0.0, theta, 0.0)
            tile.indoor_face_indices = (1,)
            tile.indoor_light_factor = INDOOR_LIGHT_FACTOR
            tile.indoor_normal_override = INDOOR_NORMAL
            tiles.append(tile)

        self.attach_shapes(tiles)
        return tiles

    def _wall_segment_vertical_bounds(
        self,
        *,
        base_y: float,
        y_min: float,
        y_max: float,
        span_axis: str,
        center_x: float,
        center_z: float,
        center_along_start: float,
        center_along_end: float,
        nx: float,
        nz: float,
        half_width: float,
        terrain_height_at: Optional[Callable[[float, float], float]],
        terrain_embed_depth: float,
        terrain_sample_spacing: float,
    ) -> tuple[float, float]:
        top_y = float(base_y) + float(y_max)
        bottom_y = float(base_y) + float(y_min)
        if terrain_height_at is None or float(y_min) > 1e-6:
            return bottom_y, top_y

        min_ground_y = self._sample_min_ground_y_under_wall_segment(
            span_axis=span_axis,
            center_x=center_x,
            center_z=center_z,
            center_along_start=center_along_start,
            center_along_end=center_along_end,
            nx=nx,
            nz=nz,
            half_width=half_width,
            terrain_height_at=terrain_height_at,
            terrain_sample_spacing=terrain_sample_spacing,
        )
        if min_ground_y is not None:
            bottom_y = min(
                bottom_y,
                min_ground_y - max(0.0, float(terrain_embed_depth)),
            )
        return bottom_y, top_y

    def _sample_min_ground_y_under_wall_segment(
        self,
        *,
        span_axis: str,
        center_x: float,
        center_z: float,
        center_along_start: float,
        center_along_end: float,
        nx: float,
        nz: float,
        half_width: float,
        terrain_height_at: Callable[[float, float], float],
        terrain_sample_spacing: float,
    ) -> float | None:
        span = max(0.0, float(center_along_end) - float(center_along_start))
        spacing = max(1.0, float(terrain_sample_spacing))
        sample_count = max(2, int(math.ceil(span / spacing)) + 1)
        offsets = (-half_width, 0.0, half_width)
        min_y: float | None = None

        for index in range(sample_count):
            if sample_count == 1:
                along = (center_along_start + center_along_end) * 0.5
            else:
                t = index / (sample_count - 1)
                along = center_along_start + (center_along_end - center_along_start) * t
            for offset in offsets:
                if span_axis == "x":
                    x = along
                    z = center_z + nz * offset
                else:
                    x = center_x + nx * offset
                    z = along
                try:
                    ground_y = float(terrain_height_at(x, z))
                except (TypeError, ValueError, AttributeError):
                    continue
                if min_y is None or ground_y < min_y:
                    min_y = ground_y

        return min_y

    def _gabled_roof_geometry(
        self,
        dimensions: tuple[float, float],
        *,
        roof_overhang: float,
    ) -> _GabledRoofGeometry:
        outer_x, outer_z = (max(1.0, float(value)) for value in dimensions)
        overhang = max(0.0, float(roof_overhang))
        ridge_axis = "x" if outer_x >= outer_z else "z"
        short_span = outer_z if ridge_axis == "x" else outer_x
        long_span = outer_x if ridge_axis == "x" else outer_z
        run = short_span * 0.5 + overhang
        rise = max(
            self.ROOF_MIN_RISE,
            min(self.ROOF_MAX_RISE, short_span * self.ROOF_RISE_RATIO),
        )
        # Keep very small hand-authored footprints from producing a near-wall.
        rise = min(rise, max(4.0, run * 0.72))
        pitch = math.atan2(rise, run)
        return _GabledRoofGeometry(
            ridge_axis=ridge_axis,
            run=run,
            ridge_half_length=long_span * 0.5 + overhang,
            rise=rise,
            pitch=pitch,
            slope_half_length=math.hypot(run, rise) * 0.5,
        )

    @staticmethod
    def style_palette(detail_variant: int = 0) -> _BuildingPalette:
        """Return a deterministic natural palette for one building."""

        return _BUILDING_PALETTES[int(detail_variant) % len(_BUILDING_PALETTES)]

    @staticmethod
    def _shade_color(
        color: tuple[float, float, float],
        factor: float,
    ) -> tuple[float, float, float]:
        return tuple(
            max(0.0, min(1.0, float(component) * float(factor)))
            for component in color
        )

    @classmethod
    def _paint_tile(
        cls,
        tile: WallTile,
        color: tuple[float, float, float],
    ) -> None:
        """Give a box subtle face variation like nearby procedural props."""

        face_factors = (1.0, 0.72, 0.88, 0.82, 1.10, 0.62)
        tile.face_colors = [
            cls._shade_color(color, factor)
            for factor in face_factors[: len(tile.faces)]
        ]

    @classmethod
    def _paint_tiles(
        cls,
        tiles: list[WallTile],
        color: tuple[float, float, float],
    ) -> None:
        for tile in tiles:
            cls._paint_tile(tile, color)

    def _create_gabled_roof_panels(
        self,
        *,
        geometry: _GabledRoofGeometry,
        wall_height: float,
        roof_thickness: float,
        texture: Optional[int],
        base_y: float,
        fallback_color: tuple[float, float, float],
    ) -> list[WallTile]:
        thickness = max(0.5, float(roof_thickness))
        cx = float(self.position.x)
        cz = float(self.position.z)
        wall_top = float(base_y) + float(wall_height)
        center_y = (
            wall_top
            + geometry.rise * 0.5
            + thickness * math.cos(geometry.pitch) * 0.5
        )

        if geometry.ridge_axis == "z":
            panel_specs = (
                (
                    Vector3(cx - geometry.run * 0.5, center_y, cz),
                    Vector3(0.0, 0.0, math.pi * 0.5 + geometry.pitch),
                ),
                (
                    Vector3(cx + geometry.run * 0.5, center_y, cz),
                    Vector3(0.0, 0.0, math.pi * 0.5 - geometry.pitch),
                ),
            )
        else:
            panel_specs = (
                (
                    Vector3(cx, center_y, cz - geometry.run * 0.5),
                    Vector3(
                        math.pi * 0.5 - geometry.pitch,
                        -math.pi * 0.5,
                        0.0,
                    ),
                ),
                (
                    Vector3(cx, center_y, cz + geometry.run * 0.5),
                    Vector3(
                        geometry.pitch - math.pi * 0.5,
                        math.pi * 0.5,
                        0.0,
                    ),
                ),
            )

        panels: list[WallTile] = []
        for panel_index, (position, rotation) in enumerate(panel_specs):
            panel = WallTile(
                position=position,
                width=thickness * 0.5,
                height=geometry.slope_half_length,
                depth=geometry.ridge_half_length,
                texture=texture,
                uv_repeat=(1.0, 1.0),
                thickness=thickness,
            )
            panel.rotation = rotation
            panel.indoor_face_indices = (1,)
            panel.indoor_light_factor = INDOOR_LIGHT_FACTOR
            panel.indoor_normal_override = INDOOR_NORMAL
            panel.flip_texture_v = (
                geometry.ridge_axis == "z" and panel_index == 1
            )
            self._paint_tile(panel, fallback_color)
            panels.append(panel)
        return panels

    def _create_gable_fill(
        self,
        *,
        dimensions: tuple[float, float],
        geometry: _GabledRoofGeometry,
        wall_height: float,
        wall_thickness: float,
        texture: Optional[int],
        base_y: float,
        fallback_color: tuple[float, float, float],
    ) -> list[WallTile]:
        outer_x, outer_z = dimensions
        gable_span = outer_z if geometry.ridge_axis == "x" else outer_x
        gable_sides = (
            ("east", "west")
            if geometry.ridge_axis == "x"
            else ("north", "south")
        )
        strip_count = max(
            1,
            int(math.ceil(float(gable_span) / self.GABLE_STRIP_WIDTH)),
        )
        if strip_count % 2 == 0:
            strip_count += 1
        strip_width = float(gable_span) / strip_count
        span_start = -float(gable_span) * 0.5
        wall_top = float(base_y) + float(wall_height)
        pieces: list[WallTile] = []

        for strip_index in range(strip_count):
            span_min = span_start + strip_index * strip_width
            span_max = span_min + strip_width
            nearest_to_ridge = min(abs(span_min), abs(span_max))
            strip_height = geometry.rise * max(
                0.0,
                1.0 - nearest_to_ridge / max(1e-6, geometry.run),
            )
            if strip_height <= 0.25:
                continue
            for side in gable_sides:
                pieces.append(
                    self._create_wall_detail_segment(
                        dimensions=dimensions,
                        side=side,
                        span_min=span_min,
                        span_max=span_max,
                        bottom_y=wall_top - 0.35,
                        top_y=wall_top + strip_height,
                        wall_thickness=wall_thickness,
                        texture=texture,
                        fallback_color=fallback_color,
                    )
                )
        return pieces

    def _create_wall_detail_segment(
        self,
        *,
        dimensions: tuple[float, float],
        side: str,
        span_min: float,
        span_max: float,
        bottom_y: float,
        top_y: float,
        wall_thickness: float,
        texture: Optional[int],
        fallback_color: tuple[float, float, float],
    ) -> WallTile:
        nx, nz = self._normal_from_side(side)
        outer_x, outer_z = dimensions
        center_x, center_z, _, span_axis = self._get_side_geometry(
            nx,
            nz,
            outer_x,
            outer_z,
            wall_thickness,
        )
        span_center = (float(span_min) + float(span_max)) * 0.5
        span_half = max(0.05, (float(span_max) - float(span_min)) * 0.5)
        half_thickness = max(0.25, float(wall_thickness) * 0.5)
        if span_axis == "x":
            position = Vector3(
                center_x + span_center,
                (bottom_y + top_y) * 0.5,
                center_z,
            )
            width, depth = half_thickness, span_half
        else:
            position = Vector3(
                center_x,
                (bottom_y + top_y) * 0.5,
                center_z + span_center,
            )
            width, depth = half_thickness, span_half

        tile = WallTile(
            position=position,
            width=width,
            height=max(0.05, (top_y - bottom_y) * 0.5),
            depth=depth,
            texture=texture,
            uv_repeat=(1.0, 1.0),
            thickness=max(0.5, float(wall_thickness)),
        )
        tile.rotation = Vector3(0.0, math.atan2(nz, nx), 0.0)
        tile.indoor_face_indices = (1,)
        tile.indoor_light_factor = INDOOR_LIGHT_FACTOR
        tile.indoor_normal_override = INDOOR_NORMAL
        tile.flip_texture_u = str(side).lower() in {"north", "west"}
        texture_size = get_texture_size(texture)
        if texture_size:
            texture_width, texture_height = texture_size
            tile.uv_offset = (
                float(span_min) / max(1.0, float(texture_width)),
                float(bottom_y) / max(1.0, float(texture_height)),
            )
        self._paint_tile(tile, fallback_color)
        return tile

    def _create_facade_details(
        self,
        *,
        dimensions: tuple[float, float],
        geometry: _GabledRoofGeometry,
        wall_height: float,
        wall_thickness: float,
        wall_texture: Optional[int],
        roof_texture: Optional[int],
        doorway_side: str,
        doorway_width: float,
        doorway_height: float,
        windows: Optional[list[dict]],
        base_y: float,
        roof_thickness: float,
        detail_variant: int,
        chimney: bool,
        palette: _BuildingPalette,
    ) -> list[WallTile]:
        outer_x, outer_z = (float(value) for value in dimensions)
        cx = float(self.position.x)
        cz = float(self.position.z)
        half_x = outer_x * 0.5
        half_z = outer_z * 0.5
        wall_top = float(base_y) + float(wall_height)
        thickness = max(0.5, float(wall_thickness))
        projection = max(1.5, thickness * 0.7)
        trim_depth = thickness + projection * 2.0
        details: list[WallTile] = []
        door_nx, door_nz = self._normal_from_side(doorway_side)
        if door_nz > 0.0:
            doorway_side_key = "north"
        elif door_nx > 0.0:
            doorway_side_key = "east"
        elif door_nx < 0.0:
            doorway_side_key = "west"
        else:
            doorway_side_key = "south"
        door_span = (
            outer_x
            if doorway_side_key in {"north", "south"}
            else outer_z
        )
        door_width = self._resolve_doorway_width(door_span, doorway_width)
        door_height = self._resolve_doorway_height(wall_height, doorway_height)

        # A low projecting plinth hides the terrain seam and gives the walls a base.
        foundation_bottom = float(base_y) - 6.0
        foundation_top = float(base_y) + 5.0
        foundation_y = (foundation_bottom + foundation_top) * 0.5
        foundation_height = foundation_top - foundation_bottom
        for side in ("north", "east", "south", "west"):
            side_span = outer_x if side in {"north", "south"} else outer_z
            segments = [(-side_span * 0.5, side_span * 0.5)]
            if side == doorway_side_key:
                gap_half = min(
                    side_span * 0.5,
                    door_width * 0.5 + 2.0,
                )
                segments = [
                    (-side_span * 0.5, -gap_half),
                    (gap_half, side_span * 0.5),
                ]
            for segment_min, segment_max in segments:
                segment_size = segment_max - segment_min
                if segment_size <= 0.1:
                    continue
                details.append(
                    self._create_wall_feature_box(
                        dimensions=dimensions,
                        side=side,
                        offset=(segment_min + segment_max) * 0.5,
                        tangent_size=segment_size,
                        center_y=foundation_y,
                        size_y=foundation_height,
                        normal_depth=trim_depth,
                        texture=wall_texture,
                        fallback_color=palette.stone,
                    )
                )

        # Heavy corner posts and an eave beam give the shell the same chunky,
        # low-poly construction language as the procedural trees and fences.
        eave_beam_height = 6.0
        post_size = min(8.0, max(6.0, min(outer_x, outer_z) * 0.055))
        post_bottom = foundation_top
        post_top = wall_top - eave_beam_height
        if post_top > post_bottom:
            for x_sign in (-1.0, 1.0):
                for z_sign in (-1.0, 1.0):
                    details.append(
                        self._axis_aligned_detail_box(
                            center=Vector3(
                                cx
                                + x_sign
                                * (half_x + post_size * 0.5 - 2.0),
                                (post_bottom + post_top) * 0.5,
                                cz
                                + z_sign
                                * (half_z + post_size * 0.5 - 2.0),
                            ),
                            size_x=post_size,
                            size_y=post_top - post_bottom,
                            size_z=post_size,
                            texture=wall_texture,
                            fallback_color=palette.timber,
                        )
                    )

        eave_beam_y = wall_top - eave_beam_height * 0.5
        for side in ("north", "east", "south", "west"):
            side_span = outer_x if side in {"north", "south"} else outer_z
            details.append(
                self._create_wall_feature_box(
                    dimensions=dimensions,
                    side=side,
                    offset=0.0,
                    tangent_size=side_span,
                    center_y=eave_beam_y,
                    size_y=eave_beam_height,
                    normal_depth=trim_depth,
                    texture=wall_texture,
                    fallback_color=palette.timber,
                )
            )

        # Divide broad procedural footprints into human-scale bays. Posts that
        # would cover a door or window are omitted, preserving every opening.
        protected_by_side: dict[str, list[tuple[float, float]]] = {
            side: [] for side in ("north", "east", "south", "west")
        }
        protected_by_side[doorway_side_key].append((0.0, door_width * 0.5 + 8.0))
        for window in windows or ():
            if not isinstance(window, dict):
                continue
            try:
                side = str(window.get("side", "north")).lower()
                offset = float(window.get("offset", 0.0))
                width = max(4.0, float(window.get("width", 0.0)))
            except (TypeError, ValueError):
                continue
            if side in protected_by_side:
                protected_by_side[side].append((offset, width * 0.5 + 8.0))

        bay_post_width = 5.0
        for side in ("north", "east", "south", "west"):
            side_span = outer_x if side in {"north", "south"} else outer_z
            post_count = max(0, int(math.ceil(side_span / 92.0)) - 1)
            for post_index in range(1, post_count + 1):
                offset = -side_span * 0.5 + side_span * post_index / (
                    post_count + 1
                )
                if any(
                    abs(offset - center) <= half_width + bay_post_width
                    for center, half_width in protected_by_side[side]
                ):
                    continue
                details.append(
                    self._create_wall_feature_box(
                        dimensions=dimensions,
                        side=side,
                        offset=offset,
                        tangent_size=bay_post_width,
                        center_y=(post_bottom + post_top) * 0.5,
                        size_y=post_top - post_bottom,
                        normal_depth=trim_depth,
                        texture=wall_texture,
                        fallback_color=palette.timber,
                    )
                )

        # Give the centered entrance an architectural surround without changing
        # the actual doorway or animated door collision.
        jamb_width = 5.0
        opening_trim_depth = max(4.5, thickness + 2.0)
        for offset in (
            -(door_width + jamb_width) * 0.5,
            (door_width + jamb_width) * 0.5,
        ):
            details.append(
                self._create_wall_feature_box(
                    dimensions=dimensions,
                    side=doorway_side_key,
                    offset=offset,
                    tangent_size=jamb_width,
                    center_y=float(base_y) + door_height * 0.5,
                    size_y=door_height,
                    normal_depth=opening_trim_depth,
                    texture=wall_texture,
                    fallback_color=palette.timber,
                )
            )
        details.append(
            self._create_wall_feature_box(
                dimensions=dimensions,
                side=doorway_side_key,
                offset=0.0,
                tangent_size=door_width + jamb_width * 2.0,
                center_y=float(base_y) + door_height + jamb_width * 0.5,
                size_y=jamb_width,
                normal_depth=opening_trim_depth,
                texture=wall_texture,
                fallback_color=palette.timber,
            )
        )

        # Chunky timber surrounds integrate the small pixel-art windows into the
        # larger low-poly facade; stone sills anchor them at the bottom.
        for window in windows or ():
            if not isinstance(window, dict):
                continue
            try:
                side = str(window.get("side", "north")).lower()
                offset = float(window.get("offset", 0.0))
                window_width = max(4.0, float(window.get("width", 0.0)))
                window_height = max(4.0, float(window.get("height", 0.0)))
                sill_height = max(0.0, float(window.get("sill_height", 0.0)))
            except (TypeError, ValueError):
                continue
            details.append(
                self._create_wall_feature_box(
                    dimensions=dimensions,
                    side=side,
                    offset=offset,
                    tangent_size=window_width + 7.0,
                    center_y=float(base_y) + sill_height - 1.5,
                    size_y=3.0,
                    normal_depth=opening_trim_depth,
                    texture=wall_texture,
                    fallback_color=palette.stone,
                )
            )
            frame_width = 4.0
            for frame_offset in (
                offset - (window_width + frame_width) * 0.5,
                offset + (window_width + frame_width) * 0.5,
            ):
                details.append(
                    self._create_wall_feature_box(
                        dimensions=dimensions,
                        side=side,
                        offset=frame_offset,
                        tangent_size=frame_width,
                        center_y=float(base_y) + sill_height + window_height * 0.5,
                        size_y=window_height,
                        normal_depth=opening_trim_depth,
                        texture=wall_texture,
                        fallback_color=palette.timber,
                    )
                )
            details.append(
                self._create_wall_feature_box(
                    dimensions=dimensions,
                    side=side,
                    offset=offset,
                    tangent_size=window_width + frame_width * 2.0,
                    center_y=float(base_y) + sill_height + window_height + 1.5,
                    size_y=3.0,
                    normal_depth=opening_trim_depth,
                    texture=wall_texture,
                    fallback_color=palette.timber,
                )
            )

        # Roof edge boards hide the stepped gable fill and make the silhouette crisp.
        roof_edge = max(4.5, float(roof_thickness) + 1.0)
        eave_y = wall_top + float(roof_thickness) * math.cos(geometry.pitch) * 0.5
        ridge_y = eave_y + geometry.rise
        if geometry.ridge_axis == "x":
            for sign in (-1.0, 1.0):
                details.append(
                    self._axis_aligned_detail_box(
                        center=Vector3(
                            cx,
                            eave_y,
                            cz + sign * (geometry.run - roof_edge * 0.5),
                        ),
                        size_x=geometry.ridge_half_length * 2.0,
                        size_y=roof_edge,
                        size_z=roof_edge,
                        texture=roof_texture,
                        fallback_color=palette.timber,
                    )
                )
            ridge_size = (geometry.ridge_half_length * 2.0, 6.0)
            details.append(
                self._axis_aligned_detail_box(
                    center=Vector3(cx, ridge_y, cz),
                    size_x=ridge_size[0],
                    size_y=5.0,
                    size_z=ridge_size[1],
                    texture=roof_texture,
                    fallback_color=palette.timber,
                )
            )
        else:
            for sign in (-1.0, 1.0):
                details.append(
                    self._axis_aligned_detail_box(
                        center=Vector3(
                            cx + sign * (geometry.run - roof_edge * 0.5),
                            eave_y,
                            cz,
                        ),
                        size_x=roof_edge,
                        size_y=roof_edge,
                        size_z=geometry.ridge_half_length * 2.0,
                        texture=roof_texture,
                        fallback_color=palette.timber,
                    )
                )
            details.append(
                self._axis_aligned_detail_box(
                    center=Vector3(cx, ridge_y, cz),
                    size_x=6.0,
                    size_y=5.0,
                    size_z=geometry.ridge_half_length * 2.0,
                    texture=roof_texture,
                    fallback_color=palette.timber,
                )
            )

        # Exposed gable beams make the taller roof silhouette legible from the
        # end walls without adding fine texture detail.
        gable_sides = (
            ("east", "west")
            if geometry.ridge_axis == "x"
            else ("north", "south")
        )
        gable_span = outer_z if geometry.ridge_axis == "x" else outer_x
        for side in gable_sides:
            details.append(
                self._create_wall_feature_box(
                    dimensions=dimensions,
                    side=side,
                    offset=0.0,
                    tangent_size=gable_span,
                    center_y=wall_top + 2.0,
                    size_y=4.0,
                    normal_depth=trim_depth,
                    texture=wall_texture,
                    fallback_color=palette.timber,
                )
            )
            gable_post_height = max(1.0, geometry.rise - 4.0)
            details.append(
                self._create_wall_feature_box(
                    dimensions=dimensions,
                    side=side,
                    offset=0.0,
                    tangent_size=5.0,
                    center_y=wall_top + 4.0 + gable_post_height * 0.5,
                    size_y=gable_post_height,
                    normal_depth=trim_depth,
                    texture=wall_texture,
                    fallback_color=palette.timber,
                )
            )

        if chimney:
            details.extend(
                self._create_chimney_details(
                    geometry=geometry,
                    wall_top=wall_top,
                    roof_thickness=roof_thickness,
                    texture=wall_texture,
                    fallback_color=palette.stone,
                    detail_variant=detail_variant,
                )
            )

        return details

    def _create_wall_feature_box(
        self,
        *,
        dimensions: tuple[float, float],
        side: str,
        offset: float,
        tangent_size: float,
        center_y: float,
        size_y: float,
        normal_depth: float,
        texture: Optional[int],
        fallback_color: tuple[float, float, float],
    ) -> WallTile:
        outer_x, outer_z = dimensions
        side_key = str(side).lower()
        nx, nz = self._normal_from_side(side_key)
        cx = float(self.position.x)
        cz = float(self.position.z)
        half_x = float(outer_x) * 0.5
        half_z = float(outer_z) * 0.5
        # Keep decorative surfaces fully in front of the structural shell.
        # A tiny gap avoids coplanar depth fighting while remaining invisible.
        outward_offset = max(0.0, float(normal_depth)) * 0.5 + 0.08
        if abs(nz) > 0.0:
            center = Vector3(
                cx + float(offset),
                center_y,
                cz + nz * (half_z + outward_offset),
            )
            size_x, size_z = tangent_size, normal_depth
        else:
            center = Vector3(
                cx + nx * (half_x + outward_offset),
                center_y,
                cz + float(offset),
            )
            size_x, size_z = normal_depth, tangent_size
        tile = self._axis_aligned_detail_box(
            center=center,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            texture=texture,
            fallback_color=fallback_color,
        )
        inward_face = (
            1
            if nx > 0.0
            else 0
            if nx < 0.0
            else 3
            if nz > 0.0
            else 2
        )
        tile.indoor_face_indices = (inward_face,)
        tile.indoor_light_factor = INDOOR_LIGHT_FACTOR
        tile.indoor_normal_override = INDOOR_NORMAL
        return tile

    def _create_chimney_details(
        self,
        *,
        geometry: _GabledRoofGeometry,
        wall_top: float,
        roof_thickness: float,
        texture: Optional[int],
        fallback_color: tuple[float, float, float],
        detail_variant: int,
    ) -> list[WallTile]:
        variant = int(detail_variant)
        long_sign = -1.0 if variant % 2 else 1.0
        cross_sign = -1.0 if (variant // 2) % 2 else 1.0
        long_offset = long_sign * max(
            0.0,
            (geometry.ridge_half_length - 18.0) * 0.38,
        )
        cross_offset = cross_sign * geometry.run * 0.22
        roof_surface_y = (
            float(wall_top)
            + float(roof_thickness) * math.cos(geometry.pitch) * 0.5
            + geometry.rise
            * (1.0 - abs(cross_offset) / max(1e-6, geometry.run))
        )
        stack_height = 25.0 + (variant % 3) * 2.0
        stack_bottom = roof_surface_y - 3.0
        stack_top = stack_bottom + stack_height
        cx = float(self.position.x)
        cz = float(self.position.z)
        if geometry.ridge_axis == "x":
            stack_center = Vector3(
                cx + long_offset,
                (stack_bottom + stack_top) * 0.5,
                cz + cross_offset,
            )
        else:
            stack_center = Vector3(
                cx + cross_offset,
                (stack_bottom + stack_top) * 0.5,
                cz + long_offset,
            )

        stack = self._axis_aligned_detail_box(
            center=stack_center,
            size_x=12.0,
            size_y=stack_height,
            size_z=12.0,
            texture=texture,
            fallback_color=fallback_color,
        )
        cap = self._axis_aligned_detail_box(
            center=Vector3(stack_center.x, stack_top - 0.5, stack_center.z),
            size_x=15.0,
            size_y=4.0,
            size_z=15.0,
            texture=texture,
            fallback_color=fallback_color,
        )
        return [stack, cap]

    def _axis_aligned_detail_box(
        self,
        *,
        center: Vector3,
        size_x: float,
        size_y: float,
        size_z: float,
        texture: Optional[int],
        fallback_color: tuple[float, float, float],
    ) -> WallTile:
        size_x = max(0.1, float(size_x))
        size_y = max(0.1, float(size_y))
        size_z = max(0.1, float(size_z))
        tile = WallTile(
            position=center,
            width=size_x * 0.5,
            height=size_y * 0.5,
            depth=size_z * 0.5,
            texture=texture,
            uv_repeat=(1.0, 1.0),
            thickness=size_x,
        )
        tile.sample_texture_edges = False
        self._paint_tile(tile, fallback_color)
        return tile

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
        self,
        nx: float,
        nz: float,
        outer_x: float,
        outer_z: float,
        wall_thickness: float,
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

    def _resolve_doorway_width(
        self, span: float, doorway_width: Optional[float]
    ) -> float:
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

    def dispose(self) -> None:
        """Release references to authored CPU geometry during scene teardown."""
        self.shapes.clear()
        self._bbox = None

    def _update_bounding_box(self) -> None:
        """Compute bounding box from attached shapes."""
        if not self.shapes:
            self._bbox = None
            return

        bounds = [
            float("inf"),
            float("-inf"),
            float("inf"),
            float("-inf"),
        ]  # min_x, max_x, min_z, max_z

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
            self.position.x - half,
            self.position.x + half,
            self.position.z - half,
            self.position.z + half,
        )

    def get_bounding_box(self) -> Optional[tuple[float, float, float, float]]:
        """Get cached bounding box or None."""
        return self._bbox

    def contains_point(self, x: float, z: float, margin: float = 0.0) -> bool:
        """Check if point is within building footprint."""
        min_x, max_x, min_z, max_z = self.bounds
        return (
            min_x - margin <= x <= max_x + margin
            and min_z - margin <= z <= max_z + margin
        )

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
