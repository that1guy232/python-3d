"""Ground grid builder extracted from renderer.py."""

from __future__ import annotations
from dataclasses import dataclass
import math
import os
import numpy as np
import pygame
from pygame.math import Vector3
from game.world.objects.ground_tile import GroundTile
from game.world.lighting_receivers import GROUND_LIGHTING_RECEIVER
from game.resources.paths import TEXTURES_PATH
from engine.core.mesh import BatchedMesh, GroundHeightSampler
from engine.rendering.geometry_lighting import uses_dynamic_textured_lighting
from engine.rendering.lighting import (
    apply_brightness_modifiers,
    apply_covered_regions,
    apply_directional_sunlight,
    covered_region_factor_at,
    covered_region_mask,
    region_light_openings,
    with_textured_normals,
)


@dataclass(frozen=True)
class TerrainFlattenPad:
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    height: float
    blend_margin: float

    def sampler_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.min_x,
            self.max_x,
            self.min_z,
            self.max_z,
            self.height,
            self.blend_margin,
        )


class TexturedGroundGridBuilder:
    def __init__(
        self,
        count: int,
        tile_size: float,
        gap: float,
        texture,
        brightness_modifiers: list[callable] = None,
        default_brightness: float = 1.0,
        lighting=None,
        sun_direction=None,
        covered_regions=None,
        environment_volumes=None,
        dynamic_lighting: bool | None = None,
    ):
        self.count = count
        self.tile_size = tile_size
        self.gap = gap
        self.texture = texture
        self.brightness_modifiers = brightness_modifiers or []
        self.w = tile_size / 2.0
        self.h = 5
        self.d = tile_size / 2.0
        self.spacing = tile_size + gap
        self.default_brightness = default_brightness
        self.lighting = lighting
        self.sun_direction = None if lighting is not None else sun_direction
        self.covered_regions = covered_regions or []
        self.environment_volumes = (
            None if environment_volumes is None else list(environment_volumes)
        )
        self.dynamic_lighting = dynamic_lighting

        tile = GroundTile(
            position=Vector3(0, 0, 0), width=self.w, height=self.h, depth=self.d
        )
        self.base = tile.local_vertices
        self.faces = tile.faces
        self.face_colors = tile.face_colors
        self.uv_top = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.uv_dummy = [(0, 0)] * 4

        if len(self.faces) == 1:
            self.face_uvs = [self.uv_top]
        else:
            self.face_uvs = [
                self.uv_dummy,
                self.uv_dummy,
                self.uv_dummy,
                self.uv_top,
                self.uv_dummy,
                self.uv_dummy,
            ]

        self.heightmap_path = os.path.join(TEXTURES_PATH, "heightmap.png")
        self.heightmap_amp = 80.0
        self.terrain_pad_min_blend = 96.0
        self.terrain_pad_max_blend = 180.0
        self.terrain_pad_sample_spacing = max(8.0, self.spacing)
        self.terrain_flatten_pads: list[TerrainFlattenPad] = []

    def _make_tile_vertex_array(self) -> np.ndarray:
        tile_vertices_local = []
        for face_idx, (face, color) in enumerate(zip(self.faces, self.face_colors)):
            a, b, c, d_idx = face
            face_uv = self.face_uvs[face_idx]
            tri_indices = [a, b, c, a, c, d_idx]
            tri_uvs = [
                face_uv[0],
                face_uv[1],
                face_uv[2],
                face_uv[0],
                face_uv[2],
                face_uv[3],
            ]
            for idx, uv in zip(tri_indices, tri_uvs):
                v = self.base[idx]
                # Normalize color components (0–1 instead of 0–255)
                r, g, b = [c / 255.0 for c in color]
                tile_vertices_local.append((v.x, v.y, v.z, r, g, b, uv[0], uv[1]))
        return np.array(tile_vertices_local, dtype=np.float32)

    def _load_heightmap(self):
        try:
            if not pygame.get_init():
                pygame.init()
            surf = pygame.image.load(self.heightmap_path)
            surf = surf.convert()
            import pygame.surfarray as surfarray

            hm = surfarray.array3d(surf).astype(np.float32) / 255.0
            hm_w, hm_h = hm.shape[0], hm.shape[1]
            world_min_x = -self.w
            world_max_x = (self.count - 1) * self.spacing + self.w
            world_min_z = -self.d
            world_max_z = (self.count - 1) * self.spacing + self.d
            return hm, hm_w, hm_h, world_min_x, world_max_x, world_min_z, world_max_z
        except Exception as e:
            print(
                f"Warning: failed to load heightmap '{self.heightmap_path}': {e}; using flat ground"
            )
            return None, 0, 0, 0.0, 0.0, 0.0, 0.0

    def _sample_base_heights_vectorized(self, world_coords, heightmap_data):
        """Sample the unmodified heightmap for multiple world coordinates."""
        coords = np.asarray(world_coords, dtype=np.float32)
        if len(coords) == 0:
            return np.zeros(0, dtype=np.float32)

        (
            heightmap_arr,
            hm_w,
            hm_h,
            world_min_x,
            world_max_x,
            world_min_z,
            world_max_z,
        ) = heightmap_data
        if heightmap_arr is None:
            heights = np.full(len(coords), self.h, dtype=np.float32)
        else:
            wx = coords[:, 0]
            wz = coords[:, 1]
            # Vectorized UV calculation
            ux = (wx - world_min_x) / max(1e-12, (world_max_x - world_min_x))
            uz = (wz - world_min_z) / max(1e-12, (world_max_z - world_min_z))
            ux = np.clip(ux, 0.0, 1.0)
            uz = np.clip(uz, 0.0, 1.0)

            px = ux * max(0, hm_w - 1)
            py = uz * max(0, hm_h - 1)
            x0 = np.floor(px).astype(np.int32)
            y0 = np.floor(py).astype(np.int32)
            x1 = np.clip(x0 + 1, 0, max(0, hm_w - 1))
            y1 = np.clip(y0 + 1, 0, max(0, hm_h - 1))
            tx = (px - x0).astype(np.float32)
            ty = (py - y0).astype(np.float32)

            lum00 = heightmap_arr[x0, y0].mean(axis=1)
            lum10 = heightmap_arr[x1, y0].mean(axis=1)
            lum01 = heightmap_arr[x0, y1].mean(axis=1)
            lum11 = heightmap_arr[x1, y1].mean(axis=1)
            lum0 = lum00 * (1.0 - tx) + lum10 * tx
            lum1 = lum01 * (1.0 - tx) + lum11 * tx
            lum = lum0 * (1.0 - ty) + lum1 * ty
            heights = self.h + (lum - 0.5) * 2.0 * self.heightmap_amp
        return heights.astype(np.float32)

    @staticmethod
    def _smooth01(values):
        values = np.clip(values, 0.0, 1.0)
        return values * values * (3.0 - 2.0 * values)

    def _apply_terrain_pads_vectorized(
        self,
        world_coords,
        heights,
        terrain_pads: list[TerrainFlattenPad] | tuple[TerrainFlattenPad, ...] | None,
    ) -> np.ndarray:
        if not terrain_pads:
            return np.asarray(heights, dtype=np.float32)

        coords = np.asarray(world_coords, dtype=np.float32)
        adjusted = np.asarray(heights, dtype=np.float32).copy()
        if len(coords) == 0:
            return adjusted

        wx = coords[:, 0]
        wz = coords[:, 1]
        for pad in terrain_pads:
            inside = (
                (wx >= pad.min_x)
                & (wx <= pad.max_x)
                & (wz >= pad.min_z)
                & (wz <= pad.max_z)
            )
            influence = np.zeros(len(coords), dtype=np.float32)
            influence[inside] = 1.0

            margin = max(0.0, float(pad.blend_margin))
            if margin > 1e-6:
                dx = np.maximum(np.maximum(pad.min_x - wx, 0.0), wx - pad.max_x)
                dz = np.maximum(np.maximum(pad.min_z - wz, 0.0), wz - pad.max_z)
                distance = np.sqrt(dx * dx + dz * dz)
                near = (~inside) & (distance < margin)
                if np.any(near):
                    influence[near] = 1.0 - self._smooth01(distance[near] / margin)

            affected = influence > 0.0
            if not np.any(affected):
                continue
            adjusted[affected] += (float(pad.height) - adjusted[affected]) * influence[
                affected
            ]

        return adjusted.astype(np.float32)

    def _sample_heights_vectorized(
        self,
        world_coords,
        heightmap_data,
        terrain_pads: (
            list[TerrainFlattenPad] | tuple[TerrainFlattenPad, ...] | None
        ) = None,
    ):
        """Sample terrain heights after building-pad flattening."""
        heights = self._sample_base_heights_vectorized(world_coords, heightmap_data)
        return self._apply_terrain_pads_vectorized(
            world_coords,
            heights,
            terrain_pads,
        )

    def _sample_smooth_normals_vectorized(
        self,
        world_coords,
        heightmap_data,
        terrain_pads: list[TerrainFlattenPad] | tuple[TerrainFlattenPad, ...] | None,
    ) -> np.ndarray:
        coords = np.asarray(world_coords, dtype=np.float32)
        if len(coords) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        step = max(1.0, min(float(self.spacing) * 0.25, 6.0))
        left = coords.copy()
        right = coords.copy()
        near = coords.copy()
        far = coords.copy()
        left[:, 0] -= step
        right[:, 0] += step
        near[:, 1] -= step
        far[:, 1] += step

        h_left = self._sample_heights_vectorized(left, heightmap_data, terrain_pads)
        h_right = self._sample_heights_vectorized(right, heightmap_data, terrain_pads)
        h_near = self._sample_heights_vectorized(near, heightmap_data, terrain_pads)
        h_far = self._sample_heights_vectorized(far, heightmap_data, terrain_pads)

        normals = np.column_stack(
            (
                h_left - h_right,
                np.full(len(coords), 2.0 * step, dtype=np.float32),
                h_near - h_far,
            )
        ).astype(np.float32)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-8
        normals[valid] /= lengths[valid, np.newaxis]
        normals[~valid] = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        return normals

    @staticmethod
    def _region_values(region) -> tuple[float, float, float, float] | None:
        try:
            if isinstance(region, dict):
                min_x = float(region["min_x"])
                max_x = float(region["max_x"])
                min_z = float(region["min_z"])
                max_z = float(region["max_z"])
            elif all(
                hasattr(region, name)
                for name in ("min_x", "max_x", "min_z", "max_z")
            ):
                min_x = float(region.min_x)
                max_x = float(region.max_x)
                min_z = float(region.min_z)
                max_z = float(region.max_z)
            else:
                min_x = float(region[0])
                max_x = float(region[1])
                min_z = float(region[2])
                max_z = float(region[3])
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            return None

        if max_x < min_x:
            min_x, max_x = max_x, min_x
        if max_z < min_z:
            min_z, max_z = max_z, min_z
        return min_x, max_x, min_z, max_z

    def _terrain_pad_blend_margin(self, width: float, depth: float) -> float:
        footprint_scale = max(1.0, min(float(width), float(depth)) * 0.55)
        return max(
            self.spacing * 3.0,
            min(
                float(self.terrain_pad_max_blend),
                max(float(self.terrain_pad_min_blend), footprint_scale),
            ),
        )

    def _sample_region_average_base_height(
        self,
        *,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
        heightmap_data,
    ) -> float:
        width = max(0.0, float(max_x) - float(min_x))
        depth = max(0.0, float(max_z) - float(min_z))
        if width <= 0.0 or depth <= 0.0:
            coords = np.array(
                [[(min_x + max_x) * 0.5, (min_z + max_z) * 0.5]],
                dtype=np.float32,
            )
            return float(
                self._sample_base_heights_vectorized(coords, heightmap_data)[0]
            )

        spacing = max(4.0, float(self.terrain_pad_sample_spacing))
        x_count = max(2, int(math.ceil(width / spacing)) + 1)
        z_count = max(2, int(math.ceil(depth / spacing)) + 1)
        xs = np.linspace(min_x, max_x, x_count, dtype=np.float32)
        zs = np.linspace(min_z, max_z, z_count, dtype=np.float32)
        coords = np.array(
            [(float(x), float(z)) for z in zs for x in xs],
            dtype=np.float32,
        )
        heights = self._sample_base_heights_vectorized(coords, heightmap_data)
        if len(heights) == 0:
            return float(self.h)
        return float(np.mean(heights))

    def _build_terrain_flatten_pads(self, heightmap_data) -> list[TerrainFlattenPad]:
        if not heightmap_data or heightmap_data[0] is None:
            return []

        pads: list[TerrainFlattenPad] = []
        terrain_footprints = (
            self.environment_volumes
            if self.environment_volumes is not None
            else self.covered_regions
        )
        for footprint in terrain_footprints:
            values = self._region_values(footprint)
            if values is None:
                continue
            min_x, max_x, min_z, max_z = values
            width = max_x - min_x
            depth = max_z - min_z
            if width <= 1e-6 or depth <= 1e-6:
                continue

            avg_y = self._sample_region_average_base_height(
                min_x=min_x,
                max_x=max_x,
                min_z=min_z,
                max_z=max_z,
                heightmap_data=heightmap_data,
            )
            pads.append(
                TerrainFlattenPad(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    height=avg_y,
                    blend_margin=self._terrain_pad_blend_margin(width, depth),
                )
            )
        return pads

    def _covered_regions_for_tile(
        self,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
    ) -> list[tuple[float, float, float, float]]:
        regions: list[tuple[float, float, float, float]] = []
        for region in self.covered_regions:
            values = self._region_values(region)
            if values is None:
                continue
            region_min_x, region_max_x, region_min_z, region_max_z = values
            if (
                region_max_x <= min_x
                or region_min_x >= max_x
                or region_max_z <= min_z
                or region_min_z >= max_z
            ):
                continue
            regions.append(values)
        return regions

    @staticmethod
    def _append_breakpoint(
        values: list[float],
        value: float,
        low: float,
        high: float,
    ) -> None:
        if low + 1e-6 < value < high - 1e-6:
            values.append(float(value))

    def _append_terrain_pad_breakpoints(
        self,
        x_breaks: list[float],
        z_breaks: list[float],
        pad: TerrainFlattenPad,
        *,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
    ) -> None:
        x_values, z_values = self._terrain_pad_breakpoint_values(pad)
        for value in x_values:
            self._append_breakpoint(x_breaks, value, min_x, max_x)
        for value in z_values:
            self._append_breakpoint(z_breaks, value, min_z, max_z)

    @staticmethod
    def _terrain_pad_breakpoint_values(
        pad: TerrainFlattenPad,
    ) -> tuple[list[float], list[float]]:
        """Return the world-axis cuts needed to represent one flatten pad."""
        margin = max(0.0, float(pad.blend_margin))
        x_values: list[float] = []
        z_values: list[float] = []
        for fraction in (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0):
            offset = margin * fraction
            x_values.extend((pad.min_x - offset, pad.max_x + offset))
            z_values.extend((pad.min_z - offset, pad.max_z + offset))
        return x_values, z_values

    @staticmethod
    def _covered_region_breakpoint_values(
        region,
    ) -> tuple[list[float], list[float]]:
        values = TexturedGroundGridBuilder._region_values(region)
        if values is None:
            return [], []

        region_min_x, region_max_x, region_min_z, region_max_z = values
        x_values = [region_min_x, region_max_x]
        z_values = [region_min_z, region_max_z]

        for opening in region_light_openings(region):
            try:
                side = str(opening.get("side", "")).lower()
                width = float(opening.get("width", 48.0))
                depth = float(opening.get("depth", 64.0))
                side_fade = float(opening.get("side_fade", width * 0.25))
                center_x = float(
                    opening.get("center_x", (region_min_x + region_max_x) * 0.5)
                )
                center_z = float(
                    opening.get("center_z", (region_min_z + region_max_z) * 0.5)
                )
            except (TypeError, ValueError):
                continue

            half = width * 0.5
            if side in {"north", "south"}:
                x_values.extend(
                    (
                        center_x - half - side_fade,
                        center_x - half,
                        center_x + half,
                        center_x + half + side_fade,
                    )
                )
                edge_z = region_max_z if side == "north" else region_min_z
                step = -depth if side == "north" else depth
                z_values.extend(
                    (edge_z + step * 0.35, edge_z + step * 0.7, edge_z + step)
                )
            elif side in {"east", "west"}:
                z_values.extend(
                    (
                        center_z - half - side_fade,
                        center_z - half,
                        center_z + half,
                        center_z + half + side_fade,
                    )
                )
                edge_x = region_max_x if side == "east" else region_min_x
                step = -depth if side == "east" else depth
                x_values.extend(
                    (edge_x + step * 0.35, edge_x + step * 0.7, edge_x + step)
                )

        return x_values, z_values

    def _append_covered_region_breakpoints(
        self,
        x_breaks: list[float],
        z_breaks: list[float],
        region,
        *,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
    ) -> None:
        x_values, z_values = self._covered_region_breakpoint_values(region)
        for value in x_values:
            self._append_breakpoint(x_breaks, value, min_x, max_x)
        for value in z_values:
            self._append_breakpoint(z_breaks, value, min_z, max_z)

    def _append_ground_quad(
        self,
        rows: list[tuple[float, ...]],
        *,
        x0: float,
        x1: float,
        z0: float,
        z1: float,
        tile_min_x: float,
        tile_min_z: float,
        color_factor: float | None = None,
    ) -> None:
        if x1 <= x0 or z1 <= z0:
            return

        span = max(1e-6, float(self.tile_size))
        u0 = (x0 - tile_min_x) / span
        u1 = (x1 - tile_min_x) / span
        v0 = (z0 - tile_min_z) / span
        v1 = (z1 - tile_min_z) / span

        def vertex(
            x: float,
            z: float,
            u: float,
            v: float,
        ) -> tuple[float, float, float, float, float, float, float, float]:
            factor = (
                covered_region_factor_at(
                    x,
                    z,
                    covered_regions=self.covered_regions,
                )
                if color_factor is None
                else float(color_factor)
            )
            r = g = b = max(0.0, min(1.0, factor))
            return (x, 0.0, z, r, g, b, u, v)

        rows.extend(
            (
                vertex(x0, z0, u0, v0),
                vertex(x1, z0, u1, v0),
                vertex(x1, z1, u1, v1),
                vertex(x0, z0, u0, v0),
                vertex(x1, z1, u1, v1),
                vertex(x0, z1, u0, v1),
            )
        )

    def _build_ground_vertex_rows(
        self,
        *,
        terrain_pads: list[TerrainFlattenPad] | tuple[TerrainFlattenPad, ...] = (),
        apply_region_colors: bool = True,
    ) -> tuple[np.ndarray, list[tuple[int, int, int, float, float]]]:
        rows: list[tuple[float, ...]] = []
        corner_coords_list: list[tuple[int, int, int, float, float]] = []

        # Axis cuts must be shared across an entire row/column of terrain
        # tiles. Applying building-pad cuts only to nearby tiles creates
        # T-junctions: one side samples the curved heightmap at extra points
        # while its neighbor spans those points with one straight edge. The
        # unmatched edges open into visible sky-colored wedges. Precomputing
        # the cuts by axis keeps every shared edge identical and avoids the
        # old pads-per-tile O(count**2 * pads) work.
        x_breaks_by_grid: list[list[float]] = []
        z_breaks_by_grid: list[list[float]] = []
        for grid_index in range(self.count):
            center = grid_index * self.spacing
            x_breaks_by_grid.append([center - self.w, center + self.w])
            z_breaks_by_grid.append([center - self.d, center + self.d])

        terrain_x_values: list[float] = []
        terrain_z_values: list[float] = []
        for pad in terrain_pads:
            x_values, z_values = self._terrain_pad_breakpoint_values(pad)
            terrain_x_values.extend(x_values)
            terrain_z_values.extend(z_values)

        region_x_values: list[float] = []
        region_z_values: list[float] = []
        if apply_region_colors:
            for region in self.covered_regions:
                x_values, z_values = self._covered_region_breakpoint_values(region)
                region_x_values.extend(x_values)
                region_z_values.extend(z_values)

        for grid_index in range(self.count):
            center = grid_index * self.spacing
            min_axis = center - self.w
            max_axis = center + self.w
            for value in (*terrain_x_values, *region_x_values):
                self._append_breakpoint(
                    x_breaks_by_grid[grid_index], value, min_axis, max_axis
                )
            for value in (*terrain_z_values, *region_z_values):
                self._append_breakpoint(
                    z_breaks_by_grid[grid_index], value, min_axis, max_axis
                )
            x_breaks_by_grid[grid_index] = sorted(
                set(round(value, 6) for value in x_breaks_by_grid[grid_index])
            )
            z_breaks_by_grid[grid_index] = sorted(
                set(round(value, 6) for value in z_breaks_by_grid[grid_index])
            )

        for gx in range(self.count):
            for gz in range(self.count):
                tx = gx * self.spacing
                tz = gz * self.spacing
                min_x = tx - self.w
                max_x = tx + self.w
                min_z = tz - self.d
                max_z = tz + self.d
                corner_coords_list.extend(
                    [
                        (gx, gz, 0, min_x, min_z),
                        (gx, gz, 1, max_x, min_z),
                        (gx, gz, 2, max_x, max_z),
                        (gx, gz, 3, min_x, max_z),
                    ]
                )

                tile_regions = (
                    self._covered_regions_for_tile(
                        min_x,
                        max_x,
                        min_z,
                        max_z,
                    )
                    if apply_region_colors
                    else []
                )
                x_breaks = x_breaks_by_grid[gx]
                z_breaks = z_breaks_by_grid[gz]

                if not tile_regions and len(x_breaks) == 2 and len(z_breaks) == 2:
                    self._append_ground_quad(
                        rows,
                        x0=min_x,
                        x1=max_x,
                        z0=min_z,
                        z1=max_z,
                        tile_min_x=min_x,
                        tile_min_z=min_z,
                        color_factor=1.0,
                    )
                    continue

                for ix in range(len(x_breaks) - 1):
                    x0 = x_breaks[ix]
                    x1 = x_breaks[ix + 1]
                    for iz in range(len(z_breaks) - 1):
                        z0 = z_breaks[iz]
                        z1 = z_breaks[iz + 1]
                        self._append_ground_quad(
                            rows,
                            x0=x0,
                            x1=x1,
                            z0=z0,
                            z1=z1,
                            tile_min_x=min_x,
                            tile_min_z=min_z,
                            color_factor=1.0 if not apply_region_colors else None,
                        )

        if not rows:
            return np.zeros((0, 8), dtype=np.float32), corner_coords_list
        return np.array(rows, dtype=np.float32), corner_coords_list

    def build(self) -> BatchedMesh:
        tile_vertex_array = self._make_tile_vertex_array()
        heightmap_data = self._load_heightmap()
        terrain_pads = self._build_terrain_flatten_pads(heightmap_data)
        self.terrain_flatten_pads = terrain_pads

        # Pre-allocate final vertex array
        vertices_per_tile = len(tile_vertex_array)
        total_vertices = vertices_per_tile * self.count * self.count
        vertex_data = np.zeros((total_vertices, 8), dtype=np.float32)
        corner_heights = np.zeros((self.count, self.count, 4), dtype=np.float32)
        shader_lighting = uses_dynamic_textured_lighting(self.dynamic_lighting)

        if total_vertices == 0:
            empty = np.zeros((0, 8), dtype=np.float32)
            return BatchedMesh.from_vertex_data(
                empty,
                texture=self.texture,
                casts_sun_shadows=False,
                exposure_baseline=self.default_brightness,
                lighting_receiver=GROUND_LIGHTING_RECEIVER,
            )

        split_covered_regions = bool(self.covered_regions) and not shader_lighting
        split_terrain_pads = bool(terrain_pads)
        if split_covered_regions or split_terrain_pads:
            vertex_data, corner_coords_list = self._build_ground_vertex_rows(
                terrain_pads=terrain_pads,
                apply_region_colors=not shader_lighting,
            )
        else:
            vertex_idx = 0
            corner_coords_list = []
            for gx in range(self.count):
                for gz in range(self.count):
                    tx = gx * self.spacing
                    tz = gz * self.spacing
                    vertex_data[vertex_idx : vertex_idx + vertices_per_tile] = (
                        tile_vertex_array
                    )
                    vertex_data[vertex_idx : vertex_idx + vertices_per_tile, 0] += tx
                    vertex_data[vertex_idx : vertex_idx + vertices_per_tile, 2] += tz
                    corner_coords_list.extend(
                        [
                            (gx, gz, 0, tx - self.w, tz - self.d),
                            (gx, gz, 1, tx + self.w, tz - self.d),
                            (gx, gz, 2, tx + self.w, tz + self.d),
                            (gx, gz, 3, tx - self.w, tz + self.d),
                        ]
                    )
                    vertex_idx += vertices_per_tile

        # Batch process all vertex heights
        vertex_world_coords = vertex_data[:, [0, 2]]  # Extract x, z coordinates
        vertex_heights = self._sample_heights_vectorized(
            vertex_world_coords,
            heightmap_data,
            terrain_pads,
        )
        vertex_data[:, 1] = vertex_heights  # Update y coordinates
        terrain_normals = (
            self._sample_smooth_normals_vectorized(
                vertex_world_coords,
                heightmap_data,
                terrain_pads,
            )
            if terrain_pads
            else None
        )

        if shader_lighting:
            vertex_data = with_textured_normals(
                vertex_data,
                normals=terrain_normals,
                prefer_upward_normals=True,
            )
        else:
            if split_covered_regions:
                receiver_factors = np.clip(
                    np.max(vertex_data[:, 3:6], axis=1),
                    0.0,
                    1.0,
                )
                indoor_receivers = receiver_factors < 0.995
            else:
                receiver_factors = None
                indoor_receivers = covered_region_mask(
                    vertex_data,
                    covered_regions=self.covered_regions,
                )
            apply_directional_sunlight(
                vertex_data,
                lighting=self.lighting,
                sun_direction=self.sun_direction,
                normals=terrain_normals,
                prefer_upward_normals=True,
            )
            if not split_covered_regions:
                apply_covered_regions(
                    vertex_data,
                    covered_regions=self.covered_regions,
                )
                receiver_factors = np.clip(
                    np.max(vertex_data[:, 3:6], axis=1),
                    0.0,
                    1.0,
                )
            apply_brightness_modifiers(
                vertex_data,
                modifiers=self.brightness_modifiers,
                default_brightness=self.default_brightness,
                receiver_mask=indoor_receivers,
                receiver_factors=receiver_factors,
                surface_floor_mask=np.ones(len(vertex_data), dtype=bool),
            )

        # Batch process all corner heights (unchanged)
        corner_world_coords = np.array(
            [[coord[3], coord[4]] for coord in corner_coords_list]
        )
        corner_heights_flat = self._sample_base_heights_vectorized(
            corner_world_coords,
            heightmap_data,
        )
        # Assign corner heights to the corner_heights array
        for i, (gx, gz, corner_idx, _, _) in enumerate(corner_coords_list):
            corner_heights[gx, gz, corner_idx] = corner_heights_flat[i]

        mesh = BatchedMesh.from_vertex_data(
            vertex_data,
            texture=self.texture,
            casts_sun_shadows=False,
            exposure_baseline=self.default_brightness,
            lighting_receiver=GROUND_LIGHTING_RECEIVER,
        )
        mesh.height_sampler = GroundHeightSampler(
            count=self.count,
            spacing=self.spacing,
            half=self.w,
            heights=corner_heights,
            height_adjustments=[pad.sampler_tuple() for pad in terrain_pads],
            surface_vertices=vertex_data[:, :3],
        )
        # Store the template tile vertex array and grid params so the mesh can
        # be updated at runtime when heights change.
        mesh._tile_vertex_array = tile_vertex_array
        mesh._count = int(self.count)
        mesh._spacing = float(self.spacing)
        mesh._half = float(self.w)
        mesh._terrain_flatten_pads = [pad.sampler_tuple() for pad in terrain_pads]
        return mesh
