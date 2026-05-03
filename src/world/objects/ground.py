"""Ground grid builder extracted from renderer.py."""

from __future__ import annotations
import os
import numpy as np
import pygame
from pygame.math import Vector3
from world.objects.ground_tile import GroundTile
from textures.resource_path import TEXTURES_PATH
from core.mesh import BatchedMesh, GroundHeightSampler
from core.compat_shader import texture_color_exposure_shader_available
from engine.rendering.lighting import (
    apply_brightness_modifiers,
    apply_covered_regions,
    apply_directional_sunlight,
    covered_region_factor_at,
    covered_region_mask,
    with_textured_normals,
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
        self.sun_direction = sun_direction
        self.covered_regions = covered_regions or []
        
        tile = GroundTile(
            position=Vector3(0, 0, 0),
            width=self.w,
            height=self.h,
            depth=self.d
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
                self.uv_dummy, self.uv_dummy, self.uv_dummy,
                self.uv_top,
                self.uv_dummy, self.uv_dummy,
            ]
        
        self.heightmap_path = os.path.join(TEXTURES_PATH, "heightmap.png")
        self.heightmap_amp = 80.0

    def _make_tile_vertex_array(self) -> np.ndarray:
        tile_vertices_local = []
        for face_idx, (face, color) in enumerate(zip(self.faces, self.face_colors)):
            a, b, c, d_idx = face
            face_uv = self.face_uvs[face_idx]
            tri_indices = [a, b, c, a, c, d_idx]
            tri_uvs = [
                face_uv[0], face_uv[1], face_uv[2],
                face_uv[0], face_uv[2], face_uv[3],
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

    def _sample_heights_vectorized(self, world_coords, heightmap_data):
        """Sample heights for multiple world coordinates at once"""
        heightmap_arr, hm_w, hm_h, world_min_x, world_max_x, world_min_z, world_max_z = heightmap_data
        if heightmap_arr is None:
            heights = np.full(len(world_coords), self.h, dtype=np.float32)
        else:
            wx = world_coords[:, 0]
            wz = world_coords[:, 1]
            # Vectorized UV calculation
            ux = (wx - world_min_x) / max(1e-12, (world_max_x - world_min_x))
            uz = (wz - world_min_z) / max(1e-12, (world_max_z - world_min_z))
            ux = np.clip(ux, 0.0, 1.0)
            uz = np.clip(uz, 0.0, 1.0)
            px = (ux * (hm_w - 1)).astype(int)
            py = (uz * (hm_h - 1)).astype(int)
            # Sample heights
            rgb_values = heightmap_arr[px, py]
            lum = rgb_values.mean(axis=1)
            heights = self.h + (lum - 0.5) * 2.0 * self.heightmap_amp
        return heights.astype(np.float32)

    @staticmethod
    def _region_values(region) -> tuple[float, float, float, float] | None:
        try:
            if isinstance(region, dict):
                min_x = float(region["min_x"])
                max_x = float(region["max_x"])
                min_z = float(region["min_z"])
                max_z = float(region["max_z"])
            else:
                min_x = float(region[0])
                max_x = float(region[1])
                min_z = float(region[2])
                max_z = float(region[3])
        except (KeyError, IndexError, TypeError, ValueError):
            return None

        if max_x < min_x:
            min_x, max_x = max_x, min_x
        if max_z < min_z:
            min_z, max_z = max_z, min_z
        return min_x, max_x, min_z, max_z

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
    ) -> tuple[np.ndarray, list[tuple[int, int, int, float, float]]]:
        rows: list[tuple[float, ...]] = []
        corner_coords_list: list[tuple[int, int, int, float, float]] = []

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

                tile_regions = self._covered_regions_for_tile(
                    min_x,
                    max_x,
                    min_z,
                    max_z,
                )
                if not tile_regions:
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

                x_breaks = [min_x, max_x]
                z_breaks = [min_z, max_z]
                doorway_breaks_x: list[float] = []
                doorway_breaks_z: list[float] = []
                for (
                    region_min_x,
                    region_max_x,
                    region_min_z,
                    region_max_z,
                ) in tile_regions:
                    self._append_breakpoint(x_breaks, region_min_x, min_x, max_x)
                    self._append_breakpoint(x_breaks, region_max_x, min_x, max_x)
                    self._append_breakpoint(z_breaks, region_min_z, min_z, max_z)
                    self._append_breakpoint(z_breaks, region_max_z, min_z, max_z)

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
                    doorway = region.get("doorway") if isinstance(region, dict) else None
                    if not isinstance(doorway, dict):
                        continue
                    try:
                        side = str(doorway.get("side", "")).lower()
                        width = float(doorway.get("width", 48.0))
                        depth = float(doorway.get("depth", 64.0))
                        side_fade = float(doorway.get("side_fade", width * 0.25))
                        center_x = float(doorway.get("center_x", (region_min_x + region_max_x) * 0.5))
                        center_z = float(doorway.get("center_z", (region_min_z + region_max_z) * 0.5))
                    except (TypeError, ValueError):
                        continue

                    half = width * 0.5
                    if side in {"north", "south"}:
                        doorway_breaks_x.extend(
                            (
                                center_x - half - side_fade,
                                center_x - half,
                                center_x + half,
                                center_x + half + side_fade,
                            )
                        )
                        edge_z = region_max_z if side == "north" else region_min_z
                        step = -depth if side == "north" else depth
                        doorway_breaks_z.extend(
                            (
                                edge_z + step * 0.35,
                                edge_z + step * 0.7,
                                edge_z + step,
                            )
                        )
                    elif side in {"east", "west"}:
                        doorway_breaks_z.extend(
                            (
                                center_z - half - side_fade,
                                center_z - half,
                                center_z + half,
                                center_z + half + side_fade,
                            )
                        )
                        edge_x = region_max_x if side == "east" else region_min_x
                        step = -depth if side == "east" else depth
                        doorway_breaks_x.extend(
                            (
                                edge_x + step * 0.35,
                                edge_x + step * 0.7,
                                edge_x + step,
                            )
                        )

                for value in doorway_breaks_x:
                    self._append_breakpoint(x_breaks, value, min_x, max_x)
                for value in doorway_breaks_z:
                    self._append_breakpoint(z_breaks, value, min_z, max_z)

                x_breaks = sorted(set(round(value, 6) for value in x_breaks))
                z_breaks = sorted(set(round(value, 6) for value in z_breaks))
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
                        )

        if not rows:
            return np.zeros((0, 8), dtype=np.float32), corner_coords_list
        return np.array(rows, dtype=np.float32), corner_coords_list

    def build(self) -> BatchedMesh:
        tile_vertex_array = self._make_tile_vertex_array()
        heightmap_data = self._load_heightmap()
        
        # Pre-allocate final vertex array
        vertices_per_tile = len(tile_vertex_array)
        total_vertices = vertices_per_tile * self.count * self.count
        vertex_data = np.zeros((total_vertices, 8), dtype=np.float32)
        corner_heights = np.zeros((self.count, self.count, 4), dtype=np.float32)
        shader_lighting = texture_color_exposure_shader_available()
        
        if total_vertices == 0:
            empty = np.zeros((0, 8), dtype=np.float32)
            return BatchedMesh.from_vertex_data(
                empty,
                texture=self.texture,
                exposure_baseline=self.default_brightness,
            )
        
        split_covered_regions = bool(self.covered_regions)
        if split_covered_regions:
            vertex_data, corner_coords_list = self._build_ground_vertex_rows()
        else:
            vertex_idx = 0
            corner_coords_list = []
            for gx in range(self.count):
                for gz in range(self.count):
                    tx = gx * self.spacing
                    tz = gz * self.spacing
                    vertex_data[vertex_idx:vertex_idx + vertices_per_tile] = tile_vertex_array
                    vertex_data[vertex_idx:vertex_idx + vertices_per_tile, 0] += tx
                    vertex_data[vertex_idx:vertex_idx + vertices_per_tile, 2] += tz
                    corner_coords_list.extend([
                        (gx, gz, 0, tx - self.w, tz - self.d),
                        (gx, gz, 1, tx + self.w, tz - self.d),
                        (gx, gz, 2, tx + self.w, tz + self.d),
                        (gx, gz, 3, tx - self.w, tz + self.d)
                    ])
                    vertex_idx += vertices_per_tile
        
        # Batch process all vertex heights
        vertex_world_coords = vertex_data[:, [0, 2]]  # Extract x, z coordinates
        vertex_heights = self._sample_heights_vectorized(vertex_world_coords, heightmap_data)
        vertex_data[:, 1] = vertex_heights  # Update y coordinates
        
        if shader_lighting:
            vertex_data = with_textured_normals(
                vertex_data,
                prefer_upward_normals=True,
            )
            if not split_covered_regions:
                apply_covered_regions(
                    vertex_data,
                    covered_regions=self.covered_regions,
                )
        else:
            if split_covered_regions:
                indoor_receivers = np.max(vertex_data[:, 3:6], axis=1) < 0.995
            else:
                indoor_receivers = covered_region_mask(
                    vertex_data,
                    covered_regions=self.covered_regions,
                )
            apply_directional_sunlight(
                vertex_data,
                lighting=self.lighting,
                sun_direction=self.sun_direction,
                prefer_upward_normals=True,
            )
            if not split_covered_regions:
                apply_covered_regions(
                    vertex_data,
                    covered_regions=self.covered_regions,
                )
            apply_brightness_modifiers(
                vertex_data,
                modifiers=self.brightness_modifiers,
                default_brightness=self.default_brightness,
                receiver_mask=indoor_receivers,
                surface_floor_mask=np.ones(len(vertex_data), dtype=bool),
            )
        
        # Batch process all corner heights (unchanged)
        corner_world_coords = np.array([[coord[3], coord[4]] for coord in corner_coords_list])
        corner_heights_flat = self._sample_heights_vectorized(corner_world_coords, heightmap_data)
        # Assign corner heights to the corner_heights array
        for i, (gx, gz, corner_idx, _, _) in enumerate(corner_coords_list):
            corner_heights[gx, gz, corner_idx] = corner_heights_flat[i]
        
        mesh = BatchedMesh.from_vertex_data(
            vertex_data,
            texture=self.texture,
            exposure_baseline=self.default_brightness,
        )
        mesh.height_sampler = GroundHeightSampler(
            count=self.count,
            spacing=self.spacing,
            half=self.w,
            heights=corner_heights
        )
        # Store the template tile vertex array and grid params so the mesh can
        # be updated at runtime when heights change.
        mesh._tile_vertex_array = tile_vertex_array
        mesh._count = int(self.count)
        mesh._spacing = float(self.spacing)
        mesh._half = float(self.w)
        return mesh
