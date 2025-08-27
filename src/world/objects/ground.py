"""Ground grid builder extracted from renderer.py."""

from __future__ import annotations
import os
import numpy as np
import pygame
from pygame.math import Vector3
from world.ground_tile import GroundTile
from textures.resoucepath import TEXTURES_PATH
from textures.texture_utils import get_texture_size
from OpenGL.GL import (
    glGenBuffers,
    glBindBuffer,
    glBufferData,
    GL_ARRAY_BUFFER,
    GL_STATIC_DRAW,
)
from core.mesh import BatchedMesh, GroundHeightSampler

class TexturedGroundGridBuilder:
    def __init__(
        self,
        count: int,
        tile_size: float,
        gap: float,
        texture,
        brightness_modifiers: list[callable] = None,
        default_brightness: float = 1.0,
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

    def build(self) -> BatchedMesh:
        tile_vertex_array = self._make_tile_vertex_array()
        heightmap_data = self._load_heightmap()
        
        # Pre-allocate final vertex array
        vertices_per_tile = len(tile_vertex_array)
        total_vertices = vertices_per_tile * self.count * self.count
        vertex_data = np.zeros((total_vertices, 8), dtype=np.float32)
        corner_heights = np.zeros((self.count, self.count, 4), dtype=np.float32)
        
        if total_vertices == 0:
            empty = np.zeros((0, 8), dtype=np.float32)
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, empty.nbytes, empty, GL_STATIC_DRAW)
            return BatchedMesh(vbo_vertices=vbo, vertex_count=0, texture=self.texture)
        
        vertex_idx = 0
        # Pre-calculate all grid positions
        grid_positions = []
        corner_coords_list = []
        tile_start_indices = []
        for gx in range(self.count):
            for gz in range(self.count):
                tx = gx * self.spacing
                tz = gz * self.spacing
                grid_positions.append((gx, gz, tx, tz))
                # Store tile start index for vertex processing
                tile_start_indices.append(vertex_idx)
                # Copy tile data directly into final array
                vertex_data[vertex_idx:vertex_idx + vertices_per_tile] = tile_vertex_array
                # Apply translation
                vertex_data[vertex_idx:vertex_idx + vertices_per_tile, 0] += tx
                vertex_data[vertex_idx:vertex_idx + vertices_per_tile, 2] += tz
                # Prepare corner coordinates for batch processing
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
        
        if self.brightness_modifiers:
            N = len(vertex_data)
            # Initialize to default_brightness so modifiers multiply that
            # baseline. This prevents a visible seam when default_brightness != 1.0.
            brightness_factor = np.full(N, self.default_brightness, dtype=np.float32)
            is_modified = np.zeros(N, dtype=bool)  # Track which vertices have modifiers
            
            # 1. Identify vertices affected by ANY modifier
            for modifier in self.brightness_modifiers:
                try:
                    position, radius, brightness_value, fall_off = modifier
                    center_x = position.x
                    center_z = position.z
                    # Calculate distances to all vertices at once
                    dx = vertex_world_coords[:, 0] - center_x
                    dz = vertex_world_coords[:, 1] - center_z
                    distances = np.sqrt(dx*dx + dz*dz)
                    # Mark vertices within radius
                    within_radius = distances <= radius
                    is_modified |= within_radius
                except (ValueError, AttributeError, IndexError) as e:
                    print(f"Warning: Invalid modifier {modifier}, skipping. Error: {e}")
            
            brightness_factor[~is_modified] = self.default_brightness
            
            # 3. Apply modifiers multiplicatively to modified vertices
            for modifier in self.brightness_modifiers:
                try:
                    position, radius, brightness_value, fall_off = modifier  # Now using all 4 values
                    center_x = position.x
                    center_z = position.z
                    dx = vertex_world_coords[:, 0] - center_x
                    dz = vertex_world_coords[:, 1] - center_z
                    distances = np.sqrt(dx*dx + dz*dz)
                    within_radius = distances <= radius
                    # Normalized distance in [0,1] (0=center, 1=radius). Avoid divide-by-zero.
                    norm = distances / np.maximum(radius, 1e-12)
                    norm = np.clip(norm, 0.0, 1.0)
                    # Interpret fall_off as exponent: 1.0 = linear, >1 = steeper, <1 = smoother
                    attenuation = (1.0 - norm) ** np.maximum(fall_off, 0.0)
                    
                    if self.default_brightness == 0:
                        rel = brightness_value
                    else:
                        rel = brightness_value / self.default_brightness
                    modifier_effect = 1.0 + (rel - 1.0) * attenuation
                    
                    # Apply only to vertices inside radius
                    brightness_factor[within_radius] *= modifier_effect[within_radius]
                except (ValueError, AttributeError, IndexError) as e:
                    print(f"Warning: Invalid modifier {modifier}, skipping. Error: {e}")
                    continue
            
            vertex_data[:, 3:6] = vertex_data[:, 3:6] * brightness_factor[:, np.newaxis]
        else:
            # No modifiers: apply default to ALL vertices
            vertex_data[:, 3:6] *= self.default_brightness
        
        ###########################################################
        # END NEW BRIGHTNESS LOGIC #
        ###########################################################
        
        # Batch process all corner heights (unchanged)
        corner_world_coords = np.array([[coord[3], coord[4]] for coord in corner_coords_list])
        corner_heights_flat = self._sample_heights_vectorized(corner_world_coords, heightmap_data)
        # Assign corner heights to the corner_heights array
        for i, (gx, gz, corner_idx, _, _) in enumerate(corner_coords_list):
            corner_heights[gx, gz, corner_idx] = corner_heights_flat[i]
        
        # Create VBO with the modified vertex data
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        
        mesh = BatchedMesh(
            vbo_vertices=vbo,
            vertex_count=vertex_data.shape[0],
            texture=self.texture
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