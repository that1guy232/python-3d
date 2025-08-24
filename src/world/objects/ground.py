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
        height_modifiers: list[callable] | None = None,
    ):
        self.count = count
        self.tile_size = tile_size
        self.gap = gap
        self.texture = texture
        self.height_modifiers = height_modifiers
        self.w = tile_size / 2.0
        self.h = 5
        self.d = tile_size / 2.0
        self.spacing = tile_size + gap

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
                tile_vertices_local.append((v.x, v.y, v.z, *color, uv[0], uv[1]))
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

    def _sample_height(
        self,
        wx: float,
        wz: float,
        heightmap_arr,
        hm_w,
        hm_h,
        world_min_x,
        world_max_x,
        world_min_z,
        world_max_z,
    ) -> float:
        if heightmap_arr is None:
            return self.h
        ux = (wx - world_min_x) / max(1e-12, (world_max_x - world_min_x))
        uz = (wz - world_min_z) / max(1e-12, (world_max_z - world_min_z))
        ux = min(1.0, max(0.0, ux))
        uz = min(1.0, max(0.0, uz))
        px = int(ux * (hm_w - 1))
        py = int(uz * (hm_h - 1))
        rgb = heightmap_arr[px, py]
        lum = float(rgb[0] + rgb[1] + rgb[2]) / 3.0
        base_y = self.h + (lum - 0.5) * 2.0 * self.heightmap_amp
        if self.height_modifiers:
            y = float(base_y)
            for mod in self.height_modifiers:
                try:
                    y = float(mod(wx, wz, y))
                except Exception:
                    pass
            return y
        return float(base_y)

    def build(self) -> BatchedMesh:
        tile_vertex_array = self._make_tile_vertex_array()

        (
            heightmap_arr,
            hm_w,
            hm_h,
            world_min_x,
            world_max_x,
            world_min_z,
            world_max_z,
        ) = self._load_heightmap()

        vertices = []
        corner_heights = np.zeros((self.count, self.count, 4), dtype=np.float32)
        for gx in range(self.count):
            for gz in range(self.count):
                tx = gx * self.spacing
                tz = gz * self.spacing
                translated = tile_vertex_array.copy()
                translated[:, 0] += tx
                translated[:, 2] += tz

                for i in range(len(translated)):
                    x, _, z = translated[i, 0], translated[i, 1], translated[i, 2]
                    translated[i, 1] = self._sample_height(
                        x,
                        z,
                        heightmap_arr,
                        hm_w,
                        hm_h,
                        world_min_x,
                        world_max_x,
                        world_min_z,
                        world_max_z,
                    )

                vertices.append(translated)

                a_y = self._sample_height(
                    tx - self.w,
                    tz - self.d,
                    heightmap_arr,
                    hm_w,
                    hm_h,
                    world_min_x,
                    world_max_x,
                    world_min_z,
                    world_max_z,
                )
                b_y = self._sample_height(
                    tx + self.w,
                    tz - self.d,
                    heightmap_arr,
                    hm_w,
                    hm_h,
                    world_min_x,
                    world_max_x,
                    world_min_z,
                    world_max_z,
                )
                c_y = self._sample_height(
                    tx + self.w,
                    tz + self.d,
                    heightmap_arr,
                    hm_w,
                    hm_h,
                    world_min_x,
                    world_max_x,
                    world_min_z,
                    world_max_z,
                )
                d_y = self._sample_height(
                    tx - self.w,
                    tz + self.d,
                    heightmap_arr,
                    hm_w,
                    hm_h,
                    world_min_x,
                    world_max_x,
                    world_min_z,
                    world_max_z,
                )
                corner_heights[gx, gz, 0] = a_y
                corner_heights[gx, gz, 1] = b_y
                corner_heights[gx, gz, 2] = c_y
                corner_heights[gx, gz, 3] = d_y

        if not vertices:
            empty = np.zeros((0, 8), dtype=np.float32)
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, empty.nbytes, empty, GL_STATIC_DRAW)
            return BatchedMesh(vbo_vertices=vbo, vertex_count=0, texture=self.texture)

        vertex_data = np.vstack(vertices)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        mesh = BatchedMesh(
            vbo_vertices=vbo, vertex_count=vertex_data.shape[0], texture=self.texture
        )

        mesh.height_sampler = GroundHeightSampler(
            count=self.count, spacing=self.spacing, half=self.w, heights=corner_heights
        )

        # Store the template tile vertex array and grid params so the mesh can
        # be updated at runtime when heights change.
        mesh._tile_vertex_array = tile_vertex_array
        mesh._count = int(self.count)
        mesh._spacing = float(self.spacing)
        mesh._half = float(self.w)

        return mesh
