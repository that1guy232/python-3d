from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from pygame.math import Vector3  # noqa: E402
from OpenGL.GL import (  # noqa: E402
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGB,
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glClear,
    glClearColor,
    glColor4f,
    glDeleteTextures,
    glDisable,
    glEnd,
    glFinish,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glNormal3f,
    glReadPixels,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex3f,
    glViewport,
)

from engine.core.gl_state import use_fixed_pipeline  # noqa: E402
from engine.core.mesh import BatchedMesh  # noqa: E402
from engine.rendering.lighting_adapter import RenderLightingAdapter  # noqa: E402
from engine.rendering.lighting_state import (  # noqa: E402
    DirectionalLightSnapshot,
    LightingSnapshot,
)
from engine.rendering.packet_shader import (  # noqa: E402
    get_packet_texture_lighting_shader,
    reset_packet_texture_lighting_shader,
)
from engine.rendering.point_shadow import (  # noqa: E402
    PointShadowCasterShader,
    PointShadowMap,
    point_light_face_matrices,
)
from engine.render_style_state import (  # noqa: E402
    update_render_fog_state,
    update_render_shine_state,
    update_render_vibrance_state,
)
from game.world.lighting_receivers import GROUND_LIGHTING_RECEIVER  # noqa: E402
from game.world.objects.torch import Torch  # noqa: E402


AUTHORED_TORCH_LIGHT = Torch.point_light_for_building_spec(
    {
        "position": Vector3(0.0, 0.0, 1.0),
        "base_y": 0.0,
        "width": 1.0,
        "depth": 1.0,
        "doorway_side": "south",
    },
    {
        "side": "north",
        "mount_height": 0.0,
        "light_color": (1.0, 1.0, 1.0),
        "light_intensity": 1.5,
        "light_range": 4.0,
    },
    light_id="fixture:torch",
)
LIGHT_POSITION = AUTHORED_TORCH_LIGHT.position
LIGHT_RANGE = AUTHORED_TORCH_LIGHT.range


def _packet():
    directional = DirectionalLightSnapshot(
        sun_position=(0.0, 0.0, 2.0),
        sun_target=(0.0, 0.0, 0.0),
        sun_direction=(0.0, 0.0, -1.0),
        light_direction=(0.0, 0.0, 1.0),
        ambient=0.0,
        diffuse=0.0,
        max_factor=1.0,
        tint=(1.0, 1.0, 1.0),
    )
    snapshot = LightingSnapshot(
        revision=1,
        base_brightness=1.0,
        sky_color=(0.0, 0.0, 0.0, 1.0),
        directional=directional,
        point_lights=(AUTHORED_TORCH_LIGHT,),
    )
    return RenderLightingAdapter().packet_for(snapshot, GROUND_LIGHTING_RECEIVER)


class PointShadowMathTests(unittest.TestCase):
    def test_each_cube_face_maps_its_forward_point_inside_clip_volume(self) -> None:
        matrices = point_light_face_matrices(
            LIGHT_POSITION,
            near=0.1,
            far=LIGHT_RANGE,
        )
        directions = (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
        for values, direction in zip(matrices, directions):
            matrix = np.asarray(values, dtype=np.float32).reshape(4, 4)
            point = np.asarray(
                (*(
                    LIGHT_POSITION[index] + direction[index]
                    for index in range(3)
                ), 1.0),
                dtype=np.float32,
            )
            clip = matrix @ point
            ndc = clip[:3] / clip[3]
            self.assertTrue(np.all(np.abs(ndc) <= 1.0), ndc)


class PointShadowFramebufferTests(unittest.TestCase):
    def setUp(self) -> None:
        pygame.init()
        try:
            pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.HIDDEN)
        except Exception as exc:
            pygame.quit()
            self.skipTest(f"hidden OpenGL context unavailable: {exc}")
        glViewport(0, 0, 64, 64)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        update_render_fog_state(enabled=False)
        update_render_vibrance_state(1.0)
        update_render_shine_state(enabled=False)
        reset_packet_texture_lighting_shader()
        self.shader = get_packet_texture_lighting_shader()
        if self.shader is None:
            self.skipTest("packet shader unavailable")
        self.shadow = PointShadowMap.create(128)
        self.caster = PointShadowCasterShader.create()
        self.matrices = point_light_face_matrices(
            LIGHT_POSITION,
            near=0.1,
            far=LIGHT_RANGE,
        )
        self.white = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.white)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            1,
            1,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            bytes((255, 255, 255, 255)),
        )
        wall_data = np.asarray(
            (
                (-0.35, -0.35, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0),
                (0.35, -0.35, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0),
                (0.35, 0.35, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0),
                (-0.35, 0.35, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        self.wall = BatchedMesh.from_vertex_data(
            wall_data,
            texture=self.white,
            shine_enabled=False,
            draw_mode=GL_QUADS,
        )

    def tearDown(self) -> None:
        try:
            if getattr(self, "shader", None) is not None:
                self.shader.set_point_shadows(())
            if getattr(self, "caster", None) is not None:
                self.caster.dispose()
            if getattr(self, "shadow", None) is not None:
                self.shadow.dispose()
            if getattr(self, "wall", None) is not None:
                self.wall.dispose()
            if getattr(self, "white", 0):
                glDeleteTextures(1, [self.white])
            reset_packet_texture_lighting_shader()
        finally:
            pygame.display.quit()
            pygame.quit()

    @staticmethod
    def _quad(x0: float, x1: float, y0: float, y1: float, z: float) -> None:
        glBegin(GL_QUADS)
        for x, y, u, v in (
            (x0, y0, 0.0, 0.0),
            (x1, y0, 1.0, 0.0),
            (x1, y1, 1.0, 1.0),
            (x0, y1, 0.0, 1.0),
        ):
            glTexCoord2f(u, v)
            glVertex3f(x, y, z)
        glEnd()

    def _render_wall(self) -> None:
        for face_index, matrix in enumerate(self.matrices):
            with self.shadow.render_face(face_index):
                self.wall.draw_point_shadow(
                    self.caster,
                    matrix,
                    LIGHT_POSITION,
                    LIGHT_RANGE,
                )

    def _draw_receiver(self) -> None:
        glViewport(0, 0, 64, 64)
        glDisable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glBindTexture(GL_TEXTURE_2D, self.white)
        self.shader.set_point_shadows(
            (
                self.shadow.binding(
                    AUTHORED_TORCH_LIGHT.light_id,
                    LIGHT_POSITION,
                    LIGHT_RANGE,
                    bias=0.02,
                ),
            )
        )
        self.shader.bind(_packet())
        glBegin(GL_QUADS)
        for x, y, u, v in (
            (-1.0, -1.0, 0.0, 0.0),
            (1.0, -1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0, 1.0),
            (-1.0, 1.0, 0.0, 1.0),
        ):
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glNormal3f(0.0, 0.0, 1.0)
            glTexCoord2f(u, v)
            glVertex3f(x, y, 0.0)
        glEnd()
        use_fixed_pipeline()
        glFinish()

    @staticmethod
    def _pixel(x: int, y: int) -> tuple[int, int, int]:
        value = glReadPixels(x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
        raw = value.tobytes() if hasattr(value, "tobytes") else bytes(value)
        return int(raw[0]), int(raw[1]), int(raw[2])

    def test_opaque_wall_blocks_authored_torch_without_receiver_hacks(self) -> None:
        self._render_wall()
        self._draw_receiver()
        blocked = self._pixel(32, 32)
        visible = self._pixel(8, 32)
        self.assertTrue(all(channel <= 10 for channel in blocked), blocked)
        self.assertTrue(all(channel >= 60 for channel in visible), visible)
