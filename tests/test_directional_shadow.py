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
from engine.rendering.directional_shadow import (  # noqa: E402
    DirectionalShadowMap,
    ShadowCasterShader,
    directional_light_matrix,
    directional_shadow_bias,
)
from engine.rendering.lighting_adapter import RenderLightingAdapter  # noqa: E402
from engine.rendering.lighting_state import (  # noqa: E402
    DirectionalLightSnapshot,
    LightingSnapshot,
)
from engine.rendering.packet_shader import (  # noqa: E402
    get_packet_texture_lighting_shader,
    reset_packet_texture_lighting_shader,
)
from engine.render_style_state import (  # noqa: E402
    update_render_fog_state,
    update_render_shine_state,
    update_render_vibrance_state,
)
from game.world.lighting_receivers import GROUND_LIGHTING_RECEIVER  # noqa: E402
from game.world.objects.door import Door, DoorRenderBatch  # noqa: E402


def _packet(*, ambient: float = 0.2, diffuse: float = 0.8):
    directional = DirectionalLightSnapshot(
        sun_position=(0.0, 0.0, 2.0),
        sun_target=(0.0, 0.0, 0.0),
        sun_direction=(0.0, 0.0, -1.0),
        light_direction=(0.0, 0.0, 1.0),
        ambient=ambient,
        diffuse=diffuse,
        max_factor=1.0,
        tint=(1.0, 1.0, 1.0),
    )
    snapshot = LightingSnapshot(
        revision=1,
        base_brightness=1.0,
        sky_color=(0.0, 0.0, 0.0, 1.0),
        directional=directional,
    )
    return RenderLightingAdapter().packet_for(snapshot, GROUND_LIGHTING_RECEIVER)


class DirectionalShadowMathTests(unittest.TestCase):
    def test_directional_matrix_maps_center_inside_clip_volume(self) -> None:
        values = directional_light_matrix(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            extent=2.0,
            near=0.1,
            far=4.0,
        )
        matrix = np.asarray(values, dtype=np.float32).reshape(4, 4)
        clip = matrix @ np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
        ndc = clip[:3] / clip[3]
        self.assertTrue(np.all(np.abs(ndc) <= 1.0), ndc)

    def test_depth_bias_is_world_scaled_for_large_shadow_volumes(self) -> None:
        self.assertAlmostEqual(
            directional_shadow_bias(near=1.0, far=4000.0),
            0.2 / 3999.0,
        )


class DirectionalShadowFramebufferTests(unittest.TestCase):
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
        self.shadow = DirectionalShadowMap.create(128)
        self.caster = ShadowCasterShader.create()
        self.matrix = directional_light_matrix(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            extent=1.0,
            near=0.1,
            far=4.0,
        )
        self.textures: list[int] = []
        self.white = self._texture(1, 1, bytes((255, 255, 255, 255)))

    def tearDown(self) -> None:
        try:
            if getattr(self, "shader", None) is not None:
                self.shader.set_directional_shadow(None)
            if getattr(self, "caster", None) is not None:
                self.caster.dispose()
            if getattr(self, "shadow", None) is not None:
                self.shadow.dispose()
            if getattr(self, "textures", None):
                glDeleteTextures(len(self.textures), self.textures)
            reset_packet_texture_lighting_shader()
        finally:
            pygame.display.quit()
            pygame.quit()

    def _texture(self, width: int, height: int, pixels: bytes) -> int:
        texture = int(glGenTextures(1))
        self.textures.append(texture)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            width,
            height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pixels,
        )
        return texture

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

    def _render_shadow(self, draws) -> None:
        with self.shadow.render_depth():
            for draw in draws:
                texture, cutout, bounds = draw
                self.caster.bind(
                    self.matrix,
                    texture=texture,
                    alpha_cutout=cutout,
                    alpha_cutoff=0.5,
                )
                self._quad(*bounds, 0.5)

    def _render_mesh_shadows(self, meshes) -> None:
        with self.shadow.render_depth():
            for mesh in meshes:
                mesh.draw_shadow(self.caster, self.matrix)

    def _draw_receiver(
        self,
        *,
        bias: float = 0.0005,
        packet=None,
    ) -> None:
        glViewport(0, 0, 64, 64)
        glDisable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glBindTexture(GL_TEXTURE_2D, self.white)
        self.shader.set_directional_shadow(self.shadow.binding(self.matrix, bias=bias))
        self.shader.bind(packet or _packet())
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

    def test_opaque_caster_blocks_direct_sun(self) -> None:
        self._render_shadow(((self.white, False, (-0.35, 0.35, -0.35, 0.35)),))
        self._draw_receiver()
        shadowed = self._pixel(32, 32)
        lit = self._pixel(8, 8)
        self.assertTrue(all(channel <= 70 for channel in shadowed), shadowed)
        self.assertTrue(all(channel >= 240 for channel in lit), lit)

    def test_world_scaled_bias_keeps_near_contact_shadow_attached(self) -> None:
        self.matrix = directional_light_matrix(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            extent=1.0,
            near=1.0,
            far=4000.0,
        )
        self._render_shadow(((self.white, False, (-0.35, 0.35, -0.35, 0.35)),))
        self._draw_receiver(bias=directional_shadow_bias(near=1.0, far=4000.0))

        shadowed = self._pixel(32, 32)
        self.assertTrue(all(channel <= 70 for channel in shadowed), shadowed)

    def test_opaque_caster_occludes_camera_aligned_sun_shine(self) -> None:
        update_render_shine_state(
            enabled=True,
            strength=1.0,
            power=8.0,
            fresnel=0.0,
            tint=(1.0, 1.0, 1.0),
        )
        unlit_packet = _packet(ambient=0.0, diffuse=0.0)

        self._render_shadow(())
        self._draw_receiver(packet=unlit_packet)
        visible_shine = self._pixel(32, 32)

        self._render_shadow(((self.white, False, (-0.35, 0.35, -0.35, 0.35)),))
        self._draw_receiver(packet=unlit_packet)
        blocked_shine = self._pixel(32, 32)

        self.assertTrue(all(channel >= 15 for channel in visible_shine), visible_shine)
        self.assertTrue(all(channel <= 10 for channel in blocked_shine), blocked_shine)
        self.assertGreater(visible_shine[0], blocked_shine[0] + 12)

    def test_geometric_gap_admits_sunlight(self) -> None:
        self._render_shadow(
            (
                (self.white, False, (-0.8, -0.15, -0.5, 0.5)),
                (self.white, False, (0.15, 0.8, -0.5, 0.5)),
            )
        )
        self._draw_receiver()
        gap = self._pixel(32, 32)
        roof = self._pixel(16, 32)
        self.assertTrue(all(channel >= 240 for channel in gap), gap)
        self.assertTrue(all(channel <= 70 for channel in roof), roof)

    def test_alpha_cutout_texel_admits_sunlight(self) -> None:
        cutout = self._texture(
            2,
            1,
            bytes((255, 255, 255, 255, 255, 255, 255, 0)),
        )
        vertex_data = np.asarray(
            (
                (-0.8, -0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0),
                (0.8, -0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0),
                (0.8, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0),
                (-0.8, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        mesh = BatchedMesh.from_vertex_data(
            vertex_data,
            texture=cutout,
            alpha_test=True,
            alpha_cutoff=0.5,
            shine_enabled=False,
            draw_mode=GL_QUADS,
        )
        try:
            self._render_mesh_shadows(mesh.shadow_meshes())
            self._draw_receiver()
            opaque_side = self._pixel(20, 32)
            transparent_side = self._pixel(44, 32)
            self.assertTrue(
                all(channel <= 80 for channel in opaque_side),
                opaque_side,
            )
            self.assertTrue(
                all(channel >= 230 for channel in transparent_side),
                transparent_side,
            )
        finally:
            mesh.dispose()

    def test_moving_caster_moves_shadow(self) -> None:
        self._render_shadow(((self.white, False, (-0.75, -0.15, -0.3, 0.3)),))
        self._draw_receiver()
        left_before = self._pixel(18, 32)
        right_before = self._pixel(46, 32)

        self._render_shadow(((self.white, False, (0.15, 0.75, -0.3, 0.3)),))
        self._draw_receiver()
        left_after = self._pixel(18, 32)
        right_after = self._pixel(46, 32)

        self.assertLess(left_before[0], left_after[0] - 120)
        self.assertLess(right_after[0], right_before[0] - 120)

    def test_opening_real_door_batch_updates_shared_shadow_geometry(self) -> None:
        door = Door(
            Vector3(0.0, 0.0, 0.5),
            camera=object(),
            texture=self.white,
            width=0.7,
            height=0.7,
            thickness=0.04,
            side="south",
            swing_speed=10.0,
        )
        batch = DoorRenderBatch((door,))
        try:
            self._render_mesh_shadows(batch.shadow_meshes())
            self._draw_receiver()
            center_closed = self._pixel(32, 32)

            door.target_open = True
            door.update(1.0)
            self._render_mesh_shadows(batch.shadow_meshes())
            self._draw_receiver()
            center_open = self._pixel(32, 32)

            self.assertLess(center_closed[0], center_open[0] - 120)
        finally:
            batch.dispose()
