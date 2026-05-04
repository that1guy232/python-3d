"""World rendering pipeline owned by the game layer."""

from __future__ import annotations

from contextlib import nullcontext
import math

from OpenGL.GL import (
    glBegin,
    glClear,
    glClearColor,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glFogf,
    glFogfv,
    glFogi,
    glLoadIdentity,
    glMatrixMode,
    glRotatef,
    glTranslatef,
    glVertex2f,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_EXP2,
    GL_FOG,
    GL_FOG_COLOR,
    GL_FOG_DENSITY,
    GL_FOG_MODE,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_TEXTURE_2D,
)
from OpenGL.GLU import gluPerspective

from config import FOGDENSITY, FOV, HEADBOB_ENABLED, HEIGHT, LIGHT_BLUE, WIDTH
from core.compat_shader import set_texture_fog_state


class WorldRenderer:
    """Render a WorldScene without making the world package own render systems."""

    def __init__(self, scene) -> None:
        self.scene = scene

    def _profile(self, name: str):
        profiler = getattr(self.scene, "profiler", None)
        if profiler is None or not getattr(profiler, "enabled", False):
            return nullcontext()
        return profiler.section(name)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _fog_enabled(self) -> bool:
        return bool(getattr(self.scene, "fog_enabled", True))

    def _apply_fog_state(self) -> None:
        if self._fog_enabled():
            glEnable(GL_FOG)
        else:
            glDisable(GL_FOG)

    def _sky_rgba(self) -> list[float]:
        brightness = self._clamp01(getattr(self.scene.camera, "brightness_default", 1.0))
        lighting = getattr(self.scene, "lighting", None)
        sky_color = getattr(lighting, "sky_color", LIGHT_BLUE)
        return [
            self._clamp01(channel * brightness)
            for channel in sky_color[:3]
        ] + [1.0]

    def draw_sky(self) -> None:  # pragma: no cover - visual
        scene = self.scene
        with self._profile("render.sky"):
            lighting = getattr(scene, "lighting", None)
            scene.sky.draw(
                scene.camera,
                sun_direction=getattr(
                    lighting,
                    "sun_direction",
                    getattr(scene, "sun_direction", None),
                ),
                lighting=lighting,
                fog_enabled=self._fog_enabled(),
            )

    def draw(self, enable_timing: bool = False) -> None:  # pragma: no cover - visual
        scene = self.scene
        with self._profile("draw.lighting_sync"):
            sync_lighting = getattr(scene, "_sync_lighting_uniforms", None)
            if callable(sync_lighting):
                sync_lighting()
            self._apply_fog_state()

        with self._profile("draw.ground"):
            scene.ground_mesh.draw()

        with self._profile("draw.fences"):
            for mesh in getattr(scene, "fence_meshes", ()):
                mesh.draw()

        with self._profile("draw.world_objects"):
            scene.draw_world_objects(enable_timing=enable_timing)

        with self._profile("draw.world_hud"):
            try:
                if getattr(scene, "hud_visible", True):
                    scene._hud.draw()
            except Exception:
                pass

    def render(
        self, *, show_hud: bool = True, text=None, fps: float | None = None
    ) -> None:  # pragma: no cover - visual
        scene = self.scene
        rgba = self._sky_rgba()
        fog_enabled = self._fog_enabled()
        fog_density = max(0.0, float(getattr(scene, "fog_density", FOGDENSITY)))

        with self._profile("render.setup"):
            if fog_enabled:
                glEnable(GL_FOG)
                glFogi(GL_FOG_MODE, GL_EXP2)
                glFogf(GL_FOG_DENSITY, fog_density)
            else:
                glDisable(GL_FOG)

            glFogfv(GL_FOG_COLOR, rgba)
            set_texture_fog_state(
                enabled=fog_enabled,
                density=fog_density,
                color=rgba,
                compile_shader=False,
            )
            glClearColor(*rgba)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(float(getattr(scene, "fov", FOV)), WIDTH / HEIGHT, 1, 1_000_000.0)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

        self.draw_sky()

        with self._profile("render.camera_transform"):
            glRotatef(math.degrees(-scene.camera.rotation.x), 1, 0, 0)
            glRotatef(math.degrees(-scene.camera.rotation.y), 0, 1, 0)
            off_x, off_y = scene._headbob.offsets()
            headbob_enabled = getattr(scene._headbob, "enabled", HEADBOB_ENABLED)
            idle_enabled = getattr(scene._headbob, "idle_enabled", True)
            if headbob_enabled:
                glTranslatef(-off_x, -off_y, 0)
            elif idle_enabled and off_y != 0.0:
                glTranslatef(0, -off_y, 0)
            glTranslatef(
                -scene.camera.position.x,
                -scene.camera.position.y,
                -scene.camera.position.z,
            )

        with self._profile("render.world"):
            self.draw()

        if show_hud and text is not None and fps is not None:
            with self._profile("render.hud_text"):
                self.draw_hud(text, fps)

    def draw_hud(self, text, fps: float) -> None:  # pragma: no cover - visual
        scene = self.scene
        if getattr(scene, "inventory_open", False):
            text.begin()
            text.draw_text(
                f"FPS: {fps:5.1f}",
                12,
                10,
                key="fps",
                align="topleft",
                color=[255, 0, 0, 0],
            )
            menu = "Inventory (press I to close)\n\n- Slot 1\n- Slot 2\n- Slot 3"
            text.draw_text_multiline(
                menu,
                WIDTH // 2,
                HEIGHT // 2,
                align="center",
                color=[255, 255, 255, 255],
            )
            text.end()
        elif getattr(scene, "paused", False):
            self.draw_pause_menu(text)
        else:
            if not getattr(scene, "debug_text_visible", True):
                return
            text.begin()
            text.draw_text(
                f"FPS: {fps:5.1f}",
                12,
                10,
                key="fps",
                align="topleft",
                color=[255, 0, 0, 0],
            )
            lorem = (
                "Lore Epsum: Vivamus sed nibh.\n"
                "Curabitur at leo quis nunc posuere congue.\n"
                "Praesent tristique sem at augue pharetra."
            )
            text.draw_text_multiline(lorem, 12, HEIGHT - 12, align="bottomleft")
            text.end()

    def _active_pause_menu(self):
        if getattr(self.scene, "showing_settings_menu", False):
            return getattr(self.scene, "setting_menu", None)
        return getattr(self.scene, "pause_menu", None)

    def draw_pause_menu(self, text) -> None:  # pragma: no cover - visual
        import pygame

        text.begin()
        glDisable(GL_TEXTURE_2D)
        glColor4f(0.0, 0.0, 0.0, 0.6)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(WIDTH, 0)
        glVertex2f(WIDTH, HEIGHT)
        glVertex2f(0, HEIGHT)
        glEnd()

        menu = self._active_pause_menu()
        if menu is None:
            glEnable(GL_TEXTURE_2D)
            text.end()
            return

        buttons = menu.compute_buttons()
        mx, my = pygame.mouse.get_pos()
        for button in buttons:
            x, y, w, h = button["rect"]
            hovered = x <= mx <= x + w and y <= my <= y + h
            is_slider = button.get("type") == "slider"

            glColor4f(0.12, 0.12, 0.12, 0.95 if hovered else 0.85)
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + w, y)
            glVertex2f(x + w, y + h)
            glVertex2f(x, y + h)
            glEnd()

            glColor4f(0.2, 0.2, 0.2, 0.9 if hovered else 0.75)
            glBegin(GL_QUADS)
            glVertex2f(x + 2, y + 2)
            glVertex2f(x + w - 2, y + 2)
            glVertex2f(x + w - 2, y + h - 2)
            glVertex2f(x + 2, y + h - 2)
            glEnd()

            if is_slider:
                padding = getattr(menu, "slider_horizontal_padding", 18)
                track_x = x + padding
                track_w = max(1, w - padding * 2)
                track_y = y + h - 12
                track_h = 5
                ratio = max(0.0, min(1.0, float(button.get("ratio", 0.0))))
                fill_w = track_w * ratio
                knob_x = track_x + fill_w

                glColor4f(0.08, 0.08, 0.08, 0.9)
                glBegin(GL_QUADS)
                glVertex2f(track_x, track_y)
                glVertex2f(track_x + track_w, track_y)
                glVertex2f(track_x + track_w, track_y + track_h)
                glVertex2f(track_x, track_y + track_h)
                glEnd()

                glColor4f(0.35, 0.58, 0.86, 0.95)
                glBegin(GL_QUADS)
                glVertex2f(track_x, track_y)
                glVertex2f(track_x + fill_w, track_y)
                glVertex2f(track_x + fill_w, track_y + track_h)
                glVertex2f(track_x, track_y + track_h)
                glEnd()

                glColor4f(0.88, 0.92, 0.98, 1.0)
                glBegin(GL_QUADS)
                glVertex2f(knob_x - 5, track_y - 5)
                glVertex2f(knob_x + 5, track_y - 5)
                glVertex2f(knob_x + 5, track_y + track_h + 5)
                glVertex2f(knob_x - 5, track_y + track_h + 5)
                glEnd()

        glEnable(GL_TEXTURE_2D)
        for button in buttons:
            x, y, w, h = button["rect"]
            if button.get("type") == "slider":
                padding = getattr(menu, "slider_horizontal_padding", 18)
                text.draw_text(
                    button["label"],
                    x + padding,
                    y + 6,
                    color=(255, 255, 255, 255),
                    align="topleft",
                )
                text.draw_text(
                    button.get("value_text", ""),
                    x + w - padding,
                    y + 6,
                    color=(220, 230, 245, 255),
                    align="topright",
                )
            else:
                text.draw_text(
                    button["label"],
                    x + w / 2,
                    y + h / 2,
                    color=(255, 255, 255, 255),
                    align="center",
                )

        title = getattr(menu, "title", None)
        if title and buttons:
            text.draw_text(
                title,
                WIDTH // 2,
                buttons[0]["rect"][1] - 40,
                color=(230, 230, 230, 255),
                align="center",
            )

        text.end()

    def compute_pause_buttons(self, width: int = WIDTH, height: int = HEIGHT):
        menu = self._active_pause_menu()
        if menu is None:
            return []
        return menu.compute_buttons(width=width, height=height)

    def handle_pause_click(self, pos) -> None:
        menu = self._active_pause_menu()
        if menu is not None:
            menu.handle_click(pos)

    def handle_pause_motion(self, pos) -> None:
        menu = self._active_pause_menu()
        if menu is not None:
            menu.handle_motion(pos)

    def handle_pause_release(self, pos) -> None:
        menu = self._active_pause_menu()
        if menu is not None:
            menu.handle_release(pos)
