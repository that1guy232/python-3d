"""World rendering pipeline owned by the game layer."""

from __future__ import annotations

from contextlib import nullcontext
import math
import time

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
    glGetDoublev,
    glGetIntegerv,
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
    GL_MODELVIEW_MATRIX,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_PROJECTION_MATRIX,
    GL_QUADS,
    GL_TEXTURE_2D,
    GL_VIEWPORT,
)
from OpenGL.GLU import gluPerspective, gluProject

from game.config import FOGDENSITY, FOV, HEADBOB_ENABLED, HEIGHT, LIGHT_BLUE, VIEWDISTANCE, WIDTH
from engine.core.compat_shader import set_texture_fog_state
from engine.core.mesh import BatchedMesh


class WorldRenderer:
    """Render a WorldScene without making the world package own render systems."""

    def __init__(self, scene) -> None:
        self.scene = scene
        self._fps_label = "FPS:   0.0"
        self._fps_label_update_s = 0.0

    def _profile(self, name: str):
        profiler = getattr(self.scene, "profiler", None)
        if profiler is None or not getattr(profiler, "enabled", False):
            return nullcontext()
        return profiler.section(name)

    def _count(self, name: str, amount: float = 1.0) -> None:
        profiler = getattr(self.scene, "profiler", None)
        if profiler is not None and getattr(profiler, "enabled", False):
            profiler.count(name, amount)

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
                clouds_enabled=getattr(scene, "clouds_enabled", True),
                cloud_density=getattr(scene, "cloud_density", 1.0),
                cloud_speed=getattr(scene, "cloud_speed", 1.0),
                cloud_opacity=getattr(scene, "cloud_opacity", 1.0),
                profiler=getattr(scene, "profiler", None),
            )

    def draw(self, enable_timing: bool = False) -> None:  # pragma: no cover - visual
        scene = self.scene
        with self._profile("draw.lighting_sync"):
            sync_lighting = getattr(scene, "_sync_lighting_uniforms", None)
            if callable(sync_lighting):
                sync_lighting()
            self._apply_fog_state()

        with self._profile("draw.ground"):
            scene.ground_mesh.draw(camera=scene.camera, view_distance=VIEWDISTANCE)

        with self._profile("draw.fences"):
            BatchedMesh.draw_many(
                getattr(scene, "fence_meshes", ()),
                camera=scene.camera,
                view_distance=VIEWDISTANCE,
            )

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
        battle_overlay = getattr(scene, "battle_overlay", None)
        if battle_overlay is not None:
            battle_overlay.sync_state()
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

        profiler = getattr(scene, "profiler", None)
        if profiler is not None and getattr(profiler, "enabled", False):
            profiler.count("render.hud_text.enabled", float(self._hud_text_will_draw()))
            profiler.count("render.minimap.visible", float(getattr(scene, "minimap_visible", True)))

        if (
            show_hud
            and text is not None
            and fps is not None
            and self._hud_text_will_draw()
        ):
            with self._profile("render.hud_text"):
                self.draw_hud(text, fps)

    def _hud_text_will_draw(self) -> bool:
        scene = self.scene
        return bool(
            getattr(scene, "battle_mode", False)
            or getattr(scene, "inventory_open", False)
            or getattr(scene, "paused", False)
            or self._controls_text_visible()
        )

    def _controls_text_visible(self) -> bool:
        scene = self.scene
        return bool(
            getattr(
                scene,
                "controls_text_visible",
                getattr(scene, "debug_text_visible", True),
            )
        )

    def _controls_text(self) -> str:
        lines = [
            "Controls",
            "WASD: Move",
            "Mouse: Look",
            "Esc: Pause",
        ]
        prompt = None
        focused_prompt = getattr(self.scene, "focused_interaction_prompt", None)
        if callable(focused_prompt):
            prompt = focused_prompt()
        if prompt:
            lines.append(str(prompt))
        return "\n".join(lines)

    def draw_hud(self, text, fps: float) -> None:  # pragma: no cover - visual
        scene = self.scene
        fps_label = self._fps_text(fps)
        if getattr(scene, "battle_mode", False):
            self._count("hud_text.battle_frames")
            with self._profile("hud_text.battle_menu"):
                self.draw_battle_menu(text)
        elif getattr(scene, "inventory_open", False):
            self._count("hud_text.inventory_frames")
            with self._profile("hud_text.inventory_menu"):
                self.draw_inventory(text, fps_label)
        elif getattr(scene, "paused", False):
            self._count("hud_text.pause_frames")
            with self._profile("hud_text.pause_menu"):
                self.draw_pause_menu(text)
        else:
            if not self._controls_text_visible():
                self._count("hud_text.skipped")
                return
            self._count("hud_text.controls_frames")
            with self._profile("hud_text.begin"):
                text.begin()
            try:
                with self._profile("hud_text.fps"):
                    text.draw_text(
                        fps_label,
                        12,
                        10,
                        key="fps",
                        align="topleft",
                        color=[255, 0, 0, 0],
                    )
                controls = self._controls_text()
                with self._profile("hud_text.controls"):
                    text.draw_text_multiline(
                        controls,
                        12,
                        HEIGHT - 12,
                        align="bottomleft",
                    )
            finally:
                with self._profile("hud_text.end"):
                    text.end()

    def _fps_text(self, fps: float) -> str:
        now_s = time.perf_counter()
        if now_s >= self._fps_label_update_s:
            self._fps_label = f"FPS: {fps:5.1f}"
            self._fps_label_update_s = now_s + 0.25
        return self._fps_label

    def _player_stat_rows(self) -> list[tuple[str, str]]:
        stats = getattr(self.scene, "player_stats", None)
        if stats is None:
            return []

        hp = max(0, int(getattr(stats, "hp", 5)))
        max_hp = max(1, int(getattr(stats, "max_hp", max(1, hp))))
        mana = max(0, int(getattr(stats, "mana", 5)))
        max_mana = max(1, int(getattr(stats, "max_mana", max(1, mana))))
        strength = max(0, int(getattr(stats, "strength", 1)))
        dexterity = max(0, int(getattr(stats, "dexterity", 1)))
        elemental = max(0, int(getattr(stats, "elemental_damage", 0)))
        card_draw = max(0, int(getattr(stats, "card_draw", 1)))

        crit_percent = getattr(stats, "crit_percent", None)
        if callable(crit_percent):
            crit = int(crit_percent())
        else:
            crit = int(round(max(0.0, float(getattr(stats, "crit_chance", 0.0)))))

        return [
            ("HP", f"{hp}/{max_hp}"),
            ("Mana", f"{mana}/{max_mana}"),
            ("Strength", str(strength)),
            ("Dexterity", str(dexterity)),
            ("Crit Chance", f"{crit}%"),
            ("Elemental Damage", str(elemental)),
            ("Card Draw", str(card_draw)),
        ]

    @staticmethod
    def _inventory_item_label(item) -> str:
        if item is None:
            return ""
        if isinstance(item, dict):
            for key in ("name", "label", "title", "id"):
                value = item.get(key)
                if value:
                    return str(value)
            return ""
        name = getattr(item, "name", None) or getattr(item, "label", None)
        if name:
            return str(name)
        return str(item)

    @staticmethod
    def _fit_text_width(text, label: str, max_width: float) -> str:
        if not label:
            return ""
        try:
            if text.font.size(label)[0] <= max_width:
                return label
            for length in range(len(label) - 1, 0, -1):
                candidate = label[:length] + "."
                if text.font.size(candidate)[0] <= max_width:
                    return candidate
        except Exception:
            return label[:8]
        return ""

    @staticmethod
    def _inventory_panel_rect(
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> tuple[float, float, float, float]:
        outer_w = min(float(width) - 72.0, 930.0)
        outer_h = min(float(height) - 72.0, 560.0)
        outer_x = (float(width) - outer_w) * 0.5
        outer_y = (float(height) - outer_h) * 0.5
        return outer_x, outer_y, outer_w, outer_h

    def _inventory_close_rect(
        self,
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> tuple[float, float, float, float]:
        outer_x, outer_y, outer_w, _outer_h = self._inventory_panel_rect(width, height)
        size = 34.0
        return outer_x + outer_w - size - 20.0, outer_y + 18.0, size, size

    def draw_inventory(self, text, fps_label: str) -> None:  # pragma: no cover - visual
        import pygame

        text.begin()
        try:
            with self._profile("hud_text.fps"):
                text.draw_text(
                    fps_label,
                    12,
                    10,
                    key="fps",
                    align="topleft",
                    color=[255, 0, 0, 0],
                )

            outer_x, outer_y, outer_w, outer_h = self._inventory_panel_rect()
            padding = 24.0
            gap = 22.0
            stats_w = min(280.0, max(230.0, outer_w * 0.31))
            grid_x = outer_x + padding
            grid_y = outer_y + 82.0
            grid_w = outer_w - padding * 2.0 - stats_w - gap
            stats_x = grid_x + grid_w + gap
            stats_y = grid_y
            stats_h = outer_h - 108.0

            rows = self._player_stat_rows()
            items = list(getattr(self.scene, "inventory_items", ()) or ())
            cols = 6
            visible_rows = 4
            slot_gap = 10.0
            slot_size = min(
                72.0,
                max(
                    44.0,
                    min(
                        (grid_w - slot_gap * (cols - 1)) / cols,
                        (stats_h - slot_gap * (visible_rows - 1)) / visible_rows,
                    ),
                ),
            )
            grid_h = visible_rows * slot_size + (visible_rows - 1) * slot_gap
            slot_count = cols * visible_rows
            close_x, close_y, close_w, close_h = self._inventory_close_rect()
            mx, my = pygame.mouse.get_pos()
            close_hovered = close_x <= mx <= close_x + close_w and close_y <= my <= close_y + close_h

            glDisable(GL_TEXTURE_2D)
            self._draw_overlay_rect(0, 0, WIDTH, HEIGHT, (0.0, 0.0, 0.0, 0.55))
            self._draw_overlay_rect(
                outer_x,
                outer_y,
                outer_w,
                outer_h,
                (0.035, 0.04, 0.045, 0.96),
            )
            self._draw_overlay_rect(
                outer_x + 4.0,
                outer_y + 4.0,
                outer_w - 8.0,
                outer_h - 8.0,
                (0.085, 0.075, 0.065, 0.9),
            )
            self._draw_overlay_rect(
                grid_x - 12.0,
                grid_y - 14.0,
                grid_w + 24.0,
                grid_h + 28.0,
                (0.025, 0.026, 0.03, 0.72),
            )
            self._draw_overlay_rect(
                stats_x,
                outer_y + 62.0,
                stats_w,
                outer_h - 86.0,
                (0.025, 0.026, 0.03, 0.72),
            )
            self._draw_overlay_rect(
                close_x,
                close_y,
                close_w,
                close_h,
                (0.30, 0.12, 0.10, 0.96) if close_hovered else (0.12, 0.08, 0.075, 0.92),
            )
            self._draw_overlay_rect(
                close_x + 3.0,
                close_y + 3.0,
                close_w - 6.0,
                close_h - 6.0,
                (0.48, 0.18, 0.14, 0.74) if close_hovered else (0.22, 0.13, 0.11, 0.64),
            )

            for index in range(slot_count):
                col = index % cols
                row = index // cols
                x = grid_x + col * (slot_size + slot_gap)
                y = grid_y + row * (slot_size + slot_gap)
                filled = index < len(items)
                self._draw_overlay_rect(
                    x,
                    y,
                    slot_size,
                    slot_size,
                    (
                        (0.12, 0.105, 0.09, 0.96)
                        if filled
                        else (0.06, 0.058, 0.055, 0.92)
                    ),
                )
                self._draw_overlay_rect(
                    x + 3.0,
                    y + 3.0,
                    slot_size - 6.0,
                    slot_size - 6.0,
                    (
                        (0.23, 0.18, 0.12, 0.54)
                        if filled
                        else (0.11, 0.105, 0.1, 0.5)
                    ),
                )

            glEnable(GL_TEXTURE_2D)
            text.draw_text(
                "Inventory",
                outer_x + padding,
                outer_y + 24.0,
                color=(255, 245, 230, 255),
                align="topleft",
            )
            text.draw_text(
                "Stats",
                stats_x + 16.0,
                outer_y + 24.0,
                color=(255, 245, 230, 255),
                align="topleft",
            )
            text.draw_text(
                "X",
                close_x + close_w * 0.5,
                close_y + close_h * 0.5,
                color=(255, 245, 230, 255),
                align="center",
            )

            for index, item in enumerate(items[:slot_count]):
                label = self._inventory_item_label(item)
                if not label:
                    continue
                col = index % cols
                row = index // cols
                x = grid_x + col * (slot_size + slot_gap)
                y = grid_y + row * (slot_size + slot_gap)
                label = self._fit_text_width(text, label, slot_size - 10.0)
                if not label:
                    continue
                text.draw_text(
                    label,
                    x + slot_size * 0.5,
                    y + slot_size * 0.5,
                    color=(245, 235, 215, 255),
                    align="center",
                )

            stat_line_h = 38.0
            for index, (label, value) in enumerate(rows):
                y = stats_y + index * stat_line_h
                text.draw_text(
                    label,
                    stats_x + 18.0,
                    y,
                    color=(210, 214, 220, 255),
                    align="topleft",
                )
                text.draw_text(
                    value,
                    stats_x + stats_w - 18.0,
                    y,
                    color=(255, 245, 230, 255),
                    align="topright",
                )
        finally:
            text.end()

    def handle_inventory_click(self, pos) -> bool:
        mx, my = pos
        x, y, w, h = self._inventory_close_rect()
        if not (x <= mx <= x + w and y <= my <= y + h):
            return False

        self.scene.inventory_open = False
        self.scene.paused = False
        self.scene.showing_settings_menu = False

        import pygame

        try:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
        except pygame.error:
            pass
        return True

    def _active_pause_menu(self):
        if getattr(self.scene, "showing_settings_menu", False):
            return getattr(self.scene, "setting_menu", None)
        return getattr(self.scene, "pause_menu", None)

    def _active_battle_goblin(self):
        goblin = getattr(self.scene, "active_battle_goblin", None)
        if goblin is None or not getattr(goblin, "enabled", True):
            return None
        return goblin

    def _battle_goblin_hp(self, goblin) -> tuple[int, int]:
        max_hp = max(1, int(getattr(goblin, "max_hp", 5)))
        hp = max(0, min(max_hp, int(getattr(goblin, "hp", max_hp))))
        return hp, max_hp

    def _battle_player_stat_lines(self) -> list[str]:
        rows = dict(self._player_stat_rows())
        if not rows:
            return []
        return [
            f"STR {rows['Strength']}  DEX {rows['Dexterity']}",
            f"Crit {rows['Crit Chance']}  Elem {rows['Elemental Damage']}",
            f"Draw {rows['Card Draw']}",
        ]

    def _battle_hp_anchor(self) -> tuple[float, float] | None:
        goblin = self._active_battle_goblin()
        if goblin is None:
            return None

        position = getattr(goblin, "position", None)
        if position is None:
            return None

        try:
            sprite_height = max(
                1.0,
                float(
                    getattr(
                        goblin,
                        "_sprite_height",
                        getattr(goblin, "DEFAULT_HEIGHT", 42.0),
                    )
                ),
            )
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            win_x, win_y, win_z = gluProject(
                float(position.x),
                float(position.y) + sprite_height * 0.62,
                float(position.z),
                modelview,
                projection,
                viewport,
            )
        except Exception:
            return None

        if win_z < 0.0 or win_z > 1.0:
            return None

        viewport_x = float(viewport[0])
        viewport_y = float(viewport[1])
        viewport_h = float(viewport[3])
        screen_x = float(win_x) - viewport_x
        screen_y = viewport_h - (float(win_y) - viewport_y)
        return (
            max(0.0, min(float(WIDTH), screen_x)),
            max(0.0, min(float(HEIGHT), screen_y)),
        )

    @staticmethod
    def _draw_overlay_rect(
        x: float,
        y: float,
        w: float,
        h: float,
        color: tuple[float, float, float, float],
    ) -> None:
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def _draw_battle_hp_plate(
        self,
        text,
        anchor: tuple[float, float],
    ) -> None:  # pragma: no cover - visual
        goblin = self._active_battle_goblin()
        if goblin is None:
            return

        hp, max_hp = self._battle_goblin_hp(goblin)
        label = f"HP {hp}/{max_hp}"
        try:
            label_w = text.font.size(label)[0]
        except Exception:
            label_w = 72

        plate_w = max(112.0, float(label_w) + 28.0)
        plate_h = 42.0
        x = max(10.0, min(float(WIDTH) - plate_w - 10.0, anchor[0] - plate_w * 0.5))
        y = max(10.0, min(float(HEIGHT) - plate_h - 10.0, anchor[1] - plate_h - 8.0))
        ratio = hp / max_hp

        glDisable(GL_TEXTURE_2D)
        self._draw_overlay_rect(x, y, plate_w, plate_h, (0.04, 0.035, 0.03, 0.9))
        self._draw_overlay_rect(x + 4, y + 4, plate_w - 8, plate_h - 8, (0.12, 0.08, 0.06, 0.86))
        bar_x = x + 10.0
        bar_y = y + plate_h - 14.0
        bar_w = plate_w - 20.0
        self._draw_overlay_rect(bar_x, bar_y, bar_w, 7.0, (0.18, 0.03, 0.03, 0.95))
        self._draw_overlay_rect(
            bar_x,
            bar_y,
            bar_w * ratio,
            7.0,
            (0.86, 0.14, 0.08, 0.96),
        )
        glEnable(GL_TEXTURE_2D)
        text.draw_text(
            label,
            x + plate_w * 0.5,
            y + 16.0,
            color=(255, 245, 230, 255),
            align="center",
        )

    def _draw_battle_player_stats(self, text) -> None:  # pragma: no cover - visual
        lines = self._battle_player_stat_lines()
        if not lines:
            return

        try:
            max_label_w = max(text.font.size(line)[0] for line in lines)
            line_h = max(16, text.font.get_height())
        except Exception:
            max_label_w = 142
            line_h = 18

        panel_w = max(150.0, float(max_label_w) + 24.0)
        panel_h = 18.0 + float(len(lines) * line_h)
        x = 14.0
        y = 14.0

        glDisable(GL_TEXTURE_2D)
        self._draw_overlay_rect(x, y, panel_w, panel_h, (0.035, 0.035, 0.04, 0.9))
        self._draw_overlay_rect(x + 4, y + 4, panel_w - 8, panel_h - 8, (0.08, 0.075, 0.07, 0.86))
        glEnable(GL_TEXTURE_2D)

        for index, line in enumerate(lines):
            text.draw_text(
                line,
                x + 12.0,
                y + 9.0 + index * line_h,
                color=(235, 240, 250, 255),
                align="topleft",
            )

    def draw_battle_menu(self, text) -> None:  # pragma: no cover - visual
        hp_anchor = self._battle_hp_anchor()
        text.begin()
        glDisable(GL_TEXTURE_2D)

        if hp_anchor is not None:
            self._draw_battle_hp_plate(text, hp_anchor)
        self._draw_battle_player_stats(text)
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        if battle_overlay is not None:
            battle_overlay.draw(text)

        glEnable(GL_TEXTURE_2D)
        text.draw_text(
            "Battle mode",
            WIDTH // 2,
            24,
            color=(255, 245, 230, 255),
            align="center",
        )

        text.end()

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

    def compute_battle_buttons(self, width: int = WIDTH, height: int = HEIGHT):
        return []

    def handle_battle_click(self, pos) -> bool:
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        if battle_overlay is not None:
            return bool(battle_overlay.handle_mouse_down(pos))
        return False

    def handle_battle_motion(self, pos) -> bool:
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        if battle_overlay is not None:
            return bool(battle_overlay.handle_mouse_motion(pos))
        return False

    def handle_battle_release(self, pos) -> bool:
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        if battle_overlay is not None:
            return bool(battle_overlay.handle_mouse_up(pos))
        return False

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
