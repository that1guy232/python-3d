"""World rendering pipeline owned by the game layer."""

from __future__ import annotations

from contextlib import nullcontext
import math
import time

from OpenGL.GL import (
    glClear,
    glClearColor,
    glDisable,
    glEnable,
    glFogf,
    glFogfv,
    glFogi,
    glLoadIdentity,
    glMatrixMode,
    glRotatef,
    glTranslatef,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_EXP2,
    GL_FOG,
    GL_FOG_COLOR,
    GL_FOG_DENSITY,
    GL_FOG_MODE,
    GL_MODELVIEW,
    GL_PROJECTION,
)
from OpenGL.GLU import gluPerspective

from game.config import FOGDENSITY, FOV, HEADBOB_ENABLED, HEIGHT, LIGHT_BLUE, VIEWDISTANCE, WIDTH
from engine.core.compat_shader import set_texture_fog_state
from engine.core.mesh import BatchedMesh
from engine.rendering.sprite import draw_sprites_batched
from game.world.objects.goblin import draw_goblin_shadows_batched
from game.world.ui.battle_panel import BattlePanel
from game.world.ui.inventory_panel import InventoryPanel
from game.world.ui.pause_panel import PauseMenuPanel


class WorldRenderer:
    """Render a WorldScene without making the world package own render systems."""

    def __init__(self, scene) -> None:
        self.scene = scene
        self.battle_panel = BattlePanel(scene)
        self.inventory_panel = InventoryPanel(scene)
        self.pause_panel = PauseMenuPanel(scene)
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
            self.draw_world_objects(enable_timing=enable_timing)

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

    def _approx_object_position(self, obj) -> tuple[float, float, float] | None:
        position = getattr(obj, "position", None)
        if position is not None:
            try:
                return (
                    float(position.x),
                    float(getattr(position, "y", 0.0)),
                    float(position.z),
                )
            except Exception:
                pass

        center = getattr(obj, "center", None)
        if center is not None:
            try:
                return (
                    float(center.x),
                    float(getattr(center, "y", 0.0)),
                    float(center.z),
                )
            except Exception:
                pass

        if hasattr(obj, "start") and hasattr(obj, "end"):
            try:
                scene = self.scene
                start = obj.start
                end = obj.end
                cx = (float(start.x) + float(end.x)) * 0.5
                cz = (float(start.z) + float(end.z)) * 0.5
                cy = float(
                    getattr(
                        obj,
                        "ground_y",
                        getattr(scene.camera.position, "y", 0.0)
                        if getattr(scene, "camera", None)
                        else 0.0,
                    )
                )
                return (cx, cy, cz)
            except Exception:
                pass

        return None

    @staticmethod
    def _sphere_for_vertices(vertices):
        if not vertices:
            return None
        min_x = min(float(v.x) for v in vertices)
        max_x = max(float(v.x) for v in vertices)
        min_y = min(float(v.y) for v in vertices)
        max_y = max(float(v.y) for v in vertices)
        min_z = min(float(v.z) for v in vertices)
        max_z = max(float(v.z) for v in vertices)
        center = (
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            (min_z + max_z) * 0.5,
        )
        radius = (
            ((max_x - min_x) * 0.5) ** 2
            + ((max_y - min_y) * 0.5) ** 2
            + ((max_z - min_z) * 0.5) ** 2
        ) ** 0.5
        return center, radius

    def _object_render_sphere(self, obj):
        center = getattr(obj, "bounds_center", None)
        if center is not None:
            return center, float(getattr(obj, "bounds_radius", 0.0))

        for method_name in ("get_render_bounding_sphere", "get_bounding_sphere"):
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    sphere = method()
                except Exception:
                    sphere = None
                if sphere:
                    return sphere

        visual_vertices = getattr(obj, "_visual_vertices", None)
        if callable(visual_vertices):
            try:
                sphere = self._sphere_for_vertices(visual_vertices())
            except Exception:
                sphere = None
            if sphere:
                return sphere

        get_vertices = getattr(obj, "get_world_vertices", None)
        if callable(get_vertices):
            try:
                sphere = self._sphere_for_vertices(get_vertices())
            except Exception:
                sphere = None
            if sphere:
                return sphere

        get_bounds = getattr(obj, "get_bounding_box", None)
        if callable(get_bounds):
            try:
                bbox = get_bounds()
                if bbox:
                    min_x, max_x, min_z, max_z = (float(v) for v in bbox)
                    scene = self.scene
                    pos = self._approx_object_position(obj)
                    cy = pos[1] if pos is not None else (
                        float(getattr(scene.camera.position, "y", 0.0))
                        if getattr(scene, "camera", None)
                        else 0.0
                    )
                    center = (
                        (min_x + max_x) * 0.5,
                        cy,
                        (min_z + max_z) * 0.5,
                    )
                    radius = (
                        ((max_x - min_x) * 0.5) ** 2
                        + ((max_z - min_z) * 0.5) ** 2
                    ) ** 0.5
                    return center, radius
            except Exception:
                pass

        pos = self._approx_object_position(obj)
        if pos is None:
            return None
        radius = max(
            0.0,
            float(
                getattr(
                    obj,
                    "render_radius",
                    getattr(obj, "collision_radius", 0.0),
                )
                or 0.0
            ),
        )
        return pos, radius

    def _object_visible(self, obj) -> bool:
        camera = getattr(self.scene, "camera", None)
        if camera is None:
            return True
        tester = getattr(camera, "sphere_in_frustum", None)
        if not callable(tester):
            return True
        sphere = self._object_render_sphere(obj)
        if sphere is None:
            return True
        center, radius = sphere
        return bool(tester(center, radius, far_distance=VIEWDISTANCE))

    def draw_world_objects(self, enable_timing: bool = False) -> None:  # pragma: no cover - visual
        scene = self.scene

        start_draw_decal_batches_time = time.perf_counter()
        with self._profile("objects.decal_batches"):
            profiler = getattr(scene, "profiler", None)
            for batch in scene.decal_batches:
                batch.draw(camera=scene.camera, profiler=profiler)
        end_draw_decal_batches_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing decal batches took "
                f"{end_draw_decal_batches_time - start_draw_decal_batches_time:.6f} seconds"
            )

        start_draw_wall_tiles_time = time.perf_counter()
        with self._profile("objects.wall_tiles"):
            if scene.wall_tile_batches:
                BatchedMesh.draw_many(
                    scene.wall_tile_batches,
                    camera=scene.camera,
                    view_distance=VIEWDISTANCE,
                )
            else:
                entity_ids = {id(entity) for entity in scene.entities}
                for wall in scene.wall_tiles:
                    if id(wall) in entity_ids:
                        continue
                    if not self._object_visible(wall):
                        continue
                    wall.draw(camera=scene.camera)
        end_draw_wall_tiles_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing wall tiles took "
                f"{end_draw_wall_tiles_time - start_draw_wall_tiles_time:.6f} seconds"
            )

        start_draw_polygons_time = time.perf_counter()
        with self._profile("objects.polygon_batches"):
            for batch in getattr(scene, "polygon_batches", ()) or ():
                batch.draw(camera=scene.camera, view_distance=VIEWDISTANCE)
        with self._profile("objects.polygons"):
            for polygon in scene.polygons:
                if getattr(polygon, "render_batched", False):
                    continue
                if not self._object_visible(polygon):
                    continue
                polygon.draw(camera=scene.camera)
        end_draw_polygons_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing polygons took "
                f"{end_draw_polygons_time - start_draw_polygons_time:.6f} seconds"
            )

        starting_draw_other_time = time.perf_counter()
        with self._profile("objects.road_batches"):
            for batch in getattr(scene, "road_batches", ()) or ():
                batch.draw(camera=scene.camera, view_distance=VIEWDISTANCE)

        with self._profile("objects.others"):
            for obj in scene.others:
                if getattr(obj, "render_batched", False):
                    continue
                if not self._object_visible(obj):
                    continue

                obj.draw()
        end_draw_other_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing other objects took "
                f"{end_draw_other_time - starting_draw_other_time:.6f} seconds"
            )

        start_draw_entities_time = time.perf_counter()

        with self._profile("objects.door_batches"):
            for batch in getattr(scene, "door_batches", ()) or ():
                batch.draw(camera=scene.camera, view_distance=VIEWDISTANCE)

        with self._profile("objects.window_batches"):
            for batch in getattr(scene, "window_batches", ()) or ():
                batch.draw(camera=scene.camera, view_distance=VIEWDISTANCE)

        with self._profile("objects.goblin_shadows"):
            goblins = getattr(scene, "goblins", None) or scene.entities
            draw_goblin_shadows_batched(
                goblins,
                camera=scene.camera,
                view_distance=VIEWDISTANCE,
            )

        with self._profile("objects.entities"):
            for entity in getattr(scene, "immediate_entities", ()) or ():
                if not getattr(entity, "enabled", True) or not getattr(
                    entity,
                    "visible",
                    True,
                ):
                    continue

                if not self._object_visible(entity):
                    continue

                draw_entity = getattr(entity, "draw", None)
                if callable(draw_entity):
                    with self._profile(f"entities.{type(entity).__name__}"):
                        try:
                            draw_entity(camera=scene.camera)
                        except TypeError:
                            draw_entity()
        end_draw_entities_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing entities took "
                f"{end_draw_entities_time - start_draw_entities_time:.6f} seconds"
            )

        start_draw_sprites_time = time.perf_counter()
        with self._profile("objects.sprites"):
            if scene.sprite_items and scene.camera is not None:
                draw_sprites_batched(
                    scene.sprite_items,
                    scene.camera,
                    scene.ground_height_at,
                    lighting=getattr(scene, "lighting", None),
                    sun_direction=getattr(scene, "sun_direction", None),
                    profiler=getattr(scene, "profiler", None),
                    static_data=True,
                )

        end_draw_sprites_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing sprites took "
                f"{end_draw_sprites_time - start_draw_sprites_time:.6f} seconds"
            )

    def draw_inventory(self, text, fps_label: str) -> None:  # pragma: no cover - visual
        self.inventory_panel.draw(text, fps_label, profile=self._profile)

    def draw_battle_menu(self, text) -> None:  # pragma: no cover - visual
        self.battle_panel.draw(text)

    def draw_pause_menu(self, text) -> None:  # pragma: no cover - visual
        self.pause_panel.draw(text)

