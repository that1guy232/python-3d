"""World rendering pipeline owned by the game layer."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
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

from game.config import (
    FOGDENSITY,
    FOV,
    HEADBOB_ENABLED,
    HEIGHT,
    LIGHT_BLUE,
    VIEWDISTANCE,
    WIDTH,
)
from engine.core.mesh import BatchedMesh
from engine.render_style_state import update_render_fog_state
from engine.rendering.sprite import draw_sprites_batched
from engine.rendering.packet_shader import (
    MAX_PACKET_POINT_LIGHTS,
    PacketLightingBackendUnavailable,
    get_packet_texture_lighting_shader,
)
from engine.rendering.directional_shadow import (
    DirectionalShadowBinding,
    DirectionalShadowMap,
    ShadowCasterShader,
    directional_light_matrix,
    directional_shadow_bias,
)
from engine.rendering.point_shadow import (
    PointShadowBinding,
    PointShadowCasterShader,
    PointShadowMap,
    point_light_face_matrices,
)
from game.world.lighting_receivers import (
    DECAL_LIGHTING_RECEIVER,
    ROAD_LIGHTING_RECEIVER,
    SKY_CLEAR_LIGHTING_RECEIVER,
    SKY_CLOUD_LIGHTING_RECEIVER,
    SKY_SUN_LIGHTING_RECEIVER,
    SPRITE_LIGHTING_RECEIVER,
)
from game.world.inventory import active_inventory_notice
from game.world.objects.goblin import draw_goblin_shadows_batched
from game.world.ui.battle_panel import BattlePanel
from game.world.ui.inventory_panel import InventoryPanel
from game.world.ui.pause_panel import PauseMenuPanel
from game.world.world_state import WorldRenderResources, WorldUIState


@dataclass(slots=True)
class _SunShadowCache:
    caster_revision: tuple
    center: tuple[float, float, float]
    direction: tuple[float, float, float]
    binding: DirectionalShadowBinding


@dataclass(slots=True)
class _PointShadowCache:
    light_id: str
    light_revision: tuple
    caster_revision: tuple
    binding: PointShadowBinding


class WorldRenderer:
    """Render a WorldScene without making the world package own render systems."""

    def __init__(
        self,
        scene,
        *,
        resources: WorldRenderResources | None = None,
        ui_state: WorldUIState | None = None,
        lighting_controller=None,
    ) -> None:
        self.scene = scene
        self.resources = resources or getattr(scene, "render_resources", scene)
        self.ui_state = ui_state or getattr(scene, "ui_state", scene)
        self.lighting_controller = lighting_controller or getattr(
            scene, "lighting_controller", None
        )
        self.battle_panel = BattlePanel(scene)
        self.inventory_panel = InventoryPanel(scene)
        self.pause_panel = PauseMenuPanel(scene)
        self._fps_label = "FPS:   0.0"
        self._fps_label_update_s = 0.0
        self._sun_shadow_map: DirectionalShadowMap | None = None
        self._shadow_caster_shader: ShadowCasterShader | None = None
        self._sun_shadow_cache: _SunShadowCache | None = None
        self._point_shadow_maps: list[PointShadowMap] = []
        self._point_shadow_caster_shader: PointShadowCasterShader | None = None
        self._point_shadow_cache: list[_PointShadowCache | None] = []
        self._point_shadow_update_cursor = 0
        self.sun_shadow_map_size = 2048
        self.sun_shadow_extent = min(float(VIEWDISTANCE), 1000.0)
        self.sun_shadow_camera_threshold = 4.0
        self.sun_shadow_direction_threshold_degrees = 0.25
        self.point_shadow_map_size = 256
        self.max_point_shadows = 2
        self.point_shadow_updates_per_frame = 1

    def _profile(self, name: str):
        profiler = getattr(self.scene, "profiler", None)
        if profiler is None or not getattr(profiler, "enabled", False):
            return nullcontext()
        return profiler.section(name)

    def _count(self, name: str, amount: float = 1.0) -> None:
        profiler = getattr(self.scene, "profiler", None)
        if profiler is not None and getattr(profiler, "enabled", False):
            profiler.count(name, amount)

    def _ui_value(self, name: str, default=None, *, legacy_name: str | None = None):
        if hasattr(self.ui_state, name):
            return getattr(self.ui_state, name)
        return getattr(self.scene, legacy_name or name, default)

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

    def _packet_lighting_shader(self):
        if getattr(self.scene, "lighting_backend", "legacy") != "packet":
            return None
        shader = get_packet_texture_lighting_shader()
        if shader is None:
            raise PacketLightingBackendUnavailable(
                "packet lighting backend was selected but shader compilation failed"
            )
        return shader

    def _ensure_sun_shadow_resources(
        self,
    ) -> tuple[DirectionalShadowMap, ShadowCasterShader]:
        if self._sun_shadow_map is None:
            self._sun_shadow_map = DirectionalShadowMap.create(
                self.sun_shadow_map_size
            )
        if self._shadow_caster_shader is None:
            self._shadow_caster_shader = ShadowCasterShader.create()
        return self._sun_shadow_map, self._shadow_caster_shader

    def _shadow_sources(self):
        resources = self.resources
        if resources.ground_mesh is not None:
            yield resources.ground_mesh
        for attr_name in (
            "fence_meshes",
            "wall_tile_batches",
            "road_batches",
            "door_batches",
            "window_batches",
            "polygon_batches",
            "others",
            "immediate_entities",
        ):
            yield from (getattr(resources, attr_name, ()) or ())

    def _shadow_meshes_from(self, source):
        if (
            source is None
            or not getattr(source, "visible", True)
            or not getattr(source, "casts_shadows", True)
        ):
            return ()
        provider = getattr(source, "shadow_meshes", None)
        if callable(provider):
            try:
                return tuple(provider(camera=getattr(self.scene, "camera", None)))
            except TypeError:
                return tuple(provider())
        mesh = getattr(source, "_mesh", None)
        if isinstance(mesh, BatchedMesh):
            return mesh.shadow_meshes()
        return ()

    def _shadow_mesh_records(self):
        """Yield meshes with the revision of the source that owns them."""

        for source in self._shadow_sources():
            source_revision = getattr(source, "shadow_revision", None)
            for mesh in self._shadow_meshes_from(source):
                yield mesh, (id(source), source_revision)

    @staticmethod
    def _shadow_mesh_revision(mesh, source_revision) -> tuple:
        """Return a cheap geometry-only cache key for one shadow caster."""

        center = getattr(mesh, "bounds_center", None)
        return (
            source_revision,
            id(mesh),
            getattr(mesh, "shadow_revision", None),
            int(getattr(mesh, "vbo_vertices", 0) or 0),
            int(getattr(mesh, "vertex_count", 0) or 0),
            tuple(float(value) for value in center) if center is not None else None,
            float(getattr(mesh, "bounds_radius", 0.0)),
            bool(getattr(mesh, "casts_shadows", True)),
            bool(getattr(mesh, "alpha_test", False)),
            float(getattr(mesh, "alpha_cutoff", 0.0)),
            int(getattr(mesh, "texture", 0) or 0),
        )

    @staticmethod
    def _normalized_direction(direction) -> tuple[float, float, float]:
        values = tuple(float(value) for value in direction)
        length = math.sqrt(sum(value * value for value in values))
        if length <= 1e-8:
            return (0.0, 1.0, 0.0)
        return tuple(value / length for value in values)

    def _sun_shadow_needs_refresh(
        self,
        *,
        caster_revision: tuple,
        center: tuple[float, float, float],
        direction: tuple[float, float, float],
    ) -> bool:
        cache = self._sun_shadow_cache
        if cache is None or cache.caster_revision != caster_revision:
            return True
        camera_threshold = max(0.0, float(self.sun_shadow_camera_threshold))
        if math.dist(cache.center, center) > camera_threshold:
            return True
        angle_threshold = max(
            0.0,
            float(self.sun_shadow_direction_threshold_degrees),
        )
        direction_dot = max(
            -1.0,
            min(
                1.0,
                sum(
                    previous * current
                    for previous, current in zip(cache.direction, direction)
                ),
            ),
        )
        return direction_dot < math.cos(math.radians(angle_threshold))

    def render_sun_shadow(self) -> None:
        """Render current world-space casters and bind their sun visibility."""

        packet_shader = self._packet_lighting_shader()
        if packet_shader is None:
            return
        ground = getattr(self.resources, "ground_mesh", None)
        camera = getattr(self.scene, "camera", None)
        lighting = getattr(self.scene, "lighting", None)
        if ground is None or camera is None or lighting is None:
            packet_shader.set_directional_shadow(None)
            return

        direction = self._normalized_direction(
            getattr(lighting, "light_direction", (0.0, 1.0, 0.0))
        )
        center = (
            float(camera.position.x),
            float(camera.position.y),
            float(camera.position.z),
        )
        shadow_records = tuple(self._shadow_mesh_records())
        sun_casters = tuple(
            (mesh, source_revision)
            for mesh, source_revision in shadow_records
            if getattr(mesh, "casts_sun_shadows", True)
        )
        caster_revision = tuple(
            self._shadow_mesh_revision(mesh, source_revision)
            for mesh, source_revision in sun_casters
        )
        if not self._sun_shadow_needs_refresh(
            caster_revision=caster_revision,
            center=center,
            direction=direction,
        ):
            self._count("shadow.sun.cache_hits")
            packet_shader.set_directional_shadow(self._sun_shadow_cache.binding)
            return

        extent = max(50.0, float(self.sun_shadow_extent))
        shadow_near = 1.0
        shadow_far = extent * 4.0
        light_matrix = directional_light_matrix(
            center,
            direction,
            extent=extent,
            near=shadow_near,
            far=shadow_far,
        )
        shadow_map, caster_shader = self._ensure_sun_shadow_resources()
        with shadow_map.render_depth():
            for mesh, _source_revision in sun_casters:
                mesh.draw_shadow(caster_shader, light_matrix)
        binding = shadow_map.binding(
            light_matrix,
            bias=directional_shadow_bias(
                near=shadow_near,
                far=shadow_far,
            ),
        )
        self._sun_shadow_cache = _SunShadowCache(
            caster_revision=caster_revision,
            center=center,
            direction=direction,
            binding=binding,
        )
        self._count("shadow.sun.refreshes")
        packet_shader.set_directional_shadow(binding)

    def _selected_point_lights(self, limit: int = MAX_PACKET_POINT_LIGHTS):
        lighting = getattr(self.scene, "lighting", None)
        lights = tuple(getattr(lighting, "point_lights", ()) or ())
        eligible = [
            light
            for light in lights
            if float(light.intensity) > 0.0
            and float(light.range) > 0.0
        ]
        camera = getattr(self.scene, "camera", None)
        if camera is None:
            return tuple(eligible[: max(0, int(limit))])

        camera_position = (
            float(camera.position.x),
            float(camera.position.y),
            float(camera.position.z),
        )

        def priority(light):
            distance = math.dist(camera_position, light.position)
            importance = max(0.0001, float(light.importance))
            in_range = distance <= float(light.range)
            return (int(in_range), importance / max(1.0, distance), light.light_id)

        eligible.sort(key=priority, reverse=True)
        return tuple(eligible[: max(0, int(limit))])

    def _ensure_point_shadow_resources(
        self,
        count: int,
    ) -> tuple[tuple[PointShadowMap, ...], PointShadowCasterShader]:
        while len(self._point_shadow_maps) < count:
            self._point_shadow_maps.append(
                PointShadowMap.create(self.point_shadow_map_size)
            )
            self._point_shadow_cache.append(None)
        if self._point_shadow_caster_shader is None:
            self._point_shadow_caster_shader = PointShadowCasterShader.create()
        return (
            tuple(self._point_shadow_maps),
            self._point_shadow_caster_shader,
        )

    @staticmethod
    def _mesh_intersects_point_light(mesh, position, light_range: float) -> bool:
        center = getattr(mesh, "bounds_center", None)
        if center is None:
            return True
        radius = max(0.0, float(getattr(mesh, "bounds_radius", 0.0)))
        return math.dist(center, position) <= float(light_range) + radius

    @staticmethod
    def _point_light_revision(light) -> tuple:
        return (
            str(light.light_id),
            tuple(float(value) for value in light.position),
            float(light.range),
        )

    def _assign_point_shadow_maps(self, lights) -> tuple[tuple[object, int], ...]:
        """Keep a light on its existing cube map when priority order changes."""

        assignments: list[tuple[object, int] | None] = [None] * len(lights)
        claimed: set[int] = set()
        for light_index, light in enumerate(lights):
            for map_index, cache in enumerate(self._point_shadow_cache):
                if (
                    map_index not in claimed
                    and cache is not None
                    and cache.light_id == str(light.light_id)
                ):
                    assignments[light_index] = (light, map_index)
                    claimed.add(map_index)
                    break
        available = (
            index
            for index in range(len(self._point_shadow_maps))
            if index not in claimed
        )
        for light_index, light in enumerate(lights):
            if assignments[light_index] is None:
                map_index = next(available)
                assignments[light_index] = (light, map_index)
                claimed.add(map_index)
        return tuple(
            assignment for assignment in assignments if assignment is not None
        )

    def _staggered_point_shadow_refreshes(
        self,
        assignments,
        revisions,
    ) -> set[int]:
        """Choose new maps immediately and spread ordinary invalidations out."""

        refreshes: set[int] = set()
        pending: list[int] = []
        for light, map_index in assignments:
            light_revision, caster_revision = revisions[map_index]
            cache = self._point_shadow_cache[map_index]
            if cache is None or cache.light_id != str(light.light_id):
                refreshes.add(map_index)
            elif (
                cache.light_revision != light_revision
                or cache.caster_revision != caster_revision
            ):
                pending.append(map_index)

        budget = max(0, int(self.point_shadow_updates_per_frame))
        if pending and budget:
            map_count = max(1, len(self._point_shadow_maps))
            pending.sort(
                key=lambda index: (index - self._point_shadow_update_cursor) % map_count
            )
            selected = pending[:budget]
            refreshes.update(selected)
            self._point_shadow_update_cursor = (selected[-1] + 1) % map_count
        return refreshes

    def render_point_shadows(self) -> None:
        """Render radial depth cubes for the most relevant local lights."""

        packet_shader = self._packet_lighting_shader()
        if packet_shader is None:
            return
        active_lights = self._selected_point_lights()
        packet_shader.set_active_point_lights(
            light.light_id for light in active_lights
        )
        lights = tuple(
            light for light in active_lights if light.casts_shadows
        )[: self.max_point_shadows]
        if not lights:
            packet_shader.set_point_shadows(())
            return

        _, caster_shader = self._ensure_point_shadow_resources(
            len(lights)
        )
        shadow_records = tuple(self._shadow_mesh_records())
        assignments = self._assign_point_shadow_maps(lights)
        revisions = {}
        casters_by_map = {}
        for light, map_index in assignments:
            light_range = max(0.1, float(light.range))
            position = tuple(float(value) for value in light.position)
            caster_records = tuple(
                (mesh, source_revision)
                for mesh, source_revision in shadow_records
                if self._mesh_intersects_point_light(mesh, position, light_range)
            )
            casters_by_map[map_index] = tuple(
                mesh for mesh, _source_revision in caster_records
            )
            revisions[map_index] = (
                self._point_light_revision(light),
                tuple(
                    self._shadow_mesh_revision(mesh, source_revision)
                    for mesh, source_revision in caster_records
                ),
            )

        refreshes = self._staggered_point_shadow_refreshes(
            assignments,
            revisions,
        )
        bindings = []
        for light, map_index in assignments:
            shadow_map = self._point_shadow_maps[map_index]
            light_range = max(0.1, float(light.range))
            position = tuple(float(value) for value in light.position)
            if map_index in refreshes:
                near = max(0.1, min(1.0, light_range * 0.01))
                matrices = point_light_face_matrices(
                    position,
                    near=near,
                    far=light_range,
                )
                for face_index, matrix in enumerate(matrices):
                    with shadow_map.render_face(face_index):
                        for mesh in casters_by_map[map_index]:
                            mesh.draw_point_shadow(
                                caster_shader,
                                matrix,
                                position,
                                light_range,
                            )
                binding = shadow_map.binding(
                    light.light_id,
                    position,
                    light_range,
                    bias=max(0.1, min(0.75, light_range * 0.003)),
                )
                light_revision, caster_revision = revisions[map_index]
                self._point_shadow_cache[map_index] = _PointShadowCache(
                    light_id=str(light.light_id),
                    light_revision=light_revision,
                    caster_revision=caster_revision,
                    binding=binding,
                )
                self._count("shadow.point.refreshes")
                self._count("shadow.point.faces", 6.0)
            else:
                self._count("shadow.point.cache_hits")
            bindings.append(self._point_shadow_cache[map_index].binding)
        packet_shader.set_point_shadows(bindings)

    def dispose(self) -> None:
        if self._sun_shadow_map is not None or self._point_shadow_maps:
            packet_shader = get_packet_texture_lighting_shader()
            if packet_shader is not None:
                packet_shader.set_directional_shadow(None)
                packet_shader.set_point_shadows(())
                packet_shader.set_active_point_lights(None)
        if self._shadow_caster_shader is not None:
            self._shadow_caster_shader.dispose()
            self._shadow_caster_shader = None
        if self._sun_shadow_map is not None:
            self._sun_shadow_map.dispose()
            self._sun_shadow_map = None
        self._sun_shadow_cache = None
        if self._point_shadow_caster_shader is not None:
            self._point_shadow_caster_shader.dispose()
            self._point_shadow_caster_shader = None
        for shadow_map in self._point_shadow_maps:
            shadow_map.dispose()
        self._point_shadow_maps.clear()
        self._point_shadow_cache.clear()
        self._point_shadow_update_cursor = 0

    def _lighting_packet_for(self, mesh):
        if self.lighting_controller is None or mesh is None:
            return None
        receiver = getattr(mesh, "lighting_receiver", None)
        if receiver is None:
            return None
        return self._receiver_packet_for(receiver)

    def _receiver_packet_for(self, receiver):
        if self.lighting_controller is None or receiver is None:
            return None
        return self.lighting_controller.render_packet_for(receiver)

    def _sky_rgba(self) -> list[float]:
        packet = self._receiver_packet_for(SKY_CLEAR_LIGHTING_RECEIVER)
        if packet is not None:
            return [
                self._clamp01(channel * packet.exposure)
                for channel in packet.sky_color[:3]
            ] + [1.0]
        brightness = (
            self._clamp01(
                getattr(self.scene.camera, "brightness_default", 1.0)
            )
            if SKY_CLEAR_LIGHTING_RECEIVER.exposure
            else 1.0
        )
        lighting = getattr(self.scene, "lighting", None)
        sky_color = getattr(lighting, "sky_color", LIGHT_BLUE)
        return [self._clamp01(channel * brightness) for channel in sky_color[:3]] + [
            1.0
        ]

    def draw_sky(self) -> None:  # pragma: no cover - visual
        scene = self.scene
        with self._profile("render.sky"):
            lighting = getattr(scene, "lighting", None)
            sun_packet = self._receiver_packet_for(SKY_SUN_LIGHTING_RECEIVER)
            cloud_packet = self._receiver_packet_for(SKY_CLOUD_LIGHTING_RECEIVER)
            self.resources.sky.draw(
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
                sun_receiver=SKY_SUN_LIGHTING_RECEIVER,
                cloud_receiver=SKY_CLOUD_LIGHTING_RECEIVER,
                sun_packet=sun_packet,
                cloud_packet=cloud_packet,
            )

    def draw(self, enable_timing: bool = False) -> None:  # pragma: no cover - visual
        scene = self.scene
        with self._profile("draw.lighting_sync"):
            if self.lighting_controller is not None:
                self.lighting_controller.sync_uniforms()
            self._apply_fog_state()

        with self._profile("draw.ground"):
            packet_shader = self._packet_lighting_shader()
            lighting_packet = (
                self._lighting_packet_for(self.resources.ground_mesh)
                if packet_shader is not None
                else None
            )
            if packet_shader is not None and lighting_packet is None:
                raise RuntimeError(
                    "packet lighting backend has no ground receiver packet"
                )
            self.resources.ground_mesh.draw(
                camera=scene.camera,
                view_distance=VIEWDISTANCE,
                lighting_packet=lighting_packet,
                packet_shader=packet_shader,
            )

        with self._profile("draw.fences"):
            packet_shader = self._packet_lighting_shader()
            BatchedMesh.draw_many(
                self.resources.fence_meshes,
                camera=scene.camera,
                view_distance=VIEWDISTANCE,
                lighting_packets=(
                    self.lighting_controller.render_packets
                    if packet_shader is not None
                    and self.lighting_controller is not None
                    else None
                ),
                packet_shader=packet_shader,
                require_lighting_packets=packet_shader is not None,
            )

        with self._profile("draw.world_objects"):
            self.draw_world_objects(enable_timing=enable_timing)

        with self._profile("draw.world_hud"):
            try:
                hud = self._ui_value("hud", legacy_name="_hud")
                if self._ui_value("hud_visible", True) and hud is not None:
                    hud.draw()
            except Exception:
                pass

    def render(
        self, *, show_hud: bool = True, text=None, fps: float | None = None
    ) -> None:  # pragma: no cover - visual
        scene = self.scene
        battle_overlay = self._ui_value("battle_overlay")
        if battle_overlay is not None:
            battle_overlay.sync_state()
        if self.lighting_controller is not None:
            self.lighting_controller.sync_uniforms()
        with self._profile("render.sun_shadow"):
            self.render_sun_shadow()
        with self._profile("render.point_shadows"):
            self.render_point_shadows()
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
            update_render_fog_state(
                enabled=fog_enabled,
                density=fog_density,
                color=rgba,
            )
            glClearColor(*rgba)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(
                float(getattr(scene, "fov", FOV)), WIDTH / HEIGHT, 1, 1_000_000.0
            )

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
            profiler.count(
                "render.minimap.visible",
                float(self._ui_value("minimap_visible", True)),
            )

        if (
            show_hud
            and text is not None
            and fps is not None
            and self._hud_text_will_draw()
        ):
            with self._profile("render.hud_text"):
                self.draw_hud(text, fps)

    def _hud_text_will_draw(self) -> bool:
        return bool(
            self._ui_value("battle_mode", False)
            or self._ui_value("inventory_open", False)
            or self._ui_value("paused", False)
            or self._controls_text_visible()
            or active_inventory_notice(self.scene)
        )

    def _controls_text_visible(self) -> bool:
        return bool(
            self._ui_value(
                "controls_text_visible",
                self._ui_value("debug_text_visible", True),
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
        fps_label = self._fps_text(fps)
        if self._ui_value("battle_mode", False):
            self._count("hud_text.battle_frames")
            with self._profile("hud_text.battle_menu"):
                self.draw_battle_menu(text)
        elif self._ui_value("inventory_open", False):
            self._count("hud_text.inventory_frames")
            with self._profile("hud_text.inventory_menu"):
                self.draw_inventory(text, fps_label)
        elif self._ui_value("paused", False):
            self._count("hud_text.pause_frames")
            with self._profile("hud_text.pause_menu"):
                self.draw_pause_menu(text)
        else:
            if self._controls_text_visible():
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
            else:
                self._count("hud_text.skipped")

        self.draw_inventory_notice(text)

    def draw_inventory_notice(self, text) -> None:  # pragma: no cover - visual
        notice = active_inventory_notice(self.scene)
        if not notice:
            return
        text.begin()
        try:
            text.draw_text(
                notice,
                WIDTH - 18.0,
                HEIGHT - 18.0,
                key="inventory_notice",
                align="bottomright",
                color=(255, 236, 180, 255),
            )
        finally:
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
                        (
                            getattr(scene.camera.position, "y", 0.0)
                            if getattr(scene, "camera", None)
                            else 0.0
                        ),
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
                    cy = (
                        pos[1]
                        if pos is not None
                        else (
                            float(getattr(scene.camera.position, "y", 0.0))
                            if getattr(scene, "camera", None)
                            else 0.0
                        )
                    )
                    center = (
                        (min_x + max_x) * 0.5,
                        cy,
                        (min_z + max_z) * 0.5,
                    )
                    radius = (
                        ((max_x - min_x) * 0.5) ** 2 + ((max_z - min_z) * 0.5) ** 2
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

    def draw_world_objects(
        self, enable_timing: bool = False
    ) -> None:  # pragma: no cover - visual
        scene = self.scene
        resources = self.resources

        start_draw_decal_batches_time = time.perf_counter()
        with self._profile("objects.decal_batches"):
            profiler = getattr(scene, "profiler", None)
            packet_shader = self._packet_lighting_shader()
            decal_packet = (
                self.lighting_controller.render_packet_for(
                    DECAL_LIGHTING_RECEIVER
                )
                if packet_shader is not None
                and self.lighting_controller is not None
                else None
            )
            if packet_shader is not None and decal_packet is None:
                raise RuntimeError(
                    "packet lighting backend has no decal receiver packet"
                )
            for batch in resources.decal_batches:
                batch.draw(
                    camera=scene.camera,
                    profiler=profiler,
                    lighting_packet=decal_packet,
                    packet_shader=packet_shader,
                )
        end_draw_decal_batches_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing decal batches took "
                f"{end_draw_decal_batches_time - start_draw_decal_batches_time:.6f} seconds"
            )

        start_draw_wall_tiles_time = time.perf_counter()
        with self._profile("objects.wall_tiles"):
            if resources.wall_tile_batches:
                packet_shader = self._packet_lighting_shader()
                BatchedMesh.draw_many(
                    resources.wall_tile_batches,
                    camera=scene.camera,
                    view_distance=VIEWDISTANCE,
                    lighting_packets=(
                        self.lighting_controller.render_packets
                        if packet_shader is not None
                        and self.lighting_controller is not None
                        else None
                    ),
                    packet_shader=packet_shader,
                    require_lighting_packets=packet_shader is not None,
                )
            else:
                entity_ids = {id(entity) for entity in resources.entities}
                for wall in resources.wall_tiles:
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
            packet_shader = self._packet_lighting_shader()
            for batch in resources.polygon_batches:
                batch.draw(
                    camera=scene.camera,
                    view_distance=VIEWDISTANCE,
                    lighting_packets=(
                        self.lighting_controller.render_packets
                        if packet_shader is not None
                        and self.lighting_controller is not None
                        else None
                    ),
                    packet_shader=packet_shader,
                )
        with self._profile("objects.polygons"):
            for polygon in resources.polygons:
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
            packet_shader = self._packet_lighting_shader()
            for batch in resources.road_batches:
                batch.draw(
                    camera=scene.camera,
                    view_distance=VIEWDISTANCE,
                    lighting_packets=(
                        self.lighting_controller.render_packets
                        if packet_shader is not None
                        and self.lighting_controller is not None
                        else None
                    ),
                    packet_shader=packet_shader,
                )

        with self._profile("objects.others"):
            for obj in resources.others:
                if getattr(obj, "render_batched", False):
                    continue
                if not self._object_visible(obj):
                    continue

                mesh = getattr(obj, "_mesh", None)
                receiver = getattr(mesh, "lighting_receiver", None)
                if (
                    packet_shader is not None
                    and receiver is not None
                    and receiver is ROAD_LIGHTING_RECEIVER
                ):
                    lighting_packet = self._lighting_packet_for(mesh)
                    if lighting_packet is None:
                        raise RuntimeError(
                            "packet lighting backend has no direct-road packet"
                        )
                    obj.draw(
                        camera=scene.camera,
                        view_distance=VIEWDISTANCE,
                        lighting_packet=lighting_packet,
                        packet_shader=packet_shader,
                    )
                else:
                    obj.draw()
        end_draw_other_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing other objects took "
                f"{end_draw_other_time - starting_draw_other_time:.6f} seconds"
            )

        start_draw_entities_time = time.perf_counter()

        with self._profile("objects.door_batches"):
            packet_shader = self._packet_lighting_shader()
            for batch in resources.door_batches:
                batch.draw(
                    camera=scene.camera,
                    view_distance=VIEWDISTANCE,
                    lighting_packets=(
                        self.lighting_controller.render_packets
                        if packet_shader is not None
                        and self.lighting_controller is not None
                        else None
                    ),
                    packet_shader=packet_shader,
                )

        with self._profile("objects.window_batches"):
            for batch in resources.window_batches:
                batch.draw(
                    camera=scene.camera,
                    view_distance=VIEWDISTANCE,
                    lighting_packets=(
                        self.lighting_controller.render_packets
                        if packet_shader is not None
                        and self.lighting_controller is not None
                        else None
                    ),
                    packet_shader=packet_shader,
                )

        with self._profile("objects.goblin_shadows"):
            build_state = getattr(scene, "build_state", scene)
            goblins = getattr(build_state, "goblins", None) or resources.entities
            draw_goblin_shadows_batched(
                goblins,
                camera=scene.camera,
                view_distance=VIEWDISTANCE,
            )

        with self._profile("objects.entities"):
            for entity in resources.immediate_entities:
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
                        packet_receiver = getattr(
                            entity,
                            "packet_lighting_receiver",
                            None,
                        )
                        if (
                            packet_shader is not None
                            and packet_receiver is not None
                        ):
                            lighting_packet = (
                                self.lighting_controller.render_packet_for(
                                    packet_receiver
                                )
                                if self.lighting_controller is not None
                                else None
                            )
                            if lighting_packet is None:
                                raise RuntimeError(
                                    "packet lighting backend has no dynamic "
                                    "object receiver packet"
                                )
                            draw_entity(
                                camera=scene.camera,
                                lighting_packet=lighting_packet,
                                packet_shader=packet_shader,
                            )
                        else:
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
            if resources.sprite_items and scene.camera is not None:
                packet_shader = self._packet_lighting_shader()
                sprite_packet = (
                    self.lighting_controller.render_packet_for(
                        SPRITE_LIGHTING_RECEIVER
                    )
                    if packet_shader is not None
                    and self.lighting_controller is not None
                    else None
                )
                if packet_shader is not None and sprite_packet is None:
                    raise RuntimeError(
                        "packet lighting backend has no sprite receiver packet"
                    )
                draw_sprites_batched(
                    resources.sprite_items,
                    scene.camera,
                    scene.ground_height_at,
                    lighting=getattr(scene, "lighting", None),
                    sun_direction=getattr(scene, "sun_direction", None),
                    profiler=getattr(scene, "profiler", None),
                    static_data=True,
                    lighting_receiver=SPRITE_LIGHTING_RECEIVER,
                    lighting_packet=sprite_packet,
                    packet_shader=packet_shader,
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
