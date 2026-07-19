"""Static lighting rebuild and shader sync policy for WorldScene."""

from __future__ import annotations

from dataclasses import dataclass

from engine.rendering.lighting_adapter import RenderLightingAdapter
from game.world.environment import environment_render_snapshot
from game.world.lighting_receivers import (
    FENCE_LIGHTING_RECEIVER,
    GROUND_LIGHTING_RECEIVER,
    PACKET_RUNTIME_LIGHTING_RECEIVERS,
    ROLLBACK_ONLY_LIGHTING_RECEIVER_IDS,
    ROAD_LIGHTING_RECEIVER,
    TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
)
from game.world.world_state import WorldBuildState, WorldRenderResources

LEGACY_LIGHTING_ALIAS_NAMES = (
    "sun_pos",
    "sun_direction",
    "brightness_modifiers",
    "covered_regions",
)

class PacketStaticGeometryError(RuntimeError):
    """Packet mode encountered static geometry owned by the rollback path."""


@dataclass(slots=True)
class LightingUpdateDiagnostics:
    """Observable update counts used by migration baselines and profiling."""

    camera_projection_rebuilds: int = 0
    legacy_alias_projections: int = 0
    uniform_sync_attempts: int = 0
    uniform_sync_cache_hits: int = 0
    shader_state_updates: int = 0
    shader_uniform_uploads: int = 0
    static_refreshes: int = 0
    ground_rebuilds: int = 0
    road_refreshes: int = 0
    wall_batch_rebuilds: int = 0
    fence_rebuilds: int = 0
    render_packet_rebuilds: int = 0


class StaticLightingController:
    """Own lighting rebuilds that still operate on scene-owned resources."""

    def __init__(
        self,
        scene,
        *,
        resources: WorldRenderResources | None = None,
        build_state: WorldBuildState | None = None,
    ) -> None:
        self.scene = scene
        self.resources = resources or getattr(scene, "render_resources", scene)
        self.build_state = build_state or getattr(scene, "build_state", scene)
        self.diagnostics = LightingUpdateDiagnostics()
        self.render_adapter = RenderLightingAdapter()
        self.render_snapshot = None
        self.render_environment_snapshot = None
        self.render_packets = {}
        self._render_packet_input_key = None
        self._legacy_bridge = None

    def _get_legacy_bridge(self):
        """Create the rollback adapter only after an explicit legacy request."""

        if self._legacy_bridge is None:
            from game.world.legacy_lighting_bridge import LegacyLightingBridge

            self._legacy_bridge = LegacyLightingBridge(
                self.scene,
                self.resources,
                self.build_state,
                self.diagnostics,
                LEGACY_LIGHTING_ALIAS_NAMES,
            )
        return self._legacy_bridge

    @property
    def legacy_bridge_instantiated(self) -> bool:
        return self._legacy_bridge is not None

    @property
    def legacy_covered_regions(self):
        if self._legacy_bridge is None:
            return ()
        return self._legacy_bridge.covered_regions

    def prepare_render_packets(self, snapshot, environment_snapshot=None) -> None:
        """Cache packets for receiver families migrated to the new backend."""

        if snapshot is None:
            self.render_snapshot = None
            self.render_environment_snapshot = None
            self.render_packets = {}
            self._render_packet_input_key = None
            return
        input_key = (int(snapshot.revision), environment_snapshot)
        self.render_snapshot = snapshot
        self.render_environment_snapshot = environment_snapshot
        self.render_packets = {
            receiver.receiver_id: self.render_adapter.packet_for(
                snapshot,
                receiver,
                environment_snapshot,
            )
            for receiver in PACKET_RUNTIME_LIGHTING_RECEIVERS
        }
        if self._render_packet_input_key != input_key:
            self.diagnostics.render_packet_rebuilds += 1
            self.render_adapter.retain_inputs(
                snapshot.revision,
                environment_snapshot,
            )
            self._render_packet_input_key = input_key

    def render_packet_for(self, receiver):
        """Return a packet for any receiver from the latest prepared snapshot."""

        receiver_id = getattr(receiver, "receiver_id", None)
        if (
            self.uses_packet_backend()
            and receiver_id in ROLLBACK_ONLY_LIGHTING_RECEIVER_IDS
        ):
            raise RuntimeError(
                "packet lighting backend cannot packetize rollback-only receiver "
                f"{receiver_id!r}"
            )
        if self.render_snapshot is None:
            return None
        packet = self.render_adapter.packet_for(
            self.render_snapshot,
            receiver,
            self.render_environment_snapshot,
        )
        self.render_packets[receiver_id] = packet
        return packet

    def sync_aliases(self):
        """Delegate rollback scene projection to the lazy legacy bridge."""

        return self._get_legacy_bridge().sync_aliases()

    def legacy_brightness_modifiers(self) -> list[dict[str, object]]:
        """Materialize local lights for rollback builders and scene aliases."""

        return self._get_legacy_bridge().legacy_brightness_modifiers()

    def clear_legacy_aliases(self) -> None:
        """Remove scene projections that packet runtime does not consume."""

        if self._legacy_bridge is not None:
            self._legacy_bridge.clear_aliases()
            return
        for name in LEGACY_LIGHTING_ALIAS_NAMES:
            if hasattr(self.scene, name):
                try:
                    delattr(self.scene, name)
                except AttributeError:
                    pass

    def set_legacy_covered_regions(self, regions) -> list[object]:
        """Install the mutable region dictionaries owned by rollback mode."""

        return self._get_legacy_bridge().set_covered_regions(regions)

    def prepare_legacy_runtime(self) -> None:
        """Materialize rollback-only scene regions and exterior-door bindings."""

        self._get_legacy_bridge().prepare_runtime()

    def activate_backend(self, backend: str) -> str:
        """Cross the explicit packet/rollback state-projection boundary."""

        backend_name = str(backend).strip().lower()
        if backend_name not in ("legacy", "packet"):
            raise ValueError("lighting backend must be 'legacy' or 'packet'")
        self.scene.lighting_backend = backend_name
        if backend_name == "legacy":
            self.prepare_legacy_runtime()
        else:
            if self._legacy_bridge is not None:
                self._legacy_bridge.deactivate()
            else:
                self.clear_legacy_aliases()
                self.invalidate_texture_lighting_cache()
        return backend_name

    def invalidate_texture_lighting_cache(self) -> None:
        self.scene._texture_lighting_sync_key = None

    def uses_packet_backend(self) -> bool:
        return getattr(self.scene, "lighting_backend", "legacy") == "packet"

    def set_brightness(self, value: float) -> float:
        """Set global brightness and refresh all baked lighting consumers."""
        scene = self.scene
        camera = scene.camera
        brightness = float(value)
        setter = getattr(camera, "set_brightness_default", None)
        if callable(setter):
            brightness = float(setter(brightness))
        else:
            camera.brightness_default = brightness
            cache = getattr(camera, "_brightness_cache", None)
            if cache is not None:
                cache.clear()

        lighting = getattr(scene, "lighting", None)
        if lighting is not None:
            lighting.set_base_brightness(brightness)

        if (
            getattr(scene, "_initialized", False)
            and getattr(scene, "_last_static_lighting_brightness", None) == brightness
        ):
            return brightness

        if getattr(scene, "_initialized", False):
            self.sync_local_lights_to_camera()
            brightness_modifiers = (
                getattr(lighting, "local_lights", ())
                if lighting is not None
                else getattr(scene, "brightness_modifiers", ())
            )
            if brightness_modifiers and self.sync_uniforms():
                if not self.uses_packet_backend():
                    self._get_legacy_bridge().apply_untextured_static_exposure_cpu(
                        brightness
                    )
                scene._last_static_lighting_brightness = brightness
            elif brightness_modifiers:
                self.refresh_static()
            else:
                self.apply_static_exposure(brightness)
                scene._last_static_lighting_brightness = brightness
        return brightness

    def sync_local_lights_to_camera(self) -> None:
        """Refresh the camera's read-only point-query projection."""

        scene = self.scene
        lighting = getattr(scene, "lighting", None)
        if lighting is None:
            return
        install = getattr(lighting, "project_local_lights_to_camera", None)
        if callable(install):
            camera = getattr(scene, "camera", None)
            install(camera)
            if camera is not None:
                self.diagnostics.camera_projection_rebuilds += 1
        if not self.uses_packet_backend():
            self.sync_aliases()

    def refresh_static(self) -> None:
        """Rebuild static VBOs whose vertex colors contain brightness."""
        scene = self.scene
        if not getattr(scene, "_initialized", False):
            return
        self.diagnostics.static_refreshes += 1

        self.sync_local_lights_to_camera()
        camera = scene.camera
        brightness = float(getattr(camera, "brightness_default", 1.0))
        lighting = getattr(scene, "lighting", None)
        packet_backend = self.uses_packet_backend()
        if lighting is not None:
            lighting.set_base_brightness(brightness)
        if packet_backend:
            self.validate_packet_static_geometry()
            if not self.sync_uniforms(compile_shader=False):
                raise PacketStaticGeometryError(
                    "packet lighting refresh has no authoritative lighting snapshot"
                )
            scene._last_static_lighting_brightness = brightness
            return
        self._get_legacy_bridge().rebuild_static(
            lighting=lighting,
            camera=camera,
            brightness=brightness,
        )
        self.sync_uniforms(compile_shader=False)
        scene._last_static_lighting_brightness = brightness

    def apply_static_exposure(self, brightness: float) -> None:
        """Apply global exposure without rebuilding static meshes."""
        exposure = float(brightness)
        if self.sync_uniforms(base_brightness=exposure):
            if not self.uses_packet_backend():
                self._get_legacy_bridge().apply_untextured_static_exposure_cpu(
                    exposure
                )
            return

        if self.uses_packet_backend():
            raise PacketStaticGeometryError(
                "packet lighting exposure update has no authoritative snapshot"
            )
        self._get_legacy_bridge().apply_static_exposure_cpu(exposure)

    def sync_uniforms(
        self,
        *,
        base_brightness: float | None = None,
        compile_shader: bool = True,
    ) -> bool:
        self.diagnostics.uniform_sync_attempts += 1
        scene = self.scene
        camera = getattr(scene, "camera", None)
        lighting = getattr(scene, "lighting", None)
        brightness = (
            float(base_brightness)
            if base_brightness is not None
            else float(getattr(camera, "brightness_default", 1.0))
        )
        snapshot = None
        packet_backend = self.uses_packet_backend()
        if lighting is not None:
            lighting.set_base_brightness(brightness)
            make_snapshot = getattr(lighting, "snapshot", None)
            if callable(make_snapshot):
                snapshot = make_snapshot()
            if not packet_backend:
                self.sync_aliases()
        environment_snapshot = (
            None
            if packet_backend
            else environment_render_snapshot(
                getattr(scene, "environment_volumes", ())
            )
        )
        self.prepare_render_packets(snapshot, environment_snapshot)
        if packet_backend:
            return snapshot is not None
        return self._get_legacy_bridge().sync_shader(
            brightness=brightness,
            lighting=lighting,
            snapshot=snapshot,
            camera=camera,
            compile_shader=compile_shader,
        )

    @staticmethod
    def uses_texture_shader(obj) -> bool:
        return getattr(obj, "texture", None) is not None

    @classmethod
    def uses_dynamic_textured_geometry(cls, obj) -> bool:
        mesh = getattr(obj, "_mesh", None) or obj
        return cls.uses_texture_shader(mesh) and int(
            getattr(mesh, "vertex_width", 0)
        ) >= 11

    @staticmethod
    def _receiver_id(obj) -> str | None:
        mesh = getattr(obj, "_mesh", None) or obj
        receiver = getattr(mesh, "lighting_receiver", None)
        return getattr(receiver, "receiver_id", None)

    def packet_static_geometry_violations(self) -> tuple[str, ...]:
        """Describe packet static resources that still depend on rollback data."""

        resources = self.resources
        expected: list[tuple[str, object, str]] = []
        ground = getattr(resources, "ground_mesh", None)
        if ground is not None:
            expected.append(("ground", ground, GROUND_LIGHTING_RECEIVER.receiver_id))

        for index, road_batch in enumerate(
            getattr(resources, "road_batches", ()) or ()
        ):
            meshes = tuple(getattr(road_batch, "_meshes", ()) or ())
            if not meshes:
                meshes = (road_batch,)
            for mesh_index, road in enumerate(meshes):
                expected.append(
                    (
                        f"road_batch[{index}].mesh[{mesh_index}]",
                        road,
                        ROAD_LIGHTING_RECEIVER.receiver_id,
                    )
                )
        for index, wall in enumerate(
            getattr(resources, "wall_tile_batches", ()) or ()
        ):
            expected.append(
                (
                    f"wall_batch[{index}]",
                    wall,
                    TEXTURED_STATIC_WALL_LIGHTING_RECEIVER.receiver_id,
                )
            )
        for index, fence in enumerate(getattr(resources, "fence_meshes", ()) or ()):
            expected.append(
                (f"fence[{index}]", fence, FENCE_LIGHTING_RECEIVER.receiver_id)
            )

        seen_roads: set[int] = set()
        for index, road in enumerate(self.packet_road_resources()):
            if road is None or id(road) in seen_roads:
                continue
            seen_roads.add(id(road))
            expected.append(
                (f"road[{index}]", road, ROAD_LIGHTING_RECEIVER.receiver_id)
            )

        violations = []
        for label, value, expected_receiver_id in expected:
            mesh = getattr(value, "_mesh", None) or value
            if not self.uses_dynamic_textured_geometry(mesh):
                violations.append(f"{label}:rollback-shaped")
                continue
            actual_receiver_id = self._receiver_id(mesh)
            if actual_receiver_id != expected_receiver_id:
                violations.append(
                    f"{label}:receiver={actual_receiver_id!r},"
                    f"expected={expected_receiver_id!r}"
                )
        return tuple(violations)

    def validate_packet_static_geometry(self) -> None:
        violations = self.packet_static_geometry_violations()
        if violations:
            raise PacketStaticGeometryError(
                "packet lighting cannot refresh rollback-only static geometry; "
                f"violations={list(violations)!r}"
            )

    def packet_road_resources(self):
        """Enumerate roads that packet static validation must inspect."""

        return [
            getattr(self.resources, "road", None),
            *(getattr(self.build_state, "roads", ()) or ()),
            *(
                obj
                for obj in (getattr(self.resources, "others", ()) or ())
                if hasattr(obj, "refresh_lighting") or hasattr(obj, "set_exposure")
            ),
        ]
