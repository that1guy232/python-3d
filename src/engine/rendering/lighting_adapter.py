"""Backend-neutral receiver lighting packets and legacy projections."""

from __future__ import annotations

from dataclasses import dataclass

from engine.lighting_receiver import LightingReceiver, ReceiverShaderFlags
from engine.rendering.lighting_state import (
    DirectionalLightSnapshot,
    LightingSnapshot,
    LocalBrightnessLight,
    PointLight,
)
from engine.rendering.render_environment import RenderEnvironmentSnapshot


LEGACY_LOCAL_LIGHT_CAPACITY = 64


@dataclass(frozen=True, slots=True)
class ReceiverLightingPacket:
    """Uncapped lighting input for one receiver and one scene revision."""

    source_revision: int
    receiver: LightingReceiver
    scene_directional: DirectionalLightSnapshot
    sky_color: tuple[float, float, float, float]
    directional: DirectionalLightSnapshot | None
    local_lights: tuple[LocalBrightnessLight, ...]
    point_lights: tuple[PointLight, ...]
    environment_enabled: bool
    environment: RenderEnvironmentSnapshot | None
    exposure: float
    local_light_reference: float
    fog_enabled: bool
    shine_enabled: bool


@dataclass(frozen=True, slots=True)
class LegacyLightingProjection:
    """The subset of a receiver packet representable by the old shader."""

    source_revision: int
    receiver_id: str
    shader_flags: ReceiverShaderFlags
    base_brightness: float
    directional: DirectionalLightSnapshot | None
    local_lights: tuple[LocalBrightnessLight, ...]
    omitted_local_light_ids: tuple[str, ...]


class RenderLightingAdapter:
    """Build immutable backend inputs without legacy shader coupling or caps."""

    def __init__(self) -> None:
        self._cache: dict[
            tuple[int, LightingReceiver, RenderEnvironmentSnapshot | None],
            ReceiverLightingPacket,
        ] = {}

    def packet_for(
        self,
        snapshot: LightingSnapshot,
        receiver: LightingReceiver,
        environment: RenderEnvironmentSnapshot | None = None,
    ) -> ReceiverLightingPacket:
        receiver_environment = environment if receiver.environment else None
        key = (int(snapshot.revision), receiver, receiver_environment)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        point_lights = tuple(snapshot.point_lights) if receiver.point else ()
        point_light_ids = {light.light_id for light in point_lights}
        packet = ReceiverLightingPacket(
            source_revision=int(snapshot.revision),
            receiver=receiver,
            scene_directional=snapshot.directional,
            sky_color=tuple(float(value) for value in snapshot.sky_color),
            directional=(
                snapshot.directional if receiver.directional else None
            ),
            local_lights=(
                tuple(
                    light
                    for light in snapshot.local_lights
                    if light.light_id not in point_light_ids
                )
                if receiver.local
                else ()
            ),
            point_lights=point_lights,
            environment_enabled=receiver.environment,
            environment=receiver_environment,
            exposure=(
                float(snapshot.base_brightness) if receiver.exposure else 1.0
            ),
            local_light_reference=float(snapshot.base_brightness),
            fog_enabled=receiver.fog,
            shine_enabled=receiver.shine,
        )
        self._cache[key] = packet
        return packet

    def retain_inputs(
        self,
        revision: int,
        environment: RenderEnvironmentSnapshot | None,
    ) -> None:
        """Keep only packets matching the controller's current frame inputs."""

        current_revision = int(revision)
        self._cache = {
            key: packet
            for key, packet in self._cache.items()
            if key[0] == current_revision
            and (key[2] is None or key[2] == environment)
        }


class LegacyLightingAdapter:
    """Project a backend-neutral packet into current GLSL 1.20 constraints."""

    def __init__(
        self,
        *,
        local_light_capacity: int = LEGACY_LOCAL_LIGHT_CAPACITY,
    ) -> None:
        if int(local_light_capacity) < 0:
            raise ValueError("legacy local-light capacity cannot be negative")
        self.local_light_capacity = int(local_light_capacity)

    def project(
        self,
        packet: ReceiverLightingPacket,
        *,
        has_normals: bool,
    ) -> LegacyLightingProjection:
        flags = packet.receiver.compatibility_shader_flags(
            has_normals=has_normals
        )
        represented = packet.local_lights[: self.local_light_capacity]
        omitted = packet.local_lights[self.local_light_capacity :]
        return LegacyLightingProjection(
            source_revision=packet.source_revision,
            receiver_id=packet.receiver.receiver_id,
            shader_flags=flags,
            base_brightness=packet.exposure,
            directional=packet.directional if flags.directional else None,
            local_lights=tuple(represented) if flags.scene_lighting else (),
            omitted_local_light_ids=tuple(
                light.light_id for light in omitted
            ),
        )
