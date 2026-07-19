"""Packet-driven GPU lighting backend with capability-sized record textures."""

from __future__ import annotations

from dataclasses import dataclass, field

from OpenGL.GL import (
    GL_TEXTURE_2D,
    GL_TEXTURE_CUBE_MAP,
    GL_TEXTURE0,
    GL_TEXTURE4,
    GL_TEXTURE5,
    GL_TEXTURE6,
    GL_TRUE,
    glActiveTexture,
    glBindTexture,
    glGetUniformLocation,
    glUniform1f,
    glUniform1i,
    glUniform2f,
    glUniform3f,
    glUniform4f,
    glUniformMatrix4fv,
    glUseProgram,
)

from engine.rendering.gl_program import compile_program
from engine.rendering.directional_shadow import DirectionalShadowBinding
from engine.rendering.point_shadow import PointShadowBinding
from engine.lighting_receiver import LightingEvaluation, LocalLightPolicy
from engine.rendering.lighting_adapter import ReceiverLightingPacket
from engine.rendering.packet_shader_source import (
    PACKET_LIGHTING_FRAGMENT_SOURCE,
    PACKET_LIGHTING_VERTEX_SOURCE,
)
from engine.rendering.packet_gpu_storage import (
    PacketGpuStorage,
    PacketGpuStorageLimits,
)
from engine.render_style_state import (
    get_render_fog_state,
    get_render_shine_state,
    get_render_vibrance_state,
)


class PacketLightingCapacityError(RuntimeError):
    """A packet cannot be represented by the experimental GPU backend."""


class PacketLightingBackendUnavailable(RuntimeError):
    """The explicitly selected packet GPU backend could not be created."""


MAX_PACKET_POINT_LIGHTS = 16
MAX_PACKET_POINT_SHADOWS = 2


def validate_packet_capacity(
    packet: ReceiverLightingPacket,
    limits: PacketGpuStorageLimits,
) -> None:
    """Raise instead of silently dropping data required by this receiver."""

    if len(packet.local_lights) > limits.local_lights:
        overflow = packet.local_lights[limits.local_lights:]
        raise PacketLightingCapacityError(
            "packet lighting backend local-light capacity exceeded; "
            f"limit={limits.local_lights}, received={len(packet.local_lights)}, "
            f"overflow_ids={[light.light_id for light in overflow]!r}"
        )

    if len(packet.point_lights) > MAX_PACKET_POINT_LIGHTS:
        overflow = packet.point_lights[MAX_PACKET_POINT_LIGHTS:]
        raise PacketLightingCapacityError(
            "packet lighting backend point-light capacity exceeded; "
            f"limit={MAX_PACKET_POINT_LIGHTS}, "
            f"received={len(packet.point_lights)}, "
            f"overflow_ids={[light.light_id for light in overflow]!r}"
        )

    environment = packet.environment
    regions = environment.regions if environment is not None else ()
    if len(regions) > limits.environment_regions:
        overflow = regions[limits.environment_regions:]
        raise PacketLightingCapacityError(
            "packet lighting backend environment-region capacity exceeded; "
            f"limit={limits.environment_regions}, received={len(regions)}, "
            f"overflow_ids={[region.volume_id for region in overflow]!r}"
        )

    portals = tuple(
        portal
        for region in regions
        for portal in region.portals
    )
    if len(portals) > limits.environment_portals:
        overflow = portals[limits.environment_portals:]
        raise PacketLightingCapacityError(
            "packet lighting backend environment-portal capacity exceeded; "
            f"limit={limits.environment_portals}, received={len(portals)}, "
            f"overflow_ids={[portal.portal_id for portal in overflow]!r}"
        )


@dataclass(slots=True)
class PacketTextureLightingShader:
    """Separate GLSL program bound directly from one receiver packet."""

    program: int
    locations: dict[str, int]
    storage: PacketGpuStorage
    _uploaded_key: tuple | None = field(default=None, init=False, repr=False)
    directional_shadow: DirectionalShadowBinding | None = field(
        default=None,
        init=False,
        repr=False,
    )
    point_shadows: tuple[PointShadowBinding, ...] = field(
        default=(),
        init=False,
        repr=False,
    )
    active_point_light_ids: tuple[str, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def set_directional_shadow(
        self,
        shadow: DirectionalShadowBinding | None,
    ) -> None:
        """Select the completed sun shadow map used by later material draws."""

        if shadow is not None and self.storage.limits.texture_image_units < 5:
            raise PacketLightingBackendUnavailable(
                "sun shadow sampling requires five fragment texture units; "
                f"available={self.storage.limits.texture_image_units}"
            )
        if shadow != self.directional_shadow:
            self.directional_shadow = shadow
            self._uploaded_key = None

    def set_point_shadows(self, shadows) -> None:
        values = tuple(shadows or ())
        if len(values) > MAX_PACKET_POINT_SHADOWS:
            raise ValueError(
                f"at most {MAX_PACKET_POINT_SHADOWS} point shadows may be bound"
            )
        if not all(isinstance(value, PointShadowBinding) for value in values):
            raise TypeError("point shadows must be PointShadowBinding values")
        if values and self.storage.limits.texture_image_units < 7:
            raise PacketLightingBackendUnavailable(
                "point shadow sampling requires seven fragment texture units; "
                f"available={self.storage.limits.texture_image_units}"
            )
        if values != self.point_shadows:
            self.point_shadows = values
            self._uploaded_key = None

    def set_active_point_lights(self, light_ids) -> None:
        """Select the camera-relevant point-light subset for later packets."""

        values = (
            None
            if light_ids is None
            else tuple(str(value) for value in light_ids)
        )
        if values is not None and len(values) > MAX_PACKET_POINT_LIGHTS:
            raise ValueError(
                f"at most {MAX_PACKET_POINT_LIGHTS} active point lights may be selected"
            )
        if values != self.active_point_light_ids:
            self.active_point_light_ids = values
            self._uploaded_key = None

    def bind(
        self,
        packet: ReceiverLightingPacket,
        *,
        directional_normal_stream: bool = False,
    ) -> None:
        if self.active_point_light_ids is None:
            validate_packet_capacity(packet, self.storage.limits)
        glUseProgram(self.program)
        self.storage.upload_and_bind(packet)
        glActiveTexture(GL_TEXTURE0)
        self._uniform1i("u_texture", 0)
        self._uniform1i("u_local_light_records", 1)
        self._uniform1i("u_environment_region_records", 2)
        self._uniform1i("u_environment_portal_records", 3)
        self._uniform1i("u_sun_shadow_map", 4)
        self._uniform1i("u_point_shadow_map0", 5)
        self._uniform1i("u_point_shadow_map1", 6)
        self._bind_directional_shadow_texture()
        self._bind_point_shadow_textures()
        local_width, region_width, portal_width = self.storage.widths
        self._uniform1f("u_local_light_record_width", local_width)
        self._uniform1f("u_environment_region_record_width", region_width)
        self._uniform1f("u_environment_portal_record_width", portal_width)
        self._uniform1i(
            "u_directional_normal_stream",
            int(directional_normal_stream),
        )

        fog = get_render_fog_state()
        shine = get_render_shine_state()
        vibrance = get_render_vibrance_state()
        state_key = (
            packet,
            bool(directional_normal_stream),
            fog,
            shine,
            vibrance,
            self.directional_shadow,
            self.point_shadows,
            self.active_point_light_ids,
        )
        if self._uploaded_key == state_key:
            return

        receiver = packet.receiver
        dynamic = receiver.evaluation is LightingEvaluation.DYNAMIC
        self._uniform1f("u_exposure", packet.exposure)
        self._uniform1f("u_local_reference", packet.local_light_reference)
        self._uniform1i(
            "u_local_lighting_enabled",
            int(dynamic and receiver.local),
        )
        self._uniform1i(
            "u_local_point_query_policy",
            int(receiver.local_light_policy is LocalLightPolicy.POINT_QUERY),
        )
        self._uniform1i(
            "u_directional_enabled",
            int(dynamic and receiver.directional),
        )
        self._uniform1i(
            "u_clamp_directional_material",
            int(receiver.clamp_directional_material),
        )
        self._uniform1i(
            "u_clamp_lit_material",
            int(receiver.clamp_lit_material),
        )
        self._uniform1i(
            "u_environment_enabled",
            int(dynamic and packet.environment_enabled),
        )

        directional = packet.directional
        if directional is not None:
            self._uniform3f("u_light_dir", directional.light_direction)
            self._uniform1f("u_light_ambient", directional.ambient)
            self._uniform1f("u_light_diffuse", directional.diffuse)
            self._uniform1f("u_light_max_factor", directional.max_factor)

        self._upload_directional_shadow()

        self._upload_local_lights(packet)
        self._upload_point_lights(packet)
        self._upload_environment(packet)

        self._uniform1i("u_fog_enabled", int(receiver.fog and fog.enabled))
        self._uniform1f("u_fog_density", fog.density)
        self._uniform4f("u_fog_color", fog.color)
        self._uniform1f("u_vibrance", vibrance.vibrance)
        self._uniform1i("u_shine_enabled", int(receiver.shine and shine.enabled))
        self._uniform1f("u_shine_strength", shine.strength)
        self._uniform1f("u_shine_power", shine.power)
        self._uniform1f("u_shine_fresnel", shine.fresnel)
        self._uniform3f("u_shine_tint", shine.tint)
        self._uploaded_key = state_key

    def _upload_local_lights(self, packet: ReceiverLightingPacket) -> None:
        self._uniform1i("u_brightness_area_count", len(packet.local_lights))

    def _upload_point_lights(self, packet: ReceiverLightingPacket) -> None:
        point_lights = tuple(packet.point_lights)
        if self.active_point_light_ids is not None:
            by_id = {light.light_id: light for light in point_lights}
            point_lights = tuple(
                by_id[light_id]
                for light_id in self.active_point_light_ids
                if light_id in by_id
            )
        self._uniform1i("u_point_light_count", len(point_lights))
        shadow_slots = {
            shadow.light_id: index
            for index, shadow in enumerate(self.point_shadows)
        }
        for index, light in enumerate(point_lights):
            self._uniform4f(
                f"u_point_light_position_ranges[{index}]",
                (*light.position, light.range),
            )
            self._uniform4f(
                f"u_point_light_color_intensities[{index}]",
                (*light.color, light.intensity),
            )
            self._uniform1i(
                f"u_point_light_shadow_slots[{index}]",
                shadow_slots.get(
                    light.light_id,
                    -2 if light.casts_shadows else -1,
                ),
            )
        self._uniform1f(
            "u_point_shadow_bias0",
            self.point_shadows[0].bias if len(self.point_shadows) > 0 else 0.0,
        )
        self._uniform1f(
            "u_point_shadow_bias1",
            self.point_shadows[1].bias if len(self.point_shadows) > 1 else 0.0,
        )

    def _upload_environment(self, packet: ReceiverLightingPacket) -> None:
        environment = packet.environment
        regions = environment.regions if environment is not None else ()
        self._uniform1i("u_environment_region_count", len(regions))
        self._uniform1i(
            "u_environment_opening_count",
            sum(len(region.portals) for region in regions),
        )

    def _bind_directional_shadow_texture(self) -> None:
        shadow = self.directional_shadow
        try:
            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, int(shadow.texture if shadow else 0))
        finally:
            glActiveTexture(GL_TEXTURE0)

    def _bind_point_shadow_textures(self) -> None:
        units = (GL_TEXTURE5, GL_TEXTURE6)
        try:
            for index, unit in enumerate(units):
                texture = (
                    self.point_shadows[index].texture
                    if index < len(self.point_shadows)
                    else 0
                )
                glActiveTexture(unit)
                glBindTexture(GL_TEXTURE_CUBE_MAP, int(texture))
        finally:
            glActiveTexture(GL_TEXTURE0)

    def _upload_directional_shadow(self) -> None:
        shadow = self.directional_shadow
        self._uniform1i("u_sun_shadow_enabled", int(shadow is not None))
        if shadow is None:
            self._uniform2f("u_sun_shadow_texel_size", (1.0, 1.0))
            self._uniform1f("u_sun_shadow_bias", 0.0)
            self._uniform_matrix4(
                "u_sun_shadow_matrix",
                (
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ),
            )
            return
        self._uniform2f("u_sun_shadow_texel_size", shadow.texel_size)
        self._uniform1f("u_sun_shadow_bias", shadow.bias)
        self._uniform_matrix4("u_sun_shadow_matrix", shadow.light_matrix)

    def _uniform1i(self, name: str, value: int) -> None:
        location = self.locations[name]
        if location >= 0:
            glUniform1i(location, int(value))

    def _uniform1f(self, name: str, value: float) -> None:
        location = self.locations[name]
        if location >= 0:
            glUniform1f(location, float(value))

    def _uniform2f(self, name: str, value) -> None:
        location = self.locations[name]
        if location >= 0:
            glUniform2f(location, float(value[0]), float(value[1]))

    def _uniform3f(self, name: str, value) -> None:
        location = self.locations[name]
        if location >= 0:
            glUniform3f(location, float(value[0]), float(value[1]), float(value[2]))

    def _uniform4f(self, name: str, value) -> None:
        location = self.locations[name]
        if location >= 0:
            self._location_uniform4f(location, value)

    def _uniform_matrix4(self, name: str, value) -> None:
        location = self.locations[name]
        if location >= 0:
            glUniformMatrix4fv(location, 1, GL_TRUE, value)

    @staticmethod
    def _location_uniform4f(location: int, value) -> None:
        if location >= 0:
            glUniform4f(
                location,
                float(value[0]),
                float(value[1]),
                float(value[2]),
                float(value[3]),
            )


_packet_shader: PacketTextureLightingShader | None = None
_packet_shader_failed = False


def _locations(program: int, names: tuple[str, ...]) -> dict[str, int]:
    return {
        name: int(glGetUniformLocation(program, name))
        for name in names
    }


def get_packet_texture_lighting_shader() -> PacketTextureLightingShader | None:
    """Compile the replacement shader lazily; return None if unsupported."""

    global _packet_shader, _packet_shader_failed
    if _packet_shader is not None:
        return _packet_shader
    if _packet_shader_failed:
        return None

    try:
        program = compile_program(
            PACKET_LIGHTING_VERTEX_SOURCE,
            PACKET_LIGHTING_FRAGMENT_SOURCE,
        )
        scalar_names = (
            "u_texture",
            "u_exposure",
            "u_local_reference",
            "u_local_lighting_enabled",
            "u_local_point_query_policy",
            "u_directional_enabled",
            "u_directional_normal_stream",
            "u_clamp_directional_material",
            "u_clamp_lit_material",
            "u_light_dir",
            "u_light_ambient",
            "u_light_diffuse",
            "u_light_max_factor",
            "u_sun_shadow_enabled",
            "u_sun_shadow_map",
            "u_sun_shadow_texel_size",
            "u_sun_shadow_bias",
            "u_sun_shadow_matrix",
            "u_point_light_count",
            "u_point_shadow_map0",
            "u_point_shadow_map1",
            "u_point_shadow_bias0",
            "u_point_shadow_bias1",
            "u_brightness_area_count",
            "u_local_light_records",
            "u_local_light_record_width",
            "u_environment_enabled",
            "u_environment_region_count",
            "u_environment_region_records",
            "u_environment_region_record_width",
            "u_environment_opening_count",
            "u_environment_portal_records",
            "u_environment_portal_record_width",
            "u_fog_enabled",
            "u_fog_density",
            "u_fog_color",
            "u_vibrance",
            "u_shine_enabled",
            "u_shine_strength",
            "u_shine_power",
            "u_shine_fresnel",
            "u_shine_tint",
            *tuple(
                f"u_point_light_position_ranges[{index}]"
                for index in range(MAX_PACKET_POINT_LIGHTS)
            ),
            *tuple(
                f"u_point_light_color_intensities[{index}]"
                for index in range(MAX_PACKET_POINT_LIGHTS)
            ),
            *tuple(
                f"u_point_light_shadow_slots[{index}]"
                for index in range(MAX_PACKET_POINT_LIGHTS)
            ),
        )
        _packet_shader = PacketTextureLightingShader(
            program=program,
            locations=_locations(program, scalar_names),
            storage=PacketGpuStorage.from_context(),
        )
        return _packet_shader
    except Exception as exc:
        _packet_shader_failed = True
        print(f"Warning: packet lighting shader unavailable: {exc}")
        return None


def reset_packet_texture_lighting_shader() -> None:
    """Forget lazy shader state; intended for context recreation and tests."""

    global _packet_shader, _packet_shader_failed
    if _packet_shader is not None:
        try:
            _packet_shader.storage.dispose()
        except Exception:
            pass
    _packet_shader = None
    _packet_shader_failed = False
