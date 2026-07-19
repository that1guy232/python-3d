"""Texture-backed variable-length records for packet lighting GLSL 1.20."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_FLOAT,
    GL_MAX_TEXTURE_IMAGE_UNITS,
    GL_MAX_TEXTURE_SIZE,
    GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS,
    GL_NEAREST,
    GL_RGBA,
    GL_RGBA32F,
    GL_TEXTURE_2D,
    GL_TEXTURE0,
    GL_TEXTURE1,
    GL_TEXTURE2,
    GL_TEXTURE3,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    glActiveTexture,
    glBindTexture,
    glDeleteTextures,
    glGenTextures,
    glGetIntegerv,
    glTexImage2D,
    glTexParameteri,
)

from engine.rendering.lighting_adapter import ReceiverLightingPacket


LOCAL_LIGHT_TEXELS = 3
ENVIRONMENT_REGION_TEXELS = 2
ENVIRONMENT_PORTAL_TEXELS = 2
PACKET_DATA_TEXTURE_UNITS = 4


@dataclass(frozen=True, slots=True)
class PacketGpuStorageLimits:
    """Record capacities derived from the active context's texture width."""

    max_texture_size: int
    texture_image_units: int
    vertex_texture_image_units: int = 1

    @property
    def local_lights(self) -> int:
        return self.max_texture_size // LOCAL_LIGHT_TEXELS

    @property
    def environment_regions(self) -> int:
        return self.max_texture_size // ENVIRONMENT_REGION_TEXELS

    @property
    def environment_portals(self) -> int:
        return self.max_texture_size // ENVIRONMENT_PORTAL_TEXELS

    @classmethod
    def from_context(cls) -> "PacketGpuStorageLimits":
        return cls(
            max_texture_size=int(glGetIntegerv(GL_MAX_TEXTURE_SIZE)),
            texture_image_units=int(glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS)),
            vertex_texture_image_units=int(
                glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS)
            ),
        )


def _portal_side_code(side: str) -> float:
    try:
        return float(("north", "east", "south", "west").index(str(side)))
    except ValueError as exc:
        raise ValueError(f"unsupported render portal side: {side!r}") from exc


def pack_local_light_texels(packet: ReceiverLightingPacket) -> np.ndarray:
    """Encode local-light records as three RGBA32F texels each."""

    lights = packet.local_lights
    texels = np.zeros((max(1, len(lights) * LOCAL_LIGHT_TEXELS), 4), dtype=np.float32)
    for index, light in enumerate(lights):
        offset = index * LOCAL_LIGHT_TEXELS
        texels[offset] = (
            light.center[0],
            light.center[2],
            light.radius,
            light.value,
        )
        texels[offset + 1] = (
            light.falloff,
            float(light.indoor_only),
            light.floor_scale,
            float(light.bounds is not None),
        )
        if light.bounds is not None:
            texels[offset + 2] = light.bounds
    return texels


def pack_environment_texels(
    packet: ReceiverLightingPacket,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode regions and their flattened portals into RGBA32F texels."""

    environment = packet.environment
    regions = environment.regions if environment is not None else ()
    region_texels = np.zeros(
        (max(1, len(regions) * ENVIRONMENT_REGION_TEXELS), 4),
        dtype=np.float32,
    )
    portals = tuple(portal for region in regions for portal in region.portals)
    portal_texels = np.zeros(
        (max(1, len(portals) * ENVIRONMENT_PORTAL_TEXELS), 4),
        dtype=np.float32,
    )

    portal_index = 0
    for region_index, region in enumerate(regions):
        region_offset = region_index * ENVIRONMENT_REGION_TEXELS
        region_texels[region_offset] = (
            region.min_x,
            region.max_x,
            region.min_z,
            region.max_z,
        )
        region_texels[region_offset + 1, 0] = region.indoor_factor
        for portal in region.portals:
            portal_offset = portal_index * ENVIRONMENT_PORTAL_TEXELS
            portal_texels[portal_offset] = (
                float(region_index),
                _portal_side_code(portal.side),
                portal.center_x,
                portal.center_z,
            )
            portal_texels[portal_offset + 1] = (
                portal.width,
                portal.depth,
                portal.side_fade,
                portal.factor,
            )
            portal_index += 1
    return region_texels, portal_texels


def _texture_ids() -> tuple[int, int, int]:
    generated = np.asarray(glGenTextures(3)).reshape(-1)
    if generated.size != 3:
        raise RuntimeError("packet lighting could not allocate three data textures")
    return tuple(int(value) for value in generated)


@dataclass(slots=True)
class PacketGpuStorage:
    """Own and bind variable-length float textures for one packet program."""

    limits: PacketGpuStorageLimits
    texture_ids: tuple[int, int, int] = field(default_factory=_texture_ids)
    _uploaded_packet: ReceiverLightingPacket | None = field(
        default=None,
        init=False,
        repr=False,
    )
    widths: tuple[int, int, int] = field(
        default=(1, 1, 1),
        init=False,
    )

    @classmethod
    def from_context(cls) -> "PacketGpuStorage":
        limits = PacketGpuStorageLimits.from_context()
        if limits.max_texture_size < LOCAL_LIGHT_TEXELS:
            raise RuntimeError(
                "packet lighting requires a float data texture at least "
                f"{LOCAL_LIGHT_TEXELS} texels wide"
            )
        if limits.texture_image_units < PACKET_DATA_TEXTURE_UNITS:
            raise RuntimeError(
                "packet lighting requires four fragment texture units; "
                f"available={limits.texture_image_units}"
            )
        if limits.vertex_texture_image_units < 1:
            raise RuntimeError(
                "packet lighting polygon receivers require one vertex texture "
                "unit; available=0"
            )
        storage = cls(limits=limits)
        try:
            empty = np.zeros((1, 4), dtype=np.float32)
            storage._upload_arrays((empty, empty, empty))
        except Exception as exc:
            storage.dispose()
            raise RuntimeError(
                "packet lighting requires renderable RGBA32F data textures"
            ) from exc
        return storage

    def _upload_arrays(self, arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        units = (GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3)
        try:
            for unit, texture, data in zip(units, self.texture_ids, arrays):
                glActiveTexture(unit)
                glBindTexture(GL_TEXTURE_2D, texture)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GL_RGBA32F,
                    int(data.shape[0]),
                    1,
                    0,
                    GL_RGBA,
                    GL_FLOAT,
                    np.ascontiguousarray(data),
                )
        finally:
            glActiveTexture(GL_TEXTURE0)
        self.widths = tuple(int(data.shape[0]) for data in arrays)

    def upload_and_bind(self, packet: ReceiverLightingPacket) -> None:
        if self._uploaded_packet != packet:
            local = pack_local_light_texels(packet)
            regions, portals = pack_environment_texels(packet)
            arrays = (local, regions, portals)
            self._upload_arrays(arrays)
            self._uploaded_packet = packet
        else:
            try:
                for unit, texture in zip(
                    (GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3),
                    self.texture_ids,
                ):
                    glActiveTexture(unit)
                    glBindTexture(GL_TEXTURE_2D, texture)
            finally:
                glActiveTexture(GL_TEXTURE0)

    def dispose(self) -> None:
        textures = tuple(texture for texture in self.texture_ids if texture)
        if textures:
            try:
                glDeleteTextures(list(textures))
            except TypeError:
                glDeleteTextures(len(textures), list(textures))
        self.texture_ids = (0, 0, 0)
        self._uploaded_packet = None
