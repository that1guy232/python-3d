"""Reusable packet lighting and shadows for independently authored scenes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace

from engine.rendering.lighting.receiver import LightingReceiver
from engine.rendering.lighting.adapter import RenderLightingAdapter
from engine.rendering.shaders.packet import (
    PacketLightingBackendUnavailable,
    get_packet_texture_lighting_shader,
)
from engine.rendering.shadows.renderer import SceneShadowRenderer


class SceneLightingRenderer:
    """Connect a SceneLighting model to meshes, props, sprites, and shadows.

    Register the scene's receiver policies at construction, then call
    begin_frame(camera=..., lighting=...) before drawing. Pass batch_draw_args()
    to mesh batches and draw_args(receiver) to individual packet-aware objects.
    set_enabled(False) displays the original material colors without changing
    the authored lights or rebuilding geometry. Construction needs no GL context.
    """

    def __init__(
        self,
        *,
        receivers: Iterable[LightingReceiver] = (),
        shadow_sources: Callable[[], Iterable[object]] = tuple,
        view_distance: float = 2000.0,
        enabled: bool = True,
    ) -> None:
        self.receivers = tuple(receivers)
        self._enabled = bool(enabled)
        self.render_adapter = RenderLightingAdapter()
        self.render_snapshot = None
        self.render_environment_snapshot = None
        self.render_packets = {}
        self._render_packet_input_key = None
        self.shadows = SceneShadowRenderer(
            sources=shadow_sources, view_distance=view_distance
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> bool:
        """Switch between scene lighting and unlit material inspection."""
        enabled = bool(enabled)
        if self._enabled != enabled:
            self._enabled = enabled
            self._render_packet_input_key = None
            self.render_packets.clear()
        return self._enabled

    def toggle(self) -> bool:
        return self.set_enabled(not self.enabled)

    @property
    def shader(self):
        shader = get_packet_texture_lighting_shader()
        if shader is None:
            raise PacketLightingBackendUnavailable(
                "scene lighting requires the packet lighting shader"
            )
        return shader

    def prepare_render_packets(self, snapshot, environment_snapshot=None) -> bool:
        """Prepare immutable inputs; return whether the packet cache changed."""
        if snapshot is None:
            changed = self.render_snapshot is not None
            self.render_snapshot = None
            self.render_environment_snapshot = None
            self.render_packets.clear()
            self.render_adapter = RenderLightingAdapter()
            self._render_packet_input_key = None
            return changed
        # Include values, not only a scene-local revision: mutable sun settings
        # and a different level can legitimately have the same revision number.
        input_key = (snapshot, environment_snapshot, self.enabled)
        if input_key == self._render_packet_input_key:
            return False
        self.render_snapshot = snapshot
        self.render_environment_snapshot = environment_snapshot
        self.render_adapter = RenderLightingAdapter()
        self.render_packets = {}
        for receiver in self.receivers:
            self.render_packet_for(receiver)
        self._render_packet_input_key = input_key
        return True

    def render_packet_for(self, receiver: LightingReceiver):
        if self.render_snapshot is None:
            return None
        if not self.enabled:
            receiver = replace(
                receiver,
                directional=False,
                local=False,
                point=False,
                environment=False,
                exposure=False,
                shine=False,
                receives_sun_shadows=False,
            )
        packet = self.render_adapter.packet_for(
            self.render_snapshot, receiver, self.render_environment_snapshot
        )
        self.render_packets[receiver.receiver_id] = packet
        return packet

    def begin_frame(
        self, *, camera, lighting, environment_snapshot=None, profiler=None
    ) -> None:
        """Prepare this scene's lighting and bind only its own shadow maps."""
        self.prepare_render_packets(lighting.snapshot(), environment_snapshot)
        self.shadows.render(
            camera=camera,
            lighting=lighting if self.enabled else None,
            packet_shader=self.shader,
            profiler=profiler,
        )

    def batch_draw_args(self) -> dict:
        """Keyword arguments shared by BatchedMesh and world-object batches."""
        if self.render_snapshot is None:
            raise RuntimeError("call begin_frame before drawing scene lighting")
        return {"lighting_packets": self.render_packets, "packet_shader": self.shader}

    def draw_args(self, receiver: LightingReceiver) -> dict:
        packet = self.render_packet_for(receiver)
        if packet is None:
            raise RuntimeError("call begin_frame before drawing scene lighting")
        return {"lighting_packet": packet, "packet_shader": self.shader}

    def dispose(self) -> None:
        """Release scene-owned shadow maps while the GL context still exists."""
        self.shadows.dispose()
        self.prepare_render_packets(None)
