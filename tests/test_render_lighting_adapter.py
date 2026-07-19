from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.lighting_receiver import (  # noqa: E402
    LightingReceiver,
    ReceiverCompatibilityError,
)
from engine.rendering.lighting_adapter import (  # noqa: E402
    LEGACY_LOCAL_LIGHT_CAPACITY,
    LegacyLightingAdapter,
    RenderLightingAdapter,
)
from engine.rendering.lighting_state import (  # noqa: E402
    DirectionalLightSnapshot,
    LightingSnapshot,
    LocalBrightnessLight,
    PointLight,
)
from game.world.environment import (  # noqa: E402
    EnvironmentPortal,
    EnvironmentVolume,
    environment_render_snapshot,
)
from game.world.lighting_controller import StaticLightingController  # noqa: E402
from game.world.lighting_receivers import (  # noqa: E402
    CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
    CPU_BAKED_SLAB_LIGHTING_RECEIVER,
    DECAL_LIGHTING_RECEIVER,
    DYNAMIC_OBJECT_LIGHTING_RECEIVER,
    DYNAMIC_SLAB_LIGHTING_RECEIVER,
    FENCE_LIGHTING_RECEIVER,
    GROUND_LIGHTING_RECEIVER,
    PACKET_RUNTIME_LIGHTING_RECEIVERS,
    ROLLBACK_ONLY_LIGHTING_RECEIVERS,
    ROAD_LIGHTING_RECEIVER,
    SKY_CLEAR_LIGHTING_RECEIVER,
    SKY_CLOUD_LIGHTING_RECEIVER,
    SKY_SUN_LIGHTING_RECEIVER,
    SPRITE_LIGHTING_RECEIVER,
    TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
    UNTEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
)


def directional() -> DirectionalLightSnapshot:
    return DirectionalLightSnapshot(
        sun_position=(10.0, 20.0, 30.0),
        sun_target=(0.0, 0.0, 0.0),
        sun_direction=(-10.0, -20.0, -30.0),
        light_direction=(10.0, 20.0, 30.0),
        ambient=0.72,
        diffuse=0.48,
        max_factor=1.15,
        tint=(1.0, 0.9, 0.8),
    )


def local_light(index: int) -> LocalBrightnessLight:
    return LocalBrightnessLight(
        light_id=f"test:{index}",
        center=(float(index), 0.0, 0.0),
        radius=20.0,
        value=1.5,
    )


def snapshot(
    *,
    revision: int = 1,
    light_count: int = 2,
) -> LightingSnapshot:
    return LightingSnapshot(
        revision=revision,
        base_brightness=0.65,
        sky_color=(0.4, 0.6, 0.8, 1.0),
        directional=directional(),
        local_lights=tuple(local_light(index) for index in range(light_count)),
    )


class RenderLightingAdapterTests(unittest.TestCase):
    @staticmethod
    def environment_volume() -> EnvironmentVolume:
        return EnvironmentVolume(
            volume_id="test:room",
            min_x=-10.0,
            max_x=10.0,
            min_z=-20.0,
            max_z=20.0,
            indoor_factor=0.34,
            portals=(
                EnvironmentPortal(
                    portal_id="test:door",
                    kind="doorway",
                    side="north",
                    center_x=0.0,
                    center_z=20.0,
                    width=8.0,
                    depth=12.0,
                    side_fade=3.0,
                    closed_factor=0.34,
                    open_factor=1.0,
                ),
            ),
        )

    def test_new_packet_keeps_local_light_and_exposure_independent(self) -> None:
        receiver = LightingReceiver(
            receiver_id="test.local_without_exposure",
            directional=False,
            local=True,
            environment=False,
            exposure=False,
            fog=False,
            shine=False,
        )
        packet = RenderLightingAdapter().packet_for(snapshot(), receiver)

        self.assertEqual(len(packet.local_lights), 2)
        self.assertEqual(packet.exposure, 1.0)
        self.assertEqual(packet.local_light_reference, 0.65)
        self.assertEqual(packet.sky_color, (0.4, 0.6, 0.8, 1.0))
        self.assertEqual(packet.scene_directional, directional())
        self.assertIsNone(packet.directional)
        with self.assertRaises(ReceiverCompatibilityError):
            LegacyLightingAdapter().project(packet, has_normals=True)

    def test_point_reception_is_independent_from_legacy_scalar_channel(self) -> None:
        point = PointLight(
            light_id="test:point",
            position=(1.0, 2.0, 3.0),
            color=(1.0, 0.8, 0.6),
            intensity=2.0,
            range=12.0,
        )
        source = LightingSnapshot(
            revision=4,
            base_brightness=1.0,
            sky_color=(0.0, 0.0, 0.0, 1.0),
            directional=directional(),
            local_lights=(local_light(0),),
            point_lights=(point,),
        )
        packet = RenderLightingAdapter().packet_for(
            source,
            DYNAMIC_OBJECT_LIGHTING_RECEIVER,
        )

        self.assertFalse(DYNAMIC_OBJECT_LIGHTING_RECEIVER.local)
        self.assertTrue(DYNAMIC_OBJECT_LIGHTING_RECEIVER.point)
        self.assertEqual(packet.local_lights, ())
        self.assertEqual(packet.point_lights, (point,))

    def test_new_packet_is_uncapped_and_legacy_projection_reports_omissions(self) -> None:
        light_count = LEGACY_LOCAL_LIGHT_CAPACITY + 6
        packet = RenderLightingAdapter().packet_for(
            snapshot(light_count=light_count),
            GROUND_LIGHTING_RECEIVER,
        )
        projection = LegacyLightingAdapter().project(
            packet,
            has_normals=True,
        )

        self.assertEqual(len(packet.local_lights), light_count)
        self.assertEqual(len(projection.local_lights), LEGACY_LOCAL_LIGHT_CAPACITY)
        self.assertEqual(
            projection.omitted_local_light_ids,
            tuple(
                f"test:{index}"
                for index in range(LEGACY_LOCAL_LIGHT_CAPACITY, light_count)
            ),
        )

    def test_packet_cache_is_revision_and_receiver_specific(self) -> None:
        adapter = RenderLightingAdapter()
        first = adapter.packet_for(snapshot(revision=4), GROUND_LIGHTING_RECEIVER)
        repeated = adapter.packet_for(
            snapshot(revision=4),
            GROUND_LIGHTING_RECEIVER,
        )
        changed = adapter.packet_for(
            snapshot(revision=5),
            GROUND_LIGHTING_RECEIVER,
        )

        self.assertIs(first, repeated)
        self.assertIsNot(first, changed)

    def test_controller_prepares_migrated_receiver_packets_by_revision(self) -> None:
        scene = SimpleNamespace()
        controller = StaticLightingController(
            scene,
            resources=SimpleNamespace(),
            build_state=SimpleNamespace(),
        )
        first_snapshot = snapshot(revision=8)
        controller.prepare_render_packets(first_snapshot)

        self.assertEqual(
            set(controller.render_packets),
            {
                receiver.receiver_id
                for receiver in PACKET_RUNTIME_LIGHTING_RECEIVERS
            },
        )
        self.assertTrue(
            set(controller.render_packets).isdisjoint(
                receiver.receiver_id
                for receiver in ROLLBACK_ONLY_LIGHTING_RECEIVERS
            )
        )
        self.assertEqual(controller.diagnostics.render_packet_rebuilds, 1)
        controller.prepare_render_packets(first_snapshot)
        self.assertEqual(controller.diagnostics.render_packet_rebuilds, 1)
        controller.prepare_render_packets(snapshot(revision=9))
        self.assertEqual(controller.diagnostics.render_packet_rebuilds, 2)

    def test_packet_controller_rejects_rollback_only_receiver_request(self) -> None:
        scene = SimpleNamespace(lighting_backend="packet")
        controller = StaticLightingController(
            scene,
            resources=SimpleNamespace(),
            build_state=SimpleNamespace(),
        )
        controller.prepare_render_packets(snapshot(revision=8))

        with self.assertRaisesRegex(RuntimeError, "rollback-only receiver"):
            controller.render_packet_for(CPU_BAKED_OBJECT_LIGHTING_RECEIVER)

    def test_typed_environment_projection_is_immutable_and_not_legacy_owned(self) -> None:
        volume = self.environment_volume()
        closed = environment_render_snapshot([volume])
        legacy = volume.to_legacy_dict()
        legacy["factor"] = 0.99
        volume.portals[0].set_openness(1.0)
        opened = environment_render_snapshot([volume])

        self.assertEqual(closed.regions[0].indoor_factor, 0.34)
        self.assertEqual(closed.regions[0].portals[0].factor, 0.34)
        self.assertEqual(opened.regions[0].portals[0].factor, 1.0)
        self.assertNotEqual(closed, opened)

    def test_environment_snapshot_only_reaches_receivers_that_request_it(self) -> None:
        render_environment = environment_render_snapshot(
            [self.environment_volume()]
        )
        adapter = RenderLightingAdapter()
        ground_packet = adapter.packet_for(
            snapshot(),
            GROUND_LIGHTING_RECEIVER,
            render_environment,
        )
        wall_packet = adapter.packet_for(
            snapshot(),
            TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
            render_environment,
        )

        self.assertIs(ground_packet.environment, render_environment)
        self.assertIsNone(wall_packet.environment)

    def test_environment_change_rebuilds_controller_packets_at_same_light_revision(self) -> None:
        controller = StaticLightingController(
            SimpleNamespace(),
            resources=SimpleNamespace(),
            build_state=SimpleNamespace(),
        )
        lighting_snapshot = snapshot(revision=12)
        volume = self.environment_volume()
        controller.prepare_render_packets(
            lighting_snapshot,
            environment_render_snapshot([volume]),
        )
        volume.portals[0].set_openness(1.0)
        controller.prepare_render_packets(
            lighting_snapshot,
            environment_render_snapshot([volume]),
        )

        self.assertEqual(controller.diagnostics.render_packet_rebuilds, 2)


if __name__ == "__main__":
    unittest.main()
