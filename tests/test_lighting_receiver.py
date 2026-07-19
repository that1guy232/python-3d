from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.core.mesh import BatchedMesh  # noqa: E402
from engine.lighting_receiver import (  # noqa: E402
    LightingEvaluation,
    LightingReceiver,
    LocalLightPolicy,
    ReceiverCompatibilityError,
    ReceiverShaderFlags,
)
from game.world.lighting_receivers import (  # noqa: E402
    ALL_WORLD_LIGHTING_RECEIVERS,
    CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
    CPU_BAKED_SLAB_LIGHTING_RECEIVER,
    DECAL_LIGHTING_RECEIVER,
    DYNAMIC_OBJECT_LIGHTING_RECEIVER,
    DYNAMIC_POLYGON_LIGHTING_RECEIVER,
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
from game.world.objects.ground import TexturedGroundGridBuilder  # noqa: E402
from game.world.objects.wall_tile import build_wall_tile_batches  # noqa: E402


class LightingReceiverContractTests(unittest.TestCase):
    def test_contract_is_immutable_and_requires_stable_identity(self) -> None:
        with self.assertRaises(ValueError):
            LightingReceiver(
                receiver_id="",
                directional=True,
                local=True,
                environment=True,
                exposure=True,
                fog=True,
                shine=True,
            )

        with self.assertRaises(FrozenInstanceError):
            GROUND_LIGHTING_RECEIVER.environment = False

        with self.assertRaises(TypeError):
            LightingReceiver(
                receiver_id="test.invalid_evaluation",
                directional=True,
                local=True,
                environment=True,
                exposure=True,
                fog=True,
                shine=True,
                evaluation="dynamic",
            )

    def test_contract_projects_all_supported_shader_channels(self) -> None:
        self.assertEqual(
            GROUND_LIGHTING_RECEIVER.compatibility_shader_flags(
                has_normals=True
            ),
            ReceiverShaderFlags(
                scene_lighting=True,
                directional=True,
                environment=True,
                fog=True,
                shine=True,
            ),
        )
        self.assertEqual(
            GROUND_LIGHTING_RECEIVER.compatibility_shader_flags(
                has_normals=False
            ),
            ReceiverShaderFlags(
                scene_lighting=False,
                directional=False,
                environment=False,
                fog=True,
                shine=False,
            ),
        )

    def test_legacy_shader_rejects_split_local_and_exposure_channels(self) -> None:
        receiver = LightingReceiver(
            receiver_id="test.split_channels",
            directional=False,
            local=True,
            environment=False,
            exposure=False,
            fog=True,
            shine=False,
        )

        with self.assertRaises(ReceiverCompatibilityError):
            receiver.compatibility_shader_flags(has_normals=True)

    def test_polygon_point_query_policy_is_packet_only(self) -> None:
        self.assertIs(
            DYNAMIC_POLYGON_LIGHTING_RECEIVER.local_light_policy,
            LocalLightPolicy.POINT_QUERY,
        )
        self.assertTrue(DYNAMIC_POLYGON_LIGHTING_RECEIVER.clamp_lit_material)
        self.assertFalse(
            DYNAMIC_POLYGON_LIGHTING_RECEIVER.clamp_directional_material
        )
        with self.assertRaisesRegex(
            ReceiverCompatibilityError,
            "only supports surface evaluation",
        ):
            DYNAMIC_POLYGON_LIGHTING_RECEIVER.compatibility_shader_flags(
                has_normals=True
            )

    def test_cpu_baked_receiver_declares_channels_without_double_lighting(self) -> None:
        self.assertEqual(
            UNTEXTURED_STATIC_WALL_LIGHTING_RECEIVER.compatibility_shader_flags(
                has_normals=True
            ),
            ReceiverShaderFlags(
                scene_lighting=False,
                directional=False,
                environment=False,
                fog=True,
                shine=False,
            ),
        )
        self.assertIs(
            UNTEXTURED_STATIC_WALL_LIGHTING_RECEIVER.evaluation,
            LightingEvaluation.CPU_BAKED,
        )
        self.assertEqual(
            CPU_BAKED_SLAB_LIGHTING_RECEIVER.compatibility_shader_flags(
                has_normals=True
            ),
            ReceiverShaderFlags(
                scene_lighting=False,
                directional=False,
                environment=False,
                fog=True,
                shine=True,
            ),
        )

    def test_explicit_mesh_contract_overrides_legacy_policy_flags(self) -> None:
        mesh = BatchedMesh(
            1,
            3,
            texture=1,
            vertex_width=11,
            shader_lighting=False,
            environment_lighting=False,
            shine_enabled=False,
            lighting_receiver=GROUND_LIGHTING_RECEIVER,
            owns_vbo=False,
        )

        self.assertEqual(
            BatchedMesh._prepared_draw_key(mesh, object()),
            (False, True, False, True, True, True, True, True),
        )

    def test_initial_world_profiles_preserve_ground_and_wall_policies(self) -> None:
        self.assertEqual(GROUND_LIGHTING_RECEIVER.receiver_id, "world.ground")
        self.assertTrue(GROUND_LIGHTING_RECEIVER.environment)
        self.assertTrue(TEXTURED_STATIC_WALL_LIGHTING_RECEIVER.local)
        self.assertFalse(TEXTURED_STATIC_WALL_LIGHTING_RECEIVER.environment)
        self.assertTrue(UNTEXTURED_STATIC_WALL_LIGHTING_RECEIVER.directional)
        self.assertFalse(UNTEXTURED_STATIC_WALL_LIGHTING_RECEIVER.local)
        self.assertTrue(UNTEXTURED_STATIC_WALL_LIGHTING_RECEIVER.exposure)
        self.assertFalse(UNTEXTURED_STATIC_WALL_LIGHTING_RECEIVER.shine)

    def test_world_receiver_matrix_has_stable_unique_identities(self) -> None:
        receiver_ids = [
            receiver.receiver_id for receiver in ALL_WORLD_LIGHTING_RECEIVERS
        ]
        self.assertEqual(len(receiver_ids), len(set(receiver_ids)))
        self.assertTrue(all(receiver_id.startswith("world.") for receiver_id in receiver_ids))

    def test_packet_runtime_and_rollback_receiver_sets_are_disjoint(self) -> None:
        packet_ids = {
            receiver.receiver_id for receiver in PACKET_RUNTIME_LIGHTING_RECEIVERS
        }
        rollback_ids = {
            receiver.receiver_id for receiver in ROLLBACK_ONLY_LIGHTING_RECEIVERS
        }

        self.assertTrue(packet_ids.isdisjoint(rollback_ids))
        self.assertEqual(
            packet_ids | rollback_ids,
            {
                receiver.receiver_id for receiver in ALL_WORLD_LIGHTING_RECEIVERS
            },
        )
        self.assertTrue(
            all(
                receiver.evaluation is not LightingEvaluation.CPU_BAKED
                for receiver in PACKET_RUNTIME_LIGHTING_RECEIVERS
            )
        )
        self.assertTrue(
            all(
                receiver.evaluation is LightingEvaluation.CPU_BAKED
                for receiver in ROLLBACK_ONLY_LIGHTING_RECEIVERS
            )
        )

    def test_ground_builder_attaches_explicit_receiver(self) -> None:
        builder = TexturedGroundGridBuilder(
            count=0,
            tile_size=10.0,
            gap=0.0,
            texture=1,
            dynamic_lighting=False,
        )
        sentinel = object()
        with (
            patch.object(builder, "_load_heightmap", return_value=np.zeros((1, 1))),
            patch.object(builder, "_build_terrain_flatten_pads", return_value=[]),
            patch(
                "game.world.objects.ground.BatchedMesh.from_vertex_data",
                return_value=sentinel,
            ) as create_mesh,
        ):
            result = builder.build()

        self.assertIs(result, sentinel)
        self.assertIs(
            create_mesh.call_args.kwargs["lighting_receiver"],
            GROUND_LIGHTING_RECEIVER,
        )
        self.assertFalse(create_mesh.call_args.kwargs["casts_sun_shadows"])

    def test_wall_batches_choose_explicit_textured_and_untextured_profiles(self) -> None:
        textured = SimpleNamespace(texture=7)
        untextured = SimpleNamespace(texture=0)
        vertex_data = np.zeros((3, 11), dtype=np.float32)
        sentinels = [object(), object()]
        with (
            patch(
                "game.world.objects.wall_tile._tile_vertex_data",
                return_value=vertex_data,
            ),
            patch("game.world.objects.wall_tile.glBindTexture"),
            patch("game.world.objects.wall_tile.glTexParameteri"),
            patch(
                "game.world.objects.wall_tile.BatchedMesh.from_vertex_data",
                side_effect=sentinels,
            ) as create_mesh,
        ):
            result = build_wall_tile_batches(
                [textured, untextured],
                dynamic_lighting=True,
            )

        self.assertEqual(result, sentinels)
        receivers = [
            call.kwargs["lighting_receiver"]
            for call in create_mesh.call_args_list
        ]
        self.assertEqual(
            receivers,
            [
                TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
                UNTEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
            ],
        )


if __name__ == "__main__":
    unittest.main()
