from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pygame.math import Vector3


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.rendering.sprite import WorldSprite  # noqa: E402
from engine.rendering.sky_renderer import SkyRenderer  # noqa: E402
from game.world.lighting_receivers import (  # noqa: E402
    DYNAMIC_OBJECT_LIGHTING_RECEIVER,
    SKY_CLEAR_LIGHTING_RECEIVER,
    SKY_CLOUD_LIGHTING_RECEIVER,
    SKY_SUN_LIGHTING_RECEIVER,
    SPRITE_LIGHTING_RECEIVER,
)
from game.world.objects.chest import Chest  # noqa: E402
from game.world.world_renderer import WorldRenderer  # noqa: E402


class LightingReceiverOwnershipTests(unittest.TestCase):
    def test_chest_declares_explicit_packet_receiver(self) -> None:
        self.assertIs(
            Chest.packet_lighting_receiver,
            DYNAMIC_OBJECT_LIGHTING_RECEIVER,
        )

    def test_every_world_mesh_upload_has_an_explicit_receiver(self) -> None:
        source_paths = list((ROOT / "src/game/world/objects").rglob("*.py"))
        source_paths.extend(
            [
                ROOT / "src/engine/rendering/decal.py",
                ROOT / "src/engine/rendering/decal_batch.py",
            ]
        )
        uploads: list[str] = []
        missing: list[str] = []
        for source_path in source_paths:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if not (
                    isinstance(function, ast.Attribute)
                    and function.attr == "from_vertex_data"
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "BatchedMesh"
                ):
                    continue
                location = f"{source_path.relative_to(ROOT)}:{node.lineno}"
                uploads.append(location)
                if not any(
                    keyword.arg == "lighting_receiver"
                    for keyword in node.keywords
                ):
                    missing.append(location)

        self.assertGreaterEqual(len(uploads), 14)
        self.assertEqual(missing, [])

    def test_world_sprite_defaults_to_the_explicit_sprite_contract(self) -> None:
        sprite = WorldSprite(
            position=Vector3(),
            size=(1.0, 1.0),
            texture=1,
            camera=SimpleNamespace(),
        )
        self.assertIs(sprite.lighting_receiver, SPRITE_LIGHTING_RECEIVER)

    def test_world_renderer_routes_explicit_sky_receivers(self) -> None:
        sky = Mock()
        clear_packet = SimpleNamespace(
            sky_color=(0.8, 0.6, 0.4, 1.0),
            exposure=0.25,
        )
        sun_packet = object()
        cloud_packet = object()
        packets = {
            SKY_CLEAR_LIGHTING_RECEIVER.receiver_id: clear_packet,
            SKY_SUN_LIGHTING_RECEIVER.receiver_id: sun_packet,
            SKY_CLOUD_LIGHTING_RECEIVER.receiver_id: cloud_packet,
        }
        renderer = WorldRenderer.__new__(WorldRenderer)
        renderer.scene = SimpleNamespace(
            camera=SimpleNamespace(brightness_default=0.5),
            lighting=SimpleNamespace(
                sky_color=(0.4, 0.6, 0.8),
                sun_direction=Vector3(0.0, -1.0, 0.0),
            ),
        )
        renderer.resources = SimpleNamespace(sky=sky)
        renderer.lighting_controller = Mock()
        renderer.lighting_controller.render_packet_for.side_effect = (
            lambda receiver: packets[receiver.receiver_id]
        )

        self.assertIs(SKY_CLEAR_LIGHTING_RECEIVER.exposure, True)
        self.assertEqual(renderer._sky_rgba(), [0.2, 0.15, 0.1, 1.0])
        renderer.draw_sky()

        kwargs = sky.draw.call_args.kwargs
        self.assertIs(kwargs["sun_receiver"], SKY_SUN_LIGHTING_RECEIVER)
        self.assertIs(kwargs["cloud_receiver"], SKY_CLOUD_LIGHTING_RECEIVER)
        self.assertIs(kwargs["sun_packet"], sun_packet)
        self.assertIs(kwargs["cloud_packet"], cloud_packet)

    def test_sky_renderer_uses_packet_exposure_direction_and_tint(self) -> None:
        renderer = SkyRenderer.__new__(SkyRenderer)
        renderer._sun_tex = 3
        renderer._clouds = Mock()
        renderer.sun_half_size = 6000.0
        scene_directional = SimpleNamespace(
            sun_direction=(-1.0, -1.0, -1.0),
            tint=(1.0, 0.8, 0.6),
        )
        sun_packet = SimpleNamespace(
            exposure=0.5,
            scene_directional=scene_directional,
        )
        cloud_packet = SimpleNamespace(
            exposure=0.4,
            scene_directional=scene_directional,
        )
        camera = SimpleNamespace(
            brightness_default=0.1,
            rotation=Vector3(),
        )

        with (
            patch("engine.rendering.sky_renderer.glDisable"),
            patch("engine.rendering.sky_renderer.glEnable"),
            patch("engine.rendering.sky_renderer.glBlendFunc"),
            patch("engine.rendering.sky_renderer.glPushMatrix"),
            patch("engine.rendering.sky_renderer.glPopMatrix"),
            patch("engine.rendering.sky_renderer.glRotatef"),
            patch("engine.rendering.sky_renderer.glBindTexture"),
            patch("engine.rendering.sky_renderer.glColor4f"),
            patch.object(renderer, "_draw_sky_quad") as draw_quad,
        ):
            renderer.draw(
                camera,
                lighting=SimpleNamespace(
                    sun_direction=Vector3(0.0, -1.0, 0.0),
                    sun_tint=(0.1, 0.1, 0.1),
                ),
                sun_packet=sun_packet,
                cloud_packet=cloud_packet,
            )

        self.assertEqual(
            draw_quad.call_args.kwargs["color"],
            (0.5, 0.4, 0.3, 1.0),
        )
        cloud_kwargs = renderer._clouds.draw.call_args.kwargs
        self.assertEqual(cloud_kwargs["brightness"], 0.4)
        self.assertEqual(cloud_kwargs["sun_tint"], (1.0, 0.8, 0.6))
        self.assertIsNone(cloud_kwargs["lighting"])


if __name__ == "__main__":
    unittest.main()
