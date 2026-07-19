from __future__ import annotations

import ast
import os
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pygame  # noqa: E402
from pygame.math import Vector3  # noqa: E402
from OpenGL.GL import (  # noqa: E402
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGB,
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glClear,
    glClearColor,
    glColor4f,
    glDisable,
    glEnd,
    glFinish,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glNormal3f,
    glReadPixels,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTranslatef,
    glVertex3f,
    glViewport,
)

from engine.core.mesh import BatchedMesh  # noqa: E402
from engine.rendering.lighting_adapter import RenderLightingAdapter  # noqa: E402
from engine.rendering.lighting_state import (  # noqa: E402
    DirectionalLightSnapshot,
    LightingSnapshot,
    LocalBrightnessLight,
    PointLight,
)
from engine.rendering.packet_shader import (  # noqa: E402
    PACKET_LIGHTING_FRAGMENT_SOURCE,
    PacketLightingCapacityError,
    get_packet_texture_lighting_shader,
    reset_packet_texture_lighting_shader,
    validate_packet_capacity,
)
from engine.rendering.packet_gpu_storage import (  # noqa: E402
    PacketGpuStorage,
    PacketGpuStorageLimits,
    pack_environment_texels,
    pack_local_light_texels,
)
from engine.rendering.render_environment import (  # noqa: E402
    RenderEnvironmentPortal,
    RenderEnvironmentRegion,
    RenderEnvironmentSnapshot,
)
from engine.rendering.sprite import WorldSprite  # noqa: E402
from game.world.lighting_receivers import GROUND_LIGHTING_RECEIVER  # noqa: E402
from game.world.lighting_receivers import (  # noqa: E402
    CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
    CPU_BAKED_SLAB_LIGHTING_RECEIVER,
    DECAL_LIGHTING_RECEIVER,
    DYNAMIC_OBJECT_LIGHTING_RECEIVER,
    DYNAMIC_POLYGON_LIGHTING_RECEIVER,
    DYNAMIC_SLAB_LIGHTING_RECEIVER,
    FENCE_LIGHTING_RECEIVER,
    ROAD_LIGHTING_RECEIVER,
    SPRITE_LIGHTING_RECEIVER,
    TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
)
from game.world.objects.road import Road, RoadRenderBatch  # noqa: E402
from game.world.world_renderer import WorldRenderer  # noqa: E402
from game.world.worldscene import WorldScene  # noqa: E402


def _directional() -> DirectionalLightSnapshot:
    return DirectionalLightSnapshot(
        sun_position=(0.0, 1.0, 0.0),
        sun_target=(0.0, 0.0, 0.0),
        sun_direction=(0.0, -1.0, 0.0),
        light_direction=(0.0, 1.0, 0.0),
        ambient=1.0,
        diffuse=0.0,
        max_factor=1.0,
        tint=(1.0, 1.0, 1.0),
    )


def _snapshot(local_lights=()) -> LightingSnapshot:
    return LightingSnapshot(
        revision=1,
        base_brightness=0.5,
        sky_color=(0.7, 0.8, 1.0, 1.0),
        directional=_directional(),
        local_lights=tuple(local_lights),
    )


def _region(index: int, portals=()) -> RenderEnvironmentRegion:
    return RenderEnvironmentRegion(
        volume_id=f"volume:{index}",
        min_x=-0.8,
        max_x=0.8,
        min_z=-0.8,
        max_z=0.8,
        indoor_factor=0.34,
        portals=tuple(portals),
    )


class PacketShaderContractTests(unittest.TestCase):
    def test_packet_controller_contains_no_cpu_baked_rebuild_policy(self) -> None:
        controller_source = (
            ROOT / "src/game/world/lighting_controller.py"
        ).read_text(encoding="utf-8")
        bridge_source = (
            ROOT / "src/game/world/legacy_lighting_bridge.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(
            "from game.world import world_builder",
            controller_source,
        )
        self.assertNotIn("build_wall_tile_batches", controller_source)
        for method_name in (
            "apply_untextured_static_exposure_cpu",
            "apply_static_exposure_cpu",
            "road_lighting_candidates",
            "dispose_renderable",
            "dispose_renderable_batches",
        ):
            self.assertNotIn(f"def {method_name}", controller_source)
            self.assertIn(f"def {method_name}", bridge_source)

    def test_runtime_modules_do_not_eagerly_import_compat_shader(self) -> None:
        eager_imports = []
        for path in sorted((ROOT / "src").rglob("*.py")):
            if path.name == "compat_shader.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and (
                    node.module == "engine.core.compat_shader"
                ):
                    eager_imports.append(f"{path.relative_to(ROOT)}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    if any(
                        alias.name == "engine.core.compat_shader"
                        for alias in node.names
                    ):
                        eager_imports.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}"
                        )

        self.assertEqual(eager_imports, [])

    def test_packet_program_source_and_compiler_are_not_legacy_owned(self) -> None:
        packet_module = (
            ROOT / "src/engine/rendering/packet_shader.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("_TEXTURE_COLOR_EXPOSURE", packet_module)
        self.assertNotIn("_compile_program", packet_module)
        self.assertNotIn("engine.core.compat_shader", packet_module)
        self.assertIn("packet_shader_source", packet_module)
        self.assertIn("gl_program", packet_module)
        self.assertIn("packet_gpu_storage", packet_module)
        self.assertIn("u_local_light_records", PACKET_LIGHTING_FRAGMENT_SOURCE)
        self.assertIn(
            "u_environment_region_records",
            PACKET_LIGHTING_FRAGMENT_SOURCE,
        )
        self.assertNotIn("u_brightness_areas[", PACKET_LIGHTING_FRAGMENT_SOURCE)
        self.assertNotIn("u_environment_regions[", PACKET_LIGHTING_FRAGMENT_SOURCE)

    def test_legacy_style_setters_update_neutral_render_state(self) -> None:
        from engine.core.compat_shader import (
            set_texture_fog_state,
            set_texture_shine_state,
            set_texture_vibrance_state,
        )
        from engine.render_style_state import (
            get_render_fog_state,
            get_render_shine_state,
            get_render_vibrance_state,
        )

        shader_patcher = patch(
            "engine.core.compat_shader._texture_color_exposure_shader",
            None,
        )
        shader_patcher.start()
        try:
            set_texture_fog_state(
                enabled=True,
                density=0.025,
                color=(0.1, 0.2, 0.3, 1.0),
                compile_shader=False,
            )
            set_texture_vibrance_state(1.3, compile_shader=False)
            set_texture_shine_state(
                enabled=False,
                strength=0.22,
                power=17.0,
                fresnel=0.11,
                tint=(0.8, 0.7, 0.6),
                compile_shader=False,
            )
            self.assertEqual(get_render_fog_state().density, 0.025)
            self.assertEqual(get_render_vibrance_state().vibrance, 1.3)
            self.assertEqual(get_render_shine_state().tint, (0.8, 0.7, 0.6))
        finally:
            try:
                set_texture_fog_state(
                    enabled=False,
                    density=0.0,
                    color=(0.7, 0.8, 1.0, 1.0),
                    compile_shader=False,
                )
                set_texture_vibrance_state(1.0, compile_shader=False)
                set_texture_shine_state(
                    enabled=True,
                    strength=0.38,
                    power=28.0,
                    fresnel=0.18,
                    tint=(1.0, 0.96, 0.86),
                    compile_shader=False,
                )
            finally:
                shader_patcher.stop()

    def test_source_separates_local_lighting_from_exposure(self) -> None:
        self.assertIn("u_local_lighting_enabled", PACKET_LIGHTING_FRAGMENT_SOURCE)
        self.assertIn("u_local_point_query_policy", PACKET_LIGHTING_FRAGMENT_SOURCE)
        self.assertIn("u_clamp_lit_material", PACKET_LIGHTING_FRAGMENT_SOURCE)
        self.assertIn("u_local_reference", PACKET_LIGHTING_FRAGMENT_SOURCE)
        self.assertIn("float brightness = 1.0;", PACKET_LIGHTING_FRAGMENT_SOURCE)
        self.assertIn(
            "bool raster_visibility = u_sun_shadow_enabled != 0;",
            PACKET_LIGHTING_FRAGMENT_SOURCE,
        )
        self.assertNotIn("u_scene_lighting_enabled", PACKET_LIGHTING_FRAGMENT_SOURCE)

    def test_capacity_failures_report_stable_overflow_ids(self) -> None:
        local_limits = PacketGpuStorageLimits(
            max_texture_size=64 * 3,
            texture_image_units=4,
        )
        lights = tuple(
            LocalBrightnessLight(
                light_id=f"light:{index}",
                center=(0.0, 0.0, 0.0),
                radius=1.0,
                value=1.0,
            )
            for index in range(local_limits.local_lights + 1)
        )
        packet = RenderLightingAdapter().packet_for(
            _snapshot(lights),
            GROUND_LIGHTING_RECEIVER,
        )
        with self.assertRaisesRegex(PacketLightingCapacityError, "light:64"):
            validate_packet_capacity(packet, local_limits)

        region_limits = PacketGpuStorageLimits(
            max_texture_size=32 * 2,
            texture_image_units=4,
        )
        regions = tuple(
            _region(index)
            for index in range(region_limits.environment_regions + 1)
        )
        packet = RenderLightingAdapter().packet_for(
            _snapshot(),
            GROUND_LIGHTING_RECEIVER,
            RenderEnvironmentSnapshot(regions),
        )
        with self.assertRaisesRegex(PacketLightingCapacityError, "volume:32"):
            validate_packet_capacity(packet, region_limits)

        portal_limits = PacketGpuStorageLimits(
            max_texture_size=64 * 2,
            texture_image_units=4,
        )
        portals = tuple(
            RenderEnvironmentPortal(
                portal_id=f"portal:{index}",
                kind="window",
                side="north",
                center_x=0.0,
                center_z=0.8,
                width=0.4,
                depth=0.6,
                side_fade=0.1,
                factor=0.86,
            )
            for index in range(portal_limits.environment_portals + 1)
        )
        packet = RenderLightingAdapter().packet_for(
            _snapshot(),
            GROUND_LIGHTING_RECEIVER,
            RenderEnvironmentSnapshot((_region(0, portals),)),
        )
        with self.assertRaisesRegex(PacketLightingCapacityError, "portal:64"):
            validate_packet_capacity(packet, portal_limits)

    def test_packet_storage_requires_vertex_texture_sampling(self) -> None:
        limits = PacketGpuStorageLimits(
            max_texture_size=1024,
            texture_image_units=4,
            vertex_texture_image_units=0,
        )
        with (
            patch.object(PacketGpuStorageLimits, "from_context", return_value=limits),
            self.assertRaisesRegex(RuntimeError, "one vertex texture unit"),
        ):
            PacketGpuStorage.from_context()

    def test_texture_record_packing_exceeds_legacy_uniform_caps(self) -> None:
        lights = tuple(
            LocalBrightnessLight(
                light_id=f"light:{index}",
                center=(float(index), 0.0, float(index + 1)),
                radius=2.0,
                value=1.5,
                bounds=(-1.0, 1.0, -2.0, 2.0),
                indoor_only=True,
                floor_scale=0.25,
            )
            for index in range(65)
        )
        packet = RenderLightingAdapter().packet_for(
            _snapshot(lights),
            GROUND_LIGHTING_RECEIVER,
            RenderEnvironmentSnapshot(
                tuple(_region(index) for index in range(33))
            ),
        )
        limits = PacketGpuStorageLimits(
            max_texture_size=1024,
            texture_image_units=4,
        )

        validate_packet_capacity(packet, limits)
        local_texels = pack_local_light_texels(packet)
        region_texels, portal_texels = pack_environment_texels(packet)
        self.assertEqual(local_texels.shape, (65 * 3, 4))
        self.assertEqual(region_texels.shape, (33 * 2, 4))
        self.assertEqual(portal_texels.shape, (1, 4))
        self.assertEqual(tuple(local_texels[-3]), (64.0, 65.0, 2.0, 1.5))

    def test_mesh_direct_and_batched_paths_accept_packet_backend(self) -> None:
        mesh = BatchedMesh(
            3,
            3,
            texture=7,
            vertex_width=11,
            lighting_receiver=GROUND_LIGHTING_RECEIVER,
            owns_vbo=False,
        )
        packet = RenderLightingAdapter().packet_for(
            _snapshot(),
            GROUND_LIGHTING_RECEIVER,
        )
        shader = Mock()
        gl_names = (
            "glBindBuffer",
            "glEnableClientState",
            "glVertexPointer",
            "glColorPointer",
            "glNormalPointer",
            "glTexCoordPointer",
            "glClientActiveTexture",
            "glEnable",
            "glBlendFunc",
            "glTexEnvi",
            "glBindTexture",
            "glDrawArrays",
            "glDisable",
            "glDisableClientState",
            "use_fixed_pipeline",
        )
        patchers = [patch(f"engine.core.mesh.{name}") for name in gl_names]
        mocks = [patcher.start() for patcher in patchers]
        try:
            with patch(
                "engine.core.mesh.get_legacy_texture_shader",
                side_effect=AssertionError("packet draw requested legacy shader"),
            ) as legacy_shader:
                mesh.draw(lighting_packet=packet, packet_shader=shader)
            legacy_shader.assert_not_called()
        finally:
            for patcher in reversed(patchers):
                patcher.stop()
        shader.bind.assert_called_once_with(
            packet,
            directional_normal_stream=False,
        )
        self.assertTrue(mocks)

        with (
            patch.object(
                BatchedMesh,
                "_draw_textured_prepared_run",
            ) as draw_run,
            patch("engine.core.mesh.get_legacy_texture_shader") as legacy_shader,
            patch("engine.core.mesh.use_fixed_pipeline"),
        ):
            BatchedMesh.draw_many(
                [mesh],
                lighting_packets={GROUND_LIGHTING_RECEIVER.receiver_id: packet},
                packet_shader=shader,
            )
        self.assertIs(draw_run.call_args.kwargs["lighting_packet"], packet)
        self.assertIs(draw_run.call_args.kwargs["packet_shader"], shader)
        legacy_shader.assert_not_called()

    def test_fresh_packet_world_import_loads_no_rollback_modules(self) -> None:
        script = (
            "import sys; "
            f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
            "from game.world.worldscene import WorldScene; "
            "scene = WorldScene(defer_setup=True, lighting_backend='packet'); "
            "print('compat_loaded=' + str("
            "'engine.core.compat_shader' in sys.modules)); "
            "print('bridge_loaded=' + str("
            "'game.world.legacy_lighting_bridge' in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("compat_loaded=False", completed.stdout)
        self.assertIn("bridge_loaded=False", completed.stdout)

    def test_packet_sprite_draw_does_not_request_legacy_shader(self) -> None:
        camera = SimpleNamespace(
            _right=Vector3(1.0, 0.0, 0.0),
            _forward=Vector3(0.0, 0.0, -1.0),
        )
        sprite = WorldSprite(
            position=Vector3(),
            size=(2.0, 3.0),
            texture=7,
            camera=camera,
        )
        packet = RenderLightingAdapter().packet_for(
            _snapshot(),
            SPRITE_LIGHTING_RECEIVER,
        )
        shader = Mock()
        gl_names = (
            "glEnable",
            "glDisable",
            "glBlendFunc",
            "glTexEnvi",
            "glBindTexture",
            "glNormal3f",
            "glColor4f",
            "glBegin",
            "glTexCoord2f",
            "glVertex3f",
            "glEnd",
            "use_fixed_pipeline",
        )
        patchers = [patch(f"engine.rendering.sprite.{name}") for name in gl_names]
        for patcher in patchers:
            patcher.start()
        try:
            with patch(
                "engine.rendering.sprite.get_legacy_texture_shader",
                side_effect=AssertionError("packet sprite requested legacy shader"),
            ) as legacy_shader:
                sprite.draw(
                    lighting_packet=packet,
                    packet_shader=shader,
                )
        finally:
            for patcher in reversed(patchers):
                patcher.stop()

        legacy_shader.assert_not_called()
        shader.bind.assert_called_once_with(packet)

    def test_world_scene_backend_selection_is_explicit(self) -> None:
        default_scene = WorldScene(defer_setup=True)
        self.assertEqual(default_scene.lighting_backend, "packet")
        scene = WorldScene(defer_setup=True, lighting_backend="packet")
        self.assertEqual(scene.lighting_backend, "packet")
        with self.assertRaises(ValueError):
            WorldScene(defer_setup=True, lighting_backend="unknown")

    def test_direct_and_batched_road_paths_forward_packet_bindings(self) -> None:
        packet = RenderLightingAdapter().packet_for(
            _snapshot(),
            ROAD_LIGHTING_RECEIVER,
        )
        shader = Mock()
        road = Road.__new__(Road)
        road._mesh = Mock()
        road.draw(
            camera="camera",
            view_distance=123.0,
            lighting_packet=packet,
            packet_shader=shader,
        )
        road._mesh.draw.assert_called_once_with(
            camera="camera",
            view_distance=123.0,
            lighting_packet=packet,
            packet_shader=shader,
        )

        batch = RoadRenderBatch.__new__(RoadRenderBatch)
        batch._meshes = [Mock()]
        packets = {ROAD_LIGHTING_RECEIVER.receiver_id: packet}
        with patch("game.world.objects.road.BatchedMesh.draw_many") as draw_many:
            batch.draw(
                camera="camera",
                view_distance=123.0,
                lighting_packets=packets,
                packet_shader=shader,
            )
        draw_many.assert_called_once_with(
            batch._meshes,
            camera="camera",
            view_distance=123.0,
            lighting_packets=packets,
            packet_shader=shader,
            require_lighting_packets=True,
        )

    def test_selected_packet_backend_does_not_silently_fall_back(self) -> None:
        renderer = WorldRenderer.__new__(WorldRenderer)
        renderer.scene = Mock(lighting_backend="packet")
        with patch(
            "game.world.world_renderer.get_packet_texture_lighting_shader",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "compilation failed"):
                renderer._packet_lighting_shader()

    def test_world_renderer_routes_custom_and_baked_receiver_packets(self) -> None:
        shader = Mock()
        decal_batch = Mock()
        polygon_batch = Mock()
        door_batch = Mock()
        window_batch = Mock()
        sprite_item = object()
        baked_entity = SimpleNamespace(
            enabled=True,
            visible=True,
            lighting_receiver=CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
            packet_lighting_receiver=DYNAMIC_OBJECT_LIGHTING_RECEIVER,
            draw=Mock(),
        )
        rollback_only_entity = SimpleNamespace(
            enabled=True,
            visible=True,
            lighting_receiver=CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
            draw=Mock(),
        )
        controller = Mock()
        packet_by_receiver = {
            receiver.receiver_id: RenderLightingAdapter().packet_for(
                _snapshot(),
                receiver,
            )
            for receiver in (
                DECAL_LIGHTING_RECEIVER,
                SPRITE_LIGHTING_RECEIVER,
                CPU_BAKED_SLAB_LIGHTING_RECEIVER,
                CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
                DYNAMIC_SLAB_LIGHTING_RECEIVER,
                DYNAMIC_OBJECT_LIGHTING_RECEIVER,
            )
        }
        controller.render_packet_for.side_effect = (
            lambda receiver: packet_by_receiver.get(receiver.receiver_id)
        )
        controller.render_packets = packet_by_receiver
        renderer = WorldRenderer.__new__(WorldRenderer)
        renderer.scene = SimpleNamespace(
            camera=SimpleNamespace(),
            profiler=None,
            build_state=SimpleNamespace(goblins=[]),
            lighting=None,
            sun_direction=None,
            ground_height_at=Mock(),
        )
        renderer.resources = SimpleNamespace(
            decal_batches=[decal_batch],
            wall_tile_batches=[],
            wall_tiles=[],
            entities=[],
            polygon_batches=[polygon_batch],
            polygons=[],
            road_batches=[],
            others=[],
            door_batches=[door_batch],
            window_batches=[window_batch],
            immediate_entities=[baked_entity, rollback_only_entity],
            sprite_items=[sprite_item],
        )
        renderer.lighting_controller = controller
        with (
            patch.object(renderer, "_packet_lighting_shader", return_value=shader),
            patch("game.world.world_renderer.draw_sprites_batched") as draw_sprites,
            patch("game.world.world_renderer.draw_goblin_shadows_batched"),
        ):
            renderer.draw_world_objects()

        decal_batch.draw.assert_called_once_with(
            camera=renderer.scene.camera,
            profiler=None,
            lighting_packet=packet_by_receiver[DECAL_LIGHTING_RECEIVER.receiver_id],
            packet_shader=shader,
        )
        self.assertIs(
            draw_sprites.call_args.kwargs["lighting_packet"],
            packet_by_receiver[SPRITE_LIGHTING_RECEIVER.receiver_id],
        )
        self.assertIs(draw_sprites.call_args.kwargs["packet_shader"], shader)
        for batch in (polygon_batch, door_batch, window_batch):
            self.assertIs(batch.draw.call_args.kwargs["packet_shader"], shader)
            self.assertIs(
                batch.draw.call_args.kwargs["lighting_packets"],
                controller.render_packets,
            )
        baked_entity.draw.assert_called_once_with(
            camera=renderer.scene.camera,
            lighting_packet=packet_by_receiver[
                DYNAMIC_OBJECT_LIGHTING_RECEIVER.receiver_id
            ],
            packet_shader=shader,
        )
        rollback_only_entity.draw.assert_called_once_with(
            camera=renderer.scene.camera
        )


class PacketShaderFramebufferTests(unittest.TestCase):
    def setUp(self) -> None:
        pygame.init()
        try:
            pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.HIDDEN)
        except Exception as exc:
            pygame.quit()
            self.skipTest(f"hidden OpenGL context unavailable: {exc}")
        glViewport(0, 0, 64, 64)
        glDisable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        reset_packet_texture_lighting_shader()

    def tearDown(self) -> None:
        pygame.display.quit()
        pygame.quit()

    @staticmethod
    def _white_texture() -> int:
        texture = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            1,
            1,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            bytes((255, 255, 255, 255)),
        )
        return texture

    @staticmethod
    def _pixel() -> tuple[int, int, int]:
        value = glReadPixels(32, 32, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
        raw = value.tobytes() if hasattr(value, "tobytes") else bytes(value)
        return int(raw[0]), int(raw[1]), int(raw[2])

    def test_packet_shader_compiles_and_renders_closed_interior_sample(self) -> None:
        from engine.core.compat_shader import (
            set_texture_fog_state,
            set_texture_shine_state,
            set_texture_vibrance_state,
            use_fixed_pipeline,
        )

        shader = get_packet_texture_lighting_shader()
        self.assertIsNotNone(shader)
        set_texture_fog_state(enabled=False, compile_shader=False)
        set_texture_vibrance_state(1.0, compile_shader=False)
        set_texture_shine_state(enabled=False, compile_shader=False)
        environment = RenderEnvironmentSnapshot((_region(0),))
        packet = RenderLightingAdapter().packet_for(
            _snapshot(),
            GROUND_LIGHTING_RECEIVER,
            environment,
        )
        texture = self._white_texture()

        glClear(GL_COLOR_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_2D, texture)
        shader.bind(packet)
        glBegin(GL_QUADS)
        for x, y, u, v in (
            (-0.18, -0.18, 0.0, 0.0),
            (0.18, -0.18, 1.0, 0.0),
            (0.18, 0.18, 1.0, 1.0),
            (-0.18, 0.18, 0.0, 1.0),
        ):
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glNormal3f(0.0, 1.0, 0.0)
            glTexCoord2f(u, v)
            glVertex3f(x, y, 0.0)
        glEnd()
        use_fixed_pipeline()

        sample = self._pixel()
        self.assertTrue(
            all(abs(channel - 43) <= 1 for channel in sample),
            sample,
        )

    def test_camera_selected_point_lights_allow_larger_scene_collection(self) -> None:
        shader = get_packet_texture_lighting_shader()
        self.assertIsNotNone(shader)
        lights = tuple(
            PointLight(
                light_id=f"point:{index}",
                position=(float(index), 1.0, 0.0),
                color=(1.0, 1.0, 1.0),
                intensity=1.0,
                range=10.0,
                casts_shadows=False,
            )
            for index in range(17)
        )
        snapshot = LightingSnapshot(
            revision=22,
            base_brightness=1.0,
            sky_color=(0.0, 0.0, 0.0, 1.0),
            directional=_directional(),
            point_lights=lights,
        )
        packet = RenderLightingAdapter().packet_for(
            snapshot,
            GROUND_LIGHTING_RECEIVER,
        )
        shader.set_active_point_lights(
            light.light_id for light in lights[:16]
        )
        shader.bind(packet)
        self.assertEqual(len(shader.active_point_light_ids), 16)

    def _draw_sample(
        self,
        texture: int,
        bind,
        *,
        position,
        normal,
        color,
    ) -> tuple[int, int, int]:
        from engine.core.compat_shader import use_fixed_pipeline

        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(-position[0], -position[1], 0.0)
        glBindTexture(GL_TEXTURE_2D, texture)
        bind()
        glBegin(GL_QUADS)
        for x, y, u, v in (
            (-0.18, -0.18, 0.0, 0.0),
            (0.18, -0.18, 1.0, 0.0),
            (0.18, 0.18, 1.0, 1.0),
            (-0.18, 0.18, 0.0, 1.0),
        ):
            glColor4f(color[0], color[1], color[2], 1.0)
            glNormal3f(normal[0], normal[1], normal[2])
            glTexCoord2f(u, v)
            glVertex3f(position[0] + x, position[1] + y, position[2])
        glEnd()
        use_fixed_pipeline()
        glFinish()
        return self._pixel()

    def test_texture_storage_renders_light_beyond_legacy_cap(self) -> None:
        from engine.render_style_state import (
            update_render_fog_state,
            update_render_shine_state,
            update_render_vibrance_state,
        )

        shader = get_packet_texture_lighting_shader()
        self.assertIsNotNone(shader)
        if shader.storage.limits.local_lights < 65:
            self.skipTest("context texture width cannot represent 65 local lights")
        update_render_fog_state(enabled=False)
        update_render_vibrance_state(1.0)
        update_render_shine_state(enabled=False)

        inactive = tuple(
            LocalBrightnessLight(
                light_id=f"inactive:{index}",
                center=(100.0 + index, 0.0, 100.0),
                radius=0.1,
                value=1.0,
            )
            for index in range(64)
        )
        active_65th = LocalBrightnessLight(
            light_id="active:64",
            center=(0.65, 0.0, 0.65),
            radius=1.0,
            value=1.0,
        )
        baseline_packet = RenderLightingAdapter().packet_for(
            _snapshot(inactive),
            GROUND_LIGHTING_RECEIVER,
        )
        extended_packet = RenderLightingAdapter().packet_for(
            _snapshot((*inactive, active_65th)),
            GROUND_LIGHTING_RECEIVER,
        )
        texture = self._white_texture()
        baseline = self._draw_sample(
            texture,
            lambda: shader.bind(baseline_packet),
            position=(0.65, 0.0, 0.65),
            normal=(0.0, 1.0, 0.0),
            color=(0.5, 0.5, 0.5),
        )
        extended = self._draw_sample(
            texture,
            lambda: shader.bind(extended_packet),
            position=(0.65, 0.0, 0.65),
            normal=(0.0, 1.0, 0.0),
            color=(0.5, 0.5, 0.5),
        )

        self.assertTrue(all(channel >= 120 for channel in extended), extended)
        self.assertTrue(
            all(after - before >= 55 for before, after in zip(baseline, extended)),
            (baseline, extended),
        )

    def test_texture_storage_renders_region_and_portal_beyond_legacy_caps(
        self,
    ) -> None:
        from engine.render_style_state import (
            update_render_fog_state,
            update_render_shine_state,
            update_render_vibrance_state,
        )

        shader = get_packet_texture_lighting_shader()
        self.assertIsNotNone(shader)
        if (
            shader.storage.limits.environment_regions < 33
            or shader.storage.limits.environment_portals < 65
        ):
            self.skipTest("context texture width cannot exceed legacy environment caps")
        update_render_fog_state(enabled=False)
        update_render_vibrance_state(1.0)
        update_render_shine_state(enabled=False)
        texture = self._white_texture()

        far_regions = tuple(
            RenderEnvironmentRegion(
                volume_id=f"far:{index}",
                min_x=10.0 + index,
                max_x=10.5 + index,
                min_z=10.0,
                max_z=10.5,
                indoor_factor=0.34,
            )
            for index in range(32)
        )
        active_region = RenderEnvironmentRegion(
            volume_id="active:32",
            min_x=-0.8,
            max_x=0.8,
            min_z=-0.8,
            max_z=0.8,
            indoor_factor=0.34,
        )
        clear_packet = RenderLightingAdapter().packet_for(
            _snapshot(),
            GROUND_LIGHTING_RECEIVER,
        )
        region_packet = RenderLightingAdapter().packet_for(
            _snapshot(),
            GROUND_LIGHTING_RECEIVER,
            RenderEnvironmentSnapshot((*far_regions, active_region)),
        )
        clear = self._draw_sample(
            texture,
            lambda: shader.bind(clear_packet),
            position=(0.65, 0.0, 0.65),
            normal=(0.0, 1.0, 0.0),
            color=(1.0, 1.0, 1.0),
        )
        region_darkened = self._draw_sample(
            texture,
            lambda: shader.bind(region_packet),
            position=(0.65, 0.0, 0.65),
            normal=(0.0, 1.0, 0.0),
            color=(1.0, 1.0, 1.0),
        )
        self.assertTrue(
            all(before - after >= 75 for before, after in zip(clear, region_darkened)),
            (clear, region_darkened),
        )

        inactive_portals = tuple(
            RenderEnvironmentPortal(
                portal_id=f"inactive:{index}",
                kind="window",
                side="north",
                center_x=100.0 + index,
                center_z=0.8,
                width=0.4,
                depth=0.6,
                side_fade=0.1,
                factor=1.0,
            )
            for index in range(64)
        )
        active_portal = RenderEnvironmentPortal(
            portal_id="active:64",
            kind="window",
            side="north",
            center_x=0.65,
            center_z=0.8,
            width=0.4,
            depth=0.6,
            side_fade=0.1,
            factor=1.0,
        )
        portal_region = RenderEnvironmentRegion(
            volume_id="portal-room",
            min_x=-0.8,
            max_x=0.8,
            min_z=-0.8,
            max_z=0.8,
            indoor_factor=0.34,
            portals=(*inactive_portals, active_portal),
        )
        portal_packet = RenderLightingAdapter().packet_for(
            _snapshot(),
            GROUND_LIGHTING_RECEIVER,
            RenderEnvironmentSnapshot((portal_region,)),
        )
        portal_lit = self._draw_sample(
            texture,
            lambda: shader.bind(portal_packet),
            position=(0.65, 0.0, 0.65),
            normal=(0.0, 1.0, 0.0),
            color=(1.0, 1.0, 1.0),
        )
        self.assertTrue(
            all(after - before >= 55 for before, after in zip(region_darkened, portal_lit)),
            (region_darkened, portal_lit),
        )

    def test_packet_shader_matches_migrated_legacy_receiver_matrix(self) -> None:
        from engine.core.compat_shader import (
            get_texture_color_exposure_shader,
            set_texture_fog_state,
            set_texture_lighting_state,
            set_texture_shine_state,
            set_texture_vibrance_state,
        )

        packet_shader = get_packet_texture_lighting_shader()
        legacy_shader = get_texture_color_exposure_shader()
        self.assertIsNotNone(packet_shader)
        self.assertIsNotNone(legacy_shader)
        set_texture_fog_state(enabled=False, compile_shader=False)
        set_texture_vibrance_state(1.0, compile_shader=False)
        set_texture_shine_state(enabled=False, compile_shader=False)

        directional = DirectionalLightSnapshot(
            sun_position=(1.0, 1.0, 1.0),
            sun_target=(0.0, 0.0, 0.0),
            sun_direction=(-1.0, -1.0, -1.0),
            light_direction=(1.0, 1.0, 1.0),
            ambient=0.2,
            diffuse=0.8,
            max_factor=1.0,
            tint=(1.0, 1.0, 1.0),
        )
        torch = LocalBrightnessLight(
            light_id="fixture:torch",
            center=(0.25, 0.0, 0.25),
            radius=0.12,
            value=3.4,
            falloff=2.2,
            bounds=(-0.3, 0.3, -0.3, 0.3),
            indoor_only=True,
            floor_scale=0.28,
        )
        lighting = LightingSnapshot(
            revision=9,
            base_brightness=0.8,
            sky_color=(0.7, 0.8, 1.0, 1.0),
            directional=directional,
            local_lights=(torch,),
        )
        environment = RenderEnvironmentSnapshot(
            (
                RenderEnvironmentRegion(
                    volume_id="fixture:room",
                    min_x=-0.3,
                    max_x=0.3,
                    min_z=-0.3,
                    max_z=0.3,
                    indoor_factor=0.34,
                ),
            )
        )
        legacy_region = {
            "min_x": -0.3,
            "max_x": 0.3,
            "min_z": -0.3,
            "max_z": 0.3,
            "factor": 0.34,
            "openings": [],
        }
        set_texture_lighting_state(
            lighting=lighting,
            brightness_areas=lighting.local_lights,
            covered_regions=[legacy_region],
            exposure_scale=1.0,
            compile_shader=False,
        )
        texture = self._white_texture()
        adapter = RenderLightingAdapter()
        cases = (
            (
                GROUND_LIGHTING_RECEIVER,
                (0.65, 0.0, 0.65),
                (0.0, 1.0, 0.0),
                (0.72, 0.82, 0.62),
            ),
            (
                ROAD_LIGHTING_RECEIVER,
                (0.65, 0.0, 0.65),
                (0.0, 1.0, 0.0),
                (0.48, 0.46, 0.42),
            ),
            (
                FENCE_LIGHTING_RECEIVER,
                (0.65, 0.0, 0.65),
                (1.0, 0.0, 0.0),
                (0.58, 0.46, 0.32),
            ),
            (
                SPRITE_LIGHTING_RECEIVER,
                (0.65, 0.0, 0.65),
                (-0.35, 1.0, -0.35),
                (0.40, 0.80, 0.40),
            ),
            (
                SPRITE_LIGHTING_RECEIVER,
                (0.65, 0.0, 0.65),
                (-0.35, 1.0, -0.35),
                (1.0, 0.82, 0.55),
            ),
            (
                DECAL_LIGHTING_RECEIVER,
                (0.65, 0.0, 0.65),
                (0.0, 1.0, 0.0),
                (0.24, 0.24, 0.24),
            ),
            (
                CPU_BAKED_SLAB_LIGHTING_RECEIVER,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.52, 0.34, 0.18),
            ),
            (
                CPU_BAKED_SLAB_LIGHTING_RECEIVER,
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.42, 0.62, 0.82),
            ),
            (
                CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.55, 0.36, 0.16),
            ),
            (
                CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
                (0.65, 0.0, 0.65),
                (0.0, 0.0, -1.0),
                (0.68, 0.30, 0.50),
            ),
            (
                GROUND_LIGHTING_RECEIVER,
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.72, 0.82, 0.62),
            ),
            (
                GROUND_LIGHTING_RECEIVER,
                (0.25, 0.0, 0.25),
                (0.0, 1.0, 0.0),
                (0.72, 0.82, 0.62),
            ),
            *(
                (
                    TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
                    (0.65, 0.0, 0.65),
                    normal,
                    (0.75, 0.65, 0.55),
                )
                for normal in (
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, -1.0),
                    (-1.0, 0.0, 0.0),
                )
            ),
        )
        for receiver, position, normal, color in cases:
            packet = adapter.packet_for(lighting, receiver, environment)
            legacy_flags = receiver.compatibility_shader_flags(
                has_normals=True
            )
            legacy = self._draw_sample(
                texture,
                lambda flags=legacy_flags: legacy_shader.bind(
                    scene_lighting_enabled=flags.scene_lighting,
                    directional_enabled=flags.directional,
                    environment_enabled=flags.environment,
                    fog_enabled=False,
                    shine_enabled=False,
                ),
                position=position,
                normal=normal,
                color=color,
            )
            replacement = self._draw_sample(
                texture,
                lambda packet=packet: packet_shader.bind(packet),
                position=position,
                normal=normal,
                color=color,
            )
            self.assertTrue(
                all(abs(a - b) <= 1 for a, b in zip(legacy, replacement)),
                (receiver.receiver_id, legacy, replacement),
            )

        baked_migration_cases = (
            (
                CPU_BAKED_SLAB_LIGHTING_RECEIVER,
                DYNAMIC_SLAB_LIGHTING_RECEIVER,
                (0.0, 1.0, 0.0),
                (0.82, 0.82, 0.82),
            ),
            (
                CPU_BAKED_SLAB_LIGHTING_RECEIVER,
                DYNAMIC_SLAB_LIGHTING_RECEIVER,
                (-1.0, 0.0, 0.0),
                (0.72, 0.72, 0.72),
            ),
            (
                CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
                DYNAMIC_OBJECT_LIGHTING_RECEIVER,
                (0.0, 1.0, 0.0),
                (0.92, 0.92, 0.92),
            ),
            (
                CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
                DYNAMIC_OBJECT_LIGHTING_RECEIVER,
                (0.0, 0.0, -1.0),
                (0.75, 0.75, 0.75),
            ),
        )
        for legacy_receiver, dynamic_receiver, normal, material_color in (
            baked_migration_cases
        ):
            ndotl = max(0.0, sum(normal) / math.sqrt(3.0))
            sun_factor = min(
                directional.max_factor,
                directional.ambient + directional.diffuse * ndotl,
            )
            baked_color = tuple(
                channel * sun_factor for channel in material_color
            )
            legacy_flags = legacy_receiver.compatibility_shader_flags(
                has_normals=True
            )
            legacy = self._draw_sample(
                texture,
                lambda flags=legacy_flags: legacy_shader.bind(
                    scene_lighting_enabled=flags.scene_lighting,
                    directional_enabled=flags.directional,
                    environment_enabled=flags.environment,
                    fog_enabled=False,
                    shine_enabled=False,
                ),
                position=(0.65, 0.0, 0.65),
                normal=normal,
                color=baked_color,
            )
            packet = adapter.packet_for(
                lighting,
                dynamic_receiver,
                environment,
            )
            replacement = self._draw_sample(
                texture,
                lambda packet=packet: packet_shader.bind(packet),
                position=(0.65, 0.0, 0.65),
                normal=normal,
                color=material_color,
            )
            self.assertTrue(
                all(abs(a - b) <= 1 for a, b in zip(legacy, replacement)),
                (dynamic_receiver.receiver_id, legacy, replacement),
            )

        # Textured polygons historically call Camera.get_brightness_at(point)
        # without a surface classification. Indoor-only lights therefore apply
        # unconditionally, floor_scale is ignored, and the combined local + sun
        # material is clamped before texture modulation.
        polygon_position = (0.25, 0.0, 0.25)
        polygon_normal = (0.0, 1.0, 0.0)
        polygon_ndotl = 1.0 / math.sqrt(3.0)
        polygon_sun = min(
            directional.max_factor,
            directional.ambient + directional.diffuse * polygon_ndotl,
        )
        polygon_light = LocalBrightnessLight(
            light_id="fixture:polygon-point-query",
            center=polygon_position,
            radius=2.0,
            value=3.4,
            falloff=0.0,
            bounds=(-1.0, 1.0, -1.0, 1.0),
            indoor_only=True,
            floor_scale=0.0,
        )
        polygon_lighting = LightingSnapshot(
            revision=10,
            base_brightness=lighting.base_brightness,
            sky_color=lighting.sky_color,
            directional=directional,
            local_lights=(polygon_light,),
        )
        polygon_baked = min(1.0, polygon_light.value * polygon_sun)
        legacy_flags = (
            CPU_BAKED_OBJECT_LIGHTING_RECEIVER.compatibility_shader_flags(
                has_normals=True
            )
        )
        legacy = self._draw_sample(
            texture,
            lambda: legacy_shader.bind(
                scene_lighting_enabled=legacy_flags.scene_lighting,
                directional_enabled=legacy_flags.directional,
                environment_enabled=legacy_flags.environment,
                fog_enabled=False,
                shine_enabled=False,
            ),
            position=polygon_position,
            normal=polygon_normal,
            color=(polygon_baked, polygon_baked, polygon_baked),
        )
        polygon_packet = adapter.packet_for(
            polygon_lighting,
            DYNAMIC_POLYGON_LIGHTING_RECEIVER,
            environment,
        )
        replacement = self._draw_sample(
            texture,
            lambda: packet_shader.bind(polygon_packet),
            position=polygon_position,
            normal=polygon_normal,
            color=(1.0, 1.0, 1.0),
        )
        self.assertTrue(all(channel >= 254 for channel in replacement), replacement)
        self.assertTrue(
            all(abs(a - b) <= 1 for a, b in zip(legacy, replacement)),
            (DYNAMIC_POLYGON_LIGHTING_RECEIVER.receiver_id, legacy, replacement),
        )


if __name__ == "__main__":
    unittest.main()
