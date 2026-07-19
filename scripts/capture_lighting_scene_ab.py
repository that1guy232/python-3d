"""Capture deterministic full-scene legacy/packet lighting comparisons."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from engine.camera import CameraBrightnessArea  # noqa: E402
from OpenGL.GL import (  # noqa: E402
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_LEQUAL,
    GL_RENDERER,
    GL_RGB,
    GL_SHADING_LANGUAGE_VERSION,
    GL_UNSIGNED_BYTE,
    GL_VENDOR,
    GL_VERSION,
    glDepthFunc,
    glDisable,
    glEnable,
    glFinish,
    glGetString,
    glReadPixels,
    glViewport,
)
from pygame.math import Vector3  # noqa: E402

from engine.rendering.frame_comparison import (  # noqa: E402
    FrameComparisonThresholds,
    amplified_rgb_difference,
    compare_rgb_frames,
)
from engine.rendering.lighting_state import LocalBrightnessLight  # noqa: E402
from engine.rendering.packet_shader import (  # noqa: E402
    get_packet_texture_lighting_shader,
    reset_packet_texture_lighting_shader,
)
from game.world.world_content import WorldContent, building  # noqa: E402
from game.world.lighting_controller import (  # noqa: E402
    LEGACY_LIGHTING_ALIAS_NAMES,
)
from game.world.lighting_receivers import (  # noqa: E402
    PACKET_RUNTIME_LIGHTING_RECEIVER_IDS,
    ROLLBACK_ONLY_LIGHTING_RECEIVER_IDS,
)
from game.world.worldscene import WorldScene  # noqa: E402
from game.world.world_lighting_plan import AuthoredOpeningLight  # noqa: E402


def _save_rgb(path: Path, frame: np.ndarray) -> None:
    surface = pygame.image.frombuffer(
        np.ascontiguousarray(frame).tobytes(),
        (int(frame.shape[1]), int(frame.shape[0])),
        "RGB",
    )
    pygame.image.save(surface, str(path))


def _read_rgb(width: int, height: int) -> np.ndarray:
    glFinish()
    raw = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    return np.ascontiguousarray(np.flipud(frame))


def _window_transmission_probe(frame: np.ndarray) -> dict[str, float | bool]:
    """Measure the fixture's direct-sun floor band against adjacent shadow."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("window transmission probe requires an RGB frame")

    height, width = frame.shape[:2]

    def roi(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        left = max(0, min(width - 1, int(round(width * x0))))
        right = max(left + 1, min(width, int(round(width * x1))))
        top = max(0, min(height - 1, int(round(height * y0))))
        bottom = max(top + 1, min(height, int(round(height * y1))))
        return frame[top:bottom, left:right, :3].astype(np.float32)

    def mean_luma(pixels: np.ndarray) -> float:
        return float(
            np.mean(
                pixels[..., 0] * 0.2126
                + pixels[..., 1] * 0.7152
                + pixels[..., 2] * 0.0722
            )
        )

    # Stable normalized regions in the deterministic window_interior fixture.
    shadow_luma = mean_luma(roi(190 / 480, 265 / 480, 205 / 270, 250 / 270))
    sunlit_luma = mean_luma(roi(300 / 480, 430 / 480, 205 / 270, 250 / 270))
    delta = sunlit_luma - shadow_luma
    ratio = sunlit_luma / max(shadow_luma, 0.001)
    return {
        "shadow_mean_luma": shadow_luma,
        "sunlit_mean_luma": sunlit_luma,
        "sunlit_minus_shadow": delta,
        "sunlit_to_shadow_ratio": ratio,
        "minimum_delta": 3.0,
        "minimum_ratio": 1.15,
        "passed": bool(delta >= 3.0 and ratio >= 1.15),
    }


def _torch_emissive_probe(frame: np.ndarray) -> dict[str, float | int | bool]:
    """Require a bright warm flame tip in the deterministic close-up view."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("torch emissive probe requires an RGB frame")
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, int(round(width * (228 / 480)))))
    right = max(left + 1, min(width, int(round(width * (245 / 480)))))
    top = max(0, min(height - 1, int(round(height * (92 / 270)))))
    bottom = max(top + 1, min(height, int(round(height * (125 / 270)))))
    pixels = frame[top:bottom, left:right, :3].astype(np.float32)
    luma = (
        pixels[..., 0] * 0.2126
        + pixels[..., 1] * 0.7152
        + pixels[..., 2] * 0.0722
    )
    warm = (
        (pixels[..., 0] >= 120.0)
        & (pixels[..., 0] >= pixels[..., 1] * 1.05)
        & (pixels[..., 1] >= pixels[..., 2] * 1.4)
    )
    highlight_luma = float(np.percentile(luma, 99.0))
    warm_pixels = int(np.count_nonzero(warm))
    return {
        "flame_tip_p99_luma": highlight_luma,
        "warm_flame_pixel_count": warm_pixels,
        "minimum_p99_luma": 110.0,
        "minimum_warm_pixel_count": 20,
        "passed": bool(highlight_luma >= 110.0 and warm_pixels >= 20),
    }


def _fence_shadow_gap_probe(frame: np.ndarray) -> dict[str, float | int | bool]:
    """Require sunlit ground to survive between fence cutout shadows."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("fence shadow probe requires an RGB frame")
    height, width = frame.shape[:2]
    left = 0
    right = max(1, min(width, int(round(width * (300 / 800)))))
    top = max(0, min(height - 1, int(round(height * (180 / 450)))))
    bottom = max(top + 1, min(height, int(round(height * (300 / 450)))))
    pixels = frame[top:bottom, left:right, :3].astype(np.float32)
    ground = (
        (pixels[..., 1] > pixels[..., 0] * 1.15)
        & (pixels[..., 1] > pixels[..., 2] * 1.25)
    )
    luma = (
        pixels[..., 0] * 0.2126
        + pixels[..., 1] * 0.7152
        + pixels[..., 2] * 0.0722
    )[ground]
    ground_pixels = int(luma.size)
    lower_quartile_luma = float(np.percentile(luma, 25.0)) if luma.size else 0.0
    return {
        "sampled_ground_pixel_count": ground_pixels,
        "ground_p25_luma": lower_quartile_luma,
        "minimum_ground_pixel_count": 1000,
        "minimum_ground_p25_luma": 38.0,
        "passed": bool(ground_pixels >= 1000 and lower_quartile_luma >= 38.0),
    }


def _decode_gl_string(name: int) -> str:
    value = glGetString(name)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _device_identity() -> dict[str, str]:
    return {
        "vendor": _decode_gl_string(GL_VENDOR),
        "renderer": _decode_gl_string(GL_RENDERER),
        "opengl_version": _decode_gl_string(GL_VERSION),
        "glsl_version": _decode_gl_string(GL_SHADING_LANGUAGE_VERSION),
    }


def _look_at(camera, position: Vector3, target: Vector3) -> None:
    forward = target - position
    if forward.length_squared() <= 1e-9:
        forward = Vector3(0.0, 0.0, -1.0)
    else:
        forward = forward.normalize()
    camera.position = position.copy()
    camera.rotation.x = math.asin(max(-1.0, min(1.0, forward.y)))
    camera.rotation.y = math.atan2(-forward.x, -forward.z)
    camera.update_rotation(0.0)


def _capture(scene: WorldScene, backend: str, width: int, height: int, warmup: int):
    scene.set_lighting_backend(backend)
    for _ in range(max(0, warmup)):
        scene.render(show_hud=False, text=None, fps=0.0)
    scene.render(show_hud=False, text=None, fps=0.0)
    return _read_rgb(width, height)


def _fixture_content() -> WorldContent:
    return WorldContent(
        buildings=(
            building(
                (650.0, 0.0, 380.0),
                width=230.0,
                depth=180.0,
                height=72.0,
                doorway_side="south",
                windows=(
                    {"side": "north", "offset": -45.0},
                    {"side": "east", "offset": 18.0},
                    {"side": "west", "offset": -22.0},
                ),
                torches=(
                    {"side": "north", "offset": 35.0},
                    {"side": "east", "offset": -28.0},
                ),
            ),
        )
    )


def _configure_scene(scene: WorldScene) -> None:
    scene.clouds_enabled = False
    scene.hud_visible = False
    scene.compass_visible = False
    scene.minimap_visible = False
    scene.held_item_visible = False
    scene.test_light_visible = False
    scene.controls_text_visible = False
    scene.debug_text_visible = False


def _fixture_viewpoints(scene: WorldScene):
    building_center = Vector3(650.0, 0.0, 380.0)
    exterior_ground = scene.ground_height_at(650.0, 640.0)
    interior_ground = scene.ground_height_at(650.0, 380.0)
    overview_ground = scene.ground_height_at(920.0, 720.0)
    torch_ground = scene.ground_height_at(685.0, 400.0)
    torch_closeup_ground = scene.ground_height_at(685.0, 430.0)
    window_ground = scene.ground_height_at(605.0, 400.0)
    # The deterministic fixture's north boundary near x=400 has the flattest
    # unobstructed inward run, making the fence's alternating shadow gaps easy
    # to audit without confusing them with terrain folds or tree shadows.
    fence_x = 400.0
    fence_z = float(scene.ground_bounds[3])
    fence_view_x = fence_x + 90.0
    fence_view_z = fence_z - 260.0
    fence_ground = scene.ground_height_at(fence_view_x, fence_view_z)
    fence_target_x = fence_x - 35.0
    fence_target_z = fence_z - 70.0
    fence_target_ground = scene.ground_height_at(fence_target_x, fence_target_z)
    return (
        (
            "building_exterior",
            Vector3(650.0, exterior_ground + 32.0, 640.0),
            Vector3(650.0, exterior_ground + 31.0, 380.0),
        ),
        (
            "building_interior",
            Vector3(650.0, interior_ground + 27.0, 380.0),
            Vector3(650.0, interior_ground + 27.0, 500.0),
        ),
        (
            "world_overview",
            Vector3(920.0, overview_ground + 190.0, 720.0),
            Vector3(building_center.x, 25.0, building_center.z),
        ),
        (
            "torch_interior",
            Vector3(685.0, torch_ground + 27.0, 400.0),
            Vector3(685.0, torch_ground + 28.0, 470.0),
        ),
        (
            "torch_closeup",
            Vector3(685.0, torch_closeup_ground + 36.0, 430.0),
            Vector3(685.0, torch_closeup_ground + 36.0, 462.0),
        ),
        (
            "window_interior",
            Vector3(605.0, window_ground + 27.0, 400.0),
            Vector3(605.0, window_ground + 28.0, 470.0),
        ),
        (
            "fence_shadow",
            Vector3(fence_view_x, fence_ground + 120.0, fence_view_z),
            Vector3(fence_target_x, fence_target_ground + 4.0, fence_target_z),
        ),
    )


_DOORWAY_NORMALS = {
    "north": Vector3(0.0, 0.0, 1.0),
    "east": Vector3(1.0, 0.0, 0.0),
    "south": Vector3(0.0, 0.0, -1.0),
    "west": Vector3(-1.0, 0.0, 0.0),
}


def _generated_viewpoints(scene: WorldScene):
    specs = tuple(getattr(scene, "building_specs", ()) or ())
    if not specs:
        raise RuntimeError("generated-world capture requires at least one building")

    spec = specs[0]
    center = Vector3(spec["position"])
    base_y = float(
        spec.get("base_y", scene.ground_height_at(center.x, center.z))
    )
    doorway_side = str(spec.get("doorway_side", "south")).lower()
    doorway_normal = _DOORWAY_NORMALS.get(
        doorway_side,
        _DOORWAY_NORMALS["south"],
    )
    face_half_extent = (
        float(spec["width"]) * 0.5
        if abs(doorway_normal.x) > 0.5
        else float(spec["depth"]) * 0.5
    )
    exterior = center + doorway_normal * (face_half_extent + 160.0)
    exterior.y = scene.ground_height_at(exterior.x, exterior.z) + 32.0
    interior = Vector3(center.x, base_y + 27.0, center.z)
    interior_target = interior - doorway_normal * 120.0
    overview_x = center.x + 270.0
    overview_z = center.z + 340.0
    overview = Vector3(
        overview_x,
        scene.ground_height_at(overview_x, overview_z) + 190.0,
        overview_z,
    )
    return (
        (
            "building_exterior",
            exterior,
            Vector3(center.x, base_y + 31.0, center.z),
        ),
        ("building_interior", interior, interior_target),
        (
            "world_overview",
            overview,
            Vector3(center.x, base_y + 25.0, center.z),
        ),
    )


def _viewpoints(scene: WorldScene, world_mode: str):
    if world_mode == "generated":
        return _generated_viewpoints(scene)
    return _fixture_viewpoints(scene)


def _primary_building_summary(scene: WorldScene) -> dict[str, object] | None:
    specs = tuple(getattr(scene, "building_specs", ()) or ())
    if not specs:
        return None
    spec = specs[0]
    return {
        "position": list(Vector3(spec["position"])),
        "base_y": float(spec.get("base_y", 0.0)),
        "width": float(spec["width"]),
        "depth": float(spec["depth"]),
        "height": float(spec["height"]),
        "doorway_side": str(spec.get("doorway_side", "south")),
    }


def _resource_counts(scene: WorldScene) -> dict[str, int]:
    resources = scene.render_resources
    return {
        "local_lights": len(scene.lighting.snapshot().local_lights),
        "point_lights": len(scene.lighting.snapshot().point_lights),
        "environment_regions": len(scene.environment_volumes),
        "environment_portals": sum(
            len(volume.portals) for volume in scene.environment_volumes
        ),
        "fence_meshes": len(resources.fence_meshes),
        "wall_batches": len(resources.wall_tile_batches),
        "road_batches": len(resources.road_batches),
        "door_batches": len(resources.door_batches),
        "window_batches": len(resources.window_batches),
        "polygon_batches": len(resources.polygon_batches),
        "decal_batches": len(resources.decal_batches),
        "sprites": len(resources.sprite_items),
        "immediate_entities": len(resources.immediate_entities),
    }


def _dynamic_geometry_contract(scene: WorldScene) -> dict[str, object]:
    resources = scene.render_resources
    families = {
        "ground": [resources.ground_mesh],
        "roads": [
            mesh
            for batch in resources.road_batches
            for mesh in getattr(batch, "_meshes", (batch,))
        ],
        "walls": [
            batch
            for batch in resources.wall_tile_batches
            if getattr(batch, "texture", None) is not None
        ],
        "fences": list(resources.fence_meshes),
    }
    widths = {
        name: [int(getattr(mesh, "vertex_width", 0)) for mesh in meshes]
        for name, meshes in families.items()
    }
    populated = all(bool(values) for values in widths.values())
    static_geometry_violations = list(
        scene.lighting_controller.packet_static_geometry_violations()
    )
    ground_sun_caster_disabled = not bool(
        getattr(resources.ground_mesh, "casts_sun_shadows", True)
    )
    fence_alpha_cutout_shadows = bool(families["fences"]) and all(
        bool(getattr(mesh, "alpha_test", False))
        and 0.0 < float(getattr(mesh, "alpha_cutoff", 0.0)) < 1.0
        for mesh in families["fences"]
    )
    return {
        "vertex_widths": widths,
        "minimum_dynamic_width": 11,
        "ground_sun_caster_disabled": ground_sun_caster_disabled,
        "fence_alpha_cutout_shadows": fence_alpha_cutout_shadows,
        "static_geometry_violations": static_geometry_violations,
        "passed": populated
        and all(width >= 11 for values in widths.values() for width in values)
        and ground_sun_caster_disabled
        and fence_alpha_cutout_shadows
        and not static_geometry_violations,
    }


def _packet_receiver_contract(scene: WorldScene) -> dict[str, object]:
    prepared = frozenset(scene.lighting_controller.render_packets)
    missing = sorted(PACKET_RUNTIME_LIGHTING_RECEIVER_IDS - prepared)
    rollback_packets = sorted(ROLLBACK_ONLY_LIGHTING_RECEIVER_IDS & prepared)
    unexpected = sorted(
        prepared
        - PACKET_RUNTIME_LIGHTING_RECEIVER_IDS
        - ROLLBACK_ONLY_LIGHTING_RECEIVER_IDS
    )
    return {
        "prepared_receiver_ids": sorted(prepared),
        "expected_runtime_receiver_ids": sorted(
            PACKET_RUNTIME_LIGHTING_RECEIVER_IDS
        ),
        "missing_runtime_receiver_ids": missing,
        "rollback_receiver_ids": sorted(ROLLBACK_ONLY_LIGHTING_RECEIVER_IDS),
        "prepared_rollback_receiver_ids": rollback_packets,
        "unexpected_receiver_ids": unexpected,
        "passed": not missing and not rollback_packets and not unexpected,
    }


def _packet_construction_contract(
    scene: WorldScene,
    *,
    loaded_modules=None,
) -> dict[str, object]:
    modules = sys.modules if loaded_modules is None else loaded_modules
    legacy_module_loaded = "engine.core.compat_shader" in modules
    legacy_bridge_module_loaded = "game.world.legacy_lighting_bridge" in modules
    aliases_present = [
        name for name in LEGACY_LIGHTING_ALIAS_NAMES if hasattr(scene, name)
    ]
    return {
        "legacy_compat_module_loaded": legacy_module_loaded,
        "legacy_bridge_module_loaded": legacy_bridge_module_loaded,
        "legacy_lighting_aliases_present": aliases_present,
        "passed": (
            not legacy_module_loaded
            and not legacy_bridge_module_loaded
            and not aliases_present
        ),
    }


def _packet_local_light_ownership_contract(
    scene: WorldScene,
) -> dict[str, object]:
    violations: list[str] = []
    legacy_lighting_mutation_names = (
        "set_brightness_modifiers",
        "extend_brightness_modifiers",
        "add_brightness_modifier",
        "update_brightness_modifier",
        "remove_brightness_modifiers",
        "install_brightness_modifiers_on_camera",
    )
    legacy_camera_projection_names = (
        "brightness_areas",
        "_brightness_areas_optimized",
        "add_brightness_area",
        "set_brightness_areas",
        "clear_brightness_areas",
    )
    collections = {
        "torch_light_modifiers": (LocalBrightnessLight,),
        "doorway_light_modifiers_by_region": (AuthoredOpeningLight,),
        "doorway_light_modifiers": (AuthoredOpeningLight,),
        "window_light_modifiers": (AuthoredOpeningLight,),
        "opening_light_modifiers": (AuthoredOpeningLight,),
    }
    for name, expected_types in collections.items():
        for index, value in enumerate(getattr(scene, name, ()) or ()):
            if value is None and name in (
                "doorway_light_modifiers_by_region",
                "doorway_light_modifiers",
            ):
                continue
            if not isinstance(value, expected_types):
                violations.append(f"{name}[{index}]:{type(value).__name__}")

    lighting = getattr(scene, "lighting", None)
    local_light_view = getattr(lighting, "local_lights", ())
    if local_light_view is None:
        local_light_view = ()
    local_lights = tuple(local_light_view)
    local_light_view_immutable = isinstance(local_light_view, tuple)
    if not local_light_view_immutable:
        violations.append("lighting:mutable-local-light-view")
    for index, value in enumerate(local_lights):
        if not isinstance(value, LocalBrightnessLight):
            violations.append(f"lighting.local_lights[{index}]:{type(value).__name__}")

    for index, door in enumerate(getattr(scene.build_state, "doors", ()) or ()):
        value = getattr(door, "_doorway_brightness_modifier", None)
        if value is not None and not isinstance(value, LocalBrightnessLight):
            violations.append(
                f"build_state.doors[{index}]._doorway_brightness_modifier:"
                f"{type(value).__name__}"
            )

    stored_projection = bool(
        lighting is not None
        and "brightness_modifiers" in getattr(lighting, "__dict__", {})
    )
    legacy_projection_api = bool(
        lighting is not None and hasattr(lighting, "brightness_modifiers")
    )
    if legacy_projection_api:
        violations.append("lighting:legacy-brightness-projection-api")
    legacy_lighting_mutation_apis = [
        name
        for name in legacy_lighting_mutation_names
        if lighting is not None and hasattr(lighting, name)
    ]
    for name in legacy_lighting_mutation_apis:
        violations.append(f"lighting:legacy-mutation-api:{name}")
    lighting_stores_legacy_regions = bool(
        lighting is not None
        and "covered_regions" in getattr(lighting, "__dict__", {})
    )
    if lighting_stores_legacy_regions:
        violations.append("lighting:stored-legacy-covered-regions")
    legacy_region_count = len(
        getattr(
            getattr(scene, "lighting_controller", None),
            "legacy_covered_regions",
            (),
        )
        or ()
    )
    if legacy_region_count:
        violations.append("controller:packet-owned-legacy-covered-regions")
    camera = getattr(scene, "camera", None)
    camera_lights = tuple(
        getattr(camera, "brightness_query_lights", ()) or ()
    )
    for index, value in enumerate(camera_lights):
        if not isinstance(value, CameraBrightnessArea):
            violations.append(
                f"camera.brightness_query_lights[{index}]:{type(value).__name__}"
            )
    camera_stored_projection = bool(
        camera is not None
        and (
            "brightness_areas" in getattr(camera, "__dict__", {})
            or "_brightness_areas_optimized" in getattr(camera, "__dict__", {})
        )
    )
    if camera_stored_projection:
        violations.append("camera:stored-legacy-brightness-projection")
    legacy_camera_projection_apis = [
        name
        for name in legacy_camera_projection_names
        if camera is not None and hasattr(camera, name)
    ]
    for name in legacy_camera_projection_apis:
        violations.append(f"camera:legacy-projection-api:{name}")
    local_ids = tuple(light.light_id for light in local_lights)
    camera_ids = tuple(light.light_id for light in camera_lights)
    if local_ids != camera_ids:
        violations.append("camera:typed-light-id-mismatch")
    camera_revision = getattr(camera, "_brightness_source_revision", None)
    lighting_revision = getattr(lighting, "revision", None)
    if camera_revision != lighting_revision:
        violations.append("camera:source-revision-mismatch")
    return {
        "local_light_count": len(local_lights),
        "camera_query_light_count": len(camera_lights),
        "local_light_view_immutable": local_light_view_immutable,
        "legacy_projection_stored_on_lighting": stored_projection,
        "legacy_projection_api_on_lighting": legacy_projection_api,
        "legacy_mutation_apis_on_lighting": legacy_lighting_mutation_apis,
        "legacy_regions_stored_on_lighting": lighting_stores_legacy_regions,
        "controller_legacy_region_count": legacy_region_count,
        "legacy_projection_stored_on_camera": camera_stored_projection,
        "legacy_projection_apis_on_camera": legacy_camera_projection_apis,
        "violations": violations,
        "passed": not stored_projection and not violations,
    }


def capture(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.set_mode(
        (args.width, args.height),
        pygame.OPENGL | pygame.HIDDEN,
    )
    glViewport(0, 0, args.width, args.height)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glDisable(GL_CULL_FACE)

    scene = None
    try:
        fixture_mode = args.world_mode == "fixture"
        scene = WorldScene(
            grid_count=52,
            grid_tile_size=25,
            grid_gap=0,
            tree_count=18,
            grass_count=24,
            rock_count=12,
            building_count=1 if fixture_mode else args.building_count,
            world_content=_fixture_content() if fixture_mode else None,
            world_random_seed=args.seed,
            lighting_backend="packet",
        )
        _configure_scene(scene)
        thresholds = FrameComparisonThresholds(
            channel_tolerance=args.channel_tolerance,
            min_stable_pixel_ratio=args.min_stable_pixel_ratio,
            max_mean_absolute_error=args.max_mean_absolute_error,
            max_p99_absolute_error=args.max_p99_absolute_error,
            max_absolute_error=args.max_absolute_error,
            max_changed_pixel_ratio=args.max_changed_pixel_ratio,
        )
        report: dict[str, object] = {
            "schema_version": 3,
            "seed": args.seed,
            "device": _device_identity(),
            "world": {
                "mode": args.world_mode,
                "requested_building_count": (
                    1 if fixture_mode else args.building_count
                ),
                "actual_building_count": len(scene.building_specs),
                "primary_building": _primary_building_summary(scene),
            },
            "resolution": [args.width, args.height],
            "thresholds": {
                name: getattr(thresholds, name)
                for name in thresholds.__dataclass_fields__
            },
            "visual_acceptance_policy": {
                "packet_vs_legacy": "diagnostic_only",
                "reason": (
                    "geometry shadows intentionally replace legacy scalar/region pixels"
                ),
            },
            "resource_counts": _resource_counts(scene),
            "dynamic_geometry_contract": _dynamic_geometry_contract(scene),
            "packet_receiver_contract": _packet_receiver_contract(scene),
            "packet_alias_contract": {
                "samples": [],
                "passed": True,
            },
            "packet_construction_contract": _packet_construction_contract(scene),
            "packet_local_light_ownership_contract": (
                _packet_local_light_ownership_contract(scene)
            ),
            "viewpoints": {},
        }

        all_passed = (
            bool(report["dynamic_geometry_contract"]["passed"])
            and bool(report["packet_receiver_contract"]["passed"])
            and bool(report["packet_construction_contract"]["passed"])
            and bool(report["packet_local_light_ownership_contract"]["passed"])
        )
        for name, position, target in _viewpoints(scene, args.world_mode):
            isolated_fence_items = None
            if fixture_mode and name == "fence_shadow":
                # Keep this evidence view about the fence material itself;
                # nearby deterministic trees and goblins otherwise cover the
                # ground strip where the post/gap shadow lands.
                isolated_fence_items = (
                    scene.render_resources.sprite_items,
                    scene.render_resources.immediate_entities,
                )
                scene.render_resources.sprite_items = []
                scene.render_resources.immediate_entities = []
            _look_at(scene.camera, position, target)
            legacy_before = _capture(
                scene,
                "legacy",
                args.width,
                args.height,
                args.warmup_frames,
            )
            from engine.core.compat_shader import (  # noqa: PLC0415
                get_texture_color_exposure_scale,
                get_texture_lighting_state,
            )

            legacy_state_before_packet = get_texture_lighting_state()
            legacy_direction_before_packet = list(
                legacy_state_before_packet.light_direction
            )
            legacy_updates_before_packet = int(
                scene.lighting_controller.diagnostics.shader_state_updates
            )
            aliases_before_packet = int(
                scene.lighting_controller.diagnostics.legacy_alias_projections
            )
            packet = _capture(
                scene,
                "packet",
                args.width,
                args.height,
                args.warmup_frames,
            )
            packet_environment_absent = (
                scene.lighting_controller.render_environment_snapshot is None
                and all(
                    packet_value.environment is None
                    for packet_value in scene.lighting_controller.render_packets.values()
                )
            )
            aliases_after_packet = int(
                scene.lighting_controller.diagnostics.legacy_alias_projections
            )
            legacy_regions_after_packet = len(
                scene.lighting_controller.legacy_covered_regions
            )
            alias_sample_passed = (
                aliases_after_packet == aliases_before_packet
                and legacy_regions_after_packet == 0
                and not hasattr(scene.lighting, "covered_regions")
                and packet_environment_absent
            )
            report["packet_alias_contract"]["samples"].append(
                {
                    "viewpoint": name,
                    "before_packet": aliases_before_packet,
                    "after_packet": aliases_after_packet,
                    "legacy_regions_after_packet": legacy_regions_after_packet,
                    "render_environment_absent": packet_environment_absent,
                    "passed": alias_sample_passed,
                }
            )
            report["packet_alias_contract"]["passed"] = bool(
                report["packet_alias_contract"]["passed"]
                and alias_sample_passed
            )
            legacy_after = _capture(
                scene,
                "legacy",
                args.width,
                args.height,
                args.warmup_frames,
            )
            if "backend_inputs" not in report:
                legacy_state = get_texture_lighting_state()
                ground_packet = scene.lighting_controller.render_packet_for(
                    scene.render_resources.ground_mesh.lighting_receiver
                )
                report["backend_inputs"] = {
                    "camera_brightness": float(scene.camera.brightness_default),
                    "legacy_base_brightness": float(
                        legacy_state.base_brightness
                    ),
                    "legacy_exposure_scale": float(
                        get_texture_color_exposure_scale()
                    ),
                    "legacy_light_direction": list(
                        legacy_state.light_direction
                    ),
                    "legacy_light_direction_before_packet": (
                        legacy_direction_before_packet
                    ),
                    "packet_exposure": float(ground_packet.exposure),
                    "packet_local_reference": float(
                        ground_packet.local_light_reference
                    ),
                    "packet_light_direction": list(
                        ground_packet.directional.light_direction
                    ),
                    "controller_shader_state_updates": int(
                        scene.lighting_controller.diagnostics.shader_state_updates
                    ),
                    "controller_updates_before_packet": (
                        legacy_updates_before_packet
                    ),
                    "controller_shader_uniform_uploads": int(
                        scene.lighting_controller.diagnostics.shader_uniform_uploads
                    ),
                    "controller_uniform_sync_cache_hits": int(
                        scene.lighting_controller.diagnostics.uniform_sync_cache_hits
                    ),
                }
            drift = compare_rgb_frames(
                legacy_before,
                legacy_after,
                thresholds=thresholds,
            )
            parity = compare_rgb_frames(
                legacy_before,
                packet,
                drift_reference=legacy_after,
                thresholds=thresholds,
            )
            all_passed = (
                all_passed
                and alias_sample_passed
                and drift.passed
            )
            viewpoint_report = {
                "camera_position": list(position),
                "camera_target": list(target),
                "legacy_drift": drift.to_dict(),
                "packet_parity": parity.to_dict(),
                "packet_parity_acceptance": "diagnostic_only",
            }
            if fixture_mode and name == "window_interior":
                window_probe = _window_transmission_probe(packet)
                viewpoint_report["window_transmission"] = window_probe
                all_passed = all_passed and bool(window_probe["passed"])
            if fixture_mode and name == "torch_closeup":
                torch_probe = _torch_emissive_probe(packet)
                viewpoint_report["torch_emissive"] = torch_probe
                all_passed = all_passed and bool(torch_probe["passed"])
            if fixture_mode and name == "fence_shadow":
                fence_probe = _fence_shadow_gap_probe(packet)
                viewpoint_report["fence_shadow_gaps"] = fence_probe
                all_passed = all_passed and bool(fence_probe["passed"])
            report["viewpoints"][name] = viewpoint_report

            _save_rgb(output_dir / f"{name}_legacy.png", legacy_before)
            _save_rgb(output_dir / f"{name}_packet.png", packet)
            _save_rgb(output_dir / f"{name}_legacy_after.png", legacy_after)
            _save_rgb(
                output_dir / f"{name}_difference_x8.png",
                amplified_rgb_difference(legacy_before, packet, scale=8),
            )
            if isolated_fence_items is not None:
                (
                    scene.render_resources.sprite_items,
                    scene.render_resources.immediate_entities,
                ) = isolated_fence_items

        shadowed_point_light_count = sum(
            1
            for light in scene.lighting.point_lights
            if light.casts_shadows and light.intensity > 0.0 and light.range > 0.0
        )
        expected_point_shadow_maps = min(
            scene.renderer.max_point_shadows,
            shadowed_point_light_count,
        )
        raster_shadow_contract = {
            "sun_shadow_map_created": scene.renderer._sun_shadow_map is not None,
            "point_shadow_map_count": len(scene.renderer._point_shadow_maps),
            "expected_point_shadow_map_count": expected_point_shadow_maps,
        }
        raster_shadow_contract["passed"] = bool(
            raster_shadow_contract["sun_shadow_map_created"]
            and raster_shadow_contract["point_shadow_map_count"]
            >= expected_point_shadow_maps
        )
        report["raster_shadow_resource_contract"] = raster_shadow_contract
        all_passed = all_passed and bool(raster_shadow_contract["passed"])

        shader = get_packet_texture_lighting_shader()
        if shader is not None:
            limits = shader.storage.limits
            report["packet_device_limits"] = {
                "max_texture_size": limits.max_texture_size,
                "fragment_texture_units": limits.texture_image_units,
                "vertex_texture_units": limits.vertex_texture_image_units,
                "local_lights": limits.local_lights,
                "environment_regions": limits.environment_regions,
                "environment_portals": limits.environment_portals,
            }
        report["passed"] = all_passed
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        if scene is not None:
            scene.dispose()
        reset_packet_texture_lighting_shader()
        pygame.quit()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "lighting_scene_ab"),
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--seed", type=int, default=7349)
    parser.add_argument(
        "--world-mode",
        choices=("fixture", "generated"),
        default="fixture",
    )
    parser.add_argument("--building-count", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=1)
    parser.add_argument("--channel-tolerance", type=int, default=2)
    parser.add_argument("--min-stable-pixel-ratio", type=float, default=0.995)
    parser.add_argument("--max-mean-absolute-error", type=float, default=1.0)
    parser.add_argument("--max-p99-absolute-error", type=float, default=2.0)
    parser.add_argument("--max-absolute-error", type=int, default=16)
    parser.add_argument("--max-changed-pixel-ratio", type=float, default=0.002)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.world_mode == "generated" and args.building_count < 1:
        raise SystemExit("--building-count must be at least 1 in generated mode")
    report = capture(args)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] or not args.require_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
