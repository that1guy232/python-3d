"""Profile packet-lighting record scans in a hidden OpenGL context."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from statistics import median
import sys
import time


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from OpenGL.GL import (  # noqa: E402
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_RENDERER,
    GL_RGBA,
    GL_SHADING_LANGUAGE_VERSION,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    GL_VENDOR,
    GL_VERSION,
    glBegin,
    glBindTexture,
    glClear,
    glClearColor,
    glColor4f,
    glDisable,
    glEnd,
    glFinish,
    glGenTextures,
    glGetString,
    glLoadIdentity,
    glMatrixMode,
    glNormal3f,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex3f,
    glViewport,
)

from engine.core.gl_state import use_fixed_pipeline  # noqa: E402
from engine.rendering.lighting_adapter import RenderLightingAdapter  # noqa: E402
from engine.rendering.lighting_state import (  # noqa: E402
    DirectionalLightSnapshot,
    LightingSnapshot,
    LocalBrightnessLight,
)
from engine.rendering.packet_shader import (  # noqa: E402
    get_packet_texture_lighting_shader,
    reset_packet_texture_lighting_shader,
)
from engine.rendering.render_environment import (  # noqa: E402
    RenderEnvironmentPortal,
    RenderEnvironmentRegion,
    RenderEnvironmentSnapshot,
)
from engine.render_style_state import (  # noqa: E402
    update_render_fog_state,
    update_render_shine_state,
    update_render_vibrance_state,
)
from game.world.lighting_receivers import GROUND_LIGHTING_RECEIVER  # noqa: E402


@dataclass(frozen=True, slots=True)
class ProfileScenario:
    name: str
    local_lights: int
    regions: int
    portals: int
    region_layout: str = "partitioned"

    @property
    def record_bytes(self) -> int:
        texels = self.local_lights * 3 + self.regions * 2 + self.portals * 2
        return texels * 4 * 4

    @property
    def estimated_record_visits_per_fragment(self) -> int:
        portal_visits = 0
        if self.regions and self.portals:
            portal_visits = (
                self.regions * self.portals
                if self.region_layout == "overlapping"
                else self.portals
            )
        return self.local_lights + self.regions + portal_visits


def default_scenarios() -> tuple[ProfileScenario, ...]:
    """Return representative and deliberately scaled shader workloads."""

    return (
        ProfileScenario("baseline", 0, 0, 0),
        ProfileScenario("generated_reference", 17, 3, 13),
        ProfileScenario("local_64", 64, 0, 0),
        ProfileScenario("local_128", 128, 0, 0),
        ProfileScenario("local_256", 256, 0, 0),
        ProfileScenario("environment_8x4", 0, 8, 32),
        ProfileScenario("environment_16x4", 0, 16, 64),
        ProfileScenario("combined_64_16x4", 64, 16, 64),
        ProfileScenario("overlap_8x4", 0, 8, 32, "overlapping"),
    )


def _directional() -> DirectionalLightSnapshot:
    return DirectionalLightSnapshot(
        sun_position=(1.0, 2.0, 1.0),
        sun_target=(0.0, 0.0, 0.0),
        sun_direction=(-0.4, -0.8, -0.4),
        light_direction=(0.4, 0.8, 0.4),
        ambient=0.52,
        diffuse=0.48,
        max_factor=1.0,
        tint=(1.0, 1.0, 1.0),
    )


def _lights(count: int) -> tuple[LocalBrightnessLight, ...]:
    return tuple(
        LocalBrightnessLight(
            light_id=f"profile:light:{index}",
            center=(0.0, 0.0, 0.0),
            radius=4.0,
            value=1.25 + (index % 3) * 0.05,
            falloff=1.5,
            floor_scale=0.75,
        )
        for index in range(count)
    )


def _portal_counts(region_count: int, portal_count: int) -> tuple[int, ...]:
    if region_count <= 0:
        return ()
    quotient, remainder = divmod(portal_count, region_count)
    return tuple(
        quotient + int(index < remainder) for index in range(region_count)
    )


def _environment(scenario: ProfileScenario) -> RenderEnvironmentSnapshot:
    portal_counts = _portal_counts(scenario.regions, scenario.portals)
    regions: list[RenderEnvironmentRegion] = []
    for region_index, region_portals in enumerate(portal_counts):
        if scenario.region_layout == "overlapping":
            min_x, max_x = -1.0, 1.0
        else:
            width = 2.0 / max(1, scenario.regions)
            min_x = -1.0 + region_index * width
            max_x = min_x + width
        portals = tuple(
            RenderEnvironmentPortal(
                portal_id=f"profile:portal:{region_index}:{portal_index}",
                kind="window",
                side="north",
                center_x=(min_x + max_x) * 0.5,
                center_z=1.0,
                width=max(0.05, (max_x - min_x) * 0.4),
                depth=0.25,
                side_fade=0.1,
                factor=0.86,
            )
            for portal_index in range(region_portals)
        )
        regions.append(
            RenderEnvironmentRegion(
                volume_id=f"profile:region:{region_index}",
                min_x=min_x,
                max_x=max_x,
                min_z=-1.0,
                max_z=1.0,
                indoor_factor=0.34,
                portals=portals,
            )
        )
    return RenderEnvironmentSnapshot(tuple(regions))


def _packet(scenario: ProfileScenario, revision: int):
    snapshot = LightingSnapshot(
        revision=revision,
        base_brightness=0.8,
        sky_color=(0.7, 0.8, 1.0, 1.0),
        directional=_directional(),
        local_lights=_lights(scenario.local_lights),
    )
    return RenderLightingAdapter().packet_for(
        snapshot,
        GROUND_LIGHTING_RECEIVER,
        _environment(scenario),
    )


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
        bytes((180, 200, 160, 255)),
    )
    return texture


def _draw(shader, packet, texture: int) -> None:
    glClear(GL_COLOR_BUFFER_BIT)
    shader.bind(packet)
    glBindTexture(GL_TEXTURE_2D, texture)
    glBegin(GL_QUADS)
    for x, y, u, v in (
        (-1.0, -1.0, 0.0, 0.0),
        (1.0, -1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (-1.0, 1.0, 0.0, 1.0),
    ):
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glNormal3f(0.0, 1.0, 0.0)
        glTexCoord2f(u, v)
        glVertex3f(x, y, 0.0)
    glEnd()
    use_fixed_pipeline()


def _percentile(samples: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), percentile))


def _measure_scenario(
    shader,
    texture: int,
    scenario: ProfileScenario,
    *,
    revision: int,
    width: int,
    height: int,
    warmup_frames: int,
    measured_frames: int,
    draws_per_sample: int,
    baseline_median_ms: float,
) -> dict[str, object]:
    packet = _packet(scenario, revision)
    started = time.perf_counter_ns()
    _draw(shader, packet, texture)
    glFinish()
    first_frame_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    for _ in range(warmup_frames):
        for _draw_index in range(draws_per_sample):
            _draw(shader, packet, texture)
        glFinish()

    samples: list[float] = []
    for _ in range(measured_frames):
        started = time.perf_counter_ns()
        for _draw_index in range(draws_per_sample):
            _draw(shader, packet, texture)
        glFinish()
        batch_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        samples.append(batch_ms / draws_per_sample)

    median_ms = float(median(samples))
    pixel_count = width * height
    incremental_ms = max(0.0, median_ms - baseline_median_ms)
    return {
        **asdict(scenario),
        "record_bytes": scenario.record_bytes,
        "estimated_record_visits_per_fragment": (
            scenario.estimated_record_visits_per_fragment
        ),
        "first_frame_ms": first_frame_ms,
        "draws_per_sample": draws_per_sample,
        "steady_frame_ms": {
            "minimum": min(samples),
            "median": median_ms,
            "p95": _percentile(samples, 95),
            "maximum": max(samples),
        },
        "incremental_over_baseline_ms": incremental_ms,
        "median_nanoseconds_per_fragment": median_ms * 1_000_000.0 / pixel_count,
        "incremental_nanoseconds_per_fragment": (
            incremental_ms * 1_000_000.0 / pixel_count
        ),
    }


def _decode_gl_string(name: int) -> str:
    value = glGetString(name)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def profile(args: argparse.Namespace) -> dict[str, object]:
    if args.width < 1 or args.height < 1:
        raise ValueError("profile resolution must be positive")
    if args.frames < 1:
        raise ValueError("--frames must be at least 1")
    if args.warmup_frames < 1:
        raise ValueError("--warmup-frames must be at least 1")
    if args.draws_per_sample < 1:
        raise ValueError("--draws-per-sample must be at least 1")

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 0)
    pygame.display.set_mode(
        (args.width, args.height),
        pygame.OPENGL | pygame.HIDDEN,
    )
    glViewport(0, 0, args.width, args.height)
    glDisable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    reset_packet_texture_lighting_shader()

    try:
        update_render_fog_state(enabled=False)
        update_render_shine_state(enabled=False)
        update_render_vibrance_state(1.0)
        compile_started = time.perf_counter_ns()
        shader = get_packet_texture_lighting_shader()
        glFinish()
        compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
        if shader is None:
            raise RuntimeError("packet lighting shader is unavailable")
        texture = _white_texture()
        scenarios = default_scenarios()
        results: list[dict[str, object]] = []
        baseline_median_ms = 0.0
        for revision, scenario in enumerate(scenarios, start=1):
            result = _measure_scenario(
                shader,
                texture,
                scenario,
                revision=revision,
                width=args.width,
                height=args.height,
                warmup_frames=args.warmup_frames,
                measured_frames=args.frames,
                draws_per_sample=args.draws_per_sample,
                baseline_median_ms=baseline_median_ms,
            )
            if scenario.name == "baseline":
                baseline_median_ms = float(result["steady_frame_ms"]["median"])
                result["incremental_over_baseline_ms"] = 0.0
                result["incremental_nanoseconds_per_fragment"] = 0.0
            results.append(result)

        reference = next(
            result for result in results if result["name"] == "generated_reference"
        )
        reference_ms = float(reference["steady_frame_ms"]["median"])
        target_frame_ms = 1000.0 / args.target_fps
        reference_budget_ms = target_frame_ms * args.reference_budget_share
        report = {
            "schema_version": 1,
            "device": {
                "vendor": _decode_gl_string(GL_VENDOR),
                "renderer": _decode_gl_string(GL_RENDERER),
                "opengl_version": _decode_gl_string(GL_VERSION),
                "glsl_version": _decode_gl_string(GL_SHADING_LANGUAGE_VERSION),
                "max_texture_size": shader.storage.limits.max_texture_size,
                "fragment_texture_units": shader.storage.limits.texture_image_units,
                "vertex_texture_units": (
                    shader.storage.limits.vertex_texture_image_units
                ),
            },
            "resolution": [args.width, args.height],
            "pixel_count": args.width * args.height,
            "warmup_frames": args.warmup_frames,
            "measured_samples": args.frames,
            "draws_per_sample": args.draws_per_sample,
            "shader_creation_ms": compile_ms,
            "target_fps": args.target_fps,
            "target_frame_ms": target_frame_ms,
            "generated_reference_budget": {
                "frame_share": args.reference_budget_share,
                "budget_ms": reference_budget_ms,
                "measured_median_ms": reference_ms,
                "passed": reference_ms <= reference_budget_ms,
            },
            "results": results,
        }
        report["passed"] = bool(report["generated_reference_budget"]["passed"])
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        reset_packet_texture_lighting_shader()
        pygame.quit()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "lighting_packet_profile.json"),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--draws-per-sample", type=int, default=20)
    parser.add_argument("--target-fps", type=float, default=61.0)
    parser.add_argument("--reference-budget-share", type=float, default=0.25)
    parser.add_argument("--require-budget", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.target_fps <= 0.0:
        raise SystemExit("--target-fps must be positive")
    if not 0.0 < args.reference_budget_share <= 1.0:
        raise SystemExit("--reference-budget-share must be in (0, 1]")
    report = profile(args)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] or not args.require_budget else 2


if __name__ == "__main__":
    raise SystemExit(main())
