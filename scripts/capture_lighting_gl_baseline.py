"""Capture deterministic compatibility-shader lighting samples as JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pygame  # noqa: E402
import numpy as np  # noqa: E402
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
    glDeleteTextures,
    glDisable,
    glEnd,
    glEnable,
    glFinish,
    glGenTextures,
    glGetString,
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
    GL_RENDERER,
    GL_VERSION,
)

from engine.core.compat_shader import (  # noqa: E402
    get_texture_color_exposure_shader,
    set_texture_fog_state,
    set_texture_lighting_state,
    set_texture_shine_state,
    set_texture_vibrance_state,
    use_fixed_pipeline,
)
from engine.rendering.lighting import (  # noqa: E402
    apply_brightness_modifiers,
    covered_region_factor_at,
    sunlight_factor_for_normal,
)


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _pixel_rgb() -> tuple[int, int, int]:
    value = glReadPixels(32, 32, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
    if hasattr(value, "tobytes"):
        value = value.tobytes()
    raw = bytes(value)
    return int(raw[0]), int(raw[1]), int(raw[2])


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


def _draw_shader_sample(
    shader,
    texture: int,
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.8),
    normal: tuple[float, float, float] = (0.0, 1.0, 0.0),
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    scene_lighting: bool = True,
    directional: bool = True,
    environment: bool = True,
) -> tuple[int, int, int]:
    glClear(GL_COLOR_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(-position[0], -position[1], 0.0)
    glBindTexture(GL_TEXTURE_2D, texture)
    shader.bind(
        scene_lighting_enabled=scene_lighting,
        directional_enabled=directional,
        environment_enabled=environment,
        shine_enabled=False,
    )
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
    return _pixel_rgb()


def _draw_fallback_sample(
    texture: int,
    color: tuple[float, float, float],
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.8),
) -> tuple[int, int, int]:
    glClear(GL_COLOR_BUFFER_BIT)
    use_fixed_pipeline()
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(-position[0], -position[1], 0.0)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)
    glBegin(GL_QUADS)
    for x, y, u, v in (
        (-0.18, -0.18, 0.0, 0.0),
        (0.18, -0.18, 1.0, 0.0),
        (0.18, 0.18, 1.0, 1.0),
        (-0.18, 0.18, 0.0, 1.0),
    ):
        glColor4f(color[0], color[1], color[2], 1.0)
        glTexCoord2f(u, v)
        glVertex3f(position[0] + x, position[1] + y, position[2])
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glFinish()
    return _pixel_rgb()


def _scaled_color(
    color: tuple[float, float, float],
    factor: float,
) -> tuple[float, float, float]:
    return tuple(float(channel) * float(factor) for channel in color)


def _capture_receiver_samples(shader, texture: int) -> dict[str, object]:
    region = {
        "min_x": -0.3,
        "max_x": 0.3,
        "min_z": -0.3,
        "max_z": 0.3,
        "factor": 0.34,
        "openings": [],
    }
    sun_direction = (-1.0, -1.0, -1.0)
    torch_area = {
        "light_id": "fixture:torch",
        "center": (0.25, 0.0, 0.25),
        "radius": 0.12,
        "value": 3.4,
        "falloff": 2.2,
        "bounds": (-0.3, 0.3, -0.3, 0.3),
        "indoor_only": True,
        "floor_scale": 0.28,
    }
    set_texture_lighting_state(
        base_brightness=1.0,
        sun_direction=sun_direction,
        brightness_areas=[torch_area],
        covered_regions=[region],
        ambient=0.2,
        diffuse=0.8,
        max_factor=1.0,
        exposure_scale=1.0,
        compile_shader=False,
    )

    exterior = (0.65, 0.0, 0.65)
    interior = (0.0, 0.0, 0.0)
    fake_sprite_normal = (-0.35, 1.0, -0.35)
    receivers = {
        "ground_exterior": {
            "position": exterior,
            "normal": (0.0, 1.0, 0.0),
            "color": (0.72, 0.82, 0.62),
            "scene": True,
            "directional": True,
            "environment": True,
        },
        "ground_interior": {
            "position": interior,
            "normal": (0.0, 1.0, 0.0),
            "color": (0.72, 0.82, 0.62),
            "scene": True,
            "directional": True,
            "environment": True,
        },
        "ground_torch_lit": {
            "position": (0.25, 0.0, 0.25),
            "normal": (0.0, 1.0, 0.0),
            "color": (0.72, 0.82, 0.62),
            "scene": True,
            "directional": True,
            "environment": True,
            "local": True,
        },
        "road": {
            "position": exterior,
            "normal": (0.0, 1.0, 0.0),
            "color": (0.48, 0.46, 0.42),
            "scene": True,
            "directional": True,
            "environment": True,
        },
        "fence": {
            "position": exterior,
            "normal": (1.0, 0.0, 0.0),
            "color": (0.58, 0.46, 0.32),
            "scene": True,
            "directional": True,
            "environment": True,
        },
        "wall_north": {
            "position": exterior,
            "normal": (0.0, 0.0, 1.0),
            "color": (0.75, 0.65, 0.55),
            "scene": True,
            "directional": True,
            "environment": False,
        },
        "wall_east": {
            "position": exterior,
            "normal": (1.0, 0.0, 0.0),
            "color": (0.75, 0.65, 0.55),
            "scene": True,
            "directional": True,
            "environment": False,
        },
        "wall_south": {
            "position": exterior,
            "normal": (0.0, 0.0, -1.0),
            "color": (0.75, 0.65, 0.55),
            "scene": True,
            "directional": True,
            "environment": False,
        },
        "wall_west": {
            "position": exterior,
            "normal": (-1.0, 0.0, 0.0),
            "color": (0.75, 0.65, 0.55),
            "scene": True,
            "directional": True,
            "environment": False,
        },
        "door_slab": {
            "position": interior,
            "normal": (0.0, 0.0, 1.0),
            "color": (0.52, 0.34, 0.18),
            "scene": False,
            "directional": False,
            "environment": False,
            "cpu_directional": True,
        },
        "window_slab": {
            "position": interior,
            "normal": (1.0, 0.0, 0.0),
            "color": (0.42, 0.62, 0.82),
            "scene": False,
            "directional": False,
            "environment": False,
            "cpu_directional": True,
        },
        "chest": {
            "position": interior,
            "normal": (0.0, 1.0, 0.0),
            "color": (0.55, 0.36, 0.16),
            "scene": False,
            "directional": False,
            "environment": False,
            "cpu_directional": True,
        },
        "polygon": {
            "position": exterior,
            "normal": (0.0, 0.0, -1.0),
            "color": (0.68, 0.30, 0.50),
            "scene": False,
            "directional": False,
            "environment": False,
            "cpu_directional": True,
        },
        "sprite": {
            "position": exterior,
            "normal": fake_sprite_normal,
            "color": (0.40, 0.80, 0.40),
            "scene": True,
            "directional": True,
            "environment": False,
        },
        "torch_sprite": {
            "position": exterior,
            "normal": fake_sprite_normal,
            "color": (1.0, 0.82, 0.55),
            "scene": True,
            "directional": True,
            "environment": False,
        },
        "decal": {
            "position": exterior,
            "normal": (0.0, 1.0, 0.0),
            "color": (0.24, 0.24, 0.24),
            "scene": False,
            "directional": False,
            "environment": False,
        },
        "sky_clear": {
            "position": exterior,
            "normal": (0.0, 1.0, 0.0),
            "color": (0.70, 0.80, 1.0),
            "fixed": True,
        },
        "sun_billboard": {
            "position": exterior,
            "normal": (0.0, 1.0, 0.0),
            "color": (1.0, 0.96, 0.86),
            "fixed": True,
        },
        "cloud": {
            "position": exterior,
            "normal": (0.0, 1.0, 0.0),
            "color": (1.0, 0.992, 0.9762),
            "fixed": True,
        },
    }

    shader_samples: dict[str, list[int]] = {}
    fallback_samples: dict[str, list[int]] = {}
    policies: dict[str, dict[str, bool]] = {}
    for name, spec in receivers.items():
        position = spec["position"]
        normal = spec["normal"]
        base_color = spec["color"]
        fixed = bool(spec.get("fixed", False))
        scene_lighting = bool(spec.get("scene", False))
        directional = bool(spec.get("directional", False))
        environment = bool(spec.get("environment", False))
        cpu_directional = bool(spec.get("cpu_directional", directional))
        cpu_environment = bool(spec.get("cpu_environment", environment))
        local = bool(spec.get("local", False))

        fallback_color = base_color
        if cpu_directional:
            fallback_color = _scaled_color(
                fallback_color,
                sunlight_factor_for_normal(
                    normal,
                    sun_direction=sun_direction,
                    ambient=0.2,
                    diffuse=0.8,
                    max_factor=1.0,
                ),
            )
        if cpu_environment:
            fallback_color = _scaled_color(
                fallback_color,
                covered_region_factor_at(
                    position[0],
                    position[2],
                    covered_regions=[region],
                ),
            )
        if local:
            receiver_factor = covered_region_factor_at(
                position[0],
                position[2],
                covered_regions=[region],
            )
            vertex = np.array(
                [
                    [
                        position[0],
                        position[1],
                        position[2],
                        fallback_color[0],
                        fallback_color[1],
                        fallback_color[2],
                    ]
                ],
                dtype=np.float32,
            )
            apply_brightness_modifiers(
                vertex,
                modifiers=[torch_area],
                default_brightness=1.0,
                receiver_mask=np.array([True]),
                receiver_factors=np.array([receiver_factor], dtype=np.float32),
                surface_floor_mask=np.array([True]),
            )
            fallback_color = tuple(float(value) for value in vertex[0, 3:6])

        shader_color = fallback_color if not scene_lighting else base_color
        if fixed:
            shader_samples[name] = list(
                _draw_fallback_sample(texture, base_color, position=position)
            )
        else:
            shader_samples[name] = list(
                _draw_shader_sample(
                    shader,
                    texture,
                    position=position,
                    normal=normal,
                    color=shader_color,
                    scene_lighting=scene_lighting,
                    directional=directional,
                    environment=environment,
                )
            )
        fallback_samples[name] = list(
            _draw_fallback_sample(
                texture,
                fallback_color if not fixed else base_color,
                position=position,
            )
        )
        policies[name] = {
            "scene": scene_lighting,
            "directional": directional,
            "environment": environment,
            "local_fixture": local,
            "fixed": fixed,
        }

    return {
        "policies": policies,
        "shader_samples": shader_samples,
        "fallback_samples": fallback_samples,
    }


def capture() -> dict[str, object]:
    pygame.init()
    texture = 0
    try:
        pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.HIDDEN)
        glViewport(0, 0, 64, 64)
        glDisable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        shader = get_texture_color_exposure_shader()
        if shader is None:
            return {"status": "unavailable", "reason": "shader compilation failed"}
        texture = _white_texture()
        set_texture_fog_state(enabled=False, compile_shader=False)
        set_texture_vibrance_state(1.0, compile_shader=False)
        set_texture_shine_state(enabled=False, compile_shader=False)

        shader_samples: dict[str, dict[str, list[int]]] = {}
        fallback_samples: dict[str, dict[str, list[int]]] = {}
        for exposure in (0.5, 1.0, 1.5):
            shader_exposure_samples: dict[str, list[int]] = {}
            fallback_exposure_samples: dict[str, list[int]] = {}
            for label, edge_factor in (
                ("closed", 0.34),
                ("half", 0.67),
                ("open", 1.0),
            ):
                doorway = {
                    "type": "doorway",
                    "side": "north",
                    "center_x": 0.0,
                    "center_z": 0.8,
                    "width": 0.4,
                    "depth": 0.6,
                    "side_fade": 0.1,
                    "edge_factor": edge_factor,
                }
                region = {
                    "min_x": -0.8,
                    "max_x": 0.8,
                    "min_z": -0.8,
                    "max_z": 0.8,
                    "factor": 0.34,
                    "doorway": doorway,
                    "windows": [],
                    "openings": [doorway],
                }
                set_texture_lighting_state(
                    base_brightness=exposure,
                    sun_direction=(0.0, -1.0, 0.0),
                    brightness_areas=(),
                    covered_regions=[region],
                    ambient=1.0,
                    diffuse=0.0,
                    max_factor=1.0,
                    exposure_scale=1.0,
                    compile_shader=False,
                )
                shader_exposure_samples[label] = list(
                    _draw_shader_sample(shader, texture)
                )
                receiver_factor = covered_region_factor_at(
                    0.0,
                    0.8,
                    covered_regions=[region],
                )
                vertex = np.array(
                    [
                        [
                            0.0,
                            0.0,
                            0.8,
                            receiver_factor,
                            receiver_factor,
                            receiver_factor,
                        ]
                    ],
                    dtype=np.float32,
                )
                apply_brightness_modifiers(
                    vertex,
                    modifiers=(),
                    default_brightness=exposure,
                )
                fallback_exposure_samples[label] = list(
                    _draw_fallback_sample(
                        texture,
                        tuple(float(value) for value in vertex[0, 3:6]),
                    )
                )
            key = f"{exposure:.1f}"
            shader_samples[key] = shader_exposure_samples
            fallback_samples[key] = fallback_exposure_samples

        receiver_samples = _capture_receiver_samples(shader, texture)

        return {
            "status": "ok",
            "version": _text(glGetString(GL_VERSION)),
            "renderer": _text(glGetString(GL_RENDERER)),
            "shader_samples": shader_samples,
            "fallback_samples": fallback_samples,
            "receiver_samples": receiver_samples,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if texture:
            try:
                glDeleteTextures([texture])
            except Exception:
                pass
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    print(json.dumps(capture(), sort_keys=True))
