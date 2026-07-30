"""Procedural beach and stylized water surrounding the playable world."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_FLOAT,
    GL_STATIC_DRAW,
    GL_TEXTURE_2D,
    GL_TRIANGLES,
    GL_TRUE,
    GL_VERTEX_ARRAY,
    glBindBuffer,
    glBufferData,
    glDeleteBuffers,
    glDeleteProgram,
    glDisable,
    glDisableClientState,
    glDepthMask,
    glDrawArrays,
    glEnable,
    glEnableClientState,
    glGenBuffers,
    glGetUniformLocation,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniform4f,
    glUseProgram,
    glVertexPointer,
)

from engine.core.gl_state import use_fixed_pipeline
from engine.core.mesh import BatchedMesh
from engine.render_style_state import get_render_fog_state
from engine.rendering.gl_program import compile_program


Bounds = tuple[float, float, float, float]
HeightFn = Callable[[float, float], float]


WATER_VERTEX_SOURCE = r"""#version 120
uniform float u_time;

varying vec3 v_world_position;
varying float v_fog_distance;
varying vec3 v_normal;

void main()
{
    vec4 world = gl_Vertex;
    float phase_a = world.x * 0.0065 + world.z * 0.0028 + u_time * 0.48;
    float phase_b = world.z * 0.0080 - world.x * 0.0018 - u_time * 0.38;
    float phase_c = (world.x + world.z) * 0.0160 + u_time * 0.85;

    float wave_a = sin(phase_a);
    float wave_b = cos(phase_b);
    float wave_c = sin(phase_c);
    world.y += wave_a * 0.72 + wave_b * 0.48 + wave_c * 0.16;

    float dhdx = cos(phase_a) * 0.0065 * 0.72
        + sin(phase_b) * 0.0018 * 0.48
        + cos(phase_c) * 0.0160 * 0.16;
    float dhdz = cos(phase_a) * 0.0028 * 0.72
        - sin(phase_b) * 0.0080 * 0.48
        + cos(phase_c) * 0.0160 * 0.16;

    vec4 eye_position = gl_ModelViewMatrix * world;
    gl_Position = gl_ProjectionMatrix * eye_position;
    v_world_position = world.xyz;
    v_fog_distance = length(eye_position.xyz);
    v_normal = normalize(vec3(-dhdx, 1.0, -dhdz));
}
"""


WATER_FRAGMENT_SOURCE = r"""#version 120
uniform float u_time;
uniform vec3 u_camera_position;
uniform vec3 u_light_direction;
uniform vec3 u_sun_tint;
uniform vec4 u_terrain_bounds;
uniform vec4 u_water_bounds;
uniform float u_shore_width;
uniform float u_horizon_fade_width;
uniform float u_exposure;
uniform int u_fog_enabled;
uniform float u_fog_density;
uniform vec4 u_fog_color;

varying vec3 v_world_position;
varying float v_fog_distance;
varying vec3 v_normal;

float hash21(vec2 point)
{
    return fract(sin(dot(point, vec2(127.1, 311.7))) * 43758.5453);
}

float value_noise(vec2 point)
{
    vec2 cell = floor(point);
    vec2 local = fract(point);
    local = local * local * (3.0 - 2.0 * local);
    float a = hash21(cell);
    float b = hash21(cell + vec2(1.0, 0.0));
    float c = hash21(cell + vec2(0.0, 1.0));
    float d = hash21(cell + vec2(1.0, 1.0));
    return mix(mix(a, b, local.x), mix(c, d, local.x), local.y);
}

float outside_rect_distance(vec2 point)
{
    vec2 low = vec2(u_terrain_bounds.x, u_terrain_bounds.z);
    vec2 high = vec2(u_terrain_bounds.y, u_terrain_bounds.w);
    vec2 delta = max(max(low - point, point - high), vec2(0.0));
    return max(delta.x, delta.y);
}

float outer_edge_distance(vec2 point)
{
    vec2 low = vec2(u_water_bounds.x, u_water_bounds.z);
    vec2 high = vec2(u_water_bounds.y, u_water_bounds.w);
    vec2 inset = min(point - low, high - point);
    return max(min(inset.x, inset.y), 0.0);
}

void main()
{
    float shore_distance = outside_rect_distance(v_world_position.xz);
    float deep_mix = smoothstep(
        u_shore_width * 0.80,
        u_shore_width + 760.0,
        shore_distance
    );

    vec3 shallow_color = vec3(0.055, 0.38, 0.40);
    vec3 deep_color = vec3(0.018, 0.105, 0.20);
    vec3 color = mix(shallow_color, deep_color, deep_mix);

    vec2 surface_point = v_world_position.xz;
    float drift = value_noise(
        surface_point * 0.010 + vec2(u_time * 0.025, -u_time * 0.018)
    );
    float ripple_a = 0.5 + 0.5 * sin(
        dot(surface_point, vec2(0.018, 0.007))
        + u_time * 0.62
        + drift * 1.10
    );
    float ripple_b = 0.5 + 0.5 * sin(
        dot(surface_point, vec2(-0.011, 0.021))
        - u_time * 0.46
        + drift * 0.75
    );
    float crest_a = smoothstep(0.91, 0.985, ripple_a);
    float crest_b = smoothstep(0.94, 0.992, ripple_b);
    float broad_swell = 0.5 + 0.5 * sin(
        dot(surface_point, vec2(0.0034, 0.0052))
        + u_time * 0.13
        + drift * 1.15
    );
    color += vec3(0.004, 0.010, 0.013) * (broad_swell - 0.50);
    float ripple_light = crest_a * 0.075 + crest_b * 0.040;
    color += vec3(0.14, 0.32, 0.31) * ripple_light;

    float shore_offset = shore_distance - u_shore_width;
    float foam_envelope = 1.0 - smoothstep(
        12.0,
        78.0,
        abs(shore_offset)
    );
    float foam_noise = value_noise(
        surface_point * 0.018 + vec2(-u_time * 0.06, u_time * 0.035)
    );
    float breaker = 0.5 + 0.5 * sin(
        shore_offset * 0.105
        + dot(surface_point, vec2(0.020, 0.015))
        - u_time * 1.18
        + foam_noise * 3.2
    );
    float along_shore = 0.5 + 0.5 * sin(
        dot(surface_point, vec2(-0.010, 0.016))
        + u_time * 0.24
        + foam_noise * 2.1
    );
    float foam_breakup = mix(0.20, 1.0, smoothstep(0.30, 0.76, along_shore));
    float foam = foam_envelope * smoothstep(0.70, 0.94, breaker) * foam_breakup;
    float wash = (1.0 - smoothstep(0.0, 28.0, abs(shore_offset + 18.0))) * 0.065;
    color = mix(color, vec3(0.72, 0.84, 0.76), foam * 0.68 + wash);

    vec3 normal = normalize(v_normal);
    vec3 view_direction = normalize(u_camera_position - v_world_position);
    vec3 light_direction = normalize(u_light_direction);
    vec3 reflected = reflect(-light_direction, normal);
    float reflected_light = max(dot(reflected, view_direction), 0.0);
    float tight_specular = pow(reflected_light, 92.0);
    float broad_specular = pow(reflected_light, 14.0);
    float facing = clamp(dot(normal, view_direction), 0.0, 1.0);
    float fresnel = pow(1.0 - facing, 2.6);
    color += u_sun_tint * (tight_specular * 0.48 + broad_specular * 0.045);
    color = mix(color, u_fog_color.rgb, fresnel * 0.34);
    color *= clamp(u_exposure, 0.28, 1.35);

    if (u_fog_enabled != 0) {
        float density = max(u_fog_density, 0.0);
        float fog_factor = exp(-pow(density * v_fog_distance, 2.0));
        color = mix(u_fog_color.rgb, color, clamp(fog_factor, 0.0, 1.0));
    }

    float horizon_blend = 1.0 - smoothstep(
        0.0,
        max(u_horizon_fade_width, 1.0),
        outer_edge_distance(v_world_position.xz)
    );
    color = mix(color, u_fog_color.rgb, horizon_blend);

    gl_FragColor = vec4(color, 1.0);
}
"""


def terrain_bounds_from_grid(count: int, spacing: float, half: float) -> Bounds:
    """Return the exact bounds produced by ``TexturedGroundGridBuilder``."""

    tile_count = max(1, int(count))
    grid_spacing = float(spacing)
    tile_half = max(0.0, float(half))
    maximum = (tile_count - 1) * grid_spacing + tile_half
    return (-tile_half, maximum, -tile_half, maximum)


def _smooth01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _expanded_side_point(
    bounds: Bounds,
    side: int,
    along: float,
    expansion: float,
) -> tuple[float, float]:
    min_x, max_x, min_z, max_z = bounds
    t = max(0.0, min(1.0, float(along)))
    e = max(0.0, float(expansion))
    if side == 0:  # south, west to east
        return (min_x - e + (max_x - min_x + 2.0 * e) * t, min_z - e)
    if side == 1:  # east, south to north
        return (max_x + e, min_z - e + (max_z - min_z + 2.0 * e) * t)
    if side == 2:  # north, east to west
        return (max_x + e - (max_x - min_x + 2.0 * e) * t, max_z + e)
    # west, north to south
    return (min_x - e, max_z + e - (max_z - min_z + 2.0 * e) * t)


def sample_boundary_heights(
    bounds: Bounds,
    height_at: HeightFn,
    *,
    sample_spacing: float,
) -> np.ndarray:
    """Sample the rendered terrain edge on all four sides."""

    min_x, max_x, min_z, max_z = bounds
    side_length = max(max_x - min_x, max_z - min_z)
    segments = max(1, int(math.ceil(side_length / max(1.0, sample_spacing))))
    values: list[float] = []
    for side in range(4):
        for index in range(segments + 1):
            x, z = _expanded_side_point(bounds, side, index / segments, 0.0)
            value = float(height_at(x, z))
            if math.isfinite(value):
                values.append(value)
    return np.asarray(values, dtype=np.float32)


def _sand_color(x: float, z: float, radial: float) -> tuple[float, float, float]:
    dry = np.asarray((0.78, 0.61, 0.35), dtype=np.float64)
    wet = np.asarray((0.34, 0.31, 0.23), dtype=np.float64)
    wet_mix = _smooth01((radial - 0.62) / 0.38)
    color = dry * (1.0 - wet_mix) + wet * wet_mix
    grain = (
        math.sin(x * 0.071 + z * 0.037)
        + math.sin(x * 0.019 - z * 0.053 + 1.7)
    ) * 0.028
    band = math.sin(radial * math.pi * 9.0) * 0.012
    wet_line = -0.055 * math.exp(-((radial - 0.84) / 0.075) ** 2)
    color = np.clip(color + grain + band + wet_line, 0.0, 1.0)
    return tuple(float(channel) for channel in color)


def build_beach_vertex_data(
    bounds: Bounds,
    height_at: HeightFn,
    *,
    water_level: float,
    shore_width: float,
    sample_spacing: float,
    radial_segments: int,
) -> np.ndarray:
    """Build a colored rectangular shoreline that follows the terrain edge."""

    min_x, max_x, min_z, max_z = bounds
    side_length = max(max_x - min_x, max_z - min_z)
    along_segments = max(
        1,
        int(math.ceil(side_length / max(1.0, float(sample_spacing)))),
    )
    radial_count = max(2, int(radial_segments))
    width = max(1.0, float(shore_width))
    target_y = float(water_level) + 0.9
    rows: list[tuple[float, float, float, float, float, float]] = []

    for side in range(4):
        inner_heights = []
        for along_index in range(along_segments + 1):
            along = along_index / along_segments
            inner_x, inner_z = _expanded_side_point(bounds, side, along, 0.0)
            inner_heights.append(float(height_at(inner_x, inner_z)) + 0.22)

        rings: list[list[tuple[float, float, float, float, float, float]]] = []
        for radial_index in range(radial_count + 1):
            radial = radial_index / radial_count
            eased = _smooth01(radial)
            dune = math.sin(radial * math.pi) * (2.6 * (1.0 - radial * 0.35))
            ring = []
            for along_index in range(along_segments + 1):
                along = along_index / along_segments
                x, z = _expanded_side_point(bounds, side, along, width * radial)
                inner_y = inner_heights[along_index]
                contour = (
                    math.sin(x * 0.024 + z * 0.013)
                    + math.sin(x * 0.009 - z * 0.021 + 0.8)
                ) * 0.42 * math.sin(radial * math.pi)
                y = inner_y * (1.0 - eased) + target_y * eased + dune + contour
                r, g, b = _sand_color(x, z, radial)
                ring.append((x, y, z, r, g, b))
            rings.append(ring)

        for radial_index in range(radial_count):
            inner_ring = rings[radial_index]
            outer_ring = rings[radial_index + 1]
            for along_index in range(along_segments):
                a = inner_ring[along_index]
                b = inner_ring[along_index + 1]
                c = outer_ring[along_index + 1]
                d = outer_ring[along_index]
                rows.extend((a, b, c, a, c, d))

    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _water_axis_coordinates(
    outer_minimum: float,
    inner_minimum: float,
    inner_maximum: float,
    outer_maximum: float,
    grid_size: float,
) -> np.ndarray:
    """Build one shared axis with exact cuts at both shoreline-hole edges."""

    sections = (
        (outer_minimum, inner_minimum),
        (inner_minimum, inner_maximum),
        (inner_maximum, outer_maximum),
    )
    coordinates: list[float] = []
    for start, end in sections:
        divisions = max(1, int(math.ceil((end - start) / grid_size)))
        values = np.linspace(start, end, divisions + 1, dtype=np.float64)
        if coordinates:
            values = values[1:]
        coordinates.extend(float(value) for value in values)
    return np.asarray(coordinates, dtype=np.float32)


def _water_surface_bounds(
    bounds: Bounds,
    *,
    shore_width: float,
    outer_extent: float,
) -> Bounds:
    min_x, max_x, min_z, max_z = bounds
    inner_expansion = max(1.0, float(shore_width)) * 0.72
    extent = max(inner_expansion + 1.0, float(outer_extent))
    return (
        min_x - extent,
        max_x + extent,
        min_z - extent,
        max_z + extent,
    )


def build_water_vertex_data(
    bounds: Bounds,
    *,
    water_level: float,
    shore_width: float,
    outer_extent: float,
    grid_size: float,
) -> np.ndarray:
    """Build four gridded water strips around a rectangular island."""

    min_x, max_x, min_z, max_z = bounds
    inner_expansion = max(1.0, float(shore_width)) * 0.72
    step = max(16.0, float(grid_size))
    inner_min_x = min_x - inner_expansion
    inner_max_x = max_x + inner_expansion
    inner_min_z = min_z - inner_expansion
    inner_max_z = max_z + inner_expansion
    outer_min_x, outer_max_x, outer_min_z, outer_max_z = _water_surface_bounds(
        bounds,
        shore_width=shore_width,
        outer_extent=outer_extent,
    )
    xs = _water_axis_coordinates(
        outer_min_x,
        inner_min_x,
        inner_max_x,
        outer_max_x,
        step,
    )
    zs = _water_axis_coordinates(
        outer_min_z,
        inner_min_z,
        inner_max_z,
        outer_max_z,
        step,
    )
    rows: list[tuple[float, float, float]] = []
    y = float(water_level)
    for z_index in range(len(zs) - 1):
        z0 = float(zs[z_index])
        z1 = float(zs[z_index + 1])
        inside_z = z0 >= inner_min_z and z1 <= inner_max_z
        for x_index in range(len(xs) - 1):
            x0 = float(xs[x_index])
            x1 = float(xs[x_index + 1])
            inside_x = x0 >= inner_min_x and x1 <= inner_max_x
            if inside_x and inside_z:
                continue
            rows.extend(
                (
                    (x0, y, z0),
                    (x1, y, z0),
                    (x1, y, z1),
                    (x0, y, z0),
                    (x1, y, z1),
                    (x0, y, z1),
                )
            )
    if not rows:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _single_gl_id(value) -> int:
    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        raise RuntimeError("OpenGL did not return one buffer ID")
    return int(values[0])


def _delete_gl_buffer(buffer_id: int) -> None:
    if not buffer_id:
        return
    try:
        glDeleteBuffers(1, [buffer_id])
    except TypeError:
        glDeleteBuffers([buffer_id])


class StylizedWaterRenderer:
    """Own the water shader and its static gridded surface VBO."""

    def __init__(
        self,
        vertex_data: np.ndarray,
        *,
        terrain_bounds: Bounds,
        water_bounds: Bounds,
        shore_width: float,
        horizon_fade_width: float,
    ) -> None:
        vertices = np.ascontiguousarray(vertex_data, dtype=np.float32)
        self.vertex_count = int(len(vertices))
        self.terrain_bounds = tuple(float(value) for value in terrain_bounds)
        self.water_bounds = tuple(float(value) for value in water_bounds)
        self.shore_width = float(shore_width)
        self.horizon_fade_width = max(1.0, float(horizon_fade_width))
        self.program = 0
        self.vbo = 0
        self._uniforms: dict[str, int] = {}
        try:
            self.program = compile_program(WATER_VERTEX_SOURCE, WATER_FRAGMENT_SOURCE)
            self.vbo = _single_gl_id(glGenBuffers(1))
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            uniform_names = (
                "u_time",
                "u_camera_position",
                "u_light_direction",
                "u_sun_tint",
                "u_terrain_bounds",
                "u_water_bounds",
                "u_shore_width",
                "u_horizon_fade_width",
                "u_exposure",
                "u_fog_enabled",
                "u_fog_density",
                "u_fog_color",
            )
            self._uniforms = {
                name: int(glGetUniformLocation(self.program, name))
                for name in uniform_names
            }
        except Exception:
            if self.vbo:
                try:
                    _delete_gl_buffer(self.vbo)
                except Exception:
                    pass
                self.vbo = 0
            if self.program:
                try:
                    glDeleteProgram(self.program)
                except Exception:
                    pass
                self.program = 0
            raise
        finally:
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._started_at = time.perf_counter()

    @staticmethod
    def _vector3(value, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        if value is None:
            return fallback
        try:
            if all(hasattr(value, name) for name in ("x", "y", "z")):
                return (float(value.x), float(value.y), float(value.z))
            return (float(value[0]), float(value[1]), float(value[2]))
        except (IndexError, TypeError, ValueError):
            return fallback

    def draw(self, camera, *, lighting=None) -> None:
        if not self.program or not self.vbo or self.vertex_count <= 0:
            return

        position = self._vector3(getattr(camera, "position", None), (0.0, 0.0, 0.0))
        light_direction = self._vector3(
            getattr(lighting, "light_direction", None),
            (0.35, 0.82, 0.25),
        )
        sun_tint = self._vector3(
            getattr(lighting, "sun_tint", None),
            (1.0, 0.96, 0.86),
        )
        fog = get_render_fog_state()
        min_x, max_x, min_z, max_z = self.terrain_bounds
        water_min_x, water_max_x, water_min_z, water_max_z = self.water_bounds

        glDisable(GL_BLEND)
        glDisable(GL_CULL_FACE)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        try:
            glUseProgram(self.program)
            elapsed = time.perf_counter() - self._started_at
            glUniform1f(self._uniforms["u_time"], elapsed)
            glUniform3f(self._uniforms["u_camera_position"], *position)
            glUniform3f(self._uniforms["u_light_direction"], *light_direction)
            glUniform3f(self._uniforms["u_sun_tint"], *sun_tint)
            glUniform4f(
                self._uniforms["u_terrain_bounds"], min_x, max_x, min_z, max_z
            )
            glUniform4f(
                self._uniforms["u_water_bounds"],
                water_min_x,
                water_max_x,
                water_min_z,
                water_max_z,
            )
            glUniform1f(self._uniforms["u_shore_width"], self.shore_width)
            glUniform1f(
                self._uniforms["u_horizon_fade_width"], self.horizon_fade_width
            )
            glUniform1f(
                self._uniforms["u_exposure"],
                float(getattr(camera, "brightness_default", 1.0)),
            )
            glUniform1i(self._uniforms["u_fog_enabled"], int(bool(fog.enabled)))
            glUniform1f(self._uniforms["u_fog_density"], float(fog.density))
            glUniform4f(self._uniforms["u_fog_color"], *fog.color)

            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        finally:
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            use_fixed_pipeline()
            glDisable(GL_BLEND)
            glDisable(GL_CULL_FACE)
            glDisable(GL_TEXTURE_2D)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)

    def dispose(self) -> None:
        vbo = self.vbo
        program = self.program
        self.vbo = 0
        self.program = 0
        self.vertex_count = 0
        try:
            _delete_gl_buffer(vbo)
        finally:
            if program:
                glDeleteProgram(program)


class IslandBoundary:
    """Scene-owned beach mesh and animated water renderer."""

    def __init__(
        self,
        *,
        beach_mesh: BatchedMesh,
        water: StylizedWaterRenderer,
        terrain_bounds: Bounds,
        water_level: float,
        shore_width: float,
    ) -> None:
        self.beach_mesh = beach_mesh
        self.water = water
        self.terrain_bounds = terrain_bounds
        self.water_level = float(water_level)
        self.shore_width = float(shore_width)

    def draw_water(self, camera, *, lighting=None) -> None:
        self.water.draw(camera, lighting=lighting)

    def draw_beach(self, camera) -> None:
        self.beach_mesh.set_exposure(
            float(getattr(camera, "brightness_default", 1.0))
        )
        self.beach_mesh.draw(camera=None, view_distance=None)

    def draw(self, camera, *, lighting=None) -> None:
        self.draw_water(camera, lighting=lighting)
        self.draw_beach(camera)

    def dispose(self) -> None:
        try:
            self.water.dispose()
        finally:
            self.beach_mesh.dispose()


def build_island_boundary(
    scene,
    *,
    shore_width: float,
    shore_sample_spacing: float,
    shore_radial_segments: int,
    water_drop: float,
    water_extent: float,
    water_grid_size: float,
) -> IslandBoundary:
    """Create the world-edge beach and water from the built ground sampler."""

    sampler = getattr(scene, "_ground_height_sampler", None)
    if sampler is None or not hasattr(sampler, "height_at"):
        raise RuntimeError("island boundary requires the built ground height sampler")

    terrain_bounds = tuple(
        float(value)
        for value in getattr(
            scene,
            "terrain_bounds",
            terrain_bounds_from_grid(
                getattr(scene, "_grid_count", 1),
                getattr(scene, "_grid_spacing", 1.0),
                getattr(scene, "_grid_half", 0.5),
            ),
        )
    )
    height_at = lambda x, z: float(sampler.height_at(x, z))
    boundary_heights = sample_boundary_heights(
        terrain_bounds,
        height_at,
        sample_spacing=shore_sample_spacing,
    )
    minimum_height = (
        float(np.min(boundary_heights)) if len(boundary_heights) else 5.0
    )
    water_level = minimum_height - max(1.0, float(water_drop))
    beach_vertices = build_beach_vertex_data(
        terrain_bounds,
        height_at,
        water_level=water_level,
        shore_width=shore_width,
        sample_spacing=shore_sample_spacing,
        radial_segments=shore_radial_segments,
    )
    water_vertices = build_water_vertex_data(
        terrain_bounds,
        water_level=water_level,
        shore_width=shore_width,
        outer_extent=water_extent,
        grid_size=water_grid_size,
    )
    water_bounds = _water_surface_bounds(
        terrain_bounds,
        shore_width=shore_width,
        outer_extent=water_extent,
    )
    outer_distance = max(1.0, terrain_bounds[0] - water_bounds[0])
    horizon_fade_width = max(64.0, min(1400.0, outer_distance * 0.28))
    beach_mesh = BatchedMesh.from_vertex_data(
        beach_vertices,
        texture=None,
        casts_shadows=False,
        casts_sun_shadows=False,
        casts_point_shadows=False,
        exposure_baseline=1.0,
        environment_lighting=False,
        shine_enabled=False,
    )
    try:
        water = StylizedWaterRenderer(
            water_vertices,
            terrain_bounds=terrain_bounds,
            water_bounds=water_bounds,
            shore_width=shore_width,
            horizon_fade_width=horizon_fade_width,
        )
    except Exception:
        beach_mesh.dispose()
        raise
    return IslandBoundary(
        beach_mesh=beach_mesh,
        water=water,
        terrain_bounds=terrain_bounds,
        water_level=water_level,
        shore_width=shore_width,
    )
