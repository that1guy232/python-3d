"""Small compatibility-profile shaders used by legacy render paths."""

from __future__ import annotations

from dataclasses import dataclass

from OpenGL.GL import (
    GL_COMPILE_STATUS,
    GL_FRAGMENT_SHADER,
    GL_LINK_STATUS,
    GL_CURRENT_PROGRAM,
    GL_TEXTURE0,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glAttachShader,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteProgram,
    glDeleteShader,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetIntegerv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniform4f,
    glUseProgram,
)

MAX_BRIGHTNESS_AREAS = 16


_TEXTURE_COLOR_EXPOSURE_VERTEX = """#version 120

varying vec4 v_color;
varying vec2 v_uv;
varying vec3 v_normal;
varying vec3 v_world_pos;

void main()
{
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    v_color = gl_Color;
    v_uv = gl_MultiTexCoord0.xy;
    v_normal = gl_Normal;
    v_world_pos = gl_Vertex.xyz;
}
"""


_TEXTURE_COLOR_EXPOSURE_FRAGMENT = """#version 120

uniform sampler2D u_texture;
uniform float u_exposure;
uniform float u_base_brightness;
uniform int u_scene_lighting_enabled;
uniform int u_directional_enabled;
uniform vec3 u_light_dir;
uniform float u_light_ambient;
uniform float u_light_diffuse;
uniform float u_light_max_factor;
uniform int u_brightness_area_count;
uniform vec4 u_brightness_areas[16];
uniform float u_brightness_falloffs[16];
uniform vec4 u_brightness_bounds[16];
uniform float u_brightness_indoor_only[16];

varying vec4 v_color;
varying vec2 v_uv;
varying vec3 v_normal;
varying vec3 v_world_pos;

float brightness_at(vec3 world_pos, float receiver_factor)
{
    float brightness = u_base_brightness;
    for (int i = 0; i < 16; ++i) {
        if (i >= u_brightness_area_count) {
            break;
        }
        if (u_brightness_indoor_only[i] > 0.5 && receiver_factor > 0.995) {
            continue;
        }
        vec4 area = u_brightness_areas[i];
        vec4 bounds = u_brightness_bounds[i];
        if (bounds.x <= bounds.y) {
            if (
                world_pos.x < bounds.x ||
                world_pos.x > bounds.y ||
                world_pos.z < bounds.z ||
                world_pos.z > bounds.w
            ) {
                continue;
            }
        }
        float radius = max(area.z, 0.000001);
        float dist = distance(world_pos.xz, area.xy);
        if (dist <= radius) {
            float norm_dist = clamp(dist / radius, 0.0, 1.0);
            float attenuation = pow(1.0 - norm_dist, max(u_brightness_falloffs[i], 0.0));
            float relative = u_base_brightness == 0.0 ? area.w : area.w / u_base_brightness;
            brightness *= 1.0 + (relative - 1.0) * attenuation;
        }
    }
    return brightness;
}

float sunlight_factor()
{
    if (u_directional_enabled == 0) {
        return 1.0;
    }
    vec3 normal = normalize(v_normal);
    float dot_light = max(0.0, dot(normal, normalize(u_light_dir)));
    return clamp(
        u_light_ambient + u_light_diffuse * dot_light,
        0.0,
        u_light_max_factor
    );
}

void main()
{
    vec4 texel = texture2D(u_texture, v_uv);
    float receiver_factor = max(max(v_color.r, v_color.g), v_color.b);
    float brightness = u_scene_lighting_enabled == 0
        ? u_exposure
        : brightness_at(v_world_pos, receiver_factor) * u_exposure;
    vec3 rgb = texel.rgb * v_color.rgb * brightness * sunlight_factor();
    gl_FragColor = vec4(rgb, texel.a * v_color.a);
}
"""


@dataclass(frozen=True)
class TextureColorExposureShader:
    program: int
    texture_location: int
    exposure_location: int
    base_brightness_location: int
    scene_lighting_enabled_location: int
    directional_enabled_location: int
    light_dir_location: int
    light_ambient_location: int
    light_diffuse_location: int
    light_max_factor_location: int
    brightness_area_count_location: int
    brightness_area_locations: tuple[int, ...]
    brightness_falloff_locations: tuple[int, ...]
    brightness_bound_locations: tuple[int, ...]
    brightness_indoor_only_locations: tuple[int, ...]

    def bind(
        self,
        *,
        scene_lighting_enabled: bool = False,
        directional_enabled: bool = False,
    ) -> None:
        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        if self.texture_location >= 0:
            glUniform1i(self.texture_location, 0)
        self.set_exposure_scale(_texture_color_exposure_scale)
        if self.scene_lighting_enabled_location >= 0:
            glUniform1i(
                self.scene_lighting_enabled_location,
                1 if scene_lighting_enabled else 0,
            )
        if self.directional_enabled_location >= 0:
            glUniform1i(
                self.directional_enabled_location,
                1 if directional_enabled else 0,
            )

    def set_exposure_scale(self, exposure_scale: float) -> None:
        if self.exposure_location >= 0:
            glUniform1f(self.exposure_location, float(exposure_scale))

    def apply_lighting_state(self, state: "TextureLightingState") -> None:
        if self.base_brightness_location >= 0:
            glUniform1f(self.base_brightness_location, state.base_brightness)
        if self.light_dir_location >= 0:
            glUniform3f(self.light_dir_location, *state.light_direction)
        if self.light_ambient_location >= 0:
            glUniform1f(self.light_ambient_location, state.light_ambient)
        if self.light_diffuse_location >= 0:
            glUniform1f(self.light_diffuse_location, state.light_diffuse)
        if self.light_max_factor_location >= 0:
            glUniform1f(self.light_max_factor_location, state.light_max_factor)
        if self.brightness_area_count_location >= 0:
            glUniform1i(self.brightness_area_count_location, len(state.brightness_areas))

        for index, location in enumerate(self.brightness_area_locations):
            if location < 0:
                continue
            if index < len(state.brightness_areas):
                glUniform4f(location, *state.brightness_areas[index])
            else:
                glUniform4f(location, 0.0, 0.0, 1.0, state.base_brightness)

        for index, location in enumerate(self.brightness_falloff_locations):
            if location < 0:
                continue
            value = (
                state.brightness_falloffs[index]
                if index < len(state.brightness_falloffs)
                else 1.0
            )
            glUniform1f(location, value)

        for index, location in enumerate(self.brightness_bound_locations):
            if location < 0:
                continue
            if index < len(state.brightness_bounds):
                glUniform4f(location, *state.brightness_bounds[index])
            else:
                glUniform4f(location, 1.0, 0.0, 0.0, 0.0)

        for index, location in enumerate(self.brightness_indoor_only_locations):
            if location < 0:
                continue
            value = (
                state.brightness_indoor_only[index]
                if index < len(state.brightness_indoor_only)
                else 0.0
            )
            glUniform1f(location, value)


@dataclass(frozen=True)
class TextureLightingState:
    base_brightness: float = 1.0
    light_direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    light_ambient: float = 1.0
    light_diffuse: float = 0.0
    light_max_factor: float = 1.0
    brightness_areas: tuple[tuple[float, float, float, float], ...] = ()
    brightness_falloffs: tuple[float, ...] = ()
    brightness_bounds: tuple[tuple[float, float, float, float], ...] = ()
    brightness_indoor_only: tuple[float, ...] = ()


_texture_color_exposure_shader: TextureColorExposureShader | None = None
_texture_color_exposure_failed = False
_texture_color_exposure_scale = 1.0
_texture_lighting_state = TextureLightingState()


def _decode_log(log) -> str:
    if isinstance(log, bytes):
        return log.decode("utf-8", "replace")
    return str(log)


def _compile_shader(shader_type: int, source: str) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = _decode_log(glGetShaderInfoLog(shader)).strip()
        glDeleteShader(shader)
        raise RuntimeError(log or "shader compilation failed")
    return int(shader)


def _compile_program(vertex_source: str, fragment_source: str) -> int:
    vertex_shader = 0
    fragment_shader = 0
    program = 0

    try:
        vertex_shader = _compile_shader(GL_VERTEX_SHADER, vertex_source)
        fragment_shader = _compile_shader(GL_FRAGMENT_SHADER, fragment_source)
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            log = _decode_log(glGetProgramInfoLog(program)).strip()
            raise RuntimeError(log or "shader link failed")
        return int(program)
    except Exception:
        if program:
            try:
                glDeleteProgram(program)
            except Exception:
                pass
        raise
    finally:
        if vertex_shader:
            glDeleteShader(vertex_shader)
        if fragment_shader:
            glDeleteShader(fragment_shader)


def get_texture_color_exposure_shader() -> TextureColorExposureShader | None:
    """Return the lazy compatibility shader, or None if unavailable."""

    global _texture_color_exposure_shader, _texture_color_exposure_failed

    if _texture_color_exposure_shader is not None:
        return _texture_color_exposure_shader
    if _texture_color_exposure_failed:
        return None

    try:
        program = _compile_program(
            _TEXTURE_COLOR_EXPOSURE_VERTEX,
            _TEXTURE_COLOR_EXPOSURE_FRAGMENT,
        )
        _texture_color_exposure_shader = TextureColorExposureShader(
            program=program,
            texture_location=int(glGetUniformLocation(program, "u_texture")),
            exposure_location=int(glGetUniformLocation(program, "u_exposure")),
            base_brightness_location=int(
                glGetUniformLocation(program, "u_base_brightness")
            ),
            scene_lighting_enabled_location=int(
                glGetUniformLocation(program, "u_scene_lighting_enabled")
            ),
            directional_enabled_location=int(
                glGetUniformLocation(program, "u_directional_enabled")
            ),
            light_dir_location=int(glGetUniformLocation(program, "u_light_dir")),
            light_ambient_location=int(
                glGetUniformLocation(program, "u_light_ambient")
            ),
            light_diffuse_location=int(
                glGetUniformLocation(program, "u_light_diffuse")
            ),
            light_max_factor_location=int(
                glGetUniformLocation(program, "u_light_max_factor")
            ),
            brightness_area_count_location=int(
                glGetUniformLocation(program, "u_brightness_area_count")
            ),
            brightness_area_locations=tuple(
                int(glGetUniformLocation(program, f"u_brightness_areas[{index}]"))
                for index in range(MAX_BRIGHTNESS_AREAS)
            ),
            brightness_falloff_locations=tuple(
                int(glGetUniformLocation(program, f"u_brightness_falloffs[{index}]"))
                for index in range(MAX_BRIGHTNESS_AREAS)
            ),
            brightness_bound_locations=tuple(
                int(glGetUniformLocation(program, f"u_brightness_bounds[{index}]"))
                for index in range(MAX_BRIGHTNESS_AREAS)
            ),
            brightness_indoor_only_locations=tuple(
                int(
                    glGetUniformLocation(
                        program,
                        f"u_brightness_indoor_only[{index}]",
                    )
                )
                for index in range(MAX_BRIGHTNESS_AREAS)
            ),
        )
        set_texture_lighting_state(compile_shader=False)
        return _texture_color_exposure_shader
    except Exception as exc:
        _texture_color_exposure_failed = True
        print(f"Warning: compatibility shader unavailable: {exc}")
        return None


def texture_color_exposure_shader_available() -> bool:
    return get_texture_color_exposure_shader() is not None


def get_texture_color_exposure_scale() -> float:
    return _texture_color_exposure_scale


def _current_program_id() -> int:
    current_program = glGetIntegerv(GL_CURRENT_PROGRAM)
    try:
        return int(current_program)
    except TypeError:
        return int(current_program[0])


def set_texture_color_exposure_scale(
    exposure_scale: float,
    *,
    compile_shader: bool = True,
) -> bool:
    """Update the global texture-color exposure uniform.

    Returns True when the shader path is available. When False, callers should
    keep using the old CPU-side vertex color upload.
    """

    global _texture_color_exposure_scale

    _texture_color_exposure_scale = float(exposure_scale)
    shader = (
        get_texture_color_exposure_shader()
        if compile_shader
        else _texture_color_exposure_shader
    )
    if shader is None:
        return False

    current_program = _current_program_id()
    try:
        glUseProgram(shader.program)
        shader.set_exposure_scale(_texture_color_exposure_scale)
    finally:
        glUseProgram(current_program)
    return True


def _vector3_tuple(value, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        return (float(value.x), float(value.y), float(value.z))
    except Exception:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return fallback


def _light_direction_from(lighting=None, sun_direction=None) -> tuple[float, float, float]:
    if lighting is not None:
        return _vector3_tuple(
            getattr(lighting, "light_direction", None),
            (0.0, 1.0, 0.0),
        )

    if sun_direction is None:
        return (0.0, 1.0, 0.0)

    sx, sy, sz = _vector3_tuple(sun_direction, (0.0, -1.0, 0.0))
    length = (sx * sx + sy * sy + sz * sz) ** 0.5
    if length <= 1e-8:
        return (0.0, 1.0, 0.0)
    return (-sx / length, -sy / length, -sz / length)


def _brightness_area_uniforms(
    brightness_areas,
) -> tuple[
    tuple[tuple[float, float, float, float], ...],
    tuple[float, ...],
    tuple[tuple[float, float, float, float], ...],
    tuple[float, ...],
]:
    areas: list[tuple[float, float, float, float]] = []
    falloffs: list[float] = []
    bounds_values: list[tuple[float, float, float, float]] = []
    indoor_only_values: list[float] = []
    for area in brightness_areas or ():
        if len(areas) >= MAX_BRIGHTNESS_AREAS:
            break
        try:
            if isinstance(area, dict):
                center = area["center"]
                radius = area["radius"]
                value = area["value"]
                falloff = area.get("falloff", 1.0)
                bounds = area.get("bounds")
                indoor_only = bool(area.get("indoor_only", False))
            else:
                center, radius, value, falloff = area[:4]
                bounds = area[4] if len(area) > 4 else None
                indoor_only = False
            try:
                cx = float(center.x)
                cz = float(center.z)
            except Exception:
                cx = float(center[0])
                cz = float(center[2] if len(center) > 2 else center[1])
            areas.append((cx, cz, max(float(radius), 1e-6), float(value)))
            falloffs.append(max(float(falloff), 0.0))
            if bounds is None:
                bounds_values.append((1.0, 0.0, 0.0, 0.0))
            else:
                min_x, max_x, min_z, max_z = (float(part) for part in bounds)
                if max_x < min_x:
                    min_x, max_x = max_x, min_x
                if max_z < min_z:
                    min_z, max_z = max_z, min_z
                bounds_values.append((min_x, max_x, min_z, max_z))
            indoor_only_values.append(1.0 if indoor_only else 0.0)
        except Exception:
            continue
    return (
        tuple(areas),
        tuple(falloffs),
        tuple(bounds_values),
        tuple(indoor_only_values),
    )


def set_texture_lighting_state(
    *,
    base_brightness: float | None = None,
    lighting=None,
    sun_direction=None,
    brightness_areas=None,
    ambient: float | None = None,
    diffuse: float | None = None,
    max_factor: float | None = None,
    exposure_scale: float | None = None,
    compile_shader: bool = True,
) -> bool:
    """Update shared scene-lighting uniforms for textured compatibility draws."""

    global _texture_lighting_state

    current = _texture_lighting_state
    area_values, area_falloffs, area_bounds, area_indoor_only = (
        _brightness_area_uniforms(brightness_areas)
        if brightness_areas is not None
        else (
            current.brightness_areas,
            current.brightness_falloffs,
            current.brightness_bounds,
            current.brightness_indoor_only,
        )
    )

    _texture_lighting_state = TextureLightingState(
        base_brightness=(
            float(base_brightness)
            if base_brightness is not None
            else current.base_brightness
        ),
        light_direction=_light_direction_from(lighting=lighting, sun_direction=sun_direction),
        light_ambient=(
            float(ambient)
            if ambient is not None
            else float(getattr(lighting, "ambient", current.light_ambient))
        ),
        light_diffuse=(
            float(diffuse)
            if diffuse is not None
            else float(getattr(lighting, "diffuse", current.light_diffuse))
        ),
        light_max_factor=(
            float(max_factor)
            if max_factor is not None
            else float(getattr(lighting, "max_factor", current.light_max_factor))
        ),
        brightness_areas=area_values,
        brightness_falloffs=area_falloffs,
        brightness_bounds=area_bounds,
        brightness_indoor_only=area_indoor_only,
    )

    if exposure_scale is not None:
        set_texture_color_exposure_scale(exposure_scale, compile_shader=False)

    shader = (
        get_texture_color_exposure_shader()
        if compile_shader
        else _texture_color_exposure_shader
    )
    if shader is None:
        return False

    current_program = _current_program_id()
    try:
        glUseProgram(shader.program)
        shader.apply_lighting_state(_texture_lighting_state)
    finally:
        glUseProgram(current_program)
    return True


def get_texture_lighting_state() -> TextureLightingState:
    return _texture_lighting_state


def use_fixed_pipeline() -> None:
    glUseProgram(0)
