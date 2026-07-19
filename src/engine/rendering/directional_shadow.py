"""Directional shadow-map resources and the shared raster caster shader."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math

import numpy as np
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_COMPONENT,
    GL_DEPTH_COMPONENT24,
    GL_DEPTH_TEST,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_BINDING,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_NONE,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRUE,
    GL_UNSIGNED_INT,
    GL_VIEWPORT,
    glActiveTexture,
    glBindFramebuffer,
    glBindTexture,
    glCheckFramebufferStatus,
    glClear,
    glDeleteFramebuffers,
    glDeleteProgram,
    glDeleteTextures,
    glDrawBuffer,
    glEnable,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenTextures,
    glGetIntegerv,
    glGetUniformLocation,
    glReadBuffer,
    glTexImage2D,
    glTexParameteri,
    glUniform1f,
    glUniform1i,
    glUniformMatrix4fv,
    glUseProgram,
    glViewport,
)

from engine.rendering.gl_program import compile_program


DIRECTIONAL_SHADOW_BIAS_WORLD_UNITS = 0.2


def directional_shadow_bias(
    *,
    near: float,
    far: float,
    world_units: float = DIRECTIONAL_SHADOW_BIAS_WORLD_UNITS,
) -> float:
    """Convert a world-space depth allowance into normalized shadow depth."""
    depth_range = float(far) - float(near)
    if depth_range <= 0.0:
        raise ValueError("directional shadow far plane must exceed near plane")
    return max(0.0, float(world_units)) / depth_range


DEFAULT_DIRECTIONAL_SHADOW_BIAS = directional_shadow_bias(near=1.0, far=4000.0)


SHADOW_CASTER_VERTEX_SOURCE = r"""#version 120
uniform mat4 u_light_matrix;
varying vec2 v_uv;

void main()
{
    gl_Position = u_light_matrix * gl_Vertex;
    v_uv = gl_MultiTexCoord0.xy;
}
"""


SHADOW_CASTER_FRAGMENT_SOURCE = r"""#version 120
uniform sampler2D u_texture;
uniform int u_alpha_cutout;
uniform float u_alpha_cutoff;
varying vec2 v_uv;

void main()
{
    if (u_alpha_cutout != 0) {
        if (texture2D(u_texture, v_uv).a <= u_alpha_cutoff) {
            discard;
        }
    }
    gl_FragColor = vec4(1.0);
}
"""


class DirectionalShadowUnavailable(RuntimeError):
    """The active OpenGL context cannot create the requested shadow map."""


def _single_gl_id(value) -> int:
    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        raise DirectionalShadowUnavailable("OpenGL did not return one resource ID")
    return int(values[0])


def _normalize(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(result))
    if length <= 1e-9:
        raise ValueError("direction vector must be non-zero")
    return result / length


def look_at_matrix(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Return a row-major OpenGL view matrix for a light or camera."""

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = _normalize(target - eye)
    up_vector = _normalize(up)
    side = np.cross(forward, up_vector)
    if float(np.linalg.norm(side)) <= 1e-9:
        up_vector = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        side = np.cross(forward, up_vector)
    side = _normalize(side)
    true_up = np.cross(side, forward)
    return np.array(
        (
            (side[0], side[1], side[2], -float(np.dot(side, eye))),
            (true_up[0], true_up[1], true_up[2], -float(np.dot(true_up, eye))),
            (-forward[0], -forward[1], -forward[2], float(np.dot(forward, eye))),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def orthographic_matrix(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> np.ndarray:
    """Return a row-major OpenGL orthographic projection matrix."""

    if right <= left or top <= bottom or near <= 0.0 or far <= near:
        raise ValueError("invalid orthographic shadow volume")
    return np.array(
        (
            (2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)),
            (0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)),
            (0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def directional_light_matrix(
    center,
    light_direction,
    *,
    extent: float,
    near: float = 1.0,
    far: float = 2000.0,
) -> tuple[float, ...]:
    """Fit one orthographic light camera around a game-space center."""

    extent = float(extent)
    if extent <= 0.0:
        raise ValueError("shadow extent must be positive")
    center = np.asarray(center, dtype=np.float64)
    toward_light = _normalize(light_direction)
    distance = (float(near) + float(far)) * 0.5
    eye = center + toward_light * distance
    view = look_at_matrix(eye, center)
    projection = orthographic_matrix(-extent, extent, -extent, extent, near, far)
    matrix = projection @ view
    return tuple(float(value) for value in matrix.reshape(-1))


@dataclass(frozen=True, slots=True)
class DirectionalShadowBinding:
    """Immutable material-pass input for one completed sun shadow map."""

    texture: int
    light_matrix: tuple[float, ...]
    map_size: int
    bias: float = DEFAULT_DIRECTIONAL_SHADOW_BIAS

    @property
    def texel_size(self) -> tuple[float, float]:
        inverse = 1.0 / max(1, int(self.map_size))
        return inverse, inverse


@dataclass(slots=True)
class DirectionalShadowMap:
    """Own one depth texture and framebuffer for sun visibility."""

    size: int
    framebuffer: int
    depth_texture: int

    @classmethod
    def create(cls, size: int = 1024) -> "DirectionalShadowMap":
        size = int(size)
        if size <= 0:
            raise ValueError("shadow-map size must be positive")
        texture = 0
        framebuffer = 0
        previous = int(glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        try:
            texture = _single_gl_id(glGenTextures(1))
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_DEPTH_COMPONENT24,
                size,
                size,
                0,
                GL_DEPTH_COMPONENT,
                GL_UNSIGNED_INT,
                None,
            )

            framebuffer = _single_gl_id(glGenFramebuffers(1))
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT,
                GL_TEXTURE_2D,
                texture,
                0,
            )
            glDrawBuffer(GL_NONE)
            glReadBuffer(GL_NONE)
            status = int(glCheckFramebufferStatus(GL_FRAMEBUFFER))
            if status != GL_FRAMEBUFFER_COMPLETE:
                raise DirectionalShadowUnavailable(
                    f"directional shadow framebuffer incomplete: 0x{status:04x}"
                )
            return cls(size=size, framebuffer=framebuffer, depth_texture=texture)
        except Exception:
            if framebuffer:
                glDeleteFramebuffers(1, [framebuffer])
            if texture:
                glDeleteTextures(1, [texture])
            raise
        finally:
            glBindFramebuffer(GL_FRAMEBUFFER, previous)
            glBindTexture(GL_TEXTURE_2D, 0)

    @contextmanager
    def render_depth(self):
        """Bind, clear, and restore this depth-only framebuffer."""

        previous_framebuffer = int(glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        previous_viewport = tuple(int(value) for value in glGetIntegerv(GL_VIEWPORT))
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
        glViewport(0, 0, self.size, self.size)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_DEPTH_BUFFER_BIT)
        try:
            yield
        finally:
            glUseProgram(0)
            glBindFramebuffer(GL_FRAMEBUFFER, previous_framebuffer)
            glViewport(*previous_viewport)

    def binding(
        self,
        light_matrix: tuple[float, ...],
        *,
        bias: float = DEFAULT_DIRECTIONAL_SHADOW_BIAS,
    ) -> DirectionalShadowBinding:
        if len(light_matrix) != 16:
            raise ValueError("directional shadow matrix must contain 16 values")
        return DirectionalShadowBinding(
            texture=self.depth_texture,
            light_matrix=tuple(float(value) for value in light_matrix),
            map_size=self.size,
            bias=max(0.0, float(bias)),
        )

    def dispose(self) -> None:
        if self.framebuffer:
            glDeleteFramebuffers(1, [self.framebuffer])
        if self.depth_texture:
            glDeleteTextures(1, [self.depth_texture])
        self.framebuffer = 0
        self.depth_texture = 0


@dataclass(slots=True)
class ShadowCasterShader:
    """Shared depth-pass shader for opaque and alpha-cutout materials."""

    program: int
    light_matrix_location: int
    texture_location: int
    alpha_cutout_location: int
    alpha_cutoff_location: int

    @classmethod
    def create(cls) -> "ShadowCasterShader":
        program = compile_program(
            SHADOW_CASTER_VERTEX_SOURCE,
            SHADOW_CASTER_FRAGMENT_SOURCE,
        )
        return cls(
            program=program,
            light_matrix_location=int(glGetUniformLocation(program, "u_light_matrix")),
            texture_location=int(glGetUniformLocation(program, "u_texture")),
            alpha_cutout_location=int(glGetUniformLocation(program, "u_alpha_cutout")),
            alpha_cutoff_location=int(glGetUniformLocation(program, "u_alpha_cutoff")),
        )

    def bind(
        self,
        light_matrix: tuple[float, ...],
        *,
        texture: int = 0,
        alpha_cutout: bool = False,
        alpha_cutoff: float = 0.5,
    ) -> None:
        if len(light_matrix) != 16:
            raise ValueError("directional shadow matrix must contain 16 values")
        glUseProgram(self.program)
        glUniformMatrix4fv(
            self.light_matrix_location,
            1,
            GL_TRUE,
            light_matrix,
        )
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, int(texture or 0))
        glUniform1i(self.texture_location, 0)
        glUniform1i(self.alpha_cutout_location, int(bool(alpha_cutout)))
        glUniform1f(self.alpha_cutoff_location, max(0.0, min(1.0, float(alpha_cutoff))))

    def dispose(self) -> None:
        if self.program:
            glDeleteProgram(self.program)
        self.program = 0
