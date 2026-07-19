"""Depth cube maps for occluded raster point lights."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_CLEAR_VALUE,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_COMPONENT24,
    GL_DEPTH_TEST,
    GL_FLOAT,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_BINDING,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_RENDERBUFFER,
    GL_RGBA,
    GL_RGBA32F,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_CUBE_MAP,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
    GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_R,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRUE,
    GL_VIEWPORT,
    glActiveTexture,
    glBindFramebuffer,
    glBindRenderbuffer,
    glBindTexture,
    glCheckFramebufferStatus,
    glClear,
    glClearColor,
    glDeleteFramebuffers,
    glDeleteProgram,
    glDeleteRenderbuffers,
    glDeleteTextures,
    glEnable,
    glFramebufferRenderbuffer,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenRenderbuffers,
    glGenTextures,
    glGetFloatv,
    glGetIntegerv,
    glGetUniformLocation,
    glRenderbufferStorage,
    glTexImage2D,
    glTexParameteri,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
    glViewport,
)

from engine.rendering.directional_shadow import look_at_matrix
from engine.rendering.gl_program import compile_program


POINT_SHADOW_FACES = (
    GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
)


POINT_SHADOW_CASTER_VERTEX_SOURCE = r"""#version 120
uniform mat4 u_light_matrix;
varying vec2 v_uv;
varying vec3 v_world_pos;

void main()
{
    gl_Position = u_light_matrix * gl_Vertex;
    v_uv = gl_MultiTexCoord0.xy;
    v_world_pos = gl_Vertex.xyz;
}
"""


POINT_SHADOW_CASTER_FRAGMENT_SOURCE = r"""#version 120
uniform sampler2D u_texture;
uniform int u_alpha_cutout;
uniform float u_alpha_cutoff;
uniform vec3 u_light_position;
uniform float u_light_range;
varying vec2 v_uv;
varying vec3 v_world_pos;

void main()
{
    if (u_alpha_cutout != 0) {
        if (texture2D(u_texture, v_uv).a <= u_alpha_cutoff) {
            discard;
        }
    }
    float radial_depth = distance(v_world_pos, u_light_position)
        / max(u_light_range, 0.0001);
    gl_FragColor = vec4(radial_depth, radial_depth, radial_depth, 1.0);
}
"""


class PointShadowUnavailable(RuntimeError):
    """The active context cannot create point-light shadow resources."""


def _single_gl_id(value) -> int:
    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        raise PointShadowUnavailable("OpenGL did not return one resource ID")
    return int(values[0])


def perspective_matrix(
    fov_degrees: float,
    aspect: float,
    near: float,
    far: float,
) -> np.ndarray:
    if fov_degrees <= 0.0 or fov_degrees >= 180.0:
        raise ValueError("point shadow FOV must be between 0 and 180 degrees")
    if aspect <= 0.0 or near <= 0.0 or far <= near:
        raise ValueError("invalid point shadow perspective volume")
    scale = 1.0 / np.tan(np.radians(float(fov_degrees)) * 0.5)
    return np.array(
        (
            (scale / aspect, 0.0, 0.0, 0.0),
            (0.0, scale, 0.0, 0.0),
            (0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)),
            (0.0, 0.0, -1.0, 0.0),
        ),
        dtype=np.float32,
    )


def point_light_face_matrices(
    position,
    *,
    near: float,
    far: float,
) -> tuple[tuple[float, ...], ...]:
    position = np.asarray(position, dtype=np.float64)
    directions_and_ups = (
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    )
    projection = perspective_matrix(90.0, 1.0, near, far)
    matrices = []
    for direction, up in directions_and_ups:
        target = position + np.asarray(direction, dtype=np.float64)
        matrix = projection @ look_at_matrix(position, target, up)
        matrices.append(tuple(float(value) for value in matrix.reshape(-1)))
    return tuple(matrices)


@dataclass(frozen=True, slots=True)
class PointShadowBinding:
    light_id: str
    texture: int
    position: tuple[float, float, float]
    range: float
    bias: float = 0.35


@dataclass(slots=True)
class PointShadowMap:
    size: int
    framebuffer: int
    cube_texture: int
    depth_renderbuffer: int

    @classmethod
    def create(cls, size: int = 256) -> "PointShadowMap":
        size = int(size)
        if size <= 0:
            raise ValueError("point-shadow size must be positive")
        texture = framebuffer = renderbuffer = 0
        previous = int(glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        try:
            texture = _single_gl_id(glGenTextures(1))
            glBindTexture(GL_TEXTURE_CUBE_MAP, texture)
            for face in POINT_SHADOW_FACES:
                glTexImage2D(
                    face,
                    0,
                    GL_RGBA32F,
                    size,
                    size,
                    0,
                    GL_RGBA,
                    GL_FLOAT,
                    None,
                )
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

            framebuffer = _single_gl_id(glGenFramebuffers(1))
            renderbuffer = _single_gl_id(glGenRenderbuffers(1))
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
            glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size, size)
            glFramebufferRenderbuffer(
                GL_FRAMEBUFFER,
                GL_DEPTH_ATTACHMENT,
                GL_RENDERBUFFER,
                renderbuffer,
            )
            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                POINT_SHADOW_FACES[0],
                texture,
                0,
            )
            status = int(glCheckFramebufferStatus(GL_FRAMEBUFFER))
            if status != GL_FRAMEBUFFER_COMPLETE:
                raise PointShadowUnavailable(
                    f"point shadow framebuffer incomplete: 0x{status:04x}"
                )
            return cls(size, framebuffer, texture, renderbuffer)
        except Exception:
            if renderbuffer:
                glDeleteRenderbuffers(1, [renderbuffer])
            if framebuffer:
                glDeleteFramebuffers(1, [framebuffer])
            if texture:
                glDeleteTextures(1, [texture])
            raise
        finally:
            glBindRenderbuffer(GL_RENDERBUFFER, 0)
            glBindFramebuffer(GL_FRAMEBUFFER, previous)
            glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

    @contextmanager
    def render_face(self, face_index: int):
        if face_index < 0 or face_index >= len(POINT_SHADOW_FACES):
            raise ValueError("point-shadow face index must be in range 0..5")
        previous_framebuffer = int(glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        previous_viewport = tuple(int(value) for value in glGetIntegerv(GL_VIEWPORT))
        previous_clear = tuple(float(value) for value in glGetFloatv(GL_COLOR_CLEAR_VALUE))
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            POINT_SHADOW_FACES[face_index],
            self.cube_texture,
            0,
        )
        glViewport(0, 0, self.size, self.size)
        glEnable(GL_DEPTH_TEST)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        try:
            yield
        finally:
            glUseProgram(0)
            glBindFramebuffer(GL_FRAMEBUFFER, previous_framebuffer)
            glViewport(*previous_viewport)
            glClearColor(*previous_clear)

    def binding(
        self,
        light_id: str,
        position,
        light_range: float,
        *,
        bias: float = 0.35,
    ) -> PointShadowBinding:
        return PointShadowBinding(
            light_id=str(light_id),
            texture=self.cube_texture,
            position=tuple(float(value) for value in position),
            range=max(0.0001, float(light_range)),
            bias=max(0.0, float(bias)),
        )

    def dispose(self) -> None:
        if self.depth_renderbuffer:
            glDeleteRenderbuffers(1, [self.depth_renderbuffer])
        if self.framebuffer:
            glDeleteFramebuffers(1, [self.framebuffer])
        if self.cube_texture:
            glDeleteTextures(1, [self.cube_texture])
        self.depth_renderbuffer = 0
        self.framebuffer = 0
        self.cube_texture = 0


@dataclass(slots=True)
class PointShadowCasterShader:
    program: int
    locations: dict[str, int]

    @classmethod
    def create(cls) -> "PointShadowCasterShader":
        program = compile_program(
            POINT_SHADOW_CASTER_VERTEX_SOURCE,
            POINT_SHADOW_CASTER_FRAGMENT_SOURCE,
        )
        names = (
            "u_light_matrix",
            "u_texture",
            "u_alpha_cutout",
            "u_alpha_cutoff",
            "u_light_position",
            "u_light_range",
        )
        return cls(
            program,
            {name: int(glGetUniformLocation(program, name)) for name in names},
        )

    def bind(
        self,
        light_matrix: tuple[float, ...],
        light_position,
        light_range: float,
        *,
        texture: int = 0,
        alpha_cutout: bool = False,
        alpha_cutoff: float = 0.01,
    ) -> None:
        glUseProgram(self.program)
        glUniformMatrix4fv(
            self.locations["u_light_matrix"],
            1,
            GL_TRUE,
            light_matrix,
        )
        glUniform3f(
            self.locations["u_light_position"],
            float(light_position[0]),
            float(light_position[1]),
            float(light_position[2]),
        )
        glUniform1f(self.locations["u_light_range"], float(light_range))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, int(texture or 0))
        glUniform1i(self.locations["u_texture"], 0)
        glUniform1i(self.locations["u_alpha_cutout"], int(bool(alpha_cutout)))
        glUniform1f(
            self.locations["u_alpha_cutoff"],
            max(0.0, min(1.0, float(alpha_cutoff))),
        )

    def dispose(self) -> None:
        if self.program:
            glDeleteProgram(self.program)
        self.program = 0
