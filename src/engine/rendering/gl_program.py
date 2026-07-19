"""Small OpenGL shader compile/link helper independent of legacy rendering."""

from __future__ import annotations

from OpenGL.GL import (
    GL_COMPILE_STATUS,
    GL_FRAGMENT_SHADER,
    GL_LINK_STATUS,
    GL_VERTEX_SHADER,
    glAttachShader,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteProgram,
    glDeleteShader,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glLinkProgram,
    glShaderSource,
)


def _decode_log(log) -> str:
    if isinstance(log, bytes):
        return log.decode("utf-8", "replace")
    return str(log)


def compile_shader(shader_type: int, source: str) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = _decode_log(glGetShaderInfoLog(shader)).strip()
        glDeleteShader(shader)
        raise RuntimeError(log or "shader compilation failed")
    return int(shader)


def compile_program(vertex_source: str, fragment_source: str) -> int:
    vertex_shader = 0
    fragment_shader = 0
    program = 0
    try:
        vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_source)
        fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_source)
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
