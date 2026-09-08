"""Small OpenGL state transitions without renderer-package dependencies."""

from OpenGL.GL import glUseProgram


def use_fixed_pipeline() -> None:
    """Leave any programmable shader without importing legacy renderer state."""

    glUseProgram(0)
