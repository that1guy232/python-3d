from pygame.math import Vector3
from core.object3d import Object3D
from OpenGL.GL import (
    glEnable,
    glDisable,
    glBindTexture,
    glTexParameteri,
    glBegin,
    glEnd,
    glTexCoord2f,
    glVertex3f,
    glColor3f,
    glColor4f,
    glBlendFunc,
    glAlphaFunc,
    GL_TEXTURE_2D,
    GL_QUADS,
    GL_BLEND,
    GL_ALPHA_TEST,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_GREATER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_REPEAT,
)
from textures.texture_utils import get_texture_size


class WallTile(Object3D):
    """A single, flat vertical wall (quad) used as part of the scene.

    This creates a plane positioned at local X = +width (so the face normal
    points along +X). The plane spans Y (height) and Z (depth).

    Parameters
    ----------
    position : Vector3 | None
        World-space position (center) of the tile.
    width : float
        Half-distance along X from the local origin to the plane (plane lies at x=+width).
    height : float
        Half-height along Y (extends +/- height).
    depth : float
        Half-depth along Z (extends +/- depth).
    """

    def __init__(
        self,
        position=None,
        width=50,
        height=5,
        depth=50,
        texture: int | None = None,
        uv_repeat: tuple[float, float] = (1.0, 1.0),
    ):
        self.width = width
        self.height = height
        self.depth = depth
        # OpenGL texture id (optional)
        self.texture = texture
        # UV repeat (u_repeat, v_repeat) to control tiling of the texture across the wall
        self.uv_repeat = uv_repeat

        super().__init__(position=position or Vector3(0, 0, 0))
        self._generate_vertices()

        # Single front-facing quad (facing +X)
        self.faces = [
            (0, 1, 2, 3),
        ]
        # Default face color (RGB floats 0..1)
        self.face_colors = [
            (1.0, 1.0, 1.0),
        ]

    def _generate_vertices(self):
        w = self.width  # plane at x = +width
        h = self.height
        d = self.depth
        # Local space vertices of the vertical plane (counter-clockwise when
        # looking from +X):
        #   3 ---- 2   (top)
        #   |      |
        #   0 ---- 1   (bottom)
        self.local_vertices = [
            Vector3(w, -h, -d),
            Vector3(w, -h, d),
            Vector3(w, h, d),
            Vector3(w, h, -d),
        ]

    def draw_untextured(self):
        """Draw the wall as an untextured colored quad (immediate mode)."""
        self.draw()

    def draw(self):
        """Immediate-mode draw. If self.texture is set, render textured; otherwise
        render a flat colored quad using face_colors.
        """
        world_verts = self.get_world_vertices()
        if not world_verts:
            return

        # Single-quad per face; base (u,v) mapping: bottom-left (0,0) -> top-right (1,1)
        u_base = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        # For textured walls, use the registered texture pixel size to compute
        # a sensible repeat so the texture tiles instead of stretching across
        # large walls. If the caller provided an explicit non-default
        # `uv_repeat` we keep it; otherwise compute from texture size.
        u_repeat, v_repeat = self.uv_repeat
        if self.texture:
            tex_size = get_texture_size(self.texture)
            if tex_size and (u_repeat == 1.0 and v_repeat == 1.0):
                tex_w, tex_h = tex_size
                # World spans for the quad: horizontal along Z is 2*depth, vertical along Y is 2*height
                world_u = 2.0 * self.depth
                world_v = 2.0 * self.height
                # Compute repeats as ratio of world size to texture pixel size.
                # This ensures one texture image covers ~tex_w x tex_h world units.
                u_repeat = world_u / max(1e-6, float(tex_w))
                v_repeat = world_v / max(1e-6, float(tex_h))

        uvs = [(u * u_repeat, v * v_repeat) for (u, v) in u_base]

        if self.texture:
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            # Enable alpha testing so fully transparent texture pixels are discarded
            # and do not write to the depth buffer. This allows objects behind
            # the wall to remain visible through transparent texels.
            glEnable(GL_ALPHA_TEST)
            glAlphaFunc(GL_GREATER, 0.01)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            # Ensure this texture uses repeat wrapping so uv_repeat tiles it.
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            # Use full white with full alpha so the texture's alpha is applied
            glColor4f(1.0, 1.0, 1.0, 1.0)
            for face_idx, face in enumerate(self.faces):
                a, b, c, d = face
                glBegin(GL_QUADS)
                for vi, idx in enumerate((a, b, c, d)):
                    uv = uvs[vi]
                    v = world_verts[idx]
                    glTexCoord2f(uv[0], uv[1])
                    glVertex3f(v.x, v.y, v.z)
                glEnd()
            glDisable(GL_BLEND)
            glDisable(GL_ALPHA_TEST)
            glDisable(GL_TEXTURE_2D)
        else:
            # Untextured: use per-face color (RGB floats)
            for face_idx, face in enumerate(self.faces):
                a, b, c, d = face
                color = (
                    self.face_colors[face_idx]
                    if face_idx < len(self.face_colors)
                    else (
                        1.0,
                        1.0,
                        1.0,
                    )
                )
                # normalize if color appears in 0-255 range
                if any(x > 2.0 for x in color):
                    color = tuple(x / 255.0 for x in color)
                glColor3f(*color)
                glBegin(GL_QUADS)
                for idx in (a, b, c, d):
                    v = world_verts[idx]
                    glVertex3f(v.x, v.y, v.z)
                glEnd()
