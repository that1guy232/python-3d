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
        thickness: float = 0.0,
    ):
        self.width = width
        self.height = height
        self.depth = depth
        # thickness along -X to give the wall a small box-like volume
        # 0.0 = original single-plane behavior
        self.thickness = float(thickness)
        # OpenGL texture id (optional)
        self.texture = texture
        # UV repeat (u_repeat, v_repeat) to control tiling of the texture across the wall
        self.uv_repeat = uv_repeat

        super().__init__(position=position or Vector3(0, 0, 0))
        self._generate_vertices()

        # Faces and default colors. If thickness is zero we keep the original
        # single front-facing quad. Otherwise create a thin box (6 faces).
        if self.thickness <= 0.0:
            # Single front-facing quad (facing +X)
            self.faces = [(0, 1, 2, 3)]
            self.face_colors = [(1.0, 1.0, 1.0)]
        else:
            # Vert ordering when thickness > 0: front 0..3, back 4..7
            # Faces: front, back, right(+Z), left(-Z), top(+Y), bottom(-Y)
            self.faces = [
                (0, 1, 2, 3),  # front
                (5, 4, 7, 6),  # back
                (1, 5, 6, 2),  # right (+Z)
                (4, 0, 3, 7),  # left (-Z)
                (3, 2, 6, 7),  # top (+Y)
                (4, 5, 1, 0),  # bottom (-Y)
            ]
            # default white for all faces
            self.face_colors = [(1.0, 1.0, 1.0)] * len(self.faces)

    def _generate_vertices(self):
        w = self.width  # plane at x = +width
        h = self.height
        d = self.depth
        # If thickness is zero, keep the original single-plane vertices.
        # When thickness > 0 we create an extruded thin box (8 verts) so the
        # wall has fake thickness: front (x = +w) and back (x = w - thickness).
        if self.thickness <= 0.0:
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
        else:
            t = self.thickness
            # Front quad (x = w)
            f0 = Vector3(w, -h, -d)
            f1 = Vector3(w, -h, d)
            f2 = Vector3(w, h, d)
            f3 = Vector3(w, h, -d)
            # Back quad (x = w - t)
            b0 = Vector3(w - t, -h, -d)
            b1 = Vector3(w - t, -h, d)
            b2 = Vector3(w - t, h, d)
            b3 = Vector3(w - t, h, -d)
            # Order: front(0..3), back(4..7)
            self.local_vertices = [f0, f1, f2, f3, b0, b1, b2, b3]

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
            # Draw each face; compute per-face UVs for side faces when the
            # geometry was extruded.
            for face_idx, face in enumerate(self.faces):
                # Determine world spans for this face to compute a sensible
                # repeat if caller left uv_repeat at the default (1.0, 1.0).
                if len(face) != 4:
                    continue
                a, b, c, d = face
                # Determine representative verts to compute spans
                va = world_verts[a]
                vb = world_verts[b]
                vc = world_verts[c]

                # Compute edge vectors in world space
                e1 = vb - va
                e2 = vc - vb
                # approximate spans as lengths of two edges
                span_u = e1.length()
                span_v = e2.length()

                u_r, v_r = u_repeat, v_repeat
                if tex_size and (u_repeat == 1.0 and v_repeat == 1.0):
                    tex_w, tex_h = tex_size
                    u_r = span_u / max(1e-6, float(tex_w))
                    v_r = span_v / max(1e-6, float(tex_h))

                # Default base uv mapping for a quad (bottom-left -> top-right)
                face_uvs = [
                    (0.0, 0.0),
                    (u_r, 0.0),
                    (u_r, v_r),
                    (0.0, v_r),
                ]

                # If the wall is extruded, avoid stretching by sampling a thin
                # strip from the front/back texture for the side/top/bottom faces
                # so they visually continue the edge pixels instead of stretching
                # the whole front texture across a thin face.
                if self.thickness > 0.0 and tex_size:
                    tex_w, tex_h = tex_size
                    # a small strip size in texture-space (one texel scaled by repeat)
                    strip_u = max(
                        1.0 / max(1.0, float(tex_w)) * u_repeat, 0.001 * u_repeat
                    )
                    strip_v = max(
                        1.0 / max(1.0, float(tex_h)) * v_repeat, 0.001 * v_repeat
                    )

                    # Face ordering when extruded defined in __init__:
                    # 0: front, 1: back, 2: right (+Z), 3: left (-Z), 4: top (+Y), 5: bottom (-Y)
                    if face_idx in (0, 1):
                        # front/back: unchanged (back may be flipped by winding elsewhere)
                        face_uvs = [
                            (0.0, 0.0),
                            (u_r, 0.0),
                            (u_r, v_r),
                            (0.0, v_r),
                        ]
                    elif face_idx == 2:
                        # right side: sample a thin vertical strip from the right
                        # edge of the front texture and tile vertically across height
                        u_min = max(0.0, u_repeat - strip_u)
                        u_max = u_repeat
                        face_uvs = [
                            (u_min, 0.0),
                            (u_max, 0.0),
                            (u_max, v_r),
                            (u_min, v_r),
                        ]
                    elif face_idx == 3:
                        # left side: sample a thin vertical strip from the left
                        u_min = 0.0
                        u_max = min(strip_u, u_repeat)
                        face_uvs = [
                            (u_min, 0.0),
                            (u_max, 0.0),
                            (u_max, v_r),
                            (u_min, v_r),
                        ]
                    elif face_idx == 4:
                        # top: sample a thin horizontal strip from the top row of the
                        # front texture and tile across the top face's span
                        v_min = max(0.0, v_r - strip_v)
                        v_max = v_r
                        face_uvs = [
                            (0.0, v_min),
                            (u_r, v_min),
                            (u_r, v_max),
                            (0.0, v_max),
                        ]
                    elif face_idx == 5:
                        # bottom: sample a thin horizontal strip from the bottom
                        v_min = 0.0
                        v_max = min(strip_v, v_r)
                        face_uvs = [
                            (0.0, v_min),
                            (u_r, v_min),
                            (u_r, v_max),
                            (0.0, v_max),
                        ]

                glBegin(GL_QUADS)
                for vi, idx in enumerate((a, b, c, d)):
                    uv = face_uvs[vi]
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
