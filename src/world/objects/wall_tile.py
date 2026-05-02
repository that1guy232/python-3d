import math
import numpy as np
from pygame.math import Vector3
from core.object3d import Object3D
from OpenGL.GL import (
    glEnable,
    glDisable,
    glBindTexture,
    glTexParameteri,
    glGenBuffers,
    glBindBuffer,
    glBufferData,
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
    GL_ARRAY_BUFFER,
    GL_STATIC_DRAW,
)
from core.mesh import BatchedMesh
from textures.texture_utils import get_texture_size


_LOCAL_FACE_NORMALS = (
    (1.0, 0.0, 0.0),   # front
    (-1.0, 0.0, 0.0),  # back
    (0.0, 0.0, 1.0),   # right
    (0.0, 0.0, -1.0),  # left
    (0.0, 1.0, 0.0),   # top
    (0.0, -1.0, 0.0),  # bottom
)


def _rotate_local_direction(tile, direction: tuple[float, float, float]) -> Vector3:
    tile._update_rotation_matrix()
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = tile._rotation_matrix
    x, y, z = direction
    world = Vector3(
        x * r00 + y * r01 + z * r02,
        x * r10 + y * r11 + z * r12,
        x * r20 + y * r21 + z * r22,
    )
    return world.normalize() if world.length_squared() > 1e-12 else Vector3(0, 1, 0)


def _sun_light_factor(tile, face_idx: int, sun_direction=None) -> float:
    if sun_direction is None or face_idx >= len(_LOCAL_FACE_NORMALS):
        return 1.0

    light_dir = Vector3(
        -float(sun_direction.x),
        -float(sun_direction.y),
        -float(sun_direction.z),
    )
    if light_dir.length_squared() <= 1e-12:
        return 1.0

    light_dir = light_dir.normalize()
    normal = _rotate_local_direction(tile, _LOCAL_FACE_NORMALS[face_idx])
    diffuse = max(
        0.0,
        normal.x * light_dir.x
        + normal.y * light_dir.y
        + normal.z * light_dir.z,
    )

    ambient = 0.6
    direct = 0.55
    return min(1.15, ambient + direct * diffuse)


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


    def draw(self, camera=None):
        """Immediate-mode draw. If self.texture is set, render textured; otherwise
        render a flat colored quad using face_colors.
        """
        world_verts = self.get_world_vertices()
        if not world_verts:
            return
        
        # FIXED: Use optimized camera brightness system with proper blending
        vertex_brightness = []
        for vert in world_verts:
            if camera and hasattr(camera, 'get_brightness_at'):
                # Use the optimized camera method that handles overlapping areas correctly
                brightness = camera.get_brightness_at(vert)
            else:
                # Fallback to default brightness
                brightness = getattr(camera, 'brightness_default', 0.0) if camera else 0.0
            
            vertex_brightness.append(float(brightness))

        u_repeat, v_repeat = self.uv_repeat
        tex_size = get_texture_size(self.texture)

        if self.texture:
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glEnable(GL_ALPHA_TEST)
            glAlphaFunc(GL_GREATER, 0.01)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            for face_idx, face in enumerate(self.faces):
                if len(face) != 4:
                    continue
                sun_factor = _sun_light_factor(
                    self, face_idx, getattr(self, "sun_direction", None)
                )
                a, b, c, d = face
                va = world_verts[a]
                vb = world_verts[b]
                vc = world_verts[c]

                e1 = vb - va
                e2 = vc - vb
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


                if self.thickness > 0.0 and tex_size:
                    tex_w, tex_h = tex_size
                    strip_u = max(
                        1.0 / max(1.0, float(tex_w)) * u_repeat, 0.001 * u_repeat
                    )
                    strip_v = max(
                        1.0 / max(1.0, float(tex_h)) * v_repeat, 0.001 * v_repeat
                    )


                    if face_idx in (0, 1):
                        face_uvs = [
                            (0.0, 0.0),
                            (u_r, 0.0),
                            (u_r, v_r),
                            (0.0, v_r),
                        ]
                    elif face_idx == 2:
                        u_min = max(0.0, u_repeat - strip_u)
                        u_max = u_repeat
                        face_uvs = [
                            (u_min, 0.0),
                            (u_max, 0.0),
                            (u_max, v_r),
                            (u_min, v_r),
                        ]
                    elif face_idx == 3:
                        u_min = 0.0
                        u_max = min(strip_u, u_repeat)
                        face_uvs = [
                            (u_min, 0.0),
                            (u_max, 0.0),
                            (u_max, v_r),
                            (u_min, v_r),
                        ]
                    elif face_idx == 4:
                        v_min = max(0.0, v_r - strip_v)
                        v_max = v_r
                        face_uvs = [
                            (0.0, v_min),
                            (u_r, v_min),
                            (u_r, v_max),
                            (0.0, v_max),
                        ]
                    elif face_idx == 5:
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
                    base_brightness = max(0.0, min(1.0, vertex_brightness[idx]))
                    brightness = base_brightness * sun_factor
                    brightness = max(0.0, min(1.0, brightness))
                    glColor4f(brightness, brightness, brightness, 1.0)
                    glTexCoord2f(uv[0], uv[1])
                    glVertex3f(v.x, v.y, v.z)
                glEnd()
      
            glDisable(GL_BLEND)
            glDisable(GL_ALPHA_TEST)
            glDisable(GL_TEXTURE_2D)


def _normalized_color(color) -> tuple[float, float, float]:
    if any(component > 2.0 for component in color):
        return tuple(component / 255.0 for component in color)
    return tuple(float(component) for component in color)


def _textured_face_uvs(
    tile: WallTile,
    face_idx: int,
    face: tuple[int, int, int, int],
    world_verts: list[Vector3],
    tex_size,
) -> list[tuple[float, float]]:
    a, b, c, _ = face
    va = world_verts[a]
    vb = world_verts[b]
    vc = world_verts[c]

    e1 = vb - va
    e2 = vc - vb
    span_u = e1.length()
    span_v = e2.length()

    u_repeat, v_repeat = tile.uv_repeat
    u_r, v_r = u_repeat, v_repeat
    if tex_size and (u_repeat == 1.0 and v_repeat == 1.0):
        tex_w, tex_h = tex_size
        u_r = span_u / max(1e-6, float(tex_w))
        v_r = span_v / max(1e-6, float(tex_h))

    face_uvs = [
        (0.0, 0.0),
        (u_r, 0.0),
        (u_r, v_r),
        (0.0, v_r),
    ]

    if tile.thickness > 0.0 and tex_size:
        tex_w, tex_h = tex_size
        strip_u = max(
            1.0 / max(1.0, float(tex_w)) * u_repeat,
            0.001 * u_repeat,
        )
        strip_v = max(
            1.0 / max(1.0, float(tex_h)) * v_repeat,
            0.001 * v_repeat,
        )

        if face_idx in (0, 1):
            face_uvs = [
                (0.0, 0.0),
                (u_r, 0.0),
                (u_r, v_r),
                (0.0, v_r),
            ]
        elif face_idx == 2:
            u_min = max(0.0, u_repeat - strip_u)
            u_max = u_repeat
            face_uvs = [
                (u_min, 0.0),
                (u_max, 0.0),
                (u_max, v_r),
                (u_min, v_r),
            ]
        elif face_idx == 3:
            u_min = 0.0
            u_max = min(strip_u, u_repeat)
            face_uvs = [
                (u_min, 0.0),
                (u_max, 0.0),
                (u_max, v_r),
                (u_min, v_r),
            ]
        elif face_idx == 4:
            v_min = max(0.0, v_r - strip_v)
            v_max = v_r
            face_uvs = [
                (0.0, v_min),
                (u_r, v_min),
                (u_r, v_max),
                (0.0, v_max),
            ]
        elif face_idx == 5:
            v_min = 0.0
            v_max = min(strip_v, v_r)
            face_uvs = [
                (0.0, v_min),
                (u_r, v_min),
                (u_r, v_max),
                (0.0, v_max),
            ]

    return face_uvs


def _tile_vertex_data(
    tile: WallTile,
    camera=None,
    default_brightness: float = 1.0,
    sun_direction=None,
) -> np.ndarray:
    world_verts = tile.get_world_vertices()
    if not world_verts:
        return np.zeros((0, 8 if tile.texture else 6), dtype=np.float32)

    if camera and hasattr(camera, "get_brightness_batch"):
        vertex_brightness = camera.get_brightness_batch(world_verts)
    else:
        get_brightness_at = getattr(camera, "get_brightness_at", None)
        vertex_brightness = [
            get_brightness_at(v) if callable(get_brightness_at) else default_brightness
            for v in world_verts
        ]

    textured = bool(tile.texture)
    tex_size = get_texture_size(tile.texture) if textured else None
    rows = []

    for face_idx, face in enumerate(tile.faces):
        if len(face) != 4:
            continue

        sun_factor = _sun_light_factor(tile, face_idx, sun_direction)
        a, b, c, d = face
        tri_indices = (a, b, c, a, c, d)

        if textured:
            face_uvs = _textured_face_uvs(tile, face_idx, face, world_verts, tex_size)
            tri_uvs = (
                face_uvs[0],
                face_uvs[1],
                face_uvs[2],
                face_uvs[0],
                face_uvs[2],
                face_uvs[3],
            )
            for idx, uv in zip(tri_indices, tri_uvs):
                v = world_verts[idx]
                base_brightness = max(
                    0.0,
                    min(1.0, float(vertex_brightness[idx])),
                )
                brightness = max(
                    0.0,
                    min(1.0, base_brightness * sun_factor),
                )
                rows.append(
                    (
                        v.x,
                        v.y,
                        v.z,
                        brightness,
                        brightness,
                        brightness,
                        uv[0],
                        uv[1],
                    )
                )
        else:
            color = (
                tile.face_colors[face_idx]
                if hasattr(tile, "face_colors") and face_idx < len(tile.face_colors)
                else (1.0, 1.0, 1.0)
            )
            r, g, b_col = _normalized_color(color)
            r = max(0.0, min(1.0, r * sun_factor))
            g = max(0.0, min(1.0, g * sun_factor))
            b_col = max(0.0, min(1.0, b_col * sun_factor))
            for idx in tri_indices:
                v = world_verts[idx]
                rows.append((v.x, v.y, v.z, r, g, b_col))

    columns = 8 if textured else 6
    if not rows:
        return np.zeros((0, columns), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def build_wall_tile_batches(
    tiles: list[WallTile],
    *,
    camera=None,
    default_brightness: float = 1.0,
    sun_direction=None,
) -> list[BatchedMesh]:
    """Merge static WallTile objects into VBO-backed batches grouped by texture."""
    chunks_by_texture = {}
    for tile in tiles:
        data = _tile_vertex_data(
            tile,
            camera=camera,
            default_brightness=default_brightness,
            sun_direction=sun_direction,
        )
        if data.size == 0:
            continue
        texture = tile.texture if tile.texture else None
        chunks_by_texture.setdefault(texture, []).append(data)

    batches: list[BatchedMesh] = []
    for texture, chunks in chunks_by_texture.items():
        vertex_data = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        if texture:
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        batches.append(
            BatchedMesh(
                vbo_vertices=vbo,
                vertex_count=vertex_data.shape[0],
                texture=texture if texture else None,
                alpha_test=bool(texture),
            )
        )

    return batches
