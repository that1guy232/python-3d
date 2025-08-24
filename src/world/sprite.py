"""World-space sprite (camera-facing billboard).

Draws a textured quad in 3D that always faces the camera. Uses the
fixed-function pipeline for simplicity to match the current codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pygame.math import Vector3
from OpenGL.GL import (
    glEnable,
    glDisable,
    glBindTexture,
    glBegin,
    glEnd,
    glDepthMask,
    glTexCoord2f,
    glVertex3f,
    glColor3f,
    glColor4f,
    glBlendFunc,
    GL_TEXTURE_2D,
    GL_QUADS,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
)
from config import WIDTH, HEIGHT, VIEWDISTANCE


# Internal helpers for billboard math
_EPS = 1e-6
_EPS2 = _EPS * _EPS


def _flatten_xz(v: Vector3) -> Vector3:
    return Vector3(v.x, 0.0, v.z)


def _length_sq(v: Vector3) -> float:
    # pygame's Vector3 has length_squared; fall back if missing
    try:
        return v.length_squared()  # type: ignore[attr-defined]
    except Exception:
        return v.x * v.x + v.y * v.y + v.z * v.z


def _safe_normalize(v: Vector3) -> Vector3 | None:
    if _length_sq(v) <= _EPS2:
        return None
    return v.normalize()


def _billboard_axes(
    camera, pitch_effect: bool, world_up: Vector3
) -> tuple[Vector3, Vector3] | None:
    """Compute right/up axes for a billboard.

    - pitch_effect=True: spherical billboard (tilts with camera pitch)
    - pitch_effect=False: cylindrical billboard (up locked to world_up)
    Returns (right, up) or None if degenerate.
    """
    forward = getattr(camera, "_forward", None)
    right = getattr(camera, "_right", None)
    if forward is None or right is None:
        return None

    if pitch_effect:
        r = _safe_normalize(right)
        f = _safe_normalize(forward)
        if r is None or f is None:
            return None
        up = r.cross(f)
        if _length_sq(up) <= _EPS2:
            up = world_up
        else:
            up = up.normalize()
        return r, up
    else:
        # Cylindrical: only yaw-align, keep up locked to world
        r_flat = _flatten_xz(right)
        if _length_sq(r_flat) <= _EPS2:
            f_flat = _flatten_xz(forward)
            if _length_sq(f_flat) <= _EPS2:
                return None
            r_flat = world_up.cross(f_flat)
        r = r_flat.normalize()
        up = world_up
        return r, up


@dataclass
class WorldSprite:
    position: Vector3
    size: tuple[float, float]
    texture: int
    camera: any  # expects Camera with _right and _forward vectors updated via update_rotation()
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def draw_untextured(self):  # keep interface parity
        self.draw()

    def draw(self, pitch_effect: bool = False):  # pragma: no cover - visual
        if not self.texture:
            return

        # Compute billboard axes
        # - If pitch_effect: use full camera pitch (spherical billboard)
        # - Else: lock to world up (cylindrical billboard)
        world_up = Vector3(0.0, 1.0, 0.0)

        axes = _billboard_axes(self.camera, pitch_effect, world_up)
        if not axes:
            return
        right, up = axes

        w, h = self.size
        hw, hh = w * 0.5, h * 0.5

        # Four corners in world space (counter-clockwise)
        center = self.position
        tl = center - right * hw + up * hh
        tr = center + right * hw + up * hh
        br = center + right * hw - up * hh
        bl = center - right * hw - up * hh

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glColor3f(*self.color)

        glBegin(GL_QUADS)
        # t,v ordering: (u, v) then vertex
        glTexCoord2f(0.0, 1.0)
        glVertex3f(tl.x, tl.y, tl.z)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(tr.x, tr.y, tr.z)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(br.x, br.y, br.z)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(bl.x, bl.y, bl.z)
        glEnd()

        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)


def draw_sprites_batched(
    sprites: list["WorldSprite"],
    camera,
    ground_height_fn=None,
) -> None:
    """Batch-render world sprites with optimized inner loop."""
    if not sprites:
        return

    # Precompute axes and frustum params
    # Culling space: full camera basis (includes pitch) for accurate frustum checks
    forward = camera._forward
    right_cull = camera._right.normalize()
    up_cull = right_cull.cross(forward).normalize()
    if right_cull.length() == 0 or up_cull.length() == 0:
        return

    # Rendering space: cylindrical billboard locked to world up (ignores camera pitch)
    world_up = Vector3(0.0, 1.0, 0.0)
    right_render = Vector3(camera._right.x, 0.0, camera._right.z)
    if right_render.length() < 1e-6:
        f_flat = Vector3(forward.x, 0.0, forward.z)
        if f_flat.length() < 1e-6:
            return
        right_render = world_up.cross(f_flat)
    right_render = right_render.normalize()
    up_render = world_up

    # tan(fov_y/2) derived from camera._fov_scale = (HEIGHT/2) / tan(fov_y/2)
    tan_half_fov = (HEIGHT * 0.5) / getattr(camera, "_fov_scale", 1.0)
    aspect = WIDTH / HEIGHT

    # Filter (cull) and collect for sorting in one pass
    textured = []
    cam_pos = camera.position
    for s in sprites:
        tex = getattr(s, "texture", None)
        if tex:
            rel = s.position - cam_pos
            depth = rel.dot(forward)

            # Sprite-size-aware culling slack so shadows near edges don't disappear
            # Allow a little distance behind the camera and extra to the left/right.
            # These can be overridden on the camera if desired.
            w, h = s.size
            hw, hh = w * 0.75, h * 0.75
            side_extra_base = getattr(camera, "sprite_side_cull_extra", 0.0)
            back_extra_base = getattr(camera, "sprite_back_cull_extra", 0.0)
            # Scale extras by sprite size with a small minimum
            # Bump horizontal slack a bit to keep ground-conforming shadows from getting clipped
            side_extra = max(side_extra_base, 0.9 * w)
            back_extra = max(back_extra_base, 0.75 * w)

            # Beyond far plane or too far behind camera
            if depth <= -back_extra or depth > VIEWDISTANCE:
                continue

            # Simple frustum check using camera-space extents
            x_cam = rel.dot(right_cull)
            y_cam = rel.dot(up_cull)

            # Frustum half-extents at this depth (in world units)
            half_v = depth * tan_half_fov
            half_h = half_v * aspect

            # Expand by sprite half-size to keep billboards that touch the edge
            # and add a bit more horizontally so ground shadows near edges remain visible
            if abs(x_cam) > (half_h + hw + side_extra) or abs(y_cam) > (half_v + hh):
                continue

            # Survives culling; keep for sorting
            textured.append((depth, tex, s))

    if not textured:
        return

    # Sort back-to-front (highest depth first)
    textured.sort(key=lambda x: x[0], reverse=True)

    # right/up for rendering already computed above

    # Prepare GL state for sprite rendering (no shadows)
    glDepthMask(False)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)

    # Group sprites by texture while preserving depth order
    batches = []
    current_tex = None
    current_batch = []
    for _, tex, s in textured:
        if tex != current_tex:
            if current_batch:
                batches.append((current_tex, current_batch))
            current_batch = [(s.position, s.size, s.color)]
            current_tex = tex
        else:
            current_batch.append((s.position, s.size, s.color))
    if current_batch:
        batches.append((current_tex, current_batch))

    # Render all batches (keep depth mask disabled and blend/texture enabled)

    for tex, batch in batches:
        glBindTexture(GL_TEXTURE_2D, tex)
        glBegin(GL_QUADS)
        for pos, (w, h), color in batch:
            # Pre-unpack color
            r, g, b = color

            # Compute half dimensions
            hw = w * 0.5
            hh = h * 0.5

            # Precompute scaled axis components
            rx_hw = right_render.x * hw
            ry_hw = right_render.y * hw
            rz_hw = right_render.z * hw

            ux_hh = up_render.x * hh
            uy_hh = up_render.y * hh
            uz_hh = up_render.z * hh

            # Top-left
            tl_x = pos.x - rx_hw + ux_hh
            tl_y = pos.y - ry_hw + uy_hh
            tl_z = pos.z - rz_hw + uz_hh

            # Top-right
            tr_x = pos.x + rx_hw + ux_hh
            tr_y = pos.y + ry_hw + uy_hh
            tr_z = pos.z + rz_hw + uz_hh

            # Bottom-right
            br_x = pos.x + rx_hw - ux_hh
            br_y = pos.y + ry_hw - uy_hh
            br_z = pos.z + rz_hw - uz_hh

            # Bottom-left
            bl_x = pos.x - rx_hw - ux_hh
            bl_y = pos.y - ry_hw - uy_hh
            bl_z = pos.z - rz_hw - uz_hh

            # Issue vertices with precomputed scalars (force alpha to 1 after shadow pass)
            glColor4f(r, g, b, 1.0)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(tl_x, tl_y, tl_z)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(tr_x, tr_y, tr_z)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(br_x, br_y, br_z)
            glTexCoord2f(0.0, 0.0)
            glVertex3f(bl_x, bl_y, bl_z)
        glEnd()

    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glDepthMask(True)
