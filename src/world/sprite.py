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


def _clamp01(v: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


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

        
        r, g, b = self.color

        glColor3f(r, g, b)

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
    """Optimized minimal-change batched renderer that keeps glBegin/quads but reduces Python overhead."""

    if not sprites:
        return

    # camera validity
    if not hasattr(camera, "_forward") or not hasattr(camera, "_right"):
        return

    # hoist GL functions + constants locally (faster attribute lookup)
    glBindTexture_local = glBindTexture
    glTexCoord2f_local = glTexCoord2f
    glVertex3f_local = glVertex3f
    glColor4f_local = glColor4f
    glEnable_local = glEnable
    glDisable_local = glDisable
    glBlendFunc_local = glBlendFunc
    glDepthMask_local = glDepthMask

    # hoist camera / config values
    forward = camera._forward
    cam_pos = camera.position
    fx, fy, fz = forward.x, forward.y, forward.z
    cx, cy, cz = cam_pos.x, cam_pos.y, cam_pos.z

    # small, allocation-free precomputed numpy-matrix fallback:
    use_numpy_transform = False
    R0x = R0y = R0z = R1x = R1y = R1z = R2x = R2y = R2z = None
    if hasattr(camera, "_R"):
        R = camera._R
        # try to extract rows as plain floats to avoid per-sprite numpy ops
        try:
            # works with numpy arrays or nested sequences
            R0x, R0y, R0z = float(R[0][0]), float(R[0][1]), float(R[0][2])
            R1x, R1y, R1z = float(R[1][0]), float(R[1][1]), float(R[1][2])
            R2x, R2y, R2z = float(R[2][0]), float(R[2][1]), float(R[2][2])
            use_numpy_transform = True
        except Exception:
            use_numpy_transform = False

    # frustum/culling params
    fov_scale = getattr(camera, "_fov_scale", (HEIGHT * 0.5) / 1.0)
    tan_half_fov = (HEIGHT * 0.5) / fov_scale
    aspect = WIDTH / HEIGHT

    brightness_default = getattr(camera, "brightness_default", 0.0)
    brightness_areas = getattr(camera, "brightness_areas", [])
    has_brightness_areas = bool(brightness_areas)
    get_brightness_at = getattr(camera, "get_brightness_at", None)

    # cylindrical billboard axes (render)
    world_up = Vector3(0.0, 1.0, 0.0)
    right_render = Vector3(camera._right.x, 0.0, camera._right.z)
    if _length_sq(right_render) < _EPS:
        f_flat = Vector3(forward.x, 0.0, forward.z)
        if _length_sq(f_flat) < _EPS:
            return
        right_render = world_up.cross(f_flat)
    right_render = right_render.normalize()
    up_render = world_up

    rx_x, rx_y, rx_z = right_render.x, right_render.y, right_render.z
    ux_x, ux_y, ux_z = up_render.x, up_render.y, up_render.z

    sprite_side_cull_extra = getattr(camera, "sprite_side_cull_extra", 0.0)
    sprite_back_cull_extra = getattr(camera, "sprite_back_cull_extra", 0.0)

    view_distance = VIEWDISTANCE
    tan_h = tan_half_fov
    asp = aspect

    # Prepare cull fallback axes if numpy-transform not used
    if not use_numpy_transform:
        right_cull = camera._right.normalize()
        up_cull = right_cull.cross(forward).normalize()
        if _length_sq(right_cull) == 0 or _length_sq(up_cull) == 0:
            return
        rcx, rcy, rcz = right_cull.x, right_cull.y, right_cull.z
        ucx, ucy, ucz = up_cull.x, up_cull.y, up_cull.z

    # Cull + collect visible sprites (lightweight tuples)
    visible_sprites = []
    append_vis = visible_sprites.append
    for s in sprites:
        tex = getattr(s, "texture", None)
        if not tex:
            continue

        # inline relative pos
        rel_x = s.position.x - cx
        rel_y = s.position.y - cy
        rel_z = s.position.z - cz

        # depth
        depth = rel_x * fx + rel_y * fy + rel_z * fz

        w, h = s.size
        hw_cull, hh_cull = w * 0.75, h * 0.75
        side_extra = max(sprite_side_cull_extra, 0.9 * w)
        back_extra = max(sprite_back_cull_extra, 0.75 * w)

        if depth <= -back_extra or depth > view_distance:
            continue

        # camera-space coords (avoid numpy allocation)
        if use_numpy_transform:
            x_cam = rel_x * R0x + rel_y * R0y + rel_z * R0z
            y_cam = rel_x * R1x + rel_y * R1y + rel_z * R1z
        else:
            x_cam = rel_x * rcx + rel_y * rcy + rel_z * rcz
            y_cam = rel_x * ucx + rel_y * ucy + rel_z * ucz

        half_v = depth * tan_h
        half_h = half_v * asp

        if abs(x_cam) > (half_h + hw_cull + side_extra) or abs(y_cam) > (half_v + hh_cull):
            continue

        append_vis((depth, tex, s))

    if not visible_sprites:
        return

    # back-to-front sort
    visible_sprites.sort(key=lambda x: x[0], reverse=True)

    # Group by texture
    batches = []
    cur_tex = None
    cur_batch = None
    for _, tex, s in visible_sprites:
        if tex is not cur_tex:
            if cur_batch:
                batches.append((cur_tex, cur_batch))
            cur_tex = tex
            cur_batch = [s]
        else:
            cur_batch.append(s)
    if cur_batch:
        batches.append((cur_tex, cur_batch))

    # GL state once
    glDepthMask_local(False)
    glEnable_local(GL_BLEND)
    glBlendFunc_local(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable_local(GL_TEXTURE_2D)

    # Main render: using glBegin/quads but minimized Python overhead
    for tex, batch in batches:
        glBindTexture_local(GL_TEXTURE_2D, tex)
        glBegin(GL_QUADS)

        for s in batch:
            # inline lots of values to reduce attribute access
            pos = s.position
            px, py, pz = pos.x, pos.y, pos.z
            w, h = s.size
            hw = w * 0.5
            hh = h * 0.5

            # brightness (single call)
            if has_brightness_areas and get_brightness_at is not None:
                brightness = get_brightness_at(pos)
                if brightness < brightness_default:
                    brightness = brightness_default
            else:
                brightness = brightness_default

            cr, cg, cb = s.color
            r = cr * brightness
            g = cg * brightness
            b = cb * brightness
            # clamp - faster inline than calling _clamp01 thrice
            if r < 0.0:
                r = 0.0
            elif r > 1.0:
                r = 1.0
            if g < 0.0:
                g = 0.0
            elif g > 1.0:
                g = 1.0
            if b < 0.0:
                b = 0.0
            elif b > 1.0:
                b = 1.0

            rx_hw = rx_x * hw
            ry_hw = rx_y * hw
            rz_hw = rx_z * hw

            ux_hh = ux_x * hh
            uy_hh = ux_y * hh
            uz_hh = ux_z * hh

            # corners
            tl_x = px - rx_hw + ux_hh
            tl_y = py - ry_hw + uy_hh
            tl_z = pz - rz_hw + uz_hh

            tr_x = px + rx_hw + ux_hh
            tr_y = py + ry_hw + uy_hh
            tr_z = pz + rz_hw + uz_hh

            br_x = px + rx_hw - ux_hh
            br_y = py + ry_hw - uy_hh
            br_z = pz + rz_hw - uz_hh

            bl_x = px - rx_hw - ux_hh
            bl_y = py - ry_hw - uy_hh
            bl_z = pz - rz_hw - uz_hh

            # draw (single color call per quad)
            glColor4f_local(r, g, b, 1.0)

            glTexCoord2f_local(0.0, 1.0)
            glVertex3f_local(tl_x, tl_y, tl_z)

            glTexCoord2f_local(1.0, 1.0)
            glVertex3f_local(tr_x, tr_y, tr_z)

            glTexCoord2f_local(1.0, 0.0)
            glVertex3f_local(br_x, br_y, br_z)

            glTexCoord2f_local(0.0, 0.0)
            glVertex3f_local(bl_x, bl_y, bl_z)

        glEnd()

    # restore state
    glDisable_local(GL_TEXTURE_2D)
    glDisable_local(GL_BLEND)
    glDepthMask_local(True)
