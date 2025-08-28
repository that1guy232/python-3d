from __future__ import annotations
from typing import Tuple
import math
from world.sprite import WorldSprite
from OpenGL.GL import (
    glEnable,
    glDisable,
    glBindTexture,
    glBegin,
    glEnd,
    glBlendFunc,
    glDepthMask,
    glTexCoord2f,
    glVertex3f,
    glColor4f,
    GL_TEXTURE_2D,
    GL_QUADS,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_DEPTH_TEST,
)


from core.consts import *  # FORWARD, RIGHT, WORLD_UP

class CompassOverlay(WorldSprite):
    def __init__(
        self,
        position,
        size: Tuple[float, float],
        camera,
        base_texture: int,
        needle_texture: int,
    ):
        self.position = position
        self.size = size
        self.camera = camera
        self.base_texture = base_texture
        self.needle_texture = needle_texture
        super().__init__(
            position=position, size=size, camera=camera, texture=base_texture
        )



    def draw(self, pitch_effect: bool = False):  # pragma: no cover - visual


        axes = self._billboard_axes(self.camera, pitch_effect=pitch_effect)
        right, up = axes if axes else (RIGHT, WORLD_UP)

        w, h = self.size
        hw, hh = w * 0.5, h * 0.5
        cx, cy, cz = self.position.x, self.position.y, self.position.z

        def emit_quad(tex_id, r_vec, u_vec, center=None):
            if not tex_id:
                return
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glColor4f(1.0, 1.0, 1.0, 1.0)

            rx_hw = r_vec.x * hw
            ry_hw = r_vec.y * hw
            rz_hw = r_vec.z * hw

            ux_hh = u_vec.x * hh
            uy_hh = u_vec.y * hh
            uz_hh = u_vec.z * hh

            # choose center for this quad (optionally offset)
            if center is None:
                ccx, ccy, ccz = cx, cy, cz
            else:
                ccx, ccy, ccz = center.x, center.y, center.z

            # Top-left
            tl_x = ccx - rx_hw + ux_hh
            tl_y = ccy - ry_hw + uy_hh
            tl_z = ccz - rz_hw + uz_hh

            # Top-right
            tr_x = ccx + rx_hw + ux_hh
            tr_y = ccy + ry_hw + uy_hh
            tr_z = ccz + rz_hw + uz_hh

            # Bottom-right
            br_x = ccx + rx_hw - ux_hh
            br_y = ccy + ry_hw - uy_hh
            br_z = ccz + rz_hw - uz_hh

            # Bottom-left
            bl_x = ccx - rx_hw - ux_hh
            bl_y = ccy - ry_hw - uy_hh
            bl_z = ccz - rz_hw - uz_hh

            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(tl_x, tl_y, tl_z)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(tr_x, tr_y, tr_z)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(br_x, br_y, br_z)
            glTexCoord2f(0.0, 0.0)
            glVertex3f(bl_x, bl_y, bl_z)
            glEnd()

        # Draw base (no rotation within plane). Disable depth to avoid z-fighting and unwanted occlusion.
        glDisable(GL_DEPTH_TEST)
        glDepthMask(False)
        emit_quad(self.base_texture, right, up)

        # Rotate needle within the billboard plane based on camera yaw so it points to world north
        yaw_deg = math.degrees(float(getattr(self.camera, "rotation").y))
        angle_rad = math.radians(yaw_deg - 90.0)  # match original overlay orientation
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        # 2D rotation of basis in the plane: r' = r*cos - u*sin, u' = r*sin + u*cos
        r2 = type(right)(
            right.x * cos_a - up.x * sin_a,
            right.y * cos_a - up.y * sin_a,
            right.z * cos_a - up.z * sin_a,
        )
        u2 = type(up)(
            right.x * sin_a + up.x * cos_a,
            right.y * sin_a + up.y * cos_a,
            right.z * sin_a + up.z * cos_a,
        )

        # Draw needle second; depth remains disabled so it cleanly overlays the base without z-fighting
        emit_quad(self.needle_texture, r2, u2)

        # Restore depth state
        glDepthMask(True)
        glEnable(GL_DEPTH_TEST)

        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
