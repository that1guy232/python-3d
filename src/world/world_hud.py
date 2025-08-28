"""World HUD: compass, held item (sword), and world shade overlay.

This isolates HUD setup, update, and draw responsibilities from WorldScene.
"""

from __future__ import annotations

import random
from pygame.math import Vector3
from world.sprite import WorldSprite
from ui.compass_overlay import CompassOverlay
from textures.texture_utils import load_texture
from textures.resoucepath import (
    COMPASS_BASE_TEXTURE_PATH,
    COMPASS_NEEDLE_TEXTURE_PATH,
    SWORD_TEXTURE_PATH,
    LIGHT_TEXTURE_PATH
)
from config import HEADBOB_ENABLED
from core.consts import *

class WorldHUD:
    def __init__(self, scene) -> None:
        self.scene = scene
        # World darkening overlay

        # Load compass textures (safe if missing)
        base_tex = load_texture(COMPASS_BASE_TEXTURE_PATH)
        needle_tex = load_texture(COMPASS_NEEDLE_TEXTURE_PATH)

        self._compass = CompassOverlay(
            position=Vector3(100, 100, 0),
            size=(1.5, 1.5),
            camera=self.scene.camera,
            base_texture=base_tex,
            needle_texture=needle_tex,
        )

        # Held item (sword) as a WorldSprite
        sword_tex = load_texture(SWORD_TEXTURE_PATH)
        sword_pos = Vector3(1, -1, -5)
        self.sword = WorldSprite(
            position=sword_pos,
            size=(4, 4),
            camera=self.scene.camera,
            texture=sword_tex,
        )

        self.test_light = WorldSprite(
            position=Vector3(0, 0, 0),
            size=(10, 10),
            camera=self.scene.camera,
            texture=load_texture(LIGHT_TEXTURE_PATH),
            color=(1.0, 1.0, 1.0),
        )
        # Visible only when looking down; initial false
        self._test_light_visible = True

    def update(self, dt: float) -> None:
        # Place sword and compass using scene's helper view_space_position
        self.sword.position = self.scene.view_space_position(
            dist=7.0, nx=0.65, ny=-0.65
        )
        self._compass.position = self.scene.view_space_position(
            dist=5.0, nx=-0.85, ny=0.85
        )

        # Apply headbob offsets if enabled
        if HEADBOB_ENABLED:
            try:
                off_x, off_y = self.scene._headbob.offsets()
            except Exception:
                off_x, off_y = 0.0, 0.0
            off_y = off_y * 0.85
            off_x = off_x * 0.75
            self.sword.position += Vector3(+off_x, +off_y, 0)
            self._compass.position += Vector3(+off_x * 1, +off_y * 1, 0)
       
        # Apply mouse-look sway from scene's sway controller (if present)
        try:
            sc = getattr(self.scene, "_sway_controller", None)
            if sc is not None:
                sway = sc.get_sway()
                right = self.scene.camera._right
                forward = self.scene.camera._forward
                up = getattr(self.scene.camera, "_up", right.cross(forward))

                sway_right = -sway.x * sc.right_mult
                sway_up = +sway.y * sc.up_mult
                sway_forward = -abs(sway.x) * sc.forward_mult

                self.sword.position += (
                    (right * sway_right) + (up * sway_up) + (forward * sway_forward)
                )
                self._compass.position += (
                    right * (sway_right * 0.6)
                    + up * (sway_up * 0.6)
                    + forward * (sway_forward * 0.6)
                )
        except Exception:
            # If camera axes or sway not present, skip sway
            pass
        
        cam_pos = self.scene.camera.position
        ground_y = self.scene.ground_height_at(cam_pos.x, cam_pos.z)
        cx = cam_pos.x
        cz = cam_pos.z
        elevation = 5
        self.test_light.position = Vector3(cx, ground_y + elevation, cz)

    def draw(self) -> None:
        # Draw held item and compass (world-space sprites with pitch effect)
        try:
            self.sword.draw(pitch_effect=True)
        except Exception:
            pass
        try:
            self._compass.draw(pitch_effect=True)
        except Exception:
            pass

        self.test_light.draw(pitch_effect=True)
