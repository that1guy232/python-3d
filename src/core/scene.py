from typing import List, Callable, Optional
from dataclasses import dataclass, field
from core.drawable import Drawable
from world.decal import Decal  # for decal-specific handling
from world.decal_batch import DecalBatch
from config import VIEWDISTANCE
from world.sprite import WorldSprite, draw_sprites_batched
from camera import Camera

# Type alias moved from engine.py during refactor
UpdateFn = Callable[[float], None]


@dataclass
class Scene:
    # Camera is optional so non-3D scenes (e.g., main menu) don't need one
    camera: Optional[Camera] = None
    static_meshes: List[Drawable] = field(default_factory=list)
    updaters: List[UpdateFn] = field(default_factory=list)

    def update(self, dt: float):
        for fn in self.updaters:
            fn(dt)

    # Optional per-event handler (scenes can override)
    def handle_event(self, event) -> None:
        pass

    # Default 3D draw helper for subclasses that use camera
    def draw(self):  # pragma: no cover - visual
        # Batch sprites by texture, draw others individually
        sprite_items: list[WorldSprite] = []
        decals: list[Decal] = []
        decal_batches: list[DecalBatch] = []
        others: list[Drawable] = []
        for mesh in self.static_meshes:
            if isinstance(mesh, WorldSprite):
                sprite_items.append(mesh)
            elif isinstance(mesh, Decal):
                decals.append(mesh)
            elif isinstance(mesh, DecalBatch):
                decal_batches.append(mesh)
            else:
                others.append(mesh)

        # Basic distance culling for decals (static, inexpensive)
        if decals and self.camera is not None:
            cam_pos = self.camera.position
            near_decals = [
                d for d in decals if (d.center - cam_pos).length() <= VIEWDISTANCE
            ]
        else:
            near_decals = decals

        # Draw non-sprite, non-decal first
        for m in others:
            if getattr(m, "texture", None):
                m.draw()
            else:
                m.draw_untextured()

        # Draw decals (already distance-culled)
        for d in near_decals:
            d.draw()

        # Draw any decal batches that may have been added (single VBO per texture)
        for batch in decal_batches:
            batch.draw()

        # Draw sprites batched with alpha blending
        if sprite_items and self.camera is not None:
            dist_culled = [
                s
                for s in sprite_items
                if (s.position - self.camera.position).length() <= VIEWDISTANCE
            ]
            # Provide a ground height sampler so sprite shadows can conform to terrain
            height_fn = getattr(self, "ground_height_at", None)
            draw_sprites_batched(dist_culled, self.camera, height_fn)

    # Scenes can own their full render pipeline (projection, modelview, etc.)
    def render(self):  # pragma: no cover - visual
        # By default, do nothing; 3D scenes should override
        pass
