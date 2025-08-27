from typing import List, Callable, Optional
from dataclasses import dataclass, field
from config import VIEWDISTANCE

from world.sprite import WorldSprite, draw_sprites_batched
from world.decal import Decal  # for decal-specific handling
from world.decal_batch import DecalBatch
from world.objects import WallTile
from world.objects.polygon import Polygon  # for polygon-specific handling
from world.ground_tile import GroundTile

# Type alias moved from engine.py during refactor
UpdateFn = Callable[[float], None]


@dataclass
class Scene:
    # Camera is optional so non-3D scenes (e.g., main menu) don't need one
    # Use a generic object type to avoid importing `camera` at module import time.
    camera: Optional[object] = None
    static_meshes: List[object] = field(default_factory=list)
    updaters: List[UpdateFn] = field(default_factory=list)
    # Optional screen-space night shade overlay; owned by base Scene so all
    # scenes can use it without duplicating initialization.
    # Use a lazy import inside the default_factory to avoid a circular import
    # when `world` package imports `core.scene` during module initialization.


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
        wall_tiles: list[WallTile] = []
        polygons: list[Polygon] = []
        others: list[object] = []

        import time
        total_start_time = time.perf_counter()


        sort_meshes_time = time.perf_counter()
        for mesh in self.static_meshes:
            if isinstance(mesh, WorldSprite):
                sprite_items.append(mesh)
            elif isinstance(mesh, Decal):
                decals.append(mesh)
            elif isinstance(mesh, DecalBatch):
                decal_batches.append(mesh)
            elif isinstance(mesh, WallTile):
                wall_tiles.append(mesh)
            elif isinstance(mesh, Polygon):
                polygons.append(mesh)
            elif isinstance(mesh, GroundTile):
                print("Found GroundTile")
            else:
                others.append(mesh)
        end_sort_meshes_time = time.perf_counter()
        print(f"Sorting meshes took {end_sort_meshes_time - sort_meshes_time:.6f} seconds")

        # Draw any decal batches that may have been added (single VBO per texture)
        start_draw_decal_batches_time = time.perf_counter()
        for batch in decal_batches:
            batch.draw(camera=self.camera)
        end_draw_decal_batches_time = time.perf_counter()
        print(f"Drawing decal batches took {end_draw_decal_batches_time - start_draw_decal_batches_time:.6f} seconds")

        # Draw non-sprite, non-decal first
        def _approx_pos(obj):
            """Try to approximate a world-space position for common drawable types.

            Returns (x, y, z) or None if unknown.
            """
            # Prefer common attributes
            p = getattr(obj, "position", None)
            if p is not None:
                try:
                    return (float(p.x), float(getattr(p, "y", 0.0)), float(p.z))
                except Exception:
                    pass

            c = getattr(obj, "center", None)
            if c is not None:
                try:
                    return (float(c.x), float(getattr(c, "y", 0.0)), float(c.z))
                except Exception:
                    pass

            # Buildings and other containers may expose a bbox
            if hasattr(obj, "get_bounding_box"):
                try:
                    bbox = obj.get_bounding_box()
                    if bbox:
                        min_x, max_x, min_z, max_z = bbox
                        # Compute closest point on the XZ rectangle to the camera
                        cam_x = float(getattr(self.camera.position, "x", 0.0)) if self.camera else 0.0
                        cam_z = float(getattr(self.camera.position, "z", 0.0)) if self.camera else 0.0
                        # Clamp camera X,Z into bbox to get nearest point on box
                        px = max(min_x, min(max_x, cam_x))
                        pz = max(min_z, min(max_z, cam_z))
                        cy = float(getattr(self.camera.position, "y", 0.0)) if self.camera else 0.0
                        return (px, cy, pz)
                except Exception:
                    pass

            # Road-like objects have start/end
            if hasattr(obj, "start") and hasattr(obj, "end"):
                try:
                    s = obj.start
                    e = obj.end
                    cx = (float(s.x) + float(e.x)) * 0.5
                    cz = (float(s.z) + float(e.z)) * 0.5
                    cy = float(getattr(obj, "ground_y", getattr(self.camera.position, "y", 0.0) if self.camera else 0.0))
                    return (cx, cy, cz)
                except Exception:
                    pass

            # Unknown
            return None

        starting_draw_other_time = time.perf_counter()
        cam_pos = self.camera.position if self.camera is not None else None
        vd_sq = VIEWDISTANCE * VIEWDISTANCE
        for m in others:
            # If there's a camera and we can approximate a position, distance-cull
            if cam_pos is not None:
                pos = _approx_pos(m)
                if pos is not None:
                    try:
                        dx = pos[0] - cam_pos.x
                        dy = pos[1] - cam_pos.y
                        dz = pos[2] - cam_pos.z
                        if (dx * dx + dy * dy + dz * dz) > vd_sq:
                            # skip drawing distant object
                            continue
                    except Exception:
                        # if any error, fall back to drawing
                        pass

            m.draw()
        end_draw_other_time = time.perf_counter()
        print(f"Drawing other objects took {end_draw_other_time - starting_draw_other_time:.6f} seconds")

        start_draw_wall_tiles_time = time.perf_counter()
        for w in wall_tiles:
            w.draw(camera=self.camera)
        end_draw_wall_tiles_time = time.perf_counter()
        print(f"Drawing wall tiles took {end_draw_wall_tiles_time - start_draw_wall_tiles_time:.6f} seconds")


        start_draw_polygons_time = time.perf_counter()
        for p in polygons:
            p.draw(camera=self.camera)
        end_draw_polygons_time = time.perf_counter()
        print(f"Drawing polygons took {end_draw_polygons_time - start_draw_polygons_time:.6f} seconds")


        start_draw_sprites_time = time.perf_counter()
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
        
        print(f"Drawing sprites took {time.perf_counter() - start_draw_sprites_time:.6f} seconds")


   
    # Scenes can own their full render pipeline (projection, modelview, etc.)
    def render(self):  # pragma: no cover - visual
        # By default, do nothing; 3D scenes should override
        pass
