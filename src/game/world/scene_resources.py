"""Scene-owned render resource cleanup for WorldScene."""

from __future__ import annotations

from game.world import world_runtime


class SceneResourceDisposer:
    """Dispose scene render resources once and clear scene references."""

    SINGLE_RESOURCE_ATTRS = (
        "ground_mesh",
        "road",
        "decal_batch",
        "_hud",
    )
    COLLECTION_RESOURCE_ATTRS = (
        "fence_meshes",
        "wall_tile_batches",
        "road_batches",
        "decal_batches",
        "decals",
        "roads",
        "building_roads",
        "door_batches",
        "window_batches",
        "polygon_batches",
        "showcase_chests",
        "chests",
        "others",
        "entities",
    )
    CLEARED_COLLECTION_ATTRS = (
        "fence_meshes",
        "wall_tile_batches",
        "road_batches",
        "decal_batches",
        "decals",
        "roads",
        "building_roads",
        "door_batches",
        "window_batches",
        "polygon_batches",
        "showcase_chests",
        "chests",
        "others",
        "entities",
        "immediate_entities",
    )

    def __init__(self, scene) -> None:
        self.scene = scene

    @staticmethod
    def dispose_renderable(obj) -> None:
        dispose = getattr(obj, "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                pass

    @classmethod
    def dispose_renderable_batches(cls, values) -> None:
        for value in values or ():
            cls.dispose_renderable(value)

    def dispose(self) -> None:
        """Release scene-owned VBOs before the OpenGL context is destroyed."""
        scene = self.scene
        world_runtime.stop_ambient_birds()
        disposed: set[int] = set()

        def dispose_once(obj) -> None:
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in disposed:
                return
            disposed.add(obj_id)
            self.dispose_renderable(obj)

        for attr_name in self.SINGLE_RESOURCE_ATTRS:
            dispose_once(getattr(scene, attr_name, None))

        for attr_name in self.COLLECTION_RESOURCE_ATTRS:
            for obj in getattr(scene, attr_name, ()) or ():
                dispose_once(obj)

        scene.ground_mesh = None
        scene._ground_height_sampler = None
        scene.road = None
        scene.decal_batch = None
        scene._collision_spatial_index = None

        for attr_name in self.CLEARED_COLLECTION_ATTRS:
            setattr(scene, attr_name, [])
