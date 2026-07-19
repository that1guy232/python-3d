"""Scene-owned render resource cleanup for WorldScene."""

from __future__ import annotations

from game.world import world_runtime
from game.world.world_state import WorldBuildState, WorldRenderResources, WorldUIState


class SceneResourceDisposer:
    """Dispose explicit world resource owners without requiring a scene."""

    SINGLE_RESOURCE_ATTRS = ("ground_mesh", "sky", "road", "decal_batch")
    COLLECTION_RESOURCE_ATTRS = (
        "fence_meshes",
        "wall_tile_batches",
        "road_batches",
        "decal_batches",
        "decals",
        "door_batches",
        "window_batches",
        "polygon_batches",
        "others",
        "entities",
    )
    CLEARED_COLLECTION_ATTRS = (
        "fence_meshes",
        "wall_tile_batches",
        "road_batches",
        "decal_batches",
        "decals",
        "door_batches",
        "window_batches",
        "polygon_batches",
        "others",
        "entities",
        "immediate_entities",
    )

    def __init__(
        self,
        resources: WorldRenderResources,
        ui_state: WorldUIState | None = None,
        build_state: WorldBuildState | None = None,
    ) -> None:
        owner = resources
        self.resources = getattr(owner, "render_resources", owner)
        self.ui_state = ui_state or getattr(owner, "ui_state", owner)
        self.build_state = build_state or getattr(owner, "build_state", owner)

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
        resources = self.resources
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
            dispose_once(getattr(resources, attr_name, None))

        dispose_once(
            getattr(self.ui_state, "hud", getattr(self.ui_state, "_hud", None))
        )

        for attr_name in self.COLLECTION_RESOURCE_ATTRS:
            for obj in getattr(resources, attr_name, ()) or ():
                dispose_once(obj)

        if self.build_state is not None:
            for attr_name in (
                "environment_volumes",
                "roads",
                "building_roads",
                "doors",
                "windows",
                "walls",
                "torches",
                "goblins",
                "chests",
                "showcase_chests",
                "showcase_polygons",
            ):
                for obj in getattr(self.build_state, attr_name, ()) or ():
                    dispose_once(obj)

        resources.ground_mesh = None
        resources.sky = None
        if hasattr(resources, "ground_height_sampler"):
            resources.ground_height_sampler = None
        else:
            resources._ground_height_sampler = None
        resources.road = None
        resources.decal_batch = None
        if hasattr(resources, "collision_spatial_index"):
            resources.collision_spatial_index = None
        else:
            resources._collision_spatial_index = None
        if hasattr(self.ui_state, "hud"):
            self.ui_state.hud = None
        else:
            self.ui_state._hud = None

        for attr_name in self.CLEARED_COLLECTION_ATTRS:
            values = getattr(resources, attr_name, None)
            if hasattr(values, "clear"):
                values.clear()
            else:
                setattr(resources, attr_name, [])

        if self.build_state is not None:
            for attr_name in (
                "roads",
                "building_roads",
                "doors",
                "windows",
                "walls",
                "torches",
                "goblins",
                "chests",
                "showcase_chests",
                "showcase_polygons",
            ):
                values = getattr(self.build_state, attr_name, None)
                if hasattr(values, "clear"):
                    values.clear()
