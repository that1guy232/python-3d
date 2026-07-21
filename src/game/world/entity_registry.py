"""Runtime entity registration for world scenes."""

from __future__ import annotations

from collections.abc import Callable

from engine.entity import Entity
from game.world.world_state import WorldBuildState, WorldRenderResources


class SceneEntityRegistry:
    """Keep scene entity lists and scene-facing resources in sync."""

    def __init__(
        self,
        resources: WorldRenderResources,
        build_state: WorldBuildState | None = None,
        *,
        invalidate_collision_index: Callable[[], None] | None = None,
    ) -> None:
        source = resources
        self.resources = getattr(resources, "render_resources", resources)
        self.build_state = build_state or getattr(resources, "build_state", resources)
        self.invalidate_collision_index = invalidate_collision_index or getattr(
            source, "invalidate_collision_index", lambda: None
        )

    def add(self, entity: Entity) -> Entity:
        resources = self.resources
        if entity not in resources.entities:
            resources.entities.append(entity)
        if (
            entity not in resources.immediate_entities
            and not getattr(entity, "door_render_batched", False)
            and not getattr(entity, "render_batched", False)
            and not getattr(entity, "shadow_render_batched", False)
        ):
            resources.immediate_entities.append(entity)

        creatures = getattr(self.build_state, "creatures", None)
        if (
            creatures is not None
            and getattr(entity, "combat_enabled", False)
            and entity not in creatures
        ):
            creatures.append(entity)

        get_sprites = getattr(entity, "get_sprites", None)
        if callable(get_sprites):
            for sprite in get_sprites() or ():
                if sprite not in resources.sprite_items:
                    resources.sprite_items.append(sprite)

        get_collision_meshes = getattr(entity, "get_collision_meshes", None)
        if callable(get_collision_meshes):
            for mesh in get_collision_meshes() or ():
                if mesh not in resources.wall_tiles:
                    resources.wall_tiles.append(mesh)

        return entity

    def remove(self, entity: Entity) -> None:
        if entity is None:
            return

        kill = getattr(entity, "kill", None)
        if callable(kill):
            kill()
        else:
            setattr(entity, "enabled", False)
            setattr(entity, "visible", False)

        def without_item(values):
            return [value for value in (values or []) if value is not entity]

        resources = self.resources
        resources.entities = without_item(resources.entities)
        resources.immediate_entities = without_item(resources.immediate_entities)
        build_state = self.build_state
        for attr_name in ("creatures", "goblins", "chests", "showcase_chests"):
            if hasattr(build_state, attr_name):
                setattr(
                    build_state,
                    attr_name,
                    without_item(getattr(build_state, attr_name)),
                )

        get_sprites = getattr(entity, "get_sprites", None)
        if callable(get_sprites):
            sprites = tuple(get_sprites() or ())
            if sprites:
                sprite_ids = {id(sprite) for sprite in sprites}
                resources.sprite_items = [
                    sprite
                    for sprite in resources.sprite_items
                    if id(sprite) not in sprite_ids
                ]

        get_collision_meshes = getattr(entity, "get_collision_meshes", None)
        if callable(get_collision_meshes):
            meshes = tuple(get_collision_meshes() or ())
            if meshes:
                mesh_ids = {id(mesh) for mesh in meshes}
                resources.wall_tiles = [
                    mesh for mesh in resources.wall_tiles if id(mesh) not in mesh_ids
                ]
                self.invalidate_collision_index()

        if hasattr(resources, "sprite_update_cache"):
            resources.sprite_update_cache = None
        else:
            resources._sprite_update_cache = None

    def refresh_immediate(self) -> None:
        resources = self.resources
        resources.immediate_entities = [
            entity
            for entity in resources.entities
            if not getattr(entity, "door_render_batched", False)
            and not getattr(entity, "render_batched", False)
            and not getattr(entity, "shadow_render_batched", False)
        ]
