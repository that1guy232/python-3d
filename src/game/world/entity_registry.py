"""Runtime entity registration for world scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from engine.entity import Entity

if TYPE_CHECKING:
    from game.world.worldscene import WorldScene


class SceneEntityRegistry:
    """Keep scene entity lists and scene-facing resources in sync."""

    def __init__(self, scene: WorldScene) -> None:
        self.scene = scene

    def add(self, entity: Entity) -> Entity:
        scene = self.scene
        if entity not in scene.entities:
            scene.entities.append(entity)
        if (
            entity not in scene.immediate_entities
            and not getattr(entity, "door_render_batched", False)
            and not getattr(entity, "render_batched", False)
            and not getattr(entity, "shadow_render_batched", False)
        ):
            scene.immediate_entities.append(entity)

        get_sprites = getattr(entity, "get_sprites", None)
        if callable(get_sprites):
            for sprite in get_sprites() or ():
                if sprite not in scene.sprite_items:
                    scene.sprite_items.append(sprite)

        get_collision_meshes = getattr(entity, "get_collision_meshes", None)
        if callable(get_collision_meshes):
            for mesh in get_collision_meshes() or ():
                if mesh not in scene.wall_tiles:
                    scene.wall_tiles.append(mesh)

        return entity

    def remove(self, entity: Entity) -> None:
        scene = self.scene
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

        for attr_name in (
            "entities",
            "immediate_entities",
            "goblins",
            "chests",
            "showcase_chests",
        ):
            if hasattr(scene, attr_name):
                setattr(scene, attr_name, without_item(getattr(scene, attr_name)))

        get_sprites = getattr(entity, "get_sprites", None)
        if callable(get_sprites):
            sprites = tuple(get_sprites() or ())
            if sprites:
                sprite_ids = {id(sprite) for sprite in sprites}
                scene.sprite_items = [
                    sprite
                    for sprite in scene.sprite_items
                    if id(sprite) not in sprite_ids
                ]

        get_collision_meshes = getattr(entity, "get_collision_meshes", None)
        if callable(get_collision_meshes):
            meshes = tuple(get_collision_meshes() or ())
            if meshes:
                mesh_ids = {id(mesh) for mesh in meshes}
                scene.wall_tiles = [
                    mesh for mesh in scene.wall_tiles if id(mesh) not in mesh_ids
                ]
                scene.invalidate_collision_index()

        scene._sprite_update_cache = None

    def refresh_immediate(self) -> None:
        scene = self.scene
        scene.immediate_entities = [
            entity
            for entity in scene.entities
            if not getattr(entity, "door_render_batched", False)
            and not getattr(entity, "render_batched", False)
            and not getattr(entity, "shadow_render_batched", False)
        ]
