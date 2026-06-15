from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world.entity_registry import SceneEntityRegistry


class FakeScene:
    def __init__(self) -> None:
        self.entities = []
        self.immediate_entities = []
        self.goblins = []
        self.chests = []
        self.showcase_chests = []
        self.sprite_items = []
        self.wall_tiles = []
        self._sprite_update_cache = object()
        self.collision_invalidated = False

    def invalidate_collision_index(self) -> None:
        self.collision_invalidated = True


class FakeEntity:
    def __init__(self, *, render_batched: bool = False) -> None:
        self.enabled = True
        self.visible = True
        self.render_batched = render_batched
        self.sprites = [object()]
        self.meshes = [object()]
        self.killed = False

    def get_sprites(self):
        return self.sprites

    def get_collision_meshes(self):
        return self.meshes

    def kill(self) -> None:
        self.killed = True
        self.enabled = False
        self.visible = False


class EntityRegistryTests(unittest.TestCase):
    def test_add_registers_entity_resources_once(self) -> None:
        scene = FakeScene()
        registry = SceneEntityRegistry(scene)
        entity = FakeEntity()

        self.assertIs(registry.add(entity), entity)
        registry.add(entity)

        self.assertEqual(scene.entities, [entity])
        self.assertEqual(scene.immediate_entities, [entity])
        self.assertEqual(scene.sprite_items, entity.sprites)
        self.assertEqual(scene.wall_tiles, entity.meshes)

    def test_add_skips_immediate_list_for_batched_entity(self) -> None:
        scene = FakeScene()
        registry = SceneEntityRegistry(scene)
        entity = FakeEntity(render_batched=True)

        registry.add(entity)

        self.assertEqual(scene.entities, [entity])
        self.assertEqual(scene.immediate_entities, [])

    def test_remove_unregisters_entity_resources_and_invalidates_collision(
        self,
    ) -> None:
        scene = FakeScene()
        registry = SceneEntityRegistry(scene)
        entity = FakeEntity()
        registry.add(entity)
        scene.goblins.append(entity)
        scene.chests.append(entity)
        scene.showcase_chests.append(entity)

        registry.remove(entity)

        self.assertTrue(entity.killed)
        self.assertEqual(scene.entities, [])
        self.assertEqual(scene.immediate_entities, [])
        self.assertEqual(scene.goblins, [])
        self.assertEqual(scene.chests, [])
        self.assertEqual(scene.showcase_chests, [])
        self.assertEqual(scene.sprite_items, [])
        self.assertEqual(scene.wall_tiles, [])
        self.assertTrue(scene.collision_invalidated)
        self.assertIsNone(scene._sprite_update_cache)

    def test_refresh_immediate_filters_batched_entities(self) -> None:
        scene = FakeScene()
        visible = FakeEntity()
        batched = FakeEntity(render_batched=True)
        shadow_batched = FakeEntity()
        shadow_batched.shadow_render_batched = True
        scene.entities = [visible, batched, shadow_batched]

        SceneEntityRegistry(scene).refresh_immediate()

        self.assertEqual(scene.immediate_entities, [visible])


if __name__ == "__main__":
    unittest.main()
