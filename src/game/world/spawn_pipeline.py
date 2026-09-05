"""Procedural-tree, sprite, and enemy spawning for world scenes."""

from __future__ import annotations


import math
import time
import random


from pygame.math import Vector3


from engine.textures.texture_utils import create_shadow_texture

from game.config import (
    GOBLIN_CHASE_GIVE_UP_RADIUS,
    GOBLIN_CHASE_RADIUS,
    GOBLIN_COUNT,
    GOBLIN_MIN_SEPARATION,
    GOBLIN_SPAWN_ATTEMPTS,
    GOBLIN_SPAWN_CLEARANCE,
    GOBLIN_SPAWN_TREE_RADIUS,
)

from game.actors.goblin import Goblin, GoblinRenderBatch

from game.world.builder_support import _dispose_values
from game.world.objects.grass import build_grass_meshes
from game.world.objects.rock import build_rock_meshes
from game.world.objects.tree import build_tree_meshes
from game.world.road_pipeline import _build_roads

from game.world.world_spawner import (
    spawn_world_grasses,
    spawn_world_rocks,
    spawn_world_trees,
)


def _build_roads_and_spawn_sprites(
    scene, tree_count: int, grass_count: int, rock_count: int
) -> None:

    _build_roads(scene)

    _spawn_trees(scene, tree_count)

    _build_goblins(scene)

    _spawn_grass(scene, grass_count)

    _spawn_rocks(scene, rock_count)


def _spawn_trees(scene, tree_count: int) -> None:
    start_time = time.perf_counter()
    resources = getattr(scene, "render_resources", scene)
    build_state = getattr(scene, "build_state", scene)

    _remove_previous_sprites(resources, getattr(build_state, "trees", ()))

    _dispose_values(getattr(resources, "tree_meshes", ()))
    resources.tree_meshes = []

    trees = spawn_world_trees(
        scene,
        count=tree_count,
        x_off=(scene.ground_bounds[0] + scene.ground_bounds[1]) / 2 + 35,
        z_off=(scene.ground_bounds[2] + scene.ground_bounds[3]) / 2 + 35,
        max_spawn_x=(scene.ground_bounds[1] - scene.ground_bounds[0]) / 2 - 35,
        max_spawn_z=(scene.ground_bounds[3] - scene.ground_bounds[2]) / 2 - 35,
        avoid_roads=scene.roads,
        avoid_areas=scene.buildings,
        random_seed=getattr(scene, "world_random_seed", None),
    )
    build_state.trees = trees
    resources.tree_meshes = build_tree_meshes(
        trees,
        exposure_baseline=float(
            getattr(getattr(scene, "camera", None), "brightness_default", 1.0)
        ),
    )

    print(
        f"Generated {len(trees)} procedural trees in "
        f"{len(resources.tree_meshes)} spatial mesh batches."
    )
    scene.log_timing("Spawn trees", start_time, time.perf_counter())


def _spawn_grass(scene, grass_count: int) -> None:
    start_time = time.perf_counter()
    resources = getattr(scene, "render_resources", scene)
    build_state = getattr(scene, "build_state", scene)
    _remove_previous_sprites(resources, getattr(build_state, "grasses", ()))
    _dispose_values(getattr(resources, "grass_meshes", ()))
    resources.grass_meshes = []

    grasses = spawn_world_grasses(
        scene,
        count=grass_count,
        x_off=(scene.ground_bounds[0] + scene.ground_bounds[1]) / 2,
        z_off=(scene.ground_bounds[2] + scene.ground_bounds[3]) / 2,
        max_spawn_x=(scene.ground_bounds[1] - scene.ground_bounds[0]) / 2,
        max_spawn_z=(scene.ground_bounds[3] - scene.ground_bounds[2]) / 2,
        avoid_roads=scene.roads,
        avoid_areas=scene.buildings,
        random_seed=_nature_layer_seed(scene, 0x47524153),
    )
    build_state.grasses = grasses
    resources.grass_meshes = build_grass_meshes(
        grasses,
        exposure_baseline=float(
            getattr(getattr(scene, "camera", None), "brightness_default", 1.0)
        ),
    )
    print(
        f"Generated {len(grasses)} procedural grass clumps in "
        f"{len(resources.grass_meshes)} spatial mesh batches."
    )
    scene.log_timing("Spawn grass", start_time, time.perf_counter())


def _spawn_rocks(scene, rock_count: int) -> None:
    start_time = time.perf_counter()
    resources = getattr(scene, "render_resources", scene)
    build_state = getattr(scene, "build_state", scene)
    _remove_previous_sprites(resources, getattr(build_state, "rocks", ()))
    _dispose_values(getattr(resources, "rock_meshes", ()))
    resources.rock_meshes = []

    rocks = spawn_world_rocks(
        scene,
        count=rock_count,
        x_off=(scene.ground_bounds[0] + scene.ground_bounds[1]) / 2,
        z_off=(scene.ground_bounds[2] + scene.ground_bounds[3]) / 2,
        max_spawn_x=(scene.ground_bounds[1] - scene.ground_bounds[0]) / 2,
        max_spawn_z=(scene.ground_bounds[3] - scene.ground_bounds[2]) / 2,
        avoid_roads=scene.roads,
        avoid_areas=scene.buildings,
        random_seed=_nature_layer_seed(scene, 0x524F434B),
    )
    build_state.rocks = rocks
    resources.rock_meshes = build_rock_meshes(
        rocks,
        exposure_baseline=float(
            getattr(getattr(scene, "camera", None), "brightness_default", 1.0)
        ),
    )
    print(
        f"Generated {len(rocks)} procedural rocks in "
        f"{len(resources.rock_meshes)} spatial mesh batches."
    )
    scene.log_timing("Spawn rocks", start_time, time.perf_counter())


def _nature_layer_seed(scene, salt: int) -> int | None:
    seed = getattr(scene, "world_random_seed", None)
    if seed is None:
        return None
    return (int(seed) ^ int(salt)) & ((1 << 63) - 1)


def _remove_previous_sprites(resources, values) -> None:
    previous_ids = {id(value) for value in (values or ())}
    if not previous_ids:
        return
    resources.sprite_items = [
        item for item in resources.sprite_items if id(item) not in previous_ids
    ]


def _goblin_position_blocker(scene, *, block_roads: bool = True):

    min_x, max_x, min_z, max_z = scene.ground_bounds

    buildings = list(getattr(scene, "buildings", ()) or ())

    def blocked(x: float, z: float, margin: float = 0.0) -> bool:

        clearance = max(0.0, float(margin))

        if (
            x < min_x + clearance
            or x > max_x - clearance
            or z < min_z + clearance
            or z > max_z - clearance
        ):

            return True

        if block_roads and scene.is_on_road(x, z, margin=clearance):

            return True

        for building in buildings:

            contains_point = getattr(building, "contains_point", None)

            if not callable(contains_point):

                continue

            if contains_point(x, z, margin=clearance):

                return True

        return False

    return blocked


def _random_goblin_spawn_near_tree(scene, rng: random.Random):

    trees = list(getattr(scene, "trees", ()) or ())

    min_x, max_x, min_z, max_z = scene.ground_bounds

    if not trees:

        x = rng.uniform(min_x, max_x)

        z = rng.uniform(min_z, max_z)

        return x, z

    anchor = rng.choice(trees).position

    angle = rng.uniform(0.0, math.tau)

    radius = max(0.0, float(GOBLIN_SPAWN_TREE_RADIUS)) * math.sqrt(rng.random())

    return (
        float(anchor.x) + math.cos(angle) * radius,
        float(anchor.z) + math.sin(angle) * radius,
    )


def _player_in_building_detector(scene):

    def player_in_building() -> bool:

        camera = getattr(scene, "camera", None)

        position = getattr(camera, "position", None)

        if position is None:

            return False

        x = float(position.x)

        z = float(position.z)

        for building in getattr(scene, "buildings", ()) or ():

            contains_point = getattr(building, "contains_point", None)

            if callable(contains_point) and contains_point(x, z, margin=0.0):

                return True

        return False

    return player_in_building


def _build_goblins(scene) -> None:

    resources = getattr(scene, "render_resources", scene)
    previous_batch = getattr(resources, "goblin_batch", None)
    dispose_batch = getattr(previous_batch, "dispose", None)
    if callable(dispose_batch):
        dispose_batch()
    resources.goblin_batch = None

    for goblin in getattr(scene, "goblins", ()) or ():

        try:
            remove_entity = getattr(scene, "remove_entity", None)
            if callable(remove_entity):
                remove_entity(goblin)
            else:
                if goblin in scene.entities:
                    scene.entities.remove(goblin)
                if goblin in getattr(scene, "creatures", ()):
                    scene.creatures.remove(goblin)
                for sprite in goblin.get_sprites() or ():
                    if sprite in scene.sprite_items:
                        scene.sprite_items.remove(sprite)

        except Exception:

            continue

    goblin_material_tex = getattr(scene, "goblin_material_tex", None)

    scene.goblins = []

    count = max(0, int(getattr(scene, "goblin_count", GOBLIN_COUNT)))

    if count <= 0:

        print("Spawned 0 goblins.")

        return

    shadow_texture = create_shadow_texture(
        width_px=96,
        height_px=96,
        max_alpha=0.24,
        inner_ratio=0.14,
        outer_ratio=0.92,
        falloff_exp=1.8,
        pixelated=False,
    )

    rng = random.Random(getattr(scene, "world_random_seed", None))

    spawn_blocked = _goblin_position_blocker(scene, block_roads=True)

    movement_blocked = _goblin_position_blocker(scene, block_roads=False)

    clearance = max(
        float(GOBLIN_SPAWN_CLEARANCE),
        float(getattr(Goblin, "DEFAULT_HEIGHT", 0.0)) * 0.3,
    )

    min_separation_sq = max(0.0, float(GOBLIN_MIN_SEPARATION)) ** 2

    max_attempts = max(1, int(GOBLIN_SPAWN_ATTEMPTS))

    add_entity = getattr(scene, "add_entity", None)

    player_in_building = _player_in_building_detector(scene)

    chase_radius = float(getattr(scene, "goblin_chase_radius", GOBLIN_CHASE_RADIUS))

    chase_give_up_radius = float(
        getattr(scene, "goblin_chase_give_up_radius", GOBLIN_CHASE_GIVE_UP_RADIUS)
    )

    for _ in range(count):

        spawn = None

        for _attempt in range(max_attempts):

            x, z = _random_goblin_spawn_near_tree(scene, rng)

            if spawn_blocked(x, z, clearance):

                continue

            too_close = False

            for other in scene.goblins:

                dx = other.spawn_position.x - x

                dz = other.spawn_position.z - z

                if dx * dx + dz * dz < min_separation_sq:

                    too_close = True

                    break

            if too_close:

                continue

            spawn = Vector3(x, scene.ground_height_at(x, z), z)

            break

        if spawn is None:

            continue

        try:

            goblin = Goblin(
                spawn,
                material_texture=goblin_material_tex,
                camera=scene.camera,
                ground_height_at=scene.ground_height_at,
                position_blocked=movement_blocked,
                player_in_building=player_in_building,
                chase_radius=chase_radius,
                chase_give_up_radius=chase_give_up_radius,
                shadow_texture=shadow_texture,
                sound_keys=getattr(scene, "goblin_sound_keys", None),
                rng=random.Random(rng.randrange(1 << 30)),
            )

        except (TypeError, ValueError, AttributeError):

            continue

        scene.goblins.append(goblin)

        if callable(add_entity):

            add_entity(goblin)

        else:

            scene.entities.append(goblin)

            if goblin not in scene.creatures:
                scene.creatures.append(goblin)

            scene.sprite_items.extend(goblin.get_sprites())

    resources.goblin_batch = GoblinRenderBatch(
        scene.goblins,
        material_texture=goblin_material_tex,
        exposure_baseline=float(
            getattr(getattr(scene, "camera", None), "brightness_default", 1.0)
        ),
    )

    print(f"Generated {len(scene.goblins)} procedural 3D goblins.")

    refresh_draw_entities = getattr(scene, "refresh_immediate_entities", None)

    if callable(refresh_draw_entities):

        refresh_draw_entities()

