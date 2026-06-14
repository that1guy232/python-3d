# Architecture Review

Date: 2026-06-14

## Findings

### High: Engine event failures are swallowed silently

Evidence:

- `Engine.handle_events()` catches every exception from `scene.handle_event(event)` and discards it at `src/engine/core/engine.py:120`.
- `Engine._dispose_scene()` also catches every disposal exception and discards it at `src/engine/core/engine.py:85`.
- World update exceptions are not swallowed at the engine boundary because `Engine.update()` calls `self.scene.update(dt)` directly at `src/engine/core/engine.py:133`.

Why it matters:

Input bugs, scene transition bugs, and cleanup failures can disappear during normal play. That makes regressions harder to reproduce and encourages defensive broad `try/except` blocks deeper in the code. It also means two scene hooks have different reliability semantics: update crashes loudly, event handling does not.

Recommendation:

Add a central exception policy. At minimum, log event/dispose exceptions with scene type and event type. Prefer a debug mode that re-raises exceptions, with release mode logging and continuing only where explicitly safe.

Status:

Addressed in `src/engine/core/engine.py` and `src/engine/config.py`. Scene `handle_event` and `dispose` exceptions now log scene/hook context plus traceback, and `PY3D_RERAISE_SCENE_EXCEPTIONS=1` enables debug re-raise behavior.

### High: World subsystems depend on an implicit shared-state contract

Evidence:

- `WorldScene.__init__()` initializes many public mutable lists and flags at `src/game/world/worldscene.py:58`.
- `world_builder.py` mutates scene attributes throughout construction, including roads, batches, doors, windows, chests, goblins, sprites, lighting, and collision invalidation. It has 1,290 lines, 34 functions, and 26 imports.
- `world_runtime.py` is a procedural subsystem operating on arbitrary scene attributes. It has 720 lines after the collision-index extraction.
- `world_renderer.py` reads a broad scene surface via `getattr` throughout rendering and UI handling. It has 594 lines after the UI and object-draw extractions.
- Recent extraction improved this locally: `BattleController` is now at `src/game/world/combat.py:14`, `SceneCollisionIndex` is now at `src/game/world/collision_index.py:12`, `SceneEntityRegistry` is now at `src/game/world/entity_registry.py:13`, and `StaticLightingController` is now at `src/game/world/lighting_controller.py:10`.

Why it matters:

The architecture currently has modules, but not strong boundaries. Most world modules communicate through a large mutable `WorldScene` object rather than typed subsystem state. That keeps call sites short, but makes ordering, ownership, and invariants hard to prove. For example, renderer code assumes `scene.sky`, `scene.ground_mesh`, `scene._headbob`, and many UI flags exist after setup, but those requirements are not represented by an explicit state object.

Recommendation:

Continue the extraction pattern used for combat and entity registration. Create focused state owners for construction output, lighting/exposure, collision queries, and UI/menu state. Use those owners as constructor dependencies for renderer/runtime helpers rather than letting every helper reach across the whole scene.

Status:

Started with `src/game/world/collision_index.py`. Collision source keying, spatial cell rebuilds, dynamic/fallback mesh handling, candidate lookup, and invalidation now live in `SceneCollisionIndex` instead of `world_runtime.py`. `WorldScene` owns the collision index instance and keeps compatibility delegate methods, while runtime movement asks the scene for collision candidates. `tests/test_collision_index.py` covers static, dynamic, fallback, polygon-filtered, rebuild, and invalidation behavior.

### Medium: `WorldScene` still owns rendering internals and lighting rebuild policy

Evidence:

- Previously, `WorldScene.draw_world_objects()` contained object position estimation, render sphere estimation, sorting, culling, batched drawing, and fallback drawing. It now delegates at `src/game/world/worldscene.py:325`, with that behavior moved to `WorldRenderer.draw_world_objects()`.
- Previously, `WorldScene.refresh_static_lighting()` rebuilt ground, roads, walls, fences, batches, and shader state. It now delegates at `src/game/world/worldscene.py:441`, with that behavior moved to `StaticLightingController.refresh_static()`.
- Previously, `WorldScene._sync_lighting_uniforms()` owned shader synchronization cache keys. It now delegates at `src/game/world/worldscene.py:447`, with the cache policy moved to `StaticLightingController.sync_uniforms()`.
- Previously, `WorldScene.dispose()` had detailed knowledge of renderable ownership across nearly every world subsystem. It now delegates at `src/game/world/worldscene.py:546`, with resource cleanup moved to `SceneResourceDisposer.dispose()`.

Why it matters:

`WorldScene` is no longer the only home for all behavior, but it remains the architecture's convergence point for rendering resources. That makes it difficult to change lighting, rendering order, or disposal ownership without editing the scene class.

Recommendation:

Move draw-object ordering/culling into `WorldRenderer` or a `RenderQueue` helper. Move exposure/static-light rebuild into a lighting controller. Move disposal lists into resource owners where possible, leaving `WorldScene.dispose()` as orchestration only.

Status:

Addressed in slices. `WorldRenderer.draw_world_objects()` now owns render sphere estimation, frustum visibility checks, batched object drawing, fallback object drawing, entity drawing, and sprite drawing. `WorldScene.draw_world_objects()` is a compatibility delegate only, and `tests/test_world_renderer.py` covers the pure render-sphere and frustum forwarding helpers. `src/game/world/lighting_controller.py` owns static lighting rebuild, brightness/exposure policy, texture-lighting uniform cache keys, and road/wall lighting refresh, with `tests/test_lighting_controller.py` covering alias sync, uniform-cache reuse, and exposure routing. `src/game/world/scene_resources.py` owns render-resource disposal lists, dispose-once behavior, ambient cleanup, and reference clearing, with `tests/test_scene_resources.py` covering shared-resource disposal and cleared scene state. `WorldScene` remains the orchestration/delegation point for compatibility.

### Medium: Loading progress relies on a manual step count

Evidence:

- Previously, `CREATE_WORLD_OBJECT_STEPS = 11` was defined separately from the actual build iterator.
- `WorldScene.initialize_steps()` now asks `world_builder.create_world_object_step_count()` for the world-object portion of total progress.
- The count is derived from `world_builder.create_world_object_step_specs()`, the same step metadata used by `create_world_objects_steps()`.

Why it matters:

Adding, splitting, or removing build phases can desynchronize progress reporting unless the constant is updated by hand. It is a small bug source, but it sits in the loading path and is easy to miss.

Recommendation:

Represent build steps as data first, then derive the count from the same sequence that is executed. If a step must stream substeps, expose a weighted step object instead of a standalone numeric constant.

Status:

Addressed in `src/game/world/world_builder.py` and `src/game/world/worldscene.py`. Build phases are represented as `WorldObjectBuildStep` metadata, the loading denominator is derived from those specs, and `tests/test_loading_progress.py` verifies the count stays tied to the step sequence.

### Medium: Wildcard imports obscure ownership of configuration and assets

Evidence:

- Previously, `src/game/world/worldscene.py` used `from game.config import *`; it now imports only the constants it uses.
- Previously, `src/game/world/world_setup.py` used wildcard imports from `game.config` and `game.resources.paths`; it now imports only the constants it uses.
- Previously, `src/game/world/world_runtime.py` used `from game.config import *`; it now imports only the runtime constants it uses.
- Previously, `src/engine/core/engine.py` used `from engine.config import *`; it now imports only the constants it uses.
- Previously, `src/game/resources/texture_manager.py` used `from game.resources.paths import *`; it now imports only the asset path constants it uses.

Why it matters:

Configuration is central to this project, and wildcard imports make it harder to know which module owns a constant. They also raise collision risk between engine-level and game-level config, especially because `game.config` re-exports some engine defaults.

Recommendation:

Use explicit imports for constants touched by each module. For settings-heavy modules, import the config module as a namespace, such as `import game.config as game_config`, when that improves clarity.

Status:

Addressed for the imports identified in this finding. `src/engine/core/engine.py`, `src/game/resources/texture_manager.py`, `src/game/world/world_runtime.py`, `src/game/world/world_setup.py`, and `src/game/world/worldscene.py` now use explicit imports instead of the reviewed config/resource wildcard imports.

### Medium: Renderer owns too much UI behavior

Evidence:

- `WorldRenderer` handles 3D render setup, sky, fog, world object drawing, HUD text, inventory UI, battle UI, pause menu drawing, and menu input forwarding in `src/game/world/world_renderer.py:49`.
- Previously, pause and battle input forwarding methods lived in renderer methods such as `handle_battle_click()` and `handle_pause_click()`.

Why it matters:

Rendering and UI interaction are coupled. That makes it harder to test menus without OpenGL context assumptions and harder to evolve battle/inventory UI independently.

Recommendation:

Keep `WorldRenderer` focused on OpenGL/world drawing. Move pause, battle, and inventory interaction surfaces into UI controller classes. Their draw methods can still receive a text renderer, but click/motion/release handling should not be routed through the world renderer.

Status:

Started with `src/game/world/ui/interactions.py`. Battle, pause/settings, and inventory click/motion/release forwarding now routes through `WorldUIInteractions` instead of `WorldRenderer`, and `tests/test_ui_interactions.py` covers those headless interaction paths. Continued with `src/game/world/ui/inventory_panel.py`; inventory panel drawing and inventory stat/item helpers now live outside `WorldRenderer`, with `tests/test_inventory_panel.py` covering pure panel behavior. Continued again with `src/game/world/ui/pause_panel.py`; pause/settings drawing now lives outside `WorldRenderer`, with `tests/test_pause_panel.py` covering slider geometry. Finished the current UI split with `src/game/world/ui/battle_panel.py`; battle HP/stat/title drawing now lives outside `WorldRenderer`, with `tests/test_battle_panel.py` covering HP/stat layout helpers. `WorldRenderer` still coordinates which HUD panel draws, but it no longer owns the battle, pause/settings, inventory input forwarding, or panel drawing logic identified in this finding.

### Medium: There is no test suite or automated architecture gate

Evidence:

- Repository search found no `tests` directory and no pytest/unittest references.
- `py -m compileall src/game/world` passes for the current world package, but that only proves syntax/import compilation for that slice.

Why it matters:

The project has many behavior-heavy modules with OpenGL, pygame, procedural geometry, collision, and scene state interactions. Without small headless tests, refactors like the current combat/entity extraction rely mostly on manual playtesting and compile checks.

Recommendation:

Start with headless tests for the pure or near-pure seams:

- `PlayerStats` damage rolls and crit chance.
- `BattleController` state transitions using a fake scene/entity.
- `SceneEntityRegistry` list/resource synchronization.
- `world_content` building declaration normalization.
- `world_road_planner` route selection.
- Addressed: `collision_index` candidate lookup with fake meshes.

Status:

Started with a stdlib `unittest` suite in `tests/`. Current coverage includes `BattleController` state transitions and damage flow, `SceneEntityRegistry` list/resource synchronization, and `SceneCollisionIndex` candidate lookup with fake meshes. The project now has a headless test command documented in `README.md`; additional seams in this recommendation still need coverage.

### Low: The documented naming direction conflicts with current module names

Evidence:

- The user specifically wants proper Python modules rather than more `world_` prefixed files.
- Existing docs and module names still center `world_builder.py`, `world_runtime.py`, `world_renderer.py`, `world_setup.py`, `world_spawner.py`, `world_content.py`, `world_lighting_plan.py`, and `world_road_planner.py`.
- The recent extractions use better names: `combat.py`, `collision_index.py`, `entity_registry.py`, `lighting_controller.py`, and `scene_resources.py`.

Why it matters:

Naming is not just style here. The `world_` prefix reflects the current broad subsystem modules, and it encourages future changes to add more similarly broad files. Better names point toward domain responsibilities.

Recommendation:

Do not mass-rename all existing files at once. Rename or replace them opportunistically as responsibilities are extracted:

- `world_runtime.py` toward `runtime.py`, `interaction.py`, `collision_index.py`, and `ambient_audio.py`.
- `world_renderer.py` toward `renderer.py`, `render_queue.py`, and UI-specific controllers.
- `world_builder.py` toward a `building_pipeline.py`, `terrain_pipeline.py`, `road_pipeline.py`, and `spawn_pipeline` split.

## Strengths

- The engine/game dependency direction is clean: search found no `game.*` imports under `src/engine`.
- `src/main.py` is minimal and delegates launch to `Engine` and `MainMenuScene`.
- `Scene` is intentionally small, which keeps the loop flexible.
- `LoadingScene` gives the world a nonblocking setup path through `WorldScene.initialize_steps()`.
- The world package already has several good domain seams: `world_content.py`, `world_lighting_plan.py`, `world_road_planner.py`, `player_controller.py`, `battle_cards.py`, `player_stats.py`, `combat.py`, `collision_index.py`, `entity_registry.py`, `lighting_controller.py`, and `scene_resources.py`.
- The docs in `DOCUMENTATION.md` describe the broad runtime flow and now include `combat.py`, `collision_index.py`, `entity_registry.py`, `lighting_controller.py`, and `scene_resources.py`.

## Suggested Refactor Roadmap

1. Add logging or debug re-raise behavior for swallowed engine event/dispose exceptions.
2. Add headless unit tests for `combat.py` and `entity_registry.py` to lock in the newly extracted behavior.
3. Addressed: extract `draw_world_objects()` from `WorldScene` into renderer-side code.
4. Addressed: extract collision indexing from `world_runtime.py` into `collision_index.py`.
5. Extract battle/pause/inventory input handling from `WorldRenderer` into UI controllers.
6. Addressed: replace `CREATE_WORLD_OBJECT_STEPS` with a derived build-step count.
7. Replace wildcard config imports in the highest-churn modules.
8. Split `world_builder.py` by pipeline phase after tests cover the relevant generated scene state.

## Current Metrics

- Largest files by line count:
  - `src/engine/core/compat_shader.py`: 1,429 lines
  - `src/game/world/world_builder.py`: 1,347 lines
  - `src/engine/textures/texture_utils.py`: 1,172 lines
  - `src/game/world/objects/goblin.py`: 942 lines
  - `src/game/world/objects/building.py`: 940 lines
  - `src/game/world/ui/setting_menu.py`: 860 lines
  - `src/game/world/objects/ground.py`: 796 lines
  - `src/engine/rendering/sprite.py`: 766 lines
  - `src/engine/rendering/lighting.py`: 749 lines
  - `src/game/world/world_runtime.py`: 720 lines
- World package compile check: `py -m compileall src/game/world` passes.
- Current worktree contains architecture-aligned changes:
  - `src/game/world/combat.py`
  - `src/game/world/collision_index.py`
  - `src/game/world/entity_registry.py`
  - `src/game/world/lighting_controller.py`
  - `src/game/world/scene_resources.py`
  - updates to `src/game/world/worldscene.py`
  - updates to `src/game/world/__init__.py`
