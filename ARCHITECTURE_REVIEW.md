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

- `WorldScene.__init__()` still initializes many generated world collections directly at `src/game/world/worldscene.py:58`.
- `world_builder.py` now orchestrates construction steps only; actual work is split across `building_pipeline.py`, `terrain_pipeline.py`, `road_pipeline.py`, `spawn_pipeline.py`, and `detail_pipeline.py`.
- `building_pipeline.py` still mutates scene attributes throughout building/showcase construction, including walls, doors, windows, chests, sprites, lighting, and collision-facing lists. It is the largest new construction owner at about 568 lines.
- `world_runtime.py` is a procedural subsystem operating on arbitrary scene attributes. It has about 601 lines after the collision-index extraction.
- `world_renderer.py` reads a broad scene surface via `getattr` throughout rendering and UI handling. It has about 528 lines after the UI and object-draw extractions.
- Recent extraction improved this locally: `BattleController` is now at `src/game/world/combat.py:14`, `SceneCollisionIndex` is now at `src/game/world/collision_index.py:12`, `SceneEntityRegistry` is now at `src/game/world/entity_registry.py:13`, and `StaticLightingController` is now at `src/game/world/lighting_controller.py:10`.

Why it matters:

The architecture currently has modules, but not strong boundaries. Most world modules communicate through a large mutable `WorldScene` object rather than typed subsystem state. That keeps call sites short, but makes ordering, ownership, and invariants hard to prove. For example, renderer code assumes `scene.sky`, `scene.ground_mesh`, `scene._headbob`, and many UI flags exist after setup, but those requirements are not represented by an explicit state object.

Recommendation:

Continue the extraction pattern used for combat and entity registration. Create focused state owners for construction output, lighting/exposure, collision queries, and UI/menu state. Use those owners as constructor dependencies for renderer/runtime helpers rather than letting every helper reach across the whole scene.

Status:

Progressed in focused slices. Collision source keying, spatial cell rebuilds, dynamic/fallback mesh handling, candidate lookup, and invalidation now live in `SceneCollisionIndex` instead of `world_runtime.py`. `WorldScene` owns the collision index instance and keeps compatibility delegate methods, while runtime movement asks the scene for collision candidates. Construction output is now split by pipeline ownership: `world_builder.py` keeps build-step orchestration, `building_pipeline.py` owns buildings/showcase geometry/doors/windows/chests, `terrain_pipeline.py` owns ground/fences, `road_pipeline.py` owns road construction and batching, `spawn_pipeline.py` owns sprite/enemy spawning, and `detail_pipeline.py` owns decals/ground detail. `tests/test_collision_index.py` covers collision lookup behavior, and `tests/test_world_builder_state.py` covers generated scene state for building preparation, road construction, and showcase chest registration. The remaining shared-state risk is the broad mutable `WorldScene` surface, not a single oversized builder file.

### Medium: `WorldScene` still owns rendering internals and lighting rebuild policy

Evidence:

- Previously, `WorldScene.draw_world_objects()` contained object position estimation, render sphere estimation, sorting, culling, batched drawing, and fallback drawing. It now delegates at `src/game/world/worldscene.py:332`, with that behavior moved to `WorldRenderer.draw_world_objects()`.
- Previously, `WorldScene.refresh_static_lighting()` rebuilt ground, roads, walls, fences, batches, and shader state. It now delegates at `src/game/world/worldscene.py:446`, with that behavior moved to `StaticLightingController.refresh_static()`.
- Previously, `WorldScene._sync_lighting_uniforms()` owned shader synchronization cache keys. It now delegates at `src/game/world/worldscene.py:452`, with the cache policy moved to `StaticLightingController.sync_uniforms()`.
- Previously, `WorldScene.dispose()` had detailed knowledge of renderable ownership across nearly every world subsystem. It now delegates at `src/game/world/worldscene.py:551`, with resource cleanup moved to `SceneResourceDisposer.dispose()`.

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

Addressed for the imports identified in this finding. `src/engine/core/engine.py`, `src/game/resources/texture_manager.py`, `src/game/world/world_runtime.py`, `src/game/world/world_setup.py`, and `src/game/world/worldscene.py` now use explicit imports instead of the reviewed config/resource wildcard imports. Additional opportunistic cleanup removed the remaining `engine.core.consts` wildcard imports from world UI modules.

### Medium: Renderer owns too much UI behavior

Evidence:

- `WorldRenderer` previously handled 3D render setup, sky, fog, world object drawing, HUD text, inventory UI, battle UI, pause menu drawing, and menu input forwarding in `src/game/world/world_renderer.py:49`. It now delegates menu input and panel drawing to UI-specific helpers.
- Previously, pause and battle input forwarding methods lived in renderer methods such as `handle_battle_click()` and `handle_pause_click()`.

Why it matters:

Rendering and UI interaction are coupled. That makes it harder to test menus without OpenGL context assumptions and harder to evolve battle/inventory UI independently.

Recommendation:

Keep `WorldRenderer` focused on OpenGL/world drawing. Move pause, battle, and inventory interaction surfaces into UI controller classes. Their draw methods can still receive a text renderer, but click/motion/release handling should not be routed through the world renderer.

Status:

Started with `src/game/world/ui/interactions.py`. Battle, pause/settings, and inventory click/motion/release forwarding now routes through `WorldUIInteractions` instead of `WorldRenderer`, and `tests/test_ui_interactions.py` covers those headless interaction paths. Continued with `src/game/world/ui/inventory_panel.py`; inventory panel drawing and inventory stat/item helpers now live outside `WorldRenderer`, with `tests/test_inventory_panel.py` covering pure panel behavior. Continued again with `src/game/world/ui/pause_panel.py`; pause/settings drawing now lives outside `WorldRenderer`, with `tests/test_pause_panel.py` covering slider geometry. Finished the current UI split with `src/game/world/ui/battle_panel.py`; battle HP/stat/title drawing now lives outside `WorldRenderer`, with `tests/test_battle_panel.py` covering HP/stat layout helpers. `WorldRenderer` still coordinates which HUD panel draws, but it no longer owns the battle, pause/settings, inventory input forwarding, or panel drawing logic identified in this finding.

### Medium: There is a test suite, but no automated architecture gate

Evidence:

- The repository now has a stdlib `unittest` suite under `tests/`, and `README.md` documents `py -m unittest discover -s tests`.
- The current suite covers combat, entity registration, collision indexing, loading progress, world content, world road planning, world renderer helpers, lighting, resource disposal, UI interactions/panels, player stats, and generated builder scene state.
- There is still no CI configuration or architecture-level gate that runs the suite automatically.

Why it matters:

The project has many behavior-heavy modules with OpenGL, pygame, procedural geometry, collision, and scene state interactions. The headless tests now protect several pure and near-pure seams, but regressions can still land if the tests are not run consistently.

Recommendation:

Keep expanding headless tests around pure or near-pure seams:

- Addressed: `PlayerStats` damage rolls and crit chance.
- Addressed: `BattleController` state transitions using a fake scene/entity.
- Addressed: `SceneEntityRegistry` list/resource synchronization.
- Addressed: `world_content` building declaration normalization.
- Addressed: `world_road_planner` route selection.
- Addressed: `collision_index` candidate lookup with fake meshes.
- Addressed: generated builder scene state for building preparation, roads, and showcase chest registration.
- Still open: add an automated gate that runs `py -m unittest discover -s tests` and a package compile check.

Status:

Addressed for the original test-suite gap. Current coverage includes all starter seams from this recommendation plus generated builder scene-state tests. The project now has a headless test command documented in `README.md`; a CI-level architecture gate is still not configured.

### Low: The documented naming direction conflicts with current module names

Evidence:

- The user specifically wants proper Python modules rather than more `world_` prefixed files.
- Existing docs and module names still include broad `world_runtime.py`, `world_renderer.py`, `world_setup.py`, `world_spawner.py`, `world_content.py`, `world_lighting_plan.py`, and `world_road_planner.py`.
- The recent extractions use better names: `combat.py`, `collision_index.py`, `entity_registry.py`, `lighting_controller.py`, `scene_resources.py`, `building_pipeline.py`, `terrain_pipeline.py`, `road_pipeline.py`, `spawn_pipeline.py`, and `detail_pipeline.py`.

Why it matters:

Naming is not just style here. The `world_` prefix reflects the current broad subsystem modules, and it encourages future changes to add more similarly broad files. Better names point toward domain responsibilities.

Recommendation:

Do not mass-rename all existing files at once. Rename or replace them opportunistically as responsibilities are extracted:

- `world_runtime.py` toward `runtime.py`, `interaction.py`, `collision_index.py`, and `ambient_audio.py`.
- `world_renderer.py` toward `renderer.py`, `render_queue.py`, and UI-specific controllers.
- Addressed: `world_builder.py` now delegates to `building_pipeline.py`, `terrain_pipeline.py`, `road_pipeline.py`, `spawn_pipeline.py`, and `detail_pipeline.py`.

## Strengths

- The engine/game dependency direction is clean: search found no `game.*` imports under `src/engine`.
- `src/main.py` is minimal and delegates launch to `Engine` and `MainMenuScene`.
- `Scene` is intentionally small, which keeps the loop flexible.
- `LoadingScene` gives the world a nonblocking setup path through `WorldScene.initialize_steps()`.
- The world package already has several good domain seams: `world_content.py`, `world_lighting_plan.py`, `world_road_planner.py`, `player_controller.py`, `battle_cards.py`, `player_stats.py`, `combat.py`, `collision_index.py`, `entity_registry.py`, `lighting_controller.py`, `scene_resources.py`, and the construction pipeline modules.
- The docs in `DOCUMENTATION.md` describe the broad runtime flow and now include `combat.py`, `collision_index.py`, `entity_registry.py`, `lighting_controller.py`, and `scene_resources.py`.

## Directory Structure Notes

- The top-level source split still makes sense: `src/engine/` contains reusable loop, rendering, camera, collision, sound, texture, and UI primitives, while `src/game/` contains game-specific config, resources, scenes, content, and world behavior.
- `src/game/world/objects/` and `src/game/world/ui/` are the right subpackages for repeated object and UI families. The construction pipeline modules can stay in `src/game/world/` for now because they coordinate across objects, terrain, roads, spawning, and decals rather than belonging to one object family.
- `world_road_planner.py` is large, but it is still cohesive: most of the file is route scoring, candidate generation, and road instantiation for one domain problem. Leave it intact until there is a concrete second responsibility to extract.
- Local `__pycache__/` directories exist after test and compile runs, but they are ignored by `.gitignore` and are not a source-layout issue.
- The notable structure gap was the lack of a single architecture gate entrypoint. That is now addressed by `scripts/check_architecture.py`, which runs the world compile check, headless tests, engine/game dependency-direction check, and wildcard-import check.

## Suggested Refactor Roadmap

1. Addressed: add logging or debug re-raise behavior for swallowed engine event/dispose exceptions.
2. Addressed: add headless unit tests for `combat.py` and `entity_registry.py` to lock in the newly extracted behavior.
3. Addressed: extract `draw_world_objects()` from `WorldScene` into renderer-side code.
4. Addressed: extract collision indexing from `world_runtime.py` into `collision_index.py`.
5. Addressed: extract battle/pause/inventory input handling from `WorldRenderer` into UI controllers.
6. Addressed: replace `CREATE_WORLD_OBJECT_STEPS` with a derived build-step count.
7. Addressed: replace wildcard config imports in the highest-churn modules reviewed here.
8. Addressed: split `world_builder.py` by pipeline ownership after adding generated scene-state tests.
9. Addressed: add a lightweight automated gate for `py -m unittest discover -s tests` and `py -m compileall src/game/world`.
10. Next: continue reducing broad scene coupling where it creates bugs, especially in `building_pipeline.py` and `world_runtime.py`. Leave `world_road_planner.py` cohesive for now despite its size.

## Current Metrics

- Largest files by line count:
  - `src/engine/core/compat_shader.py`: 1,284 lines
  - `src/engine/textures/texture_utils.py`: 1,010 lines
  - `src/game/world/objects/building.py`: 854 lines
  - `src/game/world/objects/goblin.py`: 830 lines
  - `src/game/world/ui/setting_menu.py`: 755 lines
  - `src/game/world/objects/ground.py`: 744 lines
  - `src/engine/rendering/sprite.py`: 661 lines
  - `src/game/world/world_road_planner.py`: 648 lines
  - `src/engine/core/mesh.py`: 637 lines
  - `src/engine/rendering/lighting.py`: 636 lines
  - `src/game/world/objects/polygon.py`: 628 lines
  - `src/game/world/world_runtime.py`: 601 lines
  - `src/game/world/objects/road.py`: 585 lines
  - `src/game/world/building_pipeline.py`: 568 lines
  - `src/game/world/world_builder.py`: 153 lines
- World package compile check: `py -m compileall src/game/world` passes.
- Architecture gate: `py scripts/check_architecture.py` runs world compile, headless tests, engine/game dependency-direction, and wildcard-import checks.
- Current worktree contains architecture-aligned changes:
  - `scripts/check_architecture.py`
  - `src/game/world/builder_support.py`
  - `src/game/world/building_pipeline.py`
  - `src/game/world/terrain_pipeline.py`
  - `src/game/world/road_pipeline.py`
  - `src/game/world/spawn_pipeline.py`
  - `src/game/world/detail_pipeline.py`
  - `src/game/world/combat.py`
  - `src/game/world/collision_index.py`
  - `src/game/world/entity_registry.py`
  - `src/game/world/lighting_controller.py`
  - `src/game/world/scene_resources.py`
  - updates to `src/game/world/worldscene.py`
  - updates to `src/game/world/__init__.py`
