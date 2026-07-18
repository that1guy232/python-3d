# Project Structure Review

Date: 2026-07-17

## Scope and Method

This is an overall structural review, not a line-by-line style review. It covers
all 91 Python files under `src/` and `scripts/check_architecture.py`. Asset files
were treated as inputs to the program rather than code. The review is based on
the current working tree, including the user's uncommitted changes in
`src/main.py` and `src/game/world/spawn_pipeline.py`; those changes were not
modified.

Checks performed:

- Parsed every Python file and reviewed its imports, public classes/functions,
  and responsibility.
- Traced the launch, loading, update, input, render, and cleanup paths.
- Built an internal import graph and checked the `engine`/`game` dependency
  direction.
- Imported all 91 source modules in one smoke test.
- Compiled all of `src/` and `scripts/`.
- Ran the repository's architecture gate.
- Compared `README.md`, `DOCUMENTATION.md`, and the previous architecture review
  with the files actually present.

## Executive Assessment

The project has a sound top-level idea: a reusable `engine` package supports a
game-specific `game` package, the entry point is small, the menu transitions
through an incremental loading scene, and the world code has several useful
domain seams. The render path also shows deliberate attention to batching,
culling, caching, resource disposal, and profiling.

The main architectural weakness is that the split into modules is often only a
physical split. `WorldScene` remains a large mutable service locator, and many
world modules read and write arbitrary attributes on it. The code therefore has
many files, but subsystem contracts and ownership are still implicit. This is
made riskier by the current absence of tests and by widespread exception
suppression.

Overall status: **promising architecture, but not yet safely changeable**. New
features can be added, but regression diagnosis will become increasingly hard
until verification, state ownership, and error policy are strengthened.

## Architecture at a Glance

```text
src/main.py
  -> Engine
      -> MainMenuScene
          -> LoadingScene
              -> WorldScene
                  -> setup/build pipelines
                  -> PlayerCameraController / BattleController
                  -> SceneEntityRegistry / SceneCollisionIndex
                  -> StaticLightingController / SceneResourceDisposer
                  -> WorldRenderer / world UI

engine.*  <- reusable loop, camera, rendering, collision, audio, UI primitives
game.*    <- content, world construction, gameplay, world objects, world UI
```

The intended dependency direction is being respected: no module under
`src/engine/` imports `game.*`. The most depended-on internal modules are
`engine.rendering.lighting`, `engine.textures.texture_utils`, `game.config`, and
`engine.core.mesh`, which is reasonable for this kind of renderer.

## Findings

### High: The documented test suite is absent and the architecture gate fails

`README.md:32-43` tells contributors to run a headless test suite and the
architecture gate. `scripts/check_architecture.py:11` points at a `tests/`
directory, but no such directory or tracked test file exists. The previous
review names many specific tests that are not present in the repository.

Observed result:

```text
py scripts/check_architecture.py
...
ImportError: Start directory is not importable: 'C:\python-3d\tests'
[check] failed: 1 check(s)
```

This is the most urgent issue because the highest-risk code is procedural
geometry, mutable scene orchestration, collision, GPU resource lifetime, and
input state. Those areas need executable regression protection before further
large refactors.

Recommendation:

1. Restore the intended tests if they exist elsewhere; otherwise create a real
   `tests/` package.
2. Start with headless tests for `world_content`, `interior_layout`,
   `world_road_planner`, `collision_index`, `combat`, `player_stats`, build-step
   metadata, and resource-disposal idempotence.
3. Make the gate compile all of `src/`, verify that at least one test was
   discovered, and return a clear error when `tests/` is absent.
4. Add CI only after the local gate is trustworthy.


### Medium: Broad exception handling makes failures look like missing visuals or input

The source contains 130 broad `except Exception`/bare handlers across 44 files.
Sixty-three of those handlers immediately `pass` or `continue`. Some are valid
best-effort compatibility fallbacks, especially around optional audio or OpenGL
capabilities, but the policy is not consistent.

Examples with architectural impact include:

- `LoadingScene.dispose()` silently suppresses target-scene cleanup failure at
  `src/engine/core/loading_scene.py:147`.
- `world_setup.setup_controllers()` has nested broad suppression around
  footsteps and callbacks at `src/game/world/world_setup.py:110-134`.
- HUD update/draw errors are ignored in `world_runtime.py:524-552` and
  `world_renderer.py:138-142`.
- Camera event and mouse-delta failures are ignored at
  `world_runtime.py:706-723`.

The engine now logs scene event and disposal failures, which is a good pattern,
but deeper subsystems can still erase the original error before it reaches that
boundary.

Recommendation: define one error policy. Catch specific expected failures;
route unexpected failures through a logger with subsystem context; provide a
debug flag that re-raises; and reserve silent fallback for explicitly optional
effects.

### Medium: Installation and runtime dependencies are not declared

The project imports `pygame`, `OpenGL`, and `numpy`, but there is no
`pyproject.toml`, requirements file, lock file, or setup metadata. The README
also does not document dependency installation or the main run command.

This means a working checkout cannot be reproduced from repository contents
alone, and Python/library compatibility is accidental. The current environment
successfully imported all 91 modules, but that only proves this machine is
configured.

Recommendation: add a minimal `pyproject.toml` declaring the supported Python
range and runtime dependencies, then document `py src/main.py` and the verified
setup/test commands. Add a lock file only if repeatable application builds are a
goal.

### Medium: Procedural generation cannot be reproduced from one world seed

World generation uses a mixture of the process-global `random` module,
unseeded `random.Random()` instances, and NumPy's global random generator.
Examples include `world_spawner.py:303-364`, `spawn_pipeline.py:234`,
`world_content.py:510`, `detail_pipeline.py`, and `objects/fence.py:82`.

As a result, a bad placement, route, collision arrangement, or performance
spike cannot be recreated reliably from a single seed. It also makes structural
tests harder to write.

Recommendation: give `WorldScene` or `WorldContent` an explicit seed and derive
named RNG streams for content, sprites, enemies, and details. Avoid mixing
Python and NumPy global RNG state; pass generators into the algorithms that need
them. Log the seed at world creation.

### Medium: The scene contract is informal and signatures drift

`engine.core.scene.Scene` is described as a protocol, but it is a permissive
base class whose `render(self)` signature at `src/engine/core/scene.py:32` does
not match the keyword arguments always sent by `Engine.render()` at
`src/engine/core/engine.py:177-184`. Other optional hooks are discovered with
`hasattr`/`getattr`, and scene transitions are requested by attaching
`next_scene` dynamically.

This works because current scenes follow the convention, but static checking
cannot guarantee it and a new scene can fail only when first rendered.

Recommendation: define a `Scene` protocol or abstract base with the actual
engine-called signatures and explicit defaults for `next_scene`, mouse state,
`update`, `handle_event`, `render`, `apply_mouse_delta`, and `dispose`. A small
`SceneTransition` result would be clearer than a dynamically attached field,
but is optional.

### Low: Repository formatting and line-ending policy is inconsistent

`world_builder.py`, `road_pipeline.py`, `terrain_pipeline.py`, and
`detail_pipeline.py` use lone carriage-return line endings. Several other files
mix LF and CRLF. This is why tools such as `rg` report entire pipeline files as
line 1 and why a small edit can appear as a whole-file diff. The current
`spawn_pipeline.py` working change demonstrates that normalization effect.

Recommendation: add `.gitattributes` with a single text policy, add a formatter
and import sorter configuration, then normalize line endings in a dedicated
commit so functional changes remain reviewable.

### Low: Documentation has drifted from the repository

`DOCUMENTATION.md` accurately explains the broad runtime flow, but its source
map omits ten current modules: `battle_cards.py`, `interior_layout.py`,
`player_stats.py`, and seven UI modules. The previous architecture review also
reported nonexistent tests and stale file sizes. Documentation should describe
current guarantees, not completed work that is no longer in the tree.

Recommendation: update documentation in the same change that adds/removes a
module or verification command. The architecture gate can cheaply check that
every non-`__init__` source module appears in the source map.

## What Is Working Well

- The dependency direction is clean: `engine.*` does not import `game.*`.
- All 91 source modules import successfully in the current environment.
- `py -m compileall -q src scripts` passes.
- The startup flow is easy to follow: entry point, engine, main menu, loading
  scene, then world scene.
- Incremental world initialization is a good fit for expensive mesh and asset
  construction.
- `WorldContent` provides a useful declarative seam ahead of mutable runtime
  geometry.
- `SceneCollisionIndex`, `SceneEntityRegistry`, `BattleController`,
  `StaticLightingController`, and `SceneResourceDisposer` are sensible domains
  to extract.
- Rendering code deliberately uses batching, culling, cached shader state, and
  profiling. Those are appropriate optimizations for a Python/OpenGL project.
- Asset paths are resolved relative to the repository rather than the current
  working directory.
- The large geometry modules are mostly cohesive algorithm families. File size
  alone is not a reason to split `compat_shader.py`, `building.py`, `ground.py`,
  `road.py`, or `world_road_planner.py`; tests and clearer internal sections
  should come first.

## Recommended Roadmap

1. **Restore verification:** add/restore tests, make the architecture gate
   honest, and run it in CI.
2. **Make setup reproducible:** declare dependencies and document install/run
   commands.
3. **Add deterministic world seeds:** pass explicit RNGs through content and
   build pipelines.
4. **Define contracts:** formalize the scene interface and introduce typed
   world build/render/UI state containers.
5. **Reduce silent failure:** centralize logging and debug re-raise behavior.
6. **Tighten ownership:** migrate callers away from `WorldScene` compatibility
   delegates and whole-scene dependencies one subsystem at a time.
7. **Normalize the repository:** establish line-ending and formatting policy in
   a standalone mechanical change.
8. **Continue feature work:** only split large cohesive algorithms when a test
   or a real second responsibility gives a clear boundary.

## File-by-File Structural Review

The notes below cover every Python code file. “Keep” means the file has a clear
overall responsibility; it does not mean every implementation detail is ideal.

### Entry, Configuration, and Game Resources

| File | Overall structural review |
| --- | --- |
| `src/main.py` | Keep. Minimal composition root that creates the first scene and starts the engine. |
| `src/game/__init__.py` | Keep as a package marker. |
| `src/game/config.py` | Keep, but separate user-facing tuning from engine constants as settings grow. |
| `src/game/main_menu.py` | Keep. Cohesive menu scene and transition into incremental world loading. |
| `src/game/resources/__init__.py` | Keep as a package marker. |
| `src/game/resources/paths.py` | Strong. Central, repository-rooted asset paths prevent working-directory bugs. |
| `src/game/resources/texture_manager.py` | Keep. Clear game-specific texture catalog; a resource object would eventually be safer than a string-keyed dict. |
| `src/game/resources/heightmapgen.py` | Keep. Standalone content-generation tool; document its CLI separately from runtime code. |

### Engine Core and Runtime

| File | Overall structural review |
| --- | --- |
| `src/engine/__init__.py` | Keep. Small public engine namespace. |
| `src/engine/config.py` | Keep. Environment flags are centralized; dependency metadata should declare the supported runtime. |
| `src/engine/entity.py` | Good minimal extension surface. Consider a protocol when entity capabilities become more varied. |
| `src/engine/collision.py` | Keep. Generic geometry/collision responsibility; broad compatibility fallbacks need tests and narrower exceptions. |
| `src/engine/core/__init__.py` | Keep as the core package export surface. |
| `src/engine/core/scene.py` | Refactor first. Its interface is smaller than the contract the engine actually calls. |
| `src/engine/core/engine.py` | Strong central loop. Add `try/finally` shutdown guarantees and align it with a formal scene contract. |
| `src/engine/core/loading_scene.py` | Good incremental-loading abstraction. Cleanup errors should be logged, and failed setup should have a visible error state. |
| `src/engine/core/performance.py` | Keep. Cohesive lightweight profiler with a useful engine/world integration point. |
| `src/engine/core/object3d.py` | Keep. Focused transform and bounds base for legacy/simple mesh objects. |
| `src/engine/core/mesh.py` | Keep. `BatchedMesh` and terrain sampling are both core geometry concerns, though they could become separate modules if either grows. |
| `src/engine/core/compat_shader.py` | Large but cohesive compatibility layer. Preserve until the fixed/compatibility render path is intentionally replaced; add shader-state tests around pure key/state helpers. |
| `src/engine/core/consts.py` | Keep for now; migrate direction constants toward their owning camera/world APIs if legacy usage shrinks. |

### Engine Camera, Rendering, Audio, Textures, and UI

| File | Overall structural review |
| --- | --- |
| `src/engine/camera/__init__.py` | Keep as the camera export surface. |
| `src/engine/camera/camera.py` | Mostly cohesive, but view/frustum state and world-brightness sampling are separate concerns. Extract brightness sampling only when tests exist. |
| `src/engine/camera/headbob.py` | Keep. Encapsulated motion state is a good seam; callback errors should not all be silent. |
| `src/engine/camera/sway_controller.py` | Keep. Small, focused smoothing controller. |
| `src/engine/rendering/__init__.py` | Keep as rendering exports. |
| `src/engine/rendering/lighting.py` | Important shared domain module. Large but cohesive mathematical core; prioritize pure unit tests over splitting. |
| `src/engine/rendering/sprite.py` | Keep. Billboard types and their batch renderer belong together; continue profiling cache complexity. |
| `src/engine/rendering/sky_renderer.py` | Keep. Clean sky-pass coordinator over clouds and lighting. |
| `src/engine/rendering/cloud_renderer.py` | Keep. Focused batched cloud implementation. |
| `src/engine/rendering/decal.py` | Keep. One terrain-conforming decal abstraction. |
| `src/engine/rendering/decal_batch.py` | Keep. Correct ownership for batching/culling many decals; disposal and rebuild behavior need tests. |
| `src/engine/sound/__init__.py` | Keep as a package marker. |
| `src/engine/sound/sound_utils.py` | Useful service, but global keyed state couples scenes. Consider an injected audio service if multiple worlds or tests need isolation. |
| `src/engine/textures/__init__.py` | Keep as a package marker. |
| `src/engine/textures/texture_utils.py` | Large but logically a texture/procedural-texture toolbox. Separate file loading/registry from procedural generation when packaging or testing requires it. |
| `src/engine/ui/__init__.py` | Keep as a package marker. |
| `src/engine/ui/menu.py` | Strong reusable button/slider model. It is one of the best headless-test targets. |
| `src/engine/ui/text_renderer.py` | Keep. Focused bridge from pygame fonts to OpenGL textures; clarify cache/resource disposal ownership. |

### World Orchestration and Domain Services

| File | Overall structural review |
| --- | --- |
| `src/game/world/__init__.py` | Lazy exports avoid import side effects, but the broad convenience surface hides ownership. Prefer direct module imports internally. |
| `src/game/world/worldscene.py` | Central lifecycle owner, but currently an oversized shared-state façade. Keep orchestration here while moving state and contracts into subsystem owners. |
| `src/game/world/world_setup.py` | Cohesive bootstrap phase, yet it attaches many undocumented scene fields. Return typed setup results instead. |
| `src/game/world/world_builder.py` | Good data-driven step orchestration. Stop re-exporting/importing every private pipeline helper; depend on one public operation per pipeline. |
| `src/game/world/builder_support.py` | Keep. Small step/disposal support; resource disposal ultimately belongs with resource owners. |
| `src/game/world/building_pipeline.py` | Largest construction coordinator and heaviest scene mutator. Split only along output ownership (structure, openings, showcase) after tests capture build invariants. |
| `src/game/world/terrain_pipeline.py` | Keep. Clear terrain/fence construction phase; normalize its line endings. |
| `src/game/world/road_pipeline.py` | Keep. Clear construction/batching phase over the separate planner and road object; normalize line endings. |
| `src/game/world/spawn_pipeline.py` | Keep as a spawn phase. Inject deterministic RNGs and use entity-registry APIs consistently. |
| `src/game/world/detail_pipeline.py` | Keep. Cohesive decal/detail construction; deterministic RNG and line-ending normalization are needed. |
| `src/game/world/world_runtime.py` | Refactor gradually. It combines frame update, audio, interaction ray tests, battle triggers, input routing, and height helpers over shared scene state. Best extraction candidates are interaction/focus and ambient audio. |
| `src/game/world/world_renderer.py` | Keep as render coordinator. Panel delegation is good; silent per-object fallbacks and dependence on many scene attributes remain risks. |
| `src/game/world/world_content.py` | Strong declarative content seam. Make the seed/RNG part of this public contract. |
| `src/game/world/interior_layout.py` | Keep. Cohesive pure-ish layout algorithm and an excellent high-value test target. |
| `src/game/world/world_lighting_plan.py` | Keep. Correctly separates authored lighting derivation from reusable engine lighting math. |
| `src/game/world/lighting_controller.py` | Good extracted policy owner, but still tightly coupled to scene internals. Move toward explicit render resources and lighting state inputs. |
| `src/game/world/world_road_planner.py` | Large but cohesive routing algorithm. Do not split by size; add deterministic route tests first. |
| `src/game/world/world_spawner.py` | Keep. Spatial-grid placement is a clear domain. Replace mixed global Python/NumPy randomness with an injected generator. |
| `src/game/world/collision_index.py` | Strong extraction with a clear responsibility. Define a collision-source protocol and test invalidation/rebuild behavior. |
| `src/game/world/entity_registry.py` | Good ownership direction. It should become the only supported way to synchronize entities, sprites, collision, and immediate rendering. |
| `src/game/world/combat.py` | Keep. Focused battle state controller; move compatibility fields into a battle-state object over time. |
| `src/game/world/player_stats.py` | Keep. Small value object and straightforward pure test target. |
| `src/game/world/player_controller.py` | Keep. Movement/input/collision belong together, though scene query dependencies should be expressed through a narrow interface. |
| `src/game/world/battle_cards.py` | Keep. Small loadout/composition owner; avoid hard-coded test-card naming as the real card catalog grows. |
| `src/game/world/scene_resources.py` | Good lifecycle extraction. Make ownership registration explicit so cleanup does not depend on a growing hard-coded list of scene attributes. |

### World Objects

| File | Overall structural review |
| --- | --- |
| `src/game/world/objects/__init__.py` | Useful object export surface; internal code should still import concrete owner modules when clarity matters. |
| `src/game/world/objects/building.py` | Large but cohesive footprint-to-structure generator. It needs invariant tests for openings, wall spans, bounds, and ground adaptation before refactoring. |
| `src/game/world/objects/ground.py` | Cohesive terrain mesh builder. Vectorized sampling and flatten pads are appropriate; test sampling/pad math without OpenGL. |
| `src/game/world/objects/ground_tile.py` | Clearly marked legacy primitive. Remove it once searches confirm no compatibility caller needs it. |
| `src/game/world/objects/road.py` | Cohesive road geometry, containment, lighting, and batching family. Keep; test centerline cleaning and joins. |
| `src/game/world/objects/fence.py` | Keep. Small builder, but take an RNG rather than choosing textures globally. |
| `src/game/world/objects/slab.py` | Strong shared geometry mixin for door/window/chest-style objects. Keep its public contract narrow. |
| `src/game/world/objects/door.py` | Cohesive interactive slab plus batch. Audio/light/collision integration makes it an important lifecycle test target. |
| `src/game/world/objects/window.py` | Cohesive fixed slab plus batch. Similar batch/cache behavior to doors suggests shared tests more than another abstraction. |
| `src/game/world/objects/chest.py` | Cohesive interactive animated object, though it owns substantial custom geometry. Keep until another object shares that geometry behavior. |
| `src/game/world/objects/torch.py` | Keep. Good bridge from building authoring to sprites and brightness modifiers. |
| `src/game/world/objects/goblin.py` | The broadest object: AI, animation, audio, movement, and rendering. First extract a testable behavior/state machine or audio policy, not arbitrary method groups. |
| `src/game/world/objects/polygon.py` | Keep. Primitive and its batch belong together; triangulation and UV mapping need pure tests. |
| `src/game/world/objects/wall_tile.py` | Keep. Static wall primitive and batch construction are cohesive; lighting/UV vertex generation deserves regression tests. |

### World UI

| File | Overall structural review |
| --- | --- |
| `src/game/world/ui/__init__.py` | Keep as UI exports. |
| `src/game/world/ui/world_hud.py` | Good HUD coordinator. Avoid silently suppressing every child overlay failure. |
| `src/game/world/ui/compass_overlay.py` | Keep. Focused overlay. |
| `src/game/world/ui/minimap_overlay.py` | Large but cohesive layered minimap renderer. Static layout and coordinate transforms are good headless-test seams. |
| `src/game/world/ui/interactions.py` | Good extraction of click/motion/release routing. It still needs a typed UI-state dependency instead of the full scene. |
| `src/game/world/ui/inventory_panel.py` | Keep. Drawing and inventory presentation helpers are cohesive. |
| `src/game/world/ui/pause_menu.py` | Keep. Menu actions are clear; engine exit/scene transition callbacks should eventually be explicit services. |
| `src/game/world/ui/pause_panel.py` | Keep. Focused panel renderer over the generic menu model. |
| `src/game/world/ui/setting_menu.py` | Too broad because every setting is implemented as repeated scene introspection and mutation. Generate options from declarative setting descriptors and split categories only after that. |
| `src/game/world/ui/battle_menu.py` | Keep. Minimal battle command menu. |
| `src/game/world/ui/battle_overlay.py` | Keep. Cohesive card/resource interaction overlay; input state transitions should be tested headlessly. |
| `src/game/world/ui/battle_panel.py` | Keep. Focused battle HUD presentation. |
| `src/game/world/ui/card.py` | Strong self-contained UI model for hover, drag, play-zone, and drawing behavior. Separate geometry/input tests from OpenGL drawing. |

### Repository Script

| File | Overall structural review |
| --- | --- |
| `scripts/check_architecture.py` | Useful single entry point, but currently fails because `tests/` is absent. Compile all source packages, require nonzero test discovery, retain dependency-direction and wildcard checks, and make missing prerequisites explicit. |

## Current Metrics and Verification Results

- Python source files under `src/`: 91.
- Additional Python scripts reviewed: 1.
- Physical source lines: 28,735.
- Nonblank, non-comment source lines: approximately 24,391.
- Internal import edges found: 205.
- External runtime libraries imported: `pygame`, `PyOpenGL`, and `numpy`.
- Full source compile: **pass**.
- Import smoke test: **91/91 modules pass** in the current environment.
- Engine-to-game dependency-direction check: **pass**.
- Wildcard-import check: **pass**.
- Architecture gate: **fail**, because `tests/` does not exist.

Largest files by physical lines:

| File | Lines | Structural judgment |
| --- | ---: | --- |
| `src/engine/core/compat_shader.py` | 1,439 | Large compatibility subsystem; cohesive enough to keep. |
| `src/engine/textures/texture_utils.py` | 1,197 | Two future seams: loading/registry and procedural generation. |
| `src/game/world/objects/building.py` | 1,067 | Cohesive building geometry algorithm; test before splitting. |
| `src/game/world/objects/goblin.py` | 944 | Multiple behavioral responsibilities; best large-file extraction candidate. |
| `src/game/world/ui/setting_menu.py` | 897 | Repetition points toward declarative settings metadata. |
| `src/game/world/building_pipeline.py` | 896 | Main shared-state construction hotspot. |
| `src/game/world/objects/ground.py` | 811 | Cohesive vectorized terrain builder. |
| `src/engine/rendering/sprite.py` | 770 | Cohesive sprite/batching family. |
| `src/engine/rendering/lighting.py` | 748 | Cohesive shared lighting math and state. |
| `src/engine/core/mesh.py` | 723 | Two related abstractions; acceptable at current scale. |
| `src/game/world/world_runtime.py` | 723 | Several runtime responsibilities over shared scene state. |

## Bottom Line

Do not begin with a broad module-splitting campaign. The repository already has
many modules; its limiting factor is weak contracts and missing verification,
not simply file count. Restore tests and reproducibility first, then tighten
state ownership around the existing collision, entity, lighting, resource, and
rendering seams. That will make future changes safer without destabilizing the
working 3D pipeline.
