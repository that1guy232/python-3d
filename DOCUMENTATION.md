# Program Documentation

This project is a Pygame/OpenGL first-person 3D world. The code is organized
around a small reusable engine package and focused game packages for the world,
actors, combat, player state and movement, inventory, UI, and resources.

The package boundary is intentional:

- `engine.*` contains reusable runtime services: the main loop, scene protocol,
  camera helpers, rendering primitives, collision helpers, generic UI widgets,
  texture utilities, sound helpers, and engine-level config.
- `game.world.*` is limited to world construction/runtime/rendering, authored
  content, environmental objects, lighting, and world-scene orchestration.
- The sibling `game.actors`, `game.combat`, `game.player`, `game.ui`,
  `game.inventory`, and `game.resources` modules own the other game domains.
- Dependency direction should stay one-way: game code can import engine code,
  but engine code should not import game code.

## Running the Project

The source uses Python 3.10+ syntax and directly depends on Pygame Community
Edition, PyOpenGL, and NumPy. There is no pinned dependency manifest at
present. From the repository root:

```powershell
py -m pip install pygame-ce PyOpenGL numpy
py src/main.py
```

### Controls

| Input | Action |
| --- | --- |
| `W`, `A`, `S`, `D` | Move |
| Mouse | Look |
| Left Shift | Sprint |
| Space | Jump |
| `E` | Interact with the focused door or chest |
| `I` or Tab | Toggle inventory |
| Left mouse | Move/equip items; open the inventory deck viewer; play cards or inspect the battle draw pile |
| `M` | Toggle minimap |
| Escape | Pause/resume; closes an open deck viewer, then inventory |
| `F3` | Toggle performance logging |
| `F4` | Reset the performance timing window |
| `T` on the main menu | Load the standalone Object Test Scene |
| `C` on the main menu | Load Under sea Crypt for testing |
| `L` in the Object Test Scene or crypt | Toggle scene lighting |

The Object Test Scene uses WASD, mouse look, Shift, E, and Escape; it deliberately
omits the normal world's jump, inventory, minimap, battle, and pause systems.

## Start Here

1. `src/main.py` is the entry point. `main()` creates an `Engine` with
   `make_initial_scene()` as the first-scene factory.
2. `make_initial_scene()` creates `game.scenes.main_menu.MainMenuScene`, which is the
   first active scene at boot.
3. The main menu keeps the mouse cursor visible. **Start Game** creates
   `game.world.scene.WorldScene(defer_setup=True)`; **Object Test Scene**
   creates `game.scenes.object_test.ObjectTestScene`; **Under sea Crypt** creates
   `game.levels.crypt.scene.UnderseaCryptScene` for level testing. All targets are wrapped in
   `engine.core.loading_scene.LoadingScene`.
4. `engine.core.engine.Engine` initializes Pygame/OpenGL, attaches the performance
   logger, forwards input to the active scene, calls scene update/render hooks,
   and swaps from the loading scene to its target when loading completes.
5. `engine.core.loading_scene.LoadingScene` advances
   the target's `initialize_steps()` over multiple frames so construction can show
   progress instead of blocking on a blank screen.
6. `game.world.scene.WorldScene` coordinates world lifecycle. `world/runtime`
   owns setup, gameplay updates, entity registration, resource cleanup and the
   build/render/UI state containers. `world/generation/builder.py` coordinates
   construction in the architecture, terrain and roads packages.
   `world/rendering` owns world draw order and game lighting policy; reusable
   rendering and spatial collision indexing belong to `engine`.
   Combat and UI routing remain in `game.combat` and `game.ui`.
7. `game.scenes.object_test.ObjectTestScene` is a separate, flat showroom. It owns
   its camera, movement, collision list, deterministic exhibits, render pipeline,
   interaction scan, and GPU-resource cleanup without using the world scene's
   builders, mutable state containers, renderer, or runtime systems.

## Runtime Flow

### Loading

`WorldScene.initialize_steps()` performs these phases:

- Author initial typed local lights through `world_setup.setup_local_lights()`.
- Create headbob, sway, and player-camera controllers through
  `world_setup.setup_controllers()`.
- Configure fog and scene lighting through `world_setup.setup_graphics()`.
- Load textures, sounds, sky, HUD, and menus through `world_setup.load_assets()`.
- Build terrain, the sampled island shoreline and water, buildings, roads,
  sprites, fences, decals, doors, windows, and other world objects through
  `world_builder.create_world_objects_steps()`, which delegates to focused
  construction pipeline modules.

The shoreline starts on the exact rendered terrain bounds, then uses one
continuous periodic contour to form broad coves and smaller inlets around all
four sides. Its sand rings bridge each sampled edge height to one sea plane,
placed `ISLAND_WATER_DROP` below the lowest edge sample; naturally high edges
therefore form coastal bluffs. The water cutout sits beneath the minimum beach
reach so the irregular waterline cannot expose cracks.

### Updating

Each frame, `Engine.update()` calls `scene.update(dt)`. In the world scene this
delegates to `world_runtime.update()`, which:

- Keeps ambient audio volume in sync with indoor/outdoor lighting regions.
- Initializes the camera height once the terrain sampler exists.
- Updates player rotation, movement, jump state, wall collision, and road speed.
- Updates sway, runtime entities, world sprites, HUD, and headbob.
- Maintains vertical player support against terrain and collision meshes.

### Rendering

Each frame, `Engine.render()` calls `scene.render(...)`. In the world scene this
delegates to `WorldRenderer.render()`, which:

- Reuses the packet backend's directional shadow map until meaningful
  camera/sun or caster movement, and caches up to two point-light cubemaps by
  light and spatially affected caster revision. Dynamic point-shadow refreshes
  are staggered across frames.
- Sets fog, clear color, projection, and camera transforms.
- Draws sky/clouds before world geometry.
- Draws opaque animated water, terrain, the sand beach, fences, decals, walls,
  roads, doors, windows, polygons, entities, sprites, shadows, the minimap, and
  HUD. Water implements the shared exponential fog equation in GLSL so its
  outer edge disappears into the clear-color horizon, with an independent
  outer-ring blend preserving that transition when scene fog is disabled. The
  water plane stays geometrically flat so mesh tessellation cannot show through;
  animated domain-warped cellular caustics, foam, and surface normals are
  evaluated per fragment, with foam following the same contour as the beach.
- Supplies receiver-specific lighting packets to the packet shader and uses
  static VBO batches when available, falling back to immediate rendering for
  objects that need it. Generated goblins reuse a static atlas of low-poly idle
  and walking poses, as do skeletons through their own pose atlas; per-actor GPU transforms avoid rebuilding their geometry as
  they roam.

`BatchedMesh` submits VBO offsets through raw PyOpenGL pointer entry points,
avoiding Python client-array conversion and retention on each draw. These calls
require a bound VBO; client-memory arrays must use the regular wrappers. The
packet shader caches uniform values for its program, so switching receiver
families only uploads changed values. Matrix values are copied into cache keys
to detect in-place edits, and a new shader instance starts with an empty cache.

`engine.rendering.lighting.renderer.SceneLightingRenderer` provides reusable packet
lighting and shadow ownership without a dependency on a world scene. Register
receiver policies and a callable that supplies shadow-casting objects, then call
`begin_frame(camera=..., lighting=...)`. Pass `batch_draw_args()` to mesh batches
or `draw_args(receiver)` to individual objects. `set_enabled()` and `toggle()`
switch lighting without mutating authored lights or rebuilding geometry. Dispose
the renderer before releasing the GL context. `StaticLightingController` extends
this renderer with the world's rebuild and compatibility policies.

`SceneShadowRenderer` keeps a two-cubemap budget by default. The packet shader
supports up to eight point-shadow samplers, bounded by the context's available
texture units. A compact scene can increase `max_point_shadows` to retain all its
lights. The crypt retains all eight torch maps and completes their initial faces
during loading. Traversal then reuses those maps, preventing distant torchlight
from switching off as camera priorities change. Geometry changes still trigger
staged refreshes, so the opening reliquary updates its affected shadows.

The crypt reuses world movement, collision, combat, inventory and UI.
`game/levels/crypt/layout.py` generates a procession through enclosed halls and four side
chapels, joined by seven narrow burial walks. The main route includes three long
passages; their 76–92 unit width and 70 unit barrel-vault crown contrast with the
larger chambers. Chamber dimensions, passage lengths and contents vary by seed.
Pass `world_random_seed` to `UnderseaCryptScene` to reproduce a plan; menu entry
chooses a fresh seed. No maze carving is used.

`game/levels/crypt/geometry.py` owns shared columns and solid arch supports, recessed niches,
arched passage portals, collision shapes and wall-mounted iron torch brackets.
Floor borders and burial inlays partition surfaces instead of overlapping them;
mipmapped masonry reduces shimmer in long views. The level's `water.py` adds translucent
animated water over the disjoint room and passage floors. The level does not
grant a starter loadout or change player movement and encounter defaults.

### Input

`Engine.handle_events()` handles quit/F3/F4 globally, forwards events to the
scene, and sends raw mouse deltas through `scene.apply_mouse_delta()`.
`world_runtime.handle_event()` owns Escape pause/resume, `I`/Tab inventory,
the `M` minimap toggle, `E` focused interaction, and mouse forwarding for
pause, settings, inventory, and battle surfaces through `WorldUIInteractions`.
`PlayerCameraController` owns WASD, sprint, jump, mouse-look targets, and
collision slides.

### Combat cards

The battle deck is assembled from card data owned by equipped items. Weapons
contribute three attacks, boots contribute one or two movement cards, body
armour contributes one or two risk-mitigation cards, and helmets contribute
one unusual-rule card or a passive rule. Legacy items without cards or a
passive retain slot-appropriate fallback cards. Empty slots supply a small base
loadout: one Strike, Quickstep, Brace, and Odd Thought.

The first complete equipment family is the four-piece goblin Scrap set. Each
defeated goblin awards one randomly selected piece rather than the whole set.
Goblin Fists supply Swipe (one damage and one Scrap), Dirty Shiv (spend one
Scrap for three damage), and Junk Bomb (spend one to three Scrap for three,
five, or seven damage). Scamper Boots supply Scamper & Scrounge (draw one and
gain one Scrap). Scrap Armor supplies Patchwork Shell (two base Guard plus one
per Scrap spent), making the same resource an offense-versus-safety decision.
The Thinking Cap contributes no card; its Trash Is Treasure passive grants one
Scrap per unplayed card at turn end. Scrap is capped at three, persists between
turns, resets at battle boundaries, and appears in the battle HUD together with
the active passive-rule label. The three non-weapon pieces temporarily share
the explicit `goblin_set_placeholder` icon entry.

Card origins are reconciled without rebuilding unchanged cards, including
while deck, hand, and discard piles are active. Outside battle, the inventory's
View Deck control opens the complete current loadout. During battle, the Deck
counter instead opens the remaining draw pile. Both modal views close with
their X control, a backdrop click, or Escape and block the underlying
inventory/card/turn input while open. Rules text wraps and scales as needed;
titles and footers scale to remain within their assigned card regions.

### Cleanup

`Engine.run()` disposes the active scene before quitting Pygame. `WorldScene`
delegates cleanup to `WorldResourceDisposer`, which releases OpenGL-backed
meshes, batches, roads, decals, entities, HUD resources, ambient audio, and
render references while the GL context still exists. The scene invalidates its
engine-owned collision index afterward. `WorldRenderer`
separately releases its directional and point-shadow GPU resources.

## Source Map

### Entry and Configuration

| File | Quick docs |
| --- | --- |
| `src/main.py` | Minimal launch file. Builds the first scene and starts `Engine.run()`. |
| `src/game/__init__.py` | Game package marker for actors, combat, inventory, player systems, UI, resources, and worlds. |
| `src/game/scenes/main_menu.py` | Boot menu with entries for the normal world, crypt level testing and the standalone object test scene. |
| `src/game/levels/crypt/scene.py` | Flooded dungeon orchestration, torch lighting, skeleton encounters and interactive reliquary. |
| `src/game/levels/crypt/layout.py` | Seeded enclosed chambers, narrow connecting passages and encounter placement. |
| `src/game/levels/crypt/geometry.py` | Shared masonry, vaults, columns, recessed burial niches, torch mounts and collision shapes. |
| `src/game/levels/crypt/water.py` | Translucent tide with animated caustics, highlights and fog. |
| `src/game/scenes/object_test.py` | Independent flat showroom scene with its own loading, first-person movement, collision, exhibit layout, rendering, interactions, and cleanup. |
| `src/engine/config.py` | Engine display, view, audio mute, and performance defaults. Reads `PY3D_*` environment flags. |
| `src/game/config.py` | Game movement, sky, island shoreline/water, player, and goblin tuning. Re-exports engine defaults for game modules. |

### Core Engine

| File | Quick docs |
| --- | --- |
| `src/engine/core/__init__.py` | Package marker and public core module names. |
| `src/engine/core/engine.py` | Pygame/OpenGL setup, main loop, event forwarding, scene update/render, profiler toggles, and scene disposal. |
| `src/engine/core/loading_scene.py` | Loading/progress scene that advances another scene's setup iterator over frames. |
| `src/engine/core/scene.py` | Base scene protocol: optional camera plus event, render, and dispose hooks. |
| `src/engine/core/performance.py` | Lightweight named-section profiler used by engine and world code. |
| `src/engine/geometry/object3d.py` | Base local-vertex object with cached world vertices, XZ/XYZ bounds, and bounding spheres. |
| `src/engine/rendering/mesh.py` | VBO-backed `BatchedMesh`, static and dynamic vertex uploads, explicit lighting-receiver projection with a legacy fallback, and `GroundHeightSampler`. |
| `src/engine/rendering/shaders/compat.py` | Compatibility-profile shader helpers for textured exposure, lighting, fog, vibrance, shine state, and fixed-capacity overflow warnings. |
| `src/engine/rendering/state.py` | Restores the fixed-function pipeline after programmable packet, shadow, or compatibility draws. |
| `src/engine/rendering/shaders/legacy.py` | Lazy boundary for loading the deprecated compatibility lighting shader only when a legacy draw requests it. |
| `src/engine/core/consts.py` | Shared direction vectors used by older movement/render helpers. |

### Engine Runtime Helpers

| File | Quick docs |
| --- | --- |
| `src/engine/__init__.py` | Engine-layer public exports. |
| `src/engine/core/entity.py` | Runtime entity base class for update/draw/interact/collision hooks. |
| `src/engine/physics/collision.py` | Mesh collision helpers for wall blocking, floor support, and vertical collision resolution. |
| `src/engine/rendering/lighting/receiver.py` | Immutable lighting receiver contract, local-light evaluation policy, sun-shadow reception, material clamps, and compatibility-shader projection. |
| `src/engine/rendering/style.py` | Process-wide fog and vibrance state shared by fixed-function and programmable render paths. |

### Lighting Backends

The packet renderer is the default backend. A legacy backend remains behind
`WorldScene.set_lighting_backend()` for rollback and targeted A/B work while
the lighting implementation is being changed.

### Engine Rendering

| File | Quick docs |
| --- | --- |
| `src/engine/rendering/__init__.py` | Rendering package exports. |
| `src/engine/rendering/lighting/model.py` | Authoritative revisioned scene-lighting owner with immutable typed local-light exposure, CPU lighting math, and textured-normal data. It stores neither local-light dictionaries nor covered regions. |
| `src/engine/rendering/lighting/state.py` | Immutable revisioned lighting snapshots plus typed directional and local scalar-light records. |
| `src/engine/rendering/lighting/adapter.py` | Builds uncapped backend-neutral receiver packets and explicit legacy projections. |
| `src/engine/rendering/shaders/packet.py` | Packet-driven lighting program, receiver-policy binding, local-to-world model transforms, and device-capacity validation. |
| `src/engine/rendering/shaders/source.py` | Standalone packet GLSL, including model-transformed vertices/normals, fragment surface lighting, and vertex-interpolated polygon point queries. |
| `src/engine/rendering/shadows/directional.py` | Directional depth-map ownership, light-space matrices, and the shared opaque/alpha-cutout caster shader. |
| `src/engine/rendering/shadows/point.py` | Point-light depth cubemaps, six face matrices, bindings, and radial-depth caster shader. |
| `src/engine/rendering/lighting/renderer.py` | Reusable scene lighting packets, enable/disable controls and shadow ownership. |
| `src/engine/rendering/shadows/renderer.py` | Scene-independent sun and point-shadow selection, caching and staged refreshes. |
| `src/engine/rendering/shaders/program.py` | Shared GLSL compile/link helper used by packet and shadow programs. |
| `src/engine/rendering/shaders/storage.py` | Persistent RGBA32F record textures with revision-equivalent subimage updates plus fragment/vertex texture capability checks for local-light, region, and portal input. |
| `src/engine/rendering/frame_comparison.py` | Stable-pixel masking and quantitative RGB parity metrics for renderer A/B gates. |
| `src/engine/rendering/lighting/geometry.py` | Resolves explicit packet/legacy vertex representation, retaining lazy shader probing only for standalone compatibility callers. |
| `src/engine/rendering/environment/state.py` | Immutable render-facing environment regions and portals. |
| `src/engine/rendering/sprite.py` | Camera-facing world sprites and animated sprites, with billboard-safe sun-shadow reception policy. |
| `src/engine/rendering/environment/sky.py` | Sky, sun/moon/star, and cloud render orchestration. |
| `src/engine/rendering/environment/clouds.py` | Batched pixel-art cloud rendering. |
| `src/engine/rendering/decal.py` | Terrain-conforming textured decal mesh. |
| `src/engine/rendering/decal_batch.py` | Combines many static decals into one or more VBO batches. |

### Engine Camera

| File | Quick docs |
| --- | --- |
| `src/engine/camera/__init__.py` | Camera package exports. |
| `src/engine/camera/camera.py` | Mutable first-person camera, cached view vectors, typed revision-tagged brightness-query projection, frustum tests, and WASD movement primitive. |
| `src/engine/camera/headbob.py` | Movement/idle headbob state, offsets, and footstep event timing. |
| `src/engine/camera/sway_controller.py` | Mouse-look weapon/HUD sway smoothing. |

### Game World Orchestration

| File | Quick docs |
| --- | --- |
| `src/game/world/__init__.py` | Public world declarations and lazy `WorldScene` entry point. |
| `src/game/world/scene.py` | Scene lifecycle orchestrator. Delegates setup/build/runtime/rendering/combat/entity registration/collision indexing/static lighting/resource cleanup while keeping compatibility state aliases. |
| `src/game/world/runtime/state.py` | Explicit `WorldBuildState`, `WorldRenderResources`, and `WorldUIState` containers plus compatibility alias descriptors. |
| `src/game/world/runtime/setup.py` | Bootstrap helpers for brightness, controllers, fog/lighting, texture/sound loading, sky, HUD, menus, and tunable scene flags. |
| `src/game/world/generation/builder.py` | World construction orchestration: build-step metadata, incremental progress, and compatibility exports for construction helpers. |
| `src/game/world/generation/support.py` | Shared construction-step and disposal helpers used by the construction pipeline modules. |
| `src/game/world/architecture/generation.py` | Building/showcase construction: authored/generated building specs, collision shells, render-only architectural details, torches, doors, windows, showcase polygons, and showcase chest setup. |
| `src/game/world/terrain/generation.py` | Terrain and boundary construction: ground mesh generation, island shoreline/water creation, and fence ring rebuilding. |
| `src/game/world/terrain/boundary.py` | Exact terrain-render bounds, continuous irregular multi-ring beach geometry, a hidden-underlap flat ocean mesh, and the animated cellular GLSL 1.20 water renderer. |
| `src/game/world/roads/generation.py` | Road construction and batching for the main road plus building access roads. |
| `src/game/world/generation/spawns.py` | Procedural nature and species-aware goblin/skeleton placement, registration and pose-atlas batching. |
| `src/game/world/generation/details.py` | Ground-detail/contact-shadow decal batching for procedural nature. |
| `src/game/world/runtime/update.py` | Per-frame runtime helpers: bounds checks, road checks, height queries, entity updates, interaction, pause/inventory input, and mouse delta forwarding. |
| `src/game/world/rendering/renderer.py` | World render pipeline: fog/projection/camera setup, sky, water/terrain/beach ordering, generated-goblin model/contact-shadow passes, object passes, and HUD text/panel delegation. |
| `src/engine/physics/spatial.py` | Spatial collision candidate index for wall and polygon meshes, including dynamic/fallback mesh handling. |
| `src/game/world/runtime/entities.py` | `WorldEntityRegistry` keeps entity/creature lists, immediate draw lists, sprites, and collision meshes synchronized. |
| `src/game/world/environment.py` | Typed indoor volumes and portals, point queries, and legacy covered-region projection during lighting migration. |
| `src/game/world/rendering/lighting.py` | Lighting snapshot adapter, packet-runtime cache, alias-free packet operation, explicit backend transition, exact packet static-geometry validation, rollback-only request guard, and migration diagnostics. |
| `src/game/world/rendering/legacy.py` | Lazily imported rollback-only owner for scene aliases, local-light/covered-region dictionaries, door-region bindings, compatibility uploads, CPU exposure, and ground/road/wall/fence rebuild policy. |
| `src/game/world/rendering/receivers.py` | Complete receiver registry with disjoint packet-runtime and rollback-only CPU/untextured contract sets. |
| `src/game/world/runtime/resources.py` | Scene resource disposer for OpenGL-backed meshes, island water shader/VBO, batches, HUD resources, entities, ambient audio, and cached render/collision references. |
| `src/game/world/content.py` | Declarative scene content. Converts hand-authored or generated building declarations into mutable runtime specs. |
| `src/game/world/architecture/interiors.py` | Shared doorway, window, and interior-feature placement geometry used while authoring buildings. |
| `src/game/world/rendering/plan.py` | Authors typed indoor environment and stable-ID opening-light records; projects compatibility dictionaries only for legacy construction. |
| `src/game/world/roads/planner.py` | Plans non-overlapping driveway routes from buildings to the road network. |
| `src/game/world/generation/placement.py` | Deterministically places procedural nature and billboard props inside world bounds while avoiding roads, buildings, and layer peers. |

### Game World Objects

| File | Quick docs |
| --- | --- |
| `src/game/world/architecture/building.py` | Building authoring helper that turns one rectangular footprint into walls and a flat collision ceiling with door/window openings, plus batched gabled roofs, trim, sills, buttresses, and chimneys. |
| `src/game/world/terrain/tree.py` | Deterministic low-poly trunk, branch, and foliage generation plus spatially chunked static tree batching. |
| `src/game/world/terrain/grass.py` | Deterministic low-poly grass-clump generation plus spatially chunked static batching. |
| `src/game/world/terrain/rock.py` | Deterministic faceted boulder generation plus spatially chunked static batching. |
| `src/engine/geometry/wall.py` | Wall primitive and static wall batching helpers, including texture UVs and lighting data. |
| `src/game/world/terrain/ground.py` | Terrain grid builder that loads heightmaps, flattens typed environment footprints, generates vertex rows, and creates the ground batch. |
| `src/engine/geometry/plane.py` | Legacy flat ground quad primitive. |
| `src/game/world/roads/road.py` | Road mesh builder, road object, point containment, lighting refresh, and road render batching. |
| `src/game/world/props/fence.py` | Textured fence-ring mesh generation around the playable ground bounds. |
| `src/game/world/architecture/door.py` | Interactive door entity with slab rendering, collision, doorway-light binding, and batching. |
| `src/game/world/props/chest.py` | Interactive chest with legacy CPU-baked and explicitly declared dynamic packet receiver representations. |
| `src/game/world/architecture/window.py` | Fixed window entity with slab rendering, wall backing, and batching. |
| `src/game/world/architecture/torch.py` | Building-mounted torch sprite plus typed local-light authoring and legacy modifier projections. |
| `src/engine/geometry/polygon.py` | Extruded polygon primitive plus legacy CPU-baked and camera-independent packet render batching. |
| `src/engine/geometry/slab.py` | Shared geometry, UV, collision, and draw helpers for textured rectangular slab entities. |

### Game Actors, Combat, Player, Inventory, and UI

| File | Quick docs |
| --- | --- |
| `src/game/actors/__init__.py` | Shared actor contract exports without pulling actor implementations into the world-object package. |
| `src/game/actors/creature.py` | Generic combat-creature contract and reusable base state for names, health, attacks, encounter ranges, intents, and creature-owned rewards. |
| `src/game/actors/goblin.py` | Runtime roaming/chasing generated-3D entity, frustum culling, static pose-atlas rendering, batched contact shadows, and random single-piece equipment rewards. |
| `src/game/actors/goblin_model.py` | Deterministic low-poly goblin geometry, visual variants, quantized walking poses, bounds, and shared immutable pose caches. |
| `src/game/actors/model_geometry.py` | Shared deterministic faceted primitives and model transforms. |
| `src/game/actors/skeleton.py` | Skeleton species using shared roaming, collision, combat and batching with its own stats and bone reward. |
| `src/game/actors/skeleton_model.py` | Procedural skull, rib cage and limbs, with four variants and eight cached walking poses. |
| `src/game/combat/__init__.py` | Combat controller and card-loadout exports. |
| `src/game/combat/controller.py` | Creature-agnostic battle controller for entering/leaving combat, announced enemy intent, turn resolution, Guard mitigation, healing, damage rolls, rewards, and battle mouse state. |
| `src/game/combat/cards.py` | Owns deck/hand/discard piles, full-loadout and remaining-draw snapshots, draw-and-reshuffle rules, automatic turn endings, passive equipment rules, Scrap state, and slot-keyed equipment/default card contributions. |
| `src/game/player/__init__.py` | Player controller and stats exports. |
| `src/game/player/controller.py` | Player camera input controller for mouse look, WASD/sprint/jump, terrain support, wall collision, and boundary sliding. |
| `src/game/player/stats.py` | Player health, mana, attributes, critical-hit chance, initial card draw, and attack-damage rolls. |
| `src/game/inventory.py` | Detailed item/card records, backpack and typed equipment slots, constrained movement/swapping, goblin equipment definitions, and timed receipt state. |
| `src/game/ui/__init__.py` | Game UI package exports. |
| `src/game/ui/world_hud.py` | World HUD owner for compass, minimap, held item, sway/headbob offsets, shade overlay, and HUD updates/drawing. |
| `src/game/ui/compass_overlay.py` | Compass overlay using base/needle textures in screen space. |
| `src/game/ui/minimap_overlay.py` | Camera-facing world-space minimap billboard with layered roads, building footprints, goblin markers, and player heading. |
| `src/game/ui/interactions.py` | Screen-space hit boxes and input routing for pause/settings, inventory, and battle surfaces. |
| `src/game/ui/inventory_panel.py` | Inventory overlay, full-deck viewer entry point, item details, drag feedback, equipment backgrounds, and player-stat presentation. |
| `src/game/ui/battle_panel.py` | Battle screen presentation, announced enemy intent/action feedback, active-enemy HP positioning, player-stat text, and active Scrap/passive-rule status. |
| `src/game/ui/battle_overlay.py` | Interactive battle resources, draggable cards, shared full-deck/draw-pile modal, deck/discard counts, and the End Turn control. |
| `src/game/ui/battle_menu.py` | Battle menu model built from generic engine menu items. |
| `src/game/ui/card.py` | Reusable battle-card data, bounded text layout, and mutable/non-mutating OpenGL drawing primitives. |
| `src/game/ui/pause_panel.py` | Screen-space rendering for the active pause or settings menu. |
| `src/game/ui/pause_menu.py` | Pause-menu options and actions. |
| `src/game/ui/setting_menu.py` | Settings menu sliders/toggles that update scene config live. |

### Engine Services and Game Resources

| File | Quick docs |
| --- | --- |
| `src/engine/ui/__init__.py` | Shared UI package namespace. |
| `src/engine/ui/menu.py` | Generic button/slider menu helpers used by pause/settings screens. |
| `src/engine/ui/text_renderer.py` | Pygame font to OpenGL texture text renderer with bounded scaling and measured multiline wrapping. |
| `src/engine/textures/__init__.py` | Generic texture utility package namespace. |
| `src/engine/textures/texture_utils.py` | Texture loading, atlas helpers, texture-size registry, and procedural texture helpers. |
| `src/engine/sound/__init__.py` | Sound package namespace. |
| `src/engine/sound/sound_utils.py` | Pygame mixer wrapper with keyed loading, playback, volume, looping, and optional-file handling. |
| `src/game/resources/__init__.py` | Game resource package namespace. |
| `src/game/resources/paths.py` | Repository-rooted game asset path constants for textures and sounds. |
| `src/game/resources/texture_manager.py` | Loads the texture groups needed by `WorldScene`. |
| `src/game/resources/heightmapgen.py` | Command-line heightmap generator for terrain assets. |

## Common Extension Points

- Add authored buildings through `WorldContent.with_buildings()` and pass the
  result to `WorldScene(world_content=...)`.
- Add loading-time construction to its feature's `generation.py` in
  `world/architecture`, `world/terrain`, or `world/roads`. Cross-feature placement
  and decals live in `world/generation/spawns.py` and `details.py`.
- Add per-frame behavior as an `Entity` when it needs update, interaction, or
  collision hooks.
- Add a combat-capable enemy by subclassing `Creature` (or implementing its
  combat capabilities), then register it with `scene.add_entity()`. Encounter
  detection, battle UI, attacks, and creature-owned rewards are generic.
- Add player movement changes in `game.player.controller` when they affect
  input or collision.
- Add rendering-only world pass changes in `game.world.rendering.renderer`.
- Add screen-space UI input behavior in `game.ui.interactions` and visual
  presentation in the matching panel or overlay module.
- Add minimap markers by registering a layer on `WorldHUD.minimap`; layer draw
  callbacks receive a `MiniMapContext` with world-to-map coordinate helpers.
- Add static lighting rules in `game.world.rendering.plan`, refresh policy in
  `game.world.rendering.lighting`, and reusable lighting math in
  `engine.rendering.lighting.model`.

## Practical Notes

- The repository does not keep a permanent automated test suite or temporary
  development scripts. The Object Test Scene is the supported manual fixture
  for visual and interaction checks of world objects.
- `WorldScene` orchestrates state rather than owning every mutable collection
  directly. New code should use `build_state`, `render_resources`, or `ui_state`;
  the scene-level descriptors are compatibility aliases.
- `WorldRenderResources` keeps both immediate object lists and batched render
  lists. When adding a new object type, decide whether it is static enough to
  batch and which owner controls its lifetime.
- Many rendering objects expose `dispose()`. Call or register disposal before
  replacing VBO-backed batches.
- Collision candidate lookup is indexed in `engine/physics/spatial.py`; invalidate or
  rebuild the index when collision shapes are added or removed after loading.
- Camera brightness is cached by position buckets. World lighting changes must
  go through `SceneLighting` so its revision advances and the controller can
  rebuild the typed camera projection and cache. `local_lights` and
  `brightness_query_lights` are immutable views.
- The profiler can be toggled with F3 and reset with F4 while the game is
  running.
