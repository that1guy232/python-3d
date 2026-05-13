# Program Documentation

This project is a Pygame/OpenGL first-person 3D world. The code is organized
around a small reusable engine package and a game package that owns the world,
terrain, buildings, roads, sprites, player movement, UI, and runtime entities.

The package boundary is intentional:

- `engine.*` contains reusable runtime services: the main loop, scene protocol,
  camera helpers, rendering primitives, collision helpers, generic UI widgets,
  texture utilities, sound helpers, and engine-level config.
- `game.*` contains project-specific code: world construction/runtime/rendering,
  authored content, world objects, world UI, game tuning, and asset catalogs.
- Dependency direction should stay one-way: game code can import engine code,
  but engine code should not import game code.

## Start Here

1. `src/main.py` is the entry point. `main()` creates an `Engine` with
   `make_initial_scene()` as the first-scene factory.
2. `make_initial_scene()` creates `game.main_menu.MainMenuScene`, which is the
   first active scene at boot.
3. The main menu keeps the mouse cursor visible and starts the game by creating
   `game.world.worldscene.WorldScene(defer_setup=True)` wrapped in
   `engine.core.loading_scene.LoadingScene`.
4. `engine.core.engine.Engine` initializes Pygame/OpenGL, attaches the performance
   logger, forwards input to the active scene, calls scene update/render hooks,
   and swaps from the loading scene to the world scene when loading completes.
5. `engine.core.loading_scene.LoadingScene` advances
   `WorldScene.initialize_steps()` over multiple frames so world construction can
   show progress instead of blocking on a blank screen.
6. `game.world.worldscene.WorldScene` owns the world state, but delegates most work:
   setup to `world_setup.py`, object construction to `world_builder.py`,
   runtime behavior to `world_runtime.py`, road planning to
   `world_road_planner.py`, and drawing to `world_renderer.py`.

## Runtime Flow

### Loading

`WorldScene.initialize_steps()` performs these phases:

- Setup brightness areas through `world_setup.setup_brightness_areas()`.
- Create headbob, sway, and player-camera controllers through
  `world_setup.setup_controllers()`.
- Configure fog and scene lighting through `world_setup.setup_graphics()`.
- Load textures, sounds, sky, HUD, and menus through `world_setup.load_assets()`.
- Build terrain, buildings, roads, sprites, fences, decals, doors, windows, and
  other world objects through `world_builder.create_world_objects_steps()`.

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

- Sets fog, clear color, projection, and camera transforms.
- Draws sky/clouds before world geometry.
- Draws terrain, fences, decals, walls, roads, doors, windows, polygons,
  entities, sprites, shadows, the minimap, and HUD.
- Uses static VBO batches when available, falling back to immediate rendering
  for objects that need it.

### Input

`Engine.handle_events()` handles quit/F3/F4 globally, forwards events to the
scene, and sends raw mouse deltas through `scene.apply_mouse_delta()`.
`world_runtime.handle_event()` owns pause/inventory toggles, the `M` minimap
toggle, pause-menu mouse input, and focused entity interaction.
`PlayerCameraController` owns WASD, sprint, jump, mouse-look targets, and
collision slides.

### Cleanup

`Engine.run()` disposes the active scene before quitting Pygame. `WorldScene`
releases OpenGL-backed meshes, batches, roads, decals, entities, and cached
collision data while the GL context still exists.

## Source Map

### Entry and Configuration

| File | Quick docs |
| --- | --- |
| `src/main.py` | Minimal launch file. Builds the first scene and starts `Engine.run()`. |
| `src/game/__init__.py` | Game package marker for project-specific world code and resources. |
| `src/game/main_menu.py` | Boot menu scene with a visible cursor and Start Game button that transitions into the loading/world flow. |
| `src/engine/config.py` | Engine display, view, audio mute, and performance defaults. Reads `PY3D_*` environment flags. |
| `src/game/config.py` | Game movement, sky, player, and goblin tuning. Re-exports engine defaults for game modules. |

### Core Engine

| File | Quick docs |
| --- | --- |
| `src/engine/core/__init__.py` | Package marker and public core module names. |
| `src/engine/core/engine.py` | Pygame/OpenGL setup, main loop, event forwarding, scene update/render, profiler toggles, and scene disposal. |
| `src/engine/core/loading_scene.py` | Loading/progress scene that advances another scene's setup iterator over frames. |
| `src/engine/core/scene.py` | Base scene protocol: optional camera plus event, render, and dispose hooks. |
| `src/engine/core/performance.py` | Lightweight named-section profiler used by engine and world code. |
| `src/engine/core/object3d.py` | Base local-vertex object with cached world vertices, XZ/XYZ bounds, and bounding spheres. |
| `src/engine/core/mesh.py` | VBO-backed `BatchedMesh` plus `GroundHeightSampler` for terrain height lookups. |
| `src/engine/core/compat_shader.py` | Compatibility-profile shader helpers for textured exposure, lighting, fog, vibrance, and shine state. |
| `src/engine/core/consts.py` | Shared direction vectors used by older movement/render helpers. |

### Engine Runtime Helpers

| File | Quick docs |
| --- | --- |
| `src/engine/__init__.py` | Engine-layer public exports. |
| `src/engine/entity.py` | Runtime entity base class for update/draw/interact/collision hooks. |
| `src/engine/collision.py` | Mesh collision helpers for wall blocking, floor support, and vertical collision resolution. |

### Engine Rendering

| File | Quick docs |
| --- | --- |
| `src/engine/rendering/__init__.py` | Rendering package exports. |
| `src/engine/rendering/lighting.py` | Shared lighting model for sunlight, indoor regions, brightness modifiers, and textured-normal data. |
| `src/engine/rendering/sprite.py` | Camera-facing world sprites and animated sprites. |
| `src/engine/rendering/sky_renderer.py` | Sky, sun/moon/star, and cloud render orchestration. |
| `src/engine/rendering/cloud_renderer.py` | Batched pixel-art cloud rendering. |
| `src/engine/rendering/decal.py` | Terrain-conforming textured decal mesh. |
| `src/engine/rendering/decal_batch.py` | Combines many static decals into one or more VBO batches. |

### Engine Camera

| File | Quick docs |
| --- | --- |
| `src/engine/camera/__init__.py` | Camera package exports. |
| `src/engine/camera/camera.py` | Mutable first-person camera, cached view vectors, brightness sampling, frustum tests, and WASD movement primitive. |
| `src/engine/camera/headbob.py` | Movement/idle headbob state, offsets, and footstep event timing. |
| `src/engine/camera/sway_controller.py` | Mouse-look weapon/HUD sway smoothing. |

### Game World Orchestration

| File | Quick docs |
| --- | --- |
| `src/game/world/__init__.py` | Lazy public exports for world objects, `WorldScene`, `WorldContent`, and content helpers. |
| `src/game/world/worldscene.py` | Central scene state owner. Delegates setup/build/runtime/rendering while keeping entity lists, batches, collision indexes, lighting aliases, and cleanup. |
| `src/game/world/world_setup.py` | Bootstrap helpers for brightness, controllers, fog/lighting, texture/sound loading, sky, HUD, menus, and tunable scene flags. |
| `src/game/world/world_builder.py` | World construction pipeline: content, terrain, buildings, lighting, roads, sprites, goblins, fences, decals, doors, windows, and render batches. |
| `src/game/world/world_runtime.py` | Per-frame runtime helpers: bounds checks, road checks, collision spatial index, height queries, entity updates, interaction, pause/inventory input, and mouse delta forwarding. |
| `src/game/world/world_renderer.py` | World render pipeline: fog/projection/camera setup, sky, terrain, object passes, HUD text, pause menu, and menu interaction forwarding. |
| `src/game/world/world_content.py` | Declarative scene content. Converts hand-authored or generated building declarations into mutable runtime specs. |
| `src/game/world/world_lighting_plan.py` | Builds indoor covered regions and doorway/window/torch brightness modifiers from building specs. |
| `src/game/world/world_road_planner.py` | Plans non-overlapping driveway routes from buildings to the road network. |
| `src/game/world/world_spawner.py` | Spawns static billboard sprites inside world bounds while avoiding roads, buildings, and other sprites. |
| `src/game/world/player_controller.py` | Player camera input controller for mouse look, WASD/sprint/jump, terrain support, wall collision, and boundary sliding. |

### Game World Objects

| File | Quick docs |
| --- | --- |
| `src/game/world/objects/__init__.py` | Convenient exports for common world object classes. |
| `src/game/world/objects/building.py` | Building authoring helper that turns one rectangular footprint into wall/roof `WallTile` pieces with door/window openings. |
| `src/game/world/objects/wall_tile.py` | Wall primitive and static wall batching helpers, including texture UVs and lighting data. |
| `src/game/world/objects/ground.py` | Terrain grid builder that loads heightmaps, flattens building pads, generates vertex rows, and creates the ground batch. |
| `src/game/world/objects/ground_tile.py` | Legacy flat ground quad primitive. |
| `src/game/world/objects/road.py` | Road mesh builder, road object, point containment, lighting refresh, and road render batching. |
| `src/game/world/objects/fence.py` | Textured fence-ring mesh generation around the playable ground bounds. |
| `src/game/world/objects/door.py` | Interactive door entity with slab rendering, collision, doorway-light binding, and batching. |
| `src/game/world/objects/chest.py` | Interactive opening chest entity with textured box rendering and collision. |
| `src/game/world/objects/window.py` | Fixed window entity with slab rendering, wall backing, and batching. |
| `src/game/world/objects/torch.py` | Building-mounted torch sprite plus helpers that derive torch light locations and brightness modifiers. |
| `src/game/world/objects/goblin.py` | Runtime roaming/chasing sprite entity with directional animation frames and batched shadows. |
| `src/game/world/objects/polygon.py` | Extruded arbitrary polygon primitive plus static polygon render batching. |
| `src/game/world/objects/slab.py` | Shared geometry, UV, collision, and draw helpers for textured rectangular slab entities. |

### Game World UI

| File | Quick docs |
| --- | --- |
| `src/game/world/ui/__init__.py` | World UI package exports. |
| `src/game/world/ui/world_hud.py` | World HUD owner for compass, minimap, held item, sway/headbob offsets, shade overlay, and HUD updates/drawing. |
| `src/game/world/ui/compass_overlay.py` | Compass overlay using base/needle textures in screen space. |
| `src/game/world/ui/minimap_overlay.py` | Screen-space minimap with layered roads, building footprints, goblin markers, and player heading. |
| `src/game/world/ui/pause_menu.py` | Pause-menu options and actions. |
| `src/game/world/ui/setting_menu.py` | Settings menu sliders/toggles that update scene config live. |

### Engine Services and Game Resources

| File | Quick docs |
| --- | --- |
| `src/engine/ui/__init__.py` | Shared UI package namespace. |
| `src/engine/ui/menu.py` | Generic button/slider menu helpers used by pause/settings screens. |
| `src/engine/ui/text_renderer.py` | Pygame font to OpenGL texture text renderer for HUD/menu labels. |
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
- Add new world object construction in `world_builder.py` when the object is
  static or created during loading.
- Add per-frame behavior as an `Entity` when it needs update, interaction, or
  collision hooks.
- Add player movement changes in `game.world.player_controller` when they affect
  input or collision.
- Add rendering-only world pass changes in `game.world.world_renderer`.
- Add minimap markers by registering a layer on `WorldHUD.minimap`; layer draw
  callbacks receive a `MiniMapContext` with world-to-map coordinate helpers.
- Add static lighting rules in `world_lighting_plan.py` and reusable lighting
  math in `engine.rendering.lighting`.

## Practical Notes

- `WorldScene` keeps both immediate object lists and batched render lists.
  When adding a new object type, decide whether it is static enough to batch.
- Many rendering objects expose `dispose()`. Call or register disposal before
  replacing VBO-backed batches.
- Collision candidate lookup is indexed in `world_runtime.py`; invalidate or
  rebuild the index when collision shapes are added or removed after loading.
- Camera brightness is cached by position buckets. Clear or invalidate caches
  when brightness areas change.
- The profiler can be toggled with F3 and reset with F4 while the game is
  running.
