# python-3d

A first-person 3D world built with Pygame Community Edition, PyOpenGL, and
NumPy. The game includes procedural terrain and world dressing, generated 3D
trees, grass, rocks, goblins and skeletons, buildings and interiors, interactive doors and chests, creature
encounters, inventory and battle UI, and packet-based
lighting with directional and point-light shadows. The terrain is
presented as an island with a procedural sand shoreline and animated stylized
water that fades into the atmospheric horizon.

## Quick start

The source uses Python 3.10+ syntax. From the repository root, install the
runtime dependencies and launch the game:

```powershell
py -m pip install pygame-ce PyOpenGL numpy
py src/main.py
```

There is currently no pinned dependency manifest, so the command above installs
the project's direct third-party dependencies.

## Controls

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

The main menu can be started with the mouse, Enter, keypad Enter, or Space.
Choose **Object Test Scene** (or press `T` on the main menu) to enter a
standalone, walkable showroom containing the core runtime world objects, all
seven tree forms, four grass forms, six rock forms, generated goblin and skeleton models,
and every fence texture variant.
In that scene, use WASD, mouse look, Shift to sprint, E to interact, and Escape
to return to the menu. Press `L` to toggle its shared scene lighting.

Choose **Under sea Crypt** (or press `C` on the main menu) to test the flooded
crypt level. Each visit generates enclosed vaulted chambers connected by narrow,
low burial corridors, with mounted torches, skeletons and an interactive reliquary.
It uses the normal player controls, settings and
starting inventory; entering the level grants no equipment. `L` toggles lighting
for inspection. Use Escape and **Exit to Main Menu** to leave.

## Project structure

The engine supplies reusable capabilities; the game owns content and rules.
Engine modules must be importable without the `game` package. Game modules may
use engine APIs; the engine receives data or callbacks when it needs scene input.

```text
src/
  engine/
    core/             Loop, scenes, entities, loading, profiling
    geometry/         Planes, walls, polygons, slabs, primitive materials
    physics/          Collision queries and spatial index
    rendering/
      lighting/       Light state, receiver contracts, lighting renderer
      shadows/        Point and directional shadow maps and caching
      shaders/        GPU programs, packet storage, compatibility backend
      environment/    Sky, clouds, render-facing environment records
    camera/ ui/ textures/ sound/
  game/
    scenes/           Main menu and object inspection scene
    levels/crypt/     Crypt scene, layout, masonry and water
    world/
      scene.py        Playable world lifecycle
      content.py      World and building declarations
      environment.py Game indoor volumes and portals
      architecture/   Buildings, interiors, doors, windows and torches
      terrain/        Ground, coastline, vegetation and nature LOD
      roads/          Road geometry, planning and construction
      props/          Chests and fences
      generation/     Loading pipeline, placement and world details
      runtime/        Setup, gameplay updates, state, entities and cleanup
      rendering/      World draw order and game lighting policy
    actors/ combat/ player/ ui/ resources/
```

Construction code lives with its feature (`architecture/generation.py`, for
example); `world/generation/builder.py` coordinates those features. Level-specific
content belongs under `game/levels`. Import primitives from `engine.geometry` and
world objects from their feature package. `game.world` exports only `WorldScene`,
`WorldContent`, `BuildingSpec`, and `building`.

See [DOCUMENTATION.md](DOCUMENTATION.md) for runtime flow, subsystem ownership,
extension points, lighting architecture, and a detailed source map.

## Performance logging

Press F3 while the game is running to toggle performance logging in the console.
Press F4 to reset the current timing window.

You can also start with logging enabled:

```powershell
$env:PY3D_PERF_LOG="1"
py src/main.py
```

Useful knobs:

```powershell
$env:PY3D_PERF_INTERVAL="3.0"  # seconds between reports
$env:PY3D_PERF_TOP="14"        # number of slowest sections to print
$env:PY3D_SETUP_TIMING="1"     # print world-loading setup timings
$env:PY3D_RERAISE_SCENE_EXCEPTIONS="1"  # crash on scene hook failures while debugging
```

## Testing policy

The repository does not keep a permanent automated test suite. Add focused
tests only while actively changing the feature they cover, and remove those
temporary tests when that work is complete.

## Declaring world content

Game code can now declare content before the build pipeline turns it into
meshes, collision, lighting, and runtime entities:

```python
from game.world import WorldContent, WorldScene, building

scene = WorldScene(
    world_content=WorldContent.with_buildings(
        [
            building((1200, 0, 900), width=240, depth=180, doorway_side="south"),
        ]
    ),
    defer_setup=True,
)
```
