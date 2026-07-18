# Current Lighting System: Architecture, Reach, and Deprecation Plan

This document describes the lighting system in the current working tree. It is
intended to be the baseline for deprecating that system and introducing a new
one without accidentally removing world-building, rendering, UI, or audio
behavior that presently depends on lighting data.

## Executive summary

The current lighting system is a hybrid compatibility renderer rather than one
cohesive service. Its visual result is assembled from five mechanisms:

1. A directional sun calculation (`ambient + diffuse * N dot L`).
2. A global scalar brightness/exposure value owned by `Camera` and mirrored into
   `SceneLighting`.
3. X/Z-plane scalar "brightness areas" used for torches and light entering
   openings.
4. Rectangular "covered regions" used to darken building interiors, with
   doorway/window exceptions.
5. Per-object authored vertex shading for interior faces, doors, windows,
   chests, polygons, and other legacy geometry.

Some geometry evaluates these inputs per fragment in a GLSL 1.20 compatibility
shader. Other geometry bakes the same or similar calculations into vertex
colors on the CPU. Several object families use only a subset of the model.
Consequently, changing brightness, opening a door, or losing shader support
does not affect every rendered object in the same way.

`SceneLighting` is the intended shared model, but it is not the sole owner.
Lighting state is duplicated or aliased across `WorldScene`, `Camera`, global
shader module state, ground/road builders, individual objects, and mutable
building-region dictionaries. `StaticLightingController` coordinates these
copies and rebuilds selected static VBOs.

The safest replacement strategy is therefore incremental: first introduce a
typed, revisioned lighting snapshot and explicit receiver/material contracts;
then adapt every current consumer; finally replace the shader and remove the
CPU-baked and compatibility aliases. A direct shader swap would leave terrain,
audio, door behavior, fallback rendering, and settings with hidden dependencies
on the old structures.

## System boundary and terminology

The core reusable model and CPU math live in
`src/engine/rendering/lighting.py`. Game-specific authoring lives in
`src/game/world/world_lighting_plan.py`. Runtime synchronization and static
rebuild policy live in `src/game/world/lighting_controller.py`. The actual GPU
implementation is part of the broader texture/fog/vibrance/shine shader in
`src/engine/core/compat_shader.py`.

The current names can be misleading:

- A **brightness area** is a scalar X/Z influence, not a conventional point
  light. It has no color, height, direction, shadowing, or geometry occlusion.
- A **covered region** is a rectangular indirect-light/indoor mask. It is also
  reused for terrain flattening and audio, so it is not purely render state.
- **Environment lighting** on a `BatchedMesh` means covered-region processing;
  it does not mean all scene lighting.
- **Shader lighting** means that the mesh has the 11-float vertex format and is
  allowed to apply scene brightness, directional light, and optionally covered
  regions in the compatibility shader.
- **Exposure** is the global brightness setting. On shader-lit geometry it is
  represented by `u_base_brightness`; on selected fallback geometry it is a CPU
  rescale of stored vertex RGB.

Contact-shadow decals and goblin shadow quads are adjacent visual systems, but
they are not generated from the sun or local lights. There is no shadow map,
ray-traced visibility, or dynamic light occlusion in the current lighting
system.

The HUD's `test_light` is also outside the lighting system despite its name. It
is a camera-following `WorldSprite` using `LIGHT_TEXTURE_PATH`; the setting only
toggles that sprite's visibility and it emits no brightness area or render
light (`src/game/world/ui/world_hud.py:52`).

## High-level data flow

```text
WorldContent / generated building specs
        |
        v
world_lighting_plan.py
  - covered rectangles + openings
  - doorway/window wall splashes
  - torch brightness areas
        |
        +---------------------------+
        |                           |
        v                           v
SceneLighting                 Camera.brightness_areas
(intended model)              (point queries + cache)
        |
        +--> WorldScene compatibility aliases
        |      sun_pos, sun_direction,
        |      brightness_modifiers, covered_regions
        |
        +--> StaticLightingController
        |      - global shader state
        |      - ground/road/wall/fence rebuilds
        |
        +--> object/builders
        |      - CPU-baked vertex RGB
        |      - normals + receiver RGB for shader path
        |
        +--> sky/cloud renderer
        |
        +--> covered-region audio query
```

At draw time, `WorldRenderer.draw()` asks the controller to synchronize global
texture-lighting uniforms before drawing the ground and world objects. Each
`BatchedMesh` then decides from its vertex layout and flags whether to enable
scene lighting, directional lighting, and covered-region lighting in the shared
shader.

## State model and ownership

### `SceneLighting`

`SceneLighting` is a mutable dataclass defined at
`src/engine/rendering/lighting.py:23`. It contains:

- `sun_position` and `sun_target`; `sun_direction` is derived as target minus
  position and `light_direction` is its inverse.
- `sky_color` and `sun_tint`.
- Directional terms: `ambient=0.72`, `diffuse=0.48`, and
  `max_factor=1.15`.
- `base_brightness`.
- A list of normalized brightness-modifier dictionaries.
- A list of covered-region objects.

`world_setup.setup_graphics()` creates it after the camera and initial
brightness setup (`src/game/world/world_setup.py:164`). The default sun is
offset from world center by `(36000, 22000, 18000)`, targeting the X/Z world
center at Y=0.

### `Camera`

`Camera` predates `SceneLighting` and remains an independent lighting owner
(`src/engine/camera/camera.py:10`). It owns:

- `brightness_default`, the user-facing global brightness.
- `brightness_areas`, a list of dictionaries.
- `_brightness_areas_optimized`, a separately copied representation with
  squared radii.
- A point-query cache bucketed to 10 world units in X/Z.

Point queries multiply the effects of all overlapping areas. `surface_indoor`
can suppress an `indoor_only` area, but point queries do not implement covered
region factors, opening gradients, receiver contribution weights, or
`floor_scale`. Camera brightness is therefore related to, but not equivalent
to, shader/mesh lighting.

### `WorldScene` aliases

`StaticLightingController.sync_aliases()` mirrors the model back to legacy
scene fields (`src/game/world/lighting_controller.py:25`):

- `scene.sun_pos`
- `scene.sun_direction`
- `scene.brightness_modifiers`
- `scene.covered_regions`

Many pipelines still read these fields with `getattr` fallbacks. They are part
of the current compatibility surface even though they are not declared in the
new `WorldBuildState`, `WorldRenderResources`, or `WorldUIState` owners.

### Global shader state

`compat_shader.py` keeps module-global immutable snapshots for lighting, fog,
vibrance, and shine. `set_texture_lighting_state()` converts the mutable scene
model into packed tuples and uploads them when the shader exists
(`src/engine/core/compat_shader.py:1183`). This makes the active lighting state
global to the OpenGL context/module, not explicit per scene or render pass.

### Object-local copies

Ground and road builders retain references to the modifier/region collections
and copies of default brightness and sun direction. Walls, doors, windows,
slabs, chests, and polygons retain `lighting` and/or `sun_direction` fields.
Several of them cache generated meshes using selected lighting values as part
of a cache key.

## Authoring and initialization lifecycle

The initialization order is significant:

1. `WorldScene.initialize_steps()` invokes brightness setup, controller setup,
   graphics setup, and asset loading (`src/game/world/worldscene.py:222`).
2. `setup_brightness_areas()` currently creates no random areas because its
   loop is `range(0)`, but establishes the legacy fields.
3. `setup_graphics()` creates `SceneLighting` and synchronizes aliases.
4. The first world-object step prepares building specs.
5. `apply_building_lighting()` derives covered regions and installs all
   building lights before the terrain mesh is generated
   (`src/game/world/building_pipeline.py:517`).
6. The ground is built using those regions and modifiers.
7. Buildings, torches, doors, and windows are built. Exterior doors are bound
   back to their covered region and doorway brightness modifier.
8. Roads, sprites, fences, and details are built.
9. Initialization records the static brightness and attempts a non-compiling
   uniform sync. The first normal draw can compile the shader.

`world_lighting_plan.py` translates building authoring data into render data:

- Every building footprint becomes a rectangular covered region with factor
  `0.34` (`building_covered_regions()`, line 64).
- The main doorway is an opening whose edge factor interpolates from `0.34`
  closed to `1.0` open.
- Windows create fixed opening gradients with edge factor `0.86`.
- Doorways and windows also create bounded, indoor-only wall-splash brightness
  areas. Their targets are `1.36` and `1.26`; their floor contribution is zero.
- Torches create bounded, indoor-only brightness areas with target `3.4`,
  falloff `2.2`, radius 95–180 units, and floor contribution scale `0.28`
  (`src/game/world/objects/torch.py:23`). The visible torch is a separate
  animated billboard created from the same modifier metadata.

Calling `apply_building_lighting()` more than once is not idempotent. It resets
the categorized scene lists, but it does not remove previously installed areas
from `Camera` or `SceneLighting`; repeat application can accumulate duplicate
light contributions.

## Lighting math

### Directional sunlight

For a normalized surface normal `N` and world-to-light vector `L`:

```text
sun = clamp(ambient + diffuse * max(0, dot(N, L)), 0, max_factor)
```

With defaults this is `clamp(0.72 + 0.48 * NdotL, 0, 1.15)`. The scalar
multiplies RGB; `sun_tint` is not used for ordinary diffuse lighting. It only
tints the sun, clouds, and default shine color.

CPU scalar and vectorized implementations are in
`sunlight_factor_for_normal()` and `apply_directional_sunlight()`
(`src/engine/rendering/lighting.py:240` and `:457`). The shader contains a
separate implementation, so equivalence depends on keeping both copies aligned.

### Brightness areas

For a point within an area's radius, attenuation is:

```text
a = (1 - distance_xz / radius) ** falloff
```

The area describes a target brightness, so its multiplicative effect relative
to base brightness `B0` is:

```text
effect = 1 + (target / B0 - 1) * a
```

Overlapping effects multiply. Bounds can additionally clip the X/Z circle to a
rectangle. `indoor_only` fades or suppresses the effect according to a receiver
factor. On upward-facing surfaces, `floor_scale` mixes the area's target back
toward `B0`.

This is scalar exposure, not emitted radiance. It can brighten texture and
vertex color but cannot produce colored light, height falloff, wall occlusion,
normal-dependent local illumination, or shadows.

### Covered regions and openings

Covered regions take the minimum factor of all containing rectangles. Within a
region, an opening raises the region factor toward its `edge_factor` using a
smooth width fade and an inward-depth fade. Door and window openings use the
same algorithm on CPU and GPU (`covered_region_factor_at()` at
`src/engine/rendering/lighting.py:670`).

The constant `INDOOR_LIGHT_FACTOR=0.34` is also imported directly by building,
wall, door, window, plan, and audio code. It is both a visual value and an
implicit semantic marker for "fully indoors."

### Shader composition order

For shader-lit textured geometry, the fragment shader approximately computes:

```text
receiver = min(max(vertex_rgb), covered_region_factor)
receiver_rgb = vertex_rgb scaled down to receiver
brightness = brightness_areas(world_position, receiver) * exposure
rgb = texture_rgb * receiver_rgb * brightness * directional_sun
rgb += optional shine
rgb = vibrance(rgb)
rgb = fog(rgb)
```

Alpha is texture alpha times vertex alpha. The shader is GLSL 1.20 and uses
compatibility built-ins such as `gl_Vertex`, `gl_Normal`, and fixed-function
matrices.

## GPU path, CPU fallback, and vertex contracts

`BatchedMesh` recognizes these formats (`src/engine/core/mesh.py:57`):

- 6 floats: position + RGB, untextured.
- 8 floats: position + RGB + UV.
- 11 floats: position + RGB + normal + UV.

A textured mesh with an 11-float source format defaults to `shader_lighting`.
An 8-float mesh may have normals added automatically for shine, but the
scene-lighting decision is computed before that conversion; such a mesh can
have shader normals while scene lighting remains disabled. This distinction is
intentional in some legacy objects but is subtle and easy to break.

When the shader is unavailable at build time, ground, road, fence, and wall
builders bake lighting into RGB. This is not a fully equivalent fallback:

- CPU and GPU paths do not use identical receiver information or composition
  order.
- Opening-aware ground subdivision exists specifically for the CPU path.
- Dynamic door lighting cannot cheaply update already baked terrain/walls.
- Different object families query `Camera`, call shared vectorized helpers, or
  use their own face-shade functions.

The shared shader has hard, silently truncating limits:

- 64 brightness areas.
- 32 covered regions.
- 64 total openings.

The fragment shader loops over fixed maxima, including a nested region/opening
loop. CPU helpers have no corresponding caps. A sufficiently large authored
world can therefore render differently on CPU and GPU with no warning.

## Receiver matrix: what is actually lit

| Receiver / pass | Current path | Directional sun | Global brightness | Areas | Covered regions | Dynamic opening response |
|---|---|---:|---:|---:|---:|---:|
| Ground | 11-float shader; CPU-baked fallback | Yes | Yes | Yes | Yes | Shader: yes; fallback: requires rebuild |
| Roads | 11-float shader; CPU-baked fallback | Yes | Yes | Yes | Flag enabled | Only if road overlaps a region |
| Fences | 11-float shader; CPU-baked fallback | Yes | Yes | Yes | Flag enabled | Only if fence overlaps a region |
| Textured wall batches | 11-float shader | Yes | Yes | Yes | Disabled; indoor faces are encoded in vertex RGB | Door wall-splash area can respond |
| Untextured walls | CPU vertex RGB | Yes | Selected exposure refresh | No point sampling in batch path | Authored indoor face factor | No |
| Door slabs/batches | CPU face shades in textured meshes with shader lighting forced off | Yes, baked/rebuilt by slab key | No consistent scene exposure | No | Authored front/back/interior factor | Door itself changes region and splash, but is not lit by them |
| Window slabs/backing | Same legacy slab path | Yes, CPU | No consistent scene exposure | No | Authored interior factor | No |
| Chests | CPU face shading, environment disabled | Yes | No consistent scene exposure | No | Object-specific indoor mode | No |
| Polygons/showcase geometry | CPU face shading, environment disabled | Yes | No consistent scene exposure | No | Object-specific | No |
| Billboard sprites | Shared texture shader when available; CPU point queries otherwise | Yes via fake billboard normal/factor | Yes | GPU: indoor-only areas usually fade out because environment is disabled and receiver is 1; CPU: camera point query applies them | Disabled | No covered/opening dimming |
| Torch billboard | Sprite path | Same as sprites | Same as sprites | Its visual color is independent of emitted scalar light | No | Animation only |
| Decals/contact shadows | Textured mesh path without scene lighting | No | Not managed by static exposure controller | No | No | No |
| Sky clear/fog color | CPU GL state | Indirectly | Yes, clamped 0–1 | No | No | No |
| Sun billboard | Fixed-function color | Position follows sun; tint uses `sun_tint` | Yes | No | No | No |
| Clouds | Fixed-function tint | Uses `sun_tint`, not NdotL | Yes | No | No | No |

This matrix is the most important compatibility reference for migration. A new
system should decide deliberately whether the inconsistencies are bugs to fix
or appearance to preserve temporarily.

## Runtime changes and invalidation

### Global brightness setting

The settings menu delegates to `WorldScene.set_brightness()`
(`src/game/world/ui/setting_menu.py:527`). The controller updates
`Camera.brightness_default` and `SceneLighting.base_brightness`.

If the shader works and areas exist, it uploads shader state and applies CPU
exposure only to untextured ground/fence/road/wall candidates. If the shader is
unavailable, it can rebuild ground, roads, walls, and fences. If there are no
areas, it uses a simpler exposure path. Door/window/chest/polygon/decal batches
are not part of the controller's exposure traversal.

`refresh_static()` disposes and rebuilds the ground VBO, refreshes road meshes
and road batches, rebuilds wall batches, and rebuilds fences
(`src/game/world/lighting_controller.py:109`). This is a large render-resource
operation, not a lightweight lighting update.

### Door opening

Every update while a bound exterior door animates, the door smooths its open
amount and mutates two shared dictionaries (`src/game/world/objects/door.py:333`):

- `region["doorway"]["edge_factor"]` moves from indoor factor to 1.0.
- The doorway wall-splash modifier radius moves from zero to its open radius.

The controller's fast shader key includes the open amount of bound doors, so a
draw uploads fresh packed uniforms. This is special-case invalidation; generic
in-place mutations of area/region contents are not detected because the fast
key otherwise uses only collection identity and length.

There is also a state-splitting defect to preserve or correct consciously:
`SceneLighting.add_brightness_modifier()` normalizes one dictionary for the
scene, while `Camera.add_brightness_area()` copies it into another dictionary
and another optimized record. A door mutates only the scene copy. Camera point
queries never see the dynamic radius. After a global brightness change,
`sync_brightness_modifiers_from_camera()` replaces the scene's modifier list
with new camera-derived dictionaries, orphaning the door's original modifier
reference. The doorway region gradient still works, but the dynamic wall
splash can disappear from subsequent shader state.

### Per-frame synchronization

`WorldRenderer.draw()` calls `sync_uniforms()` every frame before world draws
(`src/game/world/world_renderer.py:134`). A cached key avoids most uploads. The
key includes brightness, sun vector, directional coefficients, collection
identity/length, and bound door open amounts. A more expensive content-based key
exists in the controller but is not used by the draw path.

## Non-render systems touched by lighting data

### Terrain generation

The ground builder uses `covered_regions` to generate terrain flatten pads
under buildings (`src/game/world/objects/ground.py:361`). It also inserts extra
mesh breakpoints around covered-region boundaries and opening gradients for the
CPU lighting fallback. Removing covered regions from lighting without first
extracting a building-footprint/terrain-pad contract will change terrain and
possibly building placement.

### Ambient audio

`world_runtime._ambient_birds_volume()` calls
`covered_region_factor_at()` at the camera X/Z and maps `0.34` to indoor volume
and `1.0` to outdoor volume (`src/game/world/world_runtime.py:66`). Opening a
door therefore raises bird ambience through the same doorway gradient. A new
lighting system should not become the long-term owner of this acoustic state;
the shared concept should be an indoor/portal or environment-volume system.

### World authoring and content generation

`WorldContent` and generated building specs contain doorway, window, and torch
placement. Those are legitimate authored inputs for a new lighting system, but
their current conversion constants and dictionaries are embedded in
`world_lighting_plan.py` and `objects/torch.py`.

### Settings and post-processing

Brightness is presented as a graphics setting but currently changes scene
illumination, sky intensity, and fog/clear color. Fog, vibrance, and shine live
in the same compatibility shader and global state module. They are not fields
of `SceneLighting`, but replacing the shader must preserve or deliberately
relocate them.

### Resource lifecycle and performance

Lighting refresh disposes and recreates OpenGL resources owned by
`WorldRenderResources`. Road batching copies source vertex data from individual
road meshes. Any replacement that removes retained base vertex data or changes
vertex layouts must update batching, exposure fallback, culling bounds, and
resource disposal together.

## File-level touch map

### Core engine

- `src/engine/rendering/lighting.py` — model, normalization, CPU math, covered
  regions/openings, normals, and vertex-format conversion. Primary deprecation
  target.
- `src/engine/core/compat_shader.py` — GLSL implementation, uniform packing,
  global lighting/fog/vibrance/shine state, and hard array limits.
- `src/engine/core/mesh.py` — vertex contracts, shader enable flags, exposure
  fallback, baseline data retention, and batched draw state.
- `src/engine/camera/camera.py` — second brightness-area owner, point queries,
  optimized copies, cache, and global brightness.
- `src/engine/rendering/sprite.py` — billboard sunlight and camera/shader
  brightness behavior.
- `src/engine/rendering/sky_renderer.py` and `cloud_renderer.py` — sun placement,
  sky brightness, and tint.
- `src/engine/rendering/__init__.py` — exports `SceneLighting`.
- `src/game/world/__init__.py` — publicly exports `StaticLightingController`.

### World orchestration and policy

- `src/game/world/world_setup.py` — creates lighting and legacy aliases.
- `src/game/world/worldscene.py` — owns the controller and exposes compatibility
  methods (`set_brightness`, `refresh_static_lighting`, cache invalidation).
- `src/game/world/lighting_controller.py` — synchronization, shader upload,
  invalidation keys, exposure paths, and VBO rebuild policy.
- `src/game/world/world_lighting_plan.py` — converts building specs to indoor
  regions, portals/openings, and local scalar lights.
- `src/game/world/world_renderer.py` — per-frame sync, sky brightness, and passes
  lighting into sprites.
- `src/game/world/world_runtime.py` — indoor/outdoor ambient-audio coupling.
- `src/game/world/building_pipeline.py` — ordering, object wiring, door-region
  binding, and propagation of lighting fields.
- `src/game/world/terrain_pipeline.py`, `road_pipeline.py`, and
  `world_road_planner.py` — pass brightness, lighting, sun, and height resources
  into builders.
- `src/game/world/world_builder.py` — build order on which lighting preparation
  currently depends.
- `src/game/world/world_content.py` — authored/generated source data for
  openings and torches.

### Geometry and entities

- `objects/ground.py` — largest consumer; shader/fallback split, terrain pads,
  opening subdivisions, normals, covered regions, and local areas.
- `objects/road.py` and `objects/fence.py` — shader/fallback static lighting and
  exposure/rebuild hooks.
- `objects/building.py` and `objects/wall_tile.py` — authored indoor face markers,
  normal overrides, wall CPU/shader batching.
- `objects/slab.py`, `door.py`, `window.py`, `chest.py`, and `polygon.py` —
  object-specific CPU face shading and mesh-light cache keys.
- `objects/door.py` — dynamic mutation of portal and doorway-splash state.
- `objects/torch.py` — local-light constants/authoring plus visible sprite.

### UI, audio, and adjacent visuals

- `ui/setting_menu.py` — brightness control; vibrance/fog are adjacent shared
  shader controls.
- `ui/world_hud.py` and `resources/paths.py` — the misleadingly named visual
  `test_light` sprite and torch/light texture paths; these are visual assets,
  not light emitters by themselves.
- `detail_pipeline.py`, `decal.py`, `decal_batch.py`, and `objects/goblin.py` —
  independent static/contact shadow visuals that a new lighting design must
  either keep or replace, even though they do not consume light state today.

## Principal risks and design debt

1. **No single source of truth.** Scene, camera, shader globals, builders, and
   objects all retain lighting state.
2. **Mutable untyped contracts.** Areas, regions, and openings are dictionaries
   with optional keys and object-specific metadata.
3. **Incomplete invalidation.** The draw cache notices list replacement/length
   and door amounts, not arbitrary content mutation. Camera caches and optimized
   copies have separate invalidation rules.
4. **CPU/GPU divergence.** Similar formulas are implemented in Python, NumPy,
   camera point queries, object face-shade methods, and GLSL.
5. **Receiver inconsistency.** Some geometry gets all lighting layers, some
   only sun, some only exposure, and decals neither. Sprites use a fake normal
   and disable covered-region processing.
6. **Silent capacity loss.** GPU area/region/opening arrays truncate with no
   validation or diagnostics.
7. **Lighting owns non-lighting semantics.** Covered regions define terrain pads
   and indoor audio behavior.
8. **Large update cost.** Fallback changes can recreate ground, road, wall, and
   fence resources.
9. **Global render state.** Shader, lighting, fog, vibrance, shine, and exposure
   snapshots are module globals and assume one active context/scene.
10. **Compatibility-profile lock-in.** GLSL 1.20 and fixed-function arrays,
    matrices, fog, and client state make a modern pipeline replacement broader
    than changing one shader.
11. **No automated lighting tests.** The repository currently has no test suite
    covering formulas, packing limits, build invalidation, door transitions,
    receiver coverage, or fallback parity.

## Recommended target boundaries

Before selecting a rendering technique, split the current responsibilities into
explicit domains:

### Environment/space model

Own indoor volumes, building footprints, and portals with stable IDs. Terrain
flattening and ambient audio should consume this model directly. Door state
should update a portal component, not mutate a lighting dictionary.

### Lighting scene

Own a typed, authoritative snapshot of:

- Directional lights.
- Local lights with position (including Y), radius/range, color, intensity,
  attenuation, and optional bounds/channel masks.
- Ambient/environment terms.
- A monotonic revision or explicit dirty channels.

It should not be stored on `Camera`. Camera may query a read-only lighting or
environment service when gameplay/UI needs a sample.

### Receiver/material contract

Every renderable should declare whether and how it receives directional light,
local light, indoor/environment light, shadows, exposure, fog, and shine. Avoid
inferring behavior from vertex width, texture presence, or RGB magnitude.

### Render adapter

Convert the authoritative snapshot into GPU resources for a render context.
Capacity limits should be validated and observable. Per-scene/per-pass state
should replace module globals. CPU fallback, if still required, should consume
the same immutable snapshot and tested reference functions.

### Exposure and post-processing

Separate display exposure/brightness from emitted light intensity. Fog,
vibrance, and shine/material response should have explicit owners even if the
first migration continues using one shader.

## Staged deprecation plan

### Phase 0: establish behavioral baselines

- Add pure unit tests for sun factor, area overlap, bounds, floor scale, indoor
  contribution, covered-region overlap, and opening gradients.
- Add packing tests at 0, limit, and limit+1 lights/regions/openings.
- Add an integration fixture containing exterior ground, an interior floor,
  each wall orientation, a door at closed/half/open, a window, torch, road,
  fence, sprite, slab, decal, and sky.
- Capture screenshots or deterministic RGB samples for shader and forced
  fallback modes at several brightness values.
- Record VBO rebuild counts and uniform-upload counts.

### Phase 1: extract non-render environment semantics

- Introduce typed building footprint/indoor volume/portal data with stable IDs.
- Move terrain flatten-pad generation to building footprints.
- Move bird-volume calculation to indoor volume/portal queries.
- Keep an adapter that derives legacy covered-region dictionaries so visuals do
  not change yet.

### Phase 2: establish one authoritative lighting snapshot

- Introduce typed light and environment-light records plus revision numbers.
- Make authoring produce these records once.
- Make `Camera.brightness_areas` a temporary read-only adapter rather than an
  owner; remove copied optimized dictionaries or rebuild them by snapshot
  revision.
- Replace object-held `lighting`/`sun_direction` copies with a scene handle or
  immutable build snapshot.
- Emit warnings when legacy GPU capacities would truncate data.

### Phase 3: make receiver behavior explicit

- Add render/material flags for directional, local, environment, exposure,
  fog, and shine reception.
- Assign every row in the receiver matrix deliberately.
- Stop using vertex width, texture presence, or vertex RGB as the policy signal.
- Decide which current inconsistencies are fixed immediately and which need a
  legacy visual profile during migration.

### Phase 4: introduce the new renderer behind an adapter

- Feed both old and new GPU implementations from the same typed snapshot.
- Start with ground and static walls because they exercise every layer.
- Migrate roads/fences, then slabs/chests/polygons, then sprites and decals.
- Migrate sky/fog/post effects after ordinary receiver parity is measurable.
- If local lights become true colored 3D lights, calibrate old scalar targets
  (`3.4`, `1.36`, `1.26`) visually rather than treating them as physical units.

### Phase 5: remove baked/static compatibility paths

- Remove lighting-specific ground/road/fence/wall VBO rebuilds once dynamic GPU
  state covers them.
- Remove CPU face-light cache keys and author interior/material data instead.
- Remove shader availability decisions from mesh construction.
- Remove `with_textured_normals()` compatibility conversion once the new vertex
  format is explicit.

### Phase 6: remove legacy API and global state

- Remove `SceneLighting`, legacy `WorldScene` aliases, camera brightness areas,
  `StaticLightingController`, and texture-lighting globals only after searches
  show no adapters or consumers remain.
- Split or replace `compat_shader.py` so fog/vibrance/shine are not accidentally
  deleted with lighting.
- Delete old constants and dictionary normalization after content migration.

## Replacement acceptance criteria

The new system is ready to become the default when:

- One authoritative snapshot drives all render receivers and non-render systems
  consume an extracted environment model.
- Closed, opening, and open doors update light and indoor/audio behavior without
  mutating shared dictionaries or rebuilding static world meshes.
- Global exposure affects the intended receiver set consistently.
- All authored lights are represented or a visible capacity error is reported;
  there is no silent truncation.
- Shader and supported fallback modes use a documented, tested contract.
- Every receiver in the matrix has an explicit migration disposition.
- Ground shape and building placement no longer depend on a lighting-owned
  covered-region structure.
- Resource lifetime, uniform/buffer updates, and scene switching are per-context
  and do not rely on module-global state.
- Automated tests cover formulas, authoring conversion, invalidation, capacity,
  door/portal transitions, and render-level regression fixtures.

## Suggested first implementation slice

The lowest-risk first slice is not a new shader. It is a typed
`EnvironmentVolume`/`Portal` model plus a typed, revisioned `LightingSnapshot`
with adapters back to the current dictionaries and shader uniforms. That slice
fixes ownership and invalidation while keeping current visuals. Once every
consumer reads the snapshot or an explicit adapter, the renderer can be
replaced one receiver family at a time and the old system can be removed with a
finite, searchable dependency list.
