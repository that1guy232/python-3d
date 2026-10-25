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

## Sandbox 3d

A separate third game lives in `src/sandbox_3d`, alongside `src/game` and
`src/empty_game`. It opens an empty 3D scene with a camera and blank background.
Launch it from the repository root:

```powershell
py src/sandbox_main.py
```

From inside `src`, the folder and module launch commands also work:

```powershell
py sandbox_3d
# or
py -m sandbox_3d
```

Use WASD to move, mouse look, and Shift to sprint. Press **Escape** to quit.

## New game: galaxy map

A separate new game lives in `src/empty_game`, alongside `src/engine` and
the existing `src/game`. Launch it from the repository root:

```powershell
py src/empty_main.py
```

From inside `src`, the folder and module launch commands also work:

```powershell
py empty_game
# or
py -m empty_game
```

`py empty_main.py` remains valid from `src`. All launch commands use the same
game, save file, and command-line options.

The first launch opens a fixed galaxy map with a random spiral galaxy background: a warm
core, blue-gray arms, dust lanes, and stars. An OpenGL fragment shader generates
a 1024 x 1024 texture at startup; displaying the map reuses that texture, so
procedural noise is not recalculated every frame. Later launches resume the last
galaxy and its saved exploration progress. Press **R** to generate a new
galaxy, or **Escape** to exit. Earlier galaxies remain in the save; use their
displayed seed to revisit them:

```powershell
py src/empty_main.py --seed 12345
```

The map highlights **40 named stars** with cyan rings and label callouts.
They are chosen from brighter galaxy regions and spaced apart for readability.
Names and positions are generated from the seed; **R** regenerates them with
the galaxy. Labels are arranged once per map, with side callouts for crowded areas.
Hover over a star's ring or name to highlight its marker, callout, and label.
Above each name, a distinct symbol identifies the stellar body and a globe
symbol accompanies its exact planet count, including zero. A legend explains
the six stellar symbols. Counts use the same generator as the system scene
and are calculated once when the galaxy is generated.
Each system connects to **at most three nearby systems**. The route network is
connected, so every system can be reached through intermediate stops. The white
outer ring and small shader-drawn ship mark your current location; brighter cyan
links show its neighbors. Left-click a ring or name to select a destination.
The navigation panel shows the route's hop count, next stop, distance, and fuel
cost. The next link is gold and the remaining planned route is blue.
Press **Enter** or click **Jump** to travel **one connected hop**. Even when a
distant destination is selected, each jump stops at the next system; it never
skips the intervening systems or continues automatically. A short shader-drawn
flight shows the jump, then opens the system you reached. Press **Escape** to
return to the map and choose your next hop. Clicking your current system or
pressing **Enter** with it selected opens it without using fuel. Other systems
and their planets can only be entered after arriving there.

The ship starts with **20 fuel**; each hop costs fuel according to the link's
distance. Press **F** in the map or cargo screen to consume **1 Hydrogen** and
restore up to **5 fuel**, capped at 20. The control shows the exact gain before
refueling. If you have no Hydrogen and cannot afford any neighboring hop,
**F** offers an **eight-second reserve recharge**, restoring enough fuel to
leave. Jumping and recharging cannot overlap. Generating a new galaxy is blocked
during either operation. Each galaxy keeps its own ship and fuel supply.

A system's background is a procedural
starfield of distant stars, occasional brighter stars, and faint warm/cool dust.
Each star has a consistent background generated once by OpenGL into a texture
up to 1280 pixels across. The view displays the star's name; press **Escape**
to return to the same galaxy map.

Each map has a seeded mix of main-sequence stars, red and blue giants, white
dwarfs, neutron stars, and black holes. Sizes differ substantially: compact
remnants are tiny, while giants occupy much more of the view. These are visual
map scales. The selected object's type is shown beneath its name.

OpenGL shaders generate the bodies directly. Ordinary stars have varied plasma,
rotation, activity, and zero to five independently placed, changing plumes.
White dwarfs have smooth white surfaces; neutron stars have pulsing opposing
jets. Black holes have opaque dark horizons, photon rings, tilted accretion
disks, and stylized lensing of the GPU-generated starfield. No image assets are
used for these bodies. Type, radius, color, orientation, and activity remain
consistent for each system, and animation uses scene time.

Suitable stars also have simple seeded orbitals: three to six planets around
main-sequence stars, and one to three around giants and white dwarfs. Thin blue
ellipses show their inclined orbits. Rocky and banded planets move slowly, are
lit by the central star, and keep their layout on repeat visits. The first pass
does not add planets to black holes or neutron stars. Planet and orbit visuals
are generated with shaders; there are no planet image assets.
Every planet has a generated name, unique within its system. Warm-colored
callouts follow the moving planets and avoid the star, other planets, and the
HUD. Names are stable on repeat visits; adding names does not alter the orbits.
Left-click a planet or its name to open a large procedural portrait showing its
surface type, name, and parent system. **Escape** returns to the same star system, preserving its
orbital phase; press **Escape** again to return to the galaxy map.

Planet surface archetypes include cratered rocky worlds, terrestrial worlds
with continents and seas, ocean-dominated worlds, fractured ice worlds,
banded gas giants, and blue ice giants. Both the orbital miniature and the
close-up use the same seeded surface shader, including clouds and atmosphere
where appropriate. Planet types use a separate random stream, preserving the
existing names, counts, sizes, orbital phases, and speeds. Surfaces are generated
entirely in OpenGL without external planet images.

In the planet view, **left-drag the globe** to rotate it, use the **mouse wheel**
to zoom, and press **R** to reset the view. Zoom is bounded to keep the globe
visible, and dragging stops when the mouse is released or leaves the window.
Press **S** or click **Scan planet** to run a three-second surface scan. A cyan
sweep and progress bar show the scan, then a survey reveals surface or cloud
layers, atmospheric appearance, weather, and notable features appropriate to
that world. Reports describe the generated visuals. Completed surveys and view
settings are saved when leaving a planet or quitting and restored after a restart.
An unfinished scan stops when leaving.

Scans also discover up to three resources with sparse, moderate, or rich abundance,
and sometimes an anomaly. The findings are game content: rocky worlds can reveal
mineral deposits or buried ruins, while giants reveal atmospheric resources and
unusual storm signals. Newly surveyed rocky and terrestrial worlds always offer
Silicates at moderate or rich abundance, alongside their other deposits. This
does not replace Iron, Copper, or other generated resources. Resources and anomalies use separate random streams and
do not change the planet's appearance, name, or orbit. Completed reports and
discoveries are stored as snapshots, so reopening them does not reroll the results.
After scanning, click a resource or anomaly in the discoveries panel, then press
**P** or click **Launch probe**. Each galaxy starts with **five probes**. A launch
uses one probe, and each discovered site can be visited once. A four-second
OpenGL shader animation shows the probe approaching, sampling, and returning.
Resource probes recover 4-7 units from sparse deposits, 9-14 from moderate
deposits, or 18-25 from rich deposits. Investigating an anomaly records an outcome
and returns 1-4 units of survey data. Results are deterministic and saved as
snapshots; revisits cannot collect the same reward again.

Press **V** in any view for the **Resource Finder**, or click **Find materials**
in the cargo workshop. It starts with Silicates and also locates Iron, Copper,
Hydrogen, Rare Metals, Crystal Shards, and Carbon Compounds. Click a material
or use **Left/Right** to change it, select a source with **Up/Down** or the
mouse, and scroll with the wheel. Sources list the planet, system, abundance,
confirmation state, and connected hop count, with nearer systems first.
**Enter / Plot route** selects that system on the map; it still requires the
normal adjacent jumps. Prospecting is free and requires no scanner research.

Orbital readings supply unconfirmed leads; a normal **S** scan on the named
planet confirms the deposits before a **P** probe can collect them. Existing
survey reports remain authoritative, so previously surveyed worlds keep their
original findings. The finder distinguishes sampled sites, renewable extractor
stock, and planets whose extractor slot is already occupied. **DIRECT** sources
provide the selected material; **REFINE** sources provide a raw input that can
be converted into it. The detail panel names the actual deposit, effective
recipe, and both cargo stocks. **X / Open refining** opens that recipe in the
workshop without processing anything. The finder also reports convertible
cargo already aboard, including when there are no eligible planetary leads.
Collected samples
that cannot support further extraction are excluded as exhausted sources.

Press **C** in any exploration view to open ship cargo and see the remaining
probe supply, collected resources, and current flight. One probe can be in flight
per galaxy. It continues working when returning to the system or map, and an
unfinished mission resumes after restarting. Generating another galaxy pauses
the old galaxy's flight until that galaxy is revisited. Survey reports identify
collected deposits and investigated anomalies and retain their outcomes.

The cargo screen also contains a **ship workshop**. Each probe has three specific
components: an Iron hull, Copper electronics, and Silicate sensors. Recipes are:

| Fabrication | Iron | Copper | Silicates |
| --- | --- | --- | --- |
| Standard | 3 | 2 | 1 |
| Precision Fabrication | 2 | 1 | 1 |
| Microfabrication | 1 | 1 | 1 |

The workshop shows **have / need** for each ingredient, highlights shortages,
and displays how many probes the current cargo can produce. Select **1**, **5**,
or **MAX**, then press **B** or click **Build** to fabricate that batch.
MAX selects the largest affordable batch at that moment. The ingredient amounts
show the total needed for the selected batch. A build requires every ingredient;
insufficient stock leaves all cargo and probes unchanged. Each successful batch
deducts its exact inputs and saves once, with a separate receipt for each probe.
Hydrogen, Survey Data, and other resources are not substituted automatically.
Rocky and terrestrial worlds can provide all three ingredients. Every new probe
uses only one Silicate; research further reduces its Iron and Copper costs.
Use the resource finder to identify suitable deposits before committing probes.
Cargo can be scrolled with the mouse wheel as the inventory grows. New probes
can be built while another probe is in flight, using only returned cargo.
Historical crafts retain their original pooled-material costs.

The same workshop has **Probes / Refining** tabs; press **Tab** while cargo is
open to switch. In **Refining**, use **Left/Right** or the recipe arrows to
choose a recipe, select **1**, **5**, or **MAX batches**, then press **X / Enter**
or click **Refine**. Each batch uses **4 units of the named input**. Standard
refining returns **2 units**, Precision Fabrication returns **3**, and
Microfabrication returns **4**. The panel previews the total input spent and
output gained for the selected batch. Basic recipes are available immediately:

| Input | Available output |
| --- | --- |
| Dissolved Metals | Iron or Copper, chosen as separate recipes |
| Buried Metals | Iron or Copper, chosen as separate recipes |
| Water Ice | Hydrogen |
| Methane | Carbon Compounds |
| Seafloor Crystals | Crystal Shards |
| Crystal Shards | Silicates |

Refining uses collected ship cargo. Local extractor storage must be collected
first, and processing never launches a probe. **Find raw material** searches
for the selected recipe's exact input, rather than its finished output.
Fabrication research improves future conversions; recorded batches keep their
original input cost and yield. Sampler research improves resource collection
and extraction without applying an additional refining multiplier. Refining
cannot generate Survey Data, and its recipes contain no conversion cycles.
Converted goods enter the same cargo stacks used by probes, extractor
construction, research, and ship refueling. Each batch action validates all
inputs and output storage room before spending, then saves one receipt.

Mining builds on those same systems. Scan a planet, select one of its resource
deposits, and complete a **resource probe** there to assess it. Then press **E**
or click **Deploy extractor** in that deposit's panel. Each planet supports one
extractor on one chosen deposit; anomalies and Survey Data cannot be mined.
The construction panel shows the cargo you have and the required amounts:

| Extractor construction | Iron | Copper |
| --- | --- | --- |
| Standard | 4 | 2 |
| Precision Fabrication | 3 | 2 |
| Microfabrication | 2 | 1 |

Extractors require no Silicates, so the material being sought cannot block
construction of its own renewable supply.

Sparse deposits produce one unit every **12 seconds**, moderate deposits every
**8 seconds**, and rich deposits every **5 seconds** before upgrades. Improved
Samplers, Advanced Samplers, and Deep Extraction increase existing extractor
production by **25%, 50%, and 75% total**, respectively, alongside their probe
yield bonuses. Research changes future production speed immediately without
retroactively increasing already completed work. Fabrication research reduces
the cost of new extractors; existing construction receipts keep their original
cost. The same ingredients are also needed for replacement probes, making
construction compete with the ship's immediate exploration supply.

Each extractor stores **24 units locally**. Production stops when full and
resumes after collection; full storage cannot bank extra production time.
Visit the planet, select a resource in its discovery panel, and press **G** or
click **Collect** to transfer the installed extractor's stock into ship cargo.
Stored output cannot fund crafting, research, or refueling before collection.
Collected Iron, Copper, and Silicates supply probe and extractor construction;
Hydrogen supplies ship fuel and propulsion research; other research materials
use the existing technology costs. Survey Data still requires anomaly probes.

Press **O** in any view for **Outpost Logistics**. Its outpost list shows systems,
resources, local stock, and production or shuttle status. Select an outpost and
press **Enter** or click **Plot route** to return to the map with that system
selected. Plotting does not move the ship; adjacent jumps and fuel costs still
apply. Scientific survey reports remain available under **L**.
Map nameplates show a cyan outlined storage crate for a
producing system or an amber filled crate when any extractor there is full.
Planet names in the system view show **[M]** or **[FULL]**; cargo shows outpost
totals. The ordinary survey-completion symbols remain independent of mining.

Production continues while playing in that galaxy, across planet, system, map,
and overlay views. Inactive galaxies and closed games are paused. Local stocks,
fractional production, collected totals, and construction costs are saved,
including periodic production checkpoints and immediate build/collect saves.

Outposts can also supply a **home-system depot** using cargo shuttles. The home
system is the ship's saved starting location, identified by name in **O**.
Visit that system, select an outpost in **O → Outposts**, and press **B** or click
**Build & assign** to construct its shuttle. Each outpost supports one shuttle.
Construction uses ship cargo and the fabrication tier installed at that time:

| Shuttle construction | Iron | Copper | Carbon Compounds |
| --- | --- | --- | --- |
| Standard | 8 | 4 | 2 |
| Precision Fabrication | 6 | 3 | 2 |
| Microfabrication | 4 | 2 | 1 |

Shuttles wait for a full **12-unit load**. Before departing, they reserve that
stock at the source and space at the depot, then consume Hydrogen for the entire
round trip. Fuel cost is **1 Hydrogen per four route hops, rounded up**, with a
minimum of one, including local transfers. They follow the existing adjacent
links to the outpost, load the reserved goods on arrival, and return along the
same route. Basic shuttles take **6 seconds per hop**; Probe Drives, Plasma Drives,
and Autonomous Navigation reduce new trips to **4.5**, **3**, and **2.25 seconds
per hop**. Existing flights retain their saved speed. Local transfers still take
one timed pickup leg and one return leg.

In **O**, press **Tab** or click **Home Depot** to manage fuel and deliveries.
Choose **1 / 5 / MAX**, then **H** to transfer exactly that much Hydrogen from
ship cargo into the depot. **G** collects depot goods into ship cargo while at
home. Enabled routes keep a Hydrogen reserve of at least five, or enough for
the most expensive enabled round trip. Pause all routes to reclaim the reserve.
Delivered Hydrogen joins the same pool and can fuel subsequent deliveries.
Each resource has **96 units of depot storage**, including reserved inbound
loads; routes wait when there is insufficient room.

Press **P** on an outpost's logistics page to pause or resume its route. A paused
shuttle finishes its current delivery, keeping cargo and prepaid fuel safe.
Goods in local storage, a shuttle hold, or the depot are unavailable for ship
crafting, refining, research, and refueling until collected into ship cargo.
Manual planet collection excludes stock reserved for a departing shuttle.
Small green outbound and gold loaded-return markers show live shuttles on the
map, separate from the player ship. Trips and every transfer stage are saved;
industry continues across views in the active galaxy and pauses outside it.

Press **T** from the map, system, or planet view for the **research tree**, or
press **R** / click **Open research** while cargo is open. Twelve technologies
are arranged across extraction, propulsion, sensors, analysis, and fabrication
branches, with three progression columns. Arrows show the actual required
technologies, including cross-branch forks and joins. Selecting a node highlights
its connections. Click a technology or use the **arrow keys** to select it.
Its detail panel shows its effect,
prerequisites, and every required resource with **have / cost** amounts.
Press **Enter** or click the purchase button to research it once. Missing
materials or prerequisites block the purchase without spending anything.

| Research | Cost | Effect |
| --- | --- | --- |
| Improved Samplers | 5 Survey Data | +25% probe resource yield, rounded up; +25% extractor production |
| Advanced Samplers | 12 Survey Data, 10 Iron, 6 Copper; requires Improved Samplers | +50% total probe resource yield and extractor production |
| Deep Extraction | 18 Survey Data, 14 Rare Metals, 12 Crystal Shards; requires Advanced Samplers and Microfabrication | +75% total probe resource yield and extractor production |
| Probe Drives | 8 Survey Data, 10 Hydrogen | New probe missions take 3 seconds; new shuttle trips take 4.5 seconds per hop |
| Plasma Drives | 12 Survey Data, 16 Hydrogen, 8 Rare Metals; requires Probe Drives | New probe missions take 2 seconds; new shuttle trips take 3 seconds per hop |
| Autonomous Navigation | 18 Survey Data, 12 Copper, 12 Crystal Shards; requires Plasma Drives and Deep Survey Scanners | New probe missions take 1.5 seconds; new shuttle trips take 2.25 seconds per hop |
| Advanced Scanners | 7 Survey Data, 6 Silicates, 4 Copper | New surveys take 2 seconds instead of 3 |
| Deep Survey Scanners | 12 Survey Data, 12 Silicates, 8 Rare Metals; requires Advanced Scanners | New surveys take 1 second |
| Precision Fabrication | 10 Survey Data, 8 Rare Metals, 6 Carbon Compounds | Probe costs 2 Iron, 1 Copper, 1 Silicates; extractor costs 3 Iron, 2 Copper; shuttle costs 6 Iron, 3 Copper, 2 Carbon Compounds; refining returns 3 per 4 input |
| Microfabrication | 14 Survey Data, 12 Rare Metals, 10 Carbon Compounds; requires Precision Fabrication and Advanced Scanners | Probe costs 1 Iron, 1 Copper, 1 Silicates; extractor costs 2 Iron, 1 Copper; shuttle costs 4 Iron, 2 Copper, 1 Carbon Compounds; refining returns 4 per 4 input |
| Signal Analysis | 9 Survey Data, 8 Crystal Shards | New anomaly missions return 1 additional Survey Data |
| Spectral Analysis | 14 Survey Data, 14 Crystal Shards, 10 Carbon Compounds; requires Signal Analysis and Deep Survey Scanners | New anomaly missions return 2 additional Survey Data total |

Purchases are permanent for their galaxy. All six original technologies keep
their original costs and prerequisites, preserving existing purchases. Later
nodes require every connected parent; they cannot be purchased early even if
their cargo costs are affordable. Probe and fabrication upgrades apply to new launches, scans, and builds: an operation
already underway retains its original duration and recorded reward. Extractor
production uses the current sampler tier. Old crafting
records keep their original material cost. Higher sampler and signal-analysis
tiers replace the earlier total bonus; tier bonuses do not multiply. Research and
crafting costs are recorded together so all spent resources remain spent after
restarting. **T/Escape** closes research; **C** and **L** switch to cargo and the
survey log.

Green dots on the galaxy map mark visited systems with no planets surveyed yet.
A cyan outlined diamond marks a partially surveyed system; a gold dot marks a
fully surveyed system. Visiting a system with no surveyable planets completes
it immediately. Unvisited systems have no progress symbol. These statuses also
apply to existing saved progress and do not depend on probe collection.
The map shows visited and surveyed totals, and
surveyed planets have **[S]** beside their names in the system view. Press **L**
in any exploration view to open the survey log. Click a planet in the list to
read its report and discoveries, use **Up/Down** to select, and use the wheel to
scroll the list. **L** or **Escape** closes the log without leaving the view.

Progress saves automatically on visits, completed scans, and exit to
`%LOCALAPPDATA%/Python3D/GalaxyMap/exploration.json` on Windows. The save contains
the active galaxy seed, star names and positions, an archive of earlier galaxies, visits, planet view
settings, completed surveys, cargo, probe supply, visited sites, and active
missions, research purchases, and a record of materials used in crafting. It
also records the route network, ship location, fuel, completed jumps, Hydrogen
spent on refueling, unfinished jumps or reserve recharges, and mining outposts
with their production, collection, and build records, plus refining batches
with their recipe and fabrication tier snapshots. It also stores shuttle build
records, reserved trips, delivery counters, depot fuel, and collected shipments.
Unfinished ship
operations resume after restarting; inactive galaxies stay paused.
Version 1-11 saves migrate automatically, retaining existing galaxies, surveys,
cargo, flights, old crafting costs, and old extractor construction costs. Existing
survey snapshots are preserved; the common-Silicates generation applies to
unsurveyed worlds. Pre-navigation saves receive a fully fueled ship at a
previously visited system, or a main-sequence system if none have been visited.
Version 1 galaxies receive their initial five probes. A custom file can be selected for a separate
playthrough:

```powershell
py src/empty_main.py --save-file .cache/my-exploration.json --seed 12345
```

Saves use versioned JSON and atomic replacement. If an existing file cannot be
read or has an unsupported format, the game keeps it untouched, shows a save
status message, and allows play in memory. Write failures leave the prior save
intact and are retried at later save milestones.

These visual archetypes are inspired by [NASA's planet families](https://science.nasa.gov/exoplanets/planet-types/),
[Solar System ocean worlds](https://science.nasa.gov/solar-system/ocean-worlds/),
and [Neptune's atmosphere](https://science.nasa.gov/neptune/neptune-facts/).
Ocean and ice surfaces are visual variants, rather than additional formal
exoplanet classifications.

`src/empty_game/galaxy.py` owns texture generation and GPU resources;
`src/empty_game/baked_texture.py` shares the background texture lifecycle;
`src/empty_game/systems.py` owns named stars and label placement;
`src/empty_game/system_badges.py` supplies stellar-type symbols and planet counts;
`src/empty_game/survey_status.py` classifies visited, partial, and complete systems;
`src/empty_game/routes.py` generates the connected route network and finds hop paths;
`src/empty_game/navigation.py` defines saved ship, fuel, and travel records;
`src/empty_game/navigation_panel.py` displays destinations, hop costs, and refueling;
`src/empty_game/ship_renderer.py` draws the ship and jump trail with shaders;
`src/empty_game/starfield.py` generates the selected star's background;
`src/empty_game/stellar_body.py` renders its animated star using shaders only;
`src/empty_game/stellar_profile.py` assigns object types and size/activity ranges;
`src/empty_game/stellar_shaders.py` implements the different stellar appearances;
`src/empty_game/orbitals.py` supplies simple planets, orbit layouts and shaders;
`src/empty_game/planet_labels.py` places moving planet-name callouts;
`src/empty_game/planet_profile.py` defines seeded planet types and palettes;
`src/empty_game/planet_shaders.py` shares surfaces between both planet views;
`src/empty_game/planet_renderer.py` draws large planet portraits;
`src/empty_game/planet_inspection.py` stores session view and completed-scan state;
`src/empty_game/planet_scan.py` generates type-appropriate visual survey details;
`src/empty_game/discoveries.py` generates resources and anomalies;
`src/empty_game/prospecting.py` locates resource leads, respects survey snapshots, and plots routes;
`src/empty_game/exploration.py` owns progress, survey snapshots, and atomic saves;
`src/empty_game/probe_results.py` defines stable probe rewards and narratives;
`src/empty_game/workshop.py` defines exact probe ingredients, batch planning, and legacy recipe rules;
`src/empty_game/refining.py` defines versioned conversion recipes and immutable batch receipts;
`src/empty_game/refining_panel.py` supplies the refining workshop controls and previews;
`src/empty_game/mining.py` defines extractor construction, abundance rates, and local storage;
`src/empty_game/logistics.py` defines freight journeys, shuttle construction, and depot records;
`src/empty_game/logistics_panel.py` supplies route management and home-depot controls;
`src/empty_game/research.py` defines the research catalog, costs, and effects;
`src/empty_game/research_screen.py` draws the connected tree and selected technology requirements;
`src/empty_game/probe_renderer.py` draws the shader-only probe flight;
`src/empty_game/planet_probe_panel.py` handles target selection and launch controls;
`src/empty_game/survey_log.py` displays the survey archive and ship cargo;
`src/empty_game/planet_scene.py` displays the selected world;
`src/empty_game/star_scene.py` is the destination for a selected star;
`src/empty_game/scene.py` displays the map. It uses the shared engine without loading
the existing game's content.
The original game still launches with `py src/main.py`.

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
  sandbox_main.py     Launch the separate Sandbox 3d game
  sandbox_3d/
    __main__.py       Folder and module launch entry point
    scene.py          Empty 3D scene and camera controls
  empty_main.py       Launch the new galaxy map game
  empty_game/
    galaxy.py         Seeded OpenGL galaxy texture generation
    baked_texture.py  Shared procedural background texture resources
    systems.py        Named star generation and label placement
    routes.py         Connected local links with at most three per system
    navigation.py     Saved ship location, fuel, jumps, and recharge records
    navigation_panel.py Route previews, one-hop jumps, and refueling controls
    ship_renderer.py  Shader-drawn ship marker and jump trail
    starfield.py      Seeded stars and faint dust for the star scene
    stellar_body.py   Shader renderer for stars and compact remnants
    stellar_profile.py Seeded object types, sizes, activity and orientation
    stellar_shaders.py Plasma, dwarf, pulsar and black-hole shaders
    orbitals.py       Seeded planets and tilted orbit guides
    planet_labels.py  Moving planet-name labels
    planet_profile.py Planet surface types and seeded palettes
    planet_shaders.py Shared procedural surface and atmosphere shaders
    planet_renderer.py Large shader-only planet portraits
    planet_inspection.py Session view settings and completed scans
    planet_scan.py    Seeded visual survey descriptions
    discoveries.py    Seeded resource deposits and anomalies
    prospecting.py    Free resource leads, confirmation status, and route selection
    exploration.py    Exploration progress and versioned JSON saves
    probe_results.py  Deterministic probe cargo and anomaly outcomes
    workshop.py       Probe component recipes, batch planning, historical costs
    refining.py       Versioned resource conversions and fabrication yield snapshots
    refining_panel.py Refining recipe selection, batch previews, and controls
    mining.py         Extractor recipes, abundance rates, and local storage
    logistics.py      Shuttle trips, fixed loads, and home-depot records
    logistics_panel.py Outpost routes, fuel loading, and depot collection
    research.py       Research catalog, mixed costs, and ship effects
    research_screen.py Branching technology tree and purchase controls
    probe_renderer.py Shader-only probe flight and sampling effect
    planet_probe_panel.py Target selection and probe launch controls
    survey_log.py     Survey archive and cargo shared by all views
    planet_scene.py   Planet portrait and return to the system
    star_scene.py     Star destination and return to the map
    scene.py          Fixed galaxy map view and regeneration controls
    system_badges.py  Stellar-type symbols and exact planet counts
    survey_status.py  Visited, partial, and complete system status
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
