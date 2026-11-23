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

Choose **New terrain world** from the title screen to start a separate open
earthmoving world with the loader, dozer, truck, asphalt paver and road roller
ready to drive. The whole
32 × 32 model-metre landscape can be excavated: lower the loader's cutting
edge with **Down**, drive forward with **W**, then raise and curl the bucket
to carry the soil. Pour into the truck or onto the ground; use the dozer to
cut, spread spoil, build ramps and backfill holes. There is no replenishing
sand pit or construction-job requirement in this world.

New terrain worlds generate hills, ridges, saddles and low ground throughout
the landscape. Choose your own haul alignment: a loaded truck needs a wider,
smoother and gentler route than the tracked dozer. Cut a bench into a hillside,
reuse the spoil to fill dips, and grade an embankment to improve access.
There are no prescribed crossing locations or road-completion gates.

The dozer pushes a physical load across its blade width. Take shallow cuts,
raise the blade slightly as the load grows, and lift to leave or spread soil.
Deep cuts and heavy blade loads slow or stall it. Reverse with a low blade to
backdrag loose soil; the tracks steer independently and can counter-rotate.

The ground contains stable native soil above finite bedrock and loose soil
that settles after dumping. Cuts and piles change wheel support as well as
the visible mesh. Terrain worlds have their own named save slots, and retain
terrain edits, cargo, falling material and machine positions. **Tab** selects
the next machine; **Shift+Tab** selects the previous one. Press **V** or click
the HUD's **FLEET** button to choose a machine by name. **O** toggles the
overview. The optional **1/2/3** shortcuts select the dozer, loader and truck;
**4/5** select the paver and roller. **W/S** drive,
**A/D** steer, and **Shift** selects faster travel. The paver uses tracked
steering; the roller articulates and its drums rotate with travel. Both new
machines are travel-only for now: they do not pave or compact the ground.
Older terrain saves gain them at their parking spots. **F1** explains the
controls, **F5** saves, and **Esc** opens the normal
pause/settings/save menus. This first terrain implementation is a surface
heightfield: it supports open excavations and ramps, but not tunnels or
overhangs. The existing construction career remains a separate menu choice.

Natural terrain uses a stable base mesh while driving, smooth terrain normals
and cached native heights. Excavation updates small local patches rather than
switching terrain detail around the vehicle.

Start a road with **P** or the **PLAN ROAD** button. A top-down world view lets
you click to place points along your route; the line curves through them and
previews the road's width. Drag points to reshape it, use **-/+** in the toolbar
to change width, and use **Undo**, **Backspace**, or **Ctrl+Z** to undo an edit.
**Delete** or a right-click removes a selected point. Zoom with the wheel and
pan with the middle mouse button. **Enter / Finish plan** confirms the route
and returns to the machine; **Esc** leaves the editor with a draft retained.
Reopen **P** to edit it. The route and width are saved with the terrain world,
and the guide follows the ground while driving.

After confirming a route, **BUILD YOUR ROAD** shows the next ground-preparation
job and a button to drive the recommended dozer or loader. Press **B / SHOW
WORK** for orange areas to strip or cut, blue low spots to fill, and green
prepared areas. Grade stakes mark the fixed design level. Lower the dozer's
blade near the surface to clear trees and brush along the route; native
topsoil does not need stripping. Find your own source of subsoil, sand or
gravel and excavate it with the loader. Haul it directly in the bucket, or
load the dump truck for longer journeys. **Right** dumps the bucket and **Up**
tips the truck bed; creep along the blue route to leave a strip of material.
There is no dedicated quarry or automatic material loading.

Switch to the dozer and keep its blade just above ground with **Up**. **W**
spreads the loose base and **S** backdrags; overlap passes from the middle
toward the edges. A positive blade setting spreads loose fill without cutting
native ground when the chassis pitches on a slope. Only deliberately lowering
below track level selects excavation. The road follows the natural hills and
cross slopes; it no longer demands a trench. Progress measures actual deposited
base material and broad finished height across the planned width. Existing
plans receive this raised base survey on loading, preserving all cuts, piles
and cargo. Paving and compaction follow this earth and aggregate base later.

The dozer strips with the registered soil resistance and rolls its load out
through the open blade ends. Normal topsoil loads can be pushed and turned
aside; the loader is useful for larger piles and hauling spoil farther away.
Deep cuts, bedrock and genuinely excessive loads still limit the machine.

The default Sandbox now uses **bulk sand terrain**, rather than simulating a
particle for every grain. A 2.5 cm terrain grid stores material volume and its
surface height. The loader cuts volume out of the surface, the bucket and
truck carry measured litres, and dumping transfers it back to the terrain.
Steep faces slump to a stable slope without elastic bouncing. Taller piles
can spill over the pit walls. Work scales with terrain area, rather than the
number of grains represented by the material.

Launch with `py src/sandbox_main.py` to open the title screen. Choose **New
career** to name a company in a separate save slot, **Continue** to resume the
last selected career, or **Load career** to browse, rename and delete saves.
Existing W/S, A/D, lift, curl, truck tip, garage, site selection, camera,
recovery and save controls remain available.
New bulk careers begin with **Quarry entrance**: clear the loose spoil blocking
the lane, reuse it to repair the washout, and shape a smooth, truck-wide route
between the entrance posts and the loading apron. The spoil is relocated from
the quarry's existing reserve. There is no delivery-volume quota or prescribed
fill height. Your repaired ground stays in the yard; completion does not spawn
the old office building. Finishing pays $800 for the repair plus the $200 bonus
and opens the bridge and east-road jobs. **F2** toggles the surface survey.
Existing careers retain their original first job, buildings and progress.

The load and site HUD uses litres. Raise a loaded bucket fully, approach the
truck from its side, then tip into the bed; a full bed spills excess material.

The vehicle dashboard shows measured ground speed in model metres per second,
each machine's odometer and trip distance, tool angles, cargo volume and chassis
tilt. The dozer also shows its left and right track speeds. Career saves retain
each machine's counters. Press **T** to reset the current trip meter and **F1**
to show the controls.

Objective information sits in a small card above the lower-left instruments,
leaving the top of the view clear. The card keeps cash and the current task
near the dashboard; full job details remain in **G** and **F1**.

Pouring responds to the actual bucket or bed slope: use a slight tip for a
slow trickle while spreading, and tip farther for faster unloading. Curl back
or lower the truck bed to stop the flow.
The bucket also pushes loose sand while driving, including with a full load.
Tip it down and drive or reverse to spread and backdrag a dumped pile. These
contacts follow the bucket's actual movement and preserve the material volume.
With the cutting lip lowered to the ground, it also gathers the thin loose
layer left by a pass. Raising the bucket leaves that material alone.
The dozer displaces and grades the same volume terrain, and wheels sample its
surface for ground support. Only a bounded set of airborne stream segments is
used during pouring; each carries volume and is retired when it lands.

Bulk careers save separately to
`%LOCALAPPDATA%/Python3D/Sandbox3D/career_bulk.json`. On first launch, an existing
particle career is imported with its company, vehicles, material, cargo and
paid-fill history. The original particle save is preserved. Recovery returns
held volume to the ground, and moving previously paid fill cannot earn twice.
The quarry starts with about 51 litres of sand in a connected stockpile, four
times the original reserve. Older saves receive the extra stock once while
keeping their existing work, cargo and company progress. Later reloads do not
refill sand that has been moved or removed.
The bulk surface has one height per column: it supports digging and piling
loose sand, without underground tunnels or overhangs.

### Pause, settings and save slots

**Escape** opens the pause menu, stopping vehicles and the simulation. If the
fleet menu, garage or controls panel is open, Escape closes that panel first. The pause
menu offers Resume, Save game, Save slots, New career, Settings, Return to main
menu, and Quit. Returning to the main menu, quitting or closing the window does
not save a new snapshot: use **Save game** or **F5** to keep current progress.
Those leave actions discard changes since the last manual save or autosave.
Autosave remains configurable in Settings; Off also disables automatic saves
triggered by gameplay events. Starting a new career creates its own world and
saves the current one first. Deleting a slot asks for confirmation and removes
that career and its backup; it does not reset your settings or other careers.

**R** recovers only the machine you are currently using. Its held material is
left at its previous location; other vehicles, their loads and tool settings
stay in place. Recovery does not reset the world, jobs or vehicle counters.

Settings are available on the title screen and while paused. Display settings
include field of view, brightness, sunlight shadows, HUD visibility and frame
limit (30/60/120/uncapped). Controls and saving settings include camera
sensitivity, inverted orbit Y, autosave (off/15/30/60 seconds), and automatic
pause when the window loses focus. Changes apply immediately and persist
separately from careers. Mouse clicks and keyboard navigation are supported;
use Left/Right to adjust a setting and Escape to go back.

Named careers live in
`%LOCALAPPDATA%/Python3D/Sandbox3D/saves/bulk/` (or `saves/particle/` for the
comparison backend). Each has its own snapshot and last-good backup. Existing
single-file careers appear as **Original career** in the browser. Preferences
live in `preferences.json` next to the save folders. `--save-file PATH` still
opens an explicit career, and `--free-play` opens the unsaved all-equipment lot.

Use `py src/sandbox_main.py --particle-sand` to compare the original grain
backend. Its behavior and the shared vehicle controls are documented below.

### Quarry yard and useful construction

The bulk career takes place in a dedicated dirt-and-gravel quarry yard, with
haul tracks, cut rock faces and several different terrain problems. There are
no parking bays or route gates. The river chasm has a real lower bed; the bridge
adds a supported deck across it. The east haul road builds a sloping ramp up the
raised terrace. The depot drains a flooded basin and adds a supported loading deck, then relocates up to
1,500 grain equivalents of existing quarry sand there, preserving total material
and paid-fill history. The workshop adds a ramp to the service shelf. The
landing is a floating dock that the truck tows into berth; the rest of the
inlet remains open water for the export barge.

Vehicle wheels, loose fill and rendered terrain use the same ground heights.
Unfinished cliffs and water edges stop vehicles through missing or steep ground,
and completed ramps support climbing with visible suspension pitch. **O**
frames the working yard from above. Existing saves retain their project IDs and
balances; machines parked in a newly introduced void return to the quarry with
their cargo. The particle comparison and free-play modes retain the flat lot.

### Career choices, investments and work stages

The connected opening now points from **Quarry entrance** to **Spilled-load
recovery**, then **Clear ramp approach**, **Launch drainage bridge** and the
**Supply depot**. The recovery introduces loading before buying a larger fleet;
clearance pay covers the remaining machines; the bridge opens the north depot's
drainage job, and its retained ballast can later feed the screening plant. Open branches can still be chosen manually,
and resuming a save preserves its chosen destination. The cleared recovery
layby remains usable for loading and turning. See the
[gameplay cohesion map](docs/sandbox-gameplay.md) for roles, material destinations
and the opening economy.

The bulk career's next ramp contract is **Clear ramp approach**. Two heavy old
timber beams obstruct the contractor's assembly area. Use a low loader bucket
or dozer blade to push them out. Contact near a beam's middle mostly slides it;
contact near an end pivots it. Work around the ends and choose where to leave
the timber. The nearby layby is useful storage, but there is no prescribed
parking position: both complete beam footprints must leave the marked approach.
The beams collide with scenery, machines and each other and stay where moved.

When the approach is clear, the finished ramp installs automatically. This pays
$1,400 for the clearance plus the original $350 bonus and opens the landing job
and haul-gearing investment. No sand delivery, grading or hand-built sand ramp
is required. Beam positions survive saving and vehicle recovery, including after
completion. Unfinished older road contracts switch to clearance while retaining
their sand and earnings; already completed roads keep their existing ramp and
surface. The particle comparison retains its original road objective.

**Launch drainage bridge** uses the truck and dozer together. Load at least
**1.20 L** of sand into your truck, then park it stopped with the bed lowered
and all wheels inside the marked anchor bay beside the launch rig. Its ballast
anchors the rig; the cable shows when it is ready. Switch to the dozer, lower
the blade to the yellow pusher bar behind the prefabricated deck, and push
toward the river. The deck advances on rollers until it seats across the gap.
The crossing is a compact single lane. A light launch nose reaches over the
channel while the deck is staged partly over the water, leaving the starter
ditch and pit clear. The 1.20 m pushing run and truck bay sit on their own
riverbank apron.
Moving or tipping the truck pauses the launch without undoing progress.
The truck retains its sand for later work. Seating the deck pays $800 for the
launch plus the original $250 bonus and opens the depot crossing. Unfinished
older bridge jobs switch to this task while keeping their material and earnings;
completed bridges remain built. Older launch progress is carried over by its
completion fraction; previously completed wide bridges keep their original size.

**Drain the depot** replaces the next bulk delivery grid. Drive the loader down
the dry cleanout ramp beside the flooded basin and scoop the finite silt plug
away from the culvert mouth. Reverse up before lifting to load the truck, or
leave the spoil elsewhere clear of the outlet. The plug is relocated from
existing quarry sand once; no material disappears when the outlet opens.

The basin contains a finite volume of water. A clear passage makes the water
level fall and produces visible discharge at the riverbank outfall. Partial
openings drain more slowly; sand dumped back across the throat can stop the
flow without refilling the basin. Once drained, leave the outlet clear and
reverse the loader out of the deck installation area. The crew installs a
working slab on visible supports over the drained excavation, opening the
normal depot stockpile, screening investment and supply route.

This job pays **$2,000 for drainage plus the original $500 bonus**, once. There
is no sand-delivery quota or grading grid. Water progress, cleared material and
below-grade vehicle positions survive saving. Unfinished older depots switch
to drainage while keeping their material and earnings; completed older depots
retain their original reclaimed ground. The particle comparison retains its
original fill objective.

**Machine workshop** now repairs and commissions an existing service bay.
Select it on **G > Fleet & Jobs**; **Road supply** is the separate screened-sand
shipment. The workshop shell is already present, with hollow front and rear
openings and a jammed front shutter. Place the loader's lowered bucket under
the yellow lifting shoe and press **Up**. Its short sliding bracket follows the
boom's arc as the floor raises the shutter. Progress stays raised; at 0.20 m,
the shutter visibly latches open within the baseline loader's reach.

Lower the bucket and drive any owned machine into the clearly outlined service
bay. Entry completes the job immediately: the bay lights and markings turn
green, a check plaque appears, and **WORKSHOP COMPLETE +$3,250** is announced.
There is no hidden future building footprint to clear. The same hollow workshop
stays open and usable, and its completion opens the service-terrace ramp and
loader hydraulics investment. No sand or loose equipment delivery is required.

Shutter, sliding bracket and commissioning progress survive saving. Successful
commissioning pays **$2,600 work pay plus the original $650 bonus**, once.
An already seated unit from the replaced placement task counts as completed
work during migration and receives the remaining payment without a redo.
Already completed older workshops retain their existing progress and appearance.

**Riverside landing** is a towing job. Select the truck, park at the marked
shore tow post and press **E** to connect its rear hitch. The line passes
through the visible fairlead to the floating pontoon. Drive away on dry ground
to pull it **0.75 m** into berth; a taut line limits the truck's motion to the
landing's towing speed. Reversing gives slack. **E** disconnects without losing
the pontoon's position.

At the berth the landing latches, its flags turn green and the truck detaches
automatically. This pays **$1,200 towing pay plus the original $500 bonus**,
once. The permanent 3.00 m by 0.70 m working deck opens the export loading
position; the remaining inlet and the barge's exit channel stay water. There
is no sand-delivery target. Tow position and line state survive saving, and
completed legacy landings retain their existing wide dock and completion.

The starter loader and the $800 truck / $1,400 dozer keep their original access
and prices. After owning all three machines, open **G** and use **Tab** or the
page tabs to choose **Investments**. Reworked bulk construction jobs pay their
fixed work fee and original completion bonus once. Select a destination by
clicking its job card, or cycle open projects with **N**.

| Investment | Requirement | Price | Effect |
| --- | --- | --- | --- |
| Haul gearing (4) | East haul road | $1,800 | Truck travel speed increases 25%; the raised-bed creep limit still applies |
| Depot staging (5) | Supply depot | $2,400 | Relocates up to 4.95 L of existing quarry sand to the north loading pad once |
| Loader hydraulics (6) | Machine workshop | $3,200 | Lift and curl gain 30% speed and strength |
| Depot screening plant (7) | Supply depot | $2,600 | Process raw sand into screened shipments for higher-value buyers |

Each purchase requires the full fleet. The east road opens the riverside landing
job and a hauling investment; the bridge/depot branch improves northern loading
logistics. This lets the player berth the landing before visiting the north yard.
The workshop still
requires both completed branches because it sits across the river. Depot staging
preserves the sand's volume and paid history, leaves held cargo alone, and cannot
be purchased when the quarry reserve is empty. Purchases persist without replaying
deliveries or stacking equipment strength when a career resumes or recovers.

Legacy and particle sites have different placement goals: office and bridge need forgiving broad
coverage; the old depot needs a substantial stockpile across half its grid; the
particle road needs meaningful fill in all four sections along its length; the workshop and
landing need progressively more even distribution. The HUD and job board show
**Deliver fill**, then **Spread / grade** when the volume is sufficient but the
layout still needs work. The bulk ramp contract checks physical timber clearance,
the bridge checks the ballasted launch and seated deck, the bulk depot checks
the open culvert and finite drainage, the workshop checks the latched shutter
and a machine entering its service bay, and the landing checks a physically
towed pontoon seated in its berth. On the selected legacy or particle
unfinished fill site, orange inset outlines
mark areas needing fill and green outlines mark sufficient areas. Completion is
durable even if material is moved later. Older saves retain their completed sites,
balances, cargo, and original observer format.

### Spilled-load recovery and finite customer contracts

Finishing the first project opens **Spilled-load recovery** beside the entrance.
A customer truck is waiting by its lost 2.30 L load. Reclaim the spill with the
loader, raise the bucket and pour into the truck's actual bed. Missed pours land
on the ground and a full bed spills excess material; the former Local fill bay
no longer consumes deliveries. Recover at least 2.10 L, clean up the spill and
clear the ground in front of the cab. Press **E** near the driver to dispatch it
for a one-time fee of up to **$750**. The loaded truck drives out, waiting for
your machines to clear its path. Its cargo, spill and departure survive saving.
The customer does not return or generate another spill.

The completed depot opens **Road supply**, a fixed **4 L** screened-sand
contract paying **$550/L**, up to **$2,200**. This is the north-yard road crew,
separate from the earlier **Clear ramp approach** construction job. Face the
truck east ahead of the crew's moving spreader and tip gently backwards into
its actual low hopper. Creep ahead as it follows and lays a **0.42 m wide**
aggregate strip. Its 1.20 L buffer limits each pour; raw sand and excess spill
beside the hopper. Empty stock or a physical obstruction pauses the rig.
The contract finishes when the accepted material has been laid along the full
strip, leaving ordinary physical sand that can still be reshaped.

The berthed landing opens **Port export**, a fixed **6 L** screened shipment
paying **$800/L**, up to **$4,800**. Reverse along the painted arrows to the
yellow wheel stops and tip into the vessel's two numbered **3 L holds**.
Cargo stays aboard and uneven loading visibly changes the barge's trim.
When both holds are full they turn green and the loaded barge casts off
automatically, carrying the actual sand out through the open water channel.
No pickup/set-down step or manual dispatch is required.

Both contracts close permanently, reject further paid deliveries and drop out
of **N** destination cycling. Older careers preserve earnings and outstanding
quantities: fulfilled buyers stay complete, partially supplied orders require
only their remaining stock, and new actors do not recreate material already
sold. Existing hopper contents, laid material, loaded holds and departure
progress resume from saves. Partial Local fill payments still reduce the
remaining recovery fee.

At the completed depot, buy the screening plant and tip raw sand into the yellow
intake beside the loading pad. Its raw and screened bins each hold **12 L**.
**E** at the plant, or **7** on the Business page while near the depot, processes
**2 L for $160**. Park an empty truck with its bed lowered under the blue loading
gantry and press **E** to collect screened stock. Keep that shipment clean:
mixing raw sand into it or dumping it on the ground makes it bulk stock again.
Screening and loading transfer existing material rather than creating it.

Use **G**, then **Tab**, to open **Business**. **8** orders **10 L for $1,200**
at the quarry; **9** orders **10 L for $1,800** at the completed depot. Both
deliveries create physical stockpiles that must still be loaded and hauled.
Quarry supply is cheaper, while northern supply reduces travel. These expenses,
processing costs and facility investments compete for the same company cash.
These fixed contracts provide additional income alongside construction work.

For a fresh career, the entrance, recovery, clearance, bridge, depot, workshop
and landing pay **$12,000** in total. After buying the truck and dozer ($2,200),
the screening plant ($2,600), and five 2 L processing batches ($800), finishing
both screened contracts adds $7,000 and leaves **$13,400**, before optional
investments or supply purchases. The quarry's roughly 50.9 L reserve covers
the 10 L shipment requirement: 4 L becomes the road strip and 6 L departs in
the barge. Entrance spoil, culvert silt and retained bridge ballast remain
usable stock; workshop repair and harbor towing consume no sand.

### Original particle backend and shared controls

A separate third game lives in `src/sandbox_3d`, alongside `src/game` and
`src/empty_game`. One world unit is one meter. A full-size **42 × 36 m parking
lot** has 2.7 × 5.5 m parking bays, a 6 m central aisle, painted lines, curbs,
wheel stops, sidewalks, and light poles. All vehicles use **1:16 scale**.

The yellow bulldozer's width × height × length follows the **3.312 × 3.172 ×
5.436 m** envelope of a Cat D6 with a 6 SU blade
([Caterpillar technical specifications, page 4](https://www.avesco.ch/fileadmin/avescoproducts/40d591433cf082cc077e9402af499255.pdf)).
Its model envelope is **0.207 × 0.19825 × 0.33975 m**, about **34 cm long**.
The wheel loader uses a generic full-size 2.4 × 3.2 × 6.4 m working envelope,
giving a **0.15 × 0.20 × 0.40 m** model with a lifting boom and curling bucket.
The three-axle dump truck represents a generic 2.88 × 3.2 × 7.2 m machine,
with a **0.18 × 0.20 × 0.45 m** model envelope. Its rear-hinged bed and
separate swinging tailgate use the same solid walls for drawing and sand contacts.
The close follow camera makes the miniature vehicle and its sand work visible;
the overview shows their scale against the full-size lot.

The particle comparison is a **construction career with routes and project unlocks**.
You start with the loader, an empty company account, and one shared walled sand
pit (9,000 grain equivalents, or physical grains in the particle backend).
Haul sand to the amber site grid and meet its fill and spreading goal to construct
useful infrastructure:

| Project | Requirement | What it opens |
| --- | --- | --- |
| Site office | Available at the start | Bridge and haul road contracts |
| Drainage bridge | Site office | A driveable crossing into the depot district and its supply depot job |
| East haul road | Site office | A ramp up to the east terrace, the riverside landing job, haul gearing, and part of the machine workshop requirements |
| Supply depot | Drainage bridge | Reclaimed loading ground, a local sand stockpile, depot staging and part of the machine workshop requirements |
| Machine workshop | Supply depot and east haul road | Service-terrace access and loader hydraulics |
| Riverside landing | East haul road or machine workshop | A supported cargo dock over the inlet |

The bridge adds a supported crossing and the road builds a climbable ramp. Buildings have
distinct finished models, and the garage project board shows available jobs,
locked requirements and each project's benefit. The bridge and road can be
built in either order. Locked sites do not accept paid work until unlocked.

Settled fill earns **$1 per grain equivalent once**, even when it is placed at an unlocked site
other than your selected destination. Spread enough material across a site's
grid to finish its placement goal and earn its construction bonus. Money buys the **$800 dump
truck**, **$1,400 bulldozer**, and subsequent yard investments in the garage. Finished areas and the equipment yard visibly improve as sites
are built. Selecting another destination never moves the vehicles or replaces
the sand. Material remains physical throughout the lot.

Every grain has its own 3D position, gravity, friction, and tool collisions.
Settled grains sleep individually, including inside a connected windrow. They
remain solid contacts and wake for local tool sweeps, new grain impacts, or
removed support. Merged groups retain their neighbour caches as material spreads,
avoiding repeated splitting and rebuilding after the pit has been stirred up.
The world updates at a fixed 60 Hz, independent of rendering. Grain contacts
take smaller steps based on speed and gravity: gravity cannot move a grain by
more than a quarter radius before support is solved again. At the standard
6 mm radius this normally means 120 Hz contact steps, keeping deep piles from
repeatedly compressing and rebounding. Moving tools still use swept collisions.
World setup and support queries run at 60 Hz, keeping their lower overhead
while retaining every grain's volume, identity, cargo membership, and save data.
Moving-wall sweeps, surface friction and projections use cached native kernels.
Batch bounds reject distant walls before grain contact checks. Native support
queries and a grain-count-sized spatial hash keep scattered sand from adding
Python wall scans or a large sparse grid to every physics tick.
The neighbour cache removes common translation and rotation from its reference
frame, keeping loaded vehicles from rebuilding every grain pair each tick.
Grain positions and physical contacts remain independent of the vehicles.
The adjustable blade can push and grade the material. The loader's bucket can
scoop, carry, and pour it above the pavement. GPU instancing draws the grains in one
batch. The HUD shows the grain count and FPS.

Install the Sandbox dependencies once, then launch it from the repository root.
Numba compiles and caches the collision kernels on first use; the NumPy path
remains available when it is not installed.

```powershell
py -m pip install pygame-ce PyOpenGL numpy "numba>=0.64"
py src/sandbox_main.py
```

From inside `src`, the folder and module launch commands also work:

```powershell
py sandbox_3d
# or
py -m sandbox_3d
```

Press **Tab** for the next owned machine or **Shift+Tab** for the previous one.
Press **V** or click the HUD's **FLEET** button to select a machine by name.
The optional **1**, **2** and **3** shortcuts select the bulldozer, loader and
dump truck. Only owned vehicles appear in the work lot. Press **G** for the garage, where
number keys or clickable vehicle cards buy equipment and select owned machines.
All owned vehicles stay in the same parking lot; switching leaves the parked
machines and their sand in place.

| Control | Bulldozer | Wheel loader | Dump truck |
| --- | --- | --- | --- |
| W / S | Both tracks forward / reverse | Drive / reverse | Drive / reverse |
| A / D | Left / right track forward; A+D drives straight | Steer left / right | Steer left / right |
| Up / Down | Raise / lower blade | Raise / lower lift arms | Tip / lower bed |
| Left / Right | Tilt blade side to side | Curl bucket back / tip forward to dump | — |
| Shift | Faster travel | Faster travel | Faster travel |

The loader accelerates gradually and brakes when the throttle is released.
Its drive, lift arms, and bucket curl have finite strength. Sand reaction slows
them under load; an overload stalls the held control instead of forcing the
machine through a packed pile. Release that control to retry, or reverse/lift
away to unload it. The HUD identifies stalled controls.
For a first load, choose **Wheel loader** in **V** (or press **2**), drive into
the open entrance of the sand pit, curl the
bucket back with **Left**, then raise it with **Up**. Reverse away, turn to a
construction site, and hold **Right** to dump. Sand remains free physical grains
outside a container. Grains that enter the bucket pack from its back toward the
mouth, keeping the opening clear until it fills. Captured grains stay secured
to the bucket or truck bed during travel and return to free physics when you
tip to dump. The bucket load appears on the HUD. Ground limits keep the blade
and bucket above the pavement. Solid vehicle parts stop at curbs, wheel stops,
and other vehicles.
Chassis, tires, and tracks also block and displace sand, including on parked
machines. Front-tire contacts follow the steering, and body loads count toward
the loader's drive strength. The body solids leave bucket and truck bed interiors open.
Round tires ride onto supported sand piles; the vehicles rise, pitch and roll
with the surface, and their weight presses on the grains beneath them. Raise
the bucket before driving over a pile to keep its cutting edge clear.
Raise the loaded bucket fully before approaching the truck from the side.
The high-lift boom keeps the cutting lip above the truck rim throughout the
dump; the lowered bucket position and the vehicles' 1:16 scale are unchanged.
Truck loads fill from the floor upward and can form a secured, bounded heap
above the rim. The HUD shows held grains and capacity; once full, additional
sand stays loose and can spill. Choose **Dump truck** in **V** (or press **3**),
drive with the bed lowered, stop, and hold **Up** to
tip the load out of the rear. **Down** lowers the bed; travel with a raised
bed is restricted to a crawl.

Press **N** at any time to cycle available unfinished destinations. The selected
site is amber, other available sites are blue, locked sites are muted, and built
sites turn green. The HUD shows cash, the next vehicle price, site fill, the
project's benefit, and the destination's distance and direction. Every unlocked
site tracks work regardless of selection. Completing a project selects the next
available job. Press **G** for the fleet and project board.

Press **O** to switch between the close follow camera and the whole-lot
overview. **Right-drag** to orbit and tilt, **middle-drag** to pan, and use the
**mouse wheel** to zoom in either view. **Home** recenters the camera behind the
active vehicle, or restores the whole-yard view in overview. Press **R** to
recover machines at the pit while keeping sand, money and site progress in
place. **Escape** pauses; quit through the pause menu.

Shared engine sunlight and ambient lighting illuminate the lot, machines and
sand. **L** toggles lighting for comparison. Vehicles and the pit walls cast cached sun shadows
onto the pavement and sand; the shadow map follows the active work area and
fits the whole lot in overview. Grains stay GPU-instanced and receive shadows
without becoming individual shadow casters. Static lit meshes upload once,
while moving parts use model transforms and each vehicle has one shadow batch.
The three pit walls share one static shadow batch. Moving shadow poses refresh
at most 30 times per second, and each scenery/vehicle group shares its lighting
shader and material setup instead of binding them separately for every part.

The company, owned fleet, vehicle poses, secured cargo, individual sand positions, and all site
progress save automatically, including on exit. Existing careers retain their
money, fleet, paid-fill history and completed projects; the two new projects are
added to their network. **F5** saves manually. New careers use named save slots
from the title or pause menu; writes are atomic and keep a last-good backup.
The old `%LOCALAPPDATA%/Python3D/Sandbox3D/career.json` appears as **Original
career** in the particle browser. **F9** in the garage opens the new-career
screen. `--save-file PATH` uses an explicit separate save.

The original all-equipment lot, with approximately **31,000 grains** and its
delivery challenges, remains available separately:

```powershell
py src/sandbox_main.py --free-play
```

Sand and tools are independent of the vehicle. `GranularSand.update(dt, tools)`
accepts multiple moving `SandTool` wall assemblies. Each tool takes a rigid 4x4
pose, including pitch and roll. Tool dimensions and pose translations are in
meters; `make_bucket()` supplies an open-front, open-top bucket sized for a
1:16 vehicle by default, and `make_bulldozer_blade()` supplies the current blade.
The scene uses `CargoHold` to retain existing grain IDs once they enter a
container. Held grains follow its rigid pose without running loose-grain
physics; dumping releases those same IDs with the container's velocity.
The surrounding sand continues to use the gravity and contact solver.
Sequential grain constraints transmit support through dense piles rather than
softening as the number of contacts grows. Static friction stops small resting
slips, while weak short-range cohesion yields during scooping and dumping.
Contact velocity damping removes the small relative oscillations left by
position corrections, allowing dumped piles to settle and sleep without
damping a whole moving load or unsupported free flight.
Slow, supported contacts also dissipate residual tangential creep. Sleeping
uses sustained motion within a small position window rather than restarting
for every brief correction. Support includes shallow contacts held by static
friction, but still requires a strictly rising path from actual ground or tool
support. Wake impacts use incoming physical velocity; positional overlap
repairs cannot repeatedly wake a quiet pile as if another collision occurred.
Floor contact is applied after tool projection, and overlap repair cannot
create unrestricted launch velocity.
Speed-dependent substeps have a finite budget to prevent a damaged high-speed
state from locking up the simulation.

`SandWorld` manages the larger collection of heaps, waking material for moving
tools, nearby moving grains, or removed tool supports. Its `positions` and
`velocities` arrays retain all grains; sleeping material is still rendered and
can be reached by the same blade or bucket contacts.
Moving secured cargo does not solve its internal grain contacts. Its neighbor
cache rebuilds contacts around nearby free grains, and dropping cargo restores
the released grains' full contact graph. Cargo placement, wall bounds and local
waking use batched native operations when Numba is installed.
Loose sand solves a compact contact patch around disturbed grains and the
neighbors that can participate during that step. The world clock runs at
60 Hz, with gravity- and speed-limited contact substeps. An unchanged contact graph can reuse a cached working
patch while its active grains remain within the previous active seed set.
New contacts, newly disturbed grains, and cargo ownership changes rebuild it.
The patch also shrinks after half its active seeds settle, retiring old work
instead of keeping a nearly full-pit working set indefinitely.
Fast motion uses adaptive subdivisions. A render frame can catch
up at most two material ticks (1/30 second); vehicle motion uses the same bounded time, so
a long stall cannot generate an expanding simulation backlog.

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
    scenes/           Frontend, shared vehicle scene, and bulk career scene
    ui/               Menus, garage/business screens, dashboard, and telemetry
    vehicles/         Models, dimensions, motion, ground/sand contacts, and collision
    sand/             Particle and bulk simulation, cargo, tools, kernels, and drawing
    world/            Worksite terrain, yard construction, roads, and facility geometry
    gameplay/         Career progression, contracts, objectives, and quarry business
    persistence/      Career/bulk saves, named slots, and preferences
    rendering/        Lighting, mesh caching, materials, and road survey drawing
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

Sandbox scenes coordinate the feature packages. Vehicle and sand modules own
physical behavior; world modules build the worksite; gameplay modules judge work
and track progression. UI and rendering present that state, while persistence
owns saves and settings. Import modules from their feature package, such as
`sandbox_3d.vehicles.loader` or `sandbox_3d.sand.bulk_sand`. The existing sandbox
launch commands above continue to use the root entry points.

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
