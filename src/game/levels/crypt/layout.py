"""Seeded burial chambers connected by narrow, low processional passages."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class CryptRoom:
    name: str
    bounds: tuple[float, float, float, float]
    cells: tuple[tuple[int, int], ...]
    kind: str

    def contains(self, x, z):
        x0, x1, z0, z1 = self.bounds
        return x0 <= x <= x1 and z0 <= z <= z1

    def get_bounding_box(self):
        return self.bounds

    @property
    def along_z(self):
        x0, x1, z0, z1 = self.bounds
        return z1 - z0 > x1 - x0


@dataclass(frozen=True)
class CryptLayout:
    seed: int
    bay_width: int
    bay_depth: int
    nave_bays: int
    nave_aisles: int
    rooms: tuple[CryptRoom, ...]
    cells: frozenset[tuple[int, int]]
    tombs: tuple[tuple[float, float], ...]
    guardians: tuple[tuple[float, float], ...]

    def cell_bounds(self, cell):
        i, j = cell
        return ((i - .5) * self.bay_width, (i + .5) * self.bay_width,
                200 - (j + 1) * self.bay_depth, 200 - j * self.bay_depth)

    def cell_center(self, cell):
        i, j = cell
        return (i * self.bay_width, 200 - (j + .5) * self.bay_depth)

    @property
    def bounds(self):
        values = [room.bounds for room in self.rooms]
        return (min(b[0] for b in values), max(b[1] for b in values),
                min(b[2] for b in values), max(b[3] for b in values))

    @property
    def entrance(self):
        return (0., 200 - 58.)

    @property
    def reliquary(self):
        room = next(r for r in self.rooms if r.kind == "reliquary")
        return (0., room.bounds[2] + 74)

    @property
    def passages(self):
        return tuple(r for r in self.rooms if r.kind == "passage")


def generate_crypt_layout(seed: int) -> CryptLayout:
    """A clear procession through enclosed halls, with optional burial branches.

    Chambers own disjoint grid bays; passages occupy the gaps between them.
    Every route into another chamber passes through a real length of corridor.
    """
    rng = random.Random(seed)
    width = rng.choice((128, 144, 160))
    depth = rng.choice((160, 176, 192))
    rooms = []
    cells = set()

    def room(name, kind, i0, i1, j0, j1):
        occupied = tuple((i, j) for j in range(j0, j1) for i in range(i0, i1))
        if cells.intersection(occupied):
            raise ValueError("Crypt room plans must not overlap")
        cells.update(occupied)
        rooms.append(CryptRoom(name,
            ((i0 - .5) * width, (i1 - .5) * width, 200 - j1 * depth, 200 - j0 * depth),
            occupied, kind))
        return rooms[-1]

    def passage(name, bounds):
        rooms.append(CryptRoom(name, bounds, (), "passage"))

    previous = room("Drowned Vestibule", "entrance", 0, 1, 0, 1)
    row = 1
    main_halls = []
    for name, kind, passage_name in (
        ("Hall of the Departed", "nave", "Passage of Whispers"),
        ("Ossuary of the Drowned", "ossuary", "Low Funeral Walk"),
        ("Deep Reliquary", "reliquary", "Procession of Silence"),
    ):
        gap = rng.randint(2, 3)
        half_passage = rng.choice((38., 42., 46.))
        start = row + gap
        row = start + rng.randint(2, 3)
        hall = room(name, kind, -1, 2, start, row)
        passage(passage_name, (-half_passage, half_passage, hall.bounds[3], previous.bounds[2]))
        main_halls.append((hall, start, row))
        previous = hall

    # Four side chapels, separated from the large halls by enclosed burial walks.
    names = (("Chapel of Tides", "Ancestors' Rest"),
             ("Chapel of Remembrance", "Mariners' Rest"))
    for hall_index, (hall, start, end) in enumerate(main_halls[:2]):
        for side, name in zip((-1, 1), names[hall_index]):
            gap = rng.randint(1, 2)
            span = rng.randint(1, 2)
            door_row = rng.randrange(start, end)
            j0 = max(start, door_row - rng.randint(0, 1))
            j1 = min(end, max(door_row + 1, j0 + rng.randint(1, 2)))
            i0, i1 = ((2 + gap, 2 + gap + span) if side > 0
                      else (-1 - gap - span, -1 - gap))
            chapel = room(name, "chapel", i0, i1, j0, j1)
            cz = 200 - (door_row + .5) * depth
            half_passage = rng.choice((38., 42.))
            a, b = ((hall.bounds[1], chapel.bounds[0]) if side > 0
                    else (chapel.bounds[1], hall.bounds[0]))
            passage(name + " Burial Walk", (a, b, cz-half_passage, cz+half_passage))

    tombs = []
    guardians = []
    for cell in sorted(cells):
        i, j = cell
        cx, cz = i * width, 200 - (j + .5) * depth
        # Reserve a continuous processional aisle from the entrance to the altar.
        if i != 0 and j > 0:
            for sx, sz in rng.sample(((-1,-1),(-1,1),(1,-1),(1,1)), rng.randint(1, 2)):
                if rng.random() < .72:
                    tombs.append((cx + sx * width * .27, cz + sz * depth * .29))
        if j >= 2 and rng.random() < .22:
            guardians.append((cx, cz))
    rng.shuffle(guardians)
    guardians = guardians[:24]
    if not any(abs(x) < width for x, _ in guardians):
        guardians.append((0., main_halls[-1][0].bounds[3] - depth * .5))
    return CryptLayout(int(seed), width, depth, row, 3, tuple(rooms),
                       frozenset(cells), tuple(tombs), tuple(guardians))
