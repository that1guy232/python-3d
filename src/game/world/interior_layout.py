"""Connected room-layout generation for rectangular buildings."""

from __future__ import annotations

import math
from typing import Any

from game.world.objects import Door

_SIDES_BY_ENTRY_AXIS = {
    "north": ("z", -1.0),
    "south": ("z", 1.0),
    "east": ("x", -1.0),
    "west": ("x", 1.0),
}

_INTERIOR_WALL_JOIN_OVERLAP = 0.5
_MIN_ROOM_SIZE = 62.0
_MIN_SIDE_ROOM_DEPTH = 48.0
_MIN_HALL_SEGMENT_LENGTH = 68.0
_MIN_CHAIN_ROOM_LENGTH = 74.0
_HALL_WIDTH = 56.0
_MAX_HALL_SEGMENTS = 3
_MAX_CHAIN_ROOMS = 4


def _target_room_count(width: float, depth: float) -> int:
    area = max(0.0, float(width)) * max(0.0, float(depth))
    if area < 22000.0:
        return 1
    if area < 34000.0:
        return 2
    if area < 52000.0:
        return 3
    if area < 78000.0:
        return 4
    return 6


def _room(
    room_id: str,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
) -> dict[str, Any]:
    return {
        "id": room_id,
        "x_min": float(x_min),
        "x_max": float(x_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
    }


def _hallway(
    hallway_id: str,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
) -> dict[str, Any]:
    return {
        "id": hallway_id,
        "x_min": float(x_min),
        "x_max": float(x_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
    }


def _partition(
    *,
    axis: str,
    coord: float,
    span_min: float,
    span_max: float,
    side: str,
    openings: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    return {
        "axis": str(axis),
        "coord": float(coord),
        "span_min": float(span_min),
        "span_max": float(span_max),
        "side": str(side),
        "openings": [dict(opening) for opening in (openings or ())],
    }


def _door(
    door_id: str,
    *,
    x: float,
    z: float,
    side: str,
    width: float,
    height: float,
    connects: tuple[str, str],
) -> dict[str, Any]:
    return {
        "id": door_id,
        "x": float(x),
        "z": float(z),
        "side": str(side),
        "width": float(width),
        "height": float(height),
        "connects": tuple(connects),
    }


def _connection(door_id: str, a: str, b: str) -> dict[str, Any]:
    return {"door": str(door_id), "a": str(a), "b": str(b)}


def _single_room_layout(
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
) -> dict[str, Any]:
    return {
        "kind": "single_room",
        "rooms": [_room("room_0", x_min, x_max, z_min, z_max)],
        "hallways": [],
        "partitions": [],
        "doors": [],
        "connections": [_connection("exterior", "outside", "room_0")],
    }


def _segment_bounds(start: float, end: float, count: int) -> list[tuple[float, float]]:
    count = max(1, int(count))
    step = (float(end) - float(start)) / count
    return [
        (start + step * index, start + step * (index + 1)) for index in range(count)
    ]


def _hallway_layout(
    *,
    axis: str,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    target_rooms: int,
    door_width: float,
    door_height: float,
) -> dict[str, Any]:
    hall_half = _HALL_WIDTH * 0.5
    rooms: list[dict[str, Any]] = []
    hallways: list[dict[str, Any]] = []
    partitions: list[dict[str, Any]] = []
    doors: list[dict[str, Any]] = []
    connections = [_connection("exterior", "outside", "hall_0")]
    door_opening = {
        "width": float(door_width),
        "y_min": 0.0,
        "y_max": float(door_height),
    }

    if axis == "z":
        length = z_max - z_min
        max_segments = max(1, int(math.floor(length / _MIN_HALL_SEGMENT_LENGTH)))
        segments = max(
            1,
            min(_MAX_HALL_SEGMENTS, max_segments, math.ceil(target_rooms / 2)),
        )
        hallways.append(_hallway("hall_0", -hall_half, hall_half, z_min, z_max))
        z_segments = _segment_bounds(z_min, z_max, segments)
        left_openings = []
        right_openings = []

        for seg_min, seg_max in z_segments:
            z_mid = (seg_min + seg_max) * 0.5
            left_id = f"room_{len(rooms)}"
            right_id = f"room_{len(rooms) + 1}"
            rooms.append(_room(left_id, x_min, -hall_half, seg_min, seg_max))
            rooms.append(_room(right_id, hall_half, x_max, seg_min, seg_max))

            left_door_id = f"door_{len(doors)}"
            right_door_id = f"door_{len(doors) + 1}"
            left_openings.append({"offset": z_mid, **door_opening})
            right_openings.append({"offset": z_mid, **door_opening})
            doors.append(
                _door(
                    left_door_id,
                    x=-hall_half,
                    z=z_mid,
                    side="east",
                    width=door_width,
                    height=door_height,
                    connects=("hall_0", left_id),
                )
            )
            doors.append(
                _door(
                    right_door_id,
                    x=hall_half,
                    z=z_mid,
                    side="west",
                    width=door_width,
                    height=door_height,
                    connects=("hall_0", right_id),
                )
            )
            connections.append(_connection(left_door_id, "hall_0", left_id))
            connections.append(_connection(right_door_id, "hall_0", right_id))

        partitions.append(
            _partition(
                axis="x",
                coord=-hall_half,
                span_min=z_min,
                span_max=z_max,
                side="east",
                openings=left_openings,
            )
        )
        partitions.append(
            _partition(
                axis="x",
                coord=hall_half,
                span_min=z_min,
                span_max=z_max,
                side="west",
                openings=right_openings,
            )
        )
        for boundary in (segment[1] for segment in z_segments[:-1]):
            partitions.append(
                _partition(
                    axis="z",
                    coord=boundary,
                    span_min=x_min,
                    span_max=-hall_half,
                    side="north",
                )
            )
            partitions.append(
                _partition(
                    axis="z",
                    coord=boundary,
                    span_min=hall_half,
                    span_max=x_max,
                    side="north",
                )
            )
    else:
        length = x_max - x_min
        max_segments = max(1, int(math.floor(length / _MIN_HALL_SEGMENT_LENGTH)))
        segments = max(
            1,
            min(_MAX_HALL_SEGMENTS, max_segments, math.ceil(target_rooms / 2)),
        )
        hallways.append(_hallway("hall_0", x_min, x_max, -hall_half, hall_half))
        x_segments = _segment_bounds(x_min, x_max, segments)
        lower_openings = []
        upper_openings = []

        for seg_min, seg_max in x_segments:
            x_mid = (seg_min + seg_max) * 0.5
            lower_id = f"room_{len(rooms)}"
            upper_id = f"room_{len(rooms) + 1}"
            rooms.append(_room(lower_id, seg_min, seg_max, z_min, -hall_half))
            rooms.append(_room(upper_id, seg_min, seg_max, hall_half, z_max))

            lower_door_id = f"door_{len(doors)}"
            upper_door_id = f"door_{len(doors) + 1}"
            lower_openings.append({"offset": x_mid, **door_opening})
            upper_openings.append({"offset": x_mid, **door_opening})
            doors.append(
                _door(
                    lower_door_id,
                    x=x_mid,
                    z=-hall_half,
                    side="north",
                    width=door_width,
                    height=door_height,
                    connects=("hall_0", lower_id),
                )
            )
            doors.append(
                _door(
                    upper_door_id,
                    x=x_mid,
                    z=hall_half,
                    side="south",
                    width=door_width,
                    height=door_height,
                    connects=("hall_0", upper_id),
                )
            )
            connections.append(_connection(lower_door_id, "hall_0", lower_id))
            connections.append(_connection(upper_door_id, "hall_0", upper_id))

        partitions.append(
            _partition(
                axis="z",
                coord=-hall_half,
                span_min=x_min,
                span_max=x_max,
                side="north",
                openings=lower_openings,
            )
        )
        partitions.append(
            _partition(
                axis="z",
                coord=hall_half,
                span_min=x_min,
                span_max=x_max,
                side="south",
                openings=upper_openings,
            )
        )
        for boundary in (segment[1] for segment in x_segments[:-1]):
            partitions.append(
                _partition(
                    axis="x",
                    coord=boundary,
                    span_min=z_min,
                    span_max=-hall_half,
                    side="east",
                )
            )
            partitions.append(
                _partition(
                    axis="x",
                    coord=boundary,
                    span_min=hall_half,
                    span_max=z_max,
                    side="east",
                )
            )

    return {
        "kind": "hallway",
        "rooms": rooms,
        "hallways": hallways,
        "partitions": partitions,
        "doors": doors,
        "connections": connections,
    }


def _chain_partition_side(axis: str, inward_sign: float) -> str:
    if axis == "z":
        return "north" if inward_sign > 0.0 else "south"
    return "east" if inward_sign > 0.0 else "west"


def _chain_layout(
    *,
    axis: str,
    inward_sign: float,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    target_rooms: int,
    door_width: float,
    door_height: float,
) -> dict[str, Any]:
    length = (z_max - z_min) if axis == "z" else (x_max - x_min)
    max_rooms = max(1, int(math.floor(length / _MIN_CHAIN_ROOM_LENGTH)))
    room_count = max(2, min(_MAX_CHAIN_ROOMS, max_rooms, target_rooms))
    if room_count < 2:
        return _single_room_layout(x_min, x_max, z_min, z_max)

    rooms: list[dict[str, Any]] = []
    partitions: list[dict[str, Any]] = []
    doors: list[dict[str, Any]] = []
    connections: list[dict[str, Any]] = []
    side = _chain_partition_side(axis, inward_sign)
    door_opening = {
        "offset": 0.0,
        "width": float(door_width),
        "y_min": 0.0,
        "y_max": float(door_height),
    }

    if axis == "z":
        ordered = (
            _segment_bounds(z_min, z_max, room_count)
            if inward_sign > 0.0
            else list(reversed(_segment_bounds(z_min, z_max, room_count)))
        )
        for index, (seg_min, seg_max) in enumerate(ordered):
            rooms.append(
                _room(
                    f"room_{index}",
                    x_min,
                    x_max,
                    min(seg_min, seg_max),
                    max(seg_min, seg_max),
                )
            )
        connections.append(_connection("exterior", "outside", "room_0"))
        for index in range(room_count - 1):
            current = rooms[index]
            nxt = rooms[index + 1]
            boundary = current["z_max"] if inward_sign > 0.0 else current["z_min"]
            door_id = f"door_{len(doors)}"
            partitions.append(
                _partition(
                    axis="z",
                    coord=boundary,
                    span_min=x_min,
                    span_max=x_max,
                    side=side,
                    openings=[dict(door_opening)],
                )
            )
            doors.append(
                _door(
                    door_id,
                    x=0.0,
                    z=boundary,
                    side=side,
                    width=door_width,
                    height=door_height,
                    connects=(current["id"], nxt["id"]),
                )
            )
            connections.append(_connection(door_id, current["id"], nxt["id"]))
    else:
        ordered = (
            _segment_bounds(x_min, x_max, room_count)
            if inward_sign > 0.0
            else list(reversed(_segment_bounds(x_min, x_max, room_count)))
        )
        for index, (seg_min, seg_max) in enumerate(ordered):
            rooms.append(
                _room(
                    f"room_{index}",
                    min(seg_min, seg_max),
                    max(seg_min, seg_max),
                    z_min,
                    z_max,
                )
            )
        connections.append(_connection("exterior", "outside", "room_0"))
        for index in range(room_count - 1):
            current = rooms[index]
            nxt = rooms[index + 1]
            boundary = current["x_max"] if inward_sign > 0.0 else current["x_min"]
            door_id = f"door_{len(doors)}"
            partitions.append(
                _partition(
                    axis="x",
                    coord=boundary,
                    span_min=z_min,
                    span_max=z_max,
                    side=side,
                    openings=[dict(door_opening)],
                )
            )
            doors.append(
                _door(
                    door_id,
                    x=boundary,
                    z=0.0,
                    side=side,
                    width=door_width,
                    height=door_height,
                    connects=(current["id"], nxt["id"]),
                )
            )
            connections.append(_connection(door_id, current["id"], nxt["id"]))

    return {
        "kind": "room_chain",
        "rooms": rooms,
        "hallways": [],
        "partitions": partitions,
        "doors": doors,
        "connections": connections,
    }


def _layout_is_valid(layout: dict[str, Any]) -> bool:
    room_ids = {
        str(room.get("id"))
        for room in layout.get("rooms", ())
        if isinstance(room, dict) and room.get("id") is not None
    }
    hallway_ids = {
        str(hallway.get("id"))
        for hallway in layout.get("hallways", ())
        if isinstance(hallway, dict) and hallway.get("id") is not None
    }
    if not room_ids:
        return False

    nodes = room_ids | hallway_ids | {"outside"}
    graph = {node: set() for node in nodes}
    room_door_counts = {room_id: 0 for room_id in room_ids}
    for connection in layout.get("connections", ()) or ():
        try:
            a = str(connection["a"])
            b = str(connection["b"])
        except (KeyError, TypeError):
            continue
        if a not in graph or b not in graph:
            continue
        graph[a].add(b)
        graph[b].add(a)
        if a in room_door_counts:
            room_door_counts[a] += 1
        if b in room_door_counts:
            room_door_counts[b] += 1

    if any(count <= 0 for count in room_door_counts.values()):
        return False

    seen = {"outside"}
    stack = ["outside"]
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)

    return room_ids.issubset(seen)


def create_building_interior_layout(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a connected room graph plus walls/doors for a building spec."""
    try:
        width = max(1.0, float(spec["width"]))
        depth = max(1.0, float(spec["depth"]))
    except (KeyError, TypeError, ValueError):
        return _single_room_layout(-50.0, 50.0, -50.0, 50.0)

    try:
        wall_thickness = max(0.0, float(spec.get("wall_thickness", 2.5)))
    except (TypeError, ValueError):
        wall_thickness = 2.5

    # Let partition endpoints overlap the shell wall volume a little so
    # interior walls visually meet the exterior walls without cracks.
    inset = max(0.0, wall_thickness - _INTERIOR_WALL_JOIN_OVERLAP)
    x_min = -width * 0.5 + inset
    x_max = width * 0.5 - inset
    z_min = -depth * 0.5 + inset
    z_max = depth * 0.5 - inset
    inner_width = x_max - x_min
    inner_depth = z_max - z_min

    if inner_width < _MIN_ROOM_SIZE or inner_depth < _MIN_ROOM_SIZE:
        return _single_room_layout(x_min, x_max, z_min, z_max)

    target_rooms = _target_room_count(inner_width, inner_depth)
    if target_rooms <= 1:
        return _single_room_layout(x_min, x_max, z_min, z_max)

    side = str(spec.get("doorway_side", "south")).lower()
    axis, inward_sign = _SIDES_BY_ENTRY_AXIS.get(side, ("z", 1.0))
    try:
        door_width = max(8.0, float(spec.get("doorway_width", Door.DEFAULT_WIDTH)))
    except (TypeError, ValueError):
        door_width = Door.DEFAULT_WIDTH
    try:
        door_height = max(8.0, float(spec.get("doorway_height", Door.DEFAULT_HEIGHT)))
    except (TypeError, ValueError):
        door_height = Door.DEFAULT_HEIGHT

    perpendicular = inner_width if axis == "z" else inner_depth
    length = inner_depth if axis == "z" else inner_width
    hall_fits = (
        perpendicular >= _HALL_WIDTH + _MIN_SIDE_ROOM_DEPTH * 2.0
        and length >= _MIN_HALL_SEGMENT_LENGTH
    )
    if hall_fits:
        layout = _hallway_layout(
            axis=axis,
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            target_rooms=target_rooms,
            door_width=door_width,
            door_height=door_height,
        )
        if _layout_is_valid(layout):
            return layout

    chain_fits = length >= _MIN_CHAIN_ROOM_LENGTH * 2.0
    if chain_fits:
        layout = _chain_layout(
            axis=axis,
            inward_sign=inward_sign,
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            target_rooms=target_rooms,
            door_width=door_width,
            door_height=door_height,
        )
        if _layout_is_valid(layout):
            return layout

    return _single_room_layout(x_min, x_max, z_min, z_max)


__all__ = ["create_building_interior_layout"]
