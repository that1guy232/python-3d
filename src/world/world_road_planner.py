"""Building access road planning for the world scene."""

from __future__ import annotations

import math

from world.objects import Road
from world.objects.building import Building


class BuildingRoadPlanner:
    """Plan non-overlapping driveway routes from buildings to the road network."""

    def __init__(self, scene) -> None:
        self.scene = scene

    def __getattr__(self, name: str):
        return getattr(self.scene, name)

    @staticmethod
    def _doorway_normal(side: str) -> tuple[float, float]:
        side_key = str(side).lower()
        if side_key == "north":
            return (0.0, 1.0)
        if side_key == "east":
            return (1.0, 0.0)
        if side_key == "west":
            return (-1.0, 0.0)
        return (0.0, -1.0)

    @staticmethod
    def _rects_overlap(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> bool:
        return not (
            first[1] <= second[0]
            or first[0] >= second[1]
            or first[3] <= second[2]
            or first[2] >= second[3]
        )

    @staticmethod
    def _segment_road_rect(
        p0: tuple[float, float],
        p1: tuple[float, float],
        width: float,
    ) -> tuple[float, float, float, float] | None:
        x0, z0 = p0
        x1, z1 = p1
        if math.hypot(x1 - x0, z1 - z0) <= 1e-4:
            return None

        half_width = width * 0.5
        if abs(x1 - x0) <= 1e-4:
            return (
                x0 - half_width,
                x0 + half_width,
                min(z0, z1),
                max(z0, z1),
            )
        if abs(z1 - z0) <= 1e-4:
            return (
                min(x0, x1),
                max(x0, x1),
                z0 - half_width,
                z0 + half_width,
            )

        return (
            min(x0, x1) - half_width,
            max(x0, x1) + half_width,
            min(z0, z1) - half_width,
            max(z0, z1) + half_width,
        )

    @staticmethod
    def _dedupe_sorted(values: list[float], *, min_delta: float = 1.0) -> list[float]:
        result: list[float] = []
        for value in values:
            if all(abs(value - existing) >= min_delta for existing in result):
                result.append(value)
        return result

    @staticmethod
    def _route_length(route: list[tuple[float, float]]) -> float:
        return sum(
            math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            for p0, p1 in zip(route, route[1:])
        )

    @staticmethod
    def _route_to_segments(
        route: list[tuple[float, float]],
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return [
            (p0, p1)
            for p0, p1 in zip(route, route[1:])
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 1e-4
        ]

    @staticmethod
    def _segment_orientation(
        segment: tuple[tuple[float, float], tuple[float, float]],
    ) -> str | None:
        (x0, z0), (x1, z1) = segment
        if abs(x1 - x0) <= 1e-4:
            return "vertical"
        if abs(z1 - z0) <= 1e-4:
            return "horizontal"
        return None

    @staticmethod
    def _point_to_segment_distance(
        point: tuple[float, float],
        segment: tuple[tuple[float, float], tuple[float, float]],
    ) -> float:
        px, pz = point
        (x0, z0), (x1, z1) = segment
        vx = x1 - x0
        vz = z1 - z0
        len2 = vx * vx + vz * vz
        if len2 <= 1e-8:
            return math.hypot(px - x0, pz - z0)
        t = ((px - x0) * vx + (pz - z0) * vz) / len2
        t = max(0.0, min(1.0, t))
        cx = x0 + vx * t
        cz = z0 + vz * t
        return math.hypot(px - cx, pz - cz)

    @staticmethod
    def _prune_route(route: list[tuple[float, float]]) -> list[tuple[float, float]]:
        clean: list[tuple[float, float]] = []
        for point in route:
            if clean and math.hypot(point[0] - clean[-1][0], point[1] - clean[-1][1]) <= 1e-4:
                continue
            clean.append(point)

        i = 1
        while i < len(clean) - 1:
            prev_point = clean[i - 1]
            point = clean[i]
            next_point = clean[i + 1]
            same_x = (
                abs(prev_point[0] - point[0]) <= 1e-4
                and abs(point[0] - next_point[0]) <= 1e-4
            )
            same_z = (
                abs(prev_point[1] - point[1]) <= 1e-4
                and abs(point[1] - next_point[1]) <= 1e-4
            )
            if same_x or same_z:
                del clean[i]
            else:
                i += 1

        return clean

    def _network_lane_positions(
        self,
        road_network: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> tuple[list[float], list[float]]:
        lane_xs: list[float] = []
        lane_zs: list[float] = []
        for segment in road_network:
            orientation = self._segment_orientation(segment)
            (x0, z0), _ = segment
            if orientation == "vertical":
                lane_xs.append(x0)
            elif orientation == "horizontal":
                lane_zs.append(z0)
        return (
            self._dedupe_sorted(lane_xs, min_delta=1.0),
            self._dedupe_sorted(lane_zs, min_delta=1.0),
        )

    def _building_front_point(
        self,
        building: Building,
        doorway_side: str,
        start_gap: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        nx, nz = self._doorway_normal(doorway_side)
        min_x, max_x, min_z, max_z = building.bounds
        cx = float(building.position.x)
        cz = float(building.position.z)

        if nz > 0.0:
            edge = (cx, max_z)
        elif nz < 0.0:
            edge = (cx, min_z)
        elif nx > 0.0:
            edge = (max_x, cz)
        else:
            edge = (min_x, cz)

        return (edge[0] + nx * start_gap, edge[1] + nz * start_gap), (nx, nz)

    def _candidate_driveway_lane_xs(
        self,
        *,
        apron_x: float,
        doorway_normal: tuple[float, float],
        building: Building,
        buildings: list[Building],
        road_width: float,
    ) -> list[float]:
        min_world_x, max_world_x, _, _ = self.ground_bounds
        min_x, max_x, _, _ = building.bounds
        half_width = road_width * 0.5
        edge_offset = half_width + 12.0
        world_margin = half_width + 2.0
        nx, _ = doorway_normal

        values = [apron_x]
        if nx > 0.0:
            values.extend([max_x + edge_offset, max_world_x - world_margin])
            values.extend(other.bounds[1] + edge_offset for other in buildings)
        elif nx < 0.0:
            values.extend([min_x - edge_offset, min_world_x + world_margin])
            values.extend(other.bounds[0] - edge_offset for other in buildings)
        else:
            values.extend(
                [
                    min_x - edge_offset,
                    max_x + edge_offset,
                    min_world_x + world_margin,
                    max_world_x - world_margin,
                ]
            )
            for other in buildings:
                other_min_x, other_max_x, _, _ = other.bounds
                values.extend([other_min_x - edge_offset, other_max_x + edge_offset])

        clamped: list[float] = []
        for value in values:
            x = max(min_world_x + world_margin, min(max_world_x - world_margin, value))
            if nx > 0.0 and x < max_x + half_width:
                continue
            if nx < 0.0 and x > min_x - half_width:
                continue
            clamped.append(x)

        clamped.sort(key=lambda value: abs(value - apron_x))
        return self._dedupe_sorted(clamped)

    def _candidate_driveway_lane_zs(
        self,
        *,
        apron_z: float,
        doorway_normal: tuple[float, float],
        building: Building,
        buildings: list[Building],
        road_width: float,
        road_center_z: float,
    ) -> list[float]:
        _, _, min_world_z, max_world_z = self.ground_bounds
        _, _, min_z, max_z = building.bounds
        half_width = road_width * 0.5
        edge_offset = half_width + 12.0
        world_margin = half_width + 2.0
        _, nz = doorway_normal

        values = [apron_z, road_center_z]
        if nz > 0.0:
            values.extend([max_z + edge_offset, max_world_z - world_margin])
            values.extend(other.bounds[3] + edge_offset for other in buildings)
        elif nz < 0.0:
            values.extend([min_z - edge_offset, min_world_z + world_margin])
            values.extend(other.bounds[2] - edge_offset for other in buildings)
        else:
            values.extend([min_world_z + world_margin, max_world_z - world_margin])
            for other in buildings:
                _, _, other_min_z, other_max_z = other.bounds
                values.extend([other_min_z - edge_offset, other_max_z + edge_offset])

        clamped: list[float] = []
        for value in values:
            z = max(min_world_z + world_margin, min(max_world_z - world_margin, value))
            if nz > 0.0 and z < max_z + half_width:
                continue
            if nz < 0.0 and z > min_z - half_width:
                continue
            clamped.append(z)

        clamped.sort(key=lambda value: abs(value - apron_z))
        return self._dedupe_sorted(clamped)

    def _route_is_clear(
        self,
        route: list[tuple[float, float]],
        *,
        road_width: float,
        buildings: list[Building],
    ) -> bool:
        if len(route) < 2:
            return False

        min_world_x, max_world_x, min_world_z, max_world_z = self.ground_bounds
        world_bounds = (min_world_x, max_world_x, min_world_z, max_world_z)

        for p0, p1 in zip(route, route[1:]):
            road_rect = self._segment_road_rect(p0, p1, road_width)
            if road_rect is None:
                continue
            if not (
                road_rect[0] >= world_bounds[0]
                and road_rect[1] <= world_bounds[1]
                and road_rect[2] >= world_bounds[2]
                and road_rect[3] <= world_bounds[3]
            ):
                return False
            for building in buildings:
                if self._rects_overlap(road_rect, building.bounds):
                    return False

        return True

    def _candidate_routes_to_target(
        self,
        *,
        front: tuple[float, float],
        apron: tuple[float, float],
        target: tuple[float, float],
        lane_xs: list[float],
        lane_zs: list[float],
    ) -> list[list[tuple[float, float]]]:
        target_x, target_z = target
        candidates = [
            [front, apron, (target_x, apron[1]), target],
            [front, apron, (apron[0], target_z), target],
        ]

        for lane_x in lane_xs[:12]:
            candidates.append([front, apron, (lane_x, apron[1]), (lane_x, target_z), target])
        for lane_z in lane_zs[:12]:
            candidates.append([front, apron, (apron[0], lane_z), (target_x, lane_z), target])
        for lane_x in lane_xs[:8]:
            for lane_z in lane_zs[:8]:
                candidates.append(
                    [
                        front,
                        apron,
                        (lane_x, apron[1]),
                        (lane_x, lane_z),
                        (target_x, lane_z),
                        target,
                    ]
                )
                candidates.append(
                    [
                        front,
                        apron,
                        (apron[0], lane_z),
                        (lane_x, lane_z),
                        (lane_x, target_z),
                        target,
                    ]
                )

        return candidates

    def _candidate_routes_to_road_segment(
        self,
        *,
        front: tuple[float, float],
        apron: tuple[float, float],
        segment: tuple[tuple[float, float], tuple[float, float]],
        lane_xs: list[float],
        lane_zs: list[float],
    ) -> list[list[tuple[float, float]]]:
        orientation = self._segment_orientation(segment)
        if orientation is None:
            return []

        (x0, z0), (x1, z1) = segment
        targets: list[tuple[float, float]] = []
        if orientation == "horizontal":
            min_x, max_x = sorted((x0, x1))
            target_xs = [max(min_x, min(max_x, apron[0]))]
            target_xs.extend(x for x in lane_xs if min_x <= x <= max_x)
            target_xs = self._dedupe_sorted(
                sorted(target_xs, key=lambda x: abs(x - apron[0])),
                min_delta=1.0,
            )[:5]
            targets.extend((x, z0) for x in target_xs)
        else:
            min_z, max_z = sorted((z0, z1))
            target_zs = [max(min_z, min(max_z, apron[1]))]
            target_zs.extend(z for z in lane_zs if min_z <= z <= max_z)
            target_zs = self._dedupe_sorted(
                sorted(target_zs, key=lambda z: abs(z - apron[1])),
                min_delta=1.0,
            )[:5]
            targets.extend((x0, z) for z in target_zs)

        candidates: list[list[tuple[float, float]]] = []
        for target in targets:
            candidates.extend(
                self._candidate_routes_to_target(
                    front=front,
                    apron=apron,
                    target=target,
                    lane_xs=lane_xs,
                    lane_zs=lane_zs,
                )
            )
        return candidates

    def _find_building_access_route(
        self,
        *,
        building: Building,
        spec: dict,
        road_center_z: float,
        road_width: float,
        buildings: list[Building],
        road_network: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> list[tuple[float, float]]:
        start_gap = 0.5
        turn_clearance = road_width * 0.5 + 12.0
        front, normal = self._building_front_point(
            building, spec.get("doorway_side", "south"), start_gap
        )
        apron = (
            front[0] + normal[0] * turn_clearance,
            front[1] + normal[1] * turn_clearance,
        )
        lane_xs = self._candidate_driveway_lane_xs(
            apron_x=apron[0],
            doorway_normal=normal,
            building=building,
            buildings=buildings,
            road_width=road_width,
        )
        lane_zs = self._candidate_driveway_lane_zs(
            apron_z=apron[1],
            doorway_normal=normal,
            building=building,
            buildings=buildings,
            road_width=road_width,
            road_center_z=road_center_z,
        )
        network_lane_xs, network_lane_zs = self._network_lane_positions(road_network)
        lane_xs = self._dedupe_sorted(
            sorted(
                lane_xs + network_lane_xs,
                key=lambda value: min(abs(value - apron[0]), abs(value - front[0])),
            )
        )
        lane_zs = self._dedupe_sorted(
            sorted(
                lane_zs + network_lane_zs,
                key=lambda value: min(abs(value - apron[1]), abs(value - front[1])),
            )
        )

        candidates: list[list[tuple[float, float]]] = []
        ranked_network = sorted(
            road_network,
            key=lambda segment: self._point_to_segment_distance(apron, segment),
        )
        for segment in ranked_network[:24]:
            candidates.extend(
                self._candidate_routes_to_road_segment(
                    front=front,
                    apron=apron,
                    segment=segment,
                    lane_xs=lane_xs,
                    lane_zs=lane_zs,
                )
            )

        best_route: list[tuple[float, float]] = []
        best_score = float("inf")
        for candidate in candidates:
            route = self._prune_route(candidate)
            if self._route_is_clear(route, road_width=road_width, buildings=buildings):
                score = self._route_length(route)
                if score < best_score:
                    best_route = route
                    best_score = score

        return best_route

    def create_building_access_roads(
        self,
        *,
        road_center_z: float,
        road_y: float,
        main_road_segment: tuple[tuple[float, float], tuple[float, float]],
    ) -> list[Road]:
        roads: list[Road] = []
        road_network = [main_road_segment]
        planned_routes: list[tuple[list[tuple[float, float]], float]] = []
        route_requests = list(zip(self.buildings, self.building_specs))
        route_requests.sort(
            key=lambda item: abs(float(item[0].position.z) - road_center_z)
        )

        for building, spec in route_requests:
            driveway_width = min(34.0, max(18.0, float(spec["doorway_width"]) * 0.8))
            route = self._find_building_access_route(
                building=building,
                spec=spec,
                road_center_z=road_center_z,
                road_width=driveway_width,
                buildings=self.buildings,
                road_network=road_network,
            )
            if not route:
                print(
                    "Warning: could not find non-overlapping road route for "
                    f"building at ({building.position.x:.1f}, {building.position.z:.1f})."
                )
                continue

            planned_routes.append((route, driveway_width))
            road_network.extend(self._route_to_segments(route))

        self.scene.building_road_routes = [route for route, _ in planned_routes]
        self.scene.building_road_segments = [
            (segment, driveway_width)
            for route, driveway_width in planned_routes
            for segment in self._route_to_segments(route)
        ]

        for (p0, p1), driveway_width in self.scene.building_road_segments:
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) <= 1e-4:
                continue
            roads.append(
                Road(
                    start=p0,
                    end=p1,
                    ground_y=road_y,
                    width=driveway_width,
                    texture=self.road_tex,
                    px_to_world=1.0,
                    v_tiles=1.0,
                    height_sampler=self._ground_height_sampler,
                    elevation=3.0,
                    segment_length=8.0,
                    brightness_modifiers=self.brightness_modifiers,
                    default_brightness=self.camera.brightness_default,
                )
            )

        return roads


def create_building_access_roads(
    scene,
    *,
    road_center_z: float,
    road_y: float,
    main_road_segment: tuple[tuple[float, float], tuple[float, float]],
) -> list[Road]:
    planner = BuildingRoadPlanner(scene)
    return planner.create_building_access_roads(
        road_center_z=road_center_z,
        road_y=road_y,
        main_road_segment=main_road_segment,
    )
