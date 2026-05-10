"""Building access road planning for the world scene."""

from __future__ import annotations

import math

from game.world.objects import Road
from game.world.objects.building import Building

PointXZ = tuple[float, float]
RectXZ = tuple[float, float, float, float]
RoadSegment = tuple[PointXZ, PointXZ]
Route = list[PointXZ]
PlannedRoute = tuple[Route, float]

_CONNECTABLE_ROAD_LIMIT = 24
_LANE_TARGET_LIMIT = 5
_PRIMARY_LANE_LIMIT = 12
_CROSS_LANE_LIMIT = 8


class BuildingRoadPlanner:
    """Plan non-overlapping driveway routes from buildings to the road network."""

    def __init__(self, scene) -> None:
        self.scene = scene

    def __getattr__(self, name: str):
        return getattr(self.scene, name)

    @staticmethod
    def _doorway_normal(side: str) -> PointXZ:
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
        first: RectXZ,
        second: RectXZ,
    ) -> bool:
        return not (
            first[1] <= second[0]
            or first[0] >= second[1]
            or first[3] <= second[2]
            or first[2] >= second[3]
        )

    @staticmethod
    def _segment_road_rect(
        p0: PointXZ,
        p1: PointXZ,
        width: float,
    ) -> RectXZ | None:
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
    def _route_length(route: Route) -> float:
        return sum(
            math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            for p0, p1 in zip(route, route[1:])
        )

    @staticmethod
    def _segment_length(
        segment: RoadSegment,
    ) -> float:
        (x0, z0), (x1, z1) = segment
        return math.hypot(x1 - x0, z1 - z0)

    @classmethod
    def _turn_count(cls, route: Route) -> int:
        count = 0
        segments = cls._route_to_segments(route)
        for first, second in zip(segments, segments[1:]):
            (x0, z0), (x1, z1) = first
            (x2, z2), (x3, z3) = second
            ux0, uz0 = x1 - x0, z1 - z0
            ux1, uz1 = x3 - x2, z3 - z2
            len0 = math.hypot(ux0, uz0)
            len1 = math.hypot(ux1, uz1)
            if len0 <= 1e-4 or len1 <= 1e-4:
                continue
            cross = abs((ux0 / len0) * (uz1 / len1) - (uz0 / len0) * (ux1 / len1))
            if cross > 1e-3:
                count += 1
        return count

    @classmethod
    def _close_turn_penalty(
        cls, route: Route, road_width: float
    ) -> float:
        segments = cls._route_to_segments(route)
        if len(segments) < 3:
            return 0.0

        min_spacing = max(road_width * 0.85, 10.0)
        penalty = 0.0
        for index, segment in enumerate(segments):
            if index == 0 or index == len(segments) - 1:
                continue
            length = cls._segment_length(segment)
            if length < min_spacing:
                penalty += (min_spacing - length) * 20.0 + road_width * 8.0
        return penalty

    @classmethod
    def _route_score(cls, route: Route, road_width: float) -> float:
        return (
            cls._route_length(route)
            + cls._turn_count(route) * road_width * 0.5
            + cls._close_turn_penalty(route, road_width)
        )

    @staticmethod
    def _route_to_segments(
        route: Route,
    ) -> list[RoadSegment]:
        return [
            (p0, p1)
            for p0, p1 in zip(route, route[1:])
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 1e-4
        ]

    @staticmethod
    def _segment_orientation(
        segment: RoadSegment,
    ) -> str | None:
        (x0, z0), (x1, z1) = segment
        if abs(x1 - x0) <= 1e-4:
            return "vertical"
        if abs(z1 - z0) <= 1e-4:
            return "horizontal"
        return None

    @staticmethod
    def _point_to_segment_distance(
        point: PointXZ,
        segment: RoadSegment,
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
    def _prune_route(route: Route) -> Route:
        clean: Route = []
        for point in route:
            if (
                clean
                and math.hypot(
                    point[0] - clean[-1][0],
                    point[1] - clean[-1][1],
                )
                <= 1e-4
            ):
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

    @classmethod
    def _soften_close_turns(
        cls, route: Route, road_width: float
    ) -> Route:
        """Remove tiny interior doglegs that make adjacent turns overlap."""
        clean = cls._prune_route(route)
        min_spacing = max(road_width * 0.55, 8.0)

        changed = True
        while changed and len(clean) > 3:
            changed = False
            for index in range(1, len(clean) - 2):
                p0 = clean[index]
                p1 = clean[index + 1]
                if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) >= min_spacing:
                    continue

                remove_first = clean[:index] + clean[index + 1 :]
                remove_second = clean[: index + 1] + clean[index + 2 :]
                remove_first = cls._prune_route(remove_first)
                remove_second = cls._prune_route(remove_second)
                if index == 1:
                    clean = remove_second
                elif cls._route_length(remove_first) <= cls._route_length(
                    remove_second
                ):
                    clean = remove_first
                else:
                    clean = remove_second
                changed = True
                break

        return cls._prune_route(clean)

    def _network_lane_positions(
        self,
        road_network: list[RoadSegment],
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
    ) -> tuple[PointXZ, PointXZ]:
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
        doorway_normal: PointXZ,
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
        doorway_normal: PointXZ,
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
        route: Route,
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

        joint_pad = road_width * 0.5
        for point in route[1:-1]:
            joint_rect = (
                point[0] - joint_pad,
                point[0] + joint_pad,
                point[1] - joint_pad,
                point[1] + joint_pad,
            )
            if not (
                joint_rect[0] >= world_bounds[0]
                and joint_rect[1] <= world_bounds[1]
                and joint_rect[2] >= world_bounds[2]
                and joint_rect[3] <= world_bounds[3]
            ):
                return False
            for building in buildings:
                if self._rects_overlap(joint_rect, building.bounds):
                    return False

        return True

    def _candidate_routes_to_target(
        self,
        *,
        front: PointXZ,
        apron: PointXZ,
        target: PointXZ,
        lane_xs: list[float],
        lane_zs: list[float],
    ) -> list[Route]:
        target_x, target_z = target
        candidates = [
            [front, apron, (target_x, apron[1]), target],
            [front, apron, (apron[0], target_z), target],
        ]

        for lane_x in lane_xs[:_PRIMARY_LANE_LIMIT]:
            candidates.append(
                [front, apron, (lane_x, apron[1]), (lane_x, target_z), target]
            )
        for lane_z in lane_zs[:_PRIMARY_LANE_LIMIT]:
            candidates.append(
                [front, apron, (apron[0], lane_z), (target_x, lane_z), target]
            )
        for lane_x in lane_xs[:_CROSS_LANE_LIMIT]:
            for lane_z in lane_zs[:_CROSS_LANE_LIMIT]:
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
        front: PointXZ,
        apron: PointXZ,
        segment: RoadSegment,
        lane_xs: list[float],
        lane_zs: list[float],
    ) -> list[Route]:
        orientation = self._segment_orientation(segment)
        if orientation is None:
            return []

        (x0, z0), (x1, z1) = segment
        targets: list[PointXZ] = []
        if orientation == "horizontal":
            min_x, max_x = sorted((x0, x1))
            target_xs = [max(min_x, min(max_x, apron[0]))]
            target_xs.extend(x for x in lane_xs if min_x <= x <= max_x)
            target_xs = self._dedupe_sorted(
                sorted(target_xs, key=lambda x: abs(x - apron[0])),
                min_delta=1.0,
            )[:_LANE_TARGET_LIMIT]
            targets.extend((x, z0) for x in target_xs)
        else:
            min_z, max_z = sorted((z0, z1))
            target_zs = [max(min_z, min(max_z, apron[1]))]
            target_zs.extend(z for z in lane_zs if min_z <= z <= max_z)
            target_zs = self._dedupe_sorted(
                sorted(target_zs, key=lambda z: abs(z - apron[1])),
                min_delta=1.0,
            )[:_LANE_TARGET_LIMIT]
            targets.extend((x0, z) for z in target_zs)

        candidates: list[Route] = []
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
        road_network: list[RoadSegment],
    ) -> Route:
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

        candidates: list[Route] = []
        connectable_network = [
            segment
            for segment in road_network
            if self._segment_orientation(segment) is not None
        ]
        ranked_network = sorted(
            connectable_network,
            key=lambda segment: self._point_to_segment_distance(apron, segment),
        )
        for segment in ranked_network[:_CONNECTABLE_ROAD_LIMIT]:
            candidates.extend(
                self._candidate_routes_to_road_segment(
                    front=front,
                    apron=apron,
                    segment=segment,
                    lane_xs=lane_xs,
                    lane_zs=lane_zs,
                )
            )

        best_route: Route = []
        best_score = float("inf")
        for candidate in candidates:
            route = self._soften_close_turns(
                self._prune_route(candidate),
                road_width,
            )
            if self._route_is_clear(route, road_width=road_width, buildings=buildings):
                score = self._route_score(route, road_width)
                if score < best_score:
                    best_route = route
                    best_score = score

        return best_route

    def _record_planned_routes(self, planned_routes: list[PlannedRoute]) -> None:
        self.scene.building_road_routes = [route for route, _ in planned_routes]
        self.scene.building_road_segments = [
            (segment, driveway_width)
            for route, driveway_width in planned_routes
            for segment in self._route_to_segments(route)
        ]

    def _build_route_road(
        self,
        *,
        route: Route,
        driveway_width: float,
        road_y: float,
    ) -> Road:
        return Road(
            points=route,
            ground_y=road_y,
            width=driveway_width,
            texture=self.road_tex,
            v_tiles=1.0,
            height_sampler=self._ground_height_sampler,
            elevation=3.0,
            segment_length=8.0,
            brightness_modifiers=self.brightness_modifiers,
            default_brightness=self.camera.brightness_default,
            lighting=getattr(self.scene, "lighting", None),
            sun_direction=getattr(self.scene, "sun_direction", None),
        )

    def _instantiate_planned_roads(
        self, planned_routes: list[PlannedRoute], *, road_y: float
    ) -> list[Road]:
        return [
            self._build_route_road(
                route=route,
                driveway_width=driveway_width,
                road_y=road_y,
            )
            for route, driveway_width in planned_routes
            if self._route_to_segments(route)
        ]

    def create_building_access_roads(
        self,
        *,
        road_center_z: float,
        road_y: float,
        main_road_segment: RoadSegment,
    ) -> list[Road]:
        road_network = [main_road_segment]
        planned_routes: list[PlannedRoute] = []
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
                bx = float(building.position.x)
                bz = float(building.position.z)
                print(
                    "Warning: could not find non-overlapping road route for "
                    f"building at ({bx:.1f}, {bz:.1f})."
                )
                continue

            planned_routes.append((route, driveway_width))
            road_network.extend(self._route_to_segments(route))

        self._record_planned_routes(planned_routes)
        return self._instantiate_planned_roads(planned_routes, road_y=road_y)


def create_building_access_roads(
    scene,
    *,
    road_center_z: float,
    road_y: float,
    main_road_segment: RoadSegment,
) -> list[Road]:
    planner = BuildingRoadPlanner(scene)
    return planner.create_building_access_roads(
        road_center_z=road_center_z,
        road_y=road_y,
        main_road_segment=main_road_segment,
    )
