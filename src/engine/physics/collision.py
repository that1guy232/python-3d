"""Generic mesh collision helpers for scene runtimes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from pygame.math import Vector3


@dataclass(frozen=True)
class VerticalCollision:
    kind: str
    surface_y: float
    camera_y: float


def _mesh_vertices(mesh) -> list[Vector3]:
    if not hasattr(mesh, "get_world_vertices"):
        return []
    try:
        return mesh.get_world_vertices() or []
    except Exception:
        return []


def _mesh_vertical_bounds(mesh) -> tuple[float, float] | None:
    verts = _mesh_vertices(mesh)
    if not verts:
        return None
    return (min(v.y for v in verts), max(v.y for v in verts))


def _bounds_overlap_circle(bounds, x: float, z: float, radius: float) -> bool:
    if not bounds:
        return True
    min_x, max_x, min_z, max_z = bounds
    return (
        min_x - radius <= x <= max_x + radius and min_z - radius <= z <= max_z + radius
    )


def _mesh_overlaps_player_xz(mesh, x: float, z: float, radius: float) -> bool:
    if hasattr(mesh, "get_bounding_box"):
        try:
            return _bounds_overlap_circle(mesh.get_bounding_box(), x, z, radius)
        except Exception:
            pass

    verts = _mesh_vertices(mesh)
    if not verts:
        return False
    min_x = min(v.x for v in verts)
    max_x = max(v.x for v in verts)
    min_z = min(v.z for v in verts)
    max_z = max(v.z for v in verts)
    return _bounds_overlap_circle((min_x, max_x, min_z, max_z), x, z, radius)


def player_support_height_at(
    meshes: Iterable,
    x: float,
    z: float,
    foot_y: float,
    player_radius: float = 16.0,
    snap_up: float = 2.0,
) -> Optional[float]:
    """Return the highest solid top under the player footprint, if any."""
    best_y: float | None = None
    max_surface_y = foot_y + max(0.0, snap_up)

    for mesh in meshes:
        if not _mesh_overlaps_player_xz(mesh, x, z, player_radius):
            continue
        vertical_bounds = _mesh_vertical_bounds(mesh)
        if vertical_bounds is None:
            continue
        _, top_y = vertical_bounds
        if top_y > max_surface_y:
            continue
        if best_y is None or top_y > best_y:
            best_y = top_y

    return best_y


def resolve_player_vertical_collision(
    meshes: Iterable,
    old_pos: Vector3,
    new_pos: Vector3,
    *,
    foot_offset: float,
    head_offset: float,
    player_radius: float = 16.0,
) -> VerticalCollision | None:
    """Resolve vertical movement against solid mesh tops and undersides."""
    old_foot_y = old_pos.y - foot_offset
    new_foot_y = new_pos.y - foot_offset
    old_head_y = old_pos.y + head_offset
    new_head_y = new_pos.y + head_offset
    eps = 1e-6

    x = new_pos.x
    z = new_pos.z

    if new_foot_y < old_foot_y:
        best_floor: float | None = None
        for mesh in meshes:
            if not _mesh_overlaps_player_xz(mesh, x, z, player_radius):
                continue
            vertical_bounds = _mesh_vertical_bounds(mesh)
            if vertical_bounds is None:
                continue
            _, top_y = vertical_bounds
            if old_foot_y + eps >= top_y >= new_foot_y - eps:
                if best_floor is None or top_y > best_floor:
                    best_floor = top_y
        if best_floor is not None:
            return VerticalCollision(
                kind="floor",
                surface_y=best_floor,
                camera_y=best_floor + foot_offset,
            )

    if new_head_y > old_head_y:
        best_ceiling: float | None = None
        for mesh in meshes:
            if not _mesh_overlaps_player_xz(mesh, x, z, player_radius):
                continue
            vertical_bounds = _mesh_vertical_bounds(mesh)
            if vertical_bounds is None:
                continue
            bottom_y, _ = vertical_bounds
            if old_head_y - eps <= bottom_y <= new_head_y + eps:
                if best_ceiling is None or bottom_y < best_ceiling:
                    best_ceiling = bottom_y
        if best_ceiling is not None:
            return VerticalCollision(
                kind="ceiling",
                surface_y=best_ceiling,
                camera_y=best_ceiling - head_offset,
            )

    return None


def movement_blocked_by_wall(
    meshes: Iterable,
    old_pos: Vector3,
    new_pos: Vector3,
    player_radius: float = 16.0,
    player_bottom_y: float | None = None,
    player_top_y: float | None = None,
) -> Optional[Vector3]:
    """Return a blocking collision plane normal for the movement segment."""
    eps = 1e-6
    query_min_x = min(old_pos.x, new_pos.x) - player_radius
    query_max_x = max(old_pos.x, new_pos.x) + player_radius
    query_min_z = min(old_pos.z, new_pos.z) - player_radius
    query_max_z = max(old_pos.z, new_pos.z) + player_radius

    def bounds_overlap(bounds) -> bool:
        if not bounds:
            return True
        min_x, max_x, min_z, max_z = bounds
        return (
            min_x <= query_max_x
            and max_x >= query_min_x
            and min_z <= query_max_z
            and max_z >= query_min_z
        )

    def point_in_poly_2d(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            intersect = ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
            )
            if intersect:
                inside = not inside
            j = i
        return inside

    def closest_point_on_segment_2d(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[tuple[float, float], float]:
        seg_x = end[0] - start[0]
        seg_z = end[1] - start[1]
        length_sq = seg_x * seg_x + seg_z * seg_z
        if length_sq <= eps:
            closest = start
        else:
            t = (
                (point[0] - start[0]) * seg_x
                + (point[1] - start[1]) * seg_z
            ) / length_sq
            t = max(0.0, min(1.0, t))
            closest = (start[0] + seg_x * t, start[1] + seg_z * t)
        dx = point[0] - closest[0]
        dz = point[1] - closest[1]
        return closest, dx * dx + dz * dz

    def closest_points_on_segments_2d(
        player_start: tuple[float, float],
        player_end: tuple[float, float],
        wall_start: tuple[float, float],
        wall_end: tuple[float, float],
    ) -> tuple[float, tuple[float, float], tuple[float, float]]:
        player_dx = player_end[0] - player_start[0]
        player_dz = player_end[1] - player_start[1]
        wall_dx = wall_end[0] - wall_start[0]
        wall_dz = wall_end[1] - wall_start[1]
        offset_x = wall_start[0] - player_start[0]
        offset_z = wall_start[1] - player_start[1]
        denominator = player_dx * wall_dz - player_dz * wall_dx
        if abs(denominator) > eps:
            player_t = (offset_x * wall_dz - offset_z * wall_dx) / denominator
            wall_t = (offset_x * player_dz - offset_z * player_dx) / denominator
            if 0.0 <= player_t <= 1.0 and 0.0 <= wall_t <= 1.0:
                intersection = (
                    player_start[0] + player_dx * player_t,
                    player_start[1] + player_dz * player_t,
                )
                return 0.0, intersection, intersection

        candidates = []
        wall_point, distance_sq = closest_point_on_segment_2d(
            player_start, wall_start, wall_end
        )
        candidates.append((distance_sq, player_start, wall_point))
        wall_point, distance_sq = closest_point_on_segment_2d(
            player_end, wall_start, wall_end
        )
        candidates.append((distance_sq, player_end, wall_point))
        player_point, distance_sq = closest_point_on_segment_2d(
            wall_start, player_start, player_end
        )
        candidates.append((distance_sq, player_point, wall_start))
        player_point, distance_sq = closest_point_on_segment_2d(
            wall_end, player_start, player_end
        )
        candidates.append((distance_sq, player_point, wall_end))
        return min(candidates, key=lambda candidate: candidate[0])

    def wall_radius_collision_normal(
        face_verts: list[Vector3], face_normal: Vector3
    ) -> Optional[Vector3]:
        """Sweep the player's horizontal circle against a finite wall face."""
        player_start = (float(old_pos.x), float(old_pos.z))
        player_end = (float(new_pos.x), float(new_pos.z))
        wall_points = [(float(vertex.x), float(vertex.z)) for vertex in face_verts]
        edges = list(zip(wall_points, wall_points[1:] + wall_points[:1]))
        if not edges:
            return None

        def distance_to_face_sq(point: tuple[float, float]) -> float:
            return min(
                closest_point_on_segment_2d(point, edge_start, edge_end)[1]
                for edge_start, edge_end in edges
            )

        closest = min(
            (
                closest_points_on_segments_2d(
                    player_start,
                    player_end,
                    edge_start,
                    edge_end,
                )
                for edge_start, edge_end in edges
            ),
            key=lambda candidate: candidate[0],
        )
        closest_distance_sq, player_point, wall_point = closest
        radius_sq = max(0.0, float(player_radius)) ** 2
        start_distance_sq = distance_to_face_sq(player_start)

        if start_distance_sq > radius_sq + eps:
            collides = closest_distance_sq <= radius_sq + eps
        else:
            # When already touching, block only motion that moves closer. This
            # permits tangential movement and lets an overlapping player escape.
            collides = closest_distance_sq < start_distance_sq - eps
        if not collides:
            return None

        normal = Vector3(
            player_point[0] - wall_point[0],
            0.0,
            player_point[1] - wall_point[1],
        )
        if normal.length_squared() <= eps:
            normal = Vector3(
                player_start[0] - wall_point[0],
                0.0,
                player_start[1] - wall_point[1],
            )
        if normal.length_squared() <= eps:
            normal = Vector3(face_normal.x, 0.0, face_normal.z)
        if normal.length_squared() <= eps:
            return None
        normal = normal.normalize()
        movement = Vector3(
            player_end[0] - player_start[0],
            0.0,
            player_end[1] - player_start[1],
        )
        if movement.dot(normal) > 0.0:
            normal = -normal
        return normal

    def check_face(face_verts: list[Vector3]) -> Optional[Vector3]:
        if len(face_verts) < 3:
            return None

        v0 = face_verts[0]
        v1 = face_verts[1]
        v2 = face_verts[2]
        n = (v1 - v0).cross(v2 - v0)
        nlen = n.length()
        if nlen <= eps:
            return None
        n = n / nlen

        test_old = old_pos
        test_new = new_pos
        if (
            player_bottom_y is not None
            and player_top_y is not None
            and abs(n.y) <= 0.35
        ):
            face_min_y = min(v.y for v in face_verts)
            face_max_y = max(v.y for v in face_verts)
            overlap_min = max(face_min_y, player_bottom_y)
            overlap_max = min(face_max_y, player_top_y)
            if overlap_max < overlap_min:
                return None
            sample_y = (overlap_min + overlap_max) * 0.5
            test_old = Vector3(old_pos.x, sample_y, old_pos.z)
            test_new = Vector3(new_pos.x, sample_y, new_pos.z)
            radius_normal = wall_radius_collision_normal(face_verts, n)
            if radius_normal is not None:
                return radius_normal

        seg = test_new - test_old

        d0 = (test_old - v0).dot(n)
        d1 = (test_new - v0).dot(n)

        axis_u = v1 - v0
        len_u = axis_u.length()
        if len_u <= eps:
            return None
        axis_u_n = axis_u / len_u
        axis_v_temp = v2 - v0
        axis_v_proj = axis_u_n * axis_v_temp.dot(axis_u_n)
        axis_v = axis_v_temp - axis_v_proj
        len_v = axis_v.length()
        if len_v <= eps:
            return None
        axis_v_n = axis_v / len_v

        def project(pt: Vector3) -> tuple[float, float]:
            local = pt - v0
            return (local.dot(axis_u_n), local.dot(axis_v_n))

        if abs(d0) <= eps and abs(d1) <= eps:
            mid = test_old + seg * 0.5
            p2 = project(mid)
            poly2d = [project(v) for v in face_verts]
            if point_in_poly_2d(p2[0], p2[1], poly2d):
                return n
            return None

        if d0 * d1 > 0:
            if abs(d1) <= player_radius and abs(d1) < abs(d0):
                p_plane = test_new - n * d1
                p2 = project(p_plane)
                poly2d = [project(v) for v in face_verts]
                if point_in_poly_2d(p2[0], p2[1], poly2d):
                    return n
            return None

        denom = d0 - d1
        if abs(denom) <= eps:
            return None
        t = d0 / denom
        if t < 0.0 - eps or t > 1.0 + eps:
            return None
        p = test_old + seg * t
        p2 = project(p)
        poly2d = [project(v) for v in face_verts]
        if point_in_poly_2d(p2[0], p2[1], poly2d):
            return n
        return None

    for mesh in meshes:
        if hasattr(mesh, "get_bounding_box"):
            try:
                if not bounds_overlap(mesh.get_bounding_box()):
                    continue
            except Exception:
                pass

        if not hasattr(mesh, "get_world_vertices"):
            continue

        verts = mesh.get_world_vertices()
        if not verts:
            continue

        faces = getattr(mesh, "faces", None)
        if not faces:
            if len(verts) >= 4:
                faces = [list(range(4))]
            else:
                continue

        for face in faces:
            try:
                face_verts = [verts[i] for i in face]
            except Exception:
                continue
            normal = check_face(face_verts)
            if normal is not None:
                return normal
    return None


__all__ = [
    "VerticalCollision",
    "movement_blocked_by_wall",
    "player_support_height_at",
    "resolve_player_vertical_collision",
]
