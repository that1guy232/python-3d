"""Simple collision utilities extracted from WorldScene._movement_blocked_by_wall.

Expose `movement_blocked_by_wall(meshes, old_pos, new_pos, player_radius=16)`.
This keeps WorldScene focused on scene logic while reusing the collision routine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pygame.math import Vector3
from typing import Iterable, Optional
from world.objects import WallTile
from world.objects.polygon import Polygon
from core.mesh import BatchedMesh
from engine.rendering.sprite import WorldSprite


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
        min_x - radius <= x <= max_x + radius
        and min_z - radius <= z <= max_z + radius
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
    """Return the collision plane normal (pygame.Vector3) if segment old_pos->new_pos
    intersects any WallTile quad, otherwise return None.

    Logic mirrors the implementation previously inside WorldScene but exposes the
    plane normal so callers can compute sliding vectors instead of just aborting.
    """
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
        """Ray-casting point-in-polygon test (2D)."""
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

    def check_face(face_verts: list[Vector3]) -> Optional[Vector3]:
        """Check a single polygonal face (list of world-space verts).
        Returns plane normal if collision occurs, otherwise None.
        """
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

        seg = test_new - test_old

        d0 = (test_old - v0).dot(n)
        d1 = (test_new - v0).dot(n)

        # Prepare 2D projection basis on the face plane using Gram-Schmidt
        axis_u = (v1 - v0)
        len_u = axis_u.length()
        if len_u <= eps:
            return None
        axis_u_n = axis_u / len_u
        axis_v_temp = (v2 - v0)
        # make axis_v orthogonal to axis_u
        axis_v_proj = axis_u_n * axis_v_temp.dot(axis_u_n)
        axis_v = axis_v_temp - axis_v_proj
        len_v = axis_v.length()
        if len_v <= eps:
            return None
        axis_v_n = axis_v / len_v

        def project(pt: Vector3) -> tuple[float, float]:
            local = pt - v0
            return (local.dot(axis_u_n), local.dot(axis_v_n))

        # Case: segment lies in plane (both distances ~0)
        if abs(d0) <= eps and abs(d1) <= eps:
            mid = test_old + seg * 0.5
            p2 = project(mid)
            poly2d = [project(v) for v in face_verts]
            if point_in_poly_2d(p2[0], p2[1], poly2d):
                return n
            return None

        # If both on same side of plane -> maybe grazing within player_radius
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

    for m in meshes:
        if hasattr(m, "get_bounding_box"):
            try:
                if not bounds_overlap(m.get_bounding_box()):
                    continue
            except Exception:
                pass

        # if it has #get_world_Vertices() we can continues 
        if not hasattr(m, 'get_world_vertices'):
            print(m)
            print("Mesh does not have get_world_vertices, skipping.")
            continue

        verts = m.get_world_vertices()
        if not verts:
            continue
        #print(verts)

        faces = getattr(m, 'faces', None)
        if not faces:
            print("No faces found, falling back to simple quad.")
            # Fall back for simple quad-like meshes (old behavior)
            if len(verts) >= 4:
                faces = [list(range(4))]
            else:
                continue

        for face in faces:
            # Build face vertex list, skipping invalid indices
            try:
                face_verts = [verts[i] for i in face]
            except Exception:
                # malformed face indices
                continue
            n = check_face(face_verts)
            if n is not None:
                return n
    return None
