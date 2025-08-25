"""Simple collision utilities extracted from WorldScene._movement_blocked_by_wall.

Expose `movement_blocked_by_wall(meshes, old_pos, new_pos, player_radius=16)`.
This keeps WorldScene focused on scene logic while reusing the collision routine.
"""

from __future__ import annotations

from pygame.math import Vector3
from typing import Iterable, Optional
from world.objects import WallTile
from world.objects.polygon import Polygon


def movement_blocked_by_wall(
    meshes: Iterable, old_pos: Vector3, new_pos: Vector3, player_radius: float = 16.0
) -> Optional[Vector3]:
    """Return the collision plane normal (pygame.Vector3) if segment old_pos->new_pos
    intersects any WallTile quad, otherwise return None.

    Logic mirrors the implementation previously inside WorldScene but exposes the
    plane normal so callers can compute sliding vectors instead of just aborting.
    """
    seg = new_pos - old_pos
    eps = 1e-6

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

        d0 = (old_pos - v0).dot(n)
        d1 = (new_pos - v0).dot(n)

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
            mid = old_pos + seg * 0.5
            p2 = project(mid)
            poly2d = [project(v) for v in face_verts]
            if point_in_poly_2d(p2[0], p2[1], poly2d):
                return n
            return None

        # If both on same side of plane -> maybe grazing within player_radius
        if d0 * d1 > 0:
            if abs(d1) <= player_radius and abs(d1) < abs(d0):
                p_plane = new_pos - n * d1
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
        p = old_pos + seg * t
        p2 = project(p)
        poly2d = [project(v) for v in face_verts]
        if point_in_poly_2d(p2[0], p2[1], poly2d):
            return n
        return None

    for m in meshes:
        # Accept WallTile and any polygonal mesh that exposes .faces
        if not (isinstance(m, WallTile) or isinstance(m, Polygon) or hasattr(m, 'faces')):
            continue
        verts = m.get_world_vertices()
        if not verts:
            continue

        faces = getattr(m, 'faces', None)
        if not faces:
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
