"""Simple collision utilities extracted from WorldScene._movement_blocked_by_wall.

Expose `movement_blocked_by_wall(meshes, old_pos, new_pos, player_radius=16)`.
This keeps WorldScene focused on scene logic while reusing the collision routine.
"""

from __future__ import annotations

from pygame.math import Vector3
from typing import Iterable, Optional
from world.objects import WallTile


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
    for m in meshes:
        if not isinstance(m, WallTile):
            continue
        verts = m.get_world_vertices()
        if not verts or len(verts) < 4:
            continue
        v0, v1, v2, v3 = verts[:4]
        n = (v1 - v0).cross(v3 - v0)
        nlen = n.length()
        if nlen <= eps:
            continue
        n = n / nlen
        d0 = (old_pos - v0).dot(n)
        d1 = (new_pos - v0).dot(n)
        if abs(d0) <= eps and abs(d1) <= eps:
            mid = old_pos + (new_pos - old_pos) * 0.5
            axis_u = v1 - v0
            len_u = axis_u.length()
            if len_u <= eps:
                continue
            axis_u_n = axis_u / len_u
            axis_v = v3 - v0
            len_v = axis_v.length()
            if len_v <= eps:
                continue
            axis_v_n = axis_v / len_v
            local = mid - v0
            u = local.dot(axis_u_n)
            v = local.dot(axis_v_n)
            if -eps <= u <= len_u + eps and -eps <= v <= len_v + eps:
                return n
            continue
        if d0 * d1 > 0:
            if abs(d1) <= player_radius and abs(d1) < abs(d0):
                p_plane = new_pos - n * d1
                axis_u = v1 - v0
                len_u = axis_u.length()
                if len_u <= eps:
                    continue
                axis_u_n = axis_u / len_u
                axis_v = v3 - v0
                len_v = axis_v.length()
                if len_v <= eps:
                    continue
                axis_v_n = axis_v / len_v
                local = p_plane - v0
                u = local.dot(axis_u_n)
                v = local.dot(axis_v_n)
                if -eps <= u <= len_u + eps and -eps <= v <= len_v + eps:
                    return n
            continue
        denom = d0 - d1
        if abs(denom) <= eps:
            continue
        t = d0 / denom
        if t < 0.0 - eps or t > 1.0 + eps:
            continue
        p = old_pos + seg * t
        axis_u = v1 - v0
        len_u = axis_u.length()
        if len_u <= eps:
            continue
        axis_u_n = axis_u / len_u
        axis_v = v3 - v0
        len_v = axis_v.length()
        if len_v <= eps:
            continue
        axis_v_n = axis_v / len_v
        local = p - v0
        u = local.dot(axis_u_n)
        v = local.dot(axis_v_n)
        if -eps <= u <= len_u + eps and -eps <= v <= len_v + eps:
            return n
        return None
