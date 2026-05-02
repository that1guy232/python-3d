from pygame.math import Vector3
import math


class Object3D:
    def __init__(self, position=None, rotation=None):
        self.position = position or Vector3(0, 0, 0)
        self.rotation = rotation or Vector3(0, 0, 0)
        self.local_vertices = []
        self._rotation_cache = (None, None, None)
        self._rotation_matrix = None
        self._world_vertices_cache = None
        self._world_vertices_cache_key = None
        self._bounds_cache = None

    def _update_rotation_matrix(self):
        """Cache sin/cos values and the rotation matrix for the current rotation."""
        rx = self.rotation.x
        ry = self.rotation.y
        rz = self.rotation.z

        if self._rotation_cache == (rx, ry, rz) and self._rotation_matrix is not None:
            return

        self._rotation_cache = (rx, ry, rz)

        cx = math.cos(rx)
        sx = math.sin(rx)
        cy = math.cos(ry)
        sy = math.sin(ry)
        cz = math.cos(rz)
        sz = math.sin(rz)

        # Compose rotation: yaw (Y), then pitch (X), then roll (Z).
        self._rotation_matrix = (
            cz * cy + sz * sx * sy,
            -sz * cx,
            -cz * sy + sz * sx * cy,
            sz * cy - cz * sx * sy,
            cz * cx,
            -sz * sy - cz * sx * cy,
            cx * sy,
            sx,
            cx * cy,
        )

    def _rotate_vertex(self, vertex: Vector3) -> Vector3:
        """Rotate a vertex using the cached rotation matrix."""
        self._update_rotation_matrix()
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = self._rotation_matrix

        return Vector3(
            vertex.x * r00 + vertex.y * r01 + vertex.z * r02,
            vertex.x * r10 + vertex.y * r11 + vertex.z * r12,
            vertex.x * r20 + vertex.y * r21 + vertex.z * r22,
        )

    def get_world_vertices(self):
        """Get vertices in world space (with rotation and position)"""
        cache_key = self._transform_cache_key()
        if (
            self._world_vertices_cache_key == cache_key
            and self._world_vertices_cache is not None
        ):
            return self._world_vertices_cache

        world_vertices = []
        for vertex in self.local_vertices:
            rotated = self._rotate_vertex(vertex)
            world_vertices.append(rotated + self.position)
        self._world_vertices_cache_key = cache_key
        self._world_vertices_cache = world_vertices
        self._bounds_cache = None
        return world_vertices

    def _transform_cache_key(self):
        return (
            self.position.x,
            self.position.y,
            self.position.z,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
            id(self.local_vertices),
            len(self.local_vertices),
        )

    def get_bounding_box(self):
        """Return cached XZ bounds as (min_x, max_x, min_z, max_z)."""
        if (
            self._bounds_cache is not None
            and self._world_vertices_cache_key == self._transform_cache_key()
        ):
            return self._bounds_cache

        verts = self.get_world_vertices()
        if not verts:
            return None

        min_x = min(v.x for v in verts)
        max_x = max(v.x for v in verts)
        min_z = min(v.z for v in verts)
        max_z = max(v.z for v in verts)
        self._bounds_cache = (min_x, max_x, min_z, max_z)
        return self._bounds_cache
