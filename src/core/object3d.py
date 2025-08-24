from pygame.math import Vector3
import math


class Object3D:
    def __init__(self, position=None, rotation=None):
        self.position = position or Vector3(0, 0, 0)
        self.rotation = rotation or Vector3(0, 0, 0)
        self.local_vertices = []

    def get_world_vertices(self):
        """Get vertices in world space (with rotation and position)"""
        world_vertices = []
        for vertex in self.local_vertices:
            # Apply rotation
            rotated = Vector3(
                vertex.x * math.cos(self.rotation.y)
                - vertex.z * math.sin(self.rotation.y),
                vertex.y,
                vertex.x * math.sin(self.rotation.y)
                + vertex.z * math.cos(self.rotation.y),
            )
            # Apply position
            world_vertex = Vector3(
                rotated.x + self.position.x,
                rotated.y + self.position.y,
                rotated.z + self.position.z,
            )
            world_vertices.append(world_vertex)
        return world_vertices
