from pygame.math import Vector3
from core.object3d import Object3D


class GroundTile(Object3D):
    """A single, flat ground plane (quad) used as part of the scene floor.

    Parameters
    ----------
    position : Vector3 | None
        World-space position (center) of the tile.
    width : float
        Half-width in X direction (extends +/- width).
    height : float
        Height of the plane above the local origin (the plane lies at y=+height).
        Kept for compatibility with existing callers that expect the top surface at +5.
    depth : float
        Half-depth in Z direction (extends +/- depth).
    """

    def __init__(self, position=None, width=50, height=5, depth=50):
        self.width = width
        self.height = height
        self.depth = depth

        super().__init__(position=position or Vector3(0, 0, 0))
        self._generate_vertices()

        # Single top-facing quad
        self.faces = [
            (0, 1, 2, 3),  # top plane (y = +height)
        ]
        white = (255, 255, 255)
        self.face_colors = [
            white,
        ]

    def _generate_vertices(self):
        w = self.width
        y = self.height  # plane at y = +height
        d = self.depth
        # Local space vertices of the plane (counter-clockwise when looking down -Y)
        #   3 ---- 2
        #   |      |
        #   0 ---- 1
        self.local_vertices = [
            Vector3(-w, y, -d),
            Vector3(w, y, -d),
            Vector3(w, y, d),
            Vector3(-w, y, d),
        ]
