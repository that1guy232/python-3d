"""Spatial collision candidate index for WorldScene."""

from __future__ import annotations

import math

DEFAULT_COLLISION_CELL_SIZE = 128.0
COLLISION_FALLBACK_CELL_LIMIT = 256


class SceneCollisionIndex:
    """Build and query a coarse collision mesh spatial index."""

    def __init__(self, scene) -> None:
        self.scene = scene

    def iter_sources(self):
        scene = self.scene
        seen: set[int] = set()
        for mesh in getattr(scene, "wall_tiles", ()) or ():
            if mesh is None:
                continue
            mesh_id = id(mesh)
            if mesh_id in seen:
                continue
            seen.add(mesh_id)
            yield mesh, False
        for mesh in getattr(scene, "polygons", ()) or ():
            if mesh is None:
                continue
            mesh_id = id(mesh)
            if mesh_id in seen:
                continue
            seen.add(mesh_id)
            yield mesh, True

    def source_key(self):
        scene = self.scene
        wall_tiles = getattr(scene, "wall_tiles", ()) or ()
        polygons = getattr(scene, "polygons", ()) or ()

        def edge_ids(values):
            if not values:
                return (None, None)
            return (id(values[0]), id(values[-1]))

        return (
            id(wall_tiles),
            len(wall_tiles),
            *edge_ids(wall_tiles),
            id(polygons),
            len(polygons),
            *edge_ids(polygons),
        )

    @staticmethod
    def mesh_dynamic(mesh) -> bool:
        return (
            getattr(mesh, "door_render_batched", False)
            or hasattr(mesh, "open_amount")
            or type(mesh).__name__ == "Door"
        )

    @staticmethod
    def mesh_bounds(mesh):
        get_bounds = getattr(mesh, "get_bounding_box", None)
        if not callable(get_bounds):
            return None
        try:
            bounds = get_bounds()
        except Exception:
            return None
        if not bounds:
            return None
        try:
            min_x, max_x, min_z, max_z = bounds
            return (float(min_x), float(max_x), float(min_z), float(max_z))
        except Exception:
            return None

    def rebuild(self) -> dict:
        scene = self.scene
        cell_size = float(
            getattr(scene, "collision_cell_size", DEFAULT_COLLISION_CELL_SIZE)
        )
        if cell_size <= 1.0:
            cell_size = DEFAULT_COLLISION_CELL_SIZE

        cells: dict[tuple[int, int], list] = {}
        wall_cells: dict[tuple[int, int], list] = {}
        dynamic = []
        wall_dynamic = []
        fallback = []
        wall_fallback = []

        def add_to_grid(grid, mesh, bounds) -> bool:
            min_x, max_x, min_z, max_z = bounds
            min_cx = math.floor(min_x / cell_size)
            max_cx = math.floor(max_x / cell_size)
            min_cz = math.floor(min_z / cell_size)
            max_cz = math.floor(max_z / cell_size)
            cell_count = (max_cx - min_cx + 1) * (max_cz - min_cz + 1)
            if cell_count > COLLISION_FALLBACK_CELL_LIMIT:
                return False
            for cx in range(min_cx, max_cx + 1):
                for cz in range(min_cz, max_cz + 1):
                    grid.setdefault((cx, cz), []).append(mesh)
            return True

        for mesh, is_polygon in self.iter_sources():
            is_wall = not is_polygon
            if self.mesh_dynamic(mesh):
                dynamic.append(mesh)
                if is_wall:
                    wall_dynamic.append(mesh)
                continue

            bounds = self.mesh_bounds(mesh)
            if bounds is None:
                fallback.append(mesh)
                if is_wall:
                    wall_fallback.append(mesh)
                continue

            if not add_to_grid(cells, mesh, bounds):
                fallback.append(mesh)
            if is_wall and not add_to_grid(wall_cells, mesh, bounds):
                wall_fallback.append(mesh)

        index = {
            "key": self.source_key(),
            "cell_size": cell_size,
            "cells": cells,
            "wall_cells": wall_cells,
            "dynamic": tuple(dynamic),
            "wall_dynamic": tuple(wall_dynamic),
            "fallback": tuple(fallback),
            "wall_fallback": tuple(wall_fallback),
        }
        scene._collision_spatial_index = index
        return index

    def invalidate(self) -> None:
        self.scene._collision_spatial_index = None

    def index(self) -> dict:
        scene = self.scene
        index = getattr(scene, "_collision_spatial_index", None)
        key = self.source_key()
        if not isinstance(index, dict) or index.get("key") != key:
            index = self.rebuild()
        return index

    def meshes_for_bounds(
        self,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
        *,
        include_polygons: bool = True,
    ) -> list:
        index = self.index()
        cell_size = float(index.get("cell_size") or DEFAULT_COLLISION_CELL_SIZE)
        cells = index["cells"] if include_polygons else index["wall_cells"]
        dynamic = index["dynamic"] if include_polygons else index["wall_dynamic"]
        fallback = index["fallback"] if include_polygons else index["wall_fallback"]

        min_cx = math.floor(float(min_x) / cell_size)
        max_cx = math.floor(float(max_x) / cell_size)
        min_cz = math.floor(float(min_z) / cell_size)
        max_cz = math.floor(float(max_z) / cell_size)

        candidates = []
        seen: set[int] = set()

        def add(mesh) -> None:
            mesh_id = id(mesh)
            if mesh_id in seen:
                return
            seen.add(mesh_id)
            candidates.append(mesh)

        for cx in range(min_cx, max_cx + 1):
            for cz in range(min_cz, max_cz + 1):
                for mesh in cells.get((cx, cz), ()):
                    add(mesh)
        for mesh in dynamic:
            add(mesh)
        for mesh in fallback:
            add(mesh)

        return candidates

    def meshes_at(
        self,
        x: float,
        z: float,
        radius: float,
        *,
        include_polygons: bool = True,
    ) -> list:
        x = float(x)
        z = float(z)
        radius = max(0.0, float(radius))
        return self.meshes_for_bounds(
            x - radius,
            x + radius,
            z - radius,
            z + radius,
            include_polygons=include_polygons,
        )
