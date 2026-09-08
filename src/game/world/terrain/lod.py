"""Distance-selected mesh groups for batched procedural nature.

Procedural models are baked in world space, so selecting LOD per instance
would turn thousands of plants into thousands of draw calls.  This module
keeps nearby objects spatially batched: every chunk owns three VBOs and picks
one of them from camera distance each frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Iterator, Sequence

from engine.rendering.mesh import BatchedMesh


LOD_HIGH = 0
LOD_MEDIUM = 1
LOD_LOW = 2
LOD_LEVEL_COUNT = 3


@dataclass(frozen=True, slots=True)
class NatureLODPolicy:
    """Distance thresholds and shadow simplification for one model family."""

    distances: tuple[float, float, float]
    shadow_distance: float = 0.0
    shadow_lod_bias: int = 1

    def __post_init__(self) -> None:
        distances = tuple(float(value) for value in self.distances)
        if len(distances) != LOD_LEVEL_COUNT:
            raise ValueError("nature LOD requires exactly three distance stages")
        if not all(math.isfinite(value) and value > 0.0 for value in distances):
            raise ValueError("nature LOD distances must be positive and finite")
        if not distances[0] < distances[1] < distances[2]:
            raise ValueError("nature LOD distances must be strictly increasing")
        shadow_distance = float(self.shadow_distance)
        if not math.isfinite(shadow_distance) or shadow_distance < 0.0:
            raise ValueError("nature shadow distance must be finite and non-negative")
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "shadow_distance", shadow_distance)
        object.__setattr__(
            self,
            "shadow_lod_bias",
            max(0, min(LOD_LOW, int(self.shadow_lod_bias))),
        )


def _union_spheres(
    meshes: Sequence[BatchedMesh | None],
) -> tuple[tuple[float, float, float] | None, float]:
    """Return one conservative sphere enclosing the available LOD meshes."""

    spheres = [
        (mesh.bounds_center, max(0.0, float(mesh.bounds_radius)))
        for mesh in meshes
        if mesh is not None and mesh.bounds_center is not None
    ]
    if not spheres:
        return None, 0.0

    center, radius = spheres[0]
    center = tuple(float(value) for value in center)
    for other_center, other_radius in spheres[1:]:
        dx = float(other_center[0]) - center[0]
        dy = float(other_center[1]) - center[1]
        dz = float(other_center[2]) - center[2]
        separation = math.sqrt(dx * dx + dy * dy + dz * dz)
        if separation + other_radius <= radius:
            continue
        if separation + radius <= other_radius:
            center = tuple(float(value) for value in other_center)
            radius = other_radius
            continue
        combined_radius = (radius + separation + other_radius) * 0.5
        if separation > 1.0e-8:
            shift = (combined_radius - radius) / separation
            center = (
                center[0] + dx * shift,
                center[1] + dy * shift,
                center[2] + dz * shift,
            )
        radius = combined_radius
    return center, radius


@dataclass(slots=True)
class NatureLODGroup:
    """Three versions of one spatial chunk with shared selection policy."""

    levels: tuple[BatchedMesh | None, BatchedMesh | None, BatchedMesh | None]
    policy: NatureLODPolicy
    family: str = "nature"
    visible: bool = True
    bounds_center: tuple[float, float, float] | None = field(init=False)
    bounds_radius: float = field(init=False)

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        if len(levels) != LOD_LEVEL_COUNT:
            raise ValueError("NatureLODGroup requires high, medium, and low meshes")
        if not any(mesh is not None for mesh in levels):
            raise ValueError("NatureLODGroup requires at least one mesh")
        self.levels = levels
        self.family = str(self.family or "nature")
        self.bounds_center, self.bounds_radius = _union_spheres(levels)

    @property
    def casts_shadows(self) -> bool:
        return any(
            mesh is not None and bool(getattr(mesh, "casts_shadows", True))
            for mesh in self.levels
        )

    @property
    def casts_sun_shadows(self) -> bool:
        return any(
            mesh is not None and bool(getattr(mesh, "casts_sun_shadows", True))
            for mesh in self.levels
        )

    @property
    def casts_point_shadows(self) -> bool:
        return any(
            mesh is not None and bool(getattr(mesh, "casts_point_shadows", True))
            for mesh in self.levels
        )

    @property
    def shadow_revision(self) -> tuple:
        return (
            self.policy.distances,
            self.policy.shadow_distance,
            self.policy.shadow_lod_bias,
            tuple(
                (
                    id(mesh),
                    int(getattr(mesh, "vbo_vertices", 0) or 0),
                    int(getattr(mesh, "vertex_count", 0) or 0),
                )
                if mesh is not None
                else None
                for mesh in self.levels
            ),
        )

    def iter_meshes(self) -> Iterator[BatchedMesh]:
        """Yield all allocated levels exactly once."""

        seen: set[int] = set()
        for mesh in self.levels:
            if mesh is None or id(mesh) in seen:
                continue
            seen.add(id(mesh))
            yield mesh

    def _distance(self, camera) -> float:
        if camera is None or self.bounds_center is None:
            return 0.0
        position = getattr(camera, "position", None)
        if position is None:
            return 0.0
        dx = self.bounds_center[0] - float(position.x)
        dz = self.bounds_center[2] - float(position.z)
        # Distance to the chunk's conservative near edge avoids demoting a
        # close object merely because the center of its batch is farther away.
        return max(0.0, math.hypot(dx, dz) - self.bounds_radius)

    def lod_level(
        self,
        camera,
        *,
        distance_limit: float | None = None,
        lod_bias: int = 0,
    ) -> int | None:
        distance = self._distance(camera)
        maximum = self.policy.distances[LOD_LOW]
        if distance_limit is not None:
            maximum = min(maximum, max(0.0, float(distance_limit)))
        if distance > maximum:
            return None

        if distance <= self.policy.distances[LOD_HIGH]:
            level = LOD_HIGH
        elif distance <= self.policy.distances[LOD_MEDIUM]:
            level = LOD_MEDIUM
        else:
            level = LOD_LOW
        return min(LOD_LOW, level + max(0, int(lod_bias)))

    def select_mesh(
        self,
        camera=None,
        *,
        view_distance: float | None = None,
        lod_bias: int = 0,
    ) -> tuple[BatchedMesh | None, int | None]:
        if not self.visible:
            return None, None
        level = self.lod_level(
            camera,
            distance_limit=view_distance,
            lod_bias=lod_bias,
        )
        if level is None:
            return None, None

        # A generator may intentionally omit a stage.  Prefer a cheaper stage
        # first, then fall back toward high detail so selection stays robust.
        for candidate in range(level, LOD_LEVEL_COUNT):
            mesh = self.levels[candidate]
            if mesh is not None and mesh.vertex_count > 0 and mesh.vbo_vertices:
                return mesh, candidate
        for candidate in range(level - 1, -1, -1):
            mesh = self.levels[candidate]
            if mesh is not None and mesh.vertex_count > 0 and mesh.vbo_vertices:
                return mesh, candidate
        return None, None

    def shadow_meshes(self, camera=None) -> tuple[BatchedMesh, ...]:
        if not self.casts_shadows or self.policy.shadow_distance <= 0.0:
            return ()
        mesh, _level = self.select_mesh(
            camera,
            view_distance=self.policy.shadow_distance,
            lod_bias=self.policy.shadow_lod_bias,
        )
        if mesh is None or not getattr(mesh, "casts_shadows", True):
            return ()
        return (mesh,)

    def set_exposure(self, exposure: float, *, baseline: float | None = None) -> None:
        for mesh in self.iter_meshes():
            mesh.set_exposure(exposure, baseline=baseline)

    def dispose(self) -> None:
        for mesh in self.iter_meshes():
            mesh.dispose()


@dataclass(frozen=True, slots=True)
class NatureLODSelection:
    meshes: tuple[BatchedMesh, ...]
    level_counts: tuple[int, int, int]
    culled_groups: int
    selected_vertices: int


def select_nature_lod_meshes(
    sources: Iterable[object],
    camera=None,
    *,
    view_distance: float | None = None,
) -> NatureLODSelection:
    """Resolve LOD groups while passing ordinary meshes through unchanged."""

    selected: list[BatchedMesh] = []
    counts = [0, 0, 0]
    culled = 0
    vertices = 0
    for source in sources or ():
        if source is None:
            continue
        if isinstance(source, NatureLODGroup):
            mesh, level = source.select_mesh(
                camera,
                view_distance=view_distance,
            )
            if mesh is None or level is None:
                culled += 1
                continue
            counts[level] += 1
        elif isinstance(source, BatchedMesh):
            mesh = source
            counts[LOD_HIGH] += 1
        else:
            continue
        if not mesh.is_visible(camera, view_distance=view_distance):
            culled += 1
            if isinstance(source, NatureLODGroup):
                counts[level] -= 1
            else:
                counts[LOD_HIGH] -= 1
            continue
        selected.append(mesh)
        vertices += int(mesh.vertex_count)
    return NatureLODSelection(
        meshes=tuple(selected),
        level_counts=tuple(counts),
        culled_groups=culled,
        selected_vertices=vertices,
    )


def iter_all_lod_meshes(sources: Iterable[object]) -> Iterator[BatchedMesh]:
    """Flatten LOD resources for validation and non-render maintenance."""

    for source in sources or ():
        if isinstance(source, NatureLODGroup):
            yield from source.iter_meshes()
        elif isinstance(source, BatchedMesh):
            yield source


__all__ = [
    "LOD_HIGH",
    "LOD_LEVEL_COUNT",
    "LOD_LOW",
    "LOD_MEDIUM",
    "NatureLODGroup",
    "NatureLODPolicy",
    "NatureLODSelection",
    "iter_all_lod_meshes",
    "select_nature_lod_meshes",
]
