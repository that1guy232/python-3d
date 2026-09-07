"""Scene-independent, incrementally refreshed sun and point-light shadows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
import math

from engine.core.mesh import BatchedMesh
from engine.rendering.packet_shader import MAX_PACKET_POINT_LIGHTS
from engine.rendering.directional_shadow import (
    DirectionalShadowBinding, DirectionalShadowMap, ShadowCasterShader,
    directional_light_matrix, directional_shadow_bias,
)
from engine.rendering.point_shadow import (
    POINT_SHADOW_FACE_COUNT, PointShadowBinding, PointShadowCasterShader,
    PointShadowMap, point_light_face_matrices,
)


@dataclass(slots=True)
class _SunShadowCache:
    caster_revision: tuple
    center: tuple[float, float, float]
    direction: tuple[float, float, float]
    binding: DirectionalShadowBinding


@dataclass(slots=True)
class _SunShadowRefresh:
    """One coherent directional map snapshot being built in staging."""

    caster_revision: tuple
    center: tuple[float, float, float]
    direction: tuple[float, float, float]
    light_matrix: tuple[float, ...]
    bias: float
    casters: tuple[BatchedMesh, ...]
    next_caster: int = 0


@dataclass(frozen=True, slots=True)
class _ShadowViewpoint:
    """Minimal camera-shaped position used for light-relative shadow LOD."""

    x: float
    y: float
    z: float

    @property
    def position(self) -> "_ShadowViewpoint":
        return self


@dataclass(slots=True)
class _PointShadowCache:
    light_id: str
    light_revision: tuple
    caster_revision: tuple
    binding: PointShadowBinding


@dataclass(slots=True)
class _PointShadowRefresh:
    """One immutable cubemap target being built a face at a time."""

    light_id: str
    light_revision: tuple
    caster_revision: tuple
    position: tuple[float, float, float]
    light_range: float
    matrices: tuple[tuple[float, ...], ...]
    next_face: int = 0


class SceneShadowRenderer:
    """Own shadow maps for any scene exposing an iterable of shadow casters.

    Sources may be BatchedMesh objects or implement shadow_meshes(camera=...).
    Construction is CPU-only; render() lazily allocates context-owned resources.
    """

    def __init__(self, *, sources: Callable[[], Iterable[object]] = tuple,
                 view_distance: float = 2000.0) -> None:
        self.sources = sources
        self.camera = None
        self.lighting = None
        self.packet_shader = None
        self.profiler = None
        self._sun_shadow_map: DirectionalShadowMap | None = None
        self._sun_shadow_staging_map: DirectionalShadowMap | None = None
        self._shadow_caster_shader: ShadowCasterShader | None = None
        self._sun_shadow_cache: _SunShadowCache | None = None
        self._sun_shadow_refresh: _SunShadowRefresh | None = None
        self._sun_shadow_pending: _SunShadowRefresh | None = None
        self._point_shadow_maps: list[PointShadowMap] = []
        self._point_shadow_staging_maps: list[PointShadowMap] = []
        self._point_shadow_caster_shader: PointShadowCasterShader | None = None
        self._point_shadow_cache: list[_PointShadowCache | None] = []
        self._point_shadow_refreshes: list[_PointShadowRefresh | None] = []
        self._point_shadow_update_cursor = 0
        self.sun_shadow_map_size = 2048
        self.sun_shadow_extent = min(float(view_distance), 1000.0)
        self.sun_shadow_camera_threshold = 16.0
        self.sun_shadow_direction_threshold_degrees = 0.25
        self.sun_shadow_vertex_budget = 180_000
        self.sun_shadow_draw_budget = 32
        self.point_shadow_map_size = 256
        self.max_point_shadows = 2
        self.point_shadow_faces_per_frame = 1

    def render(self, *, camera, lighting, packet_shader, profiler=None) -> None:
        """Bind this scene's complete maps; None lighting selects an unlit view."""
        self.camera = camera
        self.lighting = lighting
        self.packet_shader = packet_shader
        self.profiler = profiler
        with self._profile("render.sun_shadow"):
            self.render_sun_shadow()
        with self._profile("render.point_shadows"):
            self.render_point_shadows()

    def _profile(self, name: str):
        if self.profiler is None or not self.profiler.enabled:
            return nullcontext()
        return self.profiler.section(name)

    def _count(self, name: str, amount: float = 1.0) -> None:
        if self.profiler is not None and self.profiler.enabled:
            self.profiler.count(name, amount)

    def _ensure_sun_shadow_resources(
        self,
    ) -> tuple[DirectionalShadowMap, DirectionalShadowMap, ShadowCasterShader]:
        created_active = False
        if self._sun_shadow_map is None:
            self._sun_shadow_map = DirectionalShadowMap.create(
                self.sun_shadow_map_size
            )
            created_active = True
        if self._sun_shadow_staging_map is None:
            try:
                self._sun_shadow_staging_map = DirectionalShadowMap.create(
                    self.sun_shadow_map_size
                )
            except Exception:
                if created_active and self._sun_shadow_map is not None:
                    self._sun_shadow_map.dispose()
                    self._sun_shadow_map = None
                raise
        if self._shadow_caster_shader is None:
            self._shadow_caster_shader = ShadowCasterShader.create()
        return (
            self._sun_shadow_map,
            self._sun_shadow_staging_map,
            self._shadow_caster_shader,
        )

    def _shadow_meshes_from(self, source, *, viewpoint=None):
        if (
            source is None
            or not getattr(source, "visible", True)
            or not getattr(source, "casts_shadows", True)
        ):
            return ()
        provider = getattr(source, "shadow_meshes", None)
        if callable(provider):
            try:
                camera = viewpoint
                if camera is None:
                    camera = self.camera
                return tuple(provider(camera=camera))
            except TypeError:
                return tuple(provider())
        mesh = getattr(source, "_mesh", None)
        if isinstance(mesh, BatchedMesh):
            return mesh.shadow_meshes()
        return ()

    def _shadow_mesh_records(self, *, viewpoint=None):
        """Yield meshes with the revision of the source that owns them."""

        for source in self.sources():
            source_revision = getattr(source, "shadow_revision", None)
            for mesh in self._shadow_meshes_from(source, viewpoint=viewpoint):
                yield mesh, (id(source), source_revision)

    @staticmethod
    def _shadow_mesh_revision(mesh, source_revision) -> tuple:
        """Return a cheap geometry-only cache key for one shadow caster."""

        center = getattr(mesh, "bounds_center", None)
        return (
            source_revision,
            id(mesh),
            getattr(mesh, "shadow_revision", None),
            int(getattr(mesh, "vbo_vertices", 0) or 0),
            int(getattr(mesh, "vertex_count", 0) or 0),
            tuple(float(value) for value in center) if center is not None else None,
            float(getattr(mesh, "bounds_radius", 0.0)),
            bool(getattr(mesh, "casts_shadows", True)),
            bool(getattr(mesh, "casts_sun_shadows", True)),
            bool(getattr(mesh, "casts_point_shadows", True)),
            bool(getattr(mesh, "alpha_test", False)),
            float(getattr(mesh, "alpha_cutoff", 0.0)),
            int(getattr(mesh, "texture", 0) or 0),
        )

    @staticmethod
    def _normalized_direction(direction) -> tuple[float, float, float]:
        values = tuple(float(value) for value in direction)
        length = math.sqrt(sum(value * value for value in values))
        if length <= 1e-8:
            return (0.0, 1.0, 0.0)
        return tuple(value / length for value in values)

    def _sun_shadow_needs_refresh(
        self,
        *,
        caster_revision: tuple,
        center: tuple[float, float, float],
        direction: tuple[float, float, float],
    ) -> bool:
        cache = self._sun_shadow_cache
        if cache is None:
            return True
        return self._sun_shadow_target_needs_refresh(
            cache,
            caster_revision=caster_revision,
            center=center,
            direction=direction,
        )

    def _sun_shadow_target_needs_refresh(
        self,
        target,
        *,
        caster_revision: tuple,
        center: tuple[float, float, float],
        direction: tuple[float, float, float],
    ) -> bool:
        if target.caster_revision != caster_revision:
            return True
        camera_threshold = max(0.0, float(self.sun_shadow_camera_threshold))
        if math.dist(target.center, center) > camera_threshold:
            return True
        angle_threshold = max(
            0.0,
            float(self.sun_shadow_direction_threshold_degrees),
        )
        direction_dot = max(
            -1.0,
            min(
                1.0,
                sum(
                    previous * current
                    for previous, current in zip(target.direction, direction)
                ),
            ),
        )
        return direction_dot < math.cos(math.radians(angle_threshold))

    def _new_sun_shadow_refresh(
        self,
        *,
        caster_revision: tuple,
        center: tuple[float, float, float],
        direction: tuple[float, float, float],
        casters: tuple[BatchedMesh, ...],
    ) -> _SunShadowRefresh:
        """Freeze one coherent target so movement cannot restart it forever."""

        extent = max(50.0, float(self.sun_shadow_extent))
        shadow_near = 1.0
        shadow_far = extent * 4.0
        return _SunShadowRefresh(
            caster_revision=caster_revision,
            center=center,
            direction=direction,
            light_matrix=directional_light_matrix(
                center,
                direction,
                extent=extent,
                near=shadow_near,
                far=shadow_far,
            ),
            bias=directional_shadow_bias(
                near=shadow_near,
                far=shadow_far,
            ),
            casters=casters,
        )

    @staticmethod
    def _sun_shadow_submission_end(
        casters: tuple[BatchedMesh, ...],
        start: int,
        *,
        vertex_budget: int,
        draw_budget: int,
    ) -> int:
        """Return a forward-progressing mesh boundary within both budgets."""

        index = max(0, int(start))
        vertex_limit = max(1, int(vertex_budget))
        draw_limit = max(1, int(draw_budget))
        submitted_vertices = 0
        submitted_draws = 0
        while index < len(casters) and submitted_draws < draw_limit:
            mesh_vertices = max(0, int(casters[index].vertex_count))
            if submitted_draws and submitted_vertices + mesh_vertices > vertex_limit:
                break
            submitted_vertices += mesh_vertices
            submitted_draws += 1
            index += 1
        return index

    def _record_sun_shadow_submission(
        self,
        casters: tuple[BatchedMesh, ...],
    ) -> None:
        self._count("shadow.sun.steps")
        self._count("shadow.sun.submitted_draws", len(casters))
        self._count(
            "shadow.sun.submitted_vertices",
            sum(max(0, int(mesh.vertex_count)) for mesh in casters),
        )

    def _render_initial_sun_shadow(
        self,
        shadow_map: DirectionalShadowMap,
        caster_shader: ShadowCasterShader,
        refresh: _SunShadowRefresh,
    ) -> _SunShadowCache:
        """Populate the first usable map immediately during scene warm-up."""

        with shadow_map.render_depth():
            BatchedMesh.draw_shadows_many(
                refresh.casters,
                caster_shader,
                refresh.light_matrix,
            )
        self._record_sun_shadow_submission(refresh.casters)
        binding = shadow_map.binding(refresh.light_matrix, bias=refresh.bias)
        self._count("shadow.sun.refreshes")
        return _SunShadowCache(
            caster_revision=refresh.caster_revision,
            center=refresh.center,
            direction=refresh.direction,
            binding=binding,
        )

    def _advance_sun_shadow_refresh(
        self,
        staging_map: DirectionalShadowMap,
        caster_shader: ShadowCasterShader,
    ) -> bool:
        """Submit one bounded staging slice and promote only a complete map."""

        refresh = self._sun_shadow_refresh
        if refresh is None:
            return False
        start = refresh.next_caster
        end = self._sun_shadow_submission_end(
            refresh.casters,
            start,
            vertex_budget=self.sun_shadow_vertex_budget,
            draw_budget=self.sun_shadow_draw_budget,
        )
        submission = refresh.casters[start:end]
        with staging_map.render_depth(clear=start == 0):
            BatchedMesh.draw_shadows_many(
                submission,
                caster_shader,
                refresh.light_matrix,
            )
        refresh.next_caster = end
        self._record_sun_shadow_submission(submission)

        if end < len(refresh.casters):
            self._count("shadow.sun.staging_frames")
            return False

        binding = staging_map.binding(refresh.light_matrix, bias=refresh.bias)
        self._sun_shadow_map, self._sun_shadow_staging_map = (
            staging_map,
            self._sun_shadow_map,
        )
        self._sun_shadow_cache = _SunShadowCache(
            caster_revision=refresh.caster_revision,
            center=refresh.center,
            direction=refresh.direction,
            binding=binding,
        )
        self._sun_shadow_refresh = self._sun_shadow_pending
        self._sun_shadow_pending = None
        if self._sun_shadow_refresh is not None:
            self._count("shadow.sun.refresh_starts")
        self._count("shadow.sun.refreshes")
        return True

    def render_sun_shadow(self) -> None:
        """Incrementally refresh sun visibility without exposing partial maps."""

        packet_shader = self.packet_shader
        if packet_shader is None:
            return
        camera = self.camera
        lighting = self.lighting
        if camera is None or lighting is None:
            self._sun_shadow_refresh = None
            self._sun_shadow_pending = None
            packet_shader.set_directional_shadow(None)
            return

        direction = self._normalized_direction(
            getattr(lighting, "light_direction", (0.0, 1.0, 0.0))
        )
        center = (
            float(camera.position.x),
            float(camera.position.y),
            float(camera.position.z),
        )
        shadow_records = tuple(self._shadow_mesh_records())
        sun_casters = tuple(
            (mesh, source_revision)
            for mesh, source_revision in shadow_records
            if getattr(mesh, "casts_sun_shadows", True)
            and int(getattr(mesh, "vertex_count", 0)) > 0
            and int(getattr(mesh, "vbo_vertices", 0) or 0)
        )
        caster_revision = tuple(
            self._shadow_mesh_revision(mesh, source_revision)
            for mesh, source_revision in sun_casters
        )
        casters = tuple(mesh for mesh, _source_revision in sun_casters)
        shadow_map, staging_map, caster_shader = (
            self._ensure_sun_shadow_resources()
        )

        if self._sun_shadow_cache is None:
            initial = self._new_sun_shadow_refresh(
                caster_revision=caster_revision,
                center=center,
                direction=direction,
                casters=casters,
            )
            self._sun_shadow_cache = self._render_initial_sun_shadow(
                shadow_map,
                caster_shader,
                initial,
            )
            self._sun_shadow_refresh = None
            self._sun_shadow_pending = None
            packet_shader.set_directional_shadow(self._sun_shadow_cache.binding)
            return

        refresh = self._sun_shadow_refresh
        if refresh is None and self._sun_shadow_needs_refresh(
            caster_revision=caster_revision,
            center=center,
            direction=direction,
        ):
            # Camera and light motion are frozen into this snapshot.  They may
            # request another refresh after promotion, but cannot keep this one
            # from completing while the player is continuously moving.
            self._sun_shadow_refresh = self._new_sun_shadow_refresh(
                caster_revision=caster_revision,
                center=center,
                direction=direction,
                casters=casters,
            )
            self._count("shadow.sun.refresh_starts")
        elif refresh is not None:
            # Never tear down ordinary in-flight work.  Camera-dependent LOD,
            # caster poses, and daylight can all evolve every frame; restarting
            # would starve the staging map during normal traversal.  Instead,
            # retain only the newest materially different target and begin it
            # after the current coherent snapshot is atomically promoted.
            if self._sun_shadow_target_needs_refresh(
                refresh,
                caster_revision=caster_revision,
                center=center,
                direction=direction,
            ):
                pending = self._sun_shadow_pending
                if pending is None or self._sun_shadow_target_needs_refresh(
                    pending,
                    caster_revision=caster_revision,
                    center=center,
                    direction=direction,
                ):
                    self._sun_shadow_pending = self._new_sun_shadow_refresh(
                        caster_revision=caster_revision,
                        center=center,
                        direction=direction,
                        casters=casters,
                    )
                    self._count("shadow.sun.coalesced_targets")
            else:
                self._sun_shadow_pending = None

        if self._sun_shadow_refresh is None:
            self._count("shadow.sun.cache_hits")
        else:
            self._advance_sun_shadow_refresh(staging_map, caster_shader)
        packet_shader.set_directional_shadow(self._sun_shadow_cache.binding)

    def _selected_point_lights(self, limit: int = MAX_PACKET_POINT_LIGHTS):
        lighting = self.lighting
        lights = tuple(getattr(lighting, "point_lights", ()) or ())
        eligible = [
            light
            for light in lights
            if float(light.intensity) > 0.0
            and float(light.range) > 0.0
        ]
        camera = self.camera
        if camera is None:
            return tuple(eligible[: max(0, int(limit))])

        camera_position = (
            float(camera.position.x),
            float(camera.position.y),
            float(camera.position.z),
        )

        def priority(light):
            distance = math.dist(camera_position, light.position)
            importance = max(0.0001, float(light.importance))
            in_range = distance <= float(light.range)
            return (int(in_range), importance / max(1.0, distance), light.light_id)

        eligible.sort(key=priority, reverse=True)
        return tuple(eligible[: max(0, int(limit))])

    def _ensure_point_shadow_resources(
        self,
        count: int,
    ) -> tuple[tuple[PointShadowMap, ...], PointShadowCasterShader]:
        while len(self._point_shadow_maps) < count:
            active_map = PointShadowMap.create(self.point_shadow_map_size)
            try:
                staging_map = PointShadowMap.create(self.point_shadow_map_size)
            except Exception:
                active_map.dispose()
                raise
            self._point_shadow_maps.append(active_map)
            self._point_shadow_staging_maps.append(staging_map)
            self._point_shadow_cache.append(None)
            self._point_shadow_refreshes.append(None)
        if self._point_shadow_caster_shader is None:
            self._point_shadow_caster_shader = PointShadowCasterShader.create()
        return (
            tuple(self._point_shadow_maps),
            self._point_shadow_caster_shader,
        )

    @staticmethod
    def _mesh_intersects_point_light(mesh, position, light_range: float) -> bool:
        center = getattr(mesh, "bounds_center", None)
        if center is None:
            return True
        radius = max(0.0, float(getattr(mesh, "bounds_radius", 0.0)))
        return math.dist(center, position) <= float(light_range) + radius

    @staticmethod
    def _point_light_revision(light) -> tuple:
        return (
            str(light.light_id),
            tuple(float(value) for value in light.position),
            float(light.range),
        )

    def _assign_point_shadow_maps(self, lights) -> tuple[tuple[object, int], ...]:
        """Keep complete and in-progress maps stable as light priority changes."""

        assignments: list[tuple[object, int] | None] = [None] * len(lights)
        claimed: set[int] = set()

        # A completed map takes precedence over an in-progress reassignment. This
        # keeps a usable shadow bound when both light IDs remain selected.
        for light_index, light in enumerate(lights):
            for map_index, cache in enumerate(self._point_shadow_cache):
                if (
                    map_index not in claimed
                    and cache is not None
                    and cache.light_id == str(light.light_id)
                ):
                    assignments[light_index] = (light, map_index)
                    claimed.add(map_index)
                    break

        # New lights have no completed cache to identify their slot. Retain an
        # in-flight assignment so its six faces can make forward progress even
        # when camera-distance priority changes from frame to frame.
        for light_index, light in enumerate(lights):
            if assignments[light_index] is not None:
                continue
            for map_index, refresh in enumerate(self._point_shadow_refreshes):
                if (
                    map_index not in claimed
                    and refresh is not None
                    and refresh.light_id == str(light.light_id)
                ):
                    assignments[light_index] = (light, map_index)
                    claimed.add(map_index)
                    break

        available = (
            index
            for index in range(len(self._point_shadow_maps))
            if index not in claimed
        )
        for light_index, light in enumerate(lights):
            if assignments[light_index] is None:
                map_index = next(available)
                assignments[light_index] = (light, map_index)
                claimed.add(map_index)
        return tuple(
            assignment for assignment in assignments if assignment is not None
        )

    def _prepare_point_shadow_refreshes(
        self,
        assignments,
        revisions,
    ) -> None:
        """Create or restart immutable staging targets for invalidated slots."""

        assigned_indices = {map_index for _light, map_index in assignments}
        for map_index in range(len(self._point_shadow_refreshes)):
            if map_index not in assigned_indices:
                self._point_shadow_refreshes[map_index] = None

        for light, map_index in assignments:
            light_revision, caster_revision = revisions[map_index]
            cache = self._point_shadow_cache[map_index]
            light_id = str(light.light_id)
            if (
                cache is not None
                and cache.light_id == light_id
                and cache.light_revision == light_revision
                and cache.caster_revision == caster_revision
            ):
                self._point_shadow_refreshes[map_index] = None
                continue

            refresh = self._point_shadow_refreshes[map_index]
            if (
                refresh is not None
                and refresh.light_id == light_id
                and refresh.light_revision == light_revision
                and refresh.caster_revision == caster_revision
            ):
                continue

            light_range = max(0.1, float(light.range))
            position = tuple(float(value) for value in light.position)
            near = max(0.1, min(1.0, light_range * 0.01))
            self._point_shadow_refreshes[map_index] = _PointShadowRefresh(
                light_id=light_id,
                light_revision=light_revision,
                caster_revision=caster_revision,
                position=position,
                light_range=light_range,
                matrices=point_light_face_matrices(
                    position,
                    near=near,
                    far=light_range,
                ),
            )

    def _render_point_shadow_refresh_faces(
        self,
        casters_by_map,
        caster_shader: PointShadowCasterShader,
    ) -> set[int]:
        """Spend the global face budget and atomically promote completed cubes."""

        completed: set[int] = set()
        budget = max(0, int(self.point_shadow_faces_per_frame))
        map_count = len(self._point_shadow_maps)
        for _ in range(budget):
            if not map_count:
                break
            map_index = next(
                (
                    (self._point_shadow_update_cursor + offset) % map_count
                    for offset in range(map_count)
                    if self._point_shadow_refreshes[
                        (self._point_shadow_update_cursor + offset) % map_count
                    ]
                    is not None
                ),
                None,
            )
            if map_index is None:
                break

            refresh = self._point_shadow_refreshes[map_index]
            face_index = refresh.next_face
            staging_map = self._point_shadow_staging_maps[map_index]
            with staging_map.render_face(face_index):
                BatchedMesh.draw_point_shadows_many(
                    casters_by_map[map_index],
                    caster_shader,
                    refresh.matrices[face_index],
                    refresh.position,
                    refresh.light_range,
                )
            refresh.next_face += 1
            self._count("shadow.point.faces")
            self._point_shadow_update_cursor = (map_index + 1) % map_count

            if refresh.next_face < POINT_SHADOW_FACE_COUNT:
                continue

            binding = staging_map.binding(
                refresh.light_id,
                refresh.position,
                refresh.light_range,
                bias=max(0.1, min(0.75, refresh.light_range * 0.003)),
            )
            self._point_shadow_maps[map_index], self._point_shadow_staging_maps[
                map_index
            ] = staging_map, self._point_shadow_maps[map_index]
            self._point_shadow_cache[map_index] = _PointShadowCache(
                light_id=refresh.light_id,
                light_revision=refresh.light_revision,
                caster_revision=refresh.caster_revision,
                binding=binding,
            )
            self._point_shadow_refreshes[map_index] = None
            completed.add(map_index)
            self._count("shadow.point.refreshes")
        return completed

    def render_point_shadows(self) -> None:
        """Render radial depth cubes for the most relevant local lights."""

        packet_shader = self.packet_shader
        if packet_shader is None:
            return
        active_lights = self._selected_point_lights()
        packet_shader.set_active_point_lights(
            light.light_id for light in active_lights
        )
        lights = tuple(
            light for light in active_lights if light.casts_shadows
        )[: self.max_point_shadows]
        if not lights:
            for map_index in range(len(self._point_shadow_refreshes)):
                self._point_shadow_refreshes[map_index] = None
            packet_shader.set_point_shadows(())
            return

        _, caster_shader = self._ensure_point_shadow_resources(
            len(lights)
        )
        assignments = self._assign_point_shadow_maps(lights)
        revisions = {}
        casters_by_map = {}
        for light, map_index in assignments:
            light_range = max(0.1, float(light.range))
            position = tuple(float(value) for value in light.position)
            # Nature shadow LOD must be stable for the light, not the moving
            # player camera.  Otherwise a six-face staged cube can select a new
            # mesh mid-build and restart forever during continuous traversal.
            viewpoint = _ShadowViewpoint(*position)
            shadow_records = tuple(
                self._shadow_mesh_records(viewpoint=viewpoint)
            )
            caster_records = tuple(
                (mesh, source_revision)
                for mesh, source_revision in shadow_records
                if getattr(mesh, "casts_point_shadows", True)
                and self._mesh_intersects_point_light(mesh, position, light_range)
            )
            casters_by_map[map_index] = tuple(
                mesh for mesh, _source_revision in caster_records
            )
            revisions[map_index] = (
                self._point_light_revision(light),
                tuple(
                    self._shadow_mesh_revision(mesh, source_revision)
                    for mesh, source_revision in caster_records
                ),
            )

        self._prepare_point_shadow_refreshes(
            assignments,
            revisions,
        )
        completed = self._render_point_shadow_refresh_faces(
            casters_by_map,
            caster_shader,
        )
        bindings = []
        for light, map_index in assignments:
            cache = self._point_shadow_cache[map_index]
            if cache is None or cache.light_id != str(light.light_id):
                continue
            if map_index not in completed:
                self._count("shadow.point.cache_hits")
            bindings.append(cache.binding)
        packet_shader.set_point_shadows(bindings)

    def dispose(self) -> None:
        if (
            self._sun_shadow_map is not None
            or self._sun_shadow_staging_map is not None
            or self._point_shadow_maps
        ):
            packet_shader = self.packet_shader
            if packet_shader is not None:
                packet_shader.set_directional_shadow(None)
                packet_shader.set_point_shadows(())
                packet_shader.set_active_point_lights(None)
        if self._shadow_caster_shader is not None:
            self._shadow_caster_shader.dispose()
            self._shadow_caster_shader = None
        if self._sun_shadow_map is not None:
            self._sun_shadow_map.dispose()
            self._sun_shadow_map = None
        if self._sun_shadow_staging_map is not None:
            self._sun_shadow_staging_map.dispose()
            self._sun_shadow_staging_map = None
        self._sun_shadow_cache = None
        self._sun_shadow_refresh = None
        self._sun_shadow_pending = None
        if self._point_shadow_caster_shader is not None:
            self._point_shadow_caster_shader.dispose()
            self._point_shadow_caster_shader = None
        for shadow_map in self._point_shadow_maps:
            shadow_map.dispose()
        for shadow_map in self._point_shadow_staging_maps:
            shadow_map.dispose()
        self._point_shadow_maps.clear()
        self._point_shadow_staging_maps.clear()
        self._point_shadow_cache.clear()
        self._point_shadow_refreshes.clear()
        self._point_shadow_update_cursor = 0

