"""Rollback-only scene projection and compatibility-shader synchronization."""

from __future__ import annotations

from game.world import world_builder
from game.world.objects.wall_tile import build_wall_tile_batches


def set_texture_lighting_state(**kwargs):
    """Import the deprecated compatibility shader only for a rollback upload."""

    from engine.core.compat_shader import set_texture_lighting_state as legacy_set

    return legacy_set(**kwargs)


class LegacyLightingBridge:
    """Own every mutable dictionary and shader upload required by rollback mode."""

    def __init__(self, scene, resources, build_state, diagnostics, alias_names) -> None:
        self.scene = scene
        self.resources = resources
        self.build_state = build_state
        self.diagnostics = diagnostics
        self.alias_names = tuple(alias_names)
        self.covered_regions = list(getattr(scene, "covered_regions", ()) or ())

    def legacy_brightness_modifiers(self) -> list[dict[str, object]]:
        lighting = getattr(self.scene, "lighting", None)
        return [
            light.to_legacy_dict()
            for light in getattr(lighting, "local_lights", ()) or ()
        ]

    def sync_aliases(self):
        """Materialize the four scene fields consumed by rollback builders."""

        lighting = getattr(self.scene, "lighting", None)
        if lighting is None:
            return None
        self.scene.sun_pos = lighting.sun_position
        self.scene.sun_direction = lighting.sun_direction
        self.scene.brightness_modifiers = self.legacy_brightness_modifiers()
        self.scene.covered_regions = self.covered_regions
        self.diagnostics.legacy_alias_projections += 1
        return lighting

    def clear_aliases(self) -> None:
        for name in self.alias_names:
            if hasattr(self.scene, name):
                try:
                    delattr(self.scene, name)
                except AttributeError:
                    pass

    def set_covered_regions(self, regions) -> list[object]:
        self.covered_regions = (
            regions if isinstance(regions, list) else list(regions or ())
        )
        return self.covered_regions

    def prepare_runtime(self) -> None:
        """Project typed volumes and bind rollback dictionaries to exterior doors."""

        volumes = tuple(getattr(self.scene, "environment_volumes", ()) or ())
        regions = [volume.to_legacy_dict() for volume in volumes]
        self.set_covered_regions(regions)
        self.sync_aliases()

        regions_by_portal_id = {}
        for volume, region in zip(volumes, regions):
            portal = getattr(volume, "doorway", None)
            portal_id = getattr(portal, "portal_id", None)
            if portal_id is not None:
                regions_by_portal_id[str(portal_id)] = region

        for door in getattr(self.build_state, "doors", ()) or ():
            portal = getattr(door, "_environment_portal", None)
            portal_id = getattr(portal, "portal_id", None)
            region = regions_by_portal_id.get(str(portal_id))
            bind = getattr(door, "bind_doorway_light", None)
            if region is None or not callable(bind):
                continue
            bind(
                region,
                brightness_modifier=getattr(
                    door,
                    "_doorway_brightness_modifier",
                    None,
                ),
                portal=portal,
                synchronize=False,
            )
        self.invalidate_cache()

    def deactivate(self) -> None:
        """Discard rollback projections when returning to packet mode."""

        self.covered_regions.clear()
        for door in getattr(self.build_state, "doors", ()) or ():
            portal = getattr(door, "_environment_portal", None)
            bind = getattr(door, "bind_doorway_light", None)
            if portal is None or not callable(bind):
                continue
            bind(
                None,
                brightness_modifier=getattr(
                    door,
                    "_doorway_brightness_modifier",
                    None,
                ),
                portal=portal,
                synchronize=False,
            )
        self.clear_aliases()
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        self.scene._texture_lighting_sync_key = None

    def sync_shader(
        self,
        *,
        brightness: float,
        lighting,
        snapshot,
        camera,
        compile_shader: bool,
    ) -> bool:
        """Upload one authoritative snapshot to the deprecated shader globals."""

        adapter_lighting = snapshot or lighting
        brightness_areas = (
            snapshot.local_lights
            if snapshot is not None
            else getattr(
                lighting,
                "local_lights",
                getattr(camera, "brightness_areas", ()),
            )
        )
        sun_direction = getattr(
            adapter_lighting,
            "sun_direction",
            getattr(self.scene, "sun_direction", None),
        )
        sync_key = self.texture_lighting_fast_key(
            brightness=brightness,
            lighting=adapter_lighting,
            sun_direction=sun_direction,
            brightness_areas=brightness_areas,
            covered_regions=self.covered_regions,
            compile_shader=compile_shader,
        )
        if sync_key == self.scene._texture_lighting_sync_key:
            self.diagnostics.uniform_sync_cache_hits += 1
            return self.scene._texture_lighting_sync_result

        self.diagnostics.shader_state_updates += 1
        result = set_texture_lighting_state(
            base_brightness=brightness,
            lighting=adapter_lighting,
            sun_direction=sun_direction,
            brightness_areas=brightness_areas,
            covered_regions=self.covered_regions,
            exposure_scale=1.0,
            compile_shader=compile_shader,
        )
        self.scene._texture_lighting_sync_key = sync_key
        self.scene._texture_lighting_sync_result = result
        if result:
            self.diagnostics.shader_uniform_uploads += 1
        return result

    def require_legacy_backend(self) -> None:
        if getattr(self.scene, "lighting_backend", "legacy") != "legacy":
            raise RuntimeError("rollback lighting policy requires legacy backend")

    def rebuild_static(self, *, lighting, camera, brightness: float) -> None:
        """Rebuild every static representation that bakes rollback lighting."""

        self.require_legacy_backend()
        if lighting is not None:
            self.set_covered_regions(getattr(self.scene, "covered_regions", ()))
            self.sync_aliases()
        authoritative_sun_direction = getattr(
            lighting,
            "sun_direction",
            getattr(self.scene, "sun_direction", None),
        )
        self.scene.sun_direction = authoritative_sun_direction
        receiver_sun_direction = (
            None if lighting is not None else authoritative_sun_direction
        )

        builder = getattr(self.build_state, "builder", None)
        if builder is not None:
            builder.brightness_modifiers = self.scene.brightness_modifiers
            builder.default_brightness = brightness
            builder.lighting = lighting
            builder.sun_direction = receiver_sun_direction
            builder.covered_regions = getattr(self.scene, "covered_regions", ())
            environment_volumes = getattr(self.scene, "environment_volumes", None)
            builder.environment_volumes = (
                None
                if environment_volumes is None
                else list(environment_volumes or ())
            )
            builder.dynamic_lighting = False
            self.dispose_renderable(getattr(self.resources, "ground_mesh", None))
            self.resources.ground_mesh = builder.build()
            self.diagnostics.ground_rebuilds += 1
            self.resources.ground_height_sampler = getattr(
                self.resources.ground_mesh,
                "height_sampler",
                None,
            )

        height_sampler = getattr(self.resources, "ground_height_sampler", None)
        refreshed_roads: set[int] = set()
        for road in self.road_lighting_candidates():
            if road is None or id(road) in refreshed_roads:
                continue
            refreshed_roads.add(id(road))
            refresh = getattr(road, "refresh_lighting", None)
            if callable(refresh):
                refresh(
                    brightness_modifiers=self.scene.brightness_modifiers,
                    default_brightness=brightness,
                    lighting=lighting,
                    sun_direction=receiver_sun_direction,
                    height_sampler=height_sampler,
                )
                self.diagnostics.road_refreshes += 1
        world_builder._build_road_batches(self.scene)

        for wall in getattr(self.build_state, "walls", ()) or ():
            wall.sun_direction = receiver_sun_direction
            wall.lighting = lighting

        self.dispose_renderable_batches(
            getattr(self.resources, "wall_tile_batches", ()) or ()
        )
        self.resources.wall_tile_batches = build_wall_tile_batches(
            getattr(self.build_state, "walls", []),
            camera=camera,
            default_brightness=brightness,
            sun_direction=receiver_sun_direction,
            lighting=lighting,
            dynamic_lighting=False,
        )
        self.diagnostics.wall_batch_rebuilds += 1

        if getattr(self.resources, "ground_mesh", None) is not None:
            world_builder._build_fences(self.scene)
            self.diagnostics.fence_rebuilds += 1

    @staticmethod
    def uses_texture_shader(obj) -> bool:
        return getattr(obj, "texture", None) is not None

    @staticmethod
    def set_exposure_cpu(obj, exposure: float) -> None:
        setter = getattr(obj, "set_exposure", None)
        if callable(setter):
            setter(exposure)

    def apply_untextured_static_exposure_cpu(self, exposure: float) -> None:
        self.require_legacy_backend()
        mesh = getattr(self.resources, "ground_mesh", None)
        if mesh is not None and not self.uses_texture_shader(mesh):
            self.set_exposure_cpu(mesh, exposure)

        for mesh in getattr(self.resources, "fence_meshes", ()) or ():
            if not self.uses_texture_shader(mesh):
                self.set_exposure_cpu(mesh, exposure)

        for batch in getattr(self.resources, "road_batches", ()) or ():
            if not self.uses_texture_shader(batch):
                self.set_exposure_cpu(batch, exposure)

        for mesh in getattr(self.resources, "wall_tile_batches", ()) or ():
            if not self.uses_texture_shader(mesh):
                self.set_exposure_cpu(mesh, exposure)

        refreshed_roads: set[int] = set()
        for road in self.road_lighting_candidates():
            if road is None or id(road) in refreshed_roads:
                continue
            refreshed_roads.add(id(road))
            if not self.uses_texture_shader(road):
                self.set_exposure_cpu(road, exposure)

    def apply_static_exposure_cpu(self, exposure: float) -> None:
        self.require_legacy_backend()
        mesh = getattr(self.resources, "ground_mesh", None)
        if mesh is not None:
            self.set_exposure_cpu(mesh, exposure)

        for mesh in getattr(self.resources, "fence_meshes", ()) or ():
            self.set_exposure_cpu(mesh, exposure)
        for batch in getattr(self.resources, "road_batches", ()) or ():
            self.set_exposure_cpu(batch, exposure)
        for mesh in getattr(self.resources, "wall_tile_batches", ()) or ():
            self.set_exposure_cpu(mesh, exposure)

        refreshed_roads: set[int] = set()
        for road in self.road_lighting_candidates():
            if road is None or id(road) in refreshed_roads:
                continue
            refreshed_roads.add(id(road))
            self.set_exposure_cpu(road, exposure)

    def road_lighting_candidates(self):
        return [
            getattr(self.resources, "road", None),
            *(getattr(self.build_state, "roads", ()) or ()),
            *(
                obj
                for obj in (getattr(self.resources, "others", ()) or ())
                if hasattr(obj, "refresh_lighting") or hasattr(obj, "set_exposure")
            ),
        ]

    @staticmethod
    def dispose_renderable(obj) -> None:
        dispose = getattr(obj, "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                pass

    @classmethod
    def dispose_renderable_batches(cls, values) -> None:
        for value in values or ():
            cls.dispose_renderable(value)

    def texture_lighting_fast_key(
        self,
        *,
        brightness: float,
        lighting,
        sun_direction,
        brightness_areas,
        covered_regions,
        compile_shader: bool,
    ):
        return (
            bool(compile_shader),
            self.rounded(brightness),
            self.vector_key(sun_direction),
            self.rounded(getattr(lighting, "ambient", 0.72)),
            self.rounded(getattr(lighting, "diffuse", 0.48)),
            self.rounded(getattr(lighting, "max_factor", 1.15)),
            getattr(lighting, "revision", None),
            (
                None
                if getattr(lighting, "revision", None) is not None
                else self.collection_identity_key(brightness_areas)
            ),
            self.collection_identity_key(covered_regions),
            tuple(
                self.rounded(getattr(door, "open_amount", 0.0), digits=4)
                for door in getattr(self.build_state, "doors", ()) or ()
                if getattr(door, "_doorway_light_region", None) is not None
                or getattr(door, "_doorway_brightness_modifier", None) is not None
            ),
        )

    @staticmethod
    def collection_identity_key(values):
        try:
            return (id(values), len(values))
        except Exception:
            return (id(values), None)

    @classmethod
    def texture_lighting_key(
        cls,
        *,
        brightness: float,
        lighting,
        sun_direction,
        brightness_areas,
        covered_regions,
        compile_shader: bool,
    ):
        return (
            bool(compile_shader),
            cls.rounded(brightness),
            cls.vector_key(sun_direction),
            cls.rounded(getattr(lighting, "ambient", 0.72)),
            cls.rounded(getattr(lighting, "diffuse", 0.48)),
            cls.rounded(getattr(lighting, "max_factor", 1.15)),
            cls.brightness_areas_key(brightness_areas),
            cls.covered_regions_key(covered_regions),
        )

    @staticmethod
    def rounded(value, digits: int = 5):
        try:
            return round(float(value), digits)
        except Exception:
            return None

    @classmethod
    def vector_key(cls, value):
        try:
            return (
                cls.rounded(value.x),
                cls.rounded(value.y),
                cls.rounded(value.z),
            )
        except Exception:
            try:
                return (
                    cls.rounded(value[0]),
                    cls.rounded(value[1]),
                    cls.rounded(value[2]),
                )
            except Exception:
                return None

    @classmethod
    def brightness_areas_key(cls, areas):
        values = []
        for area in areas or ():
            try:
                if isinstance(area, dict):
                    center = area.get("center")
                    bounds = area.get("bounds")
                    values.append(
                        (
                            cls.vector_key(center),
                            cls.rounded(area.get("radius")),
                            cls.rounded(area.get("value")),
                            cls.rounded(area.get("falloff", 1.0)),
                            cls.bounds_key(bounds),
                            bool(area.get("indoor_only", False)),
                            cls.rounded(area.get("floor_scale", 1.0)),
                        )
                    )
                elif all(
                    hasattr(area, name)
                    for name in ("center", "radius", "value")
                ):
                    values.append(
                        (
                            cls.vector_key(area.center),
                            cls.rounded(area.radius),
                            cls.rounded(area.value),
                            cls.rounded(getattr(area, "falloff", 1.0)),
                            cls.bounds_key(getattr(area, "bounds", None)),
                            bool(getattr(area, "indoor_only", False)),
                            cls.rounded(getattr(area, "floor_scale", 1.0)),
                        )
                    )
                else:
                    center, radius, value, falloff = area[:4]
                    values.append(
                        (
                            cls.vector_key(center),
                            cls.rounded(radius),
                            cls.rounded(value),
                            cls.rounded(falloff),
                            cls.bounds_key(area[4] if len(area) > 4 else None),
                            False,
                            1.0,
                        )
                    )
            except Exception:
                continue
        return tuple(values)

    @classmethod
    def covered_regions_key(cls, regions):
        values = []
        for region in regions or ():
            if not isinstance(region, dict):
                try:
                    values.append(tuple(cls.rounded(part) for part in region[:5]))
                except Exception:
                    continue
                continue

            openings = region.get("openings")
            if not isinstance(openings, (list, tuple)):
                openings = [
                    value
                    for value in (
                        region.get("doorway"),
                        *(region.get("windows") or ()),
                    )
                    if isinstance(value, dict)
                ]

            values.append(
                (
                    cls.rounded(region.get("min_x")),
                    cls.rounded(region.get("max_x")),
                    cls.rounded(region.get("min_z")),
                    cls.rounded(region.get("max_z")),
                    cls.rounded(region.get("factor", 1.0)),
                    tuple(cls.opening_key(opening) for opening in openings),
                )
            )
        return tuple(values)

    @classmethod
    def opening_key(cls, opening):
        if not isinstance(opening, dict):
            return None
        return (
            str(opening.get("side", "")),
            cls.rounded(opening.get("center_x")),
            cls.rounded(opening.get("center_z")),
            cls.rounded(opening.get("width")),
            cls.rounded(opening.get("depth")),
            cls.rounded(opening.get("side_fade")),
            cls.rounded(opening.get("edge_factor", 1.0)),
        )

    @classmethod
    def bounds_key(cls, bounds):
        if bounds is None:
            return None
        try:
            return tuple(cls.rounded(part) for part in bounds)
        except Exception:
            return None
