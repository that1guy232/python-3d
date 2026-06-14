"""Static lighting rebuild and shader sync policy for WorldScene."""

from __future__ import annotations

from engine.core.compat_shader import set_texture_lighting_state
from game.world import world_builder
from game.world.objects.wall_tile import build_wall_tile_batches


class StaticLightingController:
    """Own lighting rebuilds that still operate on scene-owned resources."""

    def __init__(self, scene) -> None:
        self.scene = scene

    def sync_aliases(self):
        """Keep older scene attributes pointing at the shared lighting model."""
        scene = self.scene
        lighting = getattr(scene, "lighting", None)
        if lighting is None:
            return None
        scene.sun_pos = lighting.sun_position
        scene.sun_direction = lighting.sun_direction
        scene.brightness_modifiers = lighting.brightness_modifiers
        scene.covered_regions = lighting.covered_regions
        return lighting

    def invalidate_texture_lighting_cache(self) -> None:
        scene = self.scene
        scene._texture_lighting_sync_key = None

    def set_brightness(self, value: float) -> float:
        """Set global brightness and refresh all baked lighting consumers."""
        scene = self.scene
        camera = scene.camera
        brightness = float(value)
        setter = getattr(camera, "set_brightness_default", None)
        if callable(setter):
            brightness = float(setter(brightness))
        else:
            camera.brightness_default = brightness
            cache = getattr(camera, "_brightness_cache", None)
            if cache is not None:
                cache.clear()

        lighting = getattr(scene, "lighting", None)
        if lighting is not None:
            lighting.set_base_brightness(brightness)

        if (
            getattr(scene, "_initialized", False)
            and getattr(scene, "_last_static_lighting_brightness", None) == brightness
        ):
            return brightness

        if getattr(scene, "_initialized", False):
            self.sync_brightness_modifiers_from_camera()
            if scene.brightness_modifiers and self.sync_uniforms():
                self.apply_untextured_static_exposure_cpu(brightness)
                scene._last_static_lighting_brightness = brightness
            elif scene.brightness_modifiers:
                self.refresh_static()
            else:
                self.apply_static_exposure(brightness)
                scene._last_static_lighting_brightness = brightness
        return brightness

    def sync_brightness_modifiers_from_camera(self) -> None:
        scene = self.scene
        camera = getattr(scene, "camera", None)
        areas = getattr(camera, "brightness_areas", None)
        if areas is None:
            return

        modifiers = []
        for area in areas:
            try:
                modifiers.append(
                    {
                        "center": area["center"],
                        "radius": float(area["radius"]),
                        "value": float(area["value"]),
                        "falloff": float(area.get("falloff", 1.0)),
                        "bounds": area.get("bounds"),
                        "indoor_only": bool(area.get("indoor_only", False)),
                        "floor_scale": float(area.get("floor_scale", 1.0)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        lighting = getattr(scene, "lighting", None)
        if lighting is not None:
            lighting.set_brightness_modifiers(modifiers)
            self.sync_aliases()
        else:
            scene.brightness_modifiers = modifiers
        self.invalidate_texture_lighting_cache()

    def refresh_static(self) -> None:
        """Rebuild static VBOs whose vertex colors contain brightness."""
        scene = self.scene
        if not getattr(scene, "_initialized", False):
            return

        self.sync_brightness_modifiers_from_camera()
        camera = scene.camera
        brightness = float(getattr(camera, "brightness_default", 1.0))
        lighting = getattr(scene, "lighting", None)
        if lighting is not None:
            lighting.set_base_brightness(brightness)
            lighting.set_covered_regions(getattr(scene, "covered_regions", ()))
            self.sync_aliases()
        sun_direction = getattr(
            lighting,
            "sun_direction",
            getattr(scene, "sun_direction", None),
        )
        scene.sun_direction = sun_direction

        builder = getattr(scene, "builder", None)
        if builder is not None:
            builder.brightness_modifiers = scene.brightness_modifiers
            builder.default_brightness = brightness
            builder.lighting = lighting
            builder.sun_direction = sun_direction
            builder.covered_regions = getattr(scene, "covered_regions", ())
            self.dispose_renderable(getattr(scene, "ground_mesh", None))
            scene.ground_mesh = builder.build()
            scene._ground_height_sampler = getattr(
                scene.ground_mesh,
                "height_sampler",
                None,
            )

        height_sampler = getattr(scene, "_ground_height_sampler", None)
        refreshed_roads: set[int] = set()
        for road in self.road_lighting_candidates():
            if road is None:
                continue
            if id(road) in refreshed_roads:
                continue
            refreshed_roads.add(id(road))
            refresh = getattr(road, "refresh_lighting", None)
            if callable(refresh):
                refresh(
                    brightness_modifiers=scene.brightness_modifiers,
                    default_brightness=brightness,
                    lighting=lighting,
                    sun_direction=sun_direction,
                    height_sampler=height_sampler,
                )
        world_builder._build_road_batches(scene)

        for wall in getattr(scene, "walls", ()) or ():
            wall.sun_direction = sun_direction
            wall.lighting = lighting

        self.dispose_renderable_batches(getattr(scene, "wall_tile_batches", ()))
        scene.wall_tile_batches = build_wall_tile_batches(
            getattr(scene, "walls", []),
            camera=camera,
            default_brightness=brightness,
            sun_direction=sun_direction,
            lighting=lighting,
        )

        if getattr(scene, "ground_mesh", None) is not None:
            world_builder._build_fences(scene)

        self.sync_uniforms(compile_shader=False)
        scene._last_static_lighting_brightness = brightness

    def apply_static_exposure(self, brightness: float) -> None:
        """Apply global exposure without rebuilding static meshes."""
        exposure = float(brightness)
        if self.sync_uniforms(base_brightness=exposure):
            self.apply_untextured_static_exposure_cpu(exposure)
            return

        self.apply_static_exposure_cpu(exposure)

    def sync_uniforms(
        self,
        *,
        base_brightness: float | None = None,
        compile_shader: bool = True,
    ) -> bool:
        scene = self.scene
        camera = getattr(scene, "camera", None)
        lighting = getattr(scene, "lighting", None)
        brightness = (
            float(base_brightness)
            if base_brightness is not None
            else float(getattr(camera, "brightness_default", 1.0))
        )
        brightness_areas = getattr(
            lighting,
            "brightness_modifiers",
            getattr(camera, "brightness_areas", ()),
        )
        covered_regions = getattr(
            lighting,
            "covered_regions",
            getattr(scene, "covered_regions", ()),
        )
        sun_direction = getattr(
            lighting,
            "sun_direction",
            getattr(scene, "sun_direction", None),
        )
        sync_key = self.texture_lighting_fast_key(
            brightness=brightness,
            lighting=lighting,
            sun_direction=sun_direction,
            brightness_areas=brightness_areas,
            covered_regions=covered_regions,
            compile_shader=compile_shader,
        )
        if sync_key == scene._texture_lighting_sync_key:
            return scene._texture_lighting_sync_result

        if lighting is not None:
            lighting.set_base_brightness(brightness)
            self.sync_aliases()
        result = set_texture_lighting_state(
            base_brightness=brightness,
            lighting=lighting,
            sun_direction=sun_direction,
            brightness_areas=brightness_areas,
            covered_regions=covered_regions,
            exposure_scale=1.0,
            compile_shader=compile_shader,
        )
        scene._texture_lighting_sync_key = sync_key
        scene._texture_lighting_sync_result = result
        return result

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
            self.collection_identity_key(brightness_areas),
            self.collection_identity_key(covered_regions),
            tuple(
                self.rounded(getattr(door, "open_amount", 0.0), digits=4)
                for door in getattr(self.scene, "doors", ()) or ()
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
                    for value in (region.get("doorway"), *(region.get("windows") or ()))
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

    @staticmethod
    def uses_texture_shader(obj) -> bool:
        return getattr(obj, "texture", None) is not None

    @staticmethod
    def set_exposure_cpu(obj, exposure: float) -> None:
        setter = getattr(obj, "set_exposure", None)
        if callable(setter):
            setter(exposure)

    def apply_untextured_static_exposure_cpu(self, exposure: float) -> None:
        scene = self.scene
        mesh = getattr(scene, "ground_mesh", None)
        if mesh is not None and not self.uses_texture_shader(mesh):
            self.set_exposure_cpu(mesh, exposure)

        for mesh in getattr(scene, "fence_meshes", ()) or ():
            if not self.uses_texture_shader(mesh):
                self.set_exposure_cpu(mesh, exposure)

        for batch in getattr(scene, "road_batches", ()) or ():
            if not self.uses_texture_shader(batch):
                self.set_exposure_cpu(batch, exposure)

        for mesh in getattr(scene, "wall_tile_batches", ()) or ():
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
        scene = self.scene
        mesh = getattr(scene, "ground_mesh", None)
        if mesh is not None:
            self.set_exposure_cpu(mesh, exposure)

        for mesh in getattr(scene, "fence_meshes", ()) or ():
            self.set_exposure_cpu(mesh, exposure)

        for batch in getattr(scene, "road_batches", ()) or ():
            self.set_exposure_cpu(batch, exposure)

        for mesh in getattr(scene, "wall_tile_batches", ()) or ():
            self.set_exposure_cpu(mesh, exposure)

        refreshed_roads: set[int] = set()
        for road in self.road_lighting_candidates():
            if road is None or id(road) in refreshed_roads:
                continue
            refreshed_roads.add(id(road))
            self.set_exposure_cpu(road, exposure)

    def road_lighting_candidates(self):
        scene = self.scene
        return [
            getattr(scene, "road", None),
            *(getattr(scene, "roads", ()) or ()),
            *(
                obj
                for obj in (getattr(scene, "others", ()) or ())
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
