"""Playable flooded crypt using the shared world controls, combat and lighting."""

from __future__ import annotations

import random
import time

import pygame
from pygame.math import Vector3
from OpenGL.GL import glDeleteTextures

from engine.rendering.lighting.model import SceneLighting
from engine.rendering.lighting.state import PointLight
from game.actors.skeleton import Skeleton, SkeletonRenderBatch
from game.inventory import InventoryItem, ItemType, receive_inventory_item
from game.gameplay import runtime as world_runtime
from game.levels.crypt.geometry import (
    build_crypt_steps, create_masonry_texture, create_mosaic_texture, crypt_fixtures,
)
from game.levels.crypt.layout import generate_crypt_layout
from game.levels.crypt.water import CryptWater
from game.objects.chest import Chest
from game.objects.architecture.torch import Torch
from game.gameplay.rendering.renderer import GameplayRenderer
from game.gameplay.scene import PlayableScene


class CryptReliquary(Chest):
    def interact(self, actor=None, scene=None):
        if scene is None:
            return False
        if scene.relic_recovered:
            return super().interact(actor=actor, scene=scene)
        item = InventoryItem("Pearl of the Deep", ItemType.MISC,
            "A luminous pearl recovered from the drowned crypt's reliquary.")
        if receive_inventory_item(scene,item) is None:
            scene.inventory_notice_text = "Backpack full. Make room for the pearl."
            scene.inventory_notice_expires_at = time.monotonic()+3.
            return False
        scene.relic_recovered = True
        self.open()
        return True


class CryptRenderer(GameplayRenderer):
    def draw_sky(self):
        pass

    def draw_world_objects(self, enable_timing=False):
        super().draw_world_objects(enable_timing=enable_timing)
        self.scene.water.draw(self.scene)

    def _controls_text(self):
        return "WASD move | Mouse look | E interact | I inventory | L lighting | Esc pause"

    def draw_hud(self, text, fps):
        super().draw_hud(text,fps)
        scene = self.scene
        if scene.paused or scene.inventory_open or scene.battle_mode:
            return
        room = next((r.name for r in scene.rooms if r.contains(
            scene.camera.position.x,scene.camera.position.z)), "Drowned Hall")
        objective = ("Pearl recovered - explore the crypt or leave through the pause menu."
            if scene.relic_recovered else "Find the Pearl of the Deep in the reliquary.")
        text.begin()
        text.draw_text(f"UNDER SEA CRYPT  /  {room}",240,54,
                       color=(203,226,214,255),align="topleft")
        text.draw_text(objective,240,82,color=(178,198,180,255),align="topleft")
        text.end()


class UnderseaCryptScene(PlayableScene):
    """A generated crypt. Construction stays CPU-only until loading begins."""

    def __init__(self, *, defer_setup=True, world_random_seed=None):
        seed = (random.SystemRandom().getrandbits(32) if world_random_seed is None
                else int(world_random_seed))
        self.layout = generate_crypt_layout(seed)
        self.level_seed = seed
        self.rooms = self.layout.rooms
        self.fixtures = crypt_fixtures(self.layout)
        super().__init__(world_random_seed=seed)
        self.ground_bounds = self.layout.bounds
        self.terrain_bounds = self.ground_bounds
        x0,x1,z0,z1 = self.ground_bounds
        self.world_center = Vector3((x0+x1)/2,0,(z0+z1)/2)
        x,z = self.layout.entrance
        self.camera.position = Vector3(x,25,z)
        self.camera.rotation = Vector3(0,0,0)
        self.camera.update_rotation(0)
        self._height_fn = lambda x,z: 0.
        self.ambient_birds_enabled = False
        self.elapsed = 0.
        self.relic_recovered = False
        self.reliquary = None
        self._owned_texture_ids = []
        self.water = CryptWater(self.rooms)
        self.renderer = CryptRenderer(self)
        if not defer_setup:
            for _ in self.initialize_steps():
                pass

    def initialize_steps(self):
        if self._initialized:
            yield "Ready",1.
            return
        yield "Preparing the crypt",0.
        self._setup_controllers()
        self._setup_graphics()
        self._load_assets()
        # Level atmosphere overrides the outdoor defaults, preserving input settings.
        self.lighting = SceneLighting(
            sun_position=Vector3(0,1000,0),sun_target=Vector3(0,0,0),
            sky_color=(.035,.075,.075,1.),ambient=.42,diffuse=0.,
            sun_tint=(.78,.89,.88),base_brightness=self.camera.brightness_default)
        self.fog_density = .00065
        self.clouds_enabled = False
        self.buildings = list(self.rooms)
        yield "Planning crypt chambers",.15
        stone = create_masonry_texture()
        self._owned_texture_ids.append(stone)
        mosaic = create_mosaic_texture(self.level_seed)
        self._owned_texture_ids.append(mosaic)
        for label, progress in build_crypt_steps(self,stone,mosaic):
            yield label,.15+.55*progress
        yield "Lighting the flooded chambers",.73
        self._place_lights()
        yield "Waking the crypt guardians",.80
        self._place_encounters()
        x,z = self.layout.reliquary
        self.reliquary = CryptReliquary(Vector3(x,8,z),texture=self.chest_tex,
            width=56,depth=34,interaction_distance=82,side="north")
        self.chests.append(self.reliquary)
        self.add_entity(self.reliquary)
        self.refresh_immediate_entities()
        self.rebuild_collision_index()
        world_runtime.initialize_player_spawn_height(self)
        self._initialized = True
        self._last_static_lighting_brightness = float(self.camera.brightness_default)
        self.lighting_controller.sync_uniforms(compile_shader=False)
        # Long corridors expose every torch's lit surfaces. Keep all their
        # occlusion maps resident, and finish the initial cubes behind loading.
        shadows = self.lighting_controller.shadows
        shadows.max_point_shadows = len(self.torches)
        shader = self.lighting_controller.shader
        if shadows.max_point_shadows > shader.point_shadow_capacity:
            raise RuntimeError("Under sea Crypt requires eight point-shadow texture slots")
        shadow_steps = 6 * len(self.torches)
        for step in range(shadow_steps):
            shadows.render(camera=self.camera, lighting=self.lighting, packet_shader=shader)
            yield "Preparing torch shadows", .82 + .17 * (step + 1) / shadow_steps
        yield "Under sea Crypt",1.

    def _place_lights(self):
        # Warm flames at the walls, with cool light entering below the sea.
        _walls,windows,mounts = self.fixtures
        frames = Torch.texture_or_load(self.torch_tex)
        for index,mount in enumerate(mounts):
            position = mount.position
            torch = Torch(position=Vector3(position),texture=frames[0],
                          camera=self.camera,frames=frames,size=(10.,22.),frame_duration=1/9)
            self.torches.append(torch)
            self.sprite_items.append(torch)
            self.lighting.add_point_light(PointLight(f"crypt:torch:{index}",
                position,(1.,.69,.32),2.8,520.,casts_shadows=True))
        for index,site in enumerate(windows):
            self.lighting.add_point_light(PointLight(f"crypt:sea:{index}",
                site.point(0,76,25),(.24,.72,.83),2.8,360.,casts_shadows=False))

    def _position_blocked(self, x, z, radius):
        if not self.contains_horizontal(Vector3(x,0,z)):
            return True
        for mesh in self.collision_meshes_at(x,z,radius=radius):
            if not getattr(mesh,"navigation_obstacle",True):
                continue
            x0,x1,z0,z1 = mesh.get_bounding_box()
            if x0-radius < x < x1+radius and z0-radius < z < z1+radius:
                return True
        return False

    def _place_encounters(self):
        for index,(x,z) in enumerate(self.layout.guardians):
            skeleton = Skeleton(Vector3(x,0,z),camera=self.camera,
                material_texture=self.building_material_tex,
                ground_height_at=self.ground_height_at,position_blocked=self._position_blocked,
                roam_radius=35.,chase_radius=135.,chase_give_up_radius=220.,
                rng=random.Random(self.level_seed+index))
            self.skeletons.append(skeleton)
            self.add_entity(skeleton)
        self.skeleton_batch = SkeletonRenderBatch(self.skeletons,
            material_texture=self.building_material_tex)

    def contains_horizontal(self, pos):
        return any(room.contains(pos.x,pos.z) for room in self.rooms)

    def ground_height_at(self, x, z):
        cx,cz = self.layout.reliquary
        if abs(x-cx) <= 46 and abs(z-cz) <= 32:
            return 8.
        if abs(x-cx) <= 64 and abs(z-cz) <= 40:
            return 4.
        return 0.

    def focused_interaction_prompt(self):
        if self.reliquary is not None and world_runtime._focused_interactable_entity(self) is self.reliquary:
            return "E to open reliquary" if not self.relic_recovered else "E to close / open reliquary"
        return super().focused_interaction_prompt()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_l and not (
            self.paused or self.inventory_open or self.battle_mode):
            self.lighting_controller.toggle()
            return
        super().handle_event(event)

    def update(self, dt):
        if not self.paused and not self.inventory_open and not self.battle_mode:
            self.elapsed += max(0.,float(dt))
        super().update(dt)

    def dispose(self):
        try:
            self.water.dispose()
            super().dispose()
        finally:
            if self._owned_texture_ids:
                glDeleteTextures(self._owned_texture_ids)
                self._owned_texture_ids.clear()
