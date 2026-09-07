"""Playable flooded crypt using the shared world controls, combat and lighting."""

from __future__ import annotations

import random
import time

import pygame
from pygame.math import Vector3
from OpenGL.GL import glDeleteTextures

from engine.rendering.lighting import SceneLighting
from engine.rendering.lighting_state import PointLight
from game.actors.skeleton import Skeleton, SkeletonRenderBatch
from game.inventory import InventoryItem, ItemType, receive_inventory_item
from game.world import world_runtime
from game.world.crypt_geometry import CRYPT_ROOMS, Masonry, build_crypt, create_masonry_texture
from game.world.crypt_water import CryptWater
from game.world.objects.chest import Chest
from game.world.objects.torch import Torch
from game.world.world_renderer import WorldRenderer
from game.world.worldscene import WorldScene


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


class CryptRenderer(WorldRenderer):
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
        room = next((r.name for r in CRYPT_ROOMS if r.contains(
            scene.camera.position.x,scene.camera.position.z)), "Drowned Hall")
        objective = ("Pearl recovered - explore the crypt or leave through the pause menu."
            if scene.relic_recovered else "Find the Pearl of the Deep in the reliquary.")
        text.begin()
        text.draw_text(f"UNDER SEA CRYPT  /  {room}",240,54,
                       color=(203,226,214,255),align="topleft")
        text.draw_text(objective,240,82,color=(178,198,180,255),align="topleft")
        text.end()


class UnderseaCryptScene(WorldScene):
    """A hand-authored dungeon. Construction stays CPU-only until loading begins."""

    def __init__(self, *, defer_setup=True):
        super().__init__(defer_setup=True, tree_count=0, grass_count=0,
                         rock_count=0, building_count=0, skeleton_count=6,
                         world_random_seed=712)
        self.ground_bounds = (-460,180,-940,200)
        self.terrain_bounds = self.ground_bounds
        self.world_center = Vector3(-140,0,-370)
        self.camera.position = Vector3(0,25,150)
        self.camera.rotation = Vector3(0,0,0)
        self.camera.update_rotation(0)
        self._height_fn = lambda x,z: 0.
        self.ambient_birds_enabled = False
        self.elapsed = 0.
        self.relic_recovered = False
        self.reliquary = None
        self._owned_texture_ids = []
        self.water = CryptWater(CRYPT_ROOMS)
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
        self.fog_density = .0015
        self.clouds_enabled = False
        self.buildings = list(CRYPT_ROOMS)
        yield "Laying stone and carving the vaults",.22
        stone = create_masonry_texture()
        self._owned_texture_ids.append(stone)
        build_crypt(self,stone,self.building_material_tex)
        yield "Lighting the flooded chambers",.55
        self._place_lights()
        yield "Waking the crypt guardians",.72
        self._place_encounters()
        self.reliquary = CryptReliquary(Vector3(0,8,-900),texture=self.chest_tex,
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
            raise RuntimeError("Under sea Crypt requires seven point-shadow texture slots")
        shadow_steps = 6 * len(self.torches)
        for step in range(shadow_steps):
            shadows.render(camera=self.camera, lighting=self.lighting, packet_shader=shader)
            yield "Preparing torch shadows", .82 + .17 * (step + 1) / shadow_steps
        yield "Under sea Crypt",1.

    def _place_lights(self):
        # Warm flames at the walls, with cool light entering below the sea.
        positions = ((-170,52,100),(165,52,-70),(-170,52,-330),
                     (-447,52,-140),(49,52,-500),(-160,52,-710),(160,52,-850))
        frames = Torch.texture_or_load(self.torch_tex)
        for index,position in enumerate(positions):
            torch = Torch(position=Vector3(position),texture=frames[0],
                          camera=self.camera,frames=frames,size=(10.,22.),frame_duration=1/9)
            self.torches.append(torch)
            self.sprite_items.append(torch)
            self.lighting.add_point_light(PointLight(f"crypt:torch:{index}",
                position,(1.,.69,.32),2.5,210.,casts_shadows=True))
        m = Masonry(self)
        for index,(x,y,z) in enumerate(((0,76,198),(-320,76,-22),(0,76,-938))):
            # Inset blue glass and iron bars suggest sealed openings to the sea.
            m.box((x-26,x+26,y-24,y+24,z-1,z+1),(.20,.31,.29))
            m.box((x-20,x+20,y-18,y+18,z-1.5,z+1.5),(.27,.80,.81))
            for dx in (-13,0,13):
                m.box((x+dx-1,x+dx+1,y-20,y+20,z-2,z+2),(.13,.19,.18))
            m.box((x-22,x+22,y-1,y+1,z-2,z+2),(.13,.19,.18))
            light_z = z+22 if z < -900 else z-22
            self.lighting.add_point_light(PointLight(f"crypt:sea:{index}",
                (x,y,light_z),(.24,.72,.83),3.0,280.,casts_shadows=False))
        self.wall_tile_batches.append(m.upload(self.building_material_tex))

    def _position_blocked(self, x, z, radius):
        if not self.contains_horizontal(Vector3(x,0,z)):
            return True
        for mesh in self.collision_meshes_at(x,z,radius=radius):
            x0,x1,z0,z1 = mesh.get_bounding_box()
            if x0-radius < x < x1+radius and z0-radius < z < z1+radius:
                return True
        return False

    def _place_encounters(self):
        for index,(x,z) in enumerate(((115,-220),(-330,-140),(0,-530),
                                    (-105,-720),(105,-790),(0,-840))):
            skeleton = Skeleton(Vector3(x,0,z),camera=self.camera,
                material_texture=self.building_material_tex,
                ground_height_at=self.ground_height_at,position_blocked=self._position_blocked,
                roam_radius=35.,chase_radius=135.,chase_give_up_radius=220.,
                rng=random.Random(712+index))
            self.skeletons.append(skeleton)
            self.add_entity(skeleton)
        self.skeleton_batch = SkeletonRenderBatch(self.skeletons,
            material_texture=self.building_material_tex)

    def contains_horizontal(self, pos):
        return any(room.contains(pos.x,pos.z) for room in CRYPT_ROOMS)

    def ground_height_at(self, x, z):
        if -46 <= x <= 46 and -924 <= z <= -862:
            return 8.
        if -64 <= x <= 64 and -930 <= z <= -850:
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
