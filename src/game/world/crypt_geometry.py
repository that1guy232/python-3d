"""Authored masonry, vaults and flooded rooms for the Under sea Crypt."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np
import pygame
from pygame.math import Vector3

from engine.core.mesh import BatchedMesh
from engine.textures.texture_utils import _upload_texture_surface
from game.world.lighting_receivers import (
    GROUND_LIGHTING_RECEIVER, TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
)
from game.world.objects.wall_tile import WallTile


@dataclass(frozen=True)
class CryptRoom:
    name: str
    bounds: tuple[float, float, float, float]

    def contains(self, x, z):
        x0, x1, z0, z1 = self.bounds
        return x0 <= x <= x1 and z0 <= z <= z1

    def get_bounding_box(self):
        return self.bounds


CRYPT_ROOMS = (
    CryptRoom("Drowned Hall", (-180, 180, -400, 200)),
    CryptRoom("Burial Gallery", (-460, -180, -260, -20)),
    CryptRoom("Tidal Passage", (-60, 60, -620, -400)),
    CryptRoom("Reliquary", (-180, 180, -940, -620)),
)

STONE = (0.66, 0.60, 0.47)
PALE = (0.84, 0.81, 0.70)
MOSS = (0.29, 0.38, 0.26)


class Masonry:
    """CPU geometry builder; uploads are grouped by room and material."""

    def __init__(self, scene):
        self.scene = scene
        self.rows = []
        self.rng = random.Random(712)

    def quad(self, a, b, c, d, color=STONE, uv_scale=64.0):
        a, b, c, d = (Vector3(p) for p in (a, b, c, d))
        normal = (b - a).cross(c - a).normalize()
        u, v = (b - a).length() / uv_scale, (d - a).length() / uv_scale
        points = (a, b, c, d)
        uvs = ((0, 0), (u, 0), (u, v), (0, v))
        for i in (0, 1, 2, 0, 2, 3):
            self.rows.append((*points[i], *color, *normal, *uvs[i]))

    def box(self, bounds, color=STONE, *, collision=False):
        x0, x1, y0, y1, z0, z1 = bounds
        self.quad((x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1), color)
        self.quad((x1,y0,z0), (x0,y0,z0), (x0,y1,z0), (x1,y1,z0), color)
        self.quad((x0,y0,z0), (x0,y0,z1), (x0,y1,z1), (x0,y1,z0), color)
        self.quad((x1,y0,z1), (x1,y0,z0), (x1,y1,z0), (x1,y1,z1), color)
        self.quad((x0,y1,z1), (x1,y1,z1), (x1,y1,z0), (x0,y1,z0), color)
        self.quad((x0,y0,z0), (x1,y0,z0), (x1,y0,z1), (x0,y0,z1), color)
        if collision:
            self.scene.wall_tiles.append(WallTile(
                position=Vector3((x0+x1)/2, (y0+y1)/2, (z0+z1)/2),
                width=(x1-x0)/2, thickness=x1-x0,
                height=(y1-y0)/2, depth=(z1-z0)/2,
            ))

    def wall(self, x0, x1, z0, z1, height=122):
        self.box((x0,x1,0,height,z0,z1), collision=True)
        # Uneven damp courses continue around the base of every wall.
        self.box((x0-.12,x1+.12,0,15,z0-.12,z1+.12), MOSS)
        self.box((x0-.25,x1+.25,23,26,z0-.25,z1+.25), (0.46,0.44,0.33))

    def cylinder(self, x, z, y0, y1, r0, r1, color=PALE):
        for i in range(12):
            a, b = i*math.tau/12, (i+1)*math.tau/12
            self.quad((x+r0*math.cos(a),y0,z+r0*math.sin(a)),
                      (x+r1*math.cos(a),y1,z+r1*math.sin(a)),
                      (x+r1*math.cos(b),y1,z+r1*math.sin(b)),
                      (x+r0*math.cos(b),y0,z+r0*math.sin(b)), color)

    def column(self, x, z):
        self.box((x-15,x+15,0,6,z-15,z+15), MOSS, collision=True)
        self.cylinder(x,z,6,12,14,11)
        self.cylinder(x,z,12,66,9,8)
        self.cylinder(x,z,66,72,8,14)
        self.box((x-15,x+15,72,78,z-15,z+15), PALE)
        self.scene.wall_tiles.append(WallTile(position=Vector3(x,39,z),
            width=10, thickness=20, height=39, depth=10))

    def vault(self, x, z0, z1, radius, spring=78, rise=44):
        # Inward-facing barrel ceiling, with its mortar grid following the curve.
        for i in range(24):
            a, b = i*math.pi/24, (i+1)*math.pi/24
            self.quad((x+radius*math.cos(a),spring+rise*math.sin(a),z0),
                      (x+radius*math.cos(a),spring+rise*math.sin(a),z1),
                      (x+radius*math.cos(b),spring+rise*math.sin(b),z1),
                      (x+radius*math.cos(b),spring+rise*math.sin(b),z0), PALE)

    def arch(self, x, z, radius=60, spring=78, rise=44, *, turn=False):
        start = len(self.rows)
        def point(r, h, a, depth):
            u, y = r*math.cos(a), spring+h*math.sin(a)
            return (x+depth,y,z+u) if turn else (x+u,y,z+depth)
        for i in range(15):
            a, b = (i+.035)*math.pi/15, (i+.965)*math.pi/15
            shade = self.rng.uniform(.84,1.02)
            color = tuple(c*shade for c in PALE)
            near = [point(r,h,t,-5) for r,h,t in
                    ((radius,rise,a),(radius+8,rise+8,a),
                     (radius+8,rise+8,b),(radius,rise,b))]
            far = [point(r,h,t,5) for r,h,t in
                   ((radius,rise,a),(radius+8,rise+8,a),
                    (radius+8,rise+8,b),(radius,rise,b))]
            self.quad(*near, color)
            self.quad(*reversed(far), color)
            for j in range(4):
                k = (j+1)%4
                self.quad(near[j],far[j],far[k],near[k],color)
        # Each voussoir is a single cut stone, with joints formed by the gaps.
        for index in range(start,len(self.rows)):
            self.rows[index] = (*self.rows[index][:9], .1, .05)

    def tomb(self, x, z):
        self.box((x-18,x+18,0,23,z-40,z+40), STONE, collision=True)
        self.box((x-21,x+21,23,29,z-43,z+43), PALE)
        self.box((x-2,x+2,29,30,z-22,z+22), (0.40,0.43,0.35))
        self.box((x-11,x+11,29,30,z-12,z-8), (0.40,0.43,0.35))

    def niches(self, x, z0, count):
        # Open cavities between shelves, with a recessed rear wall.
        self.wall(x+35,x+43,z0,z0+count*80)
        for i in range(count+1):
            z = z0+i*80
            self.wall(x,x+35,z-5,z+5)
        for i in range(count):
            z = z0+i*80
            for y in (0,39,78,116):
                self.box((x,x+35,y,y+6,z+5,z+75), STONE, collision=y==0)
            # Worn grave slabs and bone-colored fragments inside the recesses.
            for y in (7,46,85):
                self.box((x+8,x+28,y,y+4,z+14,z+65), (0.48,0.48,0.40))
                for n in range(3):
                    self.box((x+13+n*3,x+15+n*3,y+4,y+6,z+25,z+46), PALE)

    def upload(self, texture, *, floor=False):
        mesh = BatchedMesh.from_vertex_data(np.asarray(self.rows,np.float32),
            texture=texture, lighting_receiver=(GROUND_LIGHTING_RECEIVER if floor
                else TEXTURED_STATIC_WALL_LIGHTING_RECEIVER),
            casts_sun_shadows=False, shine_enabled=floor)
        self.rows.clear()
        return mesh


def create_masonry_texture():
    """Original repeating limestone courses; no reference photo pixels used."""
    rng = random.Random(9204)
    surface = pygame.Surface((128,128), pygame.SRCALPHA)
    surface.fill((112,111,102,255))
    for row in range(8):
        for col in range(-1,5):
            x, y = col*32+(row%2)*16, row*16
            value = rng.randint(192,229)
            pygame.draw.rect(surface,(value,value,value-5),(x+1,y+1,30,14))
            pygame.draw.line(surface,(239,237,228),(x+2,y+2),(x+29,y+2))
    pixels = pygame.surfarray.pixels3d(surface)
    noise = np.random.default_rng(9204).integers(-12,13,(128,128,1))
    pixels[:] = np.clip(pixels.astype(np.int16)+noise,0,255)
    del pixels
    return _upload_texture_surface(surface, nearest=False, repeat=True)


def build_crypt(scene, stone_texture, plain_texture):
    m = Masonry(scene)
    # Mosaic pavements remain visible through the shallow water.
    for room in CRYPT_ROOMS:
        x0,x1,z0,z1 = room.bounds
        for x in range(x0,x1,10):
            for z in range(z0,z1,10):
                border = min(x-x0,x1-x-10,z-z0,z1-z-10) < 20
                alternate = (x//10+z//10)%2
                base = ((.31,.44,.40) if alternate else (.66,.65,.51)) if border else (
                    (.46,.51,.43) if alternate else (.65,.63,.51))
                color = tuple(c*m.rng.uniform(.91,1.03) for c in base)
                m.quad((x,0,z+9.6),(x+9.6,0,z+9.6),
                       (x+9.6,0,z),(x,0,z), color)
    scene.ground_mesh = m.upload(plain_texture, floor=True)

    # Three parallel vaults on round columns, leaving an uninterrupted aisle.
    m.wall(-192,192,200,212)
    m.wall(-192,-180,-400,-250)
    m.wall(-192,-180,-130,200)
    m.niches(180,-400,7)
    m.wall(180,192,160,200)
    m.wall(-192,-60,-412,-400)
    m.wall(60,192,-412,-400)
    for x in (-120,0,120):
        m.vault(x,-400,200,60)
        for z in (-396,-320,-160,0,160,196):
            m.arch(x,z)
    for x in (-60,60):
        for z in (-320,-160,0,160):
            m.column(x,z)
    for z in (-260,-100,60):
        m.tomb(-125,z)
    m.arch(-184,-190,60,turn=True)
    scene.wall_tile_batches.append(m.upload(stone_texture))

    # Side gallery with raised sarcophagi and a sealed, barred sea opening.
    m.wall(-472,-460,-272,-8)
    m.wall(-460,-180,-272,-260)
    m.wall(-460,-180,-20,-8)
    m.wall(-192,-180,-260,-250)
    m.wall(-192,-180,-130,-20)
    m.vault(-320,-260,-20,140,spring=68,rise=54)
    for z in (-254,-140,-26):
        m.arch(-320,z,140,spring=68,rise=54)
    for x in (-400,-300):
        m.tomb(x,-210)
        m.tomb(x,-60)
    scene.wall_tile_batches.append(m.upload(stone_texture))

    # Narrow barrel-vault approach opens into the final chamber.
    m.wall(-72,-60,-620,-400)
    m.wall(60,72,-620,-400)
    m.vault(0,-620,-400,60)
    for z in (-610,-540,-470,-410):
        m.arch(0,z)
    scene.wall_tile_batches.append(m.upload(stone_texture))

    m.wall(-192,-180,-952,-620)
    m.wall(180,192,-952,-620)
    m.wall(-192,-60,-620,-608)
    m.wall(60,192,-620,-608)
    m.wall(-192,192,-952,-940)
    for x in (-120,0,120):
        m.vault(x,-940,-620,60)
        for z in (-936,-780,-626):
            m.arch(x,z)
    for x in (-60,60):
        m.column(x,-780)
        m.tomb(x*2,-850)
    # Low dais can be stepped onto; the reliquary itself is an interactive entity.
    m.box((-64,64,0,4,-930,-850), PALE)
    m.box((-46,46,4,8,-924,-862), PALE)
    scene.wall_tile_batches.append(m.upload(stone_texture))
    scene.invalidate_collision_index()
