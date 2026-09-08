"""Shared structural bays and non-overlapping surfaces for generated crypts."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np
import pygame
from pygame.math import Vector3
from OpenGL import GL as gl

from engine.rendering.mesh import BatchedMesh
from engine.textures.texture_utils import _upload_texture_surface
from game.world.rendering.receivers import (
    GROUND_LIGHTING_RECEIVER,
    TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
)
from engine.geometry.wall import WallTile

STONE = (.66, .60, .47)
PALE = (.84, .81, .70)
MOSS = (.29, .38, .26)
RIB_HALF_WIDTH = 12.
SPRING_HEIGHT = 82.
VAULT_RISE = 40.
ROOF_TOP = 142.
PASSAGE_SPRING = 46.
PASSAGE_RISE = 24.


@dataclass(frozen=True)
class WallSite:
    cell: tuple[int, int]
    side: str
    center: tuple[float, float]
    normal: tuple[int, int]
    span: float

    def point(self, u, y, inset=4.):
        x, z = self.center
        nx, nz = self.normal
        return (x+nz*u+nx*inset, y, z-nx*u+nz*inset)


@dataclass(frozen=True)
class TorchMount:
    site: WallSite

    @property
    def position(self):
        return self.site.point(0, 43., 14.)


def passage_walls(room):
    """Recessed burial shelves along the two sides of a narrow passage."""
    x0,x1,z0,z1 = room.bounds
    length = z1-z0 if room.along_z else x1-x0
    count = max(1, int(length/144))
    for k in range(count):
        start = (z0 if room.along_z else x0)+k*length/count-(4 if k == 0 else 0)
        end = (z0 if room.along_z else x0)+(k+1)*length/count+(4 if k == count-1 else 0)
        span,center = end-start,(start+end)/2
        if room.along_z:
            yield WallSite((-999,k),"west",(x0-4,center),(1,0),span)
            yield WallSite((-999,k),"east",(x1+4,center),(-1,0),span)
        else:
            yield WallSite((-999,k),"south",(center,z0-4),(0,1),span)
            yield WallSite((-999,k),"north",(center,z1+4),(0,-1),span)


def portal_width(site, layout):
    """A hall boundary is opened only where a passage terminates."""
    x,z = site.center
    for room in layout.passages:
        x0,x1,z0,z1 = room.bounds
        if room.along_z and site.normal[1] and abs(x-(x0+x1)/2) < .01:
            if min(abs(z-z0),abs(z-z1)) < .01:
                return x1-x0
        if not room.along_z and site.normal[0] and abs(z-(z0+z1)/2) < .01:
            if min(abs(x-x0),abs(x-x1)) < .01:
                return z1-z0
    return 0.


def arch_profile(width):
    return tuple((width/2*math.sin(a),PASSAGE_SPRING+PASSAGE_RISE*math.cos(a))
                 for a in np.linspace(-math.pi/2,math.pi/2,21))


def boundary_walls(layout):
    walls = []
    for i, j in sorted(layout.cells):
        x0, x1, z0, z1 = layout.cell_bounds((i, j))
        for side, neighbor, center, normal, span in (
            ("north", (i,j-1), ((x0+x1)/2,z1), (0,-1), x1-x0),
            ("south", (i,j+1), ((x0+x1)/2,z0), (0,1), x1-x0),
            ("west", (i-1,j), (x0,(z0+z1)/2), (1,0), z1-z0),
            ("east", (i+1,j), (x1,(z0+z1)/2), (-1,0), z1-z0),
        ):
            if neighbor not in layout.cells:
                walls.append(WallSite((i,j),side,center,normal,span))
    return tuple(walls)


def crypt_fixtures(layout):
    """Eight stable torch lights plus sea windows, independent of the camera."""
    walls = boundary_walls(layout)
    windows = []
    for room in layout.rooms:
        if room.kind == "passage":
            continue
        candidates = [w for w in walls if w.cell in room.cells and not portal_width(w,layout)]
        cx, cz = (room.bounds[0]+room.bounds[1])/2, (room.bounds[2]+room.bounds[3])/2
        windows.append(min(candidates,key=lambda w: math.dist(w.center,(cx,cz))))
    torches = []
    for index,room in enumerate(layout.passages):
        cx, cz = (room.bounds[0]+room.bounds[1])/2, (room.bounds[2]+room.bounds[3])/2
        candidates = list(passage_walls(room))[index%2::2]
        wall = min(candidates,key=lambda w: math.dist(w.center,(cx,cz)))
        torches.append(TorchMount(wall))
    room = next(r for r in layout.rooms if r.kind == "reliquary")
    candidates = [w for w in walls if w.cell in room.cells and w not in windows
                  and not portal_width(w,layout)]
    torches.append(TorchMount(min(candidates,key=lambda w: math.dist(w.center,layout.reliquary))))
    return walls,tuple(windows),tuple(torches)


class Masonry:
    """CPU geometry with triangle normals and explicit ownership of surfaces."""

    def __init__(self, scene, seed=None):
        self.scene = scene
        self.rows = []
        self.rng = random.Random(scene.level_seed if seed is None else seed)

    def quad(self, a, b, c, d, color=STONE, *, diagonal=False, plain=False):
        points = tuple(Vector3(p) for p in (a,b,c,d))
        indices = ((0,1,3),(1,2,3)) if diagonal else ((0,1,2),(0,2,3))
        for triangle in indices:
            pa,pb,pc = (points[i] for i in triangle)
            cross = (pb-pa).cross(pc-pa)
            if cross.length_squared() < 1e-12:
                raise ValueError("Degenerate crypt surface")
            normal = cross.normalize()
            axis = max(range(3),key=lambda i: abs(normal[i]))
            for index in triangle:
                p = points[index]
                uv = ((.1,.05) if plain else (p.x/64,p.z/64) if axis == 1 else
                      (p.z/64,p.y/64) if axis == 0 else (p.x/64,p.y/64))
                self.rows.append((*p,*color,*normal,*uv))

    def collision(self, bounds, *, navigation_obstacle=True):
        x0,x1,y0,y1,z0,z1 = bounds
        tile = WallTile(
            position=Vector3((x0+x1)/2,(y0+y1)/2,(z0+z1)/2),
            width=(x1-x0)/2,thickness=x1-x0,height=(y1-y0)/2,depth=(z1-z0)/2)
        tile.navigation_obstacle = navigation_obstacle
        self.scene.wall_tiles.append(tile)

    def box(self, bounds, color=STONE, *, collision=False, top=True, bottom=False, plain=False):
        x0,x1,y0,y1,z0,z1 = bounds
        for face in (
            ((x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)),
            ((x1,y0,z0),(x0,y0,z0),(x0,y1,z0),(x1,y1,z0)),
            ((x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)),
            ((x1,y0,z1),(x1,y0,z0),(x1,y1,z0),(x1,y1,z1)),
        ):
            self.quad(*face,color,plain=plain)
        if top:
            self.quad((x0,y1,z1),(x1,y1,z1),(x1,y1,z0),(x0,y1,z0),color,plain=plain)
        if bottom:
            self.quad((x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1),color,plain=plain)
        if collision:
            self.collision(bounds)

    def cylinder(self, x, z, y0, y1, r0, r1, color=PALE):
        for i in range(12):
            a,b = i*math.tau/12,(i+1)*math.tau/12
            self.quad((x+r0*math.cos(a),y0,z+r0*math.sin(a)),
                      (x+r1*math.cos(a),y1,z+r1*math.sin(a)),
                      (x+r1*math.cos(b),y1,z+r1*math.sin(b)),
                      (x+r0*math.cos(b),y0,z+r0*math.sin(b)),color)

    def column(self, x, z):
        self.box((x-18,x+18,0,7,z-18,z+18),MOSS,collision=True)
        self.cylinder(x,z,7,14,16,11)
        self.cylinder(x,z,14,68,10,9)
        self.cylinder(x,z,68,76,9,18)
        self.box((x-22,x+22,76,SPRING_HEIGHT,z-22,z+22),PALE,bottom=True)
        # Solid imposts receive four arches; vaulted ceilings end at the ribs.
        r = RIB_HALF_WIDTH
        self.box((x-r,x+r,SPRING_HEIGHT,ROOF_TOP,z-r,z+r),PALE)
        self.collision((x-11,x+11,7,SPRING_HEIGHT,z-11,z+11))

    def arcade(self, start, end):
        a,b = Vector3(start[0],0,start[1]),Vector3(end[0],0,end[1])
        tangent = (b-a).normalize()
        normal = Vector3(-tangent.z,0,tangent.x)
        span,r = (b-a).length(),RIB_HALF_WIDTH
        def point(u,y,side):
            return a+tangent*u+normal*side+Vector3(0,y,0)
        def underside(u):
            value = (u-span/2)/(span/2)
            return SPRING_HEIGHT+VAULT_RISE*math.sqrt(max(0.,1-value*value))
        for k in range(20):
            u0,u1 = r+(span-2*r)*k/20,r+(span-2*r)*(k+1)/20
            y0,y1 = underside(u0),underside(u1)
            color = tuple(c*self.rng.uniform(.94,1.01) for c in PALE)
            for face in (
                (point(u0,y0,-r),point(u1,y1,-r),point(u1,ROOF_TOP,-r),point(u0,ROOF_TOP,-r)),
                (point(u1,y1,r),point(u0,y0,r),point(u0,ROOF_TOP,r),point(u1,ROOF_TOP,r)),
                (point(u0,y0,r),point(u1,y1,r),point(u1,y1,-r),point(u0,y0,-r)),
                (point(u0,ROOF_TOP,-r),point(u1,ROOF_TOP,-r),point(u1,ROOF_TOP,r),point(u0,ROOF_TOP,r)),
            ):
                self.quad(*reversed(face),color)

    def vault(self, bounds):
        x0,x1,z0,z1 = bounds
        cx,cz,hx,hz = (x0+x1)/2,(z0+z1)/2,(x1-x0)/2,(z1-z0)/2
        xs = np.linspace(-1+RIB_HALF_WIDTH/hx,1-RIB_HALF_WIDTH/hx,17)
        zs = np.linspace(-1+RIB_HALF_WIDTH/hz,1-RIB_HALF_WIDTH/hz,17)
        def point(u,v):
            height = SPRING_HEIGHT+10+VAULT_RISE*max(math.sqrt(max(0,1-u*u)),math.sqrt(max(0,1-v*v)))
            return (cx+u*hx,height,cz+v*hz)
        for i in range(16):
            for j in range(16):
                u0,u1,v0,v1 = xs[i],xs[i+1],zs[j],zs[j+1]
                self.quad(point(u0,v0),point(u1,v0),point(u1,v1),point(u0,v1),PALE,
                          diagonal=(u0+u1)*(v0+v1)<0)

    def wall(self, site, *, window=False, niches=True, door_width=0., passage=False):
        half = site.span/2
        height = PASSAGE_SPRING if passage else ROOF_TOP
        portal_top = PASSAGE_SPRING+PASSAGE_RISE+2
        niche_half = min(25,site.span*.16)
        holes = ((-door_width/2,door_width/2,0,portal_top),) if door_width else ((-30,30,49,105),) if window else tuple(
            (u-niche_half,u+niche_half,y,y+(14 if passage else 25))
            for u in (-site.span*.22,site.span*.22) for y in ((9,29) if passage else (12,47,82))
        ) if niches else ()
        us = sorted({-half,half,*(h[i] for h in holes for i in (0,1))})
        ys = sorted({0.,18.,36.,height,*(h[i] for h in holes for i in (2,3))})
        for u0,u1 in zip(us,us[1:]):
            for y0,y1 in zip(ys,ys[1:]):
                if any(h[0] <= (u0+u1)/2 <= h[1] and h[2] <= (y0+y1)/2 <= h[3] for h in holes):
                    continue
                color = MOSS if y1 <= 18 else (.45,.46,.34) if y1 <= 36 else STONE
                self.quad(site.point(u0,y0),site.point(u1,y0),site.point(u1,y1),site.point(u0,y1),color)
        for u0,u1,y0,y1 in holes:
            if door_width:
                # This spandrel meets the passage's barrel vault at its actual
                # underside. The tunnel supplies the jambs through the wall.
                for (a,ya),(b,yb) in zip(arch_profile(door_width),arch_profile(door_width)[1:]):
                    self.quad(site.point(a,ya),site.point(b,yb),
                              site.point(b,portal_top),site.point(a,portal_top),PALE)
                continue
            front,back = 4.,-24.
            color = (.28,.72,.76) if window else (.38,.37,.29)
            self.quad(site.point(u0,y0,back),site.point(u1,y0,back),site.point(u1,y1,back),site.point(u0,y1,back),color,plain=window)
            self.quad(site.point(u0,y0,front),site.point(u0,y0,back),site.point(u0,y1,back),site.point(u0,y1,front),STONE)
            self.quad(site.point(u1,y0,back),site.point(u1,y0,front),site.point(u1,y1,front),site.point(u1,y1,back),STONE)
            self.quad(site.point(u0,y0,back),site.point(u0,y0,front),site.point(u1,y0,front),site.point(u1,y0,back),PALE)
            self.quad(site.point(u0,y1,front),site.point(u0,y1,back),site.point(u1,y1,back),site.point(u1,y1,front),STONE)
            if not window:
                self.quad(site.point(u0+7,y0+4,-15),site.point(u1-7,y0+4,-15),
                          site.point(u1-7,y0+12,-15),site.point(u0+7,y0+12,-15),(.66,.64,.51))
            else:
                for u in (-19,0,19):
                    self.quad(site.point(u-1,y0,-20),site.point(u+1,y0,-20),
                              site.point(u+1,y1,-20),site.point(u-1,y1,-20),(.13,.19,.18),plain=True)
        spans = ((-half,-door_width/2),(door_width/2,half)) if door_width else ((-half,half),)
        for a,b in spans:
            p0,p1 = site.point(a,0,-28),site.point(b,height,4)
            self.collision((min(p0[0],p1[0]),max(p0[0],p1[0]),0,height,
                            min(p0[2],p1[2]),max(p0[2],p1[2])))

    def passage(self, room):
        for site in passage_walls(room):
            self.wall(site,passage=True)
        x0,x1,z0,z1 = room.bounds
        width = x1-x0 if room.along_z else z1-z0
        v0,v1 = (z0-4,z1+4) if room.along_z else (x0-4,x1+4)
        def point(u,y,v):
            return ((x0+x1)/2+u,y,v) if room.along_z else (v,y,(z0+z1)/2+u)
        bands = max(1,int((v1-v0)/88))
        profile = arch_profile(width)
        for (u0,y0),(u1,y1) in zip(profile,profile[1:]):
            a,b = point(u0,min(y0,y1),v0),point(u1,ROOF_TOP,v1)
            self.collision((min(a[0],b[0]),max(a[0],b[0]),a[1],b[1],
                            min(a[2],b[2]),max(a[2],b[2])),navigation_obstacle=False)
        for k in range(bands):
            start,end = v0+(v1-v0)*k/bands,v0+(v1-v0)*(k+1)/bands
            for a,b,color in ((start,start+8,STONE),(start+8,end,PALE)):
                for (u0,y0),(u1,y1) in zip(profile,profile[1:]):
                    face = (point(u0,y0,a),point(u1,y1,a),point(u1,y1,b),point(u0,y0,b))
                    self.quad(*(face if room.along_z else reversed(face)),color)

    def torch_mount(self, mount):
        """An embedded iron plate and projecting arm physically hold the torch."""
        site = mount.site
        iron = (.18,.20,.18)
        def block(u0,u1,y0,y1,d0,d1,color):
            a,b = site.point(u0,y0,d0),site.point(u1,y1,d1)
            self.box((min(a[0],b[0]),max(a[0],b[0]),y0,y1,
                      min(a[2],b[2]),max(a[2],b[2])),color,bottom=True,plain=True)
        # The plate penetrates the stone face at depth 4, leaving no air gap.
        block(-4,4,24,39,3,6,iron)
        block(-1.5,1.5,28,31,6,14,iron)
        x,_,z = mount.position
        self.cylinder(x,z,30,39,2,2,(.31,.21,.11))
        self.cylinder(x,z,37,40,3,4,iron)

    def tomb(self, x, z):
        self.box((x-16,x+16,0,22,z-30,z+30),STONE,collision=True,top=False)
        self.box((x-18,x+18,22,28,z-32,z+32),PALE,top=False)
        # Inlays partition the lid instead of layering coplanar crosses.
        xs,zs = (-18,-2,2,18),(-32,-14,-10,20,32)
        for x0,x1 in zip(xs,xs[1:]):
            for z0,z1 in zip(zs,zs[1:]):
                cross = (x0 == -2 and -14 <= z0 < 20) or z0 == -14
                color = (.40,.43,.35) if cross else PALE
                self.quad((x+x0,28,z+z1),(x+x1,28,z+z1),(x+x1,28,z+z0),(x+x0,28,z+z0),color)

    def upload(self, texture, *, floor=False):
        if not self.rows:
            return None
        mesh = BatchedMesh.from_vertex_data(np.asarray(self.rows,np.float32),texture=texture,
            lighting_receiver=GROUND_LIGHTING_RECEIVER if floor else TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
            casts_sun_shadows=False,shine_enabled=floor)
        self.rows.clear()
        return mesh


def create_masonry_texture():
    rng = random.Random(9204)
    surface = pygame.Surface((128,128),pygame.SRCALPHA)
    surface.fill((112,111,102,255))
    for row in range(8):
        for col in range(-1,5):
            x,y = col*32+(row%2)*16,row*16
            value = rng.randint(192,229)
            pygame.draw.rect(surface,(value,value,value-5),(x+1,y+1,30,14))
            pygame.draw.line(surface,(239,237,228),(x+2,y+2),(x+29,y+2))
    pixels = pygame.surfarray.pixels3d(surface)
    noise = np.random.default_rng(9204).integers(-12,13,(128,128,1))
    pixels[:] = np.clip(pixels.astype(np.int16)+noise,0,255)
    del pixels
    return upload_crypt_texture(surface)


def create_mosaic_texture(seed):
    rng = random.Random(seed)
    surface = pygame.Surface((128,128),pygame.SRCALPHA)
    surface.fill((63,71,62,255))
    for i in range(8):
        for j in range(8):
            base = (159,160,135) if (i+j)%2 else (118,138,125)
            color = tuple(int(c*rng.uniform(.90,1.05)) for c in base)
            pygame.draw.rect(surface,color,(i*16+1,j*16+1,15,15))
    return upload_crypt_texture(surface)


def upload_crypt_texture(surface):
    texture = _upload_texture_surface(surface,nearest=False,repeat=True)
    # Long, grazing views down the crypt otherwise shimmer at the mortar lines.
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MIN_FILTER,gl.GL_LINEAR_MIPMAP_LINEAR)
    return texture


def build_crypt_steps(scene, stone_texture, floor_texture):
    layout,m = scene.layout,Masonry(scene)
    for cell in sorted(layout.cells):
        x0,x1,z0,z1 = layout.cell_bounds(cell)
        for a,b,c,d,tint in ((x0+6,x1-6,z0+6,z1-6,(1.,1.,1.)),
                            (x0,x1,z0,z0+6,(.67,.83,.75)),
                            (x0,x1,z1-6,z1,(.67,.83,.75)),
                            (x0,x0+6,z0+6,z1-6,(.67,.83,.75)),
                            (x1-6,x1,z0+6,z1-6,(.67,.83,.75))):
            m.quad((a,0,d),(b,0,d),(b,0,c),(a,0,c),tint)
    for room in layout.passages:
        x0,x1,z0,z1 = room.bounds
        m.quad((x0,0,z1),(x1,0,z1),(x1,0,z0),(x0,0,z0),(.76,.84,.77))
    scene.ground_mesh = m.upload(floor_texture,floor=True)
    yield "Laying mosaic pavements",.1
    vertices,edges = set(),set()
    for i,j in sorted(layout.cells):
        corners = ((i,j),(i+1,j),(i+1,j+1),(i,j+1))
        vertices.update(corners)
        for k in range(4):
            edges.add(tuple(sorted((corners[k],corners[(k+1)%4]))))
    def corner(v):
        return ((v[0]-.5)*layout.bay_width,200-v[1]*layout.bay_depth)
    groups = {}
    for v in sorted(vertices):
        groups.setdefault((v[0]//2,v[1]//2),[[],[],[]])[0].append(v)
    for edge in sorted(edges):
        groups.setdefault((edge[0][0]//2,edge[0][1]//2),[[],[],[]])[1].append(edge)
    for cell in sorted(layout.cells):
        groups.setdefault((cell[0]//2,cell[1]//2),[[],[],[]])[2].append(cell)
    for index,(columns,arcades,bays) in enumerate(groups.values()):
        for v in columns:
            m.column(*corner(v))
        for a,b in arcades:
            m.arcade(corner(a),corner(b))
        for cell in bays:
            m.vault(layout.cell_bounds(cell))
        scene.wall_tile_batches.append(m.upload(stone_texture))
        yield "Raising columns and groin vaults",.1+.55*(index+1)/len(groups)
    scene.crypt_structure_counts = {"bays":len(layout.cells),"columns":len(vertices),"arcades":len(edges)}
    walls,windows,torches = scene.fixtures
    for room_index,room in enumerate(layout.rooms):
        if room.kind == "passage":
            m.passage(room)
        for site in walls:
            if site.cell in room.cells:
                m.wall(site,window=site in windows,niches=room.kind != "entrance",
                       door_width=portal_width(site,layout))
        for x,z in layout.tombs:
            if room.contains(x,z):
                m.tomb(x,z)
        mesh = m.upload(stone_texture)
        if mesh is not None:
            scene.wall_tile_batches.append(mesh)
        yield "Carving burial chambers",.65+.30*(room_index+1)/len(layout.rooms)
    for mount in torches:
        m.torch_mount(mount)
    x,z = layout.reliquary
    m.box((x-64,x+64,0,4,z-40,z+40),PALE,top=False)
    for a,b,c,d in ((x-64,x+64,z-40,z-32),(x-64,x+64,z+32,z+40),
                    (x-64,x-46,z-32,z+32),(x+46,x+64,z-32,z+32)):
        m.quad((a,4,d),(b,4,d),(b,4,c),(a,4,c),PALE)
    m.box((x-46,x+46,4,8,z-32,z+32),PALE)
    scene.wall_tile_batches.append(m.upload(stone_texture))
    scene.invalidate_collision_index()
    yield "Crypt masonry complete",1.
