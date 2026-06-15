"""Texture loading utilities for OpenGL.

This module keeps a tiny registry of texture sizes so other systems can
query width/height or aspect ratio from a texture ID without tracking
pygame surfaces.
"""

import math
import random
import pygame
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from OpenGL.GL import (
    glGenTextures,
    glBindTexture,
    glTexImage2D,
    glTexParameteri,
    GL_TEXTURE_2D,
    GL_RGBA,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_LINEAR,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_NEAREST,
    GL_CLAMP_TO_EDGE,
    GL_REPEAT,
)

_TEXTURE_SIZES: Dict[int, Tuple[int, int]] = {}
_SHADOW_TEX_CACHE: Dict[Tuple[int, int, float, float, float, float], int] = {}
_POLY_SHADOW_CACHE: Dict[Tuple[Tuple[float, float], ...], int] = {}
_TREE_SHADOW_TEX_CACHE: Dict[Tuple[object, ...], int] = {}
_FOREST_FLOOR_TEX_CACHE: Dict[Tuple[object, ...], int] = {}
_PIXEL_CLOUD_ATLAS_CACHE: Dict[Tuple[object, ...], Tuple["TextureRegion", ...]] = {}


@dataclass(frozen=True)
class TextureRegion:
    """A rectangular sub-image inside a larger OpenGL texture."""

    texture: int
    uv_rect: Tuple[float, float, float, float]
    size: Tuple[int, int]


def get_texture_size(tex_id: int) -> Optional[Tuple[int, int]]:
    """Return (width, height) for a loaded texture ID, if known."""
    if isinstance(tex_id, TextureRegion):
        return tex_id.size
    return _TEXTURE_SIZES.get(tex_id)


def get_texture_aspect(tex_id: int, default: float = 1.0) -> float:
    """Return width/height aspect ratio for a loaded texture ID.

    Falls back to `default` if the ID isn't known.
    """
    if isinstance(tex_id, TextureRegion):
        w, h = tex_id.size
        return (w / h) if h else default
    wh = _TEXTURE_SIZES.get(tex_id)
    if not wh:
        return default
    w, h = wh
    return (w / h) if h else default


def _upload_texture_surface(
    surface, *, nearest: bool = True, repeat: bool = False
) -> int:
    texture_data = pygame.image.tostring(surface, "RGBA", True)
    width, height = surface.get_size()

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        texture_data,
    )

    filt = GL_NEAREST if nearest else GL_LINEAR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt)
    wrap = GL_REPEAT if repeat else GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap)

    _TEXTURE_SIZES[int(texture_id)] = (int(width), int(height))
    return int(texture_id)


def load_texture(filename):
    """Load a texture from an image file.

    Parameters
    ----------
    filename : str
        Path to the image file

    Returns
    -------
    int
        OpenGL texture ID
    """
    try:
        # Load image with pygame
        surface = pygame.image.load(filename)

        # Convert to RGBA format
        surface = surface.convert_alpha()

        texture_id = _upload_texture_surface(surface, nearest=True, repeat=False)

        # print(f"Loaded texture: {filename} (ID: {texture_id}, Size: {width}x{height})")
        return texture_id

    except Exception as e:
        print(f"Failed to load texture {filename}: {e}")
        return create_test_texture()  # Fallback to test texture


def _load_surface_or_fallback(filename: str):
    try:
        surface = pygame.image.load(filename)
        try:
            return surface.convert_alpha()
        except pygame.error:
            return surface.copy()
    except Exception as e:
        print(f"Failed to load atlas texture {filename}: {e}")
        size = 64
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        red = (255, 0, 0, 255)
        transparent = (0, 0, 0, 0)
        for y in range(size):
            for x in range(size):
                color = red if ((x // 8) + (y // 8)) % 2 == 0 else transparent
                surface.set_at((x, y), color)
        return surface


def load_texture_atlas(
    filenames: list[str],
    *,
    padding: int = 2,
    max_width: int = 1024,
) -> list[TextureRegion]:
    """Load several images into one texture and return per-image UV regions."""
    if not filenames:
        return []

    padding = max(0, int(padding))
    max_width = max(64, int(max_width))
    surfaces = [_load_surface_or_fallback(filename) for filename in filenames]

    placements: list[tuple[int, int, int, int]] = []
    x = padding
    y = padding
    row_h = 0
    atlas_w = padding

    for surface in surfaces:
        w, h = surface.get_size()
        if x > padding and x + w + padding > max_width:
            atlas_w = max(atlas_w, x)
            x = padding
            y += row_h + padding
            row_h = 0
        placements.append((x, y, w, h))
        x += w + padding
        row_h = max(row_h, h)

    atlas_w = max(atlas_w, x)
    atlas_h = y + row_h + padding
    atlas = pygame.Surface((atlas_w, atlas_h), pygame.SRCALPHA)
    atlas.fill((0, 0, 0, 0))

    for surface, (px, py, _, _) in zip(surfaces, placements):
        atlas.blit(surface, (px, py))

    atlas_tex = _upload_texture_surface(atlas, nearest=True, repeat=False)
    regions: list[TextureRegion] = []
    for px, py, w, h in placements:
        u0 = (px + 0.5) / atlas_w
        u1 = (px + w - 0.5) / atlas_w
        v0 = 1.0 - ((py + h - 0.5) / atlas_h)
        v1 = 1.0 - ((py + 0.5) / atlas_h)
        regions.append(
            TextureRegion(
                texture=atlas_tex,
                uv_rect=(u0, v0, u1, v1),
                size=(w, h),
            )
        )

    return regions


def create_test_texture():
    """Create a simple RGBA test decal texture with soft edges.

    Produces a circular spot with a radial alpha falloff so overlapping
    looks correct when alpha blended over the ground.
    """
    # Create a red/blue checkerboard test texture (opaque)
    size = 64
    surface = pygame.Surface((size, size), pygame.SRCALPHA)

    # Checkerboard tile size in pixels
    tile = 8
    # Use red squares and transparent squares (alpha) so the wall shows through
    red = (255, 0, 0, 255)
    transparent = (0, 0, 0, 0)

    for y in range(size):
        ty = y // tile
        for x in range(size):
            tx = x // tile
            if (tx + ty) % 2 == 0:
                surface.set_at((x, y), red)
            else:
                surface.set_at((x, y), transparent)

    # Convert to RGBA texture
    texture_data = pygame.image.tostring(surface, "RGBA", True)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        size,
        size,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        texture_data,
    )

    # Use nearest filtering to keep the checkerboard crisp and allow tiling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # Allow the test texture to repeat so uv_repeat on walls tiles it instead
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    _TEXTURE_SIZES[int(texture_id)] = (size, size)

    print(f"Created checkerboard test texture (ID: {texture_id})")
    return texture_id


def create_shadow_texture(
    *,
    width_px: int = 256,
    height_px: int = 256,
    max_alpha: float = 0.22,
    inner_ratio: float = 0.25,
    outer_ratio: float = 0.95,
    falloff_exp: float = 2.6,
    pixelated: bool = False,
    pixel_scale: int = 1,
) -> int:
    """Create a high‑quality blob shadow texture with smooth falloff.

    Parameters
    ----------
    width_px, height_px: int
        Texture resolution. Use square (e.g., 256x256) for best quality; the
        decal mesh can scale non‑uniformly to form ellipses.
    max_alpha: float
        Peak opacity (0..1) at the shadow core.
    inner_ratio: float
        Radius ratio (0..1) defining the fully dark core relative to the half‑size.
    outer_ratio: float
        Radius ratio (inner_ratio..1] where the alpha fades to 0.
    falloff_exp: float
        Exponent applied to radial fade for a smoother, more natural falloff.

    Returns
    -------
    int
        OpenGL texture ID.

    Notes
    -----
    - Results are cached per parameter set to avoid duplicate GL textures.
    - Uses GL_LINEAR filtering and CLAMP_TO_EDGE for soft blending.
    """
    # Clamp/sanitize inputs
    width_px = max(8, int(width_px))
    height_px = max(8, int(height_px))
    max_alpha = float(max(0.0, min(1.0, max_alpha)))
    inner_ratio = float(max(0.0, min(1.0, inner_ratio)))
    outer_ratio = float(max(inner_ratio, min(1.0, outer_ratio)))
    falloff_exp = float(max(0.1, falloff_exp))
    pixelated = bool(pixelated)
    pixel_scale = max(1, int(pixel_scale))

    cache_key = (
        width_px,
        height_px,
        max_alpha,
        inner_ratio,
        outer_ratio,
        falloff_exp,
        pixelated,
        pixel_scale,
    )
    cached = _SHADOW_TEX_CACHE.get(cache_key)
    if cached:
        return cached

    # If pixelated rendering requested, draw at lower resolution then scale up
    if pixelated and pixel_scale > 1:
        render_w = max(8, width_px // pixel_scale)
        render_h = max(8, height_px // pixel_scale)
    else:
        render_w = width_px
        render_h = height_px

    surface = pygame.Surface((render_w, render_h), pygame.SRCALPHA)

    cx, cy = (render_w - 1) * 0.5, (render_h - 1) * 0.5
    # Use the smaller half‑dimension as our base radius to keep it inside the texture
    base_half = min(cx, cy)
    r_inner = base_half * inner_ratio
    r_outer = base_half * outer_ratio
    max_a = int(255 * max_alpha)

    # Small blue‑noise‑ish dither to reduce banding, deterministic hash
    def hash01(x: int, y: int) -> float:
        n = (x * 73856093) ^ (y * 19349663)
        n = (n ^ (n >> 13)) & 0xFFFFFFFF
        return n / 0xFFFFFFFF

    for y in range(render_h):
        dy = y - cy
        for x in range(render_w):
            dx = x - cx
            # normalized radial distance in pixels
            r = math.hypot(dx, dy)
            if r <= r_inner:
                a = max_a
            elif r >= r_outer:
                a = 0
            else:
                t = (r - r_inner) / max(1e-6, (r_outer - r_inner))
                # Smooth falloff with adjustable exponent
                t = pow(t, falloff_exp)
                a = int(max_a * (1.0 - t))
                # Apply a tiny ordered‑like dither to break banding
                d = (hash01(x, y) - 0.5) * 2.0  # [-1,1]
                a = max(0, min(255, int(a + d * 2.0)))

            surface.set_at((x, y), (0, 0, 0, a))

    # If we rendered at a lower resolution for pixelation, scale up to final size
    if (render_w, render_h) != (width_px, height_px):
        # Use nearest neighbor upscale to preserve blocky pixels
        surface = pygame.transform.scale(surface, (width_px, height_px))

    texture_data = pygame.image.tostring(surface, "RGBA", True)

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width_px,
        height_px,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        texture_data,
    )

    # Use nearest filtering for pixelated shadows to preserve blockiness,
    # otherwise use linear filtering for smooth soft shadows.
    if pixelated:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    else:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    _TEXTURE_SIZES[int(tex_id)] = (width_px, height_px)
    _SHADOW_TEX_CACHE[cache_key] = int(tex_id)

    # print(f"Created shadow texture {width_px}x{height_px}, id={tex_id}")
    return int(tex_id)


def _smooth01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _hash01(ix: int, iy: int, salt: int) -> float:
    n = (ix * 73856093) ^ (iy * 19349663) ^ (salt * 83492791)
    n = (n ^ (n >> 13)) & 0xFFFFFFFF
    return n / 0xFFFFFFFF


def _value_noise(x: float, y: float, salt: int) -> float:
    ix = math.floor(x)
    iy = math.floor(y)
    fx = _smooth01(x - ix)
    fy = _smooth01(y - iy)

    a = _hash01(ix, iy, salt)
    b = _hash01(ix + 1, iy, salt)
    c = _hash01(ix, iy + 1, salt)
    d = _hash01(ix + 1, iy + 1, salt)
    ab = a + (b - a) * fx
    cd = c + (d - c) * fx
    return ab + (cd - ab) * fy


def create_pixel_cloud_atlas(
    *,
    variant_count: int = 8,
    cell_width: int = 128,
    cell_height: int = 48,
    pixel_scale: int = 4,
    seed: int = 4021,
) -> list[TextureRegion]:
    """Create a small nearest-filtered atlas of pixel-art cloud silhouettes."""

    variant_count = max(1, int(variant_count))
    cell_width = max(32, int(cell_width))
    cell_height = max(16, int(cell_height))
    pixel_scale = max(1, int(pixel_scale))
    seed = int(seed)

    cache_key = (variant_count, cell_width, cell_height, pixel_scale, seed)
    cached = _PIXEL_CLOUD_ATLAS_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    atlas_width = cell_width * variant_count
    atlas_height = cell_height
    atlas = pygame.Surface((atlas_width, atlas_height), pygame.SRCALPHA)
    atlas.fill((0, 0, 0, 0))

    low_w = max(8, cell_width // pixel_scale)
    low_h = max(4, cell_height // pixel_scale)
    inv_w = 1.0 / max(1, low_w - 1)
    inv_h = 1.0 / max(1, low_h - 1)

    regions: list[TextureRegion] = []
    for variant in range(variant_count):
        rng = random.Random(seed + variant * 9173)
        surface = pygame.Surface((low_w, low_h), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))

        blob_count = rng.randint(5, 8)
        blobs = []
        for blob_index in range(blob_count):
            t = blob_index / max(1, blob_count - 1)
            cx = 0.14 + 0.72 * t + rng.uniform(-0.055, 0.055)
            cy = rng.uniform(0.45, 0.68) - 0.10 * math.sin(t * math.pi)
            rx = rng.uniform(0.13, 0.24) * (1.12 if 0.2 < t < 0.8 else 0.9)
            ry = rng.uniform(0.18, 0.33)
            weight = rng.uniform(0.88, 1.14)
            blobs.append((cx, cy, rx, ry, weight))

        shelf_y = rng.uniform(0.61, 0.73)
        shelf_half_height = rng.uniform(0.18, 0.26)
        salt = seed + variant * 37
        for y in range(low_h):
            ny = y * inv_h
            for x in range(low_w):
                nx = x * inv_w
                value = 0.0
                for cx, cy, rx, ry, weight in blobs:
                    dx = (nx - cx) / max(1e-6, rx)
                    dy = (ny - cy) / max(1e-6, ry)
                    d = dx * dx + dy * dy
                    if d < 1.0:
                        value = max(value, (1.0 - d) * weight)

                shelf = 1.0 - abs(ny - shelf_y) / shelf_half_height
                if 0.08 < nx < 0.92 and shelf > 0.0:
                    value = max(value, shelf * 0.48)

                noise = _value_noise(nx * 8.0 + variant, ny * 5.0, salt)
                edge = value + (noise - 0.5) * 0.16
                if edge <= 0.15:
                    continue

                alpha = (
                    255 if edge > 0.32 else int(255 * _smooth01((edge - 0.15) / 0.17))
                )
                bottom_mix = _smooth01((ny - 0.45) / 0.38)
                top_mix = 1.0 - _smooth01((ny - 0.18) / 0.36)
                r = int(218 + 31 * top_mix - 16 * bottom_mix)
                g = int(232 + 20 * top_mix - 21 * bottom_mix)
                b = int(246 + 8 * top_mix - 34 * bottom_mix)

                if noise > 0.72 and alpha > 220:
                    r = min(255, r + 10)
                    g = min(255, g + 10)
                    b = min(255, b + 8)

                surface.set_at(
                    (x, y),
                    (
                        max(0, min(255, r)),
                        max(0, min(255, g)),
                        max(0, min(255, b)),
                        max(0, min(255, alpha)),
                    ),
                )

        scaled = pygame.transform.scale(surface, (cell_width, cell_height))
        px = variant * cell_width
        atlas.blit(scaled, (px, 0))

        u0 = (px + 0.5) / atlas_width
        u1 = (px + cell_width - 0.5) / atlas_width
        v0 = 1.0 - ((cell_height - 0.5) / atlas_height)
        v1 = 1.0 - (0.5 / atlas_height)
        regions.append(
            TextureRegion(
                texture=0,
                uv_rect=(u0, v0, u1, v1),
                size=(cell_width, cell_height),
            )
        )

    atlas_tex = _upload_texture_surface(atlas, nearest=True, repeat=False)
    regions = [
        TextureRegion(
            texture=atlas_tex,
            uv_rect=region.uv_rect,
            size=region.size,
        )
        for region in regions
    ]
    _PIXEL_CLOUD_ATLAS_CACHE[cache_key] = tuple(regions)
    return list(regions)


def _segment_shadow_alpha(
    x: float,
    y: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    radius0: float,
    radius1: float,
    strength: float,
) -> float:
    dx = x1 - x0
    dy = y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-8:
        return 0.0

    t = ((x - x0) * dx + (y - y0) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    px = x0 + dx * t
    py = y0 + dy * t
    radius = radius0 + (radius1 - radius0) * t
    distance = math.hypot(x - px, y - py)
    if distance >= radius:
        return 0.0

    edge = 1.0 - distance / max(1e-6, radius)
    end_taper = _smooth01(t / 0.12) * _smooth01((1.0 - t) / 0.18)
    return strength * (edge**1.35) * end_taper


def _blend_rgba(
    base_rgb: tuple[float, float, float],
    base_alpha: float,
    top_rgb: tuple[float, float, float],
    top_alpha: float,
) -> tuple[tuple[float, float, float], float]:
    top_alpha = max(0.0, min(1.0, float(top_alpha)))
    if top_alpha <= 0.0:
        return base_rgb, base_alpha

    out_alpha = top_alpha + base_alpha * (1.0 - top_alpha)
    if out_alpha <= 1e-6:
        return base_rgb, 0.0

    br, bg, bb = base_rgb
    tr, tg, tb = top_rgb
    keep = base_alpha * (1.0 - top_alpha)
    out_rgb = (
        (tr * top_alpha + br * keep) / out_alpha,
        (tg * top_alpha + bg * keep) / out_alpha,
        (tb * top_alpha + bb * keep) / out_alpha,
    )
    return out_rgb, out_alpha


def create_forest_floor_texture(
    *,
    width_px: int = 192,
    height_px: int = 192,
    variant_seed: int = 0,
    alpha_scale: float = 0.9,
    pixelated: bool = False,
    pixel_scale: int = 1,
) -> int:
    """Create a muted leaf-litter and moss decal for tree bases."""
    width_px = max(8, int(width_px))
    height_px = max(8, int(height_px))
    variant_seed = int(variant_seed)
    alpha_scale = float(max(0.0, min(1.0, alpha_scale)))
    pixelated = bool(pixelated)
    pixel_scale = max(1, int(pixel_scale))

    cache_key = (
        width_px,
        height_px,
        variant_seed,
        alpha_scale,
        pixelated,
        pixel_scale,
    )
    cached = _FOREST_FLOOR_TEX_CACHE.get(cache_key)
    if cached:
        return cached

    if pixelated and pixel_scale > 1:
        render_w = max(8, width_px // pixel_scale)
        render_h = max(8, height_px // pixel_scale)
    else:
        render_w = width_px
        render_h = height_px

    rng = random.Random(variant_seed)
    surface = pygame.Surface((render_w, render_h), pygame.SRCALPHA)
    inv_w = 1.0 / max(1, render_w - 1)
    inv_h = 1.0 / max(1, render_h - 1)
    salt = variant_seed * 31 + 9
    phase_a = rng.uniform(0.0, 100.0)
    phase_b = rng.uniform(0.0, 100.0)

    leaf_colors = (
        (0.42, 0.29, 0.12),
        (0.52, 0.39, 0.18),
        (0.33, 0.25, 0.12),
        (0.27, 0.38, 0.18),
        (0.46, 0.45, 0.24),
    )
    twig_colors = (
        (0.25, 0.17, 0.08),
        (0.34, 0.23, 0.10),
        (0.18, 0.14, 0.08),
    )

    leaves: list[
        tuple[
            float,
            float,
            float,
            float,
            float,
            float,
            tuple[float, float, float],
            float,
        ]
    ] = []
    for _ in range(44):
        radius = math.sqrt(rng.random()) * rng.uniform(0.08, 0.44)
        angle = rng.uniform(0.0, math.tau)
        cx = math.cos(angle) * radius
        cy = math.sin(angle) * radius * 0.82
        length = rng.uniform(0.018, 0.055)
        width = length * rng.uniform(0.24, 0.46)
        rot = rng.uniform(0.0, math.tau)
        color = rng.choice(leaf_colors)
        alpha = rng.uniform(0.12, 0.30) * alpha_scale
        leaves.append(
            (cx, cy, length, width, math.cos(rot), math.sin(rot), color, alpha)
        )

    twigs: list[
        tuple[float, float, float, float, float, float, tuple[float, float, float]]
    ] = []
    for _ in range(12):
        radius = math.sqrt(rng.random()) * rng.uniform(0.06, 0.38)
        angle = rng.uniform(0.0, math.tau)
        cx = math.cos(angle) * radius
        cy = math.sin(angle) * radius * 0.82
        length = rng.uniform(0.05, 0.16)
        rot = rng.uniform(0.0, math.tau)
        half_len = length * 0.5
        x0 = cx - math.cos(rot) * half_len
        y0 = cy - math.sin(rot) * half_len
        x1 = cx + math.cos(rot) * half_len
        y1 = cy + math.sin(rot) * half_len
        twigs.append(
            (
                x0,
                y0,
                x1,
                y1,
                rng.uniform(0.006, 0.011),
                rng.uniform(0.10, 0.22) * alpha_scale,
                rng.choice(twig_colors),
            )
        )

    for py in range(render_h):
        y = py * inv_h - 0.5
        for px in range(render_w):
            x = px * inv_w - 0.5

            ellipse = (x / 0.48) * (x / 0.48) + (y / 0.42) * (y / 0.42)
            if ellipse >= 1.0:
                surface.set_at((px, py), (66, 51, 31, 0))
                continue

            edge_fade = _smooth01((1.0 - ellipse) / 0.32)
            coarse = _value_noise(
                x * 5.5 + phase_a,
                y * 5.0 + phase_b,
                salt,
            )
            fine = _value_noise(
                x * 18.0 + phase_b,
                y * 16.0 + phase_a,
                salt + 7,
            )
            moss = _value_noise(
                x * 8.0 + phase_a * 0.3,
                y * 8.0 + phase_b * 0.3,
                salt + 13,
            )

            soil_rgb = (0.26, 0.20, 0.12)
            leaf_rgb = (0.42 + 0.12 * coarse, 0.32 + 0.07 * coarse, 0.16)
            moss_rgb = (0.25, 0.36 + 0.10 * moss, 0.17)
            moss_mix = max(0.0, min(0.36, (moss - 0.50) * 0.7))
            leaf_mix = max(0.0, min(0.52, 0.20 + fine * 0.34))

            rgb = (
                soil_rgb[0] * (1.0 - leaf_mix) + leaf_rgb[0] * leaf_mix,
                soil_rgb[1] * (1.0 - leaf_mix) + leaf_rgb[1] * leaf_mix,
                soil_rgb[2] * (1.0 - leaf_mix) + leaf_rgb[2] * leaf_mix,
            )
            rgb = (
                rgb[0] * (1.0 - moss_mix) + moss_rgb[0] * moss_mix,
                rgb[1] * (1.0 - moss_mix) + moss_rgb[1] * moss_mix,
                rgb[2] * (1.0 - moss_mix) + moss_rgb[2] * moss_mix,
            )
            alpha = alpha_scale * edge_fade * (0.035 + 0.070 * coarse)

            speckle = _hash01(px, py, salt + 21)
            if speckle > 0.985:
                color_index = int(_hash01(px, py, salt + 22) * len(leaf_colors))
                speckle_color = leaf_colors[color_index % len(leaf_colors)]
                speckle_alpha = 0.08 + 0.14 * _hash01(px, py, salt + 23)
                rgb, alpha = _blend_rgba(
                    rgb,
                    alpha,
                    speckle_color,
                    alpha_scale * speckle_alpha * edge_fade,
                )

            for cx, cy, rx, ry, cos_r, sin_r, color, leaf_alpha in leaves:
                dx = x - cx
                dy = y - cy
                lx = dx * cos_r + dy * sin_r
                ly = -dx * sin_r + dy * cos_r
                d = (lx / rx) * (lx / rx) + (ly / ry) * (ly / ry)
                if d < 1.0:
                    rgb, alpha = _blend_rgba(
                        rgb,
                        alpha,
                        color,
                        leaf_alpha * edge_fade * (_smooth01(1.0 - d) ** 0.75),
                    )

            for x0, y0, x1, y1, radius0, twig_alpha, color in twigs:
                amount = _segment_shadow_alpha(
                    x,
                    y,
                    x0,
                    y0,
                    x1,
                    y1,
                    radius0,
                    radius0 * 0.72,
                    twig_alpha,
                )
                if amount > 0.0:
                    rgb, alpha = _blend_rgba(rgb, alpha, color, amount * edge_fade)

            alpha = max(0.0, min(0.52, alpha))
            r = max(0, min(255, int(rgb[0] * 255)))
            g = max(0, min(255, int(rgb[1] * 255)))
            b = max(0, min(255, int(rgb[2] * 255)))
            a = max(0, min(255, int(alpha * 255)))
            surface.set_at((px, py), (r, g, b, a))

    if (render_w, render_h) != (width_px, height_px):
        surface = pygame.transform.scale(surface, (width_px, height_px))

    texture_data = pygame.image.tostring(surface, "RGBA", True)

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width_px,
        height_px,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        texture_data,
    )

    filt = GL_NEAREST if pixelated else GL_LINEAR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    _TEXTURE_SIZES[int(tex_id)] = (width_px, height_px)
    _FOREST_FLOOR_TEX_CACHE[cache_key] = int(tex_id)
    return int(tex_id)


def create_tree_shadow_texture(
    *,
    width_px: int = 256,
    height_px: int = 256,
    max_alpha: float = 0.48,
    variant_seed: int = 0,
    pixelated: bool = False,
    pixel_scale: int = 1,
) -> int:
    """Create a smooth, continuous tree shadow texture.

    The mask avoids separate canopy blobs: it uses one tapered, soft silhouette
    with subtle low-frequency density variation and a faint trunk streak.
    """
    width_px = max(8, int(width_px))
    height_px = max(8, int(height_px))
    max_alpha = float(max(0.0, min(1.0, max_alpha)))
    variant_seed = int(variant_seed)
    pixelated = bool(pixelated)
    pixel_scale = max(1, int(pixel_scale))

    cache_key = (
        width_px,
        height_px,
        max_alpha,
        variant_seed,
        pixelated,
        pixel_scale,
    )
    cached = _TREE_SHADOW_TEX_CACHE.get(cache_key)
    if cached:
        return cached

    if pixelated and pixel_scale > 1:
        render_w = max(8, width_px // pixel_scale)
        render_h = max(8, height_px // pixel_scale)
    else:
        render_w = width_px
        render_h = height_px

    rng = random.Random(variant_seed)
    surface = pygame.Surface((render_w, render_h), pygame.SRCALPHA)
    inv_w = 1.0 / max(1, render_w - 1)
    inv_h = 1.0 / max(1, render_h - 1)
    salt = variant_seed * 17 + 3
    phase_a = rng.uniform(0.0, 100.0)
    phase_b = rng.uniform(0.0, 100.0)

    for py in range(render_h):
        y = py * inv_h - 0.5
        for px in range(render_w):
            x = px * inv_w - 0.5
            alpha = 0.0
            u = x + 0.5

            length_fade = _smooth01(u / 0.09) * _smooth01((1.0 - u) / 0.11)
            if length_fade > 0.0:
                bell = max(0.0, math.sin(math.pi * u))
                radius_noise = _value_noise(u * 5.0 + phase_a, phase_b, salt)
                radius = 0.028 + 0.325 * (bell**0.62)
                radius *= 0.98 - 0.16 * _smooth01(u)
                radius *= 0.96 + 0.08 * radius_noise

                center_noise = _value_noise(u * 4.0 + phase_b, phase_a, salt + 5)
                center_y = (center_noise - 0.5) * 0.035
                center_y += math.sin((u * 2.4 + phase_a) * math.tau) * 0.012

                distance = abs(y - center_y) / max(1e-6, radius)
                if distance < 1.0:
                    edge = 1.0 - distance
                    edge_fade = _smooth01(edge)
                    density = 0.82 + 0.12 * _value_noise(
                        u * 7.0 + phase_b,
                        (y + 0.5) * 6.0 + phase_a,
                        salt + 11,
                    )
                    fine = _value_noise(
                        u * 16.0 + phase_a,
                        (y + 0.5) * 14.0 + phase_b,
                        salt + 19,
                    )
                    gap = _value_noise(
                        u * 10.0 + phase_b,
                        (y + 0.5) * 9.0 + phase_a,
                        salt + 29,
                    )

                    gap_soften = 1.0
                    if gap > 0.72:
                        gap_soften = 0.82 + (1.0 - gap) * 0.28

                    alpha = (
                        max_alpha
                        * length_fade
                        * (edge_fade**0.82)
                        * density
                        * (0.95 + 0.08 * fine)
                        * gap_soften
                    )

            alpha += max_alpha * _segment_shadow_alpha(
                x,
                y,
                -0.48,
                0.0,
                -0.08,
                0.01,
                0.024,
                0.015,
                0.15,
            )

            a = max(0, min(255, int(255 * min(max_alpha, alpha))))
            surface.set_at((px, py), (0, 0, 0, a))

    if (render_w, render_h) != (width_px, height_px):
        surface = pygame.transform.scale(surface, (width_px, height_px))

    texture_data = pygame.image.tostring(surface, "RGBA", True)

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width_px,
        height_px,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        texture_data,
    )

    filt = GL_NEAREST if pixelated else GL_LINEAR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    _TEXTURE_SIZES[int(tex_id)] = (width_px, height_px)
    _TREE_SHADOW_TEX_CACHE[cache_key] = int(tex_id)
    return int(tex_id)


def create_polygon_shadow_texture(
    points_2d: list[tuple[float, float]],
    *,
    width_px: int = 256,
    height_px: int = 256,
    max_alpha: float = 0.8,
    inner_ratio: float = 0.05,
    outer_ratio: float = 0.95,
    falloff_exp: float = 1.8,
    pixelated: bool = False,
    pixel_scale: int = 1,
) -> int:
    """Rasterize a CCW polygon into a soft shadow texture.

    The function fits the provided polygon into the texture, computes a
    distance-to-edge value for interior pixels and maps that distance to an
    alpha value with a smooth falloff. Returns an OpenGL texture id.

    Parameters mirror create_shadow_texture naming where reasonable. The
    polygon should be provided in CCW order; concave polygons are supported.
    """
    # Normalize / sanitize simple params
    width_px = max(8, int(width_px))
    height_px = max(8, int(height_px))
    max_alpha = float(max(0.0, min(1.0, max_alpha)))
    inner_ratio = float(max(0.0, min(1.0, inner_ratio)))
    outer_ratio = float(max(inner_ratio, min(1.0, outer_ratio)))
    falloff_exp = float(max(0.1, falloff_exp))
    pixelated = bool(pixelated)
    pixel_scale = max(1, int(pixel_scale))

    # Simple cache keyed by point tuple and texture params to avoid re-upload
    pts_key = tuple((float(x), float(y)) for x, y in points_2d)
    cache_key = (
        pts_key,
        width_px,
        height_px,
        max_alpha,
        inner_ratio,
        outer_ratio,
        falloff_exp,
        pixelated,
        pixel_scale,
    )
    cached = _POLY_SHADOW_CACHE.get(cache_key)
    if cached:
        return cached

    # Degenerate fallback
    if not points_2d or len(points_2d) < 3:
        return create_shadow_texture(
            width_px=width_px,
            height_px=height_px,
            max_alpha=max_alpha,
            inner_ratio=inner_ratio,
            outer_ratio=outer_ratio,
            falloff_exp=falloff_exp,
            pixelated=pixelated,
            pixel_scale=pixel_scale,
        )

    # Compute polygon bbox in input space
    xs = [p[0] for p in points_2d]
    ys = [p[1] for p in points_2d]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w = max(1e-6, max_x - min_x)
    bbox_h = max(1e-6, max_y - min_y)

    # Prepare render resolution (support pixelation by rendering lower res then upscaling)
    if pixelated and pixel_scale > 1:
        render_w = max(8, width_px // pixel_scale)
        render_h = max(8, height_px // pixel_scale)
    else:
        render_w = width_px
        render_h = height_px

    # Map polygon into texture coordinates with slight padding so falloff can occur
    pad = 2.0  # pixels padding inside render space
    scale = min((render_w - 2 * pad) / bbox_w, (render_h - 2 * pad) / bbox_h)
    # center the polygon in texture space
    tx = (render_w - (bbox_w * scale)) * 0.5 - min_x * scale
    ty = (render_h - (bbox_h * scale)) * 0.5 - min_y * scale

    # Transform function: world (poly) -> texture pixel coords
    def to_tex(p):
        return (p[0] * scale + tx, p[1] * scale + ty)

    tex_pts = [to_tex(p) for p in points_2d]

    # Compute a local center (centroid of vertices); used to derive max radius
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    c_tx, c_ty = to_tex((cx, cy))

    # Precompute max radius as farthest vertex distance from centroid (in tex pixels)
    max_r = 0.0
    for px, py in tex_pts:
        d = math.hypot(px - c_tx, py - c_ty)
        if d > max_r:
            max_r = d
    max_r = max(1.0, max_r)
    inner_r = max_r * inner_ratio
    outer_r = max_r * outer_ratio

    # Helpers: point-in-polygon (ray casting) and distance to segment
    def point_in_poly(x, y, poly):
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            intersect = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            )
            if intersect:
                inside = not inside
            j = i
        return inside

    def dist_point_segment(px, py, x1, y1, x2, y2):
        # Project point onto segment and clamp
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    surface = pygame.Surface((render_w, render_h), pygame.SRCALPHA)

    # Precompute edges for distance calculations
    edges = []
    n = len(tex_pts)
    for i in range(n):
        x1, y1 = tex_pts[i]
        x2, y2 = tex_pts[(i + 1) % n]
        edges.append((x1, y1, x2, y2))

    # For each pixel, if inside polygon compute distance to nearest edge and map to alpha
    for y in range(render_h):
        for x in range(render_w):
            # center of pixel
            px = x + 0.5
            py = y + 0.5
            if not point_in_poly(px, py, tex_pts):
                a = 0
            else:
                # find min distance to boundary
                min_d = float("inf")
                for x1, y1, x2, y2 in edges:
                    d = dist_point_segment(px, py, x1, y1, x2, y2)
                    if d < min_d:
                        min_d = d

                # normalize into [0,1] based on inner/outer radii (in pixels)
                if min_d <= inner_r:
                    a = int(255 * max_alpha)
                elif min_d >= outer_r:
                    a = 0
                else:
                    t = (min_d - inner_r) / max(1e-6, (outer_r - inner_r))
                    t = pow(t, falloff_exp)
                    aval = (1.0 - t) * max_alpha
                    a = max(0, min(255, int(255 * aval)))

            surface.set_at((x, y), (0, 0, 0, a))

    # Upscale if we rendered small for pixelation
    if (render_w, render_h) != (width_px, height_px):
        surface = pygame.transform.scale(surface, (width_px, height_px))

    texture_data = pygame.image.tostring(surface, "RGBA", True)

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width_px,
        height_px,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        texture_data,
    )

    if pixelated:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    else:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    _TEXTURE_SIZES[int(tex_id)] = (width_px, height_px)
    _POLY_SHADOW_CACHE[cache_key] = int(tex_id)

    return int(tex_id)
