"""Texture loading utilities for OpenGL.

This module keeps a tiny registry of texture sizes so other systems can
query width/height or aspect ratio from a texture ID without tracking
pygame surfaces.
"""

import math
import random
import pygame
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


def get_texture_size(tex_id: int) -> Optional[Tuple[int, int]]:
    """Return (width, height) for a loaded texture ID, if known."""
    return _TEXTURE_SIZES.get(tex_id)


def get_texture_aspect(tex_id: int, default: float = 1.0) -> float:
    """Return width/height aspect ratio for a loaded texture ID.

    Falls back to `default` if the ID isn't known.
    """
    wh = _TEXTURE_SIZES.get(tex_id)
    if not wh:
        return default
    w, h = wh
    return (w / h) if h else default


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

        # Get image data
        texture_data = pygame.image.tostring(surface, "RGBA", True)
        width, height = surface.get_size()

        # Generate OpenGL texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        # Upload texture data
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

        # Set texture parameters (use nearest to keep pixel art crisp)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        # Clamp edges to avoid sampling from opposite sides (prevents sprite seams)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Track texture size for later aspect-correct rendering
        _TEXTURE_SIZES[int(texture_id)] = (int(width), int(height))

        # print(f"Loaded texture: {filename} (ID: {texture_id}, Size: {width}x{height})")
        return texture_id

    except Exception as e:
        print(f"Failed to load texture {filename}: {e}")
        return create_test_texture()  # Fallback to test texture
        return None


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

    cache_key = (width_px, height_px, max_alpha, inner_ratio, outer_ratio, falloff_exp)
    cached = _SHADOW_TEX_CACHE.get(cache_key)
    if cached:
        return cached

    surface = pygame.Surface((width_px, height_px), pygame.SRCALPHA)

    cx, cy = (width_px - 1) * 0.5, (height_px - 1) * 0.5
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

    for y in range(height_px):
        dy = y - cy
        for x in range(width_px):
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

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    _TEXTURE_SIZES[int(tex_id)] = (width_px, height_px)
    _SHADOW_TEX_CACHE[cache_key] = int(tex_id)

    # print(f"Created shadow texture {width_px}x{height_px}, id={tex_id}")
    return int(tex_id)
