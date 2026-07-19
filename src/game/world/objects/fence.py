"""Simplified fence building utilities - minimal version."""

import random
import numpy as np
from engine.textures.texture_utils import get_texture_size
from engine.core.mesh import BatchedMesh
from engine.rendering.geometry_lighting import uses_dynamic_textured_lighting
from engine.rendering.lighting import (
    apply_brightness_modifiers,
    apply_directional_sunlight,
    with_textured_normals,
)
from game.world.lighting_receivers import FENCE_LIGHTING_RECEIVER


def build_textured_fence_ring(
    min_x,
    max_x,
    min_z,
    max_z,
    ground_y=5.0,
    height_sampler=None,
    textures=None,
    px_to_world=1.0,
    color=(1.0, 1.0, 1.0),
    wave_amp=0.0,
    wave_freq=0.02,
    wave_phase=0.0,
    slices_per_segment=None,
    brightness_modifiers=None,
    default_brightness=1.0,
    lighting=None,
    sun_direction=None,
    dynamic_lighting: bool | None = None,
):
    """Build fence ring - simplified."""
    if not textures:
        return []

    # Get segment size
    size = get_texture_size(textures[0])
    if size:
        seg_w, seg_h = float(size[0]) * px_to_world, float(size[1]) * px_to_world
    else:
        seg_w, seg_h = 200.0 * px_to_world, 200.0 * px_to_world

    # Height function
    def height_at(x, z):
        if not height_sampler:
            return ground_y
        if callable(height_sampler):
            return height_sampler(x, z)
        if hasattr(height_sampler, "height_at"):
            return height_sampler.height_at(x, z)
        return ground_y

    # Create segments around perimeter
    def make_segments(x0, z0, x1, z1):
        length = np.hypot(x1 - x0, z1 - z0)
        if length <= 1e-6:
            return []
        n = max(1, int(np.ceil(length / seg_w)))
        dx, dz = (x1 - x0) / n, (z1 - z0) / n
        return [
            (x0 + i * dx, z0 + i * dz, x0 + (i + 1) * dx, z0 + (i + 1) * dz)
            for i in range(n)
        ]

    segments = (
        make_segments(min_x, min_z, min_x, max_z)  # Left
        + make_segments(max_x, max_z, max_x, min_z)  # Right
        + make_segments(min_x, min_z, max_x, min_z)  # Bottom
        + make_segments(max_x, max_z, min_x, max_z)  # Top
    )

    if not segments:
        return []

    # Generate vertices
    verts_by_tex = {t: [] for t in textures}
    r, g, b = color

    for x0, z0, x1, z1 in segments:
        tex = random.choice(textures)
        y0, y1 = height_at(x0, z0), height_at(x1, z1)

        if wave_amp > 0:
            # Wavy panels
            dx, dz, length = x1 - x0, z1 - z0, np.hypot(x1 - x0, z1 - z0)
            if length <= 1e-6:
                continue
            n = slices_per_segment or max(2, int(np.ceil(length / (seg_w / 4))))

            for i in range(n):
                t0, t1 = i / n, (i + 1) / n
                sx0, sz0 = x0 + dx * t0, z0 + dz * t0
                sx1, sz1 = x0 + dx * t1, z0 + dz * t1

                wave0 = wave_amp * np.sin(
                    2 * np.pi * wave_freq * length * t0 + wave_phase
                )
                wave1 = wave_amp * np.sin(
                    2 * np.pi * wave_freq * length * t1 + wave_phase
                )

                by0, by1 = height_at(sx0, sz0), height_at(sx1, sz1)
                ty0, ty1 = by0 + seg_h + wave0, by1 + seg_h + wave1

                quad = [
                    (sx0, ty0, sz0, r, g, b, t0, 1.0),
                    (sx1, ty1, sz1, r, g, b, t1, 1.0),
                    (sx1, by1, sz1, r, g, b, t1, 0.0),
                    (sx0, ty0, sz0, r, g, b, t0, 1.0),
                    (sx1, by1, sz1, r, g, b, t1, 0.0),
                    (sx0, by0, sz0, r, g, b, t0, 0.0),
                ]
                verts_by_tex[tex].extend(quad)
        else:
            # Flat panels
            ty0, ty1 = y0 + seg_h, y1 + seg_h
            quad = [
                (x0, ty0, z0, r, g, b, 0.0, 1.0),
                (x1, ty1, z1, r, g, b, 1.0, 1.0),
                (x1, y1, z1, r, g, b, 1.0, 0.0),
                (x0, ty0, z0, r, g, b, 0.0, 1.0),
                (x1, y1, z1, r, g, b, 1.0, 0.0),
                (x0, y0, z0, r, g, b, 0.0, 0.0),
            ]
            verts_by_tex[tex].extend(quad)

    # Create meshes
    meshes = []
    for tex, verts in verts_by_tex.items():
        if not verts:
            continue

        vertex_data = np.array(verts, dtype=np.float32)

        if uses_dynamic_textured_lighting(dynamic_lighting):
            vertex_data = with_textured_normals(vertex_data)
        else:
            apply_brightness_modifiers(
                vertex_data,
                modifiers=brightness_modifiers,
                default_brightness=default_brightness,
            )
            apply_directional_sunlight(
                vertex_data,
                lighting=lighting,
                sun_direction=sun_direction,
            )

        meshes.append(
            BatchedMesh.from_vertex_data(
                vertex_data,
                texture=tex,
                # Fence panels are texture cutouts, not solid rectangles.  The
                # same alpha test is consumed by the visible, sun-shadow, and
                # point-shadow passes so gaps between posts and rails remain
                # open in every representation.
                alpha_test=True,
                alpha_cutoff=0.5,
                exposure_baseline=default_brightness,
                lighting_receiver=FENCE_LIGHTING_RECEIVER,
            )
        )

    return meshes
