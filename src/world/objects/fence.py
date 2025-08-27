"""Fence building utilities extracted from renderer.py."""

from __future__ import annotations
import random
import numpy as np
from textures.texture_utils import get_texture_size
from OpenGL.GL import (
    glGenBuffers,
    glBindBuffer,
    glBufferData,
    GL_ARRAY_BUFFER,
    GL_STATIC_DRAW,
)

from core.mesh import BatchedMesh


def build_textured_fence_ring(
    *,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    ground_y: float = 5.0,
    height_sampler=None,
    textures: list[int] | None = None,
    px_to_world: float = 1.0,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    wave_amp: float = 0.0,
    wave_freq: float = 0.02,
    wave_phase: float = 0.0,
    slices_per_segment: int | None = None,
    brightness_modifiers: list[callable] = None,
    default_brightness: float = 1.0,
) -> list[BatchedMesh]:
    if not textures:
        return []

    first_tex = textures[0]
    size = get_texture_size(first_tex)
    if size:
        w_px, h_px = size
        seg_nominal_w = float(w_px) * px_to_world
        seg_h = float(h_px) * px_to_world
    else:
        seg_nominal_w = 200.0 * px_to_world
        seg_h = 200.0 * px_to_world

    r, g, b = color

    if height_sampler is None:
        sampler_fn = None
    elif callable(height_sampler):
        sampler_fn = height_sampler
    elif hasattr(height_sampler, "height_at"):
        sampler_fn = lambda x, z, _hs=height_sampler: _hs.height_at(x, z)
    else:
        sampler_fn = None

    segments: list[tuple[float, float, float, float]] = []

    def add_edge_constant_x_segments(x: float, z_start: float, z_end: float):
        length = z_end - z_start
        dir_sign = 1.0 if length >= 0 else -1.0
        L = abs(length)
        if L <= 1e-6:
            return
        n = max(1, int(np.ceil(L / seg_nominal_w)))
        step = L / n
        for i in range(n):
            z0 = z_start + dir_sign * (i * step)
            z1 = z_start + dir_sign * ((i + 1) * step)
            segments.append((x, z0, x, z1))

    def add_edge_constant_z_segments(z: float, x_start: float, x_end: float):
        length = x_end - x_start
        dir_sign = 1.0 if length >= 0 else -1.0
        L = abs(length)
        if L <= 1e-6:
            return
        n = max(1, int(np.ceil(L / seg_nominal_w)))
        step = L / n
        for i in range(n):
            x0 = x_start + dir_sign * (i * step)
            x1 = x_start + dir_sign * ((i + 1) * step)
            segments.append((x0, z, x1, z))

    add_edge_constant_x_segments(min_x, min_z, max_z)
    add_edge_constant_x_segments(max_x, max_z, min_z)
    add_edge_constant_z_segments(min_z, min_x, max_x)
    add_edge_constant_z_segments(max_z, max_x, min_x)

    if not segments:
        return []

    verts_by_tex: dict[
        int, list[tuple[float, float, float, float, float, float, float, float]]
    ] = {t: [] for t in textures}

    def add_panel_flat(verts_list: list, x0: float, z0: float, x1: float, z1: float):
        y0_left = (
            float(sampler_fn(x0, z0)) if sampler_fn is not None else float(ground_y)
        )
        y0_right = (
            float(sampler_fn(x1, z1)) if sampler_fn is not None else float(ground_y)
        )
        y_top0 = y0_left + seg_h
        y_top1 = y0_right + seg_h

        verts_list.append((x0, y_top0, z0, r, g, b, 0.0, 1.0))
        verts_list.append((x1, y_top1, z1, r, g, b, 1.0, 1.0))
        verts_list.append((x1, y0_right, z1, r, g, b, 1.0, 0.0))
        verts_list.append((x0, y_top0, z0, r, g, b, 0.0, 1.0))
        verts_list.append((x1, y0_right, z1, r, g, b, 1.0, 0.0))
        verts_list.append((x0, y0_left, z0, r, g, b, 0.0, 0.0))

    def add_panel_wavey(verts_list: list, x0: float, z0: float, x1: float, z1: float):
        dx = x1 - x0
        dz = z1 - z0
        L = float(np.hypot(dx, dz))
        if L <= 1e-6:
            return
        if slices_per_segment is not None and slices_per_segment > 0:
            n = slices_per_segment
        else:
            target_slice_len = max(seg_nominal_w / 4.0, 1.0)
            n = max(2, int(np.ceil(L / target_slice_len)))

        for i in range(n):
            t0 = i / n
            t1 = (i + 1) / n
            sx0 = x0 + dx * t0
            sz0 = z0 + dz * t0
            sx1 = x0 + dx * t1
            sz1 = z0 + dz * t1

            s0 = L * t0
            s1 = L * t1
            off0 = wave_amp * float(np.sin(2.0 * np.pi * wave_freq * s0 + wave_phase))
            off1 = wave_amp * float(np.sin(2.0 * np.pi * wave_freq * s1 + wave_phase))

            y_bottom0 = (
                float(sampler_fn(sx0, sz0))
                if sampler_fn is not None
                else float(ground_y)
            )
            y_bottom1 = (
                float(sampler_fn(sx1, sz1))
                if sampler_fn is not None
                else float(ground_y)
            )
            y_top0 = y_bottom0 + seg_h + off0
            y_top1 = y_bottom1 + seg_h + off1

            u0 = t0
            u1 = t1

            verts_list.append((sx0, y_top0, sz0, r, g, b, u0, 1.0))
            verts_list.append((sx1, y_top1, sz1, r, g, b, u1, 1.0))
            verts_list.append((sx1, y_bottom1, sz1, r, g, b, u1, 0.0))
            verts_list.append((sx0, y_top0, sz0, r, g, b, u0, 1.0))
            verts_list.append((sx1, y_bottom1, sz1, r, g, b, u1, 0.0))
            verts_list.append((sx0, y_bottom0, sz0, r, g, b, u0, 0.0))

    for x0, z0, x1, z1 in segments:
        tex = random.choice(textures)
        if wave_amp > 0.0:
            add_panel_wavey(verts_by_tex[tex], x0, z0, x1, z1)
        else:
            add_panel_flat(verts_by_tex[tex], x0, z0, x1, z1)

    meshes: list[BatchedMesh] = []
    for tex, verts in verts_by_tex.items():
        if not verts:
            continue
        # Convert to numpy array for batch processing and potential lighting
        vertex_data = np.array(verts, dtype=np.float32)

        # Apply optional brightness/lighting modifiers (vectorized)
        if brightness_modifiers:
            print("test")
            try:
                N = vertex_data.shape[0]
                brightness_factor = np.full(N, float(default_brightness), dtype=np.float32)
                is_modified = np.zeros(N, dtype=bool)
                coords = vertex_data[:, [0, 2]]  # x, z

                # Mark vertices affected by any modifier
                for modifier in brightness_modifiers:
                    try:
                        position, radius, brightness_value, fall_off = modifier
                        center_x = position.x
                        center_z = position.z
                        dx = coords[:, 0] - center_x
                        dz = coords[:, 1] - center_z
                        distances = np.sqrt(dx * dx + dz * dz)
                        within_radius = distances <= radius
                        is_modified |= within_radius
                    except (ValueError, AttributeError, IndexError) as e:
                        print(f"Warning: Invalid brightness modifier {modifier}, skipping. Error: {e}")

                brightness_factor[~is_modified] = float(default_brightness)

                # Apply modifiers multiplicatively
                for modifier in brightness_modifiers:
                    try:
                        position, radius, brightness_value, fall_off = modifier
                        center_x = position.x
                        center_z = position.z
                        dx = coords[:, 0] - center_x
                        dz = coords[:, 1] - center_z
                        distances = np.sqrt(dx * dx + dz * dz)
                        within_radius = distances <= radius

                        # Normalized distance [0,1]
                        norm = distances / np.maximum(radius, 1e-12)
                        norm = np.clip(norm, 0.0, 1.0)
                        attenuation = (1.0 - norm) ** np.maximum(fall_off, 0.0)

                        if float(default_brightness) == 0.0:
                            rel = float(brightness_value)
                        else:
                            rel = float(brightness_value) / float(default_brightness)
                        modifier_effect = 1.0 + (rel - 1.0) * attenuation

                        # Apply only to vertices inside radius
                        brightness_factor[within_radius] *= modifier_effect[within_radius]
                    except (ValueError, AttributeError, IndexError) as e:
                        print(f"Warning: Invalid brightness modifier {modifier}, skipping. Error: {e}")

                # Multiply RGB columns by brightness factor
                vertex_data[:, 3:6] = vertex_data[:, 3:6] * brightness_factor[:, np.newaxis]
            except Exception as e:
                # Fallback: ensure we at least apply default brightness
                print(f"Warning: failed to apply brightness modifiers: {e}")
                vertex_data[:, 3:6] *= float(default_brightness)
        else:
            # No modifiers: apply default to all vertices
            vertex_data[:, 3:6] *= float(default_brightness)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        meshes.append(
            BatchedMesh(
                vbo_vertices=vbo, vertex_count=vertex_data.shape[0], texture=tex
            )
        )

    return meshes
