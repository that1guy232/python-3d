"""Lazy access to the deprecated texture-lighting shader."""

from __future__ import annotations


def get_legacy_texture_shader():
    """Load the compatibility module only when a legacy draw requests it."""

    from engine.core.compat_shader import get_texture_color_exposure_shader

    return get_texture_color_exposure_shader()
