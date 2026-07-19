"""Backend-aware vertex lighting representation selection."""

from __future__ import annotations


def uses_dynamic_textured_lighting(explicit: bool | None) -> bool:
    """Resolve an explicit backend choice or preserve standalone legacy behavior."""

    if explicit is not None:
        return bool(explicit)

    # Standalone engine callers historically inferred their vertex layout from
    # compatibility-shader availability. Keep that adapter lazy so an explicit
    # packet world does not consult the legacy shader during construction.
    from engine.core.compat_shader import texture_color_exposure_shader_available

    return bool(texture_color_exposure_shader_available())
