"""Base scene protocol used by the engine loop."""

from typing import List, Callable, Optional

UpdateFn = Callable[[float], None]


class Scene:
    """Minimal overridable surface for update, event, render, and disposal hooks."""

    # Camera is optional so non-3D scenes (e.g., main menu) don't need one
    # Use a generic object type to avoid importing `camera` at module import time.
    camera: Optional[object] = None
    # Scenes can request menu-style cursor behavior. When mouse_grabbed is None,
    # the engine grabs only if the cursor is hidden.
    mouse_visible: bool = False
    mouse_grabbed: Optional[bool] = None
    updaters: List[UpdateFn]
    # Optional screen-space night shade overlay; owned by base Scene so all
    # scenes can use it without duplicating initialization.
    # Use a lazy import inside the default_factory to avoid a circular import
    # when `world` package imports `core.scene` during module initialization.
    
    def __init__(self):
        self.updaters: list[UpdateFn] = []

    # Optional per-event handler (scenes can override)
    def handle_event(self, event) -> None:
        pass

    # Scenes can own their full render pipeline (projection, modelview, etc.)
    def render(self):  # pragma: no cover - visual
        # By default, do nothing; 3D scenes should override
        pass

    def dispose(self) -> None:
        """Release scene-owned graphics resources while the GL context exists."""
        pass
