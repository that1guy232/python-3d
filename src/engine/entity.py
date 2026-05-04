"""Lightweight engine-level entity base class.

Entities are runtime objects that may update, draw, handle interaction, and
optionally expose collision meshes to a scene.
"""

from __future__ import annotations

from typing import Iterable

from pygame.math import Vector3


class Entity:
    """Base class for dynamic world objects owned by a scene."""

    def __init__(self, position=None, rotation=None) -> None:
        self.position = position or Vector3(0.0, 0.0, 0.0)
        self.rotation = rotation or Vector3(0.0, 0.0, 0.0)
        self.enabled = True
        self.visible = True
        self.collision_enabled = False
        self.interaction_distance = 0.0

    def update(self, dt: float) -> None:
        """Advance runtime state."""
        pass

    def draw(self) -> None:  # pragma: no cover - visual hook
        """Draw the entity when a scene chooses immediate-mode rendering."""
        pass

    def interact(self, actor=None, scene=None) -> bool:
        """Handle a player/entity interaction.

        Returns True when the interaction was consumed.
        """
        return False

    def get_interaction_position(self) -> Vector3:
        """Return the world-space point used for interaction targeting."""
        return self.position

    def get_collision_meshes(self) -> Iterable[object]:
        """Return collision meshes supplied by this entity."""
        return ()

    def get_sprites(self) -> Iterable[object]:
        """Return billboard sprites supplied by this entity."""
        return ()
