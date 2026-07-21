"""Explicit mutable state owners for the world subsystem.

The scene orchestrates lifecycle.  These dataclasses own the collections that
builders and controllers exchange so their construction order and mutation
contract are visible in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, TYPE_CHECKING, TypeVar

from engine.entity import Entity
from game.config import (
    BASE_SPEED,
    CAMERA_FOLLOW_SMOOTH_HZ,
    CLOUDS_ENABLED,
    CLOUD_DENSITY,
    CLOUD_OPACITY,
    CLOUD_SPEED,
    FOGDENSITY,
    FOV,
    GRAVITY,
    JUMP_SPEED,
    MOUSE_SENSITIVITY,
    SPRINT_SPEED,
)
from game.inventory import empty_inventory

if TYPE_CHECKING:
    from engine.rendering.lighting_state import LocalBrightnessLight
    from game.world.environment import EnvironmentVolume
    from game.actors.creature import CombatCreature
    from game.actors.goblin import Goblin
    from game.inventory import InventoryItem
    from game.world.objects import Chest, Door, Road, Torch, Window
    from game.world.objects.building import Building
    from game.world.objects.polygon import Polygon
    from game.world.objects.wall_tile import WallTile


class Drawable(Protocol):
    """A scene-owned value that can participate in a render pass."""

    def draw(self, *args: Any, **kwargs: Any) -> None: ...


class Disposable(Protocol):
    """A GPU-backed value with explicit lifetime management."""

    def dispose(self) -> None: ...


class SpriteItem(Protocol):
    """Minimum boundary used by sprite update and batching code."""

    def update(self, dt: float) -> None: ...


class CollisionMesh(Protocol):
    """Collision source consumed by :class:`SceneCollisionIndex`."""

    def get_bounding_box(self) -> tuple[float, float, float, float]: ...


# Disposal is intentionally capability-checked because a few CPU-only
# drawables share these render passes. GPU lifetime ownership still belongs to
# WorldRenderResources and SceneResourceDisposer.
RenderValue = Drawable


@dataclass(slots=True)
class WorldBuildState:
    """Semantic objects and planning results produced while building a world."""

    building_specs: list[dict[str, Any]] = field(default_factory=list)
    initial_local_lights: list[LocalBrightnessLight] = field(default_factory=list)
    environment_volumes: list[EnvironmentVolume] = field(default_factory=list)
    buildings: list[Building] = field(default_factory=list)
    roads: list[Road] = field(default_factory=list)
    building_roads: list[Road] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)
    walls: list[WallTile] = field(default_factory=list)
    torches: list[Torch] = field(default_factory=list)
    creatures: list[CombatCreature] = field(default_factory=list)
    goblins: list[Goblin] = field(default_factory=list)
    chests: list[Chest] = field(default_factory=list)
    showcase_chests: list[Chest] = field(default_factory=list)
    showcase_polygons: list[Polygon] = field(default_factory=list)
    inventory_items: list[InventoryItem | None] = field(default_factory=empty_inventory)
    building_road_routes: list[list[tuple[float, float]]] = field(default_factory=list)
    building_road_segments: list[
        tuple[tuple[tuple[float, float], tuple[float, float]], float]
    ] = field(default_factory=list)
    builder: Any | None = None


@dataclass(slots=True)
class WorldRenderResources:
    """Render, collision, and entity resources owned by one world lifecycle."""

    sprite_items: list[SpriteItem] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    immediate_entities: list[Entity] = field(default_factory=list)
    decals: list[RenderValue] = field(default_factory=list)
    decal_batches: list[RenderValue] = field(default_factory=list)
    wall_tiles: list[CollisionMesh] = field(default_factory=list)
    wall_tile_batches: list[RenderValue] = field(default_factory=list)
    road_batches: list[RenderValue] = field(default_factory=list)
    door_batches: list[RenderValue] = field(default_factory=list)
    window_batches: list[RenderValue] = field(default_factory=list)
    polygons: list[CollisionMesh] = field(default_factory=list)
    polygon_batches: list[RenderValue] = field(default_factory=list)
    others: list[Drawable] = field(default_factory=list)
    fence_meshes: list[RenderValue] = field(default_factory=list)

    ground_mesh: RenderValue | None = None
    sky: RenderValue | None = None
    road: RenderValue | None = None
    decal_batch: RenderValue | None = None
    tree_shadow_caster: Disposable | None = None
    ground_height_sampler: Callable[[float, float], float] | None = None
    collision_cell_size: float = 128.0
    collision_spatial_index: dict[str, Any] | None = None
    sprite_update_cache: dict[str, Any] | None = None

    ground_tex: Any | None = None
    road_tex: Any | None = None
    tree_textures: list[Any] = field(default_factory=list)
    grasses_textures: list[Any] = field(default_factory=list)
    rock_textures: list[Any] = field(default_factory=list)
    fence_textures: list[Any] = field(default_factory=list)
    item_textures: dict[str, Any] = field(default_factory=dict)
    equipment_slot_textures: dict[str, Any] = field(default_factory=dict)
    wall_tex: Any | None = None
    torch_tex: Any | None = None
    door_tex: Any | None = None
    window_tex: Any | None = None
    goblin_tex: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorldUIState:
    """UI components, visibility, menu state, and user-adjustable presentation."""

    hud: Any | None = None
    battle_cards: Any | None = None
    battle_overlay: Any | None = None
    battle_menu: Any | None = None
    pause_menu: Any | None = None
    setting_menu: Any | None = None
    paused: bool = False
    inventory_open: bool = False
    showing_settings_menu: bool = False
    battle_mode: bool = False
    active_battle_creature: Entity | None = None
    inventory_selected_slot: int | None = None
    inventory_drag_source: int | None = None
    inventory_notice_text: str = ""
    inventory_notice_expires_at: float = 0.0
    hud_visible: bool = True
    compass_visible: bool = True
    minimap_visible: bool = True
    held_item_visible: bool = True
    test_light_visible: bool = True
    controls_text_visible: bool = True
    debug_text_visible: bool = True
    last_mouse_pos: tuple[int, int] = (0, 0)
    fov: float = FOV
    fog_enabled: bool = True
    fog_density: float = FOGDENSITY
    clouds_enabled: bool = CLOUDS_ENABLED
    cloud_density: float = CLOUD_DENSITY
    cloud_speed: float = CLOUD_SPEED
    cloud_opacity: float = CLOUD_OPACITY
    vibrance: float = 1.15
    mouse_sensitivity: float = MOUSE_SENSITIVITY
    walk_speed: float = BASE_SPEED
    sprint_speed: float = SPRINT_SPEED
    road_speed_multiplier: float = 1.5
    jump_speed: float = JUMP_SPEED
    gravity: float = GRAVITY
    camera_follow_smooth_hz: float = CAMERA_FOLLOW_SMOOTH_HZ

    @property
    def active_battle_goblin(self) -> Entity | None:
        """Compatibility alias for the former Goblin-only battle target."""
        return self.active_battle_creature

    @active_battle_goblin.setter
    def active_battle_goblin(self, value: Entity | None) -> None:
        self.active_battle_creature = value


OwnerT = TypeVar("OwnerT")
ValueT = TypeVar("ValueT")


class StateAlias(Generic[OwnerT, ValueT]):
    """Compatibility descriptor forwarding a scene attribute to its owner."""

    def __init__(self, owner_name: str, value_name: str) -> None:
        self.owner_name = owner_name
        self.value_name = value_name

    def __get__(self, instance: Any, owner: type | None = None) -> ValueT:
        if instance is None:
            return self  # type: ignore[return-value]
        return getattr(getattr(instance, self.owner_name), self.value_name)

    def __set__(self, instance: Any, value: ValueT) -> None:
        setattr(getattr(instance, self.owner_name), self.value_name, value)


def state_alias(owner_name: str, value_name: str | None = None) -> StateAlias[Any, Any]:
    """Declare a named compatibility alias without duplicating mutable state."""

    return StateAlias(owner_name, value_name or "")
