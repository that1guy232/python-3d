"""Player inventory data and slot operations."""

from __future__ import annotations

from dataclasses import dataclass
import time

INVENTORY_SLOT_COUNT = 24
INVENTORY_NOTICE_SECONDS = 3.0
TEST_GOBLIN_DROP_NAME = "Test Item"


@dataclass(frozen=True, slots=True)
class InventoryItem:
    """A named item stored in one player inventory slot."""

    name: str


def empty_inventory() -> list[InventoryItem | None]:
    """Create the fixed set of empty player inventory slots."""

    return [None] * INVENTORY_SLOT_COUNT


def inventory_slots(scene) -> list[InventoryItem | None]:
    """Return the scene's mutable inventory, padding legacy compact lists."""

    slots = getattr(scene, "inventory_items", None)
    if not isinstance(slots, list):
        slots = list(slots or ())
        scene.inventory_items = slots
    if len(slots) < INVENTORY_SLOT_COUNT:
        slots.extend([None] * (INVENTORY_SLOT_COUNT - len(slots)))
    return slots


def add_inventory_item(scene, item: InventoryItem) -> int | None:
    """Put an item in the first open slot and return that slot's index."""

    slots = inventory_slots(scene)
    for index in range(INVENTORY_SLOT_COUNT):
        if slots[index] is None:
            slots[index] = item
            return index
    return None


def move_inventory_item(scene, source: int, destination: int) -> bool:
    """Move an item between slots, swapping when the destination is occupied."""

    if not (
        0 <= int(source) < INVENTORY_SLOT_COUNT
        and 0 <= int(destination) < INVENTORY_SLOT_COUNT
    ):
        return False

    slots = inventory_slots(scene)
    source = int(source)
    destination = int(destination)
    if source == destination or slots[source] is None:
        return False
    slots[source], slots[destination] = slots[destination], slots[source]
    return True


def receive_inventory_item(
    scene,
    item: InventoryItem,
    *,
    now: float | None = None,
) -> int | None:
    """Add an item and show its temporary receipt in the bottom-right HUD."""

    slot = add_inventory_item(scene, item)
    if slot is None:
        return None

    ui = getattr(scene, "ui_state", scene)
    ui.inventory_notice_text = f"Recived {item.name}"
    ui.inventory_notice_expires_at = (
        time.monotonic() if now is None else float(now)
    ) + INVENTORY_NOTICE_SECONDS
    return slot


def active_inventory_notice(scene, *, now: float | None = None) -> str | None:
    """Return the current receipt, clearing it once its display time elapses."""

    ui = getattr(scene, "ui_state", scene)
    message = str(getattr(ui, "inventory_notice_text", "") or "")
    expires_at = float(getattr(ui, "inventory_notice_expires_at", 0.0) or 0.0)
    current = time.monotonic() if now is None else float(now)
    if not message or current >= expires_at:
        ui.inventory_notice_text = ""
        ui.inventory_notice_expires_at = 0.0
        return None
    return message
