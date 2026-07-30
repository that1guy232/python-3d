"""Player inventory data and slot operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import time

BACKPACK_SLOT_COUNT = 24
EQUIPMENT_SLOT_COUNT = 4
INVENTORY_SLOT_COUNT = BACKPACK_SLOT_COUNT + EQUIPMENT_SLOT_COUNT
INVENTORY_NOTICE_SECONDS = 3.0
GOBLIN_FISTS_NAME = "Goblin fists"
GOBLIN_FISTS_ICON = "goblin_fist"
# The non-weapon pieces temporarily reuse the fist art through a deliberately
# named texture entry, so replacing the set's placeholder is a one-line asset
# catalog change rather than an item-data migration.
GOBLIN_SET_PLACEHOLDER_ICON = "goblin_set_placeholder"
GOBLIN_BOOTS_NAME = "Goblin scamper boots"
GOBLIN_BODY_NAME = "Goblin scrap armor"
GOBLIN_HEAD_NAME = "Goblin thinking cap"


class ItemType(str, Enum):
    """Inventory item categories, including the four equippable categories."""

    MISC = "misc"
    BODY = "body"
    BOOT = "boot"
    WEAPON = "weapon"
    HELMET = "helmet"


EQUIPMENT_TYPES: tuple[ItemType, ...] = (
    ItemType.BODY,
    ItemType.BOOT,
    ItemType.WEAPON,
    ItemType.HELMET,
)
EQUIPMENT_SLOT_TYPES: dict[int, ItemType] = {
    BACKPACK_SLOT_COUNT + offset: item_type
    for offset, item_type in enumerate(EQUIPMENT_TYPES)
}


@dataclass(frozen=True, slots=True)
class ItemCard:
    """A battle card contributed while its owning item is equipped."""

    action: str
    title: str
    detail: str
    mana_cost: int
    effect: str
    amount: int = 0
    requires_odd_mana: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", str(self.action))
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "detail", str(self.detail))
        object.__setattr__(self, "mana_cost", max(0, int(self.mana_cost)))
        object.__setattr__(self, "effect", str(self.effect))
        object.__setattr__(self, "amount", max(0, int(self.amount)))
        object.__setattr__(
            self,
            "requires_odd_mana",
            bool(self.requires_odd_mana),
        )


@dataclass(frozen=True, slots=True)
class InventoryItem:
    """A categorized item with player-facing descriptive details."""

    name: str
    item_type: ItemType = ItemType.MISC
    description: str = ""
    attributes: tuple[tuple[str, str], ...] | Mapping[str, object] = ()
    icon: str = ""
    cards: tuple[ItemCard, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.item_type, ItemType):
            object.__setattr__(self, "item_type", ItemType(self.item_type))
        object.__setattr__(self, "description", str(self.description or ""))
        object.__setattr__(self, "icon", str(self.icon or ""))
        raw_attributes = self.attributes
        if isinstance(raw_attributes, Mapping):
            pairs = raw_attributes.items()
        else:
            pairs = raw_attributes or ()
        object.__setattr__(
            self,
            "attributes",
            tuple((str(label), str(value)) for label, value in pairs),
        )
        normalized_cards: list[ItemCard] = []
        for card in self.cards or ():
            if isinstance(card, ItemCard):
                normalized_cards.append(card)
            elif isinstance(card, Mapping):
                normalized_cards.append(ItemCard(**card))
            else:
                raise TypeError("Inventory item cards must be ItemCard values")
        object.__setattr__(self, "cards", tuple(normalized_cards))


def goblin_equipment_set() -> tuple[InventoryItem, ...]:
    """Return the four goblin pieces and their complete card contribution."""

    return (
        InventoryItem(
            GOBLIN_FISTS_NAME,
            ItemType.WEAPON,
            (
                "Fight like a goblin: start fast, hit below the belt, then "
                "commit to a wild finish."
            ),
            {"Cards": "Knuckle Jab, Cheap Shot, Wild Flail"},
            GOBLIN_FISTS_ICON,
            (
                ItemCard(
                    "goblin_knuckle_jab",
                    "Knuckle Jab",
                    "1 Damage",
                    0,
                    "damage",
                    1,
                ),
                ItemCard(
                    "goblin_cheap_shot",
                    "Cheap Shot",
                    "2 Damage",
                    1,
                    "damage",
                    2,
                ),
                ItemCard(
                    "goblin_wild_flail",
                    "Wild Flail",
                    "3 Damage",
                    2,
                    "damage",
                    3,
                ),
            ),
        ),
        InventoryItem(
            GOBLIN_BOOTS_NAME,
            ItemType.BOOT,
            "Keep moving and rummage deeper into your bag of tricks.",
            {"Card": "Scamper", "Draw": "+2"},
            GOBLIN_SET_PLACEHOLDER_ICON,
            (
                ItemCard(
                    "goblin_scamper",
                    "Scamper",
                    "Draw 2 from Deck",
                    1,
                    "draw",
                    2,
                ),
            ),
        ),
        InventoryItem(
            GOBLIN_BODY_NAME,
            ItemType.BODY,
            "Scrap plating lets you risk a heavier incoming hit.",
            {"Card": "Scrap Guard", "Guard": "+4"},
            GOBLIN_SET_PLACEHOLDER_ICON,
            (
                ItemCard(
                    "goblin_scrap_guard",
                    "Scrap Guard",
                    "Block 4 Next Hit",
                    1,
                    "guard",
                    4,
                ),
            ),
        ),
        InventoryItem(
            GOBLIN_HEAD_NAME,
            ItemType.HELMET,
            "Goblin arithmetic only works when your current mana is odd.",
            {"Card": "Goblin Math", "Rule": "Odd mana only"},
            GOBLIN_SET_PLACEHOLDER_ICON,
            (
                ItemCard(
                    "goblin_math",
                    "Goblin Math",
                    "Odd Mana: 3 Damage",
                    1,
                    "damage",
                    3,
                    requires_odd_mana=True,
                ),
            ),
        ),
    )


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
    """Put an item in the first open backpack slot and return its index."""

    slots = inventory_slots(scene)
    for index in range(BACKPACK_SLOT_COUNT):
        if slots[index] is None:
            slots[index] = item
            return index
    return None


def item_type(item) -> ItemType:
    """Return an item's category while tolerating legacy dict-like items."""

    if item is None:
        return ItemType.MISC
    if isinstance(item, dict):
        value = item.get("item_type", item.get("type", ItemType.MISC))
    else:
        value = getattr(item, "item_type", getattr(item, "type", ItemType.MISC))
    try:
        return value if isinstance(value, ItemType) else ItemType(value)
    except (TypeError, ValueError):
        return ItemType.MISC


def slot_accepts_item(slot: int, item) -> bool:
    """Return whether a slot may contain an item of the supplied category."""

    try:
        slot = int(slot)
    except (TypeError, ValueError):
        return False
    if not 0 <= slot < INVENTORY_SLOT_COUNT:
        return False
    if item is None or slot < BACKPACK_SLOT_COUNT:
        return True
    return item_type(item) is EQUIPMENT_SLOT_TYPES[slot]


def move_inventory_item(scene, source: int, destination: int) -> bool:
    """Move or swap items when both destination slot constraints are valid."""

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
    if not slot_accepts_item(destination, slots[source]):
        return False
    if not slot_accepts_item(source, slots[destination]):
        return False
    slots[source], slots[destination] = slots[destination], slots[source]
    battle_cards = getattr(scene, "battle_cards", None)
    sync_with_equipment = getattr(battle_cards, "sync_with_equipment", None)
    if callable(sync_with_equipment):
        sync_with_equipment()
    return True


def equipped_item(scene, item_kind: ItemType):
    """Return the item occupying the requested equipment slot, if any."""

    try:
        if not isinstance(item_kind, ItemType):
            item_kind = ItemType(item_kind)
    except (TypeError, ValueError):
        return None
    for slot, slot_kind in EQUIPMENT_SLOT_TYPES.items():
        if slot_kind is item_kind:
            return inventory_slots(scene)[slot]
    return None


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
