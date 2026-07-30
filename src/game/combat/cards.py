"""Card deck, hand, discard, and turn actions for battle mode."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game.inventory import ItemCard, ItemType, equipped_item
from game.ui.card import Card


CardOrigin = tuple[ItemType, str, int]


@dataclass(frozen=True, slots=True)
class _CardSpec:
    """Declarative card contribution from one equipment slot."""

    origin: CardOrigin
    action: str
    title: str
    detail: str
    mana_cost: int
    effect: str
    amount: int = 0
    requires_odd_mana: bool = False


class BattleCardLoadout:
    """Own the player's battle deck and its equipment-driven cards."""

    BASE_STRIKE_COUNT = 1
    EQUIPPED_WEAPON_CARD_COUNT = 3
    STRIKE_MANA_COST = 1
    DEFAULT_CARD_MANA_COST = 1
    BRACE_GUARD = 2
    ODD_THOUGHT_DAMAGE = 2

    def __init__(self, scene) -> None:
        self.scene = scene
        self._all_cards: list[Card] = []
        self._deck: list[Card] = []
        self._hand: list[Card] = []
        self._discard: list[Card] = []
        self._cards_by_origin: dict[CardOrigin, Card] = {}
        self._specs_by_card: dict[Card, _CardSpec] = {}
        self._combat_active = False
        self.sync_with_equipment()

    @property
    def cards(self) -> list[Card]:
        """Return the current hand, or the full loadout outside combat."""

        self.sync_with_equipment()
        return self._hand if self._combat_active else self._all_cards

    @property
    def loadout_cards(self) -> tuple[Card, ...]:
        """Return a read-only snapshot of every card in the equipped loadout."""

        self.sync_with_equipment()
        return tuple(self._all_cards)

    @property
    def deck_count(self) -> int:
        return len(self._deck)

    @property
    def deck_cards(self) -> tuple[Card, ...]:
        """Return remaining draw-pile cards without revealing shuffle order."""

        self.sync_with_equipment()
        remaining = set(self._deck)
        return tuple(card for card in self._all_cards if card in remaining)

    @property
    def hand_count(self) -> int:
        return len(self._hand)

    @property
    def discard_count(self) -> int:
        return len(self._discard)

    @property
    def all_cards_discarded(self) -> bool:
        """Return whether every card is exhausted into the discard pile."""

        return (
            bool(self._all_cards)
            and not self._hand
            and not self._deck
            and len(self._discard) == len(self._all_cards)
        )

    def sync_with_equipment(self) -> None:
        """Reconcile card piles with every equipment-slot contribution."""

        desired_specs = self._desired_specs()
        desired_by_origin = {spec.origin: spec for spec in desired_specs}

        for origin, card in tuple(self._cards_by_origin.items()):
            current_spec = self._specs_by_card.get(card)
            desired_spec = desired_by_origin.get(origin)
            if desired_spec is not None and current_spec == desired_spec:
                continue
            card.reset_to_home()
            self._remove_from_piles(card)
            self._cards_by_origin.pop(origin, None)
            self._specs_by_card.pop(card, None)

        ordered_cards: list[Card] = []
        for spec in desired_specs:
            card = self._cards_by_origin.get(spec.origin)
            if card is None:
                card = self._build_card(spec)
                self._cards_by_origin[spec.origin] = card
                self._specs_by_card[card] = spec
                if self._combat_active:
                    self._deck.append(card)
            ordered_cards.append(card)

        self._all_cards[:] = ordered_cards

        if not self._combat_active:
            self._deck[:] = self._all_cards
            self._hand.clear()
            self._discard.clear()

    def _desired_specs(self) -> list[_CardSpec]:
        specs: list[_CardSpec] = []
        for item_kind in (
            ItemType.WEAPON,
            ItemType.BOOT,
            ItemType.BODY,
            ItemType.HELMET,
        ):
            item = equipped_item(self.scene, item_kind)
            item_cards = self._item_cards(item)
            if item is not None and item_cards:
                item_key = f"item:{self._item_name(item)}"
                specs.extend(
                    self._item_card_spec(item_kind, item_key, index, card)
                    for index, card in enumerate(item_cards)
                )
            else:
                specs.extend(
                    self._fallback_specs(
                        item_kind,
                        "empty" if item is None else f"item:{self._item_name(item)}",
                    )
                )
        return specs

    @staticmethod
    def _item_name(item) -> str:
        if isinstance(item, dict):
            return str(item.get("name", "") or "unnamed")
        return str(getattr(item, "name", "") or "unnamed")

    @staticmethod
    def _item_cards(item) -> tuple[ItemCard, ...]:
        if item is None:
            return ()
        raw_cards = item.get("cards", ()) if isinstance(item, dict) else getattr(
            item,
            "cards",
            (),
        )
        cards: list[ItemCard] = []
        for card in raw_cards or ():
            if isinstance(card, ItemCard):
                cards.append(card)
            elif isinstance(card, dict):
                cards.append(ItemCard(**card))
        return tuple(cards)

    @staticmethod
    def _item_card_spec(
        item_kind: ItemType,
        item_key: str,
        index: int,
        card: ItemCard,
    ) -> _CardSpec:
        return _CardSpec(
            (item_kind, item_key, index),
            card.action,
            card.title,
            card.detail,
            card.mana_cost,
            card.effect,
            card.amount,
            card.requires_odd_mana,
        )

    def _fallback_specs(
        self,
        item_kind: ItemType,
        origin_key: str,
    ) -> tuple[_CardSpec, ...]:
        """Supply the base loadout and support legacy items without card data."""

        if item_kind is ItemType.WEAPON:
            count = (
                self.BASE_STRIKE_COUNT
                if origin_key == "empty"
                else self.EQUIPPED_WEAPON_CARD_COUNT
            )
            return tuple(
                self._strike_spec(index, origin_key=origin_key)
                for index in range(count)
            )
        if item_kind is ItemType.BOOT:
            return (
                _CardSpec(
                    (item_kind, origin_key, 0),
                    "quickstep",
                    "Quickstep",
                    "Draw 1 from Deck",
                    self.DEFAULT_CARD_MANA_COST,
                    "draw",
                    1,
                ),
            )
        if item_kind is ItemType.BODY:
            return (
                _CardSpec(
                    (item_kind, origin_key, 0),
                    "brace",
                    "Brace",
                    f"Block {self.BRACE_GUARD} Next Hit",
                    self.DEFAULT_CARD_MANA_COST,
                    "guard",
                    self.BRACE_GUARD,
                ),
            )
        return (
            _CardSpec(
                (item_kind, origin_key, 0),
                "odd_thought",
                "Odd Thought",
                f"Odd Mana: {self.ODD_THOUGHT_DAMAGE} Damage",
                self.DEFAULT_CARD_MANA_COST,
                "damage",
                self.ODD_THOUGHT_DAMAGE,
                requires_odd_mana=True,
            ),
        )

    def _strike_spec(self, index: int, *, origin_key: str) -> _CardSpec:
        return _CardSpec(
            (ItemType.WEAPON, origin_key, index),
            f"strike_{index + 1}",
            "Strike",
            "1 Damage",
            self.STRIKE_MANA_COST,
            "damage",
            1,
        )

    def _remove_from_piles(self, removed: Card) -> None:
        for pile in (self._deck, self._hand, self._discard):
            pile[:] = [card for card in pile if card is not removed]

    def start_battle(self) -> None:
        """Shuffle a fresh deck and draw the opening hand."""

        self.sync_with_equipment()
        self._combat_active = True
        self._deck[:] = self._all_cards
        self._hand.clear()
        self._discard.clear()
        self._shuffle(self._deck)
        self.start_player_turn()

    def end_battle(self) -> None:
        """Collapse all piles back into the out-of-combat loadout."""

        self._combat_active = False
        self._deck[:] = self._all_cards
        self._hand.clear()
        self._discard.clear()
        self.reset()

    def start_player_turn(self) -> None:
        """Restore mana and draw the player's configured hand size."""

        stats = getattr(self.scene, "player_stats", None)
        if stats is None:
            return
        max_mana = max(1, int(getattr(stats, "max_mana", 5)))
        setattr(stats, "max_mana", max_mana)
        setattr(stats, "mana", max_mana)
        draw_count = max(0, int(getattr(stats, "card_draw", 3)))
        self.draw_cards(draw_count)

    def finish_player_turn(self) -> None:
        """Discard the unplayed hand before the enemy acts."""

        for card in self._hand:
            card.reset_to_home()
        self._discard.extend(self._hand)
        self._hand.clear()

    def draw_cards(self, count: int) -> list[Card]:
        """Draw cards, recycling and shuffling discard only when needed."""

        drawn: list[Card] = []
        for _ in range(max(0, int(count))):
            if not self._deck:
                if not self._discard:
                    break
                self._deck.extend(self._discard)
                self._discard.clear()
                self._shuffle(self._deck)
            card = self._deck.pop()
            card.reset_to_home()
            self._hand.append(card)
            drawn.append(card)
        return drawn

    def reset(self) -> None:
        for card in self._all_cards:
            card.reset_to_home()

    def can_play_card(self, card: Card) -> bool:
        stats = getattr(self.scene, "player_stats", None)
        spec = self._specs_by_card.get(card)
        if spec is None or stats is None:
            return False
        mana = int(getattr(stats, "mana", 0))
        return bool(
            self._combat_active
            and getattr(self.scene, "battle_mode", False)
            and card in self._hand
            and mana >= spec.mana_cost
            and (not spec.requires_odd_mana or mana % 2 == 1)
            and (spec.effect != "draw" or bool(self._deck))
        )

    def play_card(self, card: Card) -> bool:
        """Spend mana, discard the card, and resolve its player action."""

        if not self.can_play_card(card):
            return False

        spec = self._specs_by_card[card]
        stats = self.scene.player_stats
        stats.mana = max(0, int(stats.mana) - spec.mana_cost)
        card.reset_to_home()
        self._hand.remove(card)
        self._discard.append(card)

        self._resolve_effect(spec)

        should_end_turn = stats.mana <= 0 or self.all_cards_discarded
        if getattr(self.scene, "battle_mode", False) and should_end_turn:
            end_player_turn = getattr(self.scene, "end_player_turn", None)
            if callable(end_player_turn):
                end_player_turn()
        return True

    def _resolve_effect(self, spec: _CardSpec) -> None:
        if spec.effect == "damage":
            damage_battle_creature = getattr(
                self.scene,
                "damage_battle_creature",
                None,
            )
            if callable(damage_battle_creature):
                damage_battle_creature(spec.amount)
            return

        if spec.effect == "draw":
            self._draw_from_deck(spec.amount)
            return

        if spec.effect == "guard":
            combat = getattr(self.scene, "combat", None)
            gain_guard = getattr(combat, "gain_guard", None)
            if callable(gain_guard):
                gain_guard(spec.amount)

    def _draw_from_deck(self, count: int) -> list[Card]:
        """Draw without recycling, so Quickstep can never redraw itself."""

        drawn: list[Card] = []
        for _ in range(max(0, int(count))):
            if not self._deck:
                break
            card = self._deck.pop()
            card.reset_to_home()
            self._hand.append(card)
            drawn.append(card)
        return drawn

    def _build_card(self, spec: _CardSpec) -> Card:
        card = None

        def play(_scene) -> None:
            if card is not None:
                self.play_card(card)

        def can_play(_scene) -> bool:
            return card is not None and self.can_play_card(card)

        card = Card(
            spec.action,
            spec.title,
            spec.detail,
            play,
            footer=f"{spec.mana_cost} Mana",
            can_play=can_play,
        )
        return card

    def _shuffle(self, cards: list[Card]) -> None:
        rng = getattr(self.scene, "rng", None)
        shuffle = getattr(rng, "shuffle", None)
        if callable(shuffle):
            shuffle(cards)
        else:
            random.shuffle(cards)
