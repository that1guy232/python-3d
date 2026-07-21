"""Card deck, hand, discard, and turn actions for battle mode."""

from __future__ import annotations

import random

from game.world.inventory import (
    GOBLIN_FISTS_NAME,
    GOBLIN_FISTS_STRIKE_CARD_BONUS,
    ItemType,
    equipped_item,
)
from game.world.ui.card import Card


class BattleCardLoadout:
    """Own the player's battle deck and its equipment-driven cards."""

    BASE_STRIKE_COUNT = 1
    STRIKE_MANA_COST = 1

    def __init__(self, scene) -> None:
        self.scene = scene
        self._all_cards: list[Card] = []
        self._deck: list[Card] = []
        self._hand: list[Card] = []
        self._discard: list[Card] = []
        self._combat_active = False
        self.sync_with_equipment()

    @property
    def cards(self) -> list[Card]:
        """Return the current hand, or the full loadout outside combat."""

        self.sync_with_equipment()
        return self._hand if self._combat_active else self._all_cards

    @property
    def deck_count(self) -> int:
        return len(self._deck)

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

    @staticmethod
    def _item_name(item) -> str:
        if isinstance(item, dict):
            return str(item.get("name", "") or "")
        return str(getattr(item, "name", "") or "")

    def sync_with_equipment(self) -> None:
        """Keep the complete deck aligned with the equipped weapon."""

        weapon = equipped_item(self.scene, ItemType.WEAPON)
        bonus = (
            GOBLIN_FISTS_STRIKE_CARD_BONUS
            if self._item_name(weapon) == GOBLIN_FISTS_NAME
            else 0
        )
        desired_count = self.BASE_STRIKE_COUNT + bonus
        while len(self._all_cards) < desired_count:
            card = self._build_strike(len(self._all_cards))
            self._all_cards.append(card)
            self._deck.append(card)

        if len(self._all_cards) > desired_count:
            removed = set(self._all_cards[desired_count:])
            del self._all_cards[desired_count:]
            self._deck[:] = [card for card in self._deck if card not in removed]
            self._hand[:] = [card for card in self._hand if card not in removed]
            self._discard[:] = [card for card in self._discard if card not in removed]

        if not self._combat_active:
            self._deck[:] = self._all_cards
            self._hand.clear()
            self._discard.clear()

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
        return bool(
            self._combat_active
            and getattr(self.scene, "battle_mode", False)
            and card in self._hand
            and stats is not None
            and int(getattr(stats, "mana", 0)) >= self.STRIKE_MANA_COST
        )

    def play_card(self, card: Card) -> bool:
        """Spend mana, discard the card, and resolve its player action."""

        if not self.can_play_card(card):
            return False

        stats = self.scene.player_stats
        stats.mana = max(0, int(stats.mana) - self.STRIKE_MANA_COST)
        card.reset_to_home()
        self._hand.remove(card)
        self._discard.append(card)

        damage_battle_creature = getattr(self.scene, "damage_battle_creature", None)
        if callable(damage_battle_creature):
            damage_battle_creature(1)

        should_end_turn = stats.mana <= 0 or self.all_cards_discarded
        if getattr(self.scene, "battle_mode", False) and should_end_turn:
            end_player_turn = getattr(self.scene, "end_player_turn", None)
            if callable(end_player_turn):
                end_player_turn()
        return True

    def _build_strike(self, index: int) -> Card:
        card = None

        def play(_scene) -> None:
            if card is not None:
                self.play_card(card)

        def can_play(_scene) -> bool:
            return card is not None and self.can_play_card(card)

        card = Card(
            f"strike_{index + 1}",
            "Strike",
            "1 Damage",
            play,
            footer=f"{self.STRIKE_MANA_COST} Mana",
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
