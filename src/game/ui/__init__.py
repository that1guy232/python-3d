"""Lazy exports for UI used by world, inventory, and combat scenes."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BattleResourceOverlay": (".battle_overlay", "BattleResourceOverlay"),
    "BattlePanel": (".battle_panel", "BattlePanel"),
    "Card": (".card", "Card"),
    "CompassOverlay": (".compass_overlay", "CompassOverlay"),
    "InventoryPanel": (".inventory_panel", "InventoryPanel"),
    "MiniMapOverlay": (".minimap_overlay", "MiniMapOverlay"),
    "PauseMenuPanel": (".pause_panel", "PauseMenuPanel"),
    "PauseMenu": (".pause_menu", "PauseMenu"),
    "SettingMenu": (".setting_menu", "SettingMenu"),
    "WorldHUD": (".world_hud", "WorldHUD"),
    "WorldUIInteractions": (".interactions", "WorldUIInteractions"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
