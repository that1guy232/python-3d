"""Shared button-menu helpers for simple in-game overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from config import HEIGHT, WIDTH


LabelProvider = str | Callable[[object], str]
MenuHandler = Callable[[object], None]
SliderGetter = Callable[[object], float]
SliderSetter = Callable[[object, float], None]
SliderFormatter = Callable[[float], str]


@dataclass(frozen=True)
class MenuOption:
    kind = "button"

    action: str
    label: LabelProvider
    on_click: MenuHandler

    def label_for(self, scene) -> str:
        if callable(self.label):
            return self.label(scene)
        return self.label


@dataclass(frozen=True)
class SliderOption:
    kind = "slider"

    action: str
    label: LabelProvider
    minimum: float
    maximum: float
    getter: SliderGetter
    setter: SliderSetter
    step: float | None = None
    formatter: SliderFormatter | None = None

    def label_for(self, scene) -> str:
        if callable(self.label):
            return self.label(scene)
        return self.label

    def value_for(self, scene) -> float:
        try:
            return float(self.getter(scene))
        except Exception:
            return float(self.minimum)

    def value_text_for(self, scene) -> str:
        value = self.value_for(scene)
        if self.formatter is not None:
            return self.formatter(value)
        if self.step is not None and self.step >= 1.0:
            return str(int(round(value)))
        return f"{value:.2f}"

    def ratio_for(self, scene) -> float:
        span = float(self.maximum) - float(self.minimum)
        if span <= 0:
            return 0.0
        ratio = (self.value_for(scene) - float(self.minimum)) / span
        return max(0.0, min(1.0, ratio))

    def set_from_ratio(self, scene, ratio: float) -> None:
        ratio = max(0.0, min(1.0, float(ratio)))
        value = float(self.minimum) + (float(self.maximum) - float(self.minimum)) * ratio
        if self.step:
            value = round(value / self.step) * self.step
        value = max(float(self.minimum), min(float(self.maximum), value))
        self.setter(scene, value)


MenuItem = MenuOption | SliderOption


class ButtonMenu:
    title: str | None = None
    button_height = 48
    button_spacing = 14
    button_width_max = 360
    button_width_divisor = 3
    slider_horizontal_padding = 18

    def __init__(self, scene) -> None:
        self.scene = scene
        self._active_slider_action: str | None = None

    def options(self) -> list[MenuItem]:
        return []

    def compute_buttons(self, width: int = WIDTH, height: int = HEIGHT) -> list[dict]:
        options = self.options()
        center_x = width // 2
        center_y = height // 2
        button_w = min(self.button_width_max, width // self.button_width_divisor)
        button_h = self.button_height
        spacing = self.button_spacing

        total_h = len(options) * button_h + max(0, len(options) - 1) * spacing
        top = center_y - total_h // 2
        buttons = []
        for index, option in enumerate(options):
            x = center_x - button_w // 2
            y = top + index * (button_h + spacing)
            buttons.append(
                {
                    "type": getattr(option, "kind", "button"),
                    "label": option.label_for(self.scene),
                    "rect": (x, y, button_w, button_h),
                    "action": option.action,
                }
            )
            if getattr(option, "kind", "button") == "slider":
                buttons[-1]["value"] = option.value_for(self.scene)
                buttons[-1]["value_text"] = option.value_text_for(self.scene)
                buttons[-1]["ratio"] = option.ratio_for(self.scene)
        return buttons

    def handle_click(self, pos) -> None:
        mx, my = pos
        for button in self.compute_buttons():
            x, y, w, h = button["rect"]
            if x <= mx <= x + w and y <= my <= y + h:
                if button.get("type") == "slider":
                    self._active_slider_action = button["action"]
                    self.handle_slider_drag(pos)
                else:
                    self._active_slider_action = None
                    self.handle_action(button["action"])
                return
        self._active_slider_action = None

    def handle_motion(self, pos) -> None:
        if self._active_slider_action is not None:
            self.handle_slider_drag(pos)

    def handle_release(self, pos) -> None:
        if self._active_slider_action is not None:
            self.handle_slider_drag(pos)
        self._active_slider_action = None

    def handle_action(self, action: str) -> None:
        for option in self.options():
            if option.action == action and getattr(option, "kind", "button") == "button":
                option.on_click(self.scene)
                return

    def handle_slider_drag(self, pos) -> None:
        if self._active_slider_action is None:
            return
        for button in self.compute_buttons():
            if button["action"] != self._active_slider_action:
                continue
            x, _y, w, _h = button["rect"]
            slider_x = x + self.slider_horizontal_padding
            slider_w = max(1, w - (self.slider_horizontal_padding * 2))
            ratio = (pos[0] - slider_x) / slider_w
            self.handle_slider_value(button["action"], ratio)
            return

    def handle_slider_value(self, action: str, ratio: float) -> None:
        for option in self.options():
            if option.action == action and getattr(option, "kind", "button") == "slider":
                option.set_from_ratio(self.scene, ratio)
                return
