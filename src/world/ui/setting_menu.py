"""World settings-menu options and actions."""

from __future__ import annotations

import math

from config import HEIGHT, WIDTH
from sound.sound_utils import Sounds
from ui.menu import ButtonMenu, MenuItem, MenuOption, SliderOption


class SettingMenu(ButtonMenu):
    button_height = 44
    button_spacing = 8
    button_width_max = 560
    page_size = 7

    @property
    def title(self) -> str:
        return f"Settings {self.page + 1}/{self.page_count()}"

    def __init__(self, scene) -> None:
        super().__init__(scene)
        self.page = 0
        self.settings = [
            MenuOption("toggle_audio", self.audio_label, self.toggle_audio),
            MenuOption("toggle_hud", self.hud_label, self.toggle_hud),
            MenuOption("toggle_compass", self.compass_label, self.toggle_compass),
            MenuOption("toggle_held_item", self.held_item_label, self.toggle_held_item),
            MenuOption("toggle_test_light", self.test_light_label, self.toggle_test_light),
            MenuOption("toggle_debug_text", self.debug_text_label, self.toggle_debug_text),
            MenuOption("toggle_fog", self.fog_label, self.toggle_fog),
            self._scene_slider(
                "set_fog_density",
                "Fog Density",
                "fog_density",
                0.0,
                0.002,
                0.0005,
                step=0.0001,
                formatter=lambda value: f"{value:.4f}",
            ),
            self._scene_slider(
                "set_fov",
                "FOV",
                "fov",
                70,
                110,
                90,
                step=1,
                formatter=lambda value: self._format_int(value),
                on_change=self._set_camera_fov,
            ),
            self._object_slider(
                "set_brightness",
                "Brightness",
                self._camera,
                "brightness_default",
                0.4,
                1.2,
                0.8,
                step=0.05,
                formatter=lambda value: self._format_float(value, 2),
            ),
            self._scene_slider(
                "set_mouse_sensitivity",
                "Mouse Sens",
                "mouse_sensitivity",
                0.0008,
                0.003,
                0.0015,
                step=0.0001,
                formatter=lambda value: f"{value:.4f}",
            ),
            self._object_slider(
                "set_look_smooth",
                "Look Smooth",
                self._controller,
                "rot_smooth_hz",
                0.0,
                12.0,
                4.0,
                step=0.25,
                formatter=lambda value: self._format_float(value, 2),
            ),
            self._scene_slider(
                "set_walk_speed",
                "Walk Speed",
                "walk_speed",
                80,
                220,
                120,
                step=5,
                formatter=lambda value: self._format_int(value),
            ),
            self._scene_slider(
                "set_sprint_speed",
                "Sprint Speed",
                "sprint_speed",
                120,
                320,
                180,
                step=5,
                formatter=lambda value: self._format_int(value),
            ),
            self._scene_slider(
                "set_road_boost",
                "Road Boost",
                "road_speed_multiplier",
                1.0,
                2.0,
                1.5,
                step=0.05,
                formatter=lambda value: f"{value:.2f}x",
            ),
            self._scene_slider(
                "set_jump_speed",
                "Jump Speed",
                "jump_speed",
                150,
                500,
                250,
                step=10,
                formatter=lambda value: self._format_int(value),
            ),
            self._scene_slider(
                "set_gravity",
                "Gravity",
                "gravity",
                400,
                1800,
                800,
                step=25,
                formatter=lambda value: self._format_int(value),
            ),
            self._scene_slider(
                "set_follow_smooth",
                "Ground Smooth",
                "camera_follow_smooth_hz",
                0.0,
                10.0,
                5.0,
                step=0.25,
                formatter=lambda value: self._format_float(value, 2),
            ),
            self._object_slider(
                "set_eye_height",
                "Eye Offset",
                self._camera,
                "manual_height_offset",
                -25,
                100,
                0,
                step=5,
                formatter=lambda value: self._format_int(value),
            ),
            self._object_slider(
                "set_height_speed",
                "Height Speed",
                self._camera,
                "height_adjust_speed",
                25,
                200,
                50,
                step=5,
                formatter=lambda value: self._format_int(value),
            ),
            MenuOption("toggle_headbob", self.headbob_label, self.toggle_headbob),
            self._object_slider(
                "set_headbob_speed",
                "Bob Speed",
                self._headbob,
                "frequency",
                0.25,
                1.5,
                0.5,
                step=0.05,
                formatter=lambda value: self._format_float(value, 2),
            ),
            self._object_slider(
                "set_headbob_amount",
                "Bob Amount",
                self._headbob,
                "amplitude_y",
                0.0,
                8.0,
                4.0,
                step=0.25,
                formatter=lambda value: self._format_float(value, 2),
            ),
            self._object_slider(
                "set_headbob_side",
                "Bob Side",
                self._headbob,
                "amplitude_x",
                0.0,
                5.0,
                3.0,
                step=0.25,
                formatter=lambda value: self._format_float(value, 2),
            ),
            self._object_slider(
                "set_headbob_damping",
                "Bob Damping",
                self._headbob,
                "damping",
                1.0,
                8.0,
                2.0,
                step=0.25,
                formatter=lambda value: self._format_float(value, 2),
            ),
            MenuOption("toggle_idle_sway", self.idle_sway_label, self.toggle_idle_sway),
            self._object_slider(
                "set_idle_delay",
                "Idle Delay",
                self._headbob,
                "_idle_threshold",
                0.0,
                4.0,
                1.0,
                step=0.1,
                formatter=lambda value: f"{value:.1f}s",
            ),
            self._object_slider(
                "set_idle_amount",
                "Idle Amount",
                self._headbob,
                "_idle_amplitude",
                0.0,
                1.2,
                0.35,
                step=0.05,
                formatter=lambda value: self._format_float(value, 2),
            ),
            self._object_slider(
                "set_breath_amount",
                "Breath Amount",
                self._headbob,
                "_idle_breath_amplitude",
                0.0,
                2.0,
                1.0,
                step=0.05,
                formatter=lambda value: self._format_float(value, 2),
            ),
            MenuOption("toggle_mouse_sway", self.mouse_sway_label, self.toggle_mouse_sway),
            self._object_slider(
                "set_sway_scale",
                "Sway Scale",
                self._sway,
                "mouse_scale",
                0.0,
                0.04,
                0.01,
                step=0.001,
                formatter=lambda value: self._format_float(value, 3),
            ),
            MenuOption("cycle_sway_limit", self.sway_limit_label, self.cycle_sway_limit),
            MenuOption("reset_motion", "Reset Motion Values", self.reset_motion_values),
        ]
        self.buttons = [
            MenuOption("previous_page", "Previous", self.previous_page),
            MenuOption("next_page", "Next", self.next_page),
            MenuOption("back", "Back", self.back),
        ]

    def options(self) -> list[MenuItem]:
        start = self.page * self.page_size
        end = start + self.page_size
        return [*self.settings[start:end], *self.page_buttons()]

    def compute_buttons(self, width: int = WIDTH, height: int = HEIGHT) -> list[dict]:
        self.page = max(0, min(self.page, self.page_count() - 1))
        return super().compute_buttons(width=width, height=height)

    def page_count(self) -> int:
        return max(1, math.ceil(len(self.settings) / self.page_size))

    def page_buttons(self) -> list[MenuOption]:
        buttons = []
        if self.page_count() > 1:
            if self.page > 0:
                buttons.append(self.buttons[0])
            if self.page < self.page_count() - 1:
                buttons.append(self.buttons[1])
        buttons.append(self.buttons[2])
        return buttons

    def next_page(self, scene) -> None:
        self.page = min(self.page + 1, self.page_count() - 1)

    def previous_page(self, scene) -> None:
        self.page = max(0, self.page - 1)

    def _scene_slider(
        self,
        action: str,
        label: str,
        attr: str,
        minimum: float,
        maximum: float,
        default: float,
        *,
        step: float | None = None,
        formatter=None,
        on_change=None,
    ) -> SliderOption:
        def getter(scene) -> float:
            return float(getattr(scene, attr, default))

        def setter(scene, value: float) -> None:
            setattr(scene, attr, value)
            if on_change is not None:
                on_change(scene, value)

        return SliderOption(
            action,
            label,
            minimum,
            maximum,
            getter,
            setter,
            step=step,
            formatter=formatter,
        )

    def _object_slider(
        self,
        action: str,
        label: str,
        object_getter,
        attr: str,
        minimum: float,
        maximum: float,
        default: float,
        *,
        step: float | None = None,
        formatter=None,
        on_change=None,
    ) -> SliderOption:
        def getter(scene) -> float:
            obj = object_getter(scene)
            if obj is None:
                return float(default)
            return float(getattr(obj, attr, default))

        def setter(scene, value: float) -> None:
            obj = object_getter(scene)
            if obj is None:
                return
            setattr(obj, attr, value)
            if on_change is not None:
                on_change(scene, value)

        return SliderOption(
            action,
            label,
            minimum,
            maximum,
            getter,
            setter,
            step=step,
            formatter=formatter,
        )

    @staticmethod
    def _bool_label(name: str, value: bool) -> str:
        return f"{name}: {'On' if value else 'Off'}"

    @staticmethod
    def _get_scene_value(scene, name: str, default):
        return getattr(scene, name, default)

    @staticmethod
    def _set_scene_value(scene, name: str, value) -> None:
        setattr(scene, name, value)

    @staticmethod
    def _cycle_value(current, values):
        if not values:
            return current
        try:
            current_f = float(current)
            index = min(range(len(values)), key=lambda i: abs(float(values[i]) - current_f))
        except Exception:
            try:
                index = values.index(current)
            except ValueError:
                index = -1
        return values[(index + 1) % len(values)]

    @classmethod
    def _cycle_scene_value(cls, scene, name: str, values, default=None):
        current = getattr(scene, name, values[0] if default is None else default)
        value = cls._cycle_value(current, values)
        setattr(scene, name, value)
        return value

    @classmethod
    def _cycle_object_value(cls, obj, name: str, values, default=None):
        if obj is None:
            return None
        current = getattr(obj, name, values[0] if default is None else default)
        value = cls._cycle_value(current, values)
        setattr(obj, name, value)
        return value

    @staticmethod
    def _toggle_scene_flag(scene, name: str, default: bool = True) -> bool:
        value = not getattr(scene, name, default)
        setattr(scene, name, value)
        return value

    @staticmethod
    def _camera(scene):
        return getattr(scene, "camera", None)

    @staticmethod
    def _headbob(scene):
        return getattr(scene, "_headbob", None)

    @staticmethod
    def _sway(scene):
        return getattr(scene, "_sway_controller", None)

    @staticmethod
    def _controller(scene):
        return getattr(scene, "_camera_controller", None)

    @staticmethod
    def _format_float(value, digits: int = 2) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value)

    @staticmethod
    def _format_int(value) -> str:
        try:
            return str(int(round(float(value))))
        except Exception:
            return str(value)

    @staticmethod
    def _set_camera_fov(scene, fov: float) -> None:
        camera = getattr(scene, "camera", None)
        if camera is None:
            return
        try:
            camera._fov_scale = (HEIGHT * 0.5) / math.tan(math.radians(float(fov) * 0.5))
        except Exception:
            pass

    def audio_label(self, scene) -> str:
        muted = Sounds.is_muted() if hasattr(Sounds, "is_muted") else False
        return f"Audio: {'Muted' if muted else 'On'}"

    def toggle_audio(self, scene) -> None:
        if hasattr(Sounds, "toggle_muted"):
            Sounds.toggle_muted()

    def hud_label(self, scene) -> str:
        return self._bool_label("HUD", getattr(scene, "hud_visible", True))

    def toggle_hud(self, scene) -> None:
        self._toggle_scene_flag(scene, "hud_visible", True)

    def compass_label(self, scene) -> str:
        return self._bool_label("Compass", getattr(scene, "compass_visible", True))

    def toggle_compass(self, scene) -> None:
        self._toggle_scene_flag(scene, "compass_visible", True)

    def held_item_label(self, scene) -> str:
        return self._bool_label("Held Item", getattr(scene, "held_item_visible", True))

    def toggle_held_item(self, scene) -> None:
        self._toggle_scene_flag(scene, "held_item_visible", True)

    def test_light_label(self, scene) -> str:
        return self._bool_label("Test Light", getattr(scene, "test_light_visible", True))

    def toggle_test_light(self, scene) -> None:
        self._toggle_scene_flag(scene, "test_light_visible", True)

    def debug_text_label(self, scene) -> str:
        return self._bool_label("Debug Text", getattr(scene, "debug_text_visible", True))

    def toggle_debug_text(self, scene) -> None:
        self._toggle_scene_flag(scene, "debug_text_visible", True)

    def fog_label(self, scene) -> str:
        return self._bool_label("Fog", getattr(scene, "fog_enabled", True))

    def toggle_fog(self, scene) -> None:
        self._toggle_scene_flag(scene, "fog_enabled", True)

    def fog_density_label(self, scene) -> str:
        value = getattr(scene, "fog_density", 0.0005)
        return f"Fog Density: {float(value):.4f}"

    def cycle_fog_density(self, scene) -> None:
        self._cycle_scene_value(scene, "fog_density", [0.0, 0.0002, 0.0005, 0.001, 0.002])

    def fov_label(self, scene) -> str:
        return f"FOV: {self._format_int(getattr(scene, 'fov', 90))}"

    def cycle_fov(self, scene) -> None:
        fov = self._cycle_scene_value(scene, "fov", [70, 80, 90, 100, 110], 90)
        self._set_camera_fov(scene, fov)

    def brightness_label(self, scene) -> str:
        camera = self._camera(scene)
        value = getattr(camera, "brightness_default", 0.8)
        return f"Brightness: {self._format_float(value, 1)}"

    def cycle_brightness(self, scene) -> None:
        camera = self._camera(scene)
        self._cycle_object_value(camera, "brightness_default", [0.4, 0.6, 0.8, 1.0, 1.2], 0.8)

    def mouse_sensitivity_label(self, scene) -> str:
        value = getattr(scene, "mouse_sensitivity", 0.0015)
        return f"Mouse Sens: {float(value):.4f}"

    def cycle_mouse_sensitivity(self, scene) -> None:
        self._cycle_scene_value(
            scene,
            "mouse_sensitivity",
            [0.0008, 0.0012, 0.0015, 0.002, 0.003],
            0.0015,
        )

    def look_smooth_label(self, scene) -> str:
        controller = self._controller(scene)
        value = getattr(controller, "rot_smooth_hz", 4.0)
        return f"Look Smooth: {self._format_float(value, 1)}"

    def cycle_look_smooth(self, scene) -> None:
        controller = self._controller(scene)
        self._cycle_object_value(controller, "rot_smooth_hz", [0.0, 2.0, 4.0, 8.0, 12.0], 4.0)

    def walk_speed_label(self, scene) -> str:
        return f"Walk Speed: {self._format_int(getattr(scene, 'walk_speed', 120))}"

    def cycle_walk_speed(self, scene) -> None:
        self._cycle_scene_value(scene, "walk_speed", [80, 120, 160, 220], 120)

    def sprint_speed_label(self, scene) -> str:
        return f"Sprint Speed: {self._format_int(getattr(scene, 'sprint_speed', 180))}"

    def cycle_sprint_speed(self, scene) -> None:
        self._cycle_scene_value(scene, "sprint_speed", [120, 180, 240, 320], 180)

    def road_boost_label(self, scene) -> str:
        return f"Road Boost: {self._format_float(getattr(scene, 'road_speed_multiplier', 1.5), 1)}x"

    def cycle_road_boost(self, scene) -> None:
        self._cycle_scene_value(scene, "road_speed_multiplier", [1.0, 1.25, 1.5, 2.0], 1.5)

    def jump_speed_label(self, scene) -> str:
        return f"Jump Speed: {self._format_int(getattr(scene, 'jump_speed', 250))}"

    def cycle_jump_speed(self, scene) -> None:
        self._cycle_scene_value(scene, "jump_speed", [150, 250, 350, 500], 250)

    def gravity_label(self, scene) -> str:
        return f"Gravity: {self._format_int(getattr(scene, 'gravity', 800))}"

    def cycle_gravity(self, scene) -> None:
        self._cycle_scene_value(scene, "gravity", [400, 800, 1200, 1800], 800)

    def follow_smooth_label(self, scene) -> str:
        value = getattr(scene, "camera_follow_smooth_hz", 5.0)
        return f"Ground Smooth: {self._format_float(value, 1)}"

    def cycle_follow_smooth(self, scene) -> None:
        self._cycle_scene_value(scene, "camera_follow_smooth_hz", [0.0, 2.0, 5.0, 10.0], 5.0)

    def eye_height_label(self, scene) -> str:
        camera = self._camera(scene)
        value = getattr(camera, "manual_height_offset", 0.0)
        return f"Eye Offset: {self._format_int(value)}"

    def cycle_eye_height(self, scene) -> None:
        camera = self._camera(scene)
        self._cycle_object_value(camera, "manual_height_offset", [-25, 0, 25, 50, 100], 0.0)

    def height_speed_label(self, scene) -> str:
        camera = self._camera(scene)
        value = getattr(camera, "height_adjust_speed", 50.0)
        return f"Height Speed: {self._format_int(value)}"

    def cycle_height_speed(self, scene) -> None:
        camera = self._camera(scene)
        self._cycle_object_value(camera, "height_adjust_speed", [25, 50, 100, 200], 50.0)

    @staticmethod
    def headbob_label(scene) -> str:
        headbob = getattr(scene, "_headbob", None)
        enabled = getattr(headbob, "enabled", True)
        return f"Headbob: {'On' if enabled else 'Off'}"

    def toggle_headbob(self, scene) -> None:
        headbob = getattr(scene, "_headbob", None)
        if headbob is not None:
            headbob.enabled = not getattr(headbob, "enabled", True)

    def headbob_speed_label(self, scene) -> str:
        headbob = self._headbob(scene)
        value = getattr(headbob, "frequency", 0.5)
        return f"Bob Speed: {self._format_float(value, 2)}"

    def cycle_headbob_speed(self, scene) -> None:
        headbob = self._headbob(scene)
        self._cycle_object_value(headbob, "frequency", [0.25, 0.5, 0.75, 1.0, 1.5], 0.5)

    def headbob_amount_label(self, scene) -> str:
        headbob = self._headbob(scene)
        value = getattr(headbob, "amplitude_y", 4.0)
        return f"Bob Amount: {self._format_float(value, 1)}"

    def cycle_headbob_amount(self, scene) -> None:
        headbob = self._headbob(scene)
        self._cycle_object_value(headbob, "amplitude_y", [0.0, 2.0, 4.0, 6.0, 8.0], 4.0)

    def headbob_side_label(self, scene) -> str:
        headbob = self._headbob(scene)
        value = getattr(headbob, "amplitude_x", 3.0)
        return f"Bob Side: {self._format_float(value, 1)}"

    def cycle_headbob_side(self, scene) -> None:
        headbob = self._headbob(scene)
        self._cycle_object_value(headbob, "amplitude_x", [0.0, 1.5, 3.0, 5.0], 3.0)

    def headbob_damping_label(self, scene) -> str:
        headbob = self._headbob(scene)
        value = getattr(headbob, "damping", 2.0)
        return f"Bob Damping: {self._format_float(value, 1)}"

    def cycle_headbob_damping(self, scene) -> None:
        headbob = self._headbob(scene)
        self._cycle_object_value(headbob, "damping", [1.0, 2.0, 4.0, 8.0], 2.0)

    def idle_sway_label(self, scene) -> str:
        headbob = self._headbob(scene)
        return self._bool_label("Idle Sway", getattr(headbob, "idle_enabled", True))

    def toggle_idle_sway(self, scene) -> None:
        headbob = self._headbob(scene)
        if headbob is not None:
            headbob.idle_enabled = not getattr(headbob, "idle_enabled", True)

    def idle_delay_label(self, scene) -> str:
        headbob = self._headbob(scene)
        value = getattr(headbob, "_idle_threshold", 1.0)
        return f"Idle Delay: {self._format_float(value, 1)}s"

    def cycle_idle_delay(self, scene) -> None:
        headbob = self._headbob(scene)
        self._cycle_object_value(headbob, "_idle_threshold", [0.0, 0.5, 1.0, 2.0, 4.0], 1.0)

    def idle_sway_amount_label(self, scene) -> str:
        headbob = self._headbob(scene)
        value = getattr(headbob, "_idle_amplitude", 0.35)
        return f"Idle Amount: {self._format_float(value, 2)}"

    def cycle_idle_sway_amount(self, scene) -> None:
        headbob = self._headbob(scene)
        self._cycle_object_value(headbob, "_idle_amplitude", [0.0, 0.2, 0.35, 0.7, 1.2], 0.35)

    def breath_amount_label(self, scene) -> str:
        headbob = self._headbob(scene)
        value = getattr(headbob, "_idle_breath_amplitude", 1.0)
        return f"Breath Amount: {self._format_float(value, 1)}"

    def cycle_breath_amount(self, scene) -> None:
        headbob = self._headbob(scene)
        self._cycle_object_value(headbob, "_idle_breath_amplitude", [0.0, 0.5, 1.0, 2.0], 1.0)

    def mouse_sway_label(self, scene) -> str:
        sway = self._sway(scene)
        return self._bool_label("Mouse Sway", getattr(sway, "enabled", True))

    def toggle_mouse_sway(self, scene) -> None:
        sway = self._sway(scene)
        if sway is not None:
            sway.enabled = not getattr(sway, "enabled", True)

    def sway_scale_label(self, scene) -> str:
        sway = self._sway(scene)
        value = getattr(sway, "mouse_scale", 0.01)
        return f"Sway Scale: {self._format_float(value, 3)}"

    def cycle_sway_scale(self, scene) -> None:
        sway = self._sway(scene)
        self._cycle_object_value(sway, "mouse_scale", [0.0, 0.005, 0.01, 0.02, 0.04], 0.01)

    def sway_limit_label(self, scene) -> str:
        sway = self._sway(scene)
        limit = getattr(sway, "max", None)
        x = getattr(limit, "x", 1.25)
        y = getattr(limit, "y", 0.75)
        return f"Sway Limit: {self._format_float(x, 1)}x{self._format_float(y, 1)}"

    def cycle_sway_limit(self, scene) -> None:
        sway = self._sway(scene)
        if sway is None:
            return
        x, y = self._cycle_value((getattr(sway.max, "x", 1.25), getattr(sway.max, "y", 0.75)), [
            (0.0, 0.0),
            (0.75, 0.45),
            (1.25, 0.75),
            (2.0, 1.2),
        ])
        sway.max.x = x
        sway.max.y = y

    def reset_motion_values(self, scene) -> None:
        camera = self._camera(scene)
        if camera is not None:
            camera.manual_height_offset = 0.0
            camera.height_adjust_speed = 50.0
        controller = self._controller(scene)
        if controller is not None:
            controller.rot_smooth_hz = 4.0
        headbob = self._headbob(scene)
        if headbob is not None:
            headbob.enabled = True
            headbob.frequency = 0.5
            headbob.amplitude_y = 4.0
            headbob.amplitude_x = 3.0
            headbob.damping = 2.0
            headbob.idle_enabled = True
            headbob._idle_threshold = 1.0
            headbob._idle_amplitude = 0.35
            headbob._idle_breath_amplitude = 1.0
        sway = self._sway(scene)
        if sway is not None:
            sway.enabled = True
            sway.mouse_scale = 0.01
            sway.max.x = 1.25
            sway.max.y = 0.75
        scene.mouse_sensitivity = 0.0015
        scene.walk_speed = 120
        scene.sprint_speed = 180.0
        scene.road_speed_multiplier = 1.5
        scene.jump_speed = 250.0
        scene.gravity = 800.0
        scene.camera_follow_smooth_hz = 5.0

    def back(self, scene) -> None:
        scene.showing_settings_menu = False
