"""Battle UI drawing for the world HUD."""

from __future__ import annotations

from OpenGL.GL import (
    glBegin,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glGetDoublev,
    glGetIntegerv,
    glVertex2f,
    GL_MODELVIEW_MATRIX,
    GL_PROJECTION_MATRIX,
    GL_QUADS,
    GL_TEXTURE_2D,
    GL_VIEWPORT,
)
from OpenGL.GLU import gluProject

from game.config import HEIGHT, WIDTH
from game.world.ui.inventory_panel import InventoryPanel


class BattlePanel:
    """Draw battle-mode screen-space overlays."""

    def __init__(self, scene) -> None:
        self.scene = scene

    def active_goblin(self):
        goblin = getattr(self.scene, "active_battle_goblin", None)
        if goblin is None or not getattr(goblin, "enabled", True):
            return None
        return goblin

    @staticmethod
    def goblin_hp(goblin) -> tuple[int, int]:
        max_hp = max(1, int(getattr(goblin, "max_hp", 5)))
        hp = max(0, min(max_hp, int(getattr(goblin, "hp", max_hp))))
        return hp, max_hp

    def player_stat_lines(self) -> list[str]:
        rows = dict(InventoryPanel(self.scene)._player_stat_rows())
        if not rows:
            return []
        return [
            f"STR {rows['Strength']}  DEX {rows['Dexterity']}",
            f"Crit {rows['Crit Chance']}  Elem {rows['Elemental Damage']}",
            f"Draw {rows['Card Draw']}",
        ]

    def hp_anchor(self) -> tuple[float, float] | None:
        goblin = self.active_goblin()
        if goblin is None:
            return None

        position = getattr(goblin, "position", None)
        if position is None:
            return None

        try:
            sprite_height = max(
                1.0,
                float(
                    getattr(
                        goblin,
                        "_sprite_height",
                        getattr(goblin, "DEFAULT_HEIGHT", 42.0),
                    )
                ),
            )
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            win_x, win_y, win_z = gluProject(
                float(position.x),
                float(position.y) + sprite_height * 0.62,
                float(position.z),
                modelview,
                projection,
                viewport,
            )
        except Exception:
            return None

        if win_z < 0.0 or win_z > 1.0:
            return None

        viewport_x = float(viewport[0])
        viewport_y = float(viewport[1])
        viewport_h = float(viewport[3])
        screen_x = float(win_x) - viewport_x
        screen_y = viewport_h - (float(win_y) - viewport_y)
        return (
            max(0.0, min(float(WIDTH), screen_x)),
            max(0.0, min(float(HEIGHT), screen_y)),
        )

    @staticmethod
    def overlay_rect(
        x: float,
        y: float,
        w: float,
        h: float,
        color: tuple[float, float, float, float],
    ) -> None:
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    @staticmethod
    def hp_plate_rect(
        anchor: tuple[float, float],
        label_width: float,
        *,
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> tuple[float, float, float, float]:
        plate_w = max(112.0, float(label_width) + 28.0)
        plate_h = 42.0
        x = max(10.0, min(float(width) - plate_w - 10.0, anchor[0] - plate_w * 0.5))
        y = max(10.0, min(float(height) - plate_h - 10.0, anchor[1] - plate_h - 8.0))
        return x, y, plate_w, plate_h

    @staticmethod
    def notice_rect_avoiding(
        rect: tuple[float, float, float, float],
        avoid: tuple[float, float, float, float] | None,
        *,
        width: int = WIDTH,
        height: int = HEIGHT,
        gap: float = 8.0,
    ) -> tuple[float, float, float, float]:
        """Move a notice vertically around the enemy HP plate when needed."""

        if avoid is None:
            return rect

        x, y, rect_w, rect_h = rect
        avoid_x, avoid_y, avoid_w, avoid_h = avoid

        def overlaps(candidate_x: float, candidate_y: float) -> bool:
            return not (
                candidate_x + rect_w <= avoid_x
                or candidate_x >= avoid_x + avoid_w
                or candidate_y + rect_h <= avoid_y
                or candidate_y >= avoid_y + avoid_h
            )

        if not overlaps(x, y):
            return rect

        margin = 10.0
        candidates = (
            (x, avoid_y + avoid_h + gap),
            (x, avoid_y - rect_h - gap),
            (avoid_x + avoid_w + gap, y),
            (avoid_x - rect_w - gap, y),
        )
        for candidate_x, candidate_y in candidates:
            if (
                candidate_x >= margin
                and candidate_x + rect_w <= float(width) - margin
                and candidate_y >= margin
                and candidate_y + rect_h <= float(height) - margin
                and not overlaps(candidate_x, candidate_y)
            ):
                return candidate_x, candidate_y, rect_w, rect_h

        return rect

    def draw_hp_plate(
        self,
        text,
        anchor: tuple[float, float],
    ) -> tuple[float, float, float, float] | None:  # pragma: no cover - visual
        goblin = self.active_goblin()
        if goblin is None:
            return None

        hp, max_hp = self.goblin_hp(goblin)
        label = f"HP {hp}/{max_hp}"
        try:
            label_w = text.font.size(label)[0]
        except Exception:
            label_w = 72

        x, y, plate_w, plate_h = self.hp_plate_rect(anchor, label_w)
        ratio = hp / max_hp

        glDisable(GL_TEXTURE_2D)
        self.overlay_rect(x, y, plate_w, plate_h, (0.04, 0.035, 0.03, 0.9))
        self.overlay_rect(
            x + 4, y + 4, plate_w - 8, plate_h - 8, (0.12, 0.08, 0.06, 0.86)
        )
        bar_x = x + 10.0
        bar_y = y + plate_h - 14.0
        bar_w = plate_w - 20.0
        self.overlay_rect(bar_x, bar_y, bar_w, 7.0, (0.18, 0.03, 0.03, 0.95))
        self.overlay_rect(
            bar_x,
            bar_y,
            bar_w * ratio,
            7.0,
            (0.86, 0.14, 0.08, 0.96),
        )
        glEnable(GL_TEXTURE_2D)
        text.draw_text(
            label,
            x + plate_w * 0.5,
            y + 16.0,
            color=(255, 245, 230, 255),
            align="center",
        )
        return x, y, plate_w, plate_h

    def draw_player_stats(self, text) -> None:  # pragma: no cover - visual
        lines = self.player_stat_lines()
        if not lines:
            return

        try:
            max_label_w = max(text.font.size(line)[0] for line in lines)
            line_h = max(16, text.font.get_height())
        except Exception:
            max_label_w = 142
            line_h = 18

        panel_w = max(150.0, float(max_label_w) + 24.0)
        panel_h = 18.0 + float(len(lines) * line_h)
        x = 14.0
        y = 14.0

        glDisable(GL_TEXTURE_2D)
        self.overlay_rect(x, y, panel_w, panel_h, (0.035, 0.035, 0.04, 0.9))
        self.overlay_rect(
            x + 4, y + 4, panel_w - 8, panel_h - 8, (0.08, 0.075, 0.07, 0.86)
        )
        glEnable(GL_TEXTURE_2D)

        for index, line in enumerate(lines):
            text.draw_text(
                line,
                x + 12.0,
                y + 9.0 + index * line_h,
                color=(235, 240, 250, 255),
                align="topleft",
            )

    def draw_combat_notice(
        self,
        text,
        avoid_rect: tuple[float, float, float, float] | None = None,
    ) -> bool:  # pragma: no cover - visual
        combat = getattr(self.scene, "combat", None)
        active_notice = getattr(combat, "active_combat_notice", None)
        if not callable(active_notice):
            return False

        notice = active_notice()
        if not notice:
            return False

        try:
            label_w = text.font.size(notice)[0]
        except Exception:
            label_w = 210
        plate_w = max(240.0, float(label_w) + 36.0)
        plate_h = 38.0
        x = float(WIDTH) * 0.5 - plate_w * 0.5
        y = 52.0
        x, y, plate_w, plate_h = self.notice_rect_avoiding(
            (x, y, plate_w, plate_h),
            avoid_rect,
        )

        glDisable(GL_TEXTURE_2D)
        self.overlay_rect(x, y, plate_w, plate_h, (0.04, 0.025, 0.025, 0.92))
        self.overlay_rect(
            x + 4.0,
            y + 4.0,
            plate_w - 8.0,
            plate_h - 8.0,
            (0.42, 0.06, 0.04, 0.88),
        )
        glEnable(GL_TEXTURE_2D)
        text.draw_text(
            notice,
            x + plate_w * 0.5,
            y + plate_h * 0.5,
            color=(255, 236, 218, 255),
            align="center",
        )
        return True

    def draw_goblin_intent(self, text) -> None:  # pragma: no cover - visual
        combat = getattr(self.scene, "combat", None)
        intent_text = getattr(combat, "goblin_intent_text", None)
        if not callable(intent_text):
            return

        intent = intent_text()
        if not intent:
            return

        try:
            label_w = text.font.size(intent)[0]
        except Exception:
            label_w = 250
        plate_w = max(280.0, float(label_w) + 36.0)
        plate_h = 38.0
        x = float(WIDTH) * 0.5 - plate_w * 0.5
        y = 52.0

        glDisable(GL_TEXTURE_2D)
        self.overlay_rect(x, y, plate_w, plate_h, (0.035, 0.03, 0.02, 0.94))
        self.overlay_rect(
            x + 4.0,
            y + 4.0,
            plate_w - 8.0,
            plate_h - 8.0,
            (0.48, 0.27, 0.04, 0.9),
        )
        glEnable(GL_TEXTURE_2D)
        text.draw_text(
            intent,
            float(WIDTH) * 0.5,
            y + plate_h * 0.5,
            color=(255, 244, 205, 255),
            align="center",
        )

    def draw(self, text) -> None:  # pragma: no cover - visual
        hp_anchor = self.hp_anchor()
        text.begin()
        glDisable(GL_TEXTURE_2D)

        hp_rect = None
        if hp_anchor is not None:
            hp_rect = self.draw_hp_plate(text, hp_anchor)
        self.draw_player_stats(text)
        if not self.draw_combat_notice(text, hp_rect):
            self.draw_goblin_intent(text)
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        if battle_overlay is not None:
            battle_overlay.draw(text)

        glEnable(GL_TEXTURE_2D)
        text.draw_text(
            "Battle mode",
            WIDTH // 2,
            24,
            color=(255, 245, 230, 255),
            align="center",
        )

        text.end()
