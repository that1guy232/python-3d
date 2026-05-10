"""Loading scene shown while the world scene initializes."""

from __future__ import annotations

import math
from typing import Iterable, Iterator

from OpenGL.GL import (
    glBegin,
    glClear,
    glClearColor,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glVertex2f,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FOG,
    GL_QUADS,
    GL_TEXTURE_2D,
)

from engine.config import HEIGHT, WIDTH
from engine.core.scene import Scene


def _draw_rect(x: float, y: float, w: float, h: float, color: tuple[float, ...]) -> None:
    glColor4f(*color)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


class LoadingScene(Scene):
    """Builds a target scene over several frames while drawing progress."""

    def __init__(
        self,
        target_scene,
        steps: Iterable[tuple[str, float]] | None = None,
        *,
        title: str = "Loading",
        initial_status: str = "Preparing",
    ) -> None:
        super().__init__()
        self.target_scene = target_scene
        if steps is None:
            steps = target_scene.initialize_steps()
        self._steps: Iterator[tuple[str, float]] = iter(steps)
        self.title = title
        self.status = initial_status
        self.progress = 0.0
        self.elapsed = 0.0
        self.next_scene = None
        self._target_transferred = False

    def update(self, dt: float) -> None:
        self.elapsed += dt
        if self.next_scene is not None:
            return

        try:
            label, progress = next(self._steps)
        except StopIteration:
            self.next_scene = self.target_scene
            self._target_transferred = True
            return

        self.status = label
        self.progress = max(0.0, min(1.0, float(progress)))

    def render(
        self, *, show_hud: bool = True, text=None, fps: float | None = None
    ) -> None:  # pragma: no cover - visual
        glDisable(GL_FOG)
        glDisable(GL_DEPTH_TEST)
        glClearColor(0.035, 0.043, 0.052, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if text is None:
            return

        progress = max(0.0, min(1.0, self.progress))
        pulse = 0.5 + 0.5 * math.sin(self.elapsed * 4.0)
        bar_w = min(640.0, WIDTH * 0.58)
        bar_h = 18.0
        bar_x = (WIDTH - bar_w) * 0.5
        bar_y = HEIGHT * 0.58
        fill_w = bar_w * progress

        text.begin()
        glDisable(GL_TEXTURE_2D)

        _draw_rect(0, 0, WIDTH, HEIGHT, (0.035, 0.043, 0.052, 1.0))
        _draw_rect(bar_x - 2, bar_y - 2, bar_w + 4, bar_h + 4, (0.02, 0.025, 0.03, 1.0))
        _draw_rect(bar_x, bar_y, bar_w, bar_h, (0.11, 0.13, 0.15, 1.0))
        if fill_w > 0:
            _draw_rect(bar_x, bar_y, fill_w, bar_h, (0.23, 0.56, 0.78, 1.0))
            _draw_rect(
                bar_x,
                bar_y,
                fill_w,
                max(2.0, bar_h * 0.28),
                (0.45, 0.76, 0.92, 0.35 + pulse * 0.25),
            )

        glEnable(GL_TEXTURE_2D)
        text.draw_text(
            self.title,
            WIDTH * 0.5,
            HEIGHT * 0.42,
            color=(242, 246, 248, 255),
            align="center",
        )
        text.draw_text(
            self.status,
            WIDTH * 0.5,
            bar_y - 34,
            color=(190, 205, 214, 255),
            key="loading_status",
            align="center",
        )
        text.draw_text(
            f"{int(progress * 100):d}%",
            WIDTH * 0.5,
            bar_y + 44,
            color=(220, 230, 235, 255),
            key="loading_percent",
            align="center",
        )
        text.end()

    def dispose(self) -> None:
        if self._target_transferred:
            return
        dispose = getattr(self.target_scene, "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                pass
