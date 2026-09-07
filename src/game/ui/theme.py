"""Crisp parchment and brass UI, drawn in the same pixel style as the compass.

These primitives run inside an existing screen-space pass with textures disabled.
They create no textures or per-frame resources.
"""

from __future__ import annotations

from OpenGL.GL import glBegin, glColor4f, glEnd, glVertex2f, GL_QUADS


def _rgba(r: int, g: int, b: int, a: float = 1.0) -> tuple[float, ...]:
    return r / 255.0, g / 255.0, b / 255.0, a


PAPER = _rgba(239, 219, 171)
PAPER_LIGHT = _rgba(255, 239, 197)
PAPER_DARK = _rgba(215, 192, 143)
INK = _rgba(67, 53, 35)
GOLD = _rgba(204, 146, 47)
GOLD_LIGHT = _rgba(255, 207, 99)
GOLD_DARK = _rgba(125, 83, 32)
FOREST = _rgba(49, 80, 51)
FOREST_LIGHT = _rgba(72, 109, 62)
BLUE = _rgba(53, 109, 155)
RED = _rgba(164, 67, 43)
INK_TEXT = (67, 53, 35)
MUTED_TEXT = (113, 94, 61)
CREAM_TEXT = (255, 239, 197)


def draw_rect(rect, color) -> None:
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return
    glColor4f(*color)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


def _stepped_rect(rect, color, step: float) -> None:
    x, y, w, h = rect
    draw_rect((x + step, y, w - step * 2, h), color)
    draw_rect((x, y + step, step, h - step * 2), color)
    draw_rect((x + w - step, y + step, step, h - step * 2), color)


def draw_frame(
    rect, *, fill=PAPER, border=GOLD_DARK, light=GOLD_LIGHT,
    shadow: bool = True, inset: float = 4.0,
) -> None:
    """A stepped outer silhouette with a small, hard bevel."""
    x, y, w, h = rect
    p = min(inset, w / 8, h / 8)
    if shadow:
        _stepped_rect((x + p, y + p * 2, w, h), _rgba(22, 29, 20, 0.28), p * 2)
    _stepped_rect(rect, border, p * 2)
    _stepped_rect((x + p, y + p, w - 2 * p, h - 2 * p), light, p)
    _stepped_rect((x + 2 * p, y + 2 * p, w - 4 * p, h - 4 * p), fill, p)
    draw_rect((x + 3 * p, y + h - 2 * p, w - 6 * p, p), GOLD)


def draw_inset(rect, *, fill=PAPER_DARK) -> None:
    x, y, w, h = rect
    p = min(2.0, w / 8, h / 8)
    draw_rect(rect, GOLD_DARK)
    draw_rect((x + p, y + p, w - p, h - p), PAPER_LIGHT)
    draw_rect((x + p, y + p, w - 2 * p, h - 2 * p), fill)


def draw_button(rect, *, hovered=False, pressed=False, danger=False) -> None:
    fill = RED if danger else FOREST_LIGHT if hovered else FOREST
    draw_frame(rect, fill=fill, light=GOLD_LIGHT if hovered else GOLD,
               shadow=False, inset=2.0)
    if pressed:
        x, y, w, _h = rect
        draw_rect((x + 4, y + 4, w - 8, 2), INK)


# Small silhouettes intentionally use integer pixel blocks, like the held sword.
_ICONS = {
    "sword": (
        "........##", ".......###", "......###.", ".....###..", "..#.###...",
        "..####....", "...##.....", "..#.##....", ".##.......", "##........",
    ),
    "shield": (
        ".########.", ".#......#.", ".#..##..#.", ".#..##..#.", ".########.",
        "..#.##.#..", "..#.##.#..", "...####...", "....##....", "..........",
    ),
    "boot": (
        "..#####...", "..#...#...", "..#...#...", "..#####...", "..#...#...",
        "..#...###.", ".##.....##", ".#.......#", ".#########", "..........",
    ),
    "spark": (
        "....##....", "....##....", "...####...", "..######..", "##########",
        "##########", "..######..", "...####...", "....##....", "....##....",
    ),
    "cards": (
        ".######...", ".#....#...", ".#.######.", ".#.#....#.", ".#.#.##.#.",
        ".#.#.##.#.", ".###....#.", "...#....#.", "...######.", "..........",
    ),
    "bag": (
        "...####...", "..##..##..", ".########.", ".#......#.", ".#..##..#.",
        "##..##..##", "##......##", ".#......#.", ".########.", "..........",
    ),
    "close": (
        "##......##", ".##....##.", "..##..##..", "...####...", "....##....",
        "...####...", "..##..##..", ".##....##.", "##......##", "..........",
    ),
}


def draw_pixel_icon(kind: str, rect, *, color=INK) -> None:
    pixels = _ICONS.get(kind, _ICONS["spark"])
    x, y, w, h = rect
    size = max(1, int(min(w / len(pixels[0]), h / len(pixels))))
    x = round(x + (w - size * len(pixels[0])) / 2)
    y = round(y + (h - size * len(pixels)) / 2)
    glColor4f(*color)
    glBegin(GL_QUADS)
    for row, line in enumerate(pixels):
        for col, pixel in enumerate(line):
            if pixel != "#":
                continue
            px, py = x + col * size, y + row * size
            glVertex2f(px, py)
            glVertex2f(px + size, py)
            glVertex2f(px + size, py + size)
            glVertex2f(px, py + size)
    glEnd()
