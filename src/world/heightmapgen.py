"""Heightmap generator

Generates a smooth grayscale heightmap (black=0 … white=1) using
fractal Brownian motion (fBM) Perlin noise. Saves to PNG.

Usage (examples):
  - Default 512x512:
      python heightmapgen.py
  - Larger, tiled, with specific seed:
      python heightmapgen.py -o ..\\assets\\textures\\heightmap.png \
          --width 1024 --height 1024 --scale 0.008 --octaves 6 \
          --persistence 0.5 --lacunarity 2.0 --seed 12345 --repeat 512

Notes:
  - Output is grayscale PNG with linear values. Set --gamma for tone mapping.
  - Use --ridged to generate ridged/mountainous style (abs + invert-ish).
  - Use --invert to flip black/white.
  - Use --repeat to produce a seamlessly tiling texture.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List

# We use pygame to write PNG without new dependencies
import pygame


# --------------------------- Perlin noise core ---------------------------- #


def _fade(t: float) -> float:
    # 6t^5 - 15t^4 + 10t^3 smoothstep for Perlin
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


_GRADS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
)


@dataclass
class Perlin2D:
    perm: List[int]
    repeat: int | None = None  # period for seamless tiling

    @classmethod
    def from_seed(cls, seed: int, repeat: int | None = None) -> "Perlin2D":
        rng = random.Random(seed)
        p = list(range(256))
        rng.shuffle(p)
        # duplicate to avoid overflow without mod; we'll still mod for repeat
        p = p + p
        return cls(p, repeat)

    def _hash(self, x: int, y: int) -> int:
        # If repeat is set, confine lattice to [0, repeat)
        if self.repeat is not None and self.repeat > 0:
            x %= self.repeat
            y %= self.repeat
        # Classic Perlin hashing via permutation table
        return self.perm[self.perm[x & 255] + (y & 255)] & 255

    def noise(self, x: float, y: float) -> float:
        # Integer lattice cell
        xi = math.floor(x)
        yi = math.floor(y)
        # Local coords inside cell
        xf = x - xi
        yf = y - yi

        # Hash corners -> gradient index
        h00 = self._hash(xi + 0, yi + 0)
        h10 = self._hash(xi + 1, yi + 0)
        h01 = self._hash(xi + 0, yi + 1)
        h11 = self._hash(xi + 1, yi + 1)

        # Gradient vectors
        g00 = _GRADS[h00 & 7]
        g10 = _GRADS[h10 & 7]
        g01 = _GRADS[h01 & 7]
        g11 = _GRADS[h11 & 7]

        # Dot products
        d00 = g00[0] * (xf - 0) + g00[1] * (yf - 0)
        d10 = g10[0] * (xf - 1) + g10[1] * (yf - 0)
        d01 = g01[0] * (xf - 0) + g01[1] * (yf - 1)
        d11 = g11[0] * (xf - 1) + g11[1] * (yf - 1)

        u = _fade(xf)
        v = _fade(yf)

        x1 = _lerp(d00, d10, u)
        x2 = _lerp(d01, d11, u)
        n = _lerp(x1, x2, v)

        # Perlin approx amplitude range ~[-sqrt(0.5), sqrt(0.5)] ~ [-0.707, 0.707]
        # We'll normalize after fBM accumulation, so return raw value
        return n


def fbm(
    n: Perlin2D,
    x: float,
    y: float,
    *,
    octaves: int = 5,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> float:
    value = 0.0
    amp = 1.0
    freq = 1.0
    for _ in range(max(1, octaves)):
        value += amp * n.noise(x * freq, y * freq)
        freq *= lacunarity
        amp *= gain
    return value


def fbm_ridged(
    n: Perlin2D,
    x: float,
    y: float,
    *,
    octaves: int = 6,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    ridge_gain: float = 1.0,
) -> float:
    # Ridged: use 1 - abs(noise), then accentuate ridge with squared term
    value = 0.0
    amp = 0.5  # start lower to keep range in check
    freq = 1.0
    for _ in range(max(1, octaves)):
        v = n.noise(x * freq, y * freq)
        v = 1.0 - abs(v)
        v = v * v  # sharpen
        value += v * amp * ridge_gain
        freq *= lacunarity
        amp *= gain
    # Shift to be roughly centered near 0
    return value - 0.5


# --------------------------- Image utilities ----------------------------- #


def to_surface_gray_u8(pixels01: List[float], w: int, h: int) -> pygame.Surface:
    # Build RGB byte buffer (grayscale replicated)
    buf = bytearray(w * h * 3)
    i = 0
    for v in pixels01:
        g = max(0, min(255, int(round(v * 255.0))))
        buf[i + 0] = g
        buf[i + 1] = g
        buf[i + 2] = g
        i += 3
    surf = pygame.image.frombuffer(buf, (w, h), "RGB")
    return surf.convert(24)


def box_blur(vals: List[float], w: int, h: int, iterations: int = 1) -> List[float]:
    """Apply a simple 3x3 box blur repeatedly (separable would be faster).

    This smooths high-frequency detail and makes the result more "rolly".
    """
    if iterations <= 0:
        return vals
    cur = list(vals)
    for _ in range(iterations):
        out = [0.0] * (w * h)
        for y in range(h):
            for x in range(w):
                s = 0.0
                cnt = 0
                # average over 3x3 neighborhood (clamped at borders)
                for dy in (-1, 0, 1):
                    ny = y + dy
                    if 0 <= ny < h:
                        base = ny * w
                        for dx in (-1, 0, 1):
                            nx = x + dx
                            if 0 <= nx < w:
                                s += cur[base + nx]
                                cnt += 1
                out[y * w + x] = s / max(1, cnt)
        cur = out
    return cur


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# ------------------------------ CLI runner ------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a grayscale heightmap PNG using fBM Perlin noise"
    )
    ap.add_argument(
        "-o",
        "--output",
        default=os.path.join("..", "assets", "textures", "heightmap.png"),
        help="Output PNG path",
    )
    ap.add_argument("--width", type=int, default=512, help="Image width in pixels")
    ap.add_argument("--height", type=int, default=512, help="Image height in pixels")
    ap.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Base scale (world units per pixel); lower = zoomed in",
    )
    ap.add_argument("--octaves", type=int, default=6, help="Number of fBM octaves")
    ap.add_argument(
        "--persistence",
        type=float,
        default=0.5,
        help="Gain per octave (amplitude multiplier)",
    )
    ap.add_argument(
        "--lacunarity", type=float, default=2.0, help="Frequency multiplier per octave"
    )
    ap.add_argument(
        "--seed", type=int, default=None, help="Random seed (int); defaults to random"
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Tile period in lattice units for seamless tiling (e.g., 512)",
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Apply gamma correction (e.g., 1.6 for softer midtones)",
    )
    ap.add_argument("--invert", action="store_true", help="Invert final grayscale")
    ap.add_argument(
        "--ridged", action="store_true", help="Use ridged/mountainous style fBM"
    )
    ap.add_argument(
        "--exaggeration",
        type=float,
        default=1.0,
        help=(
            "Multiply deviation from 0.5 to exaggerate amplitude. "
            "Values >1 increase hills/valleys, <1 soften them (default 1.0)"
        ),
    )
    ap.add_argument(
        "--blur",
        type=int,
        default=0,
        help=(
            "Apply N iterations of a 3x3 box blur to smooth detail and create "
            "more roll-like hills (default 0 = no blur)."
        ),
    )

    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    noise = Perlin2D.from_seed(seed, repeat=args.repeat)

    w, h = max(1, args.width), max(1, args.height)
    base_scale = max(1e-6, float(args.scale))

    # Generate raw values and keep track of min/max for normalization
    vals: List[float] = []
    vmin, vmax = float("inf"), float("-inf")
    use_ridged = bool(args.ridged)
    for j in range(h):
        y = j * base_scale
        for i in range(w):
            x = i * base_scale
            if use_ridged:
                v = fbm_ridged(
                    noise,
                    x,
                    y,
                    octaves=args.octaves,
                    lacunarity=args.lacunarity,
                    gain=args.persistence,
                )
            else:
                v = fbm(
                    noise,
                    x,
                    y,
                    octaves=args.octaves,
                    lacunarity=args.lacunarity,
                    gain=args.persistence,
                )
            vals.append(v)
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v

    # Normalize to [0,1]
    eps = 1e-12
    rng = max(eps, (vmax - vmin))
    vals01 = [(v - vmin) / rng for v in vals]

    # Optional exaggeration: multiply deviation from mid (0.5) to increase amplitude
    ex = float(args.exaggeration)
    if abs(ex - 1.0) > 1e-6:
        vals01 = [max(0.0, min(1.0, (v - 0.5) * ex + 0.5)) for v in vals01]

    # Optional box blur to make shapes rollier
    if int(args.blur) > 0:
        vals01 = box_blur(vals01, w, h, iterations=int(args.blur))

    # Optional gamma and invert
    g = max(1e-6, float(args.gamma))
    if abs(g - 1.0) > 1e-6:
        inv_g = 1.0 / g
        vals01 = [pow(v, inv_g) for v in vals01]
    if args.invert:
        vals01 = [1.0 - v for v in vals01]

    # Make sure pygame is initialized enough to write image
    if not pygame.get_init():
        pygame.init()
    try:
        surf = to_surface_gray_u8(vals01, w, h)
        out_path = os.path.abspath(args.output)
        ensure_parent_dir(out_path)
        pygame.image.save(surf, out_path)
        print(
            f"Saved heightmap: {out_path}  (seed={seed}, min={vmin:.4f}, max={vmax:.4f})"
        )
    finally:
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
