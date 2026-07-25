#!/usr/bin/env python3
"""Generate the small shared textures used by the Web-friendly clay material."""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageFilter


SIZE = 256
SEED = 0xB0B5
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "materials"


def _wrapped_delta(value: float, center: float) -> float:
    delta = abs(value - center)
    return min(delta, SIZE - delta)


def make_height() -> Image.Image:
    rng = random.Random(SEED)
    dents = [
        (
            rng.uniform(0.0, SIZE),
            rng.uniform(0.0, SIZE),
            rng.uniform(5.0, 18.0),
            rng.uniform(-0.22, 0.18),
        )
        for _ in range(34)
    ]
    pixels: list[int] = []
    for y in range(SIZE):
        for x in range(SIZE):
            ridge = (
                math.sin((x + math.sin(y * 0.061) * 15.0) * 0.115) * 0.035
                + math.sin((y - x * 0.22) * 0.051) * 0.025
                + math.sin((x + y) * 0.019) * 0.018
            )
            dent_value = 0.0
            for cx, cy, radius, strength in dents:
                dx = _wrapped_delta(x, cx)
                dy = _wrapped_delta(y, cy)
                distance = math.sqrt(dx * dx + dy * dy)
                if distance < radius:
                    falloff = 0.5 + 0.5 * math.cos(math.pi * distance / radius)
                    dent_value += strength * falloff
            value = max(0.0, min(1.0, 0.5 + ridge + dent_value))
            pixels.append(round(value * 255.0))
    image = Image.new("L", (SIZE, SIZE))
    image.putdata(pixels)
    return image.filter(ImageFilter.GaussianBlur(radius=1.15))


def make_normal(height: Image.Image) -> Image.Image:
    source = height.load()
    strength = 1.55
    pixels: list[tuple[int, int, int]] = []
    for y in range(SIZE):
        for x in range(SIZE):
            left = source[(x - 1) % SIZE, y] / 255.0
            right = source[(x + 1) % SIZE, y] / 255.0
            down = source[x, (y - 1) % SIZE] / 255.0
            up = source[x, (y + 1) % SIZE] / 255.0
            nx = (left - right) * strength
            ny = (down - up) * strength
            nz = 1.0
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            pixels.append(
                (
                    round((nx / length * 0.5 + 0.5) * 255.0),
                    round((ny / length * 0.5 + 0.5) * 255.0),
                    round((nz / length * 0.5 + 0.5) * 255.0),
                )
            )
    image = Image.new("RGB", (SIZE, SIZE))
    image.putdata(pixels)
    return image


def make_roughness(height: Image.Image) -> Image.Image:
    source = height.load()
    pixels: list[int] = []
    for y in range(SIZE):
        for x in range(SIZE):
            variation = (source[x, y] / 255.0 - 0.5) * 0.12
            pixels.append(round(max(0.0, min(1.0, 0.9 + variation)) * 255.0))
    image = Image.new("L", (SIZE, SIZE))
    image.putdata(pixels)
    return image


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    height = make_height()
    make_normal(height).save(OUTPUT_DIR / "clay_detail_normal.png", optimize=True)
    make_roughness(height).save(OUTPUT_DIR / "clay_roughness.png", optimize=True)


if __name__ == "__main__":
    main()
