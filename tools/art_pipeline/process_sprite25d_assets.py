#!/usr/bin/env python3
"""Prepare the approved ImageGen sprite sheets for the Godot runtime.

The script is intentionally deterministic: it removes the fixed magenta key,
extracts the cyan team-tint mask, normalizes character poses, and slices the
environment/effect/UI contact sheets into versioned PNG assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "art/reference/sprite25d/source"
DIRECTIONAL_SOURCE_ROOT = (
    PROJECT_ROOT / "art/reference/sprite25d/directional-v2/source"
)
BUBBLE_SOURCE_ROOT = (
    PROJECT_ROOT / "art/reference/sprite25d/bubbles-v3/source"
)
PLANT_WIND_SOURCE_ROOT = (
    PROJECT_ROOT / "art/reference/sprite25d/plant-wind-v2/source"
)
RUNTIME_ROOT = PROJECT_ROOT / "assets/art/storybook25d"

CHARACTERS = (
    "cat",
    "dog",
    "rabbit",
    "bear",
    "fox",
    "raccoon",
    "penguin",
    "capybara",
)
DIRECTIONS = ("down", "up", "left", "right")
WIND_PLANTS = (
    "conifer",
    "bush",
    "mushrooms",
    "flowers",
    "lavender",
)
ENVIRONMENT_ASSETS = (
    "floor_tile",
    "wood_rail",
    "wood_corner",
    "conifer",
    "bush",
    "mushrooms",
    "stump",
    "wood_sign",
    "rocks",
    "flowers",
    "lavender",
    "moss",
)
ITEM_EFFECT_ASSETS = (
    "leaf_shoes",
    "bubble_gourd",
    "paw_burst",
    "bubble_bomb",
    "trap_bubble",
    "pop_core",
    "foam_burst",
    "cross_splash",
)
UI_ASSETS = (
    "paper_panel",
    "button_normal",
    "button_hover",
    "button_pressed",
    "paper_badge",
    "leaf_divider",
)


def keyed_rgba(image: Image.Image) -> Image.Image:
    """Remove the exact ImageGen magenta backdrop with a soft despilled matte."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    key = np.array([255.0, 0.0, 255.0], dtype=np.float32)
    distance = np.linalg.norm(rgb - key, axis=2)
    exact_key_amount = 1.0 - np.clip((distance - 5.0) / 82.0, 0.0, 1.0)

    # ImageGen can retain a faint watercolor value pattern even when asked for
    # a flat key. Detect the whole magenta family instead of relying only on
    # distance from #ff00ff. Natural pink ears/noses are protected because they
    # have substantially more green and much lower magenta dominance.
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    magenta_floor = np.minimum(red, blue)
    magenta_dominance = magenta_floor - green
    red_blue_balance = np.abs(red - blue)
    chroma_key_amount = (
        np.clip((magenta_dominance - 38.0) / 82.0, 0.0, 1.0)
        * np.clip((magenta_floor - 82.0) / 88.0, 0.0, 1.0)
        * (1.0 - np.clip((red_blue_balance - 74.0) / 92.0, 0.0, 1.0))
    )
    key_amount = np.maximum(exact_key_amount, chroma_key_amount)
    alpha = 1.0 - key_amount

    # Recover the foreground color from the keyed composite at antialiased
    # edges. This avoids a pink fringe after texture filtering.
    safe_alpha = np.maximum(alpha[..., None], 1.0 / 255.0)
    foreground = (rgb - (1.0 - alpha[..., None]) * key) / safe_alpha
    foreground = np.clip(foreground, 0.0, 255.0)
    foreground[alpha < 0.01] = 0.0

    rgba = np.dstack((foreground, alpha[..., None] * 255.0)).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def team_mask_and_neutral_base(
    keyed: Image.Image,
    original: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    """Extract watercolor cyan markers and replace them with a neutral wash."""

    rgba = np.asarray(keyed.convert("RGBA"), dtype=np.float32)
    rgb = np.asarray(original.convert("RGB"), dtype=np.float32)
    hsv = np.asarray(original.convert("HSV"), dtype=np.float32)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    alpha = rgba[..., 3] / 255.0

    cyan_hue = np.minimum(np.abs(hue - 127.0), 255.0 - np.abs(hue - 127.0))
    hue_weight = np.clip(1.0 - cyan_hue / 34.0, 0.0, 1.0)
    saturation_weight = np.clip((saturation - 52.0) / 128.0, 0.0, 1.0)
    blue_green_weight = np.clip(
        (np.minimum(rgb[..., 1], rgb[..., 2]) - rgb[..., 0] - 12.0) / 96.0,
        0.0,
        1.0,
    )
    mask = hue_weight * saturation_weight * blue_green_weight * alpha
    mask = np.clip(mask * 1.5, 0.0, 1.0)

    luminance = (
        rgba[..., 0] * 0.2126
        + rgba[..., 1] * 0.7152
        + rgba[..., 2] * 0.0722
    )
    neutral = np.stack(
        (
            luminance * 0.88 + 31.0,
            luminance * 0.94 + 20.0,
            luminance * 0.95 + 18.0,
        ),
        axis=2,
    )
    mix = (mask * 0.92)[..., None]
    rgba[..., :3] = rgba[..., :3] * (1.0 - mix) + neutral * mix

    base = Image.fromarray(np.clip(rgba, 0.0, 255.0).astype(np.uint8), "RGBA")
    mask_rgba = np.zeros_like(rgba, dtype=np.uint8)
    mask_channel = np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8)
    mask_rgba[..., :3] = mask_channel[..., None]
    mask_rgba[..., 3] = np.asarray(keyed.getchannel("A"), dtype=np.uint8)
    return base, Image.fromarray(mask_rgba, "RGBA")


def cell(image: Image.Image, columns: int, rows: int, index: int) -> Image.Image:
    column = index % columns
    row = index // columns
    x0 = round(column * image.width / columns)
    x1 = round((column + 1) * image.width / columns)
    y0 = round(row * image.height / rows)
    y1 = round((row + 1) * image.height / rows)
    return image.crop((x0, y0, x1, y1))


def visible_bbox(image: Image.Image, threshold: int = 8) -> tuple[int, int, int, int]:
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.where(alpha > threshold)
    if xs.size == 0:
        return (0, 0, image.width, image.height)
    return (
        max(0, int(xs.min()) - 4),
        max(0, int(ys.min()) - 4),
        min(image.width, int(xs.max()) + 5),
        min(image.height, int(ys.max()) + 5),
    )


def normalized_character_pose(
    base: Image.Image,
    mask: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    bbox = visible_bbox(base)
    cropped_base = base.crop(bbox)
    cropped_mask = mask.crop(bbox)
    scale = min(448.0 / cropped_base.width, 452.0 / cropped_base.height)
    size = (
        max(1, round(cropped_base.width * scale)),
        max(1, round(cropped_base.height * scale)),
    )
    cropped_base = cropped_base.resize(size, Image.Resampling.LANCZOS)
    cropped_mask = cropped_mask.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (512, 512))
    mask_canvas = Image.new("RGBA", (512, 512))
    x = (512 - size[0]) // 2
    y = 488 - size[1]
    canvas.alpha_composite(cropped_base, (x, y))
    mask_canvas.alpha_composite(cropped_mask, (x, y))
    return canvas, mask_canvas


def normalized_bubble_skin(
    base: Image.Image,
    mask: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    """Normalize character bubbles to one grounded 512px runtime contract."""

    bbox = visible_bbox(base)
    cropped_base = base.crop(bbox)
    cropped_mask = mask.crop(bbox)
    scale = min(450.0 / cropped_base.width, 480.0 / cropped_base.height)
    size = (
        max(1, round(cropped_base.width * scale)),
        max(1, round(cropped_base.height * scale)),
    )
    cropped_base = cropped_base.resize(size, Image.Resampling.LANCZOS)
    cropped_mask = cropped_mask.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (512, 512))
    mask_canvas = Image.new("RGBA", (512, 512))
    x = (512 - size[0]) // 2
    y = 496 - size[1]
    canvas.alpha_composite(cropped_base, (x, y))
    mask_canvas.alpha_composite(cropped_mask, (x, y))
    return canvas, mask_canvas


def normalized_wind_frames(
    frames: list[Image.Image],
    target_size: tuple[int, int],
) -> list[Image.Image]:
    """Keep every redrawn plant frame on one fixed root baseline."""

    cropped_frames = [frame.crop(visible_bbox(frame)) for frame in frames]
    maximum_width = max(frame.width for frame in cropped_frames)
    maximum_height = max(frame.height for frame in cropped_frames)
    target_width, target_height = target_size
    scale = min(
        float(target_width - 2) / maximum_width,
        float(target_height - 2) / maximum_height,
    )
    normalized: list[Image.Image] = []
    for frame in cropped_frames:
        size = (
            max(1, round(frame.width * scale)),
            max(1, round(frame.height * scale)),
        )
        resized = frame.resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", target_size)
        x = (target_width - size[0]) // 2
        y = target_height - size[1] - 1
        canvas.alpha_composite(resized, (x, y))
        normalized.append(canvas)
    return normalized


def centered_component_bbox(
    image: Image.Image,
    threshold: int = 8,
) -> tuple[int, int, int, int]:
    """Return the connected alpha island nearest the cell center.

    ImageGen occasionally lets a few pixels from a neighboring contact-sheet
    cell cross a guide boundary. UI pieces are centered, so selecting the
    center island removes those fragments without redrawing the source art.
    """

    alpha = np.asarray(image.getchannel("A")) > threshold
    if not alpha.any():
        return (0, 0, image.width, image.height)
    center = np.array([image.height // 2, image.width // 2])
    candidates = np.argwhere(alpha)
    distances = np.square(candidates - center).sum(axis=1)
    start_y, start_x = candidates[int(distances.argmin())]
    visited = np.zeros_like(alpha, dtype=np.bool_)
    pending: deque[tuple[int, int]] = deque([(int(start_x), int(start_y))])
    visited[start_y, start_x] = True
    minimum_x = maximum_x = int(start_x)
    minimum_y = maximum_y = int(start_y)
    while pending:
        x, y = pending.popleft()
        minimum_x = min(minimum_x, x)
        maximum_x = max(maximum_x, x)
        minimum_y = min(minimum_y, y)
        maximum_y = max(maximum_y, y)
        for neighbor_x, neighbor_y in (
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ):
            if (
                neighbor_x < 0
                or neighbor_y < 0
                or neighbor_x >= image.width
                or neighbor_y >= image.height
                or visited[neighbor_y, neighbor_x]
                or not alpha[neighbor_y, neighbor_x]
            ):
                continue
            visited[neighbor_y, neighbor_x] = True
            pending.append((neighbor_x, neighbor_y))
    return (
        max(0, minimum_x - 4),
        max(0, minimum_y - 4),
        min(image.width, maximum_x + 5),
        min(image.height, maximum_y + 5),
    )


def trimmed_asset(
    image: Image.Image,
    max_size: int = 512,
    isolate_center: bool = False,
) -> Image.Image:
    bbox = centered_component_bbox(image) if isolate_center else visible_bbox(image)
    cropped = image.crop(bbox)
    scale = min(1.0, max_size / max(cropped.width, cropped.height))
    if scale < 1.0:
        cropped = cropped.resize(
            (round(cropped.width * scale), round(cropped.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return cropped


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wide_button_asset(
    image: Image.Image,
    output_size: tuple[int, int] = (720, 144),
) -> Image.Image:
    """Extend only the painted center so wide HUD buttons keep round ends."""

    target_width, target_height = output_size
    scale = target_height / image.height
    scaled = image.resize(
        (max(1, round(image.width * scale)), target_height),
        Image.Resampling.LANCZOS,
    )
    cap_width = min(target_height // 2, scaled.width // 3)
    left = scaled.crop((0, 0, cap_width, target_height))
    right = scaled.crop(
        (scaled.width - cap_width, 0, scaled.width, target_height)
    )
    center = scaled.crop(
        (cap_width, 0, scaled.width - cap_width, target_height)
    ).resize(
        (target_width - cap_width * 2, target_height),
        Image.Resampling.LANCZOS,
    )
    output = Image.new("RGBA", output_size)
    output.alpha_composite(left, (0, 0))
    output.alpha_composite(center, (cap_width, 0))
    output.alpha_composite(right, (target_width - cap_width, 0))
    return output


def process_directional_character(character_id: str) -> dict[str, object]:
    """Prepare one approved four-direction ImageGen character set."""

    character_dir = RUNTIME_ROOT / "characters" / character_id
    movement_manifest: dict[str, list[dict[str, str]]] = {}
    for direction in DIRECTIONS:
        source_path = (
            DIRECTIONAL_SOURCE_ROOT
            / f"{character_id}-walk-{direction}-v2.png"
        )
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source = Image.open(source_path).convert("RGB")
        frames: list[dict[str, str]] = []
        for frame_index in range(4):
            original_cell = cell(source, 2, 2, frame_index)
            keyed_cell = keyed_rgba(original_cell)
            base, mask = team_mask_and_neutral_base(
                keyed_cell,
                original_cell,
            )
            base, mask = normalized_character_pose(base, mask)
            stem = f"walk_{direction}_{frame_index}"
            save_png(base, character_dir / f"{stem}.png")
            save_png(mask, character_dir / f"{stem}_mask.png")
            frames.append(
                {
                    "texture": (
                        "res://assets/art/storybook25d/characters/"
                        f"{character_id}/{stem}.png"
                    ),
                    "mask": (
                        "res://assets/art/storybook25d/characters/"
                        f"{character_id}/{stem}_mask.png"
                    ),
                }
            )
        movement_manifest[direction] = frames

    idle_source_path = (
        DIRECTIONAL_SOURCE_ROOT
        / f"{character_id}-idle-directions-v2.png"
    )
    if not idle_source_path.exists():
        raise FileNotFoundError(idle_source_path)
    idle_source = Image.open(idle_source_path).convert("RGB")
    idle_manifest: dict[str, dict[str, str]] = {}
    for index, direction in enumerate(DIRECTIONS):
        original_cell = cell(idle_source, 2, 2, index)
        keyed_cell = keyed_rgba(original_cell)
        base, mask = team_mask_and_neutral_base(
            keyed_cell,
            original_cell,
        )
        base, mask = normalized_character_pose(base, mask)
        stem = f"idle_{direction}"
        save_png(base, character_dir / f"{stem}.png")
        save_png(mask, character_dir / f"{stem}_mask.png")
        idle_manifest[direction] = {
            "texture": (
                "res://assets/art/storybook25d/characters/"
                f"{character_id}/{stem}.png"
            ),
            "mask": (
                "res://assets/art/storybook25d/characters/"
                f"{character_id}/{stem}_mask.png"
            ),
        }

    trapped_source_path = (
        DIRECTIONAL_SOURCE_ROOT / f"{character_id}-trapped-v2.png"
    )
    if not trapped_source_path.exists():
        raise FileNotFoundError(trapped_source_path)
    trapped_source = Image.open(trapped_source_path).convert("RGB")
    trapped_keyed = keyed_rgba(trapped_source)
    trapped_base, trapped_mask = team_mask_and_neutral_base(
        trapped_keyed,
        trapped_source,
    )
    trapped_base, trapped_mask = normalized_character_pose(
        trapped_base,
        trapped_mask,
    )
    save_png(trapped_base, character_dir / "trapped.png")
    save_png(trapped_mask, character_dir / "trapped_mask.png")

    return {
        "pixel_size": 0.00265,
        "canvas": [512, 512],
        "source_sha256": {
            **{
                f"walk_{direction}": sha256(
                    DIRECTIONAL_SOURCE_ROOT
                    / f"{character_id}-walk-{direction}-v2.png"
                )
                for direction in DIRECTIONS
            },
            "idle_directions": sha256(idle_source_path),
            "trapped": sha256(trapped_source_path),
        },
        "logical_actions": [
            "Idle",
            "WalkUp",
            "WalkDown",
            "WalkLeft",
            "WalkRight",
            "Trapped",
        ],
        "movement": movement_manifest,
        "idle": idle_manifest,
        "trapped": {
            "texture": (
                "res://assets/art/storybook25d/characters/"
                f"{character_id}/trapped.png"
            ),
            "mask": (
                "res://assets/art/storybook25d/characters/"
                f"{character_id}/trapped_mask.png"
            ),
        },
    }


def process_directional_characters(
    character_ids: tuple[str, ...] = CHARACTERS,
) -> dict[str, object]:
    return {
        character_id: process_directional_character(character_id)
        for character_id in character_ids
    }


def process_environment() -> None:
    source = Image.open(SOURCE_ROOT / "environment-assets-v1.png").convert("RGB")
    for index, name in enumerate(ENVIRONMENT_ASSETS):
        output = trimmed_asset(keyed_rgba(cell(source, 4, 3, index)))
        save_png(output, RUNTIME_ROOT / "environment" / f"{name}.png")

    tile = Image.open(RUNTIME_ROOT / "environment/floor_tile.png").convert("RGBA")
    variants = (
        tile,
        tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        tile.rotate(90, expand=True),
        tile.rotate(180, expand=True),
        tile.rotate(270, expand=True),
    )
    for index, variant in enumerate(variants):
        save_png(variant, RUNTIME_ROOT / "environment" / f"floor_tile_{index}.png")
    atlas = Image.new("RGBA", (768, 512))
    for index, variant in enumerate(variants):
        atlas_cell = ImageOps.contain(
            variant,
            (248, 248),
            method=Image.Resampling.LANCZOS,
        )
        x = (index % 3) * 256 + (256 - atlas_cell.width) // 2
        y = (index // 3) * 256 + (256 - atlas_cell.height) // 2
        atlas.alpha_composite(atlas_cell, (x, y))
    save_png(atlas, RUNTIME_ROOT / "environment/floor_tile_atlas.png")

    grass = Image.open(SOURCE_ROOT / "grass-seamless-v1.png").convert("RGB")
    grass = ImageOps.fit(grass, (1024, 1024), method=Image.Resampling.LANCZOS)
    save_png(grass.convert("RGBA"), RUNTIME_ROOT / "environment/grass.png")


def process_items_and_effects() -> None:
    source = Image.open(SOURCE_ROOT / "items-effects-v1.png").convert("RGB")
    for index, name in enumerate(ITEM_EFFECT_ASSETS):
        original_cell = cell(source, 4, 2, index)
        keyed = keyed_rgba(original_cell)
        base, mask = team_mask_and_neutral_base(keyed, original_cell)
        output = trimmed_asset(base)
        mask_output = trimmed_asset(mask)
        target_root = "items" if index < 3 else "effects"
        save_png(output, RUNTIME_ROOT / target_root / f"{name}.png")
        if index >= 3:
            save_png(
                mask_output,
                RUNTIME_ROOT / target_root / f"{name}_mask.png",
            )


def process_character_bubbles() -> dict[str, object]:
    manifest: dict[str, object] = {}
    for character_id in CHARACTERS:
        source_path = BUBBLE_SOURCE_ROOT / f"{character_id}-bubble-v3.png"
        original = Image.open(source_path).convert("RGB")
        keyed = keyed_rgba(original)
        base, mask = team_mask_and_neutral_base(keyed, original)
        base, mask = normalized_bubble_skin(base, mask)
        target_root = RUNTIME_ROOT / "effects" / "character_bubbles"
        save_png(base, target_root / f"{character_id}.png")
        save_png(mask, target_root / f"{character_id}_mask.png")
        manifest[character_id] = {
            "texture": (
                "res://assets/art/storybook25d/effects/"
                f"character_bubbles/{character_id}.png"
            ),
            "mask": (
                "res://assets/art/storybook25d/effects/"
                f"character_bubbles/{character_id}_mask.png"
            ),
            "source_sha256": sha256(source_path),
        }
    return manifest


def process_plant_wind_frames() -> dict[str, object]:
    manifest: dict[str, object] = {}
    target_root = RUNTIME_ROOT / "environment" / "wind"
    for plant_id in WIND_PLANTS:
        source_path = PLANT_WIND_SOURCE_ROOT / f"{plant_id}-wind-v2.png"
        source = Image.open(source_path).convert("RGB")
        keyed_frames = [
            keyed_rgba(cell(source, 2, 2, frame_index))
            for frame_index in range(4)
        ]
        reference = Image.open(
            RUNTIME_ROOT / "environment" / f"{plant_id}.png"
        ).convert("RGBA")
        runtime_frames = normalized_wind_frames(
            keyed_frames,
            reference.size,
        )
        frame_paths: list[str] = []
        for frame_index, frame in enumerate(runtime_frames):
            path = target_root / f"{plant_id}_{frame_index}.png"
            save_png(frame, path)
            frame_paths.append(
                "res://assets/art/storybook25d/environment/wind/"
                f"{plant_id}_{frame_index}.png"
            )
        manifest[plant_id] = {
            "frames": frame_paths,
            "source_sha256": sha256(source_path),
        }
    return manifest


def process_ui() -> None:
    source = Image.open(SOURCE_ROOT / "ui-kit-v1.png").convert("RGB")
    for index, name in enumerate(UI_ASSETS):
        output = trimmed_asset(
            keyed_rgba(cell(source, 3, 2, index)),
            isolate_center=True,
        )
        save_png(output, RUNTIME_ROOT / "ui" / f"{name}.png")
        if name.startswith("button_"):
            save_png(
                wide_button_asset(output),
                RUNTIME_ROOT / "ui" / f"{name}_wide.png",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--character",
        choices=CHARACTERS,
        help="Process one directional-v2 character without rebuilding shared art.",
    )
    arguments = parser.parse_args()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    if arguments.character:
        character_id = str(arguments.character)
        manifest = {
            "version": 2,
            "generator": "Codex built-in ImageGen",
            "characters": {
                character_id: process_directional_character(character_id),
            },
        }
        (RUNTIME_ROOT / "directional_character_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Prepared directional-v2 character:", character_id)
        return
    manifest = {
        "version": 3,
        "generator": "Codex built-in ImageGen",
        "characters": process_directional_characters(),
    }
    process_environment()
    process_items_and_effects()
    manifest["character_bubbles"] = process_character_bubbles()
    manifest["plant_wind"] = process_plant_wind_frames()
    process_ui()
    (RUNTIME_ROOT / "character_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Prepared",
        len(CHARACTERS),
        "characters and",
        len(ENVIRONMENT_ASSETS) + len(ITEM_EFFECT_ASSETS) + len(UI_ASSETS),
        "shared assets.",
    )


if __name__ == "__main__":
    main()
