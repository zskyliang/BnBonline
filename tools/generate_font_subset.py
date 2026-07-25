#!/usr/bin/env python3
"""Build the runtime Noto Sans SC subset from strings shipped by the game."""

from __future__ import annotations

import argparse
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont


ROOT = Path(__file__).resolve().parents[1]
TEXT_ROOTS = (
    ROOT / "scripts",
    ROOT / "scenes",
)
TEXT_FILES = (
    ROOT / "project.godot",
)
TEXT_SUFFIXES = {".gd", ".tscn"}


def collect_codepoints() -> set[int]:
    characters = {chr(codepoint) for codepoint in range(0x20, 0x7F)}
    paths: list[Path] = list(TEXT_FILES)
    for text_root in TEXT_ROOTS:
        paths.extend(
            path
            for path in text_root.rglob("*")
            if path.is_file() and path.suffix in TEXT_SUFFIXES
        )
    for path in paths:
        characters.update(path.read_text(encoding="utf-8"))
    # Keep both regular and ideographic spaces. Web exports cannot fall back to
    # a system font for these glyphs, so dropping them produces visible boxes.
    return {
        ord(character)
        for character in characters
        if character not in {"\r", "\n", "\t"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets/fonts/NotoSansSC-BnB-Subset.ttf",
    )
    arguments = parser.parse_args()

    font = TTFont(arguments.source)
    if "fvar" in font:
        font = instantiateVariableFont(font, {"wght": 400}, inplace=True)

    options = subset.Options()
    options.layout_features = ["*"]
    options.name_IDs = ["*"]
    options.name_languages = ["*"]
    options.notdef_glyph = True
    options.recommended_glyphs = True
    subsetter = subset.Subsetter(options=options)
    codepoints = collect_codepoints()
    subsetter.populate(unicodes=codepoints)
    subsetter.subset(font)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    font.save(arguments.output)
    print(
        f"Saved {arguments.output.relative_to(ROOT)} "
        f"with {len(codepoints)} requested codepoints"
    )


if __name__ == "__main__":
    main()
