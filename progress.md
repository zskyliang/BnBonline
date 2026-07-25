Original prompt: Implement the “BnBonline 等距海岛与水泡特效重制” plan while keeping all game rules and AI behavior unchanged.

## 2026-07-25

- Replaced the old runtime maps with the fixed 15×13 `harbor-market` and
  `bell-garden` layouts, typed building placements and exact rigid footprints.
- Added the 40°/38° orthographic camera fit, opaque clay water/island, CC0
  Kenney buildings, CC0 Quaternius shoreline nature, and Bayer-dithered roof
  occlusion driven by visual-only camera rays.
- Replaced ball skins with `aqua`/`coral` clay water bubbles and added the
  0.45-second compressed-core, water-column and dithered-foam explosion stages,
  including a 48-droplet cap and pooled views.
- Migrated legacy map and bubble IDs, updated setup/HUD UI, replaced the lobby
  windmill with a clock tower and expanded the Noto Sans SC subset.
- Documented final third-party files, processing and SHA-256 values.
- Verified 98 rule checks, 116 character/3D checks, smoke flow, AI benchmark,
  four-AI stress, 1280×720/1040×600 screenshots and browser behavior.
- Exported Web Release below the 50 MiB limit. The PCK excludes old zodiac and
  sprite directories.
- Rotated the match view counter-clockwise by 90 degrees, added 85–150% live
  camera zoom with a 110% default, and exposed mouse-wheel, keyboard and HUD
  controls.
- Replaced the excessive -50 degree orbit with a fixed -5 degree horizontal
  skew, keeping the 38 degree elevation while aligning movement within 9
  degrees vertically and 4 degrees horizontally on screen.
- Replaced end-stopping character playback with distance-driven, explicitly
  looped 12 FPS gait sampling so held movement keeps alternating foot poses
  without sliding on a frozen frame.
- Regenerated the reproducible Noto Sans SC runtime subset with the new zoom
  labels and Web-critical regular/ideographic space glyphs.

## Remaining

- No known blockers.
