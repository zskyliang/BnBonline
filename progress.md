Original prompt: Implement the “BnBonline 等距海岛与水泡特效重制” plan while keeping all game rules and AI behavior unchanged.

## 2026-07-26

- Added capacity-pressure paint utility and safe current-cell barrage chaining;
  the fixed four-AI live stress match now reaches 0.00 average peak free slots
  instead of waiting on one opening bubble, while retaining virtual-fuse escape
  validation.
- Replaced the mechanical battle mix with a cute CC0 level loop, natural bubble
  placement/appearance cues, a short vocal pop explosion and pizzicato/toy-like
  interaction stingers. The explosion now receives 8 dB of additional
  attenuation and subtle upward pitch variation for softer chain reactions;
  source mapping, license links and SHA-256 records were refreshed.
- Raised safe positive-payback pickups above paint and contact decisions, while
  retaining hazard routing, late-round rejection and teammate claim ownership.
- Moved camera controls into a pausing settings modal, made live score/item
  panels translucent, and removed the always-visible camera panel.
- Changed trap timeout into automatic escape. Only an opposing touch finish now
  locks permanent 3×3 territory, rendered solely as a translucent tile shadow.
- Added persisted orthographic camera azimuth, elevation and zoom preferences,
  live right-drag orbit controls, bounded angles and a responsive HUD camera
  panel while retaining fixed map-relative movement.
- Added a renderer-independent item board with stable IDs, atomic collection,
  ten-second stage scheduling and uncapped stage-only speed, bubble-capacity and
  power stacks that survive respawns but reset at settlement.
- Imported only the spring, star and bomb pickup GLBs plus their shared palette
  from Kenney Platformer Kit, with CC0 provenance and SHA-256 records; added
  hovering, rings and an explicit bubble `+1` marker.
- Extended AI snapshots with items, remaining time and temporary stats. Added
  paint-comparable marginal item utility, safe routing, late-round/competition
  costs, per-item AI claims and a serialized tuned behavior profile.
- Expanded rules, smoke, visual, AI item benchmark and item-enabled four-AI
  stress coverage.
- Re-ran the seeded stage 1/2/4 profile search after barrage tuning; the neutral
  `x1.00` item profile wins the tie, and the holdout set reaches 100%
  participation on safe positive-payback pickups with 350% expected-territory
  improvement. Latest paint/item decision average/P95 are 1.58/2.27ms and
  3.92/4.07ms; the fixed four-AI stress P95 is 7.01ms.
- Rebuilt the Noto Sans SC runtime subset for the camera, countdown, temporary
  bonus and pickup feedback strings, then checked default and extreme camera
  composition at both target resolutions.

## 2026-07-25

- Replaced the kill-score island match with a three-minute, 15×13 paint-territory
  campaign: all 195 cells are walkable, mutable paint can be stolen, and defeat
  locks the opposing color into a permanent clipped 3×3 neighborhood.
- Added seven selectable clothing palettes with per-character material
  whitelists; player/AI clothing, bubbles, explosions, trap shells and floor
  ownership now share their team color.
- Added in-memory infinite-run progression, mandatory post-win skill selection,
  uncapped `+10` speed / `+1` bubble / `+1` power stacking, stage-scaled AI
  counts and per-stage randomized AI point distributions.
- Replaced item/box objectives with paint-swing AI scoring, pending-blast overlap
  avoidance and AI-team friendly-fire immunity while retaining fuse-aware
  escape planning and staggered scheduling.
- Rebuilt the arena renderer as two fixed 195-instance floor/lock `MultiMesh`
  batches and removed buildings, shoreline scenery and roof occlusion.
- Replaced the setup, live scoreboard and result flow for color selection,
  territory counts, locked counts, retry handling and skill-gated advancement.
- Regenerated the runtime Noto Sans SC subset for all new Chinese UI strings.
- Added paint rules, visual structure, progression smoke, AI paint benchmark and
  four-AI stress coverage.

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
