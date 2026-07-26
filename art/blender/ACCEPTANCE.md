# Storybook 2.5D production acceptance

Date: 2026-07-26

## Waddle self-acceptance

- Both arms remain broadly spread throughout the four sampled poses.
- Left and right feet alternate between planted and visibly lifted positions.
- The shared Root shifts left/right and rises during passing poses, preventing
  rigid world-space sliding.
- All eight GLBs pass numeric animation-track checks for arm rotation, leg
  displacement, and Root displacement.

Visual evidence:

- `previews/cat-waddle-contact-sheet.png`
- `previews/bear-waddle-contact-sheet.png`
- `previews/eight-character-blender-contact-sheet.png`

## Automated acceptance

| Suite | Result |
| --- | --- |
| Rules and progression | 89 checks, 0 failures |
| Storybook GLB import/budget/action contracts | 184 checks, 0 failures |
| Visual structure and Compatibility material contracts | 107 checks, 0 failures |
| End-to-end smoke | PASS |
| AI paint benchmark | avg 1.51 ms, p95 1.54 ms |
| AI item benchmark | avg 3.99 ms, p95 4.06 ms |
| Item training/holdout | PASS |
| Four-AI stress | PASS, p95 6.60 ms |
| Headless import | PASS, no persistent warnings |
| Web Release export | PASS |
| Browser console | 0 game warnings/errors |
| Browser gameplay, 1280×720 Release | stable 60 FPS |

## Snapshot matrix

`previews/acceptance/` contains 21 PNGs: lobby, character selection, normal
battle, full items, chain explosion, trapped actor, and victory settlement at
1280×720, 1040×600, and 1920×1080.

The final review confirms:

- every playable tile is visible at the default -30°/42° orthographic camera;
- exterior forest decoration does not cover tiles, actors, grid lines, or HUD;
- five simultaneous species and their local team tints remain distinguishable;
- trap-bubble and chain-explosion shapes remain readable without blocking HUD;
- the 1040×600 character grid and action buttons remain inside safe bounds.
