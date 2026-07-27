# ImageGen runtime art provenance

All bitmap assets in this directory are original project artwork produced with
the built-in ImageGen tool on 2026-07-26. The user-provided images were used
only as style and composition references.

- Character identities, poses, environment cutouts, items, effects, and UI
  were generated on a solid magenta key background.
- `tools/art_pipeline/process_sprite25d_assets.py` performs deterministic
  key removal, edge decontamination, atlas splitting, and cyan team-mask
  extraction. It does not synthesize or repaint the artwork.
- No third-party asset library, native transparent-image API, stock game-art
  pack, or external 3D generation service is used.
- Source generations, prompts, checksums, and acceptance images are archived
  under `art/reference/sprite25d/` and excluded from exports.
