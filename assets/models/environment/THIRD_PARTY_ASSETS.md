# Environment asset manifest

Downloaded on 2026-07-25. The game loads these files locally and has no runtime
network dependency.

## Kenney Fantasy Town Kit

- Author: Kenney
- Source: https://kenney.nl/assets/fantasy-town-kit
- License: CC0 1.0 Universal
- Selected files: modular walls, doors, shuttered windows, three roofs,
  market stalls, cart, gate, dock planks, round fountain and the shared
  512×512 color map under `kenney_town/`.
- Processing: copied from the official `kenney_fantasy-town-kit_2.0.zip`;
  filenames were normalized to lowercase. Geometry and texture pixels are
  unchanged.

## Quaternius Stylized Nature MegaKit

- Author: Quaternius
- Source: https://quaternius.com/packs/stylizednaturemegakit.html
- License: CC0 1.0 Universal
- Selected files: flower bush, clover, common tree, flower group, round rock
  and twisted tree under `quaternius_nature/`.
- Processing: source glTF scenes were repacked as GLB with glTF Transform
  4.2.1. Referenced textures were resized to at most 512 px before repacking;
  no source scene outside this selected subset is shipped.

Final SHA-256 values for every shipped source asset are recorded in
[`SHA256SUMS`](./SHA256SUMS). Godot-generated `.import` metadata is not included
in the hash inventory.
