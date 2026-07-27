# 3D asset provenance

All runtime GLB files under `characters/`, `items/storybook/`,
`effects/storybook/`, and `environment/storybook/` are original project assets
generated from the reproducible Blender scripts in `art/blender/scripts/`.

- Character and prop concepts were generated with the built-in ImageGen tool
  from the user-provided style reference and are stored under
  `art/reference/`.
- Geometry, rigs, UV-independent color materials, and animation were authored
  locally through Blender MCP. No Sketchfab, Poly Haven, Hyper3D, Hunyuan3D, or
  other external 3D model service was used.
- Runtime GLBs may be used and redistributed with this game. Blender sources
  and reference images are excluded from Web exports.
