# Blender MCP 资产流水线

本目录保存《森林泡泡染色战》的原创 Blender 源文件、可复现生成脚本和验收预览。
`art/.gdignore` 阻止 Godot 导入这些创作源文件，`export_presets.cfg` 也会从 Web
发行包排除整个 `art/` 目录。

## Production v1

- `characters/`：猫、狗、兔、熊、狐狸、浣熊、企鹅、水豚的独立 `.blend`。
- `props/`：叶片鞋、水泡葫芦、爪印爆发星。
- `effects/`：队色水泡和泡沫花瓣/液滴/十字扩散形状。
- `environment/`：单块地砖、森林棋盘外框与绘本大厅。
- `scripts/generate_storybook_characters.py`：生成、预览并导出八角色。
- `scripts/generate_storybook_world_assets.py`：生成、预览并导出所有世界资产。
- `previews/`：Blender 材质预览、八角色总览、Waddle 动作表及 Godot 验收图。
- `storybook-manifest.json`：预算、动作、路径和文件校验值。

角色脚本的公开入口为 `generate_character("cat")` 或
`generate_all_characters()`；世界脚本入口为 `generate_all_assets()`。正式生成
过程通过 Blender MCP 的 `execute_blender_code` 在 Blender 内执行，不使用
Sketchfab、Poly Haven、Hyper3D、Hunyuan3D 或其他外部三维生成服务。

## 统一约定

- Blender 使用 Z 轴向上，GLB 导出为 Y 轴向上。
- 原点位于脚底中心，角色高度为 1.3 单位。
- 共享骨架包含 10 根骨骼。
- 动画按 8 FPS、Constant 插值采样。
- `Waddle` 始终张开双臂，左右脚交替离地，并通过 Root 横移与抬升表现重心。
- 共享动作：`Idle`、`Waddle`、`PlaceBubble`、`Trapped`、`Defeat`、`Victory`。
- 局部队色材质：`TeamTint`、`FootRing`。
- 固定材质：`Fur`、`SpeciesMarking`、`Belly`、`Muzzle`、`Eyes`、`Pupils`、`Nose`。
