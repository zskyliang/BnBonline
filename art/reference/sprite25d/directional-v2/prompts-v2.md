# Directional v2 — ImageGen 生成记录

版本：`directional-v2`  
生成器：Codex 内置 ImageGen（identity-preserve 编辑）  
身份参考：`art/reference/sprite25d/source/<character>-actions-v1.png`  
运行时输出：`assets/art/storybook25d/characters/<character>/`

## 固定视觉约束

- 儿童绘本水彩、炭笔墨线、圆肚短腿、大眼小瞳孔、轻微错焦眼神。
- 移动时双臂或企鹅鳍始终向两侧展开约 60°，身体呈笨拙、重心不稳的摆动。
- 队色标记固定为 `#00B7C7`；色键背景固定为纯色 `#FF00FF`。
- 不生成文字、水印、地面、阴影、道具或边框。
- 左右侧面分别生成，不做镜像；上移为完全背面；下移为完全正面。

## 移动表模板

每个角色、每个方向使用一次独立 ImageGen 编辑，生成 2×2 四帧表。阅读顺序为左上、右上、左下、右下：

1. 左脚落地、右脚抬起。
2. 重心过渡、身体抬高。
3. 右脚落地、左脚抬起。
4. 重心过渡、身体抬高。

通用提示词要求保持输入角色的物种、毛色、脸部花纹、尾巴、耳朵或鳍身份；四格使用相同比例与脚底基线，并为对应方向指定真实正面、完整背面、左侧面或右侧面。

## Idle 与 Trapped 模板

- `idle-directions-v2`：2×2，顺序固定为 `down / up / left / right`，相同尺寸与脚底基线；双臂略微张开但保持静止。
- `trapped-v2`：单张正面全身姿势，不绘制泡泡本体；双臂抬起、双脚靠拢悬空、表情惊讶无助。

## 本地确定性处理

`tools/art_pipeline/process_sprite25d_assets.py` 负责：

- 洋红色键抠图和边缘去色溢；
- 提取青色队色遮罩并中和基础贴图中的占位色；
- 切分 2×2 图、统一到 `512×512`、对齐 `y=488` 脚底基线；
- 输出 21 个姿势及对应遮罩；
- 在 `character_manifest.json` 中记录所有 48 张源图的 SHA-256。

旧 v1 母版保留为历史身份参考，但不进入运行时或 Web 包。
