# Sprite3D Production v1

- 生成日期：2026-07-26
- 生成器：Codex 内置 ImageGen
- 风格锚点：用户提供的呆萌动物设计图
- 构图锚点：`source/forest-board-composition-reference.png`
- 运行时处理：`tools/art_pipeline/process_sprite25d_assets.py`
- 色键：纯洋红 `#FF00FF`
- 队色占位：纯青色 `#00B7C7`

本地脚本会在保留整张 Waddle 回退图的同时，将其确定性拆为 `body`、
`left_arm`、`right_arm`、`left_foot`、`right_foot` 五张同锚点透明图及对应
队色遮罩。运行时双脚独立绕踝部旋转 `18°` 并交替抬高 `0.08` 单位。

## 八角色六动作表

对猫、狗、兔、熊、狐狸、浣熊、企鹅、水豚分别执行同一模板，并保持已确认
身份特征：

> Create a clean 3 by 2 production sprite sheet for the specified upright
> forest animal. Match the supplied hand-painted storybook identity: 1:1
> head-to-body ratio, round belly, very short legs, oversized eyes, tiny pupils,
> slightly unfocused innocent gaze, charcoal ink outline and soft watercolor.
> Full body and front-facing in every cell, identical scale and foot baseline,
> no text, no labels, no watermark, no shadows outside the character. Exact
> cell order: Idle; Waddle with both arms spread widely for toddler balance and
> one short foot visibly lifted; PlaceBubble; Trapped; Defeat; Victory. Use a
> perfectly flat #FF00FF background. Paint only the intended local team-color
> patches in flat #00B7C7; never mark eyes, pupils, muzzle, belly or species
> identity markings. Keep every pose isolated inside its cell.

原始输出：

- `source/cat-actions-v1.png`
- `source/dog-actions-v1.png`
- `source/rabbit-actions-v1.png`
- `source/bear-actions-v1.png`
- `source/fox-actions-v1.png`
- `source/raccoon-actions-v1.png`
- `source/penguin-actions-v1.png`
- `source/capybara-actions-v1.png`

## 森林环境

> Create a 4 by 3 sheet of twelve isolated hand-painted watercolor storybook
> game assets on perfectly flat #FF00FF: cream irregular floor tile, carved
> wooden rail, wooden corner block, evergreen tree, rounded bush, red-and-blue
> mushrooms, moss patch, tree stump, blank wooden sign, small rocks, wildflower
> cluster, lavender cluster. Front-isometric presentation consistent with the
> supplied forest-board composition, charcoal outlines, clean separation,
> no text, no watermark and no cast shadows crossing cell boundaries.

草地：

> Create one square seamless watercolor grass-and-moss texture in the approved
> cozy forest storybook palette. Subtle paper grain, sparse tiny leaves, no
> flowers larger than a floor tile, no paths, objects, borders, text or
> watermark. Tile perfectly on every edge.

## 道具与水泡特效

> Create a 4 by 2 sheet of eight isolated front-isometric watercolor game
> sprites on perfectly flat #FF00FF: winged leaf shoes, double-bubble gourd,
> golden paw-print burst star, leaf-stopper soap bubble, soft oval trapped
> bubble, compact pop core, cross-shaped splash, foam-petal burst. Bold charcoal
> contour, clean readable silhouette at small size, no text or watermark.
> Where team tint is intended use only flat #00B7C7 and keep ink/highlights
> opaque and color-neutral.

## 绘本 UI

> Create a 3 by 2 sheet of six isolated watercolor storybook UI pieces on
> perfectly flat #FF00FF: warm-white paper nine-slice panel, normal button,
> hover button, pressed button, paper badge and leafy divider. Charcoal pencil
> borders, blank interiors, no words, symbols, watermark or drop shadows beyond
> each piece. Corners must remain suitable for nine-slice scaling.

## SHA-256

```text
6b15f55e8c1955545b1fd130b363d191e9a81cf3982ba54bb2110d37d0e7b17b  bear-actions-v1.png
e6b1f9b204123de071180ae7ac78e656c8810e6c8c7f8698933b9400d3b86a6b  capybara-actions-v1.png
40efeb0b5fd5577f25ff2cd6aa5a21a9905ead78d13d28e8a933a9171a04577a  cat-actions-v1.png
e6e9f816d4ba246d1f917b9b985f4391f9f29512f094689e726e6c3f29254003  dog-actions-v1.png
850dccb3985fb459d6f90b0e3506c37570de2e1121419495876392562a5f216d  environment-assets-v1.png
7f5280b80e145dc6e5e8b7aa8f0b46159efa7c668e3582ecb0cad75b4e8f1bb3  fox-actions-v1.png
ea9733f3f048f7a224f08f46fe555baeac87ad2eed3ade520394be4575da1373  grass-seamless-v1.png
44cd663ba7e77988f0ceb944e83cfcc6a7928a90ced47c525b789479f228b9e0  items-effects-v1.png
e71db9805f3f6a25a219c8c3a43fea7de4261b3b888480cc7f42528c2c12c03e  penguin-actions-v1.png
53500d3d012fd8186aeb805ac97a928bab7e6c681ca0baf22334f4eb0e475ce8  rabbit-actions-v1.png
601fd9e7da4e66b47de0debcd982727a19f13c44278c5a98935f1df8f7ce7a1f  raccoon-actions-v1.png
ac78383d9b097b7e36179976819e3fdfc81201e2fa98c5d494ace9d58cbae7a0  ui-kit-v1.png
```
