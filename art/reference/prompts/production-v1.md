# ImageGen 正式生产提示词与版本记录

- 生成日期：2026-07-26
- 版本：production-v1
- 生成器：Codex 内置 ImageGen（stylized-concept）
- 风格参考：`../source/animal-style-reference.png`
- 已批准母版：`../pilot/animal-roster-style-master.png`
- 用途：原创 Blender MCP 建模参考，不直接作为游戏贴图
- 状态：完成并进入 Blender 正式生产

## 六角色建模参考

六张角色图使用下列完整固定提示词，仅替换 `{species}` 和
`{species_details}`：

```text
Use case: stylized-concept
Asset type: Blender character modeling and animation reference sheet
Input images: the uploaded image is art-direction reference only. Create an original {species} design in the approved Forest Bubble Paint storybook language; do not copy an existing copyrighted character or the reference layout.
Primary request: Show one internally consistent upright {species} in front orthographic, right-side orthographic, back orthographic, front-right three-quarter view, plus four small readable action poses: Idle, Waddle, PlaceBubble, and Victory.
Scene/backdrop: clean warm off-white handmade paper, completely uncluttered.
Style/medium: hand-drawn children's picture-book concept art, confident charcoal ink contour, subtle watercolor washes, visible paper grain, low-saturation warm palette.
Character language: head-to-body ratio about 1:1, huge round white eyes occupying much of the face, tiny dark pupils with a slightly unfocused and gently mismatched gaze, tiny mouth, round belly, very short thick legs, upright bipedal posture, clear silhouette.
Waddle pose requirement: both short arms spread broadly sideways like a toddler balancing, one foot visibly lifted and swung forward, the other planted, torso leaning over the planted foot; stiff, clumsy, dumb-cute movement rather than gliding.
Species details: {species_details}
Runtime recolor zones: show a pale desaturated blue placeholder only on ear tips or centers, forepaws, hind paws, selected tail bands, flipper edges, and/or a small upper-back patch as anatomically appropriate. Natural fur, eyes, pupils, belly, muzzle, nose, beak, and species-defining markings stay fixed.
Composition/framing: landscape model sheet, all views and poses fully visible at consistent scale and baseline, minimal perspective distortion, strong separation between figures.
Constraints: no text, labels, arrows, frame numbers, UI, clothes, weapons, unrelated props, scenery, logos, watermark, extra limbs, cropped ears/paws/tail, realistic anatomy, anime style, or glossy 3D rendering.
```

物种变量：

- `dog`：暖黄褐毛、深棕下垂耳、奶油口鼻和肚皮、小黑鼻、短弯尾。
- `rabbit`：奶油米色毛、两只很长的直耳和粉色内耳、圆脸、白肚皮、绒球尾。
- `fox`：暖橙毛、尖耳、奶油脸颊和肚皮、深色鼻、大而蓬松的弯尾。
- `raccoon`：灰毛、深色眼罩、圆耳、奶油口鼻、蓬松环纹尾。
- `penguin`：深炭黑背部、奶油白脸和肚皮、橙色喙和脚、短鳍代替手臂。
- `capybara`：暖棕毛、宽钝口鼻、小圆耳、短尾、平静而错焦的眼神。

| 文件 | 尺寸 | SHA-256 |
| --- | --- | --- |
| `../production/dog-modeling-reference-v1.png` | 1698×926 | `911dbc9e4ceb83057a9555a83a45d62f4aafe40701eb61fb6f486de2b4776b42` |
| `../production/rabbit-modeling-reference-v1.png` | 1694×929 | `c2c3976b065d399f5e4ffb92788acf106cdc370f8fe73fa2b63941d0373c0c67` |
| `../production/fox-modeling-reference-v1.png` | 1698×926 | `386857c94a0558693e539893e81b68553de1575794a1337493d523ee599d17d9` |
| `../production/raccoon-modeling-reference-v1.png` | 1694×929 | `159fd663149638b23b7655183fb84af384c85650b8e1bd06bb5980b5567b8433` |
| `../production/penguin-modeling-reference-v1.png` | 1693×929 | `fe6854af932204d0009b4d35e5be569f67eda15e5ae11cba3bf90fec950b10ff` |
| `../production/capybara-modeling-reference-v1.png` | 1693×929 | `8208704c66c529230a6de4b7e42df3e813da92213ed6ac457799edbbb6c409da` |

## 三道具

```text
Use case: stylized-concept
Asset type: Blender game-prop modeling reference sheet
Primary request: Design exactly three original collectible props at matching scale: a speed pickup shaped as a pair of green leaf shoes with tiny cream wings; a bubble-capacity pickup shaped as a double-sphere translucent soap-bubble gourd with a leaf cork; and a power pickup shaped as a golden paw-print burst star.
Style/medium: warm off-white handmade paper, charcoal ink contours, subtle watercolor wash, visible paper grain, simple low-poly-friendly volumes, readable from a fixed 2.5D orthographic game camera.
Composition: each prop shown in front three-quarter view plus one small side view; generous spacing; clear silhouette.
Constraints: no characters, text, numbers, labels, UI, logos, watermark, photorealism, glossy product rendering, weapons, scenery, or cropped objects.
```

- 文件：`../production/items-modeling-reference-v1.png`
- 尺寸：1698×926
- SHA-256：`070f113880e2892d815ed85d838c113abbfe12d720a697f05ae4e84c71f3b817`

## 水泡与爆炸

```text
Use case: stylized-concept
Asset type: Blender and Godot VFX shape reference sheet
Primary request: Design a team-tint soap bubble with a leaf cork, a larger soft oval trap bubble, and a harmless bubble explosion kit made from outlined foam petals, round droplets, and a clean four-direction cross burst. Show a compact life-cycle sequence from placed bubble to pressure wobble to pop.
Style/medium: children's picture-book ink and watercolor on warm off-white paper; bold charcoal silhouette; translucent pastel cyan with localized team-color accents; simple Compatibility-renderer-safe geometry.
Composition: separated modelable shapes and a small sequence row, no overlap.
Constraints: no fire, smoke, sparks, weapons, violence, characters, text, labels, UI, logos, watermark, photorealism, or screen-space glow.
```

- 文件：`../production/bubble-effects-reference-v1.png`
- 尺寸：1698×926
- SHA-256：`37193386848de1757b52295e0990a44ab12af1fa94bc1bca669d7a3d8be2b0bc`

## 森林棋盘

```text
Use case: stylized-concept
Asset type: 2.5D game-environment concept for Blender modeling
Primary request: Design a fixed-camera 15-by-13 obstacle-free paint arena with cream irregular hand-painted floor tiles, a low wooden board frame, moss, paper-cut trees, mushrooms, flowers, grasses, and small wooden signs. Every tall decoration must stay strictly outside the playable rectangle and must not cover grid lines, characters, or effects.
Camera: orthographic, azimuth -30 degrees, elevation 42 degrees, full playable grid visible with safe HUD margins.
Style/medium: warm children's forest picture book, charcoal ink outline, watercolor wash, visible handmade paper grain, low-poly-friendly layered shapes.
Composition: one clean gameplay concept view plus small isolated material/prop callouts without labels.
Constraints: no characters, text, UI, buildings, roofs, obstacles inside the grid, photorealism, glossy render, watermark, or cropped playable cells.
```

- 文件：`../production/forest-board-concept-v1.png`
- 尺寸：1672×941
- SHA-256：`eeab80a94a6778d05d0719cd269938af2ee22a56725b36f3a8c4b39e320c18b1`

## 绘本大厅

```text
Use case: stylized-concept
Asset type: 2.5D game-lobby concept for Blender modeling
Primary request: Design a welcoming forest storybook clearing used as the character-selection lobby. Include a cream paper ground, low mossy wooden platform, paper-cut trees, mushrooms, flowers, grasses, tiny bunting shapes, and eight evenly spaced circular character pedestals. Leave generous central and side negative space for runtime title, selection cards, and buttons.
Camera: fixed orthographic three-quarter view matching azimuth -30 degrees and elevation 42 degrees.
Style/medium: charcoal ink outlines, soft watercolor washes, handmade paper grain, muted warm palette, low-poly-friendly layered geometry.
Constraints: no characters, text, letters, UI, logos, buildings, enclosed roofs, photorealism, glossy 3D, watermark, or decorations blocking the pedestals.
```

- 文件：`../production/storybook-lobby-concept-v1.png`
- 尺寸：1672×941
- SHA-256：`c91f6b3b7542ece9f63ec449b08f708433927a8806ba1751792a9ec8c5875fc4`
