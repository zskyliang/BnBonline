# CrazyGames 商店素材

更新日期：2026-07-27

## 可直接上传的文件

### Covers

| 文件 | 尺寸 |
| --- | ---: |
| `covers/forest-bubble-paint-battle-landscape-1920x1080.png` | 1920 × 1080 |
| `covers/forest-bubble-paint-battle-portrait-800x1200.png` | 800 × 1200 |
| `covers/forest-bubble-paint-battle-square-800x800.png` | 800 × 800 |

### Preview videos

| 文件 | 规格 | 时长 | 大小 |
| --- | --- | ---: | ---: |
| `videos/forest-bubble-paint-battle-landscape-1920x1080.mp4` | H.264、1920×1080、60 FPS、无声 | 18.03 秒 | 15,694,670 bytes |
| `videos/forest-bubble-paint-battle-portrait-1080x1620.mp4` | H.264、1080×1620、60 FPS、无声 | 18.03 秒 | 15,339,974 bytes |

两支视频都以前 1.5 秒静态封面开场，随后播放 16.5 秒真实自动试玩。没有快放、
音轨、鼠标光标、黑边、宣传文字、平台图标或过场 Logo。竖版从同一段真实
1920×1080 试玩中选择中央 600×900 动作区域并缩放到 1080×1620，使角色、
水泡和染色在竖屏缩略图中保持清楚。

## ImageGen 制作记录

使用内置 ImageGen，根据游戏最终 Web 构建的大厅和战斗截图生成；没有使用
CLI/API fallback。ImageGen 只负责无文字主视觉，准确标题
`Forest Bubble Paint Battle` 随后使用项目中的 OFL
`NotoSansSC-BnB-Subset.ttf` 排版，避免生成式文字错误。

### Landscape prompt

```text
Use case: ads-marketing
Asset type: CrazyGames landscape game cover master artwork, 16:9
Input images: final lobby character/style reference and final gameplay/mechanics
reference.
Primary request: premium, eye-catching Forest Bubble Paint Battle cover that
faithfully shows the gray cat hero placing a glossy coral animal-shaped water
bubble, penguin and raccoon rivals, a water-splash cross painting cream tiles,
and turquoise versus lavender territory in the forest arena.
Style: 2.5D hand-painted watercolor/gouache children's storybook key art,
matching the real game.
Composition: dynamic isometric arena, readable at thumbnail size, blank pale
parchment title sign across the upper area.
Constraints: only real characters and mechanics; no weapons, UI, border, logos,
watermark, letters, numbers, pseudo-text, or cropped subjects.
```

### Portrait prompt

```text
Recompose the approved landscape master as a true 2:3 portrait cover. Preserve
the same gray cat, coral cat-ear bubble, penguin, raccoon, turquoise splash,
lavender territory, golden pickup, forest arena, palette, lighting and
watercolor/gouache style. Put a completely blank parchment title sign in the
upper 20%; make the cat and bubble large in the lower-middle. No bars, UI,
logos, watermark or generated text.
```

### Square prompt

```text
Recompose the approved landscape and portrait masters as a true 1:1 cover.
Preserve the same identities, coral bubble, splash, two territory colors,
forest arena, lighting and storybook aesthetic. Use a blank parchment title
sign in the upper 23%, with the cat and bubble as the largest lower focal
point. Keep the composition simple at thumbnail size. No bars, UI, logos,
watermark or generated text.
```

原始 ImageGen 无文字图保留在 `covers/source/`，便于以后重排标题或制作季节版。

## 试玩录制

`tools/record_crazygames_preview.mjs` 使用 Playwright 打开最终 CrazyGames ZIP，
等待 SDK 和大厅初始化，点击一键开局，并真实发送 WASD 与 Space 输入。游戏
画面通过浏览器 `canvas.captureStream(60)` 采集，不再使用固定约 25 FPS 的
Playwright 视频录制器。录制过程会响应实际碰撞、泡泡容量、引信、水花、AI
和占格规则。

源 WebM、最终状态和 QA 截图保留在 `videos/source/`。最终 MP4 已用
`ffprobe` 和完整解码验证：

- 只有视频流，没有音轨。
- 尺寸分别为 1920×1080 和 1080×1620。
- 编码帧率均为 60 FPS，时长均为 18.03 秒。
- 单个文件远低于 CrazyGames 的 50 MB 上限。

最终五个上传文件的校验值位于 `SHA256SUMS`。
