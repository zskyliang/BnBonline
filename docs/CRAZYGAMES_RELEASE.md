# CrazyGames 发布手册

更新日期：2026-07-27

## 当前结论

游戏代码和 Web 发布包已经可以进入 CrazyGames Developer Portal 的 Preview
测试。当前最稳妥的首发方案是：

- 发布阶段：Basic Launch
- 设备：Desktop only
- 屏幕方向：Landscape
- 输入：Keyboard + Mouse
- 语言：English、简体中文；English 为默认回退
- Progress Save：No
- Multiplayer：No
- Ads：Basic Launch 阶段不接广告

当前不建议勾选 Mobile：游戏还没有触屏移动/放泡操作，而且 36.06 MiB 的估算
首屏传输量高于移动首页推荐的 20 MiB 门槛。

## 已完成的适配

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| CrazyGames SDK | 已完成 | 集成官方 Godot 4 SDK v1.0.2，使用 v3 Web SDK |
| Gameplay lifecycle | 已完成 | 对局开始/恢复报告 `gameplayStart`；大厅、设置、暂停、结算报告 `gameplayStop` |
| 平台静音 | 已完成 | 启动与运行中均响应 `muteAudio`，平台静音优先 |
| 平台语言 | 已完成 | 首次启动读取 SDK locale；用户已保存的语言选择优先 |
| 一次点击开局 | 已完成 | Web 大厅主按钮直接开始；角色和队色保留为独立入口 |
| 静态游戏首屏 | 已完成 | Godot 原生版和 Web 版游戏内大厅共用指定的 1920×1080 壁纸，不再创建动态动物 SubViewport |
| 静态角色预览 | 已完成 | 角色选择页固定渲染各动物 IdleDown 第 1 帧，仅在切换队色时重绘 |
| iframe/缩放 | 已完成 | 响应容器尺寸、禁用页面滚动/选择/右键菜单，无自定义全屏按钮 |
| 失焦处理 | 已完成 | 页面隐藏时暂停并静音，恢复后由玩家手动继续 |
| Web 兼容性 | 已完成 | Compatibility renderer、Web threads disabled、无 PWA |
| 包体优化 | 已完成 | 彩色角色纹理采用 Web 有损压缩，遮罩继续无损 |
| 字体 | 已完成 | 字体子集同时扫描代码和本地化 CSV，英文/中文符号完整 |
| 自动打包 | 已完成 | 构建、校验限制、生成 ZIP 一条命令完成 |
| 三种商店封面 | 已完成 | ImageGen 主视觉，标题使用项目 OFL 字体精确排版 |
| 两支 Preview Video | 已完成 | 18.03 秒、1080p、60 FPS、H.264、无声、正常速度、无黑边 |

## 构建和上传

在项目根目录执行：

```bash
./tools/package_crazygames.sh
```

每次修改代码或资源后重新执行同一条命令即可。脚本会按顺序完成资源导入、
规则检查、资源检查、视觉检查、完整 Smoke Test、Release 导出、CrazyGames
限制校验和 ZIP 完整性检查；任何一步失败都不会覆盖上一次成功生成的发布包。
成功后终端会打印 ZIP 路径、大小和 SHA-256。

输出文件：

```text
build/crazygames-upload.zip
```

2026-07-27 实测：

| 指标 | 实测 | CrazyGames 限制 |
| --- | ---: | ---: |
| 文件数 | 9 | 1500 |
| 解压后总大小 | 65.47 MiB | 250 MiB |
| 估算 gzip 首屏传输 | 36.06 MiB | 50 MiB |
| 上传 ZIP | 约 37 MiB | — |

脚本会在超限、导出缺文件或 ZIP 损坏时直接失败。上传 ZIP 后，入口文件是包根
目录的 `index.html`。本次发布包 SHA-256 为
`19c35f1709cbdf6367fdbaff761e468f4edc5aba1a89892063bef4c4345825de`。

CrazyGames Developer Portal 在游戏加载前可能显示平台侧上传的商店封面；
这是平台页面，不属于 Godot 场景。引擎加载完成后，Godot 原生版和 Web 版的
游戏内首屏现已使用同一张静态壁纸。

## Web 性能结论

先前 Preview Video 中显示的约 18–26 FPS 不是 CrazyGames 发布包本身的
性能，而是旧录制流程强制 Chrome 使用 SwiftShader 软件渲染造成的。正常的
CrazyGames 页面会使用玩家浏览器的硬件 WebGL；只有浏览器关闭 GPU、驱动被
禁用或回退到软件渲染时，才可能出现类似低帧率。

对本手册所列最终 ZIP 的 1920×1080 Chrome 实测如下：

| 场景 | WebGL 后端 | 稳态平均 FPS | 稳态最低 FPS | 最大 Draw calls | 最大对象数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 正常硬件加速 + 同时录制 | Apple M3 Pro / Metal | 60.00 | 60 | 76 | 342 |
| CPU 人为降速 4 倍 + 同时录制 | Apple M3 Pro / Metal | 59.89 | 59 | 74 | 350 |

每组稳态结果包含 62 个、间隔 250 ms 的样本，排除对局切入和录制器初始化的
前 2 秒。两次完整自动试玩均无浏览器控制台错误。Web 状态探针和 FPS HUD
每 250 ms 更新一次，状态快照仅作浅拷贝，不会逐帧进行深拷贝或 JavaScript
序列化。这些结果说明当前版本在硬件加速浏览器中有充足的 60 FPS 帧预算；
但它不能替代 CrazyGames Portal Preview 和真实低配 Chromebook 的最终验收。

## Developer Portal 推荐填写值

| 后台字段 | 建议值 |
| --- | --- |
| Title | Forest Bubble Paint Battle |
| Orientation | Landscape |
| Devices | Desktop |
| Mobile support | No |
| Multiplayer | No |
| Progress Save | No |
| Languages | English, Chinese (Simplified) |
| Controls | Keyboard, Mouse |
| Engine | Godot |

`Progress Save` 选择 No 是有意的：当前只持久化音量、语言和镜头等偏好，
闯关成长按设计只存在于本次运行内。不要仅仅为了勾选该选项而混用浏览器本地
存储和 CrazyGames Data。

### Short description

> Paint the forest arena in two-minute bubble battles. Trap rivals, lock
> territory, collect upgrades, and survive an endless animal adventure.

### Long description

> Choose a charming forest animal and a team color, then race across a
> storybook arena in fast two-minute paint battles. Place bubbles to cover the
> floor, steal open territory, trap rival animals, and lock defeated spaces for
> your team. Collect speed, capacity, and power upgrades during each round.
> Win by painting more cells than the rival team, choose a permanent run
> upgrade, and face a larger AI team in the next stage. How far can your animal
> hero go?

### Controls

> WASD or Arrow Keys: move  
> Space: place a bubble  
> Mouse Wheel or +/-: zoom  
> 0: reset zoom  
> Esc: pause or resume

## 已完成的商店素材

所有素材位于 `marketing/crazygames/`，制作方式和提示词见该目录的
`README.md`。

### 1. 商店封面

| 用途 | 文件 | 尺寸 |
| --- | --- | ---: |
| Landscape | `covers/forest-bubble-paint-battle-landscape-1920x1080.png` | 1920 × 1080 |
| Portrait | `covers/forest-bubble-paint-battle-portrait-800x1200.png` | 800 × 1200 |
| Square | `covers/forest-bubble-paint-battle-square-800x800.png` | 800 × 800 |

三张图使用相同的猫主角、珊瑚色动物水泡、企鹅/浣熊对手、青色/紫色占格和
水花十字爆炸。除准确的英文游戏名外没有其他文字、平台徽标、边框或 UI。

### 2. Preview videos

| 用途 | 文件 | 规格 | 大小 |
| --- | --- | --- | ---: |
| Landscape | `videos/forest-bubble-paint-battle-landscape-1920x1080.mp4` | 1920×1080、60 FPS、18.03 秒 | 约 15.0 MiB |
| Portrait | `videos/forest-bubble-paint-battle-portrait-1080x1620.mp4` | 1080×1620、60 FPS、18.03 秒 | 约 14.6 MiB |

两支视频均为 H.264 / yuv420p / 60 FPS，仅含视频流。前 1.5 秒使用对应封面，
后 16.5 秒来自自动实际试玩，展示移动、连续放泡、水花爆炸、双色染色和道具。
没有声音、快放、鼠标光标、黑边、宣传字幕或平台图标。竖版使用同一段真实
试玩的中央动作区域重新构图，没有伪造游戏画面。

## 仍需在 Portal 人工执行

### Portal Preview 验收

上传 `build/crazygames-upload.zip` 后，至少完成以下检查：

- 在 Developer Portal 的 Preview 中完整走一遍：加载、大厅、一键开局、暂停、
  设置、恢复、结算、再开一局。
- 在 SDK 工具中确认真正进入/恢复对局时触发 gameplay start，暂停、设置、大厅
  和结算时触发 gameplay stop。
- 切换平台的声音设置，确认 `muteAudio` 会立即静音且不会被游戏设置反向覆盖。
- Chrome 和 Edge 各测一次；在 `chrome://gpu` 确认 WebGL 为硬件加速，而非
  SwiftShader。再用一台 4 GB 内存 Chromebook 或相近低配设备测加载时间、
  帧率和长局稳定性。
- 低配机至少连续玩 10 分钟，并在爆炸、道具和多 AI 同屏时确认 HUD 大部分时间
  保持 55–60 FPS；若明显低于该范围，记录设备型号、浏览器版本和
  `chrome://gpu` 后再决定是否降低雨滴/涟漪密度。
- 检查常见 iframe 尺寸，尤其是 1280×720、1216×684、1077×606、907×510、
  821×462 和 800×450，确认按钮、HUD 和设置弹窗不被裁切。
- 确认所有资源路径为相对路径，浏览器控制台没有 404、JavaScript 异常或音频
  自动播放错误。
- 首次提交仅勾选 Desktop，不要声明 Mobile 或触屏支持。

## Basic Launch 之后

### Full Launch / 广告

Basic Launch 的广告不会启用，因此当前版本没有放置一个无法工作的广告按钮。
进入 Full Launch 前，再在自然断点（建议关卡结算后、下一关开始前）接入
CrazyGames SDK midgame ad，并做到：

- 广告期间暂停游戏并静音。
- 广告完成、跳过或报错后都能恢复。
- 不调用外部广告 SDK，不显示自定义 preroll。
- 重新用 SDK 工具验证 gameplay 和 ad 的调用顺序。

### 云端进度

如果以后要让无限闯关跨设备续玩，应将整个进度存档迁移到 CrazyGames Data，
包括关卡、成长点、角色/队色和必要的运行状态，然后才在后台把 Progress Save
改为 Yes。不要只同步其中一部分。

### 移动端

若以后要上移动端，需要同时完成：

- 屏幕虚拟方向键或摇杆、放泡按钮、暂停入口和触摸反馈。
- 触控下的设置/角色选择可用性与多尺寸竖横屏测试。
- 继续压缩或拆分首屏资源，将移动首页首屏传输控制到 20 MiB 以内。
- 鼠标、键盘和触摸三套输入都通过回归后，才在 Portal 勾选 Mobile。

## 官方依据

- [Technical requirements](https://docs.crazygames.com/requirements/technical/)
- [Gameplay requirements](https://docs.crazygames.com/requirements/gameplay/)
- [Game covers](https://docs.crazygames.com/requirements/game-covers/)
- [SDK introduction](https://docs.crazygames.com/sdk/intro/)
- [Game module](https://docs.crazygames.com/sdk/game/)
- [Video ads](https://docs.crazygames.com/sdk/video-ads/)
- [Data module](https://docs.crazygames.com/sdk/data/)
