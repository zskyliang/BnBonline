# BnBonline Godot

基于 Godot 4.6 重写的泡泡堂风格本地对战游戏。项目使用纯 GDScript，
保留原有二维规则逻辑，并以水平角 `-5°`、俯角 `38°` 的轻微斜视固定
正交镜头呈现软陶 3D 海岛，使四方向操作接近平面游戏的上下左右。

## 运行

需要 Godot 4.6 或兼容的 Godot 4.x 版本：

```bash
godot --path .
```

打开编辑器：

```bash
godot --editor --path .
```

## 操作

- 方向键或 `WASD`：移动
- 空格：放置水泡
- 数字 `1`：被困泡时自救
- 鼠标滚轮或 `+` / `-`：放大、缩小地图（85%～150%）
- 数字 `0`：恢复推荐的 110% 地图缩放
- `Esc`：暂停或继续

启动流程为“大厅 → 8 名黏土角色选角与对局设置 → 比赛 → 结算”。赛前可选择
“软陶海岛集市”和“钟楼花园”、0～4 名 AI、角色速度/水泡数/威力上限，以及
“海盐蓝”和“珊瑚橙”两种水泡皮肤。设置在桌面版
保存到 `user://settings.cfg`，Web 版保存到 `localStorage` 的
`bnb.settings.v3`。角色只影响外观，不影响任何属性或技能。

## 游戏内容

- 15×13 网格逻辑映射到 3D XZ 平面的软陶海岛集市与钟楼花园
- Kenney/Quaternius CC0 建筑和自然单元，经统一黏土材质包装后组成固定手工地图
- 8 个可选的 CC0 黏土化低模角色，最多 4 名 AI 随机使用其余不重复角色
- 0～4 名使用爆炸时间线与时间扩展寻路的规则 AI
- 三阶段软陶水花/泡沫连锁爆炸、摊位破坏和水泡数/速度/威力强化
- 半身安全、连续两帧命中、困泡、自救、接触击杀与复活规则
- 五分钟计时、击杀排行、胜利与平局结算
- 由实际位移驱动且显式循环的 12 FPS 左右脚交替步态
- 默认 110% 的正交地图缩放，以及 HUD、滚轮和键盘缩放控制
- 顶部实时 FPS 显示，便于观察多人对局性能
- Compatibility/WebGL 2 渲染器、60 FPS 上限与物理插值

详细规则见 [docs/GAME_RULES.md](docs/GAME_RULES.md)。

## 验证

```bash
godot --headless --path . --import --quit
godot --headless --path . --script res://tests/test_runner.gd
godot --headless --path . --script res://tests/character_clay_runner.gd -- --mute
godot --headless --path . --script res://tests/smoke_runner.gd -- --mute
godot --headless --path . --script res://tests/ai_benchmark_runner.gd -- --mute
godot --headless --path . --script res://tests/ai_stress_runner.gd -- --mute
```

Web Release 导出：

```bash
godot --headless --path . --export-release Web build/web/index.html
```

AI 实战评测在保持每个模拟物理步约 16.7ms 的前提下加速运行，并以非零退出码阻断回归。固定门槛为：40 个不同引信多水泡场景生存率不低于 95%，30 个同速度道具竞速场景胜率不低于 70%，30 个开阔区、通道和死角场景的复合压迫成功率不低于 80%、进攻存活率不低于 90%。压迫场景还要求至少 85% 出现两颗以上 AI 水泡并存，并且至少 80% 的峰值加权威胁达到最佳单泡的 1.5 倍。评测会输出峰值水泡数、等效威胁格、目标覆盖率、困泡/击杀率以及单次决策平均和 P95 耗时；性能目标分别低于 5ms 和 10ms。

多人压力评测会在同一局中运行 4 个 AI，检查每个 AI 仍保持约 150ms 的决策周期，同时普通情况下每个物理帧最多派发一次 AI 决策。它会输出单次决策平均/P95、决策活跃帧的 P95/P99 AI 计算峰值、同帧聚集次数和各 AI 的调度公平性；硬门槛为单次平均低于 5ms、P95 低于 10ms、P99 帧内 AI 峰值低于 16.67ms，且不得出现普通决策同帧聚集。评测还包含无危险大空间的连续走位探针，要求移动帧占比不低于 78%，并禁止出现 A→B→A 的即时格子折返。

项目结构采用逻辑与表现分离：`GameActor`、`GameBoard`、`GameBubble` 和
`ExplosionEffect` 保持二维坐标、碰撞与数值规则；`BoardView3D`、
`ActorView3D`、`BubbleView3D` 和 `ExplosionView3D` 只读取状态并映射到
3D 世界，不参与判定。重复地面、摊位和道具采用 MultiMesh，爆炸视图由对象池
回收。AI 使用只读战场快照，
危险预测、时间寻路和威胁场均为无渲染依赖的纯逻辑模块；多人对局仍由比赛控制器
错峰调度 150ms 决策周期。

角色 GLB 来自 Quaternius Ultimate Animated Character Pack，采用 CC0 许可。完整来源、下载日期
和 SHA-256 见
[assets/models/characters/THIRD_PARTY_ASSETS.md](assets/models/characters/THIRD_PARTY_ASSETS.md)。
中文字体为 OFL 授权 Noto Sans SC 的项目字符子集，许可文件位于
`assets/fonts/OFL-NotoSansSC.txt`。
新增运行时文案后，可使用
`python3 tools/generate_font_subset.py --source /path/to/NotoSansSC[wght].ttf`
重新生成只包含项目界面字符的 Web 字体。

地图建筑和自然模型的作者、来源、CC0 许可、处理记录及 SHA-256 见
[assets/models/environment/THIRD_PARTY_ASSETS.md](assets/models/environment/THIRD_PARTY_ASSETS.md)。
