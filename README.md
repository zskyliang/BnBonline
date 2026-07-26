# BnBonline Godot

基于 Godot 4.6 的单机染色占格闯关游戏。玩家选择一名软陶角色和阵营颜色，
在三分钟内用水泡爆炸覆盖地板，对抗共用另一种颜色的 AI 队。

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
- 按住鼠标右键拖动：调整水平角与俯视角
- 鼠标滚轮或 `+` / `-`：缩放地图
- 数字 `0`：恢复推荐角度与 110% 缩放
- 对局右上角“设置”：打开镜头设置模态框并暂停对局
- `Esc`：暂停或继续

## 染色闯关

- 大厅可从 8 名角色和红、橙、黄、绿、青、蓝、紫七种服装颜色中选择。
- 竞技场为完整 15×13 地板网格，不含障碍、建筑或箱子。
- 第 10 秒起每隔 10 秒随机生成速度、水泡数或威力道具；本关内可无限叠加，
  复活后保留，结算或重试时清空。
- 水泡、爆炸和被染地板都使用放泡者的阵营颜色；未锁定地板可以反复争夺。
- 困泡 3 秒后会自动脱困；只有敌方角色接触撞破困泡才算击败，并把该位置
  九宫格永久锁为击败方颜色。永久格只叠加半透明阴影。
- 每关 3 分钟，玩家覆盖数严格高于 AI 队才可晋级；平局或失败会重试本关。
- 胜利后从速度 `+10px/s`、水泡数 `+1`、威力 `+1` 中选择一个技能点。
- 第 1～4 关依次加入 1～4 名 AI，之后保持 4 名；每名 AI 每关按玩家累计点数
  重新随机属性分配。
- 闯关成长只保留到返回大厅或关闭游戏，不写入本地存档。

详细规则见 [docs/GAME_RULES.md](docs/GAME_RULES.md)。

## 验证

```bash
godot --headless --path . --import --quit
godot --headless --path . --script res://tests/test_runner.gd
godot --headless --path . --script res://tests/character_clay_runner.gd -- --mute
godot --headless --path . --script res://tests/smoke_runner.gd -- --mute
godot --headless --path . --script res://tests/ai_benchmark_runner.gd -- --mute
godot --headless --path . --script res://tests/ai_item_training_runner.gd -- --mute
godot --headless --path . --script res://tests/ai_stress_runner.gd -- --mute
```

Web Release：

```bash
godot --headless --path . --export-release Web build/web/index.html
```

## 结构

二维逻辑层负责移动、爆炸时间线、道具、染色归属、锁定状态和闯关成长；
`BoardView3D`、`ActorView3D`、`BubbleView3D`、`ItemView3D` 与 `ExplosionView3D`
只订阅状态并渲染 3D 表现。195 个地板和永久格阴影分别使用固定 `MultiMesh`，
染色时只更新受影响的实例。AI 使用只读战场快照、时间扩展寻路和错峰调度，
在确保逃生的前提下优先争抢可安全回本的道具，再比较即时染色和击败收益，
并通过道具认领避免多个 AI 重复追逐。已有水泡且仍有容量时，AI 会在每段
安全逃生路线末端优先继续放泡，主动把可用槽位压到 0～1 个。

角色 GLB 来自 Quaternius Ultimate Animated Character Pack，道具来自 CC0 的
Kenney Platformer Kit；背景音乐改用 OpenGameArt 的萌系 CC0 无缝循环，放泡、
出现和爆炸使用柔和的水泡/短促 pop，结算与交互使用 Kenney 及 OpenGameArt
的拨弦短音。许可记录保留在各自的 `THIRD_PARTY_ASSETS.md`。中文字体为 OFL
授权的 Noto Sans SC 项目字符子集；新增运行时文案后执行：

```bash
python3 tools/generate_font_subset.py --source /path/to/NotoSansSC[wght].ttf
```
