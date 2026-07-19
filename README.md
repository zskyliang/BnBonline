# BnBonline Godot

基于 Godot 4.6 重写的泡泡堂风格本地对战游戏。项目使用纯 GDScript，启动后直接进入玩家对规则 AI 的五分钟比赛。

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
- `Esc`：暂停或继续

右侧面板可以调整地图、AI 数量、角色速度/泡泡数/威力上限和玩家泡泡皮肤。地图或 AI 数量改变后会立即重开本局，其他设置即时生效并保存到 `user://settings.cfg`。

## 游戏内容

- 经典地图与风车爱心地图
- 0～4 名使用 `AStarGrid2D` 寻路的规则 AI
- 泡泡连锁爆炸、箱子破坏和泡泡数/速度/威力强化
- 半身安全、连续两帧命中、困泡、自救、接触击杀与复活规则
- 五分钟计时、击杀排行、胜利与平局结算

详细规则见 [docs/GAME_RULES.md](docs/GAME_RULES.md)。

## 验证

```bash
godot --headless --path . --import --quit
godot --headless --path . --script res://tests/test_runner.gd
godot --headless --path . --script res://tests/smoke_runner.gd -- --mute
```

项目结构采用组合场景：比赛控制器负责流程，地图、角色、泡泡、爆炸、AI 和 HUD 各自维护单一职责，并通过类型化信号通信。
