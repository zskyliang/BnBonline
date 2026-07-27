# 森林泡泡染色战

基于 Godot 4.6 的 2.5D 手绘动物染色占格闯关游戏。玩家选择一只动物和阵营颜色，
在两分钟内用水泡爆炸覆盖地板，对抗共用另一种颜色的 AI 队。

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
- 鼠标滚轮或 `+` / `-`：缩放地图
- 数字 `0`：恢复 110% 缩放
- 对局使用固定正交镜头（方位角 `0°`、俯角 `54°`），棋盘前边保持水平
- 对局右上角“设置”：打开缩放设置并暂停对局
- `Esc`：暂停或继续

## 染色闯关

- 大厅可从猫、狗、兔、熊、狐狸、浣熊、企鹅、水豚和七种局部队色中选择。
- 竞技场为完整 15×13 地板网格，不含障碍、建筑或箱子。
- 第 10 秒起每隔 10 秒随机生成速度、水泡数或威力道具；本关内可无限叠加，
  复活后保留，结算或重试时清空。
- 水泡、爆炸和被染地板都使用放泡者的阵营颜色；未锁定地板可以反复争夺。
- 困泡 3 秒后会自动脱困；只有敌方角色接触撞破困泡才算击败，并把该位置
  九宫格永久锁为击败方颜色。永久格只叠加半透明阴影。
- 每关 2 分钟，玩家覆盖数严格高于 AI 队才可晋级；平局或失败会重试本关。
- 胜利后从速度 `+10px/s`、水泡数 `+1`、威力 `+1` 中选择一个技能点。
- 第 1～4 关依次加入 1～4 名 AI，之后保持 4 名；每名 AI 每关按玩家累计点数
  重新随机属性分配。
- 闯关成长只保留到返回大厅或关闭游戏，不写入本地存档。

详细规则见 [docs/GAME_RULES.md](docs/GAME_RULES.md)。

## 验证

```bash
godot --headless --path . --import --quit
godot --headless --path . --script res://tests/test_runner.gd
godot --headless --path . --script res://tests/storybook_asset_runner.gd -- --mute
godot --headless --path . --script res://tests/storybook_visual_runner.gd -- --mute
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
只订阅状态并渲染 Sprite3D/基础 Mesh 表现。195 个地板和永久格阴影分别使用固定 `MultiMesh`，
染色时只更新受影响的实例。AI 使用只读战场快照、时间扩展寻路和错峰调度，
在确保逃生的前提下优先争抢可安全回本的道具，再比较即时染色和击败收益，
并通过道具认领避免多个 AI 重复追逐。已有水泡且仍有容量时，AI 会在每段
安全逃生路线末端优先继续放泡，主动把可用槽位压到 0～1 个。

八角色、三种道具、水泡爆炸、六种地砖、森林植被、绘本 UI 与大厅均由内置
ImageGen 原创生成，并经确定性本地脚本完成洋红抠图、边缘去色、队色遮罩和图集
切分。运行时成品位于 `assets/art/storybook25d/`，完整提示词、原始生成图、校验值
和验收快照位于 `art/reference/sprite25d/`，不会打入 Web 包。角色以单个
Sprite3D 表现 `Idle / WalkUp / WalkDown / WalkLeft / WalkRight / Trapped`
六种逻辑动作；四个移动方向分别使用独立生成的 4 帧、8 FPS 图像。上移展示
完整背面，左右为独立侧面；移动帧保持双臂张开、左右脚交替落地，并叠加约
±6° 重心摆动。停止后保持最后朝向，受困统一切换正面。旧 Blender 源文件和 GLB
已迁入 `art/blender/` 作为不可导出的历史归档，
不再参与运行时加载。对局每 10 秒在不同可用格随机刷新 3 个道具；八只动物
分别使用猫耳、垂耳、兔耳、爪印、狐尾、浣熊眼罩、企鹅鳍和水豚叶片等专属
ImageGen 水泡，视觉宽度不低于单格的 80%；三种道具也统一放大到至少 80%
单格宽度。棋盘外松树、灌木、蘑菇、花丛和薰衣草均以四张独立重绘帧播放，
速率为初版的三分之一，并为每株植物设置独立时间相位。对局循环播放用户提供
并重新制作无缝衔接的 `Puddle Jumpers`，下方叠加 CC0 轻雨和远雷环境循环。
背景音乐、环境音和音效的来源及处理记录保留在
`assets/audio/THIRD_PARTY_ASSETS.md`。对局角色使用独立于大厅预览的统一
战斗缩放，八只动物四向静止姿态的最窄可见宽度也不低于单格的 80%，碰撞与
占格规则保持不变。角色属性统一限制为速度最高 `300px/s`、同时存在的水泡数
最高 `10`、爆炸威力最高 `10`；闯关技能点、关内道具和直接数值写入都会经过
同一上限约束。中文字体为 OFL
授权的 Noto Sans SC 项目字符子集；新增运行时文案后执行：

```bash
python3 tools/generate_font_subset.py --source /path/to/NotoSansSC[wght].ttf
```
