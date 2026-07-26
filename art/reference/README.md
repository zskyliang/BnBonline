# 《森林泡泡染色战》美术参考

这里保存只用于原创建模和风格校准的图片，不作为最终角色贴图，也不会进入 Web 发行包。

## 目录

- `source/animal-style-reference.png`：用户提供的风格参考，只用于提取“手绘、圆肚短腿、大眼呆萌”的视觉语言。
- `pilot/animal-roster-style-master.png`：八角色统一风格母版。
- `pilot/cat-turnaround.png`：猫的正面、右侧、背面和右前四分之三视图。
- `pilot/cat-animation-poses.png`：猫的六个共享动作关键姿势。
- `pilot/bear-turnaround.png`：熊的正面、右侧、背面和右前四分之三视图。
- `pilot/bear-animation-poses.png`：熊的六个共享动作关键姿势。
- `prompts/pilot-v1.md`：首轮 ImageGen 的完整提示词、版本和文件校验值。
- `production/`：其余六角色、三道具、水泡特效、森林棋盘和大厅的正式建模参考。
- `prompts/production-v1.md`：正式生产轮的提示词、版本和文件校验值。

## 当前验收状态

首轮 ImageGen 参考图已于 2026-07-26 通过验收门 1；猫、熊的 Blender MCP
模型已在真实战斗镜头完成自主验收，验收门 2 同日通过。其余六角色及全部世界
资产已进入正式生产并完成运行时接入。

## 运行时约束

- 八个角色固定为猫、狗、兔、熊、狐狸、浣熊、企鹅和水豚。
- 身高统一约为 1.3 Godot 单位，模型原点位于脚底中心。
- 角色基础毛色固定；只有耳尖、四肢、尾部条纹、鳍边或背部色块参与队色替换。
- 眼睛、腹部、口鼻和物种标志不参与染色。
- 脚下使用细队色环辅助混战辨识。
- 共享动作命名为 `Idle`、`Waddle`、`PlaceBubble`、`Trapped`、`Defeat`、`Victory`。
