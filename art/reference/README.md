# 《森林泡泡染色战》美术参考

这里保存 ImageGen 风格母版、Sprite3D 运行时素材原图、完整提示词和验收快照。
`art/.gdignore` 与 Web 排除规则保证本目录不会进入发行包。

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
- `sprite25d/source/`：八角色六动作表、环境、道具、特效、UI 原始 ImageGen 图，
  以及用户确认的棋盘/大厅构图参考副本。
- `sprite25d/prompts-v1.md`：Sprite3D 正式生产提示词、色键规范、版本和校验值。
- `sprite25d/directional-v2/`：八角色四方向步态、四朝向 Idle、正面 Trapped
  原始图、提示词模板和 SHA-256 版本记录。
- `sprite25d/bubbles-v3/`：八角色专属大水泡原图、物种标记规范和战斗验收图。
- `sprite25d/plant-wind-v2/`：五类植物四帧风摆表、提示词和雨天验收图。
- `sprite25d/acceptance/`：真实 Godot 场景的多状态、多分辨率验收快照。

## 当前验收状态

首轮角色风格已于 2026-07-26 通过用户验收。随后根据最终方案改为纯
ImageGen + Sprite3D 流水线：猫的四方向步态先在真实战斗镜头完成自主验收，
再按相同模板批量生产其余七角色。旧 v1 动作表保留为历史身份参考，不进入运行时。

## 运行时约束

- 八个角色固定为猫、狗、兔、熊、狐狸、浣熊、企鹅和水豚。
- 纸偶视觉高度统一约为 1.3 Godot 单位，所有姿势使用相同脚底锚点。
- 角色基础毛色固定；只有耳尖、四肢、尾部条纹、鳍边或背部色块参与队色替换。
- 眼睛、腹部、口鼻和物种标志不参与染色。
- 脚下使用细队色环辅助混战辨识。
- 逻辑动作严格为 `Idle`、`WalkUp`、`WalkDown`、`WalkLeft`、`WalkRight`、
  `Trapped`；不保留放泡、倒地、胜利或失败动作。
- 每个移动方向独立使用 4 帧、8 FPS 动画。上移展示完整背面，左右为分别生成
  的真实侧面；双手始终向两侧展开，短腿交替落地，并叠加六度身体倾斜。
- 停止后保持最后朝向 Idle；Trapped 和大厅固定使用正面；结算保持 Idle。
- 每只角色使用独立水泡外观，队色只覆盖水彩填充区，物种标记保持固定；水泡
  可见宽度不低于单格的 80%。
- 松树、灌木、蘑菇、花丛和薰衣草使用四帧 ImageGen 重绘动画、固定根部基线
  和实例错相，禁止通过刚性旋转模拟风摆。
