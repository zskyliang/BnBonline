# BnBonline

H5 泡泡堂在线对战版，使用 Node.js + Express + Socket.IO 搭建服务端，前端修改自 [Visolleon/bnb](https://github.com/Visolleon/bnb)。

详细启动文档请看：[docs/GAME_STARTUP.md](docs/GAME_STARTUP.md)

## 1. 本地启动（Node 方式）

### 前置要求

- Node.js 18+（推荐 20）
- npm

### 启动步骤

```bash
npm install
npm start
```

默认访问地址：

- 游戏主页: [http://127.0.0.1:4000](http://127.0.0.1:4000)
- 对战模式示例: [http://127.0.0.1:4000/?mode=battle&ml=1&ml_conf=0.26&ml_move_conf=0.34&ml_margin=0.03&ml_force_move_eta=460&ml_wait_block_eta=760&ml_move_threat_ms=300&ml_model=/output/ml/models/dodge_bc_v1.onnx](http://127.0.0.1:4000/?mode=battle&ml=1&ml_conf=0.26&ml_move_conf=0.34&ml_margin=0.03&ml_force_move_eta=460&ml_wait_block_eta=760&ml_move_threat_ms=300&ml_model=/output/ml/models/dodge_bc_v1.onnx)
- 专家规则 AI 1v1 实时逐帧观测: [http://127.0.0.1:4000/?mode=expert_duel_1v1](http://127.0.0.1:4000/?mode=expert_duel_1v1)

### 可选：后台脚本启动

项目已提供脚本：

```bash
./public/game/start-game.sh
./public/game/restart-game.sh
```

Windows PowerShell 可使用：

```powershell
.\public\game\restart-game.ps1
```

如果本机限制脚本执行，可用：

```powershell
powershell -ExecutionPolicy Bypass -File .\public\game\restart-game.ps1
```

适合本地开发时快速重启服务。

## 2. Docker 一键部署（推荐移植）

### 前置要求

- Docker
- Docker Compose（Docker Desktop 内置）

### 一键启动

在项目根目录执行：

```bash
docker compose up -d --build
```

然后访问：

- [http://127.0.0.1:4000](http://127.0.0.1:4000)

### 常用命令

```bash
# 查看日志
docker compose logs -f

# 停止并删除容器
docker compose down

# 仅重启服务（不重建镜像）
docker compose restart
```

### 端口说明

- 容器内部端口固定为 `4000`
- 宿主机端口默认 `4000`
- 如需改宿主机端口，可在启动时指定环境变量：

```bash
PORT=8080 docker compose up -d --build
```

此时访问：

- [http://127.0.0.1:8080](http://127.0.0.1:8080)

### 数据持久化

`docker-compose.yml` 已把宿主机 `./output` 挂载到容器 `/app/output`，训练输出和中间文件可在主机侧保留，容器重建后不会丢失。

## 3. 移植到新机器启动

在新机器上执行：

```bash
git clone https://github.com/SineYuan/BnBonline.git
cd BnBonline
docker compose up -d --build
```

完成后直接访问浏览器即可。

## 4. 已知问题

网络延迟会导致双方游戏不同步。
