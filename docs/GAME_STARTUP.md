# 游戏启动与部署文档

本项目支持两种启动方式：

- 本地 Node 直接启动（开发调试）
- Docker Compose 一键启动（推荐迁移/部署）

## 1. 本地 Node 启动

### 1.1 前置要求

- Node.js 18+
- npm

### 1.2 启动命令

```bash
npm install
npm start
```

### 1.3 访问地址

- `http://127.0.0.1:4000`
- 专家规则 AI 1v1 实时逐帧观测：`http://127.0.0.1:4000/?mode=expert_duel_1v1`

如需改端口（例如 8080）：

```bash
PORT=8080 npm start
```

### 1.4 快速重启脚本（开发场景）

- Linux/macOS:

```bash
./public/game/restart-game.sh
```

- Windows PowerShell:

```powershell
.\public\game\restart-game.ps1
```

如遇执行策略限制，可改用：

```powershell
powershell -ExecutionPolicy Bypass -File .\public\game\restart-game.ps1
```

## 2. Docker 一键启动（推荐）

### 2.1 前置要求

- Docker
- Docker Compose（Docker Desktop 已内置）

### 2.2 一键部署

```bash
docker compose up -d --build
```

### 2.3 访问地址

- `http://127.0.0.1:4000`

### 2.4 常用维护命令

```bash
# 查看运行状态
docker compose ps

# 查看日志
docker compose logs -f

# 停止服务
docker compose down

# 重启服务
docker compose restart
```

### 2.5 宿主机端口自定义

```bash
PORT=8080 docker compose up -d --build
```

访问：

- `http://127.0.0.1:8080`

## 3. 移植到新机器

```bash
git clone https://github.com/SineYuan/BnBonline.git
cd BnBonline
docker compose up -d --build
```

如果你只想用 Node 方式（不安装 Docker）：

```bash
npm install
npm start
```

## 4. 目录与持久化说明

- 容器内服务目录：`/app`
- 持久化挂载：宿主机 `./output` -> 容器 `/app/output`
- 镜像构建会忽略 `node_modules`、`.git`、`output` 子文件等本地无关内容

## 5. 常见问题

### 5.1 端口占用（EADDRINUSE: 4000）

解决方法：

- 改端口启动：`PORT=8080 npm start` 或 `PORT=8080 docker compose up -d --build`
- 或释放本地 4000 端口后重启

### 5.2 Docker 无法连接 daemon

报错类似 `failed to connect to the docker API` 时，通常是 Docker Desktop 未启动。先启动 Docker，再执行 `docker compose` 命令。
