// =============================================================================
// BnB Smart AI System — 智能怪物系统
// =============================================================================

var MonsterCount = 3;
var MonsterThinkInterval = 150;
var MonsterStorage = [];
var DIRS = [{dx: 0, dy: -1}, {dx: 0, dy: 1}, {dx: -1, dy: 0}, {dx: 1, dy: 0}];

// =============================================================================
// 工具函数
// =============================================================================

function MapKey(x, y) {
    return x + "_" + y;
}

function ParseKey(key) {
    var p = key.split("_");
    return {X: +p[0], Y: +p[1]};
}

function IsInsideMap(x, y) {
    return x >= 0 && y >= 0 && x < 15 && y < 13;
}

function IsAIWalkable(x, y) {
    return IsInsideMap(x, y) && (townBarrierMap[y][x] === 0 || townBarrierMap[y][x] > 100);
}

function ManhattanDist(x1, y1, x2, y2) {
    return Math.abs(x1 - x2) + Math.abs(y1 - y2);
}

// =============================================================================
// BFS 全图可达性分析
// =============================================================================

function AIPathBFS(startX, startY) {
    var queue = [{x: startX, y: startY}];
    var visited = {};
    var dist = {};
    var prev = {};
    var sk = MapKey(startX, startY);
    visited[sk] = true;
    dist[sk] = 0;
    prev[sk] = null;
    var head = 0;

    while (head < queue.length) {
        var cur = queue[head++];
        var ck = MapKey(cur.x, cur.y);
        for (var i = 0; i < 4; i++) {
            var nx = cur.x + DIRS[i].dx;
            var ny = cur.y + DIRS[i].dy;
            var nk = MapKey(nx, ny);
            if (visited[nk] || !IsAIWalkable(nx, ny)) continue;
            visited[nk] = true;
            dist[nk] = dist[ck] + 1;
            prev[nk] = ck;
            queue.push({x: nx, y: ny});
        }
    }

    return {dist: dist, prev: prev};
}

// =============================================================================
// 威胁地图
// =============================================================================

function BuildThreatMap() {
    var threatMap = {};

    for (var y = 0; y < PaopaoArray.length; y++) {
        if (!PaopaoArray[y]) continue;
        for (var x = 0; x < PaopaoArray[y].length; x++) {
            var p = PaopaoArray[y][x];
            if (!p || p.IsExploded) continue;
            var cid = p.CurrentMapID.Y * 15 + p.CurrentMapID.X;
            var bp = FindPaopaoBombXY(cid, p.PaopaoStrong);
            var all = bp.X.concat(bp.Y);
            all.push(cid);
            for (var i = 0; i < all.length; i++) {
                threatMap[MapKey(all[i] % 15, parseInt(all[i] / 15, 10))] = true;
            }
        }
    }

    return threatMap;
}

function AddSimulatedBombThreat(baseThreatMap, bombX, bombY, strong) {
    var sim = {};
    for (var k in baseThreatMap) sim[k] = true;
    var cid = bombY * 15 + bombX;
    var bp = FindPaopaoBombXY(cid, strong);
    var all = bp.X.concat(bp.Y);
    all.push(cid);
    for (var i = 0; i < all.length; i++) {
        sim[MapKey(all[i] % 15, parseInt(all[i] / 15, 10))] = true;
    }
    return sim;
}

// 从 bombPos 出发，在模拟威胁地图中找到最近的安全格
function FindEscapeRoute(fromX, fromY, bombX, bombY, strong, currentThreatMap) {
    var simThreat = AddSimulatedBombThreat(currentThreatMap, bombX, bombY, strong);
    var queue = [{x: fromX, y: fromY}];
    var visited = {};
    var distMap = {};
    var sk = MapKey(fromX, fromY);
    visited[sk] = true;
    distMap[sk] = 0;
    var head = 0;

    while (head < queue.length) {
        var cur = queue[head++];
        var ck = MapKey(cur.x, cur.y);
        if (distMap[ck] > 0 && !simThreat[ck]) {
            return {X: cur.x, Y: cur.y};
        }
        if (distMap[ck] >= 8) continue;
        for (var i = 0; i < 4; i++) {
            var nx = cur.x + DIRS[i].dx;
            var ny = cur.y + DIRS[i].dy;
            var nk = MapKey(nx, ny);
            if (visited[nk] || !IsAIWalkable(nx, ny)) continue;
            visited[nk] = true;
            distMap[nk] = distMap[ck] + 1;
            queue.push({x: nx, y: ny});
        }
    }

    return null;
}

// =============================================================================
// 全局查询
// =============================================================================

function GetSinglePlayerPlayer() {
    if (singlePlayerState && singlePlayerState.Player) return singlePlayerState.Player;
    for (var i = 0; i < RoleStorage.length; i++) {
        if (RoleStorage[i].RoleNumber === 1 && !RoleStorage[i].IsDeath) return RoleStorage[i];
    }
    return null;
}

function FindAllEnemies(selfRole) {
    var enemies = [];
    for (var i = 0; i < RoleStorage.length; i++) {
        var r = RoleStorage[i];
        if (r !== selfRole && r.RoleNumber !== selfRole.RoleNumber && !r.IsDeath) {
            enemies.push(r);
        }
    }
    return enemies;
}

// =============================================================================
// 进化系统 — 跨局持久化学习
// =============================================================================

var AIEvolution = {
    deathHeatMap: {},
    playerHeatMap: {},
    totalKills: 0,
    totalDeaths: 0,
    aggression: 0.5,
    caution: 0.5,
    lastDecay: 0,

    save: function() {
        try {
            localStorage.setItem("bnb_ai_evo", JSON.stringify({
                dh: this.deathHeatMap, ph: this.playerHeatMap,
                k: this.totalKills, d: this.totalDeaths,
                a: this.aggression, c: this.caution
            }));
        } catch (e) {}
    },

    load: function() {
        try {
            var d = JSON.parse(localStorage.getItem("bnb_ai_evo"));
            if (d) {
                this.deathHeatMap = d.dh || {};
                this.playerHeatMap = d.ph || {};
                this.totalKills = d.k || 0;
                this.totalDeaths = d.d || 0;
                this.aggression = typeof d.a === "number" ? d.a : 0.5;
                this.caution = typeof d.c === "number" ? d.c : 0.5;
            }
        } catch (e) {}
    },

    onDeath: function(x, y) {
        var k = MapKey(x, y);
        this.deathHeatMap[k] = (this.deathHeatMap[k] || 0) + 1;
        this.totalDeaths++;
        this.caution = Math.min(1, this.caution + 0.05);
        this.aggression = Math.max(0.1, this.aggression - 0.03);
        this.save();
    },

    onKill: function() {
        this.totalKills++;
        this.aggression = Math.min(0.9, this.aggression + 0.03);
        this.caution = Math.max(0.1, this.caution - 0.02);
        this.save();
    },

    updatePlayer: function(pos) {
        if (!pos) return;
        var k = MapKey(pos.X, pos.Y);
        this.playerHeatMap[k] = (this.playerHeatMap[k] || 0) + 1;
    },

    getDanger: function(x, y) {
        return (this.deathHeatMap[MapKey(x, y)] || 0) * this.caution;
    },

    getPlayerHeat: function(x, y) {
        return this.playerHeatMap[MapKey(x, y)] || 0;
    },

    decay: function() {
        var now = Date.now();
        if (now - this.lastDecay < 30000) return;
        this.lastDecay = now;
        this.aggression += (0.5 - this.aggression) * 0.05;
        this.caution += (0.5 - this.caution) * 0.05;
        var k;
        for (k in this.deathHeatMap) {
            this.deathHeatMap[k] *= 0.95;
            if (this.deathHeatMap[k] < 0.1) delete this.deathHeatMap[k];
        }
        for (k in this.playerHeatMap) {
            this.playerHeatMap[k] *= 0.9;
            if (this.playerHeatMap[k] < 0.1) delete this.playerHeatMap[k];
        }
        this.save();
    }
};

AIEvolution.load();

// =============================================================================
// Monster 类
// =============================================================================

var Monster = function() {
    this.Role = new Role(2);
    this.Role.Offset = new Size(0, 17);
    this.Role.RideSize = new Size(56, 60);
    this.Role.Object.Size = new Size(56, 67);
    this.Role.AniSize = new Size(56, 70);
    this.Role.DieSize = new Size(56, 98);
    this.Role.MoveStep = 3;
    this.Role.SetRawSpeed(3);
    this.Role.CanPaopaoLength = 2;

    this.ThinkInterval = null;
    this.LastBombAt = 0;
    this.LastTargetKey = "";
    this.AttackPlan = null;
    this.State = "idle";
};

Monster.prototype.SetMap = function(x, y) {
    this.Role.SetToMap(x, y);
};

Monster.prototype.MoveToMap = function(pos) {
    if (!pos) return;
    var key = pos.X + "_" + pos.Y;
    if (this.LastTargetKey === key) return;
    this.LastTargetKey = key;
    this.Role.MoveTo(pos.X, pos.Y);
};

Monster.prototype.ClearTarget = function() {
    this.LastTargetKey = "";
};

Monster.prototype.CanDropBomb = function() {
    return Date.now() - this.LastBombAt > 800 &&
        this.Role.CanPaopaoLength > this.Role.PaopaoCount;
};

Monster.prototype.DropBomb = function() {
    this.LastBombAt = Date.now();
    this.Role.PaoPao();
    this.ClearTarget();
};

Monster.prototype.Start = function() {
    var self = this;
    self.Think();
    self.ThinkInterval = setInterval(function() {
        if (!self.Role.IsDeath && !self.Role.IsInPaopao) {
            self.Think();
        } else {
            self.Stop();
        }
    }, MonsterThinkInterval);
};

Monster.prototype.Stop = function() {
    this.Role.Stop();
    clearInterval(this.Role.movetoInterval);
    if (this.ThinkInterval) {
        clearInterval(this.ThinkInterval);
        this.ThinkInterval = null;
    }
    this.AttackPlan = null;
};

Monster.prototype.OnDeath = function(pos) {
    if (pos) AIEvolution.onDeath(pos.X, pos.Y);
};

Monster.prototype.OnKill = function() {
    AIEvolution.onKill();
};

// =============================================================================
// Think — 核心决策循环（优先级状态机）
// =============================================================================

Monster.prototype.Think = function() {
    var role = this.Role;
    var currentMap = role.CurrentMapID();
    if (!currentMap) return;

    var threatMap = BuildThreatMap();
    var currentKey = MapKey(currentMap.X, currentMap.Y);
    var player = GetSinglePlayerPlayer();

    // 已到达目标 → 清除，允许重新选择
    if (this.LastTargetKey === currentKey) {
        this.LastTargetKey = "";
    }

    AIEvolution.decay();
    if (player && !player.IsDeath) {
        AIEvolution.updatePlayer(player.CurrentMapID());
    }

    // ───── 1. EVADE — 处于危险区时立即逃跑 ─────
    if (threatMap[currentKey]) {
        this.AttackPlan = null;
        this.State = "evade";
        var safe = this.FindSafeTile(currentMap, threatMap);
        if (safe) this.MoveToMap(safe);
        return;
    }

    // ───── 2. RESCUE — 救被困泡的队友 ─────
    var rescueTarget = this.FindRescueTarget(currentMap, threatMap);
    if (rescueTarget) {
        this.State = "rescue";
        this.MoveToMap(rescueTarget);
        return;
    }

    // ───── 3. KILL — 击杀被困泡的敌人 ─────
    var killTarget = this.FindKillTarget(currentMap, threatMap);
    if (killTarget) {
        this.State = "kill";
        this.MoveToMap(killTarget);
        return;
    }

    // ───── 4. V-ATTACK — 多泡夹击攻击 ─────
    if (this.AttackPlan) {
        this.ExecuteVAttackPlan(currentMap, threatMap);
        return;
    }
    if (player && !player.IsDeath &&
        role.CanPaopaoLength >= 2 && role.PaopaoCount === 0 &&
        AIEvolution.aggression > 0.3) {
        if (this.TryPlanVAttack(player, currentMap, threatMap)) {
            return;
        }
    }

    // ───── 5. ATTACK — 直接攻击 ─────
    if (player && !player.IsDeath) {
        var playerMap = player.CurrentMapID();
        if (playerMap) {
            var dist = ManhattanDist(currentMap.X, currentMap.Y, playerMap.X, playerMap.Y);

            // 已在攻击位：放泡 + 逃跑
            if (dist === 1 && this.CanDropBomb()) {
                var escape = FindEscapeRoute(
                    currentMap.X, currentMap.Y,
                    currentMap.X, currentMap.Y,
                    role.PaopaoStrong, threatMap
                );
                if (escape) {
                    this.State = "attack";
                    this.DropBomb();
                    this.MoveToMap(escape);
                    return;
                }
            }

            // 在爆炸范围内但未到相邻位：放泡（远程攻击）
            if (dist <= role.PaopaoStrong && dist > 1 && this.CanDropBomb()) {
                if (this.WouldBlastReach(currentMap.X, currentMap.Y, role.PaopaoStrong,
                    playerMap.X, playerMap.Y)) {
                    var escapeRemote = FindEscapeRoute(
                        currentMap.X, currentMap.Y,
                        currentMap.X, currentMap.Y,
                        role.PaopaoStrong, threatMap
                    );
                    if (escapeRemote) {
                        this.State = "attack";
                        this.DropBomb();
                        this.MoveToMap(escapeRemote);
                        return;
                    }
                }
            }

            // 移动到攻击位（始终尝试，不受 aggression 门槛限制）
            var atkPos = this.FindAttackPosition(player, currentMap, threatMap);
            if (atkPos) {
                this.State = "attack";
                this.MoveToMap(atkPos);
                return;
            }
        }
    }

    // ───── 6. COLLECT — 捡道具 ─────
    var item = this.FindBestItem(currentMap, threatMap);
    if (item) {
        this.State = "collect";
        this.MoveToMap(item);
        return;
    }

    // ───── 7. BREAK — 炸箱子 ─────
    var boxAction = this.FindBoxAction(currentMap, threatMap);
    if (boxAction) {
        this.State = "break";
        if (boxAction.bomb && this.CanDropBomb()) {
            var boxEscape = FindEscapeRoute(
                currentMap.X, currentMap.Y,
                currentMap.X, currentMap.Y,
                role.PaopaoStrong, threatMap
            );
            if (boxEscape) {
                this.DropBomb();
                this.MoveToMap(boxEscape);
                return;
            }
        }
        this.MoveToMap(boxAction);
        return;
    }

    // ───── 8. PATROL — 巡逻 ─────
    this.State = "patrol";
    this.ClearTarget();
    var patrol = this.FindPatrolTarget(currentMap, threatMap);
    if (patrol) this.MoveToMap(patrol);
};

// =============================================================================
// 寻找安全格（含半身闪避退路）
// =============================================================================

Monster.prototype.FindSafeTile = function(currentMap, threatMap) {
    var bfs = AIPathBFS(currentMap.X, currentMap.Y);
    var bestTile = null;
    var bestDist = Infinity;

    for (var key in bfs.dist) {
        if (threatMap[key]) continue;
        var d = bfs.dist[key];
        var pos = ParseKey(key);
        d += AIEvolution.getDanger(pos.X, pos.Y);
        if (d < bestDist) {
            bestDist = d;
            bestTile = pos;
        }
    }

    if (bestTile) return bestTile;

    // 无完全安全格 — 退到相邻可行走格（半身闪避）
    for (var i = 0; i < 4; i++) {
        var nx = currentMap.X + DIRS[i].dx;
        var ny = currentMap.Y + DIRS[i].dy;
        if (IsAIWalkable(nx, ny)) {
            return {X: nx, Y: ny};
        }
    }

    return null;
};

// =============================================================================
// 救队友
// =============================================================================

Monster.prototype.FindRescueTarget = function(currentMap, threatMap) {
    var bfs = AIPathBFS(currentMap.X, currentMap.Y);
    var bestTarget = null;
    var bestDist = Infinity;

    for (var i = 0; i < RoleStorage.length; i++) {
        var r = RoleStorage[i];
        if (r === this.Role || r.RoleNumber !== this.Role.RoleNumber) continue;
        if (!r.IsInPaopao) continue;

        var pos = r.CurrentMapID();
        if (!pos) continue;

        // 直接走到被困队友所在格（触碰距离24px，需要重叠）
        var pk = MapKey(pos.X, pos.Y);
        if ((pk in bfs.dist) && !threatMap[pk] && bfs.dist[pk] < bestDist) {
            bestDist = bfs.dist[pk];
            bestTarget = {X: pos.X, Y: pos.Y};
        }
    }

    return bestTarget;
};

// =============================================================================
// 击杀被困敌人
// =============================================================================

Monster.prototype.FindKillTarget = function(currentMap, threatMap) {
    var bfs = AIPathBFS(currentMap.X, currentMap.Y);
    var bestTarget = null;
    var bestDist = Infinity;

    for (var i = 0; i < RoleStorage.length; i++) {
        var r = RoleStorage[i];
        if (r === this.Role || r.RoleNumber === this.Role.RoleNumber) continue;
        if (!r.IsInPaopao) continue;

        var pos = r.CurrentMapID();
        if (!pos) continue;

        // 直接走到被困敌人所在格（触碰距离24px，需要重叠）
        var pk = MapKey(pos.X, pos.Y);
        if ((pk in bfs.dist) && !threatMap[pk] && bfs.dist[pk] < bestDist) {
            bestDist = bfs.dist[pk];
            bestTarget = {X: pos.X, Y: pos.Y};
        }
    }

    return bestTarget;
};

// =============================================================================
// 爆炸范围判定
// =============================================================================

Monster.prototype.WouldBlastReach = function(bombX, bombY, strong, targetX, targetY) {
    var cid = bombY * 15 + bombX;
    var bp = FindPaopaoBombXY(cid, strong);
    var targetId = targetY * 15 + targetX;
    return bp.X.concat(bp.Y).indexOf(targetId) !== -1;
};

// =============================================================================
// 寻找攻击位
// =============================================================================

Monster.prototype.FindAttackPosition = function(player, currentMap, threatMap) {
    var playerMap = player.CurrentMapID();
    if (!playerMap) return null;
    var strong = this.Role.PaopaoStrong;
    var bfs = AIPathBFS(currentMap.X, currentMap.Y);
    var bestPos = null;
    var bestScore = Infinity;
    var candidates = [];
    var x, y, key, cid, bp, targetId, escape, score;

    // 同行位置（水平水柱穿过敌人）
    for (x = Math.max(0, playerMap.X - strong); x <= Math.min(14, playerMap.X + strong); x++) {
        if (x === playerMap.X) continue;
        candidates.push({X: x, Y: playerMap.Y});
    }
    // 同列位置（垂直水柱穿过敌人）
    for (y = Math.max(0, playerMap.Y - strong); y <= Math.min(12, playerMap.Y + strong); y++) {
        if (y === playerMap.Y) continue;
        candidates.push({X: playerMap.X, Y: y});
    }

    targetId = playerMap.Y * 15 + playerMap.X;

    for (var i = 0; i < candidates.length; i++) {
        var c = candidates[i];
        key = MapKey(c.X, c.Y);
        if (!IsAIWalkable(c.X, c.Y) || threatMap[key] || !(key in bfs.dist)) continue;

        cid = c.Y * 15 + c.X;
        bp = FindPaopaoBombXY(cid, strong);
        if (bp.X.concat(bp.Y).indexOf(targetId) === -1) continue;

        escape = FindEscapeRoute(c.X, c.Y, c.X, c.Y, strong, threatMap);
        if (!escape) continue;

        score = bfs.dist[key] + AIEvolution.getDanger(c.X, c.Y);
        if (score < bestScore) {
            bestScore = score;
            bestPos = c;
        }
    }

    // 退路：直接相邻（旧行为兜底）
    if (!bestPos) {
        var adj = [
            {X: playerMap.X - 1, Y: playerMap.Y},
            {X: playerMap.X + 1, Y: playerMap.Y},
            {X: playerMap.X, Y: playerMap.Y - 1},
            {X: playerMap.X, Y: playerMap.Y + 1}
        ];
        for (var j = 0; j < adj.length; j++) {
            var a = adj[j];
            key = MapKey(a.X, a.Y);
            if (!IsAIWalkable(a.X, a.Y) || threatMap[key] || !(key in bfs.dist)) continue;
            score = bfs.dist[key];
            if (score < bestScore) {
                bestScore = score;
                bestPos = a;
            }
        }
    }

    return bestPos;
};

// =============================================================================
// 道具拾取
// =============================================================================

Monster.prototype.ScoreItem = function(itemCode) {
    var role = this.Role;
    switch (itemCode) {
        case 101: return role.CanPaopaoLength < 5 ? 3 : 0.5;
        case 102: return role.MoveStep < RoleConstant.MaxMoveStep ? 3 : 0.5;
        case 103: return role.PaopaoStrong < RoleConstant.MaxPaopaoStrong ? 3 : 0.5;
        case 104: return role.PaopaoStrong < RoleConstant.MaxPaopaoStrong ? 8 : 0.5;
        case 105: return role.MoveStep < RoleConstant.MaxMoveStep ? 8 : 0.5;
        case 106: return role.IsCanMovePaopao ? 0.5 : 6;
        case 107: case 108: case 109:
            return role.MoveHorse !== MoveHorseObject.None ? 0.5 : 2;
        default: return 1;
    }
};

Monster.prototype.FindBestItem = function(currentMap, threatMap) {
    var bfs = AIPathBFS(currentMap.X, currentMap.Y);
    var bestItem = null;
    var bestValue = -Infinity;

    for (var y = 0; y < 13; y++) {
        for (var x = 0; x < 15; x++) {
            var v = townBarrierMap[y][x];
            if (v < 101 || v > 109) continue;

            var key = MapKey(x, y);
            if (threatMap[key] || !(key in bfs.dist)) continue;

            var dist = bfs.dist[key];
            var worth = this.ScoreItem(v);
            var value = worth / (dist + 1);

            if (value > bestValue) {
                bestValue = value;
                bestItem = {X: x, Y: y};
            }
        }
    }

    return bestItem;
};

// =============================================================================
// 炸箱子
// =============================================================================

Monster.prototype.FindBoxAction = function(currentMap, threatMap) {
    var role = this.Role;

    // 已在箱子旁边 → 立即放泡
    for (var d = 0; d < 4; d++) {
        var bx = currentMap.X + DIRS[d].dx;
        var by = currentMap.Y + DIRS[d].dy;
        if (IsInsideMap(bx, by) && townBarrierMap[by][bx] === 3) {
            var esc = FindEscapeRoute(
                currentMap.X, currentMap.Y,
                currentMap.X, currentMap.Y,
                role.PaopaoStrong, threatMap
            );
            if (esc) {
                return {X: currentMap.X, Y: currentMap.Y, bomb: true};
            }
        }
    }

    // 寻找最近的可炸箱子位
    var bfs = AIPathBFS(currentMap.X, currentMap.Y);
    var bestTarget = null;
    var bestDist = Infinity;

    for (var y = 0; y < 13; y++) {
        for (var x = 0; x < 15; x++) {
            if (townBarrierMap[y][x] !== 3) continue;

            for (var di = 0; di < 4; di++) {
                var ax = x + DIRS[di].dx;
                var ay = y + DIRS[di].dy;
                var ak = MapKey(ax, ay);
                if (!IsAIWalkable(ax, ay) || threatMap[ak] || !(ak in bfs.dist)) continue;

                var esc2 = FindEscapeRoute(ax, ay, ax, ay, role.PaopaoStrong, threatMap);
                if (!esc2) continue;

                var dd = bfs.dist[ak] + AIEvolution.getDanger(ax, ay) * 2;
                if (dd < bestDist) {
                    bestDist = dd;
                    bestTarget = {X: ax, Y: ay, bomb: false};
                }
            }
        }
    }

    return bestTarget;
};

// =============================================================================
// V 字型多泡攻击
// =============================================================================

Monster.prototype.TryPlanVAttack = function(player, currentMap, threatMap) {
    var playerMap = player.CurrentMapID();
    if (!playerMap) return false;
    var strong = this.Role.PaopaoStrong;
    if (strong < 2) return false;
    var bfs = AIPathBFS(currentMap.X, currentMap.Y);

    var hPositions = [];
    var vPositions = [];
    var x, y, key, cid, bp, targetId, esc;

    targetId = playerMap.Y * 15 + playerMap.X;

    // 同行候选位（水平水柱经过敌人）
    for (x = Math.max(0, playerMap.X - strong); x <= Math.min(14, playerMap.X + strong); x++) {
        if (x === playerMap.X) continue;
        key = MapKey(x, playerMap.Y);
        if (!IsAIWalkable(x, playerMap.Y) || threatMap[key] || !(key in bfs.dist)) continue;
        cid = playerMap.Y * 15 + x;
        bp = FindPaopaoBombXY(cid, strong);
        if (bp.X.indexOf(targetId) === -1) continue;
        esc = FindEscapeRoute(x, playerMap.Y, x, playerMap.Y, strong, threatMap);
        if (esc) hPositions.push({X: x, Y: playerMap.Y, dist: bfs.dist[key]});
    }

    // 同列候选位（垂直水柱经过敌人）
    for (y = Math.max(0, playerMap.Y - strong); y <= Math.min(12, playerMap.Y + strong); y++) {
        if (y === playerMap.Y) continue;
        key = MapKey(playerMap.X, y);
        if (!IsAIWalkable(playerMap.X, y) || threatMap[key] || !(key in bfs.dist)) continue;
        cid = y * 15 + playerMap.X;
        bp = FindPaopaoBombXY(cid, strong);
        if (bp.Y.indexOf(targetId) === -1) continue;
        esc = FindEscapeRoute(playerMap.X, y, playerMap.X, y, strong, threatMap);
        if (esc) vPositions.push({X: playerMap.X, Y: y, dist: bfs.dist[key]});
    }

    if (hPositions.length === 0 || vPositions.length === 0) return false;

    hPositions.sort(function(a, b) { return a.dist - b.dist; });
    vPositions.sort(function(a, b) { return a.dist - b.dist; });

    var pos1 = hPositions[0];
    var pos2 = vPositions[0];

    // 先去近的
    if (pos1.dist > pos2.dist) {
        var tmp = pos1; pos1 = pos2; pos2 = tmp;
    }

    var totalDist = pos1.dist + ManhattanDist(pos1.X, pos1.Y, pos2.X, pos2.Y);
    if (totalDist > 12) return false;

    this.AttackPlan = {
        target: {X: playerMap.X, Y: playerMap.Y},
        pos1: {X: pos1.X, Y: pos1.Y},
        pos2: {X: pos2.X, Y: pos2.Y},
        phase: "move1",
        startTime: Date.now()
    };
    this.State = "v_attack";
    this.ClearTarget();
    this.MoveToMap(this.AttackPlan.pos1);
    return true;
};

Monster.prototype.ExecuteVAttackPlan = function(currentMap, threatMap) {
    var plan = this.AttackPlan;
    if (!plan) return;

    // 安全优先 — 处于危险区立即放弃计划
    if (threatMap[MapKey(currentMap.X, currentMap.Y)]) {
        this.AttackPlan = null;
        this.State = "evade";
        var safe = this.FindSafeTile(currentMap, threatMap);
        if (safe) this.MoveToMap(safe);
        return;
    }

    // 验证计划有效性
    var player = GetSinglePlayerPlayer();
    if (!player || player.IsDeath) { this.AttackPlan = null; return; }

    var playerMap = player.CurrentMapID();
    if (!playerMap) { this.AttackPlan = null; return; }

    var targetMoved = ManhattanDist(plan.target.X, plan.target.Y, playerMap.X, playerMap.Y) > 2;
    var tooOld = Date.now() - plan.startTime > 6000;
    if (targetMoved || tooOld) {
        this.AttackPlan = null;
        return;
    }

    if (plan.phase === "move1") {
        if (currentMap.X === plan.pos1.X && currentMap.Y === plan.pos1.Y) {
            if (this.CanDropBomb()) {
                this.DropBomb();
                plan.phase = "move2";
                this.MoveToMap(plan.pos2);
            }
        } else {
            this.MoveToMap(plan.pos1);
        }
    } else if (plan.phase === "move2") {
        if (currentMap.X === plan.pos2.X && currentMap.Y === plan.pos2.Y) {
            if (this.CanDropBomb()) {
                this.DropBomb();
                plan.phase = "escape";
                var safeTile = this.FindSafeTile(currentMap, BuildThreatMap());
                if (safeTile) this.MoveToMap(safeTile);
            }
        } else {
            this.MoveToMap(plan.pos2);
        }
    } else {
        // escape phase
        this.AttackPlan = null;
        var safeEsc = this.FindSafeTile(currentMap, threatMap);
        if (safeEsc) this.MoveToMap(safeEsc);
    }
};

// =============================================================================
// 巡逻 — 根据玩家热力图探索
// =============================================================================

Monster.prototype.FindPatrolTarget = function(currentMap, threatMap) {
    var bfs = AIPathBFS(currentMap.X, currentMap.Y);
    var candidates = [];

    for (var key in bfs.dist) {
        if (threatMap[key]) continue;
        var d = bfs.dist[key];
        if (d < 2 || d > 12) continue;
        var pos = ParseKey(key);
        var heat = AIEvolution.getPlayerHeat(pos.X, pos.Y);
        var score = heat / (d + 1) - AIEvolution.getDanger(pos.X, pos.Y) + Math.random() * 2;
        candidates.push({pos: pos, score: score});
    }

    if (candidates.length === 0) return null;

    candidates.sort(function(a, b) { return b.score - a.score; });
    // 从前 5 名中随机选一个，增加行为多样性
    var pick = Math.min(candidates.length, 5);
    return candidates[Math.floor(Math.random() * pick)].pos;
};

// =============================================================================
// 启动怪物
// =============================================================================

function StartMonsters(count) {
    var targetCount = count || MonsterCount;
    MonsterStorage = [];

    for (var i = 0; i < targetCount; i++) {
        var mapID = {
            X: Math.floor(Math.random() * 15),
            Y: Math.floor(Math.random() * 13)
        };

        if (townBarrierMap[mapID.Y][mapID.X] === 0 && !(mapID.X === 0 && mapID.Y === 0)) {
            var monster = new Monster();
            monster.SetMap(mapID.X, mapID.Y);
            monster.Start();
            MonsterStorage.push(monster);
        } else {
            i--;
        }
    }

    return MonsterStorage;
}
