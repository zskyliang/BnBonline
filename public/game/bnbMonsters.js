// =============================================================================
// BnB Smart AI System — 智能怪物系统
// =============================================================================

var MonsterCount = 3;
var MonsterThinkInterval = 150;
var MonsterTrainingThinkInterval = 33;
var MonsterStorage = [];
var MonsterMaxPaopaoLength = typeof RoleBalanceConfig !== "undefined"
    ? RoleBalanceConfig.MaxBubbleCount
    : 8;
var AIBootcampRounds = 100;
var DIRS = [{dx: 0, dy: -1}, {dx: 0, dy: 1}, {dx: -1, dy: 0}, {dx: 1, dy: 0}];
var AIDodgePolicyStorageKey = "bnb_ai_dodge_policy";
var AIDodgePolicy = {
    forecastMs: 1750,
    safeBufferMs: 240,
    repathMs: 420,
    stuckTimeoutMs: 900,
    roamRadius: 9,
    halfBodyMinEtaMs: 220
};
var LastThreatSnapshot = null;

function ClampNumber(v, min, max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

function NormalizeDodgePolicyValue(policy) {
    policy.forecastMs = ClampNumber(parseInt(policy.forecastMs, 10) || 1750, 800, 3200);
    policy.safeBufferMs = ClampNumber(parseInt(policy.safeBufferMs, 10) || 240, 120, 800);
    policy.repathMs = ClampNumber(parseInt(policy.repathMs, 10) || 420, 150, 1200);
    policy.stuckTimeoutMs = ClampNumber(parseInt(policy.stuckTimeoutMs, 10) || 900, 400, 2600);
    policy.roamRadius = ClampNumber(parseInt(policy.roamRadius, 10) || 9, 4, 13);
    policy.halfBodyMinEtaMs = ClampNumber(parseInt(policy.halfBodyMinEtaMs, 10) || 220, 80, 900);
}

function SaveAIDodgePolicy() {
    try {
        localStorage.setItem(AIDodgePolicyStorageKey, JSON.stringify(AIDodgePolicy));
    } catch (e) {}
}

function LoadAIDodgePolicy() {
    try {
        var raw = JSON.parse(localStorage.getItem(AIDodgePolicyStorageKey));
        if (raw) {
            AIDodgePolicy.forecastMs = typeof raw.forecastMs === "number" ? raw.forecastMs : AIDodgePolicy.forecastMs;
            AIDodgePolicy.safeBufferMs = typeof raw.safeBufferMs === "number" ? raw.safeBufferMs : AIDodgePolicy.safeBufferMs;
            AIDodgePolicy.repathMs = typeof raw.repathMs === "number" ? raw.repathMs : AIDodgePolicy.repathMs;
            AIDodgePolicy.stuckTimeoutMs = typeof raw.stuckTimeoutMs === "number" ? raw.stuckTimeoutMs : AIDodgePolicy.stuckTimeoutMs;
            AIDodgePolicy.roamRadius = typeof raw.roamRadius === "number" ? raw.roamRadius : AIDodgePolicy.roamRadius;
            AIDodgePolicy.halfBodyMinEtaMs = typeof raw.halfBodyMinEtaMs === "number" ? raw.halfBodyMinEtaMs : AIDodgePolicy.halfBodyMinEtaMs;
        }
    } catch (e) {}
    NormalizeDodgePolicyValue(AIDodgePolicy);
}

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

function GetTrainingFailureHeatPenalty(x, y) {
    var key;
    if (typeof AIDodgeTrainer === "undefined" || !AIDodgeTrainer || !AIDodgeTrainer.IsRunning) {
        return 0;
    }
    if (!AIDodgeTrainer.State || !AIDodgeTrainer.State.failureHeatMap) {
        return 0;
    }
    key = MapKey(x, y);
    return AIDodgeTrainer.State.failureHeatMap[key] || 0;
}

// =============================================================================
// Offline ML (Behavioral Cloning V1)
// =============================================================================

var BNBMLDefaultModelUrl = "/output/ml/models/dodge_bc_v1.onnx";
var BNBMLCollectStorageKey = "bnb_ml_collect_state_v1";
var BNBMLConfig = {
    enableRuntime: false,
    enableCollect: false,
    freezeExpertPolicy: false,
    minConfidence: 0.34,
    minMoveConfidence: 0.4,
    top1Margin: 0.06,
    modelUrl: BNBMLDefaultModelUrl,
    preDeathWindowMs: 1500,
    predictionReuseMs: 240,
    collectWaitKeepProb: 0.15,
    forceMoveEtaMs: 520,
    waitBlockEtaMs: 760,
    moveThreatSoonMs: 280
};

function GetBNBQueryParam(name) {
    var query;
    if (typeof window === "undefined" || !window.location || !window.location.search) {
        return "";
    }
    query = new URLSearchParams(window.location.search);
    return query.get(name) || "";
}

function ParseBNBBool(raw) {
    if (raw == null) {
        return false;
    }
    if (typeof raw === "boolean") {
        return raw;
    }
    raw = String(raw).toLowerCase();
    return raw === "1" || raw === "true" || raw === "yes" || raw === "on";
}

function ParseBNBNumber(raw, fallback) {
    var v = parseFloat(raw);
    if (isNaN(v)) {
        return fallback;
    }
    return v;
}

function RefreshBNBMLConfigFromQuery() {
    var modelUrl = GetBNBQueryParam("ml_model");
    BNBMLConfig.enableRuntime = ParseBNBBool(GetBNBQueryParam("ml"));
    BNBMLConfig.enableCollect = ParseBNBBool(GetBNBQueryParam("ml_collect"));
    BNBMLConfig.freezeExpertPolicy = ParseBNBBool(GetBNBQueryParam("ml_freeze"));
    BNBMLConfig.minConfidence = ClampNumber(
        ParseBNBNumber(GetBNBQueryParam("ml_conf"), BNBMLConfig.minConfidence),
        0.05,
        0.99
    );
    BNBMLConfig.minMoveConfidence = ClampNumber(
        ParseBNBNumber(GetBNBQueryParam("ml_move_conf"), BNBMLConfig.minMoveConfidence),
        0.05,
        0.99
    );
    BNBMLConfig.top1Margin = ClampNumber(
        ParseBNBNumber(GetBNBQueryParam("ml_margin"), BNBMLConfig.top1Margin),
        0,
        0.5
    );
    BNBMLConfig.collectWaitKeepProb = ClampNumber(
        ParseBNBNumber(GetBNBQueryParam("ml_wait_keep"), BNBMLConfig.collectWaitKeepProb),
        0,
        1
    );
    BNBMLConfig.forceMoveEtaMs = ClampNumber(
        ParseBNBNumber(GetBNBQueryParam("ml_force_move_eta"), BNBMLConfig.forceMoveEtaMs),
        120,
        2000
    );
    BNBMLConfig.waitBlockEtaMs = ClampNumber(
        ParseBNBNumber(GetBNBQueryParam("ml_wait_block_eta"), BNBMLConfig.waitBlockEtaMs),
        180,
        2800
    );
    BNBMLConfig.moveThreatSoonMs = ClampNumber(
        ParseBNBNumber(GetBNBQueryParam("ml_move_threat_ms"), BNBMLConfig.moveThreatSoonMs),
        180,
        1200
    );
    if (modelUrl) {
        BNBMLConfig.modelUrl = modelUrl;
    }
}

function GetBNBMLCollectWaitKeepProb() {
    if (typeof window !== "undefined" && typeof window.BNBMLCollectWaitKeepProb === "number") {
        return ClampNumber(window.BNBMLCollectWaitKeepProb, 0, 1);
    }
    return ClampNumber(BNBMLConfig.collectWaitKeepProb, 0, 1);
}

function IsOfflineMLExpertFreezeEnabled() {
    if (typeof window !== "undefined" && ParseBNBBool(window.BNBMLFreezeExpertPolicy)) {
        return true;
    }
    return !!BNBMLConfig.freezeExpertPolicy;
}

function NormalizeActionId(action) {
    var a = parseInt(action, 10);
    if (isNaN(a) || a < 0 || a > 4) {
        return 0;
    }
    return a;
}

function EncodeDirectionToAction(direction) {
    if (direction === Direction.Up) return 1;
    if (direction === Direction.Down) return 2;
    if (direction === Direction.Left) return 3;
    if (direction === Direction.Right) return 4;
    return 0;
}

function DecodeActionToDelta(action) {
    var a = NormalizeActionId(action);
    if (a === 1) return { dx: 0, dy: -1 };
    if (a === 2) return { dx: 0, dy: 1 };
    if (a === 3) return { dx: -1, dy: 0 };
    if (a === 4) return { dx: 1, dy: 0 };
    return { dx: 0, dy: 0 };
}

function EncodeActionByTargetMap(currentMap, targetMap) {
    if (!currentMap || !targetMap) {
        return 0;
    }
    if (targetMap.X === currentMap.X && targetMap.Y === currentMap.Y) {
        return 0;
    }
    if (targetMap.X === currentMap.X && targetMap.Y === currentMap.Y - 1) {
        return 1;
    }
    if (targetMap.X === currentMap.X && targetMap.Y === currentMap.Y + 1) {
        return 2;
    }
    if (targetMap.X === currentMap.X - 1 && targetMap.Y === currentMap.Y) {
        return 3;
    }
    if (targetMap.X === currentMap.X + 1 && targetMap.Y === currentMap.Y) {
        return 4;
    }
    return 0;
}

function BuildTargetMapByAction(currentMap, action) {
    var d = DecodeActionToDelta(action);
    return {
        X: currentMap.X + d.dx,
        Y: currentMap.Y + d.dy
    };
}

function EncodeActionFromChoice(choice, currentMap) {
    if (!choice) {
        return 0;
    }
    if (choice.targetMap && currentMap) {
        return EncodeActionByTargetMap(currentMap, choice.targetMap);
    }
    if (choice.direction != null) {
        return EncodeDirectionToAction(choice.direction);
    }
    return 0;
}

function IsRoleCountableForSurvival(role) {
    return !!role && !role.IsDeath && !role.IsInPaopao;
}

function ComputeSurvivalRate(bombedCount, spawnedEffective) {
    if (spawnedEffective <= 0) {
        return 1;
    }
    return 1 - bombedCount / spawnedEffective;
}

function FormatMapPos(map) {
    if (!map) {
        return "(-,-)";
    }
    return "(" + map.X + "," + map.Y + ")";
}

function FormatDecisionPathText(startMap, targetMap, action) {
    if (!startMap) {
        return "n/a";
    }
    if (NormalizeActionId(action) === 0) {
        return FormatMapPos(startMap) + " wait";
    }
    return FormatMapPos(startMap) + "->" + FormatMapPos(targetMap || startMap);
}

var BNBMLFeatureEncoder = {
    BuildObstacleValue: function(cell) {
        if (cell === 0 || cell > 100) {
            return 0;
        }
        if (cell === 3 || cell === 8 || cell === 100) {
            return 0.5;
        }
        if (typeof IsRigidBarrierNo === "function") {
            return IsRigidBarrierNo(cell) ? 1.0 : 0.5;
        }
        return cell > 0 && cell < 100 ? 1.0 : 0.0;
    },

    CreateEmptyMap: function(rows, cols, channels) {
        var y;
        var x;
        var c;
        var map = new Array(rows);
        for (y = 0; y < rows; y++) {
            map[y] = new Array(cols);
            for (x = 0; x < cols; x++) {
                map[y][x] = new Array(channels);
                for (c = 0; c < channels; c++) {
                    map[y][x][c] = 0;
                }
            }
        }
        return map;
    },

    Encode: function(role, currentMap, snapshot) {
        var rows = typeof MapRowCount === "number" ? MapRowCount : 13;
        var cols = typeof MapColumnCount === "number" ? MapColumnCount : 15;
        var channels = 5;
        var now = Date.now();
        var forecastMs = Math.max(1, AIDodgePolicy.forecastMs || 1750);
        var fuseMs = typeof GetPaopaoFuseMs === "function" ? Math.max(1, GetPaopaoFuseMs()) : 3000;
        var src = snapshot || LastThreatSnapshot || BuildThreatSnapshot();
        var map = this.CreateEmptyMap(rows, cols, channels);
        var y;
        var x;
        var key;
        var eta;
        var bomb;
        var bombEta;
        var bombScore;
        var safeNeighbors;
        var mapPoint;
        var centerX;
        var centerY;
        var dx = 0;
        var dy = 0;

        for (y = 0; y < rows; y++) {
            for (x = 0; x < cols; x++) {
                map[y][x][0] = this.BuildObstacleValue(townBarrierMap[y][x]);
                key = MapKey(x, y);
                eta = src && src.dangerEtaMap ? src.dangerEtaMap[key] : null;
                if (typeof eta === "number" && eta <= forecastMs) {
                    map[y][x][2] = ClampNumber(1 - eta / forecastMs, 0, 1);
                }
                safeNeighbors = CountSafeNeighborTiles(x, y, src);
                map[y][x][4] = safeNeighbors >= 3 ? 1 : 0;
            }
        }

        for (y = 0; y < PaopaoArray.length; y++) {
            if (!PaopaoArray[y]) continue;
            for (x = 0; x < PaopaoArray[y].length; x++) {
                bomb = PaopaoArray[y][x];
                if (!bomb || bomb.IsExploded || y >= rows || x >= cols) continue;
                bombEta = (typeof bomb.ExplodeAt === "number" ? bomb.ExplodeAt : now + fuseMs) - now;
                if (bombEta < 0) bombEta = 0;
                bombScore = ClampNumber(1 - bombEta / fuseMs, 0, 1);
                if (bombScore > map[y][x][1]) {
                    map[y][x][1] = bombScore;
                }
            }
        }

        if (currentMap && currentMap.Y >= 0 && currentMap.Y < rows && currentMap.X >= 0 && currentMap.X < cols) {
            map[currentMap.Y][currentMap.X][3] = 1;
        }
        if (role && typeof role.MapPoint === "function" && currentMap) {
            mapPoint = role.MapPoint();
            centerX = currentMap.X * 40 + 20;
            centerY = currentMap.Y * 40 + 20;
            dx = ClampNumber((mapPoint.X - centerX) / 20, -1, 1);
            dy = ClampNumber((mapPoint.Y - centerY) / 20, -1, 1);
        }

        return {
            state_map: map,
            state_vector: [dx, dy]
        };
    },

    ToNCHWFloat32: function(stateMap) {
        var rows = stateMap.length;
        var cols = rows > 0 ? stateMap[0].length : 0;
        var channels = rows > 0 && cols > 0 ? stateMap[0][0].length : 0;
        var out = new Float32Array(channels * rows * cols);
        var c;
        var y;
        var x;
        var idx = 0;
        for (c = 0; c < channels; c++) {
            for (y = 0; y < rows; y++) {
                for (x = 0; x < cols; x++) {
                    out[idx++] = stateMap[y][x][c];
                }
            }
        }
        return out;
    }
};

var BNBMLDatasetCollector = {
    State: {
        enabled: false,
        preDeathWindowMs: 1500,
        rowsReady: [],
        stagingRows: [],
        lastOpenSample: null,
        sampleIdSeed: 0,
        samplesFinalized: 0,
        actionHist: { "0": 0, "1": 0, "2": 0, "3": 0, "4": 0 },
        spawnedBubblesEffective: 0,
        spawnedBubblesIgnoredTrapped: 0,
        bombedCount: 0
    },

    Init: function() {
        this.State.enabled = !!BNBMLConfig.enableCollect;
        this.State.preDeathWindowMs = Math.max(1000, BNBMLConfig.preDeathWindowMs || 1500);
        this.State.rowsReady = [];
        this.State.stagingRows = [];
        this.State.lastOpenSample = null;
        this.State.sampleIdSeed = 0;
        this.State.samplesFinalized = 0;
        this.State.actionHist = { "0": 0, "1": 0, "2": 0, "3": 0, "4": 0 };
        this.State.spawnedBubblesEffective = 0;
        this.State.spawnedBubblesIgnoredTrapped = 0;
        this.State.bombedCount = 0;
        this.PublishState();
    },

    ShouldCollect: function() {
        return !!this.State.enabled;
    },

    BuildSampleMeta: function(monster, currentMap, snapshot, choice) {
        var key = MapKey(currentMap.X, currentMap.Y);
        var eta = snapshot && snapshot.dangerEtaMap ? snapshot.dangerEtaMap[key] : null;
        return {
            roleNumber: monster && monster.Role ? monster.Role.RoleNumber : -1,
            x: currentMap.X,
            y: currentMap.Y,
            eta: typeof eta === "number" ? eta : null,
            activeBombs: CountActiveBombs(),
            safeNeighbors: CountSafeNeighborTiles(currentMap.X, currentMap.Y, snapshot),
            nextSafeRank: choice ? choice.safeRank : null
        };
    },

    InferTemporalPlanAction: function(monster, currentMap) {
        var plan;
        var route;
        var cursor;
        var node;
        var target;
        if (!monster || !currentMap) {
            return null;
        }
        plan = monster.CurrentTemporalPlan;
        route = plan && Array.isArray(plan.route) ? plan.route : null;
        cursor = plan && typeof plan.cursor === "number" ? plan.cursor : -1;
        if (!route || cursor < 0 || cursor >= route.length) {
            return null;
        }
        node = route[cursor];
        if (!node) {
            return null;
        }
        if (node.action === "wait") {
            return 0;
        }
        target = { X: node.x, Y: node.y };
        if (Math.abs(target.X - currentMap.X) + Math.abs(target.Y - currentMap.Y) !== 1) {
            return null;
        }
        return EncodeActionByTargetMap(currentMap, target);
    },

    FinalizeOpenSample: function(nextState, reward, done) {
        var sample = this.State.lastOpenSample;
        if (!sample) {
            return;
        }
        sample.next_state = nextState;
        sample.reward = typeof reward === "number" ? reward : 1;
        sample.done = !!done;
        this.State.stagingRows.push(sample);
        this.State.actionHist[String(sample.action)] = (this.State.actionHist[String(sample.action)] || 0) + 1;
        this.State.samplesFinalized += 1;
        this.State.lastOpenSample = null;
    },

    FlushStagingRows: function(forceAll) {
        var now = Date.now();
        var i;
        var row;
        var keep = [];
        var limitTs = now - this.State.preDeathWindowMs;
        for (i = 0; i < this.State.stagingRows.length; i++) {
            row = this.State.stagingRows[i];
            if (forceAll || row.ts <= limitTs || row.done) {
                this.State.rowsReady.push(row);
            }
            else {
                keep.push(row);
            }
        }
        this.State.stagingRows = keep;
    },

    MarkRecentPreDeath: function() {
        var now = Date.now();
        var cutoff = now - this.State.preDeathWindowMs;
        var i;
        for (i = this.State.stagingRows.length - 1; i >= 0; i--) {
            if (this.State.stagingRows[i].ts < cutoff) {
                break;
            }
            this.State.stagingRows[i].pre_death = true;
        }
        for (i = this.State.rowsReady.length - 1; i >= 0; i--) {
            if (this.State.rowsReady[i].ts < cutoff) {
                break;
            }
            this.State.rowsReady[i].pre_death = true;
        }
    },

    RecordFrame: function(monster, currentMap, snapshot) {
        var encoded;
        var choice;
        var action;
        var actionSource = "rule";
        var temporalAction;
        var newSample;
        var key;
        var eta;
        var isDangerNow;

        if (!this.ShouldCollect() || !monster || !monster.Role || !currentMap) {
            return;
        }
        if (typeof AIDodgeTrainer !== "undefined"
            && AIDodgeTrainer
            && typeof AIDodgeTrainer.IsMonsterTraining === "function"
            && !AIDodgeTrainer.IsMonsterTraining(monster)) {
            return;
        }

        encoded = BNBMLFeatureEncoder.Encode(monster.Role, currentMap, snapshot);
        choice = PickNextFrameMovementChoice(monster.Role, currentMap, snapshot);
        action = EncodeActionFromChoice(choice, currentMap);
        temporalAction = this.InferTemporalPlanAction(monster, currentMap);
        if (typeof temporalAction === "number" && temporalAction >= 0 && temporalAction <= 4) {
            if (action === 0 && temporalAction !== 0) {
                action = temporalAction;
                actionSource = "temporal_plan";
            }
            else if (action === 0 && temporalAction === 0) {
                actionSource = "temporal_plan_wait";
            }
        }
        key = MapKey(currentMap.X, currentMap.Y);
        eta = snapshot && snapshot.dangerEtaMap ? snapshot.dangerEtaMap[key] : null;
        isDangerNow = !!(snapshot && snapshot.threatMap && snapshot.threatMap[key]);

        // Reduce severe class imbalance: keep waits mostly in urgent contexts.
        if (action === 0) {
            var keepWaitProb = GetBNBMLCollectWaitKeepProb();
            if (!isDangerNow
                && (typeof eta !== "number" || eta > Math.max(320, AIDodgePolicy.safeBufferMs + 120))
                && Math.random() > keepWaitProb) {
                return;
            }
        }

        if (this.State.lastOpenSample) {
            this.FinalizeOpenSample(encoded, 1, false);
        }

        newSample = {
            id: "S" + (++this.State.sampleIdSeed),
            ts: Date.now(),
            state: {
                state_map: encoded.state_map,
                state_vector: encoded.state_vector
            },
            action: NormalizeActionId(action),
            reward: 0,
            done: false,
            next_state: null,
            pre_death: false,
            meta: this.BuildSampleMeta(monster, currentMap, snapshot, choice)
        };
        newSample.meta.action_source = actionSource;
        this.State.lastOpenSample = newSample;
        this.FlushStagingRows(false);
        this.PublishState();
    },

    OnBombed: function() {
        if (!this.ShouldCollect()) {
            return;
        }
        this.State.bombedCount += 1;
        this.MarkRecentPreDeath();
        if (this.State.lastOpenSample) {
            this.State.lastOpenSample.pre_death = true;
            this.FinalizeOpenSample(null, -1, true);
        }
        this.PublishState();
    },

    OnBubbleSpawned: function(role) {
        if (!this.ShouldCollect()) {
            return;
        }
        if (IsRoleCountableForSurvival(role)) {
            this.State.spawnedBubblesEffective += 1;
        }
        else {
            this.State.spawnedBubblesIgnoredTrapped += 1;
        }
        this.PublishState();
    },

    Drain: function(maxRows) {
        var rows;
        var count = Math.max(1, parseInt(maxRows, 10) || 2048);
        this.FlushStagingRows(false);
        rows = this.State.rowsReady.splice(0, count);
        this.PublishState();
        return rows;
    },

    ForceDrainAll: function() {
        this.FlushStagingRows(true);
        if (this.State.lastOpenSample) {
            this.FinalizeOpenSample(null, 1, false);
            this.FlushStagingRows(true);
        }
        this.PublishState();
        return this.State.rowsReady.splice(0);
    },

    PublishState: function() {
        var state = this.State;
        var survivalRate = ComputeSurvivalRate(state.bombedCount, state.spawnedBubblesEffective);
        if (typeof window === "undefined") {
            return;
        }
        window.BNBMLDatasetCollectorState = {
            enabled: !!state.enabled,
            pre_death_window_ms: state.preDeathWindowMs,
            samples_finalized: state.samplesFinalized,
            rows_ready: state.rowsReady.length,
            rows_staging: state.stagingRows.length,
            action_hist: JSON.parse(JSON.stringify(state.actionHist)),
            spawned_bubbles: state.spawnedBubblesEffective,
            spawned_bubbles_effective: state.spawnedBubblesEffective,
            spawned_bubbles_ignored_trapped: state.spawnedBubblesIgnoredTrapped,
            bombed_count: state.bombedCount,
            survival_rate: survivalRate
        };
    }
};

var BNBMLRuntime = {
    State: {
        enabled: false,
        modelUrl: BNBMLDefaultModelUrl,
        minConfidence: 0.34,
        loading: false,
        loaded: false,
        error: "",
        session: null,
        inflight: false,
        latestPrediction: null,
        inferenceCount: 0,
        usedCount: 0,
        fallbackCount: 0,
        avgLatencyMs: 0,
        actionHist: { "0": 0, "1": 0, "2": 0, "3": 0, "4": 0 },
        spawnedBubblesEffective: 0,
        spawnedBubblesIgnoredTrapped: 0,
        bombedCount: 0,
        lastDecisionSource: "",
        lastFallbackReason: "",
        decisionTrace: [],
        lastDecisionLogAt: 0,
        lastDecisionLogSig: ""
    },

    Init: function() {
        this.State.enabled = !!BNBMLConfig.enableRuntime;
        this.State.modelUrl = BNBMLConfig.modelUrl || BNBMLDefaultModelUrl;
        this.State.minConfidence = BNBMLConfig.minConfidence;
        this.State.inferenceCount = 0;
        this.State.usedCount = 0;
        this.State.fallbackCount = 0;
        this.State.avgLatencyMs = 0;
        this.State.actionHist = { "0": 0, "1": 0, "2": 0, "3": 0, "4": 0 };
        this.State.latestPrediction = null;
        this.State.spawnedBubblesEffective = 0;
        this.State.spawnedBubblesIgnoredTrapped = 0;
        this.State.bombedCount = 0;
        this.State.lastDecisionSource = "";
        this.State.lastFallbackReason = "";
        this.State.decisionTrace = [];
        this.State.lastDecisionLogAt = 0;
        this.State.lastDecisionLogSig = "";
        this.State.error = "";
        this.PublishState();
    },

    ShouldUseContext: function(currentMap, snapshot) {
        var src = snapshot || LastThreatSnapshot || BuildThreatSnapshot();
        var key;
        var eta;
        if (!this.State.enabled || !currentMap) {
            return false;
        }
        if (CountActiveBombs() <= 0) {
            return false;
        }
        key = MapKey(currentMap.X, currentMap.Y);
        eta = src && src.dangerEtaMap ? src.dangerEtaMap[key] : null;
        if (src && src.threatMap && src.threatMap[key]) {
            return true;
        }
        return typeof eta === "number" && eta <= Math.max(620, AIDodgePolicy.safeBufferMs + 320);
    },

    EnsureOrtReady: function() {
        var self = this;
        if (typeof window === "undefined") {
            return Promise.reject(new Error("window_unavailable"));
        }
        if (window.ort) {
            return Promise.resolve(window.ort);
        }
        if (window.__bnbOrtLoadingPromise) {
            return window.__bnbOrtLoadingPromise;
        }
        window.__bnbOrtLoadingPromise = new Promise(function(resolve, reject) {
            var script = document.createElement("script");
            script.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";
            script.async = true;
            script.onload = function() {
                resolve(window.ort);
            };
            script.onerror = function() {
                reject(new Error("onnxruntime_web_load_failed"));
            };
            document.head.appendChild(script);
        }).catch(function(err) {
            self.State.error = String(err && err.message ? err.message : err);
            self.PublishState();
            throw err;
        });
        return window.__bnbOrtLoadingPromise;
    },

    EnsureSession: function() {
        var self = this;
        if (!this.State.enabled) {
            return Promise.resolve(null);
        }
        if (this.State.session) {
            return Promise.resolve(this.State.session);
        }
        if (this.State.loading) {
            return Promise.resolve(null);
        }
        this.State.loading = true;
        this.PublishState();
        return this.EnsureOrtReady()
            .then(function(ort) {
                return ort.InferenceSession.create(self.State.modelUrl, {
                    executionProviders: ["wasm"]
                });
            })
            .then(function(session) {
                self.State.session = session;
                self.State.loaded = true;
                self.State.error = "";
                self.State.loading = false;
                self.PublishState();
                return session;
            })
            .catch(function(err) {
                self.State.error = String(err && err.message ? err.message : err);
                self.State.loading = false;
                self.State.loaded = false;
                self.PublishState();
                return null;
            });
    },

    SoftmaxArgmax: function(logits) {
        var maxV = -Infinity;
        var sum = 0;
        var i;
        var probs = [];
        var best = 0;
        var bestP = 0;
        for (i = 0; i < logits.length; i++) {
            if (logits[i] > maxV) {
                maxV = logits[i];
            }
        }
        for (i = 0; i < logits.length; i++) {
            probs[i] = Math.exp(logits[i] - maxV);
            sum += probs[i];
        }
        for (i = 0; i < probs.length; i++) {
            probs[i] = sum > 0 ? probs[i] / sum : 0;
            if (probs[i] > bestP) {
                bestP = probs[i];
                best = i;
            }
        }
        return {
            action: best,
            confidence: bestP,
            probs: probs
        };
    },

    ValidateAction: function(action, currentMap, snapshot) {
        var a = NormalizeActionId(action);
        var src = snapshot || LastThreatSnapshot || BuildThreatSnapshot();
        var target;
        var key;
        if (!currentMap) {
            return false;
        }
        if (a === 0) {
            key = MapKey(currentMap.X, currentMap.Y);
            if (src && src.threatMap && src.threatMap[key]) {
                return false;
            }
            if (IsTileThreatSoon(key, 120, src)) {
                return false;
            }
            return true;
        }
        target = BuildTargetMapByAction(currentMap, a);
        if (!IsAIWalkable(target.X, target.Y)) {
            return false;
        }
        key = MapKey(target.X, target.Y);
        if (IsTileThreatSoon(key, Math.max(220, BNBMLConfig.moveThreatSoonMs || 280), src)) {
            return false;
        }
        return true;
    },

    GetTileEta: function(currentMap, snapshot) {
        var src = snapshot || LastThreatSnapshot || BuildThreatSnapshot();
        var key;
        if (!currentMap || !src || !src.dangerEtaMap) {
            return null;
        }
        key = MapKey(currentMap.X, currentMap.Y);
        return typeof src.dangerEtaMap[key] === "number" ? src.dangerEtaMap[key] : null;
    },

    ShouldForceMoveNow: function(currentMap, snapshot) {
        var src = snapshot || LastThreatSnapshot || BuildThreatSnapshot();
        var key;
        var eta;
        var exits;
        if (!currentMap || !src) {
            return false;
        }
        key = MapKey(currentMap.X, currentMap.Y);
        eta = this.GetTileEta(currentMap, src);
        exits = CountSafeNeighborTiles(currentMap.X, currentMap.Y, src);
        if (src.threatMap && src.threatMap[key]) {
            return true;
        }
        if (typeof eta === "number" && eta <= Math.max(180, BNBMLConfig.forceMoveEtaMs || 520)) {
            return true;
        }
        return CountActiveBombs() > 0 && exits <= 1;
    },

    PassGuardrail: function(action, confidence, currentMap, snapshot, ranked, pred) {
        var a = NormalizeActionId(action);
        var top1;
        var top2;
        var eta;
        if (!currentMap) {
            return false;
        }
        if (a === 0) {
            eta = this.GetTileEta(currentMap, snapshot);
            if (this.ShouldForceMoveNow(currentMap, snapshot)) {
                return false;
            }
            if (typeof eta === "number" && eta <= Math.max(220, BNBMLConfig.waitBlockEtaMs || 760)) {
                return false;
            }
            top1 = pred && pred.probs ? (pred.probs[ranked[0]] || 0) : confidence;
            top2 = pred && pred.probs ? (pred.probs[ranked[1]] || 0) : 0;
            if (top1 - top2 < Math.max(0, BNBMLConfig.top1Margin || 0)) {
                return false;
            }
            return true;
        }
        if (confidence < Math.max(this.State.minConfidence, BNBMLConfig.minMoveConfidence || 0.4)) {
            return false;
        }
        return true;
    },

    SchedulePrediction: function(role, currentMap, snapshot) {
        var self = this;
        var startedAt = Date.now();
        var encoded;
        var mapTensor;
        var vectorTensor;
        var session = this.State.session;
        var inputNames;
        var outputNames;
        var mapName;
        var vecName;

        if (!session || this.State.inflight || !role || !currentMap) {
            return;
        }
        encoded = BNBMLFeatureEncoder.Encode(role, currentMap, snapshot);
        mapTensor = BNBMLFeatureEncoder.ToNCHWFloat32(encoded.state_map);
        vectorTensor = new Float32Array(encoded.state_vector);
        inputNames = session.inputNames || [];
        outputNames = session.outputNames || [];
        mapName = inputNames.length > 0 ? inputNames[0] : "state_map";
        vecName = inputNames.length > 1 ? inputNames[1] : "state_vector";
        if (inputNames.length > 1
            && inputNames[0].toLowerCase().indexOf("vector") !== -1
            && inputNames[1].toLowerCase().indexOf("map") !== -1) {
            mapName = inputNames[1];
            vecName = inputNames[0];
        }

        this.State.inflight = true;
        session.run({
            [mapName]: new window.ort.Tensor("float32", mapTensor, [1, 5, encoded.state_map.length, encoded.state_map[0].length]),
            [vecName]: new window.ort.Tensor("float32", vectorTensor, [1, vectorTensor.length])
        }).then(function(outputs) {
            var outName = outputNames.length > 0 ? outputNames[0] : Object.keys(outputs)[0];
            var data = outputs[outName] && outputs[outName].data ? outputs[outName].data : [];
            var decoded = self.SoftmaxArgmax(data);
            var latency = Date.now() - startedAt;
            self.State.inferenceCount += 1;
            self.State.avgLatencyMs = self.State.avgLatencyMs <= 0
                ? latency
                : (self.State.avgLatencyMs * 0.9 + latency * 0.1);
            self.State.latestPrediction = {
                ts: Date.now(),
                action: NormalizeActionId(decoded.action),
                confidence: decoded.confidence,
                probs: decoded.probs,
                mapKey: MapKey(currentMap.X, currentMap.Y)
            };
            self.State.inflight = false;
            self.PublishState();
        }).catch(function(err) {
            self.State.error = String(err && err.message ? err.message : err);
            self.State.inflight = false;
            self.PublishState();
        });
    },

    DecideAction: function(monster, currentMap, snapshot) {
        var pred = this.State.latestPrediction;
        var now = Date.now();
        var ranked;
        var i;
        var candidate;
        var conf;
        if (!this.State.enabled || !monster || !monster.Role || !currentMap) {
            return { ok: false, reason: "disabled_or_invalid" };
        }
        this.EnsureSession();
        if (this.State.session) {
            this.SchedulePrediction(monster.Role, currentMap, snapshot);
        }
        if (!pred || now - pred.ts > Math.max(120, BNBMLConfig.predictionReuseMs || 240)) {
            this.State.fallbackCount += 1;
            this.PublishState();
            return { ok: false, reason: "prediction_not_ready" };
        }
        if (pred.confidence < this.State.minConfidence) {
            this.State.fallbackCount += 1;
            this.PublishState();
            return { ok: false, reason: "low_confidence" };
        }

        ranked = [0, 1, 2, 3, 4];
        ranked.sort(function(a, b) {
            return (pred.probs[b] || 0) - (pred.probs[a] || 0);
        });
        candidate = null;
        for (i = 0; i < ranked.length; i++) {
            if (this.ValidateAction(ranked[i], currentMap, snapshot)) {
                conf = pred.probs && pred.probs[ranked[i]] != null ? pred.probs[ranked[i]] : pred.confidence;
                if (this.PassGuardrail(ranked[i], conf, currentMap, snapshot, ranked, pred)) {
                    candidate = ranked[i];
                    break;
                }
            }
        }
        if (candidate == null) {
            this.State.fallbackCount += 1;
            this.PublishState();
            return { ok: false, reason: "guardrail_rejected" };
        }
        this.State.usedCount += 1;
        this.State.actionHist[String(candidate)] = (this.State.actionHist[String(candidate)] || 0) + 1;
        this.PublishState();
        return {
            ok: true,
            action: candidate,
            confidence: pred.probs && pred.probs[candidate] != null ? pred.probs[candidate] : pred.confidence
        };
    },

    AppendDecisionTrace: function(info) {
        var item = info || {};
        var trace = this.State.decisionTrace;
        var msg;
        var sig;
        var canConsole = true;
        var now;
        var minInterval = 120;
        item.ts = item.ts || Date.now();
        trace.push(item);
        if (trace.length > 120) {
            trace.splice(0, trace.length - 120);
        }
        this.State.lastDecisionSource = item.source || "";
        this.State.lastFallbackReason = item.fallback_reason || "";
        msg = "[BNB-ML][battle] source=" + (item.source || "unknown")
            + " reason=" + (item.reason || "")
            + " conf=" + (typeof item.confidence === "number" ? item.confidence.toFixed(3) : "na")
            + " start=" + (item.start || "(-,-)")
            + " path=" + (item.path || "n/a");
        sig = (item.source || "")
            + "|" + (item.reason || "")
            + "|" + (item.fallback_reason || "")
            + "|" + (item.start || "")
            + "|" + (item.path || "");
        now = item.ts;
        if (this.State.lastDecisionLogSig === sig && now - this.State.lastDecisionLogAt < minInterval) {
            canConsole = false;
        }
        if (canConsole && typeof console !== "undefined" && typeof console.log === "function") {
            console.log(msg);
            this.State.lastDecisionLogAt = now;
            this.State.lastDecisionLogSig = sig;
        }
        this.PublishState();
    },

    RecordModelDecision: function(currentMap, targetMap, action, confidence) {
        this.AppendDecisionTrace({
            source: "model",
            reason: "model_action",
            fallback_reason: "",
            confidence: typeof confidence === "number" ? confidence : null,
            action: NormalizeActionId(action),
            start: FormatMapPos(currentMap),
            path: FormatDecisionPathText(currentMap, targetMap, action)
        });
    },

    RecordRuleDecision: function(currentMap, targetMap, reason) {
        this.AppendDecisionTrace({
            source: "rule",
            reason: reason || "rule_dodge",
            fallback_reason: reason || "",
            confidence: null,
            action: targetMap && currentMap && targetMap.X === currentMap.X && targetMap.Y === currentMap.Y ? 0 : null,
            start: FormatMapPos(currentMap),
            path: FormatDecisionPathText(
                currentMap,
                targetMap || currentMap,
                targetMap && currentMap && targetMap.X === currentMap.X && targetMap.Y === currentMap.Y ? 0 : 4
            )
        });
    },

    OnBubbleSpawned: function(role) {
        if (IsRoleCountableForSurvival(role)) {
            this.State.spawnedBubblesEffective += 1;
        }
        else {
            this.State.spawnedBubblesIgnoredTrapped += 1;
        }
        this.PublishState();
    },

    OnBombed: function() {
        this.State.bombedCount += 1;
        this.PublishState();
    },

    PublishState: function() {
        var s = this.State;
        var totalDecision = s.usedCount + s.fallbackCount;
        var fallbackRate = totalDecision > 0 ? s.fallbackCount / totalDecision : 0;
        var survivalRate = ComputeSurvivalRate(s.bombedCount, s.spawnedBubblesEffective);
        if (typeof window === "undefined") {
            return;
        }
        window.BNBMLRuntimeState = {
            enabled: !!s.enabled,
            model_url: s.modelUrl,
            min_confidence: s.minConfidence,
            min_move_confidence: BNBMLConfig.minMoveConfidence,
            top1_margin: BNBMLConfig.top1Margin,
            force_move_eta_ms: BNBMLConfig.forceMoveEtaMs,
            wait_block_eta_ms: BNBMLConfig.waitBlockEtaMs,
            loading: !!s.loading,
            loaded: !!s.loaded,
            inflight: !!s.inflight,
            error: s.error || "",
            inference_count: s.inferenceCount,
            used_count: s.usedCount,
            fallback_count: s.fallbackCount,
            fallback_rate: fallbackRate,
            avg_latency_ms: s.avgLatencyMs,
            action_hist: JSON.parse(JSON.stringify(s.actionHist)),
            latest_prediction: s.latestPrediction ? {
                ts: s.latestPrediction.ts,
                action: s.latestPrediction.action,
                confidence: s.latestPrediction.confidence,
                mapKey: s.latestPrediction.mapKey
            } : null,
            spawned_bubbles: s.spawnedBubblesEffective,
            spawned_bubbles_effective: s.spawnedBubblesEffective,
            spawned_bubbles_ignored_trapped: s.spawnedBubblesIgnoredTrapped,
            bombed_count: s.bombedCount,
            survival_rate: survivalRate,
            last_decision_source: s.lastDecisionSource,
            last_fallback_reason: s.lastFallbackReason,
            decision_trace_tail: s.decisionTrace.slice(Math.max(0, s.decisionTrace.length - 20))
        };
    }
};

RefreshBNBMLConfigFromQuery();
BNBMLDatasetCollector.Init();
BNBMLRuntime.Init();

if (typeof window !== "undefined") {
    window.BNBMLCollectorDrain = function(maxRows) {
        return BNBMLDatasetCollector.Drain(maxRows);
    };
    window.BNBMLCollectorDrainAll = function() {
        return BNBMLDatasetCollector.ForceDrainAll();
    };
    window.BNBMLRefreshConfig = function() {
        RefreshBNBMLConfigFromQuery();
        BNBMLDatasetCollector.Init();
        BNBMLRuntime.Init();
    };
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
    var snapshot = BuildThreatSnapshot();
    return snapshot.threatMap;
}

function BuildThreatSnapshot() {
    var eventSnapshot = null;
    var threatMap = {};
    var dangerEtaMap = {};
    var dangerEndMap = {};
    var now = Date.now();
    var horizon = AIDodgePolicy.forecastMs;
    var i;
    var j;
    var key;
    var eta;
    var startIn;
    var endIn;

    if (typeof GetBNBExplosionEventSnapshot === "function") {
        eventSnapshot = GetBNBExplosionEventSnapshot(now);
    }

    if (!eventSnapshot) {
        eventSnapshot = {
            bombs: [],
            clusters: [],
            activeWindows: []
        };
    }

    for (i = 0; i < eventSnapshot.clusters.length; i++) {
        startIn = eventSnapshot.clusters[i].startAt - now;
        endIn = eventSnapshot.clusters[i].endAt - now;
        for (j = 0; j < eventSnapshot.clusters[i].coverageMapIds.length; j++) {
            key = MapKey(
                eventSnapshot.clusters[i].coverageMapIds[j] % 15,
                parseInt(eventSnapshot.clusters[i].coverageMapIds[j] / 15, 10)
            );
            eta = startIn < 0 ? 0 : startIn;
            if (!(key in dangerEtaMap) || eta < dangerEtaMap[key]) {
                dangerEtaMap[key] = eta;
            }
            if (!(key in dangerEndMap) || endIn > dangerEndMap[key]) {
                dangerEndMap[key] = endIn;
            }
            if (startIn <= horizon && endIn > 0) {
                threatMap[key] = true;
            }
        }
    }

    for (i = 0; i < eventSnapshot.activeWindows.length; i++) {
        endIn = eventSnapshot.activeWindows[i].endAt - now;
        for (j = 0; j < eventSnapshot.activeWindows[i].coverageMapIds.length; j++) {
            key = MapKey(
                eventSnapshot.activeWindows[i].coverageMapIds[j] % 15,
                parseInt(eventSnapshot.activeWindows[i].coverageMapIds[j] / 15, 10)
            );
            if (!(key in dangerEtaMap) || 0 < dangerEtaMap[key]) {
                dangerEtaMap[key] = 0;
            }
            if (!(key in dangerEndMap) || endIn > dangerEndMap[key]) {
                dangerEndMap[key] = endIn;
            }
            if (endIn > 0) {
                threatMap[key] = true;
            }
        }
    }

    // 兼容兜底：若事件快照不可用时回退到原始泡泡预测
    if (eventSnapshot.clusters.length === 0 && eventSnapshot.activeWindows.length === 0) {
        var y;
        var x;
        var p;
        var cid;
        var bp;
        var all;
        var explodeEta;

        for (y = 0; y < PaopaoArray.length; y++) {
            if (!PaopaoArray[y]) continue;
            for (x = 0; x < PaopaoArray[y].length; x++) {
                p = PaopaoArray[y][x];
                if (!p || p.IsExploded) continue;

                cid = p.CurrentMapID.Y * 15 + p.CurrentMapID.X;
                bp = FindPaopaoBombXY(cid, p.PaopaoStrong);
                all = bp.X.concat(bp.Y);
                all.push(cid);
                explodeEta = (typeof p.ExplodeAt === "number" ? p.ExplodeAt : now + 3000) - now;
                if (explodeEta < 0) {
                    explodeEta = 0;
                }

                for (i = 0; i < all.length; i++) {
                    key = MapKey(all[i] % 15, parseInt(all[i] / 15, 10));
                    if (!(key in dangerEtaMap) || explodeEta < dangerEtaMap[key]) {
                        dangerEtaMap[key] = explodeEta;
                    }
                    if (explodeEta <= horizon) {
                        threatMap[key] = true;
                    }
                }
            }
        }
    }

    LastThreatSnapshot = {
        threatMap: threatMap,
        dangerEtaMap: dangerEtaMap,
        dangerEndMap: dangerEndMap,
        eventSnapshot: eventSnapshot,
        now: now
    };
    return LastThreatSnapshot;
}

function GetThreatSceneRevision() {
    if (typeof window === "undefined") {
        return 0;
    }
    if (typeof window.BNBThreatSceneRevision === "number") {
        return window.BNBThreatSceneRevision;
    }
    return 0;
}

function GetLastBubbleSpawnEvent() {
    if (typeof window === "undefined" || !window.BNBLastBubbleSpawnEvent) {
        return null;
    }
    return window.BNBLastBubbleSpawnEvent;
}

function BuildUnsafeIntervalsByMapNoFromEventSnapshot(eventSnapshot) {
    var byMapNo = {};
    var i;
    var j;
    var mapNo;
    var row;
    var item;
    var windows = [];

    if (!eventSnapshot) {
        return byMapNo;
    }

    if (Array.isArray(eventSnapshot.clusters)) {
        windows = windows.concat(eventSnapshot.clusters);
    }
    if (Array.isArray(eventSnapshot.activeWindows)) {
        windows = windows.concat(eventSnapshot.activeWindows);
    }

    for (i = 0; i < windows.length; i++) {
        item = windows[i];
        if (!item || !Array.isArray(item.coverageMapIds)) {
            continue;
        }
        if (typeof item.startAt !== "number" || typeof item.endAt !== "number" || item.endAt <= item.startAt) {
            continue;
        }
        for (j = 0; j < item.coverageMapIds.length; j++) {
            mapNo = parseInt(item.coverageMapIds[j], 10);
            if (isNaN(mapNo) || mapNo < 0 || mapNo >= 195) {
                continue;
            }
            row = byMapNo[mapNo];
            if (!row) {
                row = [];
                byMapNo[mapNo] = row;
            }
            row.push({
                start: item.startAt,
                end: item.endAt
            });
        }
    }

    for (mapNo in byMapNo) {
        if (!byMapNo.hasOwnProperty(mapNo)) {
            continue;
        }
        row = byMapNo[mapNo];
        row.sort(function(a, b) {
            if (a.start === b.start) {
                return a.end - b.end;
            }
            return a.start - b.start;
        });
        var merged = [];
        for (i = 0; i < row.length; i++) {
            if (merged.length === 0 || row[i].start > merged[merged.length - 1].end) {
                merged.push({
                    start: row[i].start,
                    end: row[i].end
                });
            }
            else if (row[i].end > merged[merged.length - 1].end) {
                merged[merged.length - 1].end = row[i].end;
            }
        }
        byMapNo[mapNo] = merged;
    }
    return byMapNo;
}

function BuildTileSafetyTimeline(intervals, now) {
    var timeline = [];
    var cursor = now;
    var i;
    var startAt;
    var endAt;

    if (!intervals || intervals.length === 0) {
        return [{
            from: now,
            to: null,
            safe: 1
        }];
    }

    for (i = 0; i < intervals.length; i++) {
        endAt = intervals[i].end;
        if (typeof endAt !== "number" || endAt <= now) {
            continue;
        }
        startAt = intervals[i].start;
        if (typeof startAt !== "number") {
            startAt = now;
        }
        if (startAt < now) {
            startAt = now;
        }

        if (startAt > cursor) {
            timeline.push({
                from: cursor,
                to: startAt,
                safe: 1
            });
        }

        if (endAt > cursor) {
            timeline.push({
                from: Math.max(cursor, startAt),
                to: endAt,
                safe: 0
            });
            cursor = endAt;
        }
    }

    if (timeline.length === 0) {
        timeline.push({
            from: now,
            to: null,
            safe: 1
        });
        return timeline;
    }

    timeline.push({
        from: cursor,
        to: null,
        safe: 1
    });

    var compact = [];
    for (i = 0; i < timeline.length; i++) {
        if (compact.length === 0) {
            compact.push(timeline[i]);
            continue;
        }
        var prev = compact[compact.length - 1];
        if (prev.safe === timeline[i].safe && (prev.to === timeline[i].from || prev.to == null)) {
            prev.to = timeline[i].to;
        }
        else {
            compact.push(timeline[i]);
        }
    }
    return compact;
}

function BuildTemporalSafetyModel(snapshot, now) {
    var sourceSnapshot = snapshot || LastThreatSnapshot || BuildThreatSnapshot();
    var timestamp = typeof now === "number" ? now : Date.now();
    var eventSnapshot = sourceSnapshot && sourceSnapshot.eventSnapshot
        ? sourceSnapshot.eventSnapshot
        : (typeof GetBNBExplosionEventSnapshot === "function"
            ? GetBNBExplosionEventSnapshot(timestamp)
            : { now: timestamp, bombs: [], clusters: [], activeWindows: [] });
    var unsafeIntervalsByMapNo = BuildUnsafeIntervalsByMapNoFromEventSnapshot(eventSnapshot);
    var timelinesByMapNo = {};
    var mapNo;

    for (mapNo = 0; mapNo < 195; mapNo++) {
        timelinesByMapNo[mapNo] = BuildTileSafetyTimeline(unsafeIntervalsByMapNo[mapNo] || [], timestamp);
    }

    return {
        now: timestamp,
        eventSnapshot: eventSnapshot,
        unsafeIntervalsByMapNo: unsafeIntervalsByMapNo,
        timelinesByMapNo: timelinesByMapNo
    };
}

function GetMapNoTimelineSegmentAt(mapNo, at, temporalModel) {
    var src = temporalModel;
    var time = at;
    var timeline;
    var i;
    if (!src || mapNo < 0 || mapNo >= 195) {
        return null;
    }
    if (typeof time !== "number") {
        time = src.now;
    }
    if (time < src.now) {
        time = src.now;
    }
    timeline = src.timelinesByMapNo ? src.timelinesByMapNo[mapNo] : null;
    if (!timeline || timeline.length === 0) {
        return null;
    }
    for (i = 0; i < timeline.length; i++) {
        if (time < timeline[i].from) {
            break;
        }
        if (timeline[i].to == null || time < timeline[i].to) {
            return timeline[i];
        }
    }
    return timeline[timeline.length - 1] || null;
}

function IsMapNoSafeAtTemporalTime(mapNo, at, temporalModel) {
    var seg = GetMapNoTimelineSegmentAt(mapNo, at, temporalModel);
    if (!seg) {
        return false;
    }
    return seg.safe === 1;
}

function EvaluateFootSafetyWithTemporalMapPoint(mapPoint, at, temporalModel) {
    var leftMap = GetMapIDByRelativePoint(mapPoint.X - 12, mapPoint.Y + 16);
    var rightMap = GetMapIDByRelativePoint(mapPoint.X + 12, mapPoint.Y + 16);
    var leftNo = leftMap ? leftMap.Y * 15 + leftMap.X : -1;
    var rightNo = rightMap ? rightMap.Y * 15 + rightMap.X : -1;
    var leftSeg = GetMapNoTimelineSegmentAt(leftNo, at, temporalModel);
    var rightSeg = GetMapNoTimelineSegmentAt(rightNo, at, temporalModel);
    var leftSafe = !!(leftSeg && leftSeg.safe === 1);
    var rightSafe = !!(rightSeg && rightSeg.safe === 1);
    var leftUntil = leftSeg && leftSeg.to != null ? leftSeg.to : Number.POSITIVE_INFINITY;
    var rightUntil = rightSeg && rightSeg.to != null ? rightSeg.to : Number.POSITIVE_INFINITY;
    return {
        leftNo: leftNo,
        rightNo: rightNo,
        leftSafe: leftSafe,
        rightSafe: rightSafe,
        isSafe: leftSafe && rightSafe,
        isHalfSafe: leftSafe !== rightSafe,
        safeUntil: Math.min(leftUntil, rightUntil)
    };
}

function BuildTileSamplePoints(x, y) {
    var centerX = x * 40 + 20;
    var centerY = y * 40 + 20;
    return [
        { X: centerX, Y: centerY, tag: "center" },
        { X: centerX - 11, Y: centerY, tag: "left-edge" },
        { X: centerX + 11, Y: centerY, tag: "right-edge" },
        { X: centerX, Y: centerY - 8, tag: "up-edge" },
        { X: centerX, Y: centerY + 8, tag: "down-edge" }
    ];
}

function EvaluateTileTemporalSafety(x, y, at, temporalModel) {
    var points = BuildTileSamplePoints(x, y);
    var i;
    var foot;
    var bestFull = null;
    var bestHalf = null;
    var safeDuration;

    for (i = 0; i < points.length; i++) {
        foot = EvaluateFootSafetyWithTemporalMapPoint(points[i], at, temporalModel);
        safeDuration = foot.safeUntil === Number.POSITIVE_INFINITY
            ? Number.POSITIVE_INFINITY
            : Math.max(0, foot.safeUntil - at);
        if (foot.isSafe) {
            if (!bestFull || safeDuration > bestFull.safeDurationMs) {
                bestFull = {
                    safeRank: 0,
                    safetyMode: "full",
                    foot: foot,
                    sampleTag: points[i].tag,
                    safeDurationMs: safeDuration
                };
            }
        }
        else if (foot.isHalfSafe) {
            if (!bestHalf || safeDuration > bestHalf.safeDurationMs) {
                bestHalf = {
                    safeRank: 1,
                    safetyMode: "half",
                    foot: foot,
                    sampleTag: points[i].tag,
                    safeDurationMs: safeDuration
                };
            }
        }
    }
    if (bestFull) {
        return bestFull;
    }
    if (bestHalf) {
        return bestHalf;
    }
    return {
        safeRank: 2,
        safetyMode: "unsafe",
        foot: null,
        sampleTag: "none",
        safeDurationMs: 0
    };
}

function CountTemporalSafeNeighborTiles(x, y, at, temporalModel) {
    var count = 0;
    var i;
    var nx;
    var ny;
    var evalState;
    for (i = 0; i < DIRS.length; i++) {
        nx = x + DIRS[i].dx;
        ny = y + DIRS[i].dy;
        if (!IsAIWalkable(nx, ny)) {
            continue;
        }
        evalState = EvaluateTileTemporalSafety(nx, ny, at, temporalModel);
        if (evalState.safeRank <= 1) {
            count++;
        }
    }
    return count;
}

function BuildTemporalPlanStateKey(x, y, at, baseTime, binMs) {
    var bin = Math.floor((at - baseTime) / binMs);
    return x + "_" + y + "_" + bin;
}

function BuildTemporalEvadePlan(role, currentMap, temporalModel, options) {
    var opts = options || {};
    var baseTime = temporalModel && typeof temporalModel.now === "number" ? temporalModel.now : Date.now();
    var waitMs = 33;
    var binMs = 33;
    var stepMs = Math.max(waitMs, Math.ceil(400 / Math.max(1, role.MoveStep || 1)));
    var maxHorizonMs = typeof opts.maxHorizonMs === "number"
        ? opts.maxHorizonMs
        : Math.max(2600, AIDodgePolicy.forecastMs + 1200);
    var minHoldMs = typeof opts.minHoldMs === "number" ? opts.minHoldMs : 60;
    var goalMinHoldMs = typeof opts.goalMinHoldMs === "number"
        ? opts.goalMinHoldMs
        : Math.max(220, AIDodgePolicy.safeBufferMs + 40);
    var goalMinExits = typeof opts.goalMinExits === "number" ? Math.max(1, opts.goalMinExits) : 1;
    var allowHalfBody = opts.allowHalfBody !== false;
    var preferMove = !!opts.preferMove;
    var forceMove = !!opts.forceMove;
    var panicMode = !!opts.panicMode;
    var minGoalMoveSteps = typeof opts.minGoalMoveSteps === "number"
        ? Math.max(0, opts.minGoalMoveSteps)
        : (forceMove ? 1 : 0);
    var minGoalDisplacement = typeof opts.minGoalDisplacement === "number"
        ? Math.max(0, opts.minGoalDisplacement)
        : (forceMove ? 1 : 0);
    var stayPenaltyMs = typeof opts.stayPenaltyMs === "number" ? Math.max(0, opts.stayPenaltyMs) : 0;
    var moveBonusMs = typeof opts.moveBonusMs === "number"
        ? opts.moveBonusMs
        : (preferMove ? Math.max(80, Math.round(stepMs * 0.75)) : 0);
    var preferMoveSlackMs = typeof opts.preferMoveSlackMs === "number"
        ? Math.max(0, opts.preferMoveSlackMs)
        : Math.max(180, stepMs * 2);
    var preferStayDurationMs = typeof opts.preferStayDurationMs === "number"
        ? Math.max(goalMinHoldMs, opts.preferStayDurationMs)
        : Math.max(goalMinHoldMs + stepMs * 2, 900);
    var maxArrival = baseTime + maxHorizonMs;
    var nodes = [];
    var open = [];
    var bestCost = {};
    var expanded = 0;
    var maxExpand = 32000;
    var bestGoalIdx = -1;
    var bestMovedGoalIdx = -1;
    var bestDeepMovedGoalIdx = -1;
    var bestDisplacedGoalIdx = -1;
    var startEval;
    var startExits;

    function pushNode(node) {
        nodes.push(node);
        open.push(nodes.length - 1);
    }

    function popMinNodeIndex() {
        var minAt = Infinity;
        var minOpenIndex = -1;
        var i;
        var nodeIdx;
        for (i = 0; i < open.length; i++) {
            nodeIdx = open[i];
            if (nodes[nodeIdx].t < minAt) {
                minAt = nodes[nodeIdx].t;
                minOpenIndex = i;
            }
        }
        if (minOpenIndex < 0) {
            return -1;
        }
        var picked = open[minOpenIndex];
        open.splice(minOpenIndex, 1);
        return picked;
    }

    function ResolveSafeDurationMs(raw) {
        if (raw === Number.POSITIVE_INFINITY) {
            return maxHorizonMs + 1200;
        }
        if (typeof raw !== "number") {
            return 0;
        }
        return Math.max(0, raw);
    }

    function ComputeGoalScore(node) {
        var durationMs = ResolveSafeDurationMs(node.safeDurationMs);
        var displacement = ManhattanDist(node.x, node.y, currentMap.X, currentMap.Y);
        var failureHeat = GetTrainingFailureHeatPenalty(node.x, node.y);
        var score = node.t - baseTime;
        score += node.waitCount * Math.max(12, Math.round(waitMs * 0.72));
        if (panicMode) {
            score += node.waitCount * 220;
        }
        score += node.safeRank * 220;
        score -= node.safeNeighbors * 132;
        score -= Math.min(durationMs, maxHorizonMs + 800) / 4;
        score -= Math.min(6, node.moveCount) * 30;
        score += (node.backtrackCount || 0) * 420;
        score += failureHeat * 180;
        if (node.safeNeighbors <= 1) {
            score += 240;
        }

        if (node.safeNeighbors < goalMinExits) {
            score += (goalMinExits - node.safeNeighbors) * 360;
        }
        if (minGoalMoveSteps > 0 && node.moveCount < minGoalMoveSteps) {
            score += (minGoalMoveSteps - node.moveCount) * 220;
        }
        if (minGoalDisplacement > 0 && displacement < minGoalDisplacement) {
            score += (minGoalDisplacement - displacement) * 260;
        }
        if (node.moveCount === 0) {
            if (preferMove && durationMs < preferStayDurationMs) {
                score += stayPenaltyMs;
            }
            else if (stayPenaltyMs > 0) {
                score += Math.round(stayPenaltyMs * 0.35);
            }
            if (forceMove) {
                score += stayPenaltyMs + 9000;
            }
        }
        else if (preferMove) {
            score -= moveBonusMs;
        }

        return score;
    }

    function isBetterGoal(aIdx, bIdx) {
        var a = nodes[aIdx];
        var b = nodes[bIdx];
        if (a.goalScore !== b.goalScore) {
            return a.goalScore < b.goalScore;
        }
        if (a.t !== b.t) {
            return a.t < b.t;
        }
        if (a.safeRank !== b.safeRank) {
            return a.safeRank < b.safeRank;
        }
        if (a.safeNeighbors !== b.safeNeighbors) {
            return a.safeNeighbors > b.safeNeighbors;
        }
        if (a.waitCount !== b.waitCount) {
            return a.waitCount < b.waitCount;
        }
        return a.moveCount < b.moveCount;
    }

    function acceptNeighbor(parentIdx, nx, ny, nt, action) {
        var evalState = EvaluateTileTemporalSafety(nx, ny, nt, temporalModel);
        var stateKey;
        var safeNeighbors;
        var node;
        var parentNode;
        var isImmediateBacktrack = false;
        if (evalState.safeRank > 1) {
            return;
        }
        if (!allowHalfBody && evalState.safeRank === 1) {
            return;
        }
        if (evalState.safeDurationMs < minHoldMs) {
            return;
        }
        stateKey = BuildTemporalPlanStateKey(nx, ny, nt, baseTime, binMs);
        if (typeof bestCost[stateKey] === "number" && bestCost[stateKey] <= nt) {
            return;
        }
        safeNeighbors = CountTemporalSafeNeighborTiles(nx, ny, nt, temporalModel);
        bestCost[stateKey] = nt;
        parentNode = nodes[parentIdx];
        if (action === "move" && parentNode && parentNode.parent >= 0) {
            var grandParent = nodes[parentNode.parent];
            if (grandParent && grandParent.x === nx && grandParent.y === ny) {
                isImmediateBacktrack = true;
            }
        }
        node = {
            x: nx,
            y: ny,
            t: nt,
            parent: parentIdx,
            action: action,
            safeRank: evalState.safeRank,
            safetyMode: evalState.safetyMode,
            safeDurationMs: evalState.safeDurationMs,
            safeNeighbors: safeNeighbors,
            waitCount: nodes[parentIdx].waitCount + (action === "wait" ? 1 : 0),
            moveCount: nodes[parentIdx].moveCount + (action === "move" ? 1 : 0),
            halfCount: nodes[parentIdx].halfCount + (evalState.safeRank === 1 ? 1 : 0),
            backtrackCount: nodes[parentIdx].backtrackCount + (isImmediateBacktrack ? 1 : 0),
            goalScore: Infinity
        };
        pushNode(node);
    }

    if (!temporalModel || !currentMap || !role) {
        return null;
    }

    startEval = EvaluateTileTemporalSafety(currentMap.X, currentMap.Y, baseTime, temporalModel);
    startExits = CountTemporalSafeNeighborTiles(currentMap.X, currentMap.Y, baseTime, temporalModel);
    pushNode({
        x: currentMap.X,
        y: currentMap.Y,
        t: baseTime,
        parent: -1,
        action: "start",
        safeRank: startEval.safeRank,
        safetyMode: startEval.safetyMode,
        safeDurationMs: startEval.safeDurationMs,
        safeNeighbors: startExits,
        waitCount: 0,
        moveCount: 0,
        halfCount: startEval.safeRank === 1 ? 1 : 0,
        backtrackCount: 0,
        goalScore: Infinity
    });
    bestCost[BuildTemporalPlanStateKey(currentMap.X, currentMap.Y, baseTime, baseTime, binMs)] = baseTime;

    while (open.length > 0 && expanded < maxExpand) {
        var nodeIdx = popMinNodeIndex();
        var node;
        var i;
        var nx;
        var ny;
        var nt;
        if (nodeIdx < 0) {
            break;
        }
        node = nodes[nodeIdx];
        expanded++;

        if (node.safeRank <= 1 && node.safeNeighbors >= goalMinExits && node.safeDurationMs >= goalMinHoldMs) {
            node.goalScore = ComputeGoalScore(node);
            if (bestGoalIdx < 0 || isBetterGoal(nodeIdx, bestGoalIdx)) {
                bestGoalIdx = nodeIdx;
            }
            if (node.moveCount > 0 && (bestMovedGoalIdx < 0 || isBetterGoal(nodeIdx, bestMovedGoalIdx))) {
                bestMovedGoalIdx = nodeIdx;
            }
            if (node.moveCount >= minGoalMoveSteps
                && (bestDeepMovedGoalIdx < 0 || isBetterGoal(nodeIdx, bestDeepMovedGoalIdx))) {
                bestDeepMovedGoalIdx = nodeIdx;
            }
            if (node.moveCount >= minGoalMoveSteps
                && ManhattanDist(node.x, node.y, currentMap.X, currentMap.Y) >= minGoalDisplacement
                && (bestDisplacedGoalIdx < 0 || isBetterGoal(nodeIdx, bestDisplacedGoalIdx))) {
                bestDisplacedGoalIdx = nodeIdx;
            }
        }
        if (node.t >= maxArrival) {
            continue;
        }

        nt = node.t + waitMs;
        if (nt <= maxArrival) {
            acceptNeighbor(nodeIdx, node.x, node.y, nt, "wait");
        }
        for (i = 0; i < DIRS.length; i++) {
            nx = node.x + DIRS[i].dx;
            ny = node.y + DIRS[i].dy;
            if (!IsAIWalkable(nx, ny)) {
                continue;
            }
            nt = node.t + stepMs;
            if (nt > maxArrival) {
                continue;
            }
            acceptNeighbor(nodeIdx, nx, ny, nt, "move");
        }
    }

    if (bestGoalIdx < 0) {
        return null;
    }

    var chosenGoalIdx = bestGoalIdx;
    if (forceMove && bestDisplacedGoalIdx >= 0) {
        chosenGoalIdx = bestDisplacedGoalIdx;
    }
    else if (forceMove && bestDeepMovedGoalIdx >= 0) {
        chosenGoalIdx = bestDeepMovedGoalIdx;
    }
    else if (forceMove && bestMovedGoalIdx >= 0) {
        chosenGoalIdx = bestMovedGoalIdx;
    }
    else if (!forceMove
        && preferMove
        && bestMovedGoalIdx >= 0
        && nodes[chosenGoalIdx].moveCount === 0
        && nodes[bestMovedGoalIdx].goalScore <= nodes[chosenGoalIdx].goalScore + preferMoveSlackMs) {
        chosenGoalIdx = bestMovedGoalIdx;
    }

    var route = [];
    var cursor = chosenGoalIdx;
    while (cursor >= 0) {
        route.push({
            x: nodes[cursor].x,
            y: nodes[cursor].y,
            arrivalAt: nodes[cursor].t,
            safeRank: nodes[cursor].safeRank,
            safetyMode: nodes[cursor].safetyMode,
            action: nodes[cursor].action,
            safeNeighbors: nodes[cursor].safeNeighbors,
            safeDurationMs: nodes[cursor].safeDurationMs
        });
        cursor = nodes[cursor].parent;
    }
    route.reverse();

    return {
        createdAt: baseTime,
        start: { x: currentMap.X, y: currentMap.Y },
        target: {
            x: nodes[chosenGoalIdx].x,
            y: nodes[chosenGoalIdx].y
        },
        route: route,
        cursor: 1,
        stepMs: stepMs,
        waitMs: waitMs,
        estimatedMs: Math.max(0, nodes[chosenGoalIdx].t - baseTime),
        totalSteps: Math.max(0, route.length - 1),
        moveSteps: nodes[chosenGoalIdx].moveCount,
        waitSteps: nodes[chosenGoalIdx].waitCount,
        halfBodySteps: nodes[chosenGoalIdx].halfCount,
        expandedStates: expanded,
        horizonMs: maxHorizonMs,
        goalScore: nodes[chosenGoalIdx].goalScore,
        preferMove: preferMove,
        forceMove: forceMove,
        panicMode: panicMode,
        goalMinExits: goalMinExits,
        minGoalDisplacement: minGoalDisplacement,
        allowHalfBody: allowHalfBody
    };
}

function GetRoleFootStateFromMapPoint(mapPoint, snapshot, sampleTime) {
    var src = snapshot || LastThreatSnapshot || BuildThreatSnapshot();
    var at = typeof sampleTime === "number" ? sampleTime : Date.now();
    var leftMap = GetMapIDByRelativePoint(mapPoint.X - 12, mapPoint.Y + 16);
    var rightMap = GetMapIDByRelativePoint(mapPoint.X + 12, mapPoint.Y + 16);
    var leftNo = leftMap ? leftMap.Y * 15 + leftMap.X : -1;
    var rightNo = rightMap ? rightMap.Y * 15 + rightMap.X : -1;
    var leftUnsafe = IsMapNoUnsafeAtTime(leftNo, src, at);
    var rightUnsafe = IsMapNoUnsafeAtTime(rightNo, src, at);

    return {
        leftNo: leftNo,
        rightNo: rightNo,
        leftUnsafe: leftUnsafe,
        rightUnsafe: rightUnsafe,
        isSafe: !leftUnsafe && !rightUnsafe,
        isHalfSafe: leftUnsafe !== rightUnsafe
    };
}

function IsMapNoUnsafeAtTime(mapNo, snapshot, at) {
    var src = snapshot || LastThreatSnapshot;
    var i;
    var cluster;
    var active;

    if (!src || !src.eventSnapshot || mapNo < 0) {
        return false;
    }

    for (i = 0; i < src.eventSnapshot.activeWindows.length; i++) {
        active = src.eventSnapshot.activeWindows[i];
        if (active.startAt <= at && active.endAt >= at && active.coverageMapIds.indexOf(mapNo) !== -1) {
            return true;
        }
    }

    for (i = 0; i < src.eventSnapshot.clusters.length; i++) {
        cluster = src.eventSnapshot.clusters[i];
        if (cluster.startAt <= at && cluster.endAt >= at && cluster.coverageMapIds.indexOf(mapNo) !== -1) {
            return true;
        }
    }

    return false;
}

function GetUnsafeBufferPenaltyByMapNo(mapNo, snapshot) {
    var src = snapshot || LastThreatSnapshot;
    var key;
    var eta;
    if (!src || mapNo < 0) {
        return 9999;
    }
    key = MapKey(mapNo % 15, parseInt(mapNo / 15, 10));
    eta = src.dangerEtaMap ? src.dangerEtaMap[key] : null;
    if (typeof eta !== "number") {
        return 9999;
    }
    return eta;
}

function PickNextFrameMovementChoice(role, currentMap, snapshot, options) {
    var mapPoint;
    var nextFrameMs = Math.max(16, Math.round(1000 / 60));
    var step = Math.max(1, role.MoveStep || 1);
    var candidates;
    var i;
    var c;
    var futureMapPoint;
    var footState;
    var nextMap;
    var travelPenalty;
    var unsafeEtaPenalty;
    var best = null;
    var bestScore = Infinity;
    var score;
    var safeRank;
    var halfBodyBufferOk;
    var opts = options || {};
    var minExits = typeof opts.minExits === "number" ? Math.max(1, opts.minExits) : 1;
    var exits;

    mapPoint = role.MapPoint();
    candidates = [
        { direction: null, dx: 0, dy: 0 },
        { direction: Direction.Up, dx: 0, dy: -step },
        { direction: Direction.Down, dx: 0, dy: step },
        { direction: Direction.Left, dx: -step, dy: 0 },
        { direction: Direction.Right, dx: step, dy: 0 }
    ];

    for (i = 0; i < candidates.length; i++) {
        c = candidates[i];
        futureMapPoint = {
            X: mapPoint.X + c.dx,
            Y: mapPoint.Y + c.dy
        };
        nextMap = GetMapIDByRelativePoint(futureMapPoint.X, futureMapPoint.Y);
        if (!nextMap) {
            continue;
        }
        if (opts.disallowStay && c.direction == null) {
            continue;
        }
        if (c.direction != null && !IsAIWalkable(nextMap.X, nextMap.Y)) {
            continue;
        }

        footState = GetRoleFootStateFromMapPoint(futureMapPoint, snapshot, Date.now() + nextFrameMs);
        if (footState.isSafe) {
            safeRank = 0;
        }
        else if (footState.isHalfSafe) {
            safeRank = 1;
        }
        else {
            safeRank = 2;
        }

        travelPenalty = c.direction == null ? 0 : 1;
        exits = CountSafeNeighborTiles(nextMap.X, nextMap.Y, snapshot);
        unsafeEtaPenalty = Math.min(
            GetUnsafeBufferPenaltyByMapNo(footState.leftNo, snapshot),
            GetUnsafeBufferPenaltyByMapNo(footState.rightNo, snapshot)
        );
        if (unsafeEtaPenalty === 9999) {
            unsafeEtaPenalty = 5000;
        }
        halfBodyBufferOk = safeRank !== 1 || unsafeEtaPenalty >= AIDodgePolicy.halfBodyMinEtaMs;
        // 先保证安全级，再最短位移，再尽量增大危险缓冲
        score = safeRank * 100000 + travelPenalty * 1000 + (5000 - unsafeEtaPenalty);
        // 半身可行但缓冲不足时，不要“硬蹭边”，优先找全安全步
        if (!halfBodyBufferOk) {
            score += 65000;
        }
        if (exits < minExits) {
            score += (minExits - exits) * 4200;
        }
        if (score < bestScore) {
            bestScore = score;
            best = {
                direction: c.direction,
                targetMap: { X: nextMap.X, Y: nextMap.Y },
                safeRank: safeRank,
                footState: footState,
                unsafeEtaPenalty: unsafeEtaPenalty,
                halfBodyBufferOk: halfBodyBufferOk,
                safeNeighbors: exits
            };
        }
    }

    return best;
}

function ApplyNextFrameEvadeChoice(monster, choice) {
    if (!monster || !choice) {
        return false;
    }
    if (choice.direction == null) {
        return true;
    }
    monster.MoveToMap(choice.targetMap, true);
    return true;
}

function IsTileThreatSoon(key, travelMs, snapshot) {
    var source = snapshot || LastThreatSnapshot;
    var eta;
    if (!source || !source.dangerEtaMap) {
        return false;
    }
    eta = source.dangerEtaMap[key];
    if (typeof eta !== "number") {
        return false;
    }
    return eta <= (travelMs + AIDodgePolicy.safeBufferMs);
}

function IsTileUnsafeNow(key, snapshot) {
    var src = snapshot || LastThreatSnapshot;
    if (!src || !src.threatMap) {
        return false;
    }
    return !!src.threatMap[key];
}

function EstimateEscapeSteps(startX, startY, snapshot, minExits, maxDepth) {
    var src = snapshot || LastThreatSnapshot || BuildThreatSnapshot();
    var queue = [{ x: startX, y: startY, d: 0 }];
    var visited = {};
    var head = 0;
    var targetExits = typeof minExits === "number" ? Math.max(1, minExits) : 2;
    var depthCap = typeof maxDepth === "number" ? Math.max(2, maxDepth) : 12;

    visited[MapKey(startX, startY)] = true;

    while (head < queue.length) {
        var cur = queue[head++];
        var key = MapKey(cur.x, cur.y);
        var exits = CountSafeNeighborTiles(cur.x, cur.y, src);
        if (!IsTileUnsafeNow(key, src)
            && !IsTileThreatSoon(key, cur.d * 240, src)
            && exits >= targetExits) {
            return cur.d;
        }
        if (cur.d >= depthCap) {
            continue;
        }
        for (var i = 0; i < 4; i++) {
            var nx = cur.x + DIRS[i].dx;
            var ny = cur.y + DIRS[i].dy;
            var nk = MapKey(nx, ny);
            if (visited[nk] || !IsAIWalkable(nx, ny)) {
                continue;
            }
            visited[nk] = true;
            queue.push({ x: nx, y: ny, d: cur.d + 1 });
        }
    }
    return null;
}

function CountUpcomingUnsafeNeighbors(x, y, snapshot) {
    var src = snapshot || LastThreatSnapshot || BuildThreatSnapshot();
    var count = 0;
    var i;
    var nx;
    var ny;
    var nk;

    for (i = 0; i < 4; i++) {
        nx = x + DIRS[i].dx;
        ny = y + DIRS[i].dy;
        nk = MapKey(nx, ny);
        if (!IsAIWalkable(nx, ny)) {
            count++;
            continue;
        }
        if (IsTileUnsafeNow(nk, src) || IsTileThreatSoon(nk, 260, src)) {
            count++;
        }
    }
    return count;
}

function GetTrainingGuardDirectives() {
    if (typeof AIDodgeTrainer === "undefined" || !AIDodgeTrainer || !AIDodgeTrainer.IsRunning) {
        return null;
    }
    if (typeof AIDodgeTrainer.GetActiveGuardDirectives !== "function") {
        return null;
    }
    return AIDodgeTrainer.GetActiveGuardDirectives();
}


function DistanceToNearestBomb(x, y) {
    var best = Infinity;
    for (var row = 0; row < PaopaoArray.length; row++) {
        if (!PaopaoArray[row]) continue;
        for (var col = 0; col < PaopaoArray[row].length; col++) {
            var bomb = PaopaoArray[row][col];
            var d;
            if (!bomb || bomb.IsExploded) continue;
            d = ManhattanDist(x, y, col, row);
            if (d < best) {
                best = d;
            }
        }
    }
    return best === Infinity ? 99 : best;
}

function CountActiveBombs() {
    var count = 0;
    for (var row = 0; row < PaopaoArray.length; row++) {
        if (!PaopaoArray[row]) continue;
        for (var col = 0; col < PaopaoArray[row].length; col++) {
            if (PaopaoArray[row][col] && !PaopaoArray[row][col].IsExploded) {
                count++;
            }
        }
    }
    return count;
}

function CountSafeNeighborTiles(x, y, snapshot) {
    var src = snapshot || LastThreatSnapshot || BuildThreatSnapshot();
    var count = 0;
    var i;
    for (i = 0; i < 4; i++) {
        var nx = x + DIRS[i].dx;
        var ny = y + DIRS[i].dy;
        var nk = MapKey(nx, ny);
        if (!IsAIWalkable(nx, ny)) continue;
        if (src.threatMap && src.threatMap[nk]) continue;
        if (IsTileThreatSoon(nk, 240, src)) continue;
        count++;
    }
    return count;
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
    totalScore: 1,
    matchScore: 0,
    killReward: 3,
    deathPenalty: 2,
    aggression: 0.5,
    caution: 0.5,
    trainingRounds: 0,
    positiveEpisodes: 0,
    bombCooldownMs: 800,
    tacticWeights: {
        evade: 0.9,
        trap: 0.75,
        vAttack: 0.7,
        chainBomb: 0.68,
        halfBody: 0.62
    },
    lastDecay: 0,

    clamp: function(v, min, max) {
        if (v < min) return min;
        if (v > max) return max;
        return v;
    },

    clamp01: function(v) {
        return this.clamp(v, 0, 1);
    },

    scoreByKD: function(k, d) {
        return k * this.killReward - d * this.deathPenalty;
    },

    normalizeTactics: function() {
        this.tacticWeights.evade = this.clamp(this.tacticWeights.evade, 0.3, 1.5);
        this.tacticWeights.trap = this.clamp(this.tacticWeights.trap, 0.2, 1.4);
        this.tacticWeights.vAttack = this.clamp(this.tacticWeights.vAttack, 0.2, 1.4);
        this.tacticWeights.chainBomb = this.clamp(this.tacticWeights.chainBomb, 0.2, 1.4);
        this.tacticWeights.halfBody = this.clamp(this.tacticWeights.halfBody, 0.2, 1.3);
    },

    refreshBombCooldown: function() {
        var tacticPressure = this.tacticWeights.chainBomb * 0.45 + this.tacticWeights.vAttack * 0.3 + this.tacticWeights.trap * 0.25;
        this.bombCooldownMs = this.clamp(Math.round(900 - tacticPressure * 420), 320, 900);
    },

    save: function() {
        try {
            localStorage.setItem("bnb_ai_evo", JSON.stringify({
                dh: this.deathHeatMap, ph: this.playerHeatMap,
                k: this.totalKills, d: this.totalDeaths,
                s: this.totalScore, ms: this.matchScore,
                a: this.aggression, c: this.caution,
                tr: this.trainingRounds, pe: this.positiveEpisodes,
                tw: this.tacticWeights, bc: this.bombCooldownMs
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
                this.totalScore = typeof d.s === "number" ? d.s : this.scoreByKD(this.totalKills, this.totalDeaths);
                this.matchScore = typeof d.ms === "number" ? d.ms : 0;
                this.aggression = typeof d.a === "number" ? d.a : 0.5;
                this.caution = typeof d.c === "number" ? d.c : 0.5;
                this.trainingRounds = d.tr || 0;
                this.positiveEpisodes = d.pe || 0;
                if (d.tw) {
                    this.tacticWeights.evade = typeof d.tw.evade === "number" ? d.tw.evade : this.tacticWeights.evade;
                    this.tacticWeights.trap = typeof d.tw.trap === "number" ? d.tw.trap : this.tacticWeights.trap;
                    this.tacticWeights.vAttack = typeof d.tw.vAttack === "number" ? d.tw.vAttack : this.tacticWeights.vAttack;
                    this.tacticWeights.chainBomb = typeof d.tw.chainBomb === "number" ? d.tw.chainBomb : this.tacticWeights.chainBomb;
                    this.tacticWeights.halfBody = typeof d.tw.halfBody === "number" ? d.tw.halfBody : this.tacticWeights.halfBody;
                }
                if (typeof d.bc === "number") {
                    this.bombCooldownMs = d.bc;
                }
            }
        } catch (e) {}
        this.normalizeTactics();
        this.refreshBombCooldown();
        if (this.totalScore <= 0) {
            this.totalScore = 1;
        }
    },

    startMatch: function() {
        this.matchScore = 0;
    },

    finalizeMatch: function() {
        if (this.matchScore <= 0) {
            this.runBootcamp(30);
            this.matchScore = 1;
        }
        this.totalScore = Math.max(1, this.scoreByKD(this.totalKills, this.totalDeaths));
        this.save();
    },

    simulateEpisode: function(policy) {
        var kills = 0;
        var deaths = 0;

        for (var i = 0; i < 70; i++) {
            var threatPressure = Math.random();
            var playerExposure = Math.random();
            var evadeScore = policy.caution * 0.72 + policy.tactic.evade * 0.48 + policy.tactic.halfBody * 0.22 - threatPressure * 0.95 + (Math.random() - 0.5) * 0.22;
            var attackScore = policy.aggression * 0.66 + policy.tactic.trap * 0.42 + policy.tactic.vAttack * 0.35 + policy.tactic.chainBomb * 0.35 - policy.caution * 0.18 + playerExposure * 0.25 + (Math.random() - 0.5) * 0.2;

            if (evadeScore < -0.08) {
                deaths++;
            }
            if (attackScore > 0.88) {
                kills++;
            }
            if (policy.tactic.chainBomb > 0.75 && attackScore > 0.86 && Math.random() < 0.32) {
                kills++;
            }
            if (policy.tactic.vAttack > 0.75 && attackScore > 0.84 && Math.random() < 0.2) {
                kills++;
            }
        }

        if (deaths === 0) {
            kills++;
        }

        return {
            kills: kills,
            deaths: deaths,
            score: this.scoreByKD(kills, deaths)
        };
    },

    runBootcamp: function(rounds) {
        var totalRounds = rounds || AIBootcampRounds;
        var positives = 0;

        for (var i = 0; i < totalRounds; i++) {
            var policy = {
                aggression: this.clamp01(this.aggression + (Math.random() - 0.5) * 0.45),
                caution: this.clamp01(this.caution + (Math.random() - 0.5) * 0.45),
                tactic: {
                    evade: this.clamp(this.tacticWeights.evade + (Math.random() - 0.5) * 0.5, 0.3, 1.5),
                    trap: this.clamp(this.tacticWeights.trap + (Math.random() - 0.5) * 0.5, 0.2, 1.4),
                    vAttack: this.clamp(this.tacticWeights.vAttack + (Math.random() - 0.5) * 0.5, 0.2, 1.4),
                    chainBomb: this.clamp(this.tacticWeights.chainBomb + (Math.random() - 0.5) * 0.5, 0.2, 1.4),
                    halfBody: this.clamp(this.tacticWeights.halfBody + (Math.random() - 0.5) * 0.5, 0.2, 1.3)
                }
            };
            var result = this.simulateEpisode(policy);

            if (result.score > 0) {
                var learnRate = this.clamp(0.03 + result.score * 0.01, 0.03, 0.22);
                positives++;

                this.aggression += (policy.aggression - this.aggression) * learnRate;
                this.caution += (policy.caution - this.caution) * learnRate;
                this.tacticWeights.evade += (policy.tactic.evade - this.tacticWeights.evade) * learnRate;
                this.tacticWeights.trap += (policy.tactic.trap - this.tacticWeights.trap) * learnRate;
                this.tacticWeights.vAttack += (policy.tactic.vAttack - this.tacticWeights.vAttack) * learnRate;
                this.tacticWeights.chainBomb += (policy.tactic.chainBomb - this.tacticWeights.chainBomb) * learnRate;
                this.tacticWeights.halfBody += (policy.tactic.halfBody - this.tacticWeights.halfBody) * learnRate;
                this.totalScore += result.score;
            }
            else {
                this.caution = this.clamp01(this.caution + 0.01);
                this.tacticWeights.evade = this.clamp(this.tacticWeights.evade + 0.015, 0.3, 1.5);
                this.tacticWeights.halfBody = this.clamp(this.tacticWeights.halfBody + 0.01, 0.2, 1.3);
            }
        }

        this.trainingRounds += totalRounds;
        this.positiveEpisodes += positives;
        this.aggression = this.clamp01(this.aggression);
        this.caution = this.clamp01(this.caution);
        this.normalizeTactics();
        this.refreshBombCooldown();

        if (this.totalScore <= 0) {
            this.totalScore = 1;
        }
        if (this.matchScore <= 0 && positives > 0) {
            this.matchScore = 1;
        }

        this.save();
    },

    getBombCooldownMs: function() {
        return this.bombCooldownMs;
    },

    getEvadeWeight: function() {
        return this.tacticWeights.evade;
    },

    getTrapPreference: function() {
        return this.tacticWeights.trap;
    },

    getVAttackPreference: function() {
        return this.tacticWeights.vAttack;
    },

    getChainBombPreference: function() {
        return this.tacticWeights.chainBomb;
    },

    getHalfBodyPreference: function() {
        return this.tacticWeights.halfBody;
    },

    onDeath: function(x, y) {
        var k = MapKey(x, y);
        this.deathHeatMap[k] = (this.deathHeatMap[k] || 0) + 1;
        this.totalDeaths++;
        this.matchScore -= this.deathPenalty;
        this.totalScore -= this.deathPenalty;
        this.caution = Math.min(1, this.caution + 0.06);
        this.aggression = Math.max(0.1, this.aggression - 0.04);
        this.tacticWeights.evade = Math.min(1.5, this.tacticWeights.evade + 0.04);
        this.tacticWeights.halfBody = Math.min(1.3, this.tacticWeights.halfBody + 0.03);
        this.normalizeTactics();
        this.refreshBombCooldown();
        this.save();
    },

    onKill: function() {
        this.totalKills++;
        this.matchScore += this.killReward;
        this.totalScore += this.killReward;
        this.aggression = Math.min(0.95, this.aggression + 0.03);
        this.caution = Math.max(0.1, this.caution - 0.02);
        this.tacticWeights.trap = Math.min(1.4, this.tacticWeights.trap + 0.03);
        this.tacticWeights.vAttack = Math.min(1.4, this.tacticWeights.vAttack + 0.02);
        this.tacticWeights.chainBomb = Math.min(1.4, this.tacticWeights.chainBomb + 0.025);
        this.normalizeTactics();
        this.refreshBombCooldown();
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
LoadAIDodgePolicy();
AIEvolution.runBootcamp(AIBootcampRounds);

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
    this.Role.SetMoveSpeedPxPerSec(RoleBalanceConfig.InitialSpeedPxPerSec);
    this.Role.PaopaoStrong = ClampRolePower(RoleBalanceConfig.InitialPower);
    this.Role.CanPaopaoLength = ClampRoleBubbleCount(RoleBalanceConfig.InitialBubbleCount);

    this.ThinkInterval = null;
    this.LastBombAt = 0;
    this.LastTargetKey = "";
    this.LastMoveIssuedAt = 0;
    this.LastProgressAt = Date.now();
    this.LastMapKey = "";
    this.StuckCount = 0;
    this.AttackPlan = null;
    this.State = "idle";
    this.CurrentTemporalPlan = null;
    this.LastTemporalSceneRevision = 0;
    this.TemporalPlanStepStallTicks = 0;
    this.TemporalPlanStepStallSince = 0;
    this.TemporalPlanStepKey = "";
    this.LastTemporalReplanAt = 0;
    this.TemporalFallbackTriggerCount = 0;
    this.ActiveThinkIntervalMs = MonsterThinkInterval;
};

Monster.prototype.SetMap = function(x, y) {
    this.Role.SetToMap(x, y);
};

Monster.prototype.MoveToMap = function(pos, forceRetry) {
    if (!pos) return;
    var now = Date.now();
    var key = pos.X + "_" + pos.Y;
    if (!forceRetry && this.LastTargetKey === key && now - this.LastMoveIssuedAt < AIDodgePolicy.repathMs) return;
    this.LastTargetKey = key;
    this.LastMoveIssuedAt = now;
    if (!this.Role.MoveTo(pos.X, pos.Y)) {
        this.LastTargetKey = "";
    }
};

Monster.prototype.ClearTarget = function() {
    this.LastTargetKey = "";
};

Monster.prototype.CanDropBomb = function() {
    var cooldown = (typeof AIEvolution !== "undefined" && typeof AIEvolution.getBombCooldownMs === "function")
        ? AIEvolution.getBombCooldownMs()
        : 800;
    return Date.now() - this.LastBombAt > cooldown &&
        this.Role.CanPaopaoLength > this.Role.PaopaoCount;
};

Monster.prototype.DropBomb = function() {
    this.LastBombAt = Date.now();
    if (typeof AIEvolution !== "undefined" && AIEvolution.getChainBombPreference && AIEvolution.getChainBombPreference() > 0.8) {
        this.LastBombAt -= 120;
    }
    this.Role.PaoPao();
    this.ClearTarget();
};

Monster.prototype.GetThinkIntervalMs = function() {
    var trainer = (typeof AIDodgeTrainer !== "undefined" && AIDodgeTrainer) ? AIDodgeTrainer : null;
    if (trainer && typeof trainer.IsMonsterTraining === "function" && trainer.IsMonsterTraining(this)) {
        return trainer.Config && typeof trainer.Config.trainingThinkIntervalMs === "number"
            ? trainer.Config.trainingThinkIntervalMs
            : MonsterTrainingThinkInterval;
    }
    return MonsterThinkInterval;
};

Monster.prototype.Start = function() {
    var self = this;
    var thinkMs = this.GetThinkIntervalMs();
    if (self.ThinkInterval) {
        clearInterval(self.ThinkInterval);
        self.ThinkInterval = null;
    }
    self.ActiveThinkIntervalMs = thinkMs;
    self.Think();
    self.ThinkInterval = setInterval(function() {
        if (self.Role.IsDeath) {
            self.Stop();
            return;
        }
        // 被困泡时仅暂停决策，不销毁 ThinkInterval，避免脱困后 AI 永久停摆
        if (self.Role.IsInPaopao) {
            self.Role.Stop();
            return;
        }
        self.Think();
    }, thinkMs);
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
    var threatSnapshot;
    var threatMap;
    var currentKey;
    var player = GetSinglePlayerPlayer();
    var now = Date.now();
    var stuckTarget;
    if (!currentMap) return;
    currentKey = MapKey(currentMap.X, currentMap.Y);

    threatSnapshot = BuildThreatSnapshot();
    threatMap = threatSnapshot.threatMap;

    // 已到达目标 → 清除，允许重新选择
    if (this.LastTargetKey === currentKey) {
        this.LastTargetKey = "";
    }
    if (currentKey !== this.LastMapKey) {
        this.LastMapKey = currentKey;
        this.LastProgressAt = now;
        this.StuckCount = 0;
    }
    else if (now - this.LastProgressAt > AIDodgePolicy.stuckTimeoutMs) {
        this.LastProgressAt = now;
        this.StuckCount++;
        this.ClearTarget();
        stuckTarget = this.FindSafeTile(currentMap, threatMap, threatSnapshot);
        if (!stuckTarget) {
            stuckTarget = this.FindTrainingPatrolTile(currentMap, threatMap, threatSnapshot);
        }
        if (stuckTarget) {
            this.State = "unstick";
            this.MoveToMap(stuckTarget, true);
            return;
        }
    }

    AIEvolution.decay();
    if (player && !player.IsDeath) {
        AIEvolution.updatePlayer(player.CurrentMapID());
    }

    if (typeof AIDodgeTrainer !== "undefined" && AIDodgeTrainer.IsMonsterTraining(this)) {
        this.ThinkDodgeTraining(currentMap, threatMap, threatSnapshot);
        return;
    }

    var mlDecision = this.TryOfflineMLDodgeAction(currentMap, threatSnapshot, "battle");
    if (mlDecision && mlDecision.handled) {
        return;
    }

    // ───── 1. EVADE — 处于危险区时立即逃跑 ─────
    if (threatMap[currentKey]) {
        this.AttackPlan = null;
        this.State = "evade";
        var safe = this.FindSafeTile(currentMap, threatMap, threatSnapshot);
        if (safe) {
            this.MoveToMap(safe);
            if (BNBMLRuntime && typeof BNBMLRuntime.RecordRuleDecision === "function") {
                BNBMLRuntime.RecordRuleDecision(
                    currentMap,
                    safe,
                    mlDecision && mlDecision.attempted ? ("fallback_" + (mlDecision.reason || "unknown")) : "rule_evade"
                );
            }
        }
        else {
            if (BNBMLRuntime && typeof BNBMLRuntime.RecordRuleDecision === "function") {
                BNBMLRuntime.RecordRuleDecision(
                    currentMap,
                    currentMap,
                    mlDecision && mlDecision.attempted ? ("fallback_" + (mlDecision.reason || "unknown") + "_wait") : "rule_evade_wait"
                );
            }
        }
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

    // ───── 3.5 HALF-BODY — 半身走位微调（优先拉开爆炸线） ─────
    if (player && !player.IsDeath) {
        if (this.TryHalfBodyShift(currentMap, player, threatMap)) {
            this.State = "half_body";
            return;
        }
    }

    // ───── 4. V-ATTACK — 多泡夹击攻击 ─────
    if (this.AttackPlan) {
        this.ExecuteVAttackPlan(currentMap, threatMap);
        return;
    }
    if (player && !player.IsDeath &&
        role.CanPaopaoLength >= 2 && role.PaopaoCount === 0 &&
        AIEvolution.aggression > 0.22 &&
        AIEvolution.getVAttackPreference() > 0.45) {
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

Monster.prototype.GetActiveTrainingTrainer = function() {
    if (typeof AIDodgeTrainer === "undefined" || !AIDodgeTrainer) {
        return null;
    }
    if (!AIDodgeTrainer.IsMonsterTraining(this)) {
        return null;
    }
    return AIDodgeTrainer;
};

Monster.prototype.TryOfflineMLDodgeAction = function(currentMap, threatSnapshot, modeTag) {
    var snapshot = threatSnapshot || LastThreatSnapshot || BuildThreatSnapshot();
    var decision;
    var action;
    var targetMap;
    if (!currentMap || !BNBMLRuntime || !BNBMLRuntime.ShouldUseContext(currentMap, snapshot)) {
        return { handled: false, attempted: false, reason: "context_not_applicable" };
    }
    decision = BNBMLRuntime.DecideAction(this, currentMap, snapshot);
    if (!decision || !decision.ok) {
        return {
            handled: false,
            attempted: true,
            reason: decision && decision.reason ? decision.reason : "decision_failed"
        };
    }
    action = NormalizeActionId(decision.action);
    if (action === 0) {
        this.Role.Stop();
        this.State = modeTag === "training" ? "ml_dodge_training_wait" : "ml_dodge_wait";
        if (modeTag === "battle" && typeof BNBMLRuntime.RecordModelDecision === "function") {
            BNBMLRuntime.RecordModelDecision(currentMap, currentMap, action, decision.confidence);
        }
        return { handled: true, attempted: true, reason: "model_wait", action: action };
    }
    targetMap = BuildTargetMapByAction(currentMap, action);
    if (!targetMap || !IsAIWalkable(targetMap.X, targetMap.Y)) {
        return { handled: false, attempted: true, reason: "model_action_invalid_target" };
    }
    this.MoveToMap(targetMap, true);
    this.State = modeTag === "training" ? "ml_dodge_training_move" : "ml_dodge_move";
    if (modeTag === "battle" && typeof BNBMLRuntime.RecordModelDecision === "function") {
        BNBMLRuntime.RecordModelDecision(currentMap, targetMap, action, decision.confidence);
    }
    return { handled: true, attempted: true, reason: "model_move", action: action };
};

Monster.prototype.ResetTemporalPlanState = function(sceneRevision) {
    this.CurrentTemporalPlan = null;
    this.LastTemporalSceneRevision = typeof sceneRevision === "number" ? sceneRevision : GetThreatSceneRevision();
    this.TemporalPlanStepStallTicks = 0;
    this.TemporalPlanStepStallSince = 0;
    this.TemporalPlanStepKey = "";
    this.LastTemporalReplanAt = 0;
    this.TemporalFallbackTriggerCount = 0;
};

Monster.prototype.BuildTemporalPlanPathPreview = function(plan, maxNodes) {
    var route = plan && Array.isArray(plan.route) ? plan.route : [];
    var preview = [];
    var limit = typeof maxNodes === "number" ? maxNodes : 20;
    var i;
    var node;
    var mode;
    var mark;
    for (i = 0; i < route.length && i < limit; i++) {
        node = route[i];
        mode = node.safetyMode === "half" ? "H" : "S";
        mark = node.action === "wait" ? "W" : "M";
        preview.push("(" + node.x + "," + node.y + "," + mode + "," + mark + ")");
    }
    if (route.length > limit) {
        preview.push("...");
    }
    return preview.join("->");
};

Monster.prototype.NotifyTemporalPlanBuilt = function(plan, triggerReason, temporalModel, sceneRevision) {
    var trainer = this.GetActiveTrainingTrainer();
    var bubble = GetLastBubbleSpawnEvent();
    if (!trainer || typeof trainer.OnTemporalPlanBuilt !== "function") {
        return;
    }
    trainer.OnTemporalPlanBuilt({
        reason: triggerReason,
        sceneRevision: sceneRevision,
        bubbleEvent: bubble,
        plan: plan,
        timeline: temporalModel,
        preview: this.BuildTemporalPlanPathPreview(plan, 20)
    });
};

Monster.prototype.BuildTemporalPlanForTraining = function(currentMap, threatSnapshot, triggerReason) {
    var trainer = this.GetActiveTrainingTrainer();
    var guard = trainer && typeof trainer.GetActiveGuardDirectives === "function"
        ? trainer.GetActiveGuardDirectives()
        : null;
    var sceneRevision = GetThreatSceneRevision();
    var now = Date.now();
    var temporalModel = BuildTemporalSafetyModel(threatSnapshot, now);
    var startEval = EvaluateTileTemporalSafety(currentMap.X, currentMap.Y, now, temporalModel);
    var startExits = CountTemporalSafeNeighborTiles(currentMap.X, currentMap.Y, now, temporalModel);
    var snapshotRef = threatSnapshot || LastThreatSnapshot || BuildThreatSnapshot();
    var unsafeNeighbors = CountUpcomingUnsafeNeighbors(currentMap.X, currentMap.Y, snapshotRef);
    var activeBombs = CountActiveBombs();
    var currentKey = MapKey(currentMap.X, currentMap.Y);
    var currentEta = snapshotRef && snapshotRef.dangerEtaMap && typeof snapshotRef.dangerEtaMap[currentKey] === "number"
        ? snapshotRef.dangerEtaMap[currentKey]
        : null;
    var preferMove = triggerReason === "new_bubble_event"
        || triggerReason === "path_blocked_fallback"
        || triggerReason === "panic_eta_replan"
        || triggerReason === "low_exit_replan";
    var forceMove = false;
    var panicEtaThreshold = Math.max(
        120,
        AIDodgePolicy.safeBufferMs + ((guard && guard.panicEtaBoost) ? guard.panicEtaBoost : 0) + 180
    );
    var panicMode = typeof currentEta === "number" && currentEta <= panicEtaThreshold;
    var goalMinExits = 1;
    var minGoalMoveSteps = 0;
    var minGoalDisplacement = 0;
    var stayPenaltyMs = preferMove ? Math.max(180, AIDodgePolicy.safeBufferMs + 10) : 0;
    var minHoldMs = Math.max(50, Math.round(AIDodgePolicy.safeBufferMs * 0.25));
    var goalMinHoldMs = Math.max(240, AIDodgePolicy.safeBufferMs + 40);
    var allowHalfBody = !(guard && guard.disableHalfBody);

    if (guard) {
        goalMinExits = Math.max(goalMinExits, guard.minSafeNeighbors || 1);
        if (guard.preferFarSafe) {
            goalMinExits = Math.max(goalMinExits, 2);
        }
    }
    if (startEval.safeRank > 0 || startEval.safeDurationMs < Math.max(260, AIDodgePolicy.safeBufferMs + 120)) {
        forceMove = true;
    }
    if (unsafeNeighbors >= 2 || startExits <= 1 || activeBombs >= 4) {
        forceMove = true;
    }
    if (triggerReason === "path_blocked_fallback") {
        forceMove = true;
    }
    if (triggerReason === "panic_eta_replan") {
        forceMove = true;
        goalMinExits = Math.max(goalMinExits, 2);
        minGoalMoveSteps = Math.max(minGoalMoveSteps, 2);
        minGoalDisplacement = Math.max(minGoalDisplacement, 1);
    }
    if (triggerReason === "low_exit_replan") {
        forceMove = true;
        goalMinExits = Math.max(goalMinExits, 3);
        minGoalMoveSteps = Math.max(minGoalMoveSteps, 2);
        minGoalDisplacement = Math.max(minGoalDisplacement, 2);
    }
    if (guard && guard.disallowStay) {
        forceMove = true;
    }
    if (startExits <= 1 || unsafeNeighbors >= 2) {
        goalMinExits = Math.max(goalMinExits, 2);
    }
    if (activeBombs >= 5 || unsafeNeighbors >= 3) {
        goalMinExits = Math.max(goalMinExits, 3);
    }
    if (forceMove) {
        minGoalMoveSteps = 1;
        minGoalDisplacement = 1;
    }
    if (forceMove && (startExits <= 1 || activeBombs >= 4 || unsafeNeighbors >= 3)) {
        minGoalMoveSteps = 2;
        minGoalDisplacement = 2;
    }
    if (triggerReason === "path_blocked_fallback") {
        minGoalMoveSteps = Math.max(minGoalMoveSteps, 2);
        minGoalDisplacement = Math.max(minGoalDisplacement, 2);
        goalMinExits = Math.max(goalMinExits, 2);
    }
    if (triggerReason === "low_exit_replan") {
        goalMinExits = Math.max(goalMinExits, 3);
    }
    if (activeBombs >= 4 && (startExits <= 1 || unsafeNeighbors >= 2)) {
        allowHalfBody = false;
    }
    if (panicMode) {
        forceMove = true;
        goalMinExits = Math.max(goalMinExits, 2);
        minGoalMoveSteps = 1;
        minGoalDisplacement = 1;
        stayPenaltyMs += 2600;
        minHoldMs = Math.max(35, Math.round(AIDodgePolicy.safeBufferMs * 0.16));
        goalMinHoldMs = Math.max(140, Math.round(AIDodgePolicy.safeBufferMs * 0.55));
    }
    if (forceMove) {
        stayPenaltyMs += 1400;
    }

    var plan = BuildTemporalEvadePlan(this.Role, currentMap, temporalModel, {
        maxHorizonMs: Math.max(2800, AIDodgePolicy.forecastMs + 1400),
        minHoldMs: minHoldMs,
        goalMinHoldMs: goalMinHoldMs,
        goalMinExits: goalMinExits,
        allowHalfBody: allowHalfBody,
        preferMove: preferMove,
        forceMove: forceMove,
        panicMode: panicMode,
        minGoalMoveSteps: minGoalMoveSteps,
        minGoalDisplacement: minGoalDisplacement,
        stayPenaltyMs: stayPenaltyMs,
        moveBonusMs: Math.max(80, Math.round(400 / Math.max(1, this.Role.MoveStep || 1))),
        preferMoveSlackMs: Math.max(220, AIDodgePolicy.safeBufferMs),
        preferStayDurationMs: Math.max(900, AIDodgePolicy.safeBufferMs + 520)
    });

    if (!plan && forceMove) {
        plan = BuildTemporalEvadePlan(this.Role, currentMap, temporalModel, {
            maxHorizonMs: Math.max(2600, AIDodgePolicy.forecastMs + 1200),
            minHoldMs: Math.max(25, Math.round(minHoldMs * 0.65)),
            goalMinHoldMs: Math.max(120, Math.round(goalMinHoldMs * 0.7)),
            goalMinExits: Math.max(1, goalMinExits - 1),
            allowHalfBody: allowHalfBody,
            preferMove: true,
            forceMove: true,
            panicMode: panicMode,
            minGoalMoveSteps: 1,
            minGoalDisplacement: 1,
            stayPenaltyMs: stayPenaltyMs + 900,
            moveBonusMs: Math.max(80, Math.round(340 / Math.max(1, this.Role.MoveStep || 1))),
            preferMoveSlackMs: Math.max(180, Math.round(AIDodgePolicy.safeBufferMs * 0.75)),
            preferStayDurationMs: Math.max(720, AIDodgePolicy.safeBufferMs + 380)
        });
    }

    this.LastTemporalSceneRevision = sceneRevision;
    this.LastTemporalReplanAt = Date.now();
    this.TemporalPlanStepStallTicks = 0;
    this.TemporalPlanStepStallSince = 0;
    this.TemporalPlanStepKey = "";
    this.CurrentTemporalPlan = plan;
    if (plan) {
        plan.sceneRevision = sceneRevision;
        plan.triggerReason = triggerReason;
        plan.createdAt = now;
        plan.temporalModel = temporalModel;
        plan.planContext = {
            startSafeRank: startEval.safeRank,
            startSafeDurationMs: startEval.safeDurationMs,
            startExits: startExits,
            unsafeNeighbors: unsafeNeighbors,
            activeBombs: activeBombs,
            currentEta: currentEta,
            panicEtaThreshold: panicEtaThreshold,
            panicMode: panicMode,
            preferMove: preferMove,
            forceMove: forceMove,
            minGoalMoveSteps: minGoalMoveSteps,
            minGoalDisplacement: minGoalDisplacement,
            guardDisallowStay: !!(guard && guard.disallowStay),
            guardDisableHalfBody: !!(guard && guard.disableHalfBody),
            effectiveAllowHalfBody: allowHalfBody,
            goalMinExits: goalMinExits
        };
        this.NotifyTemporalPlanBuilt(plan, triggerReason, temporalModel, sceneRevision);
        return true;
    }

    this.NotifyTemporalPlanBuilt({
        route: [],
        start: { x: currentMap.X, y: currentMap.Y },
        target: { x: currentMap.X, y: currentMap.Y },
        totalSteps: 0,
        estimatedMs: 0,
        waitSteps: 0,
        moveSteps: 0,
        halfBodySteps: 0,
        expandedStates: 0,
        horizonMs: 0,
        planContext: {
            startSafeRank: startEval.safeRank,
            startSafeDurationMs: startEval.safeDurationMs,
            startExits: startExits,
            unsafeNeighbors: unsafeNeighbors,
            activeBombs: activeBombs,
            currentEta: currentEta,
            panicEtaThreshold: panicEtaThreshold,
            panicMode: panicMode,
            preferMove: preferMove,
            forceMove: forceMove,
            minGoalMoveSteps: minGoalMoveSteps,
            minGoalDisplacement: minGoalDisplacement,
            guardDisallowStay: !!(guard && guard.disallowStay),
            guardDisableHalfBody: !!(guard && guard.disableHalfBody),
            effectiveAllowHalfBody: allowHalfBody,
            goalMinExits: goalMinExits
        }
    }, triggerReason, temporalModel, sceneRevision);
    return false;
};

Monster.prototype.AdvanceTemporalPlanCursor = function(currentMap, now) {
    var plan = this.CurrentTemporalPlan;
    var step;
    if (!plan || !Array.isArray(plan.route)) {
        return false;
    }
    if (typeof plan.cursor !== "number" || plan.cursor < 1) {
        plan.cursor = 1;
    }
    while (plan.cursor < plan.route.length) {
        step = plan.route[plan.cursor];
        if (currentMap.X === step.x && currentMap.Y === step.y && now + 25 >= step.arrivalAt) {
            plan.cursor += 1;
            this.TemporalPlanStepStallTicks = 0;
            this.TemporalPlanStepStallSince = 0;
            this.TemporalPlanStepKey = "";
            continue;
        }
        break;
    }
    return plan.cursor >= plan.route.length;
};

Monster.prototype.IsTemporalPlanBlocked = function(currentMap, now) {
    var plan = this.CurrentTemporalPlan;
    var step;
    var stepKey;
    if (!plan || !Array.isArray(plan.route) || plan.cursor >= plan.route.length) {
        return false;
    }
    step = plan.route[plan.cursor];
    if (currentMap.X === step.x && currentMap.Y === step.y && now + 25 >= step.arrivalAt) {
        this.TemporalPlanStepStallTicks = 0;
        this.TemporalPlanStepStallSince = 0;
        this.TemporalPlanStepKey = "";
        return false;
    }
    stepKey = step.x + "_" + step.y + "_" + plan.cursor;
    if (this.TemporalPlanStepKey !== stepKey) {
        this.TemporalPlanStepKey = stepKey;
        this.TemporalPlanStepStallTicks = 0;
        this.TemporalPlanStepStallSince = now;
        return false;
    }
    this.TemporalPlanStepStallTicks += 1;
    if (!this.TemporalPlanStepStallSince) {
        this.TemporalPlanStepStallSince = now;
    }
    return this.TemporalPlanStepStallTicks >= 2 || (now - this.TemporalPlanStepStallSince > 350);
};

Monster.prototype.ExecuteTemporalPlanStep = function(currentMap, threatMap, threatSnapshot) {
    var plan = this.CurrentTemporalPlan;
    var now = Date.now();
    var step;
    var immediateSafe;
    if (!plan || !Array.isArray(plan.route)) {
        return false;
    }
    if (this.AdvanceTemporalPlanCursor(currentMap, now)) {
        this.Role.Stop();
        return true;
    }
    step = plan.route[plan.cursor];
    if (!step) {
        return true;
    }
    if (step.action === "wait" || (step.x === currentMap.X && step.y === currentMap.Y)) {
        this.Role.Stop();
        return true;
    }
    if (!IsAIWalkable(step.x, step.y)) {
        return false;
    }
    immediateSafe = EvaluateTileTemporalSafety(step.x, step.y, now, plan.temporalModel || BuildTemporalSafetyModel(threatSnapshot, now));
    if (immediateSafe.safeRank > 1) {
        return false;
    }
    this.MoveToMap({ X: step.x, Y: step.y });
    return true;
};

Monster.prototype.ThinkDodgeTraining = function(currentMap, threatMap, threatSnapshot) {
    var now = Date.now();
    var sceneRevision = GetThreatSceneRevision();
    var needReplan = false;
    var replanReason = "";
    var hasBombs = CountActiveBombs() > 0;
    var safeTile;
    var currentKey = MapKey(currentMap.X, currentMap.Y);
    var currentEta = threatSnapshot && threatSnapshot.dangerEtaMap
        && typeof threatSnapshot.dangerEtaMap[currentKey] === "number"
        ? threatSnapshot.dangerEtaMap[currentKey]
        : null;
    var panicReplanEta = Math.max(140, AIDodgePolicy.safeBufferMs + 110);
    var lowExitReplanEta = Math.max(420, AIDodgePolicy.safeBufferMs + 220);
    var currentSafeExits = CountSafeNeighborTiles(currentMap.X, currentMap.Y, threatSnapshot);
    var nextRouteNode = this.CurrentTemporalPlan
        && this.CurrentTemporalPlan.route
        && this.CurrentTemporalPlan.cursor < this.CurrentTemporalPlan.route.length
        ? this.CurrentTemporalPlan.route[this.CurrentTemporalPlan.cursor]
        : null;

    BNBMLDatasetCollector.RecordFrame(this, currentMap, threatSnapshot);
    var mlDecision = this.TryOfflineMLDodgeAction(currentMap, threatSnapshot, "training");
    if (mlDecision && mlDecision.handled) {
        return;
    }

    if (!this.CurrentTemporalPlan && hasBombs && (now - this.LastTemporalReplanAt > 180)) {
        needReplan = true;
        replanReason = "path_blocked_fallback";
    }
    if (sceneRevision !== this.LastTemporalSceneRevision) {
        needReplan = true;
        replanReason = "new_bubble_event";
    }
    else if (this.IsTemporalPlanBlocked(currentMap, now) && (now - this.LastTemporalReplanAt > 120)) {
        needReplan = true;
        replanReason = "path_blocked_fallback";
        this.TemporalFallbackTriggerCount += 1;
    }
    else if (!needReplan
        && hasBombs
        && nextRouteNode
        && nextRouteNode.action === "wait"
        && typeof currentEta === "number"
        && currentEta <= panicReplanEta
        && (now - this.LastTemporalReplanAt > 70)) {
        needReplan = true;
        replanReason = "panic_eta_replan";
    }
    else if (!needReplan
        && hasBombs
        && currentSafeExits <= 1
        && (typeof currentEta !== "number" || currentEta <= lowExitReplanEta)
        && (now - this.LastTemporalReplanAt > 80)) {
        needReplan = true;
        replanReason = "low_exit_replan";
    }

    if (needReplan) {
        this.BuildTemporalPlanForTraining(currentMap, threatSnapshot, replanReason);
    }

    this.State = "temporal_dodge";
    if (this.CurrentTemporalPlan && this.ExecuteTemporalPlanStep(currentMap, threatMap, threatSnapshot)) {
        if (this.CurrentTemporalPlan.cursor < this.CurrentTemporalPlan.route.length) {
            var routeNode = this.CurrentTemporalPlan.route[this.CurrentTemporalPlan.cursor];
            if (routeNode && routeNode.safetyMode === "half") {
                this.State = "half_body_training";
            }
        }
        return;
    }

    safeTile = this.FindSafeTile(currentMap, threatMap, threatSnapshot);
    if (safeTile) {
        this.MoveToMap(safeTile, true);
        this.State = "evade_training";
        return;
    }
    this.Role.Stop();
};

Monster.prototype.FindTrainingPatrolTile = function(currentMap, threatMap, threatSnapshot) {
    var bfs = AIPathBFS(currentMap.X, currentMap.Y);
    var best = null;
    var bestScore = -Infinity;
    var guard = GetTrainingGuardDirectives();
    var minExits = guard ? guard.minSafeNeighbors : 1;
    var preferFarSafe = !!(guard && guard.preferFarSafe);

    for (var key in bfs.dist) {
        var d = bfs.dist[key];
        var pos;
        var score;
        var exits;
        var failureHeat;

        if (d < 1 || d > AIDodgePolicy.roamRadius) continue;
        if (threatMap[key]) continue;
        if (IsTileThreatSoon(key, d * 240, threatSnapshot)) continue;

        pos = ParseKey(key);
        exits = CountSafeNeighborTiles(pos.X, pos.Y, threatSnapshot);
        if (exits < minExits) {
            continue;
        }
        failureHeat = GetTrainingFailureHeatPenalty(pos.X, pos.Y);
        score = DistanceToNearestBomb(pos.X, pos.Y) * 2.05
            + exits * 1.7
            - d * 0.4
            - AIEvolution.getDanger(pos.X, pos.Y) * 2.1
            - failureHeat * 2.6
            + Math.random() * 0.6;
        if (preferFarSafe) {
            score += DistanceToNearestBomb(pos.X, pos.Y) * 0.9;
        }
        if (exits <= 1) {
            score -= 1.9;
        }

        if (score > bestScore) {
            bestScore = score;
            best = pos;
        }
    }

    if (best) {
        return best;
    }

    for (var i = 0; i < 4; i++) {
        var nx = currentMap.X + DIRS[i].dx;
        var ny = currentMap.Y + DIRS[i].dy;
        var nk = MapKey(nx, ny);
        if (!IsAIWalkable(nx, ny) || threatMap[nk] || IsTileThreatSoon(nk, 240, threatSnapshot)) {
            continue;
        }
        return {X: nx, Y: ny};
    }

    return null;
};

Monster.prototype.FindSafeTile = function(currentMap, threatMap, threatSnapshot) {
    var snapshot = threatSnapshot || LastThreatSnapshot || BuildThreatSnapshot();
    var bfs = AIPathBFS(currentMap.X, currentMap.Y);
    var bestTile = null;
    var bestDist = Infinity;
    var fallbackTile = null;
    var fallbackScore = -Infinity;
    var key;
    var guard = GetTrainingGuardDirectives();
    var minExits = guard ? guard.minSafeNeighbors : 1;
    var preferFarSafe = !!(guard && guard.preferFarSafe);

    for (key in bfs.dist) {
        var d = bfs.dist[key];
        var pos = ParseKey(key);
        var travelMs = d * 240;
        var eta = snapshot.dangerEtaMap ? snapshot.dangerEtaMap[key] : null;
        var score;
        var fallbackValue;
        var exits;
        var failureHeat;

        if (threatMap[key]) {
            continue;
        }
        if (IsTileThreatSoon(key, travelMs, snapshot)) {
            continue;
        }

        exits = CountSafeNeighborTiles(pos.X, pos.Y, snapshot);
        if (exits < minExits) {
            continue;
        }
        failureHeat = GetTrainingFailureHeatPenalty(pos.X, pos.Y);

        score = d * 0.85
            + AIEvolution.getDanger(pos.X, pos.Y) * (1.3 + AIEvolution.getEvadeWeight())
            - DistanceToNearestBomb(pos.X, pos.Y) * 0.24
            + failureHeat * (1.4 + AIEvolution.getEvadeWeight())
            - exits * 1.65;
        if (preferFarSafe) {
            score -= DistanceToNearestBomb(pos.X, pos.Y) * 0.34;
            score -= exits * 0.45;
        }
        if (exits <= 1) {
            score += 2;
        }

        if (typeof eta === "number") {
            score += Math.max(0, (AIDodgePolicy.safeBufferMs + 220 - eta) / 180);
        }

        if (score < bestDist) {
            bestDist = score;
            bestTile = pos;
        }

        fallbackValue = (typeof eta === "number" ? eta : 10000)
            - d * 150
            - failureHeat * 150
            - AIEvolution.getDanger(pos.X, pos.Y) * 240;
        if (fallbackValue > fallbackScore) {
            fallbackScore = fallbackValue;
            fallbackTile = pos;
        }
    }

    if (bestTile) return bestTile;
    if (fallbackTile) return fallbackTile;

    // 无完全安全格 — 退到相邻可行走格（半身闪避）
    for (var minExitTry = minExits; minExitTry >= 1; minExitTry--) {
        for (var i = 0; i < 4; i++) {
            var nx = currentMap.X + DIRS[i].dx;
            var ny = currentMap.Y + DIRS[i].dy;
            var nk = MapKey(nx, ny);
            if (IsAIWalkable(nx, ny)
                && !IsTileThreatSoon(nk, 220, snapshot)
                && CountSafeNeighborTiles(nx, ny, snapshot) >= minExitTry) {
                return {X: nx, Y: ny};
            }
        }
    }

    return null;
};

Monster.prototype.TryHalfBodyShift = function(currentMap, player, threatMap) {
    var pref = AIEvolution.getHalfBodyPreference();
    var playerMap = player.CurrentMapID();
    var sameRow;
    var sameCol;
    var candidates;
    var best = null;
    var bestScore = Infinity;

    if (!playerMap || pref < 0.4) return false;
    if (ManhattanDist(currentMap.X, currentMap.Y, playerMap.X, playerMap.Y) > 6) return false;
    if (Math.random() > pref * 0.35) return false;

    sameRow = currentMap.Y === playerMap.Y;
    sameCol = currentMap.X === playerMap.X;
    if (!sameRow && !sameCol) return false;

    candidates = sameRow
        ? [{dx: 0, dy: -1}, {dx: 0, dy: 1}]
        : [{dx: -1, dy: 0}, {dx: 1, dy: 0}];

    for (var i = 0; i < candidates.length; i++) {
        var nx = currentMap.X + candidates[i].dx;
        var ny = currentMap.Y + candidates[i].dy;
        var nk = MapKey(nx, ny);
        var score;
        if (!IsAIWalkable(nx, ny) || threatMap[nk]) continue;
        score = AIEvolution.getDanger(nx, ny) + ManhattanDist(nx, ny, playerMap.X, playerMap.Y) * 0.08;
        if (score < bestScore) {
            bestScore = score;
            best = {X: nx, Y: ny};
        }
    }

    if (!best) return false;
    this.MoveToMap(best);
    return true;
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

        score = bfs.dist[key] + AIEvolution.getDanger(c.X, c.Y) * (1 + AIEvolution.getEvadeWeight() * 1.2) - AIEvolution.getTrapPreference() * 0.6;
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
        case 101: return role.CanPaopaoLength < MonsterMaxPaopaoLength ? 3 : 0.5;
        case 102: return role.MoveStep < RoleConstant.MaxMoveStep ? 3 : 0.5;
        case 103: return role.PaopaoStrong < RoleConstant.MaxPaopaoStrong ? 3 : 0.5;
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
            if (v < 101 || v > 103) continue;

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
    var maxVAttackPath = 10 + Math.round(AIEvolution.getVAttackPreference() * 6);
    if (totalDist > maxVAttackPath) return false;

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
    var snapshot;
    if (!plan) return;
    snapshot = LastThreatSnapshot || BuildThreatSnapshot();

    // 安全优先 — 处于危险区立即放弃计划
    if (threatMap[MapKey(currentMap.X, currentMap.Y)]) {
        this.AttackPlan = null;
        this.State = "evade";
        var safe = this.FindSafeTile(currentMap, threatMap, snapshot);
        if (safe) this.MoveToMap(safe);
        return;
    }

    // 验证计划有效性
    var player = GetSinglePlayerPlayer();
    if (!player || player.IsDeath) { this.AttackPlan = null; return; }

    var playerMap = player.CurrentMapID();
    if (!playerMap) { this.AttackPlan = null; return; }

    var targetMoved = ManhattanDist(plan.target.X, plan.target.Y, playerMap.X, playerMap.Y) > 2;
    var tooOld = Date.now() - plan.startTime > (4200 + Math.round(AIEvolution.getVAttackPreference() * 3200));
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
                var safeSnapshot = BuildThreatSnapshot();
                var safeTile = this.FindSafeTile(currentMap, safeSnapshot.threatMap, safeSnapshot);
                if (safeTile) this.MoveToMap(safeTile);
            }
        } else {
            this.MoveToMap(plan.pos2);
        }
    } else {
        // escape phase
        this.AttackPlan = null;
        var safeEsc = this.FindSafeTile(currentMap, threatMap, snapshot);
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
    var targetCount = typeof count === "number" ? count : MonsterCount;
    var playerSpawn = typeof GetCurrentGameMapSpawn === "function" ? GetCurrentGameMapSpawn() : { X: 0, Y: 0 };
    targetCount = parseInt(targetCount, 10);
    if (isNaN(targetCount)) {
        targetCount = MonsterCount;
    }
    if (targetCount < 0) {
        targetCount = 0;
    }
    if (targetCount > 4) {
        targetCount = 4;
    }
    MonsterStorage = [];

    for (var i = 0; i < targetCount; i++) {
        var mapID = {
            X: Math.floor(Math.random() * 15),
            Y: Math.floor(Math.random() * 13)
        };

        if (townBarrierMap[mapID.Y][mapID.X] === 0 && !(mapID.X === playerSpawn.X && mapID.Y === playerSpawn.Y)) {
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

// =============================================================================
// AI 躲泡强化训练
// =============================================================================

var AIDodgeTrainer = {
    IsRunning: false,
    Monster: null,
    Role: null,
    BubbleCaster: null,
    SpawnInterval: null,
    TraceInterval: null,
    NextStepTimeout: null,
    RespawnTimeout: null,
    AttemptEndTimeout: null,
    IsAttemptActive: false,
    Config: {
        targetIterations: 10,
        maxAttemptsPerMatch: Infinity,
        unboundedRounds: true,
        stopSuccessThreshold: 3000,
        minIncrementPerRound: 50,
        trainingThinkIntervalMs: MonsterTrainingThinkInterval,
        deathLimitPerMatch: 10,
        bubbleIntervalMs: 200,
        bubbleFuseMs: 1200,
        bubbleStrong: 10,
        traceSampleMs: 120,
        matchGapMs: 900,
        respawnDelayMs: 520,
        policyFloor: {
            forecastMs: 2200,
            safeBufferMs: 320,
            repathMs: 260,
            stuckTimeoutMs: 620,
            roamRadius: 11,
            halfBodyMinEtaMs: 280
        }
    },
    State: {
        matchIndex: 0,
        matchAttempt: 0,
        roundIndex: 0,
        matchScore: 0,
        matchDeaths: 0,
        matchSurviveRounds: 0,
        totalScore: 0,
        totalDeaths: 0,
        totalAttempts: 0,
        completedMatches: 0,
        baselineScore: null,
        targetScore: null,
        latestLesson: "",
        latestDeathSummary: "",
        acceptedResults: [],
        matchHistory: [],
        roundReviews: [],
        logs: [],
        deathReasonCounts: {},
        currentMatchAttempts: [],
        currentAttemptEventMap: {},
        currentAttemptEventStats: null,
        currentMatchTargetScore: null,
        currentAttemptFailureLogs: [],
        traceFrames: [],
        failureHeatMap: {},
        reasonMemory: {},
        lastDominantReason: "",
        dominantReasonStreak: 0,
        adaptiveTuningVersion: 0,
        reasonGuards: {},
        lastBombReason: "",
        sameReasonHitStreak: 0,
        stopReason: "",
        stopAtRound: 0,
        latestReview: null,
        currentAttemptPlannerStats: null
    },

    IsMonsterTraining: function(monster) {
        return this.IsRunning && this.Monster === monster;
    },

    EnsurePanel: function() {
        var panel = document.getElementById("match-panel");
        var block;
        if (!panel) {
            return;
        }
        panel.style.display = "block";
        block = document.getElementById("ai-training-block");
        if (!block) {
            block = document.createElement("div");
            block.id = "ai-training-block";
            block.className = "config-block";
            block.innerHTML = ""
                + "<div style=\"font-weight:700;margin-bottom:6px;\">AI 躲泡强化训练</div>"
                + "<div id=\"ai-training-status\" style=\"font-size:12px;color:#cdd9f6;margin-bottom:6px;\"></div>"
                + "<div id=\"ai-training-goal\" style=\"font-size:12px;color:#c7c9ff;margin-bottom:6px;\"></div>"
                + "<div id=\"ai-training-score\" style=\"font-size:12px;color:#ffe38b;margin-bottom:6px;\"></div>"
                + "<div id=\"ai-training-policy\" style=\"font-size:12px;color:#9bb2dd;margin-bottom:6px;\"></div>"
                + "<div id=\"ai-training-lesson\" style=\"font-size:12px;color:#9ff7c5;margin-bottom:6px;\"></div>"
                + "<div id=\"ai-training-death\" style=\"font-size:12px;color:#ffb4b4;margin-bottom:6px;\"></div>"
                + "<ol id=\"ai-training-log\" style=\"margin:0;padding-left:16px;max-height:180px;overflow:auto;font-size:12px;color:#d8e6ff;\"></ol>";
            panel.appendChild(block);
        }
    },

    RenderPanel: function() {
        var statusNode = document.getElementById("ai-training-status");
        var goalNode = document.getElementById("ai-training-goal");
        var scoreNode = document.getElementById("ai-training-score");
        var policyNode = document.getElementById("ai-training-policy");
        var lessonNode = document.getElementById("ai-training-lesson");
        var deathNode = document.getElementById("ai-training-death");
        var logNode = document.getElementById("ai-training-log");
        var guard = this.GetActiveGuardDirectives();
        var guardText = guard
            ? ("｜护栏: 预警+" + guard.panicEtaBoost + "ms"
                + " 最低出口" + guard.minSafeNeighbors
                + (guard.disableHalfBody ? " 禁半身" : "")
                + (guard.disallowStay ? " 禁停留" : "")
                + (guard.preferFarSafe ? " 远离密区" : ""))
            : "｜护栏: 无";
        var i;
        if (!statusNode || !goalNode || !scoreNode || !policyNode || !lessonNode || !deathNode || !logNode) {
            return;
        }

        statusNode.textContent = "第 " + this.State.matchIndex + " 轮（尝试 " + this.State.matchAttempt + "）"
            + "｜事件结算 " + this.State.roundIndex + " 个";
        goalNode.textContent = "目标：" + (this.State.currentMatchTargetScore == null
            ? "首轮建立基线"
            : ("本轮 >= " + this.State.currentMatchTargetScore + "（基线+"
                + this.Config.minIncrementPerRound + "）"))
            + (this.State.baselineScore == null ? "" : ("｜当前基线 " + this.State.baselineScore));
        scoreNode.textContent = "当前尝试：躲泡成功 " + this.State.matchScore
            + "｜被炸 " + this.State.matchDeaths + "/" + this.Config.deathLimitPerMatch
            + "｜累计总分 " + this.State.totalScore
            + "｜总尝试 " + this.State.totalAttempts;
        policyNode.textContent = "策略: 预判 " + AIDodgePolicy.forecastMs + "ms"
            + " / 缓冲 " + AIDodgePolicy.safeBufferMs + "ms"
            + " / 重规划 " + AIDodgePolicy.repathMs + "ms"
            + " / 解卡 " + AIDodgePolicy.stuckTimeoutMs + "ms"
            + " / 巡逻半径 " + AIDodgePolicy.roamRadius
            + " / 半身最小缓冲 " + AIDodgePolicy.halfBodyMinEtaMs + "ms"
            + " / 迭代v" + this.State.adaptiveTuningVersion
            + guardText;
        lessonNode.textContent = this.State.latestLesson ? ("经验: " + this.State.latestLesson) : "经验: 训练进行中";
        deathNode.textContent = this.State.latestDeathSummary ? ("失败画像: " + this.State.latestDeathSummary) : "失败画像: 暂无";

        logNode.innerHTML = "";
        for (i = Math.max(0, this.State.logs.length - 10); i < this.State.logs.length; i++) {
            var li = document.createElement("li");
            li.textContent = this.State.logs[i];
            logNode.appendChild(li);
        }
        this.PublishRuntimeState();
    },

    AddLog: function(line) {
        var stamp = new Date().toLocaleTimeString();
        var msg = "[" + stamp + "] " + line;
        this.State.logs.push(msg);
        if (this.State.logs.length > 1200) {
            this.State.logs = this.State.logs.slice(this.State.logs.length - 1200);
        }
        if (typeof console !== "undefined" && typeof console.log === "function") {
            console.log("[AI训练] " + msg);
        }
        this.RenderPanel();
    },

    CapturePolicySnapshot: function() {
        return {
            forecastMs: AIDodgePolicy.forecastMs,
            safeBufferMs: AIDodgePolicy.safeBufferMs,
            repathMs: AIDodgePolicy.repathMs,
            stuckTimeoutMs: AIDodgePolicy.stuckTimeoutMs,
            roamRadius: AIDodgePolicy.roamRadius,
            halfBodyMinEtaMs: AIDodgePolicy.halfBodyMinEtaMs
        };
    },

    DiffPolicySnapshot: function(before, after) {
        var b = before || {};
        var a = after || {};
        return {
            forecastMs: (a.forecastMs || 0) - (b.forecastMs || 0),
            safeBufferMs: (a.safeBufferMs || 0) - (b.safeBufferMs || 0),
            repathMs: (a.repathMs || 0) - (b.repathMs || 0),
            stuckTimeoutMs: (a.stuckTimeoutMs || 0) - (b.stuckTimeoutMs || 0),
            roamRadius: (a.roamRadius || 0) - (b.roamRadius || 0),
            halfBodyMinEtaMs: (a.halfBodyMinEtaMs || 0) - (b.halfBodyMinEtaMs || 0)
        };
    },

    EnforceHighPressurePolicyFloor: function(tag) {
        var floor = this.Config.policyFloor || {};
        var before = this.CapturePolicySnapshot();
        var after;
        var diff;
        var changed = false;
        var source = tag || "attempt_start";

        if (IsOfflineMLExpertFreezeEnabled()) {
            return null;
        }

        if (typeof floor.forecastMs === "number" && AIDodgePolicy.forecastMs < floor.forecastMs) {
            AIDodgePolicy.forecastMs = floor.forecastMs;
            changed = true;
        }
        if (typeof floor.safeBufferMs === "number" && AIDodgePolicy.safeBufferMs < floor.safeBufferMs) {
            AIDodgePolicy.safeBufferMs = floor.safeBufferMs;
            changed = true;
        }
        if (typeof floor.repathMs === "number" && AIDodgePolicy.repathMs > floor.repathMs) {
            AIDodgePolicy.repathMs = floor.repathMs;
            changed = true;
        }
        if (typeof floor.stuckTimeoutMs === "number" && AIDodgePolicy.stuckTimeoutMs > floor.stuckTimeoutMs) {
            AIDodgePolicy.stuckTimeoutMs = floor.stuckTimeoutMs;
            changed = true;
        }
        if (typeof floor.roamRadius === "number" && AIDodgePolicy.roamRadius < floor.roamRadius) {
            AIDodgePolicy.roamRadius = floor.roamRadius;
            changed = true;
        }
        if (typeof floor.halfBodyMinEtaMs === "number" && AIDodgePolicy.halfBodyMinEtaMs < floor.halfBodyMinEtaMs) {
            AIDodgePolicy.halfBodyMinEtaMs = floor.halfBodyMinEtaMs;
            changed = true;
        }
        if (!changed) {
            return null;
        }

        NormalizeDodgePolicyValue(AIDodgePolicy);
        SaveAIDodgePolicy();
        after = this.CapturePolicySnapshot();
        diff = this.DiffPolicySnapshot(before, after);
        this.State.adaptiveTuningVersion += 1;
        this.AddLog("高压训练参数保底(" + source + ")：预判"
            + (diff.forecastMs >= 0 ? "+" : "") + diff.forecastMs + "ms"
            + "，缓冲" + (diff.safeBufferMs >= 0 ? "+" : "") + diff.safeBufferMs + "ms"
            + "，重规划" + (diff.repathMs >= 0 ? "+" : "") + diff.repathMs + "ms"
            + "，解卡" + (diff.stuckTimeoutMs >= 0 ? "+" : "") + diff.stuckTimeoutMs + "ms"
            + "，巡逻半径" + (diff.roamRadius >= 0 ? "+" : "") + diff.roamRadius
            + "，半身阈值" + (diff.halfBodyMinEtaMs >= 0 ? "+" : "") + diff.halfBodyMinEtaMs + "ms。");
        return diff;
    },

    PublishRuntimeState: function() {
        if (typeof window === "undefined") {
            return;
        }
        window.BNBTrainingRuntimeState = {
            ts: Date.now(),
            running: !!this.IsRunning,
            matchIndex: this.State.matchIndex,
            matchAttempt: this.State.matchAttempt,
            roundIndex: this.State.roundIndex,
            matchScore: this.State.matchScore,
            matchDeaths: this.State.matchDeaths,
            totalAttempts: this.State.totalAttempts,
            totalScore: this.State.totalScore,
            completedMatches: this.State.completedMatches,
            baselineScore: this.State.baselineScore,
            targetScore: this.State.currentMatchTargetScore,
            latestLesson: this.State.latestLesson,
            latestDeathSummary: this.State.latestDeathSummary,
            latestReview: this.State.latestReview,
            stopReason: this.State.stopReason || "",
            stopAtRound: this.State.stopAtRound || 0,
            logsTail: this.State.logs.slice(Math.max(0, this.State.logs.length - 8))
        };
    },

    ResetTimers: function() {
        if (this.SpawnInterval) {
            clearInterval(this.SpawnInterval);
            this.SpawnInterval = null;
        }
        if (this.TraceInterval) {
            clearInterval(this.TraceInterval);
            this.TraceInterval = null;
        }
        if (this.NextStepTimeout) {
            clearTimeout(this.NextStepTimeout);
            this.NextStepTimeout = null;
        }
        if (this.RespawnTimeout) {
            clearTimeout(this.RespawnTimeout);
            this.RespawnTimeout = null;
        }
        if (this.AttemptEndTimeout) {
            clearTimeout(this.AttemptEndTimeout);
            this.AttemptEndTimeout = null;
        }
    },

    BuildBubbleCaster: function() {
        this.BubbleCaster = {
            RoleNumber: 2,
            PaopaoStrong: this.Config.bubbleStrong,
            PaopaoCount: 0,
            Object: { ZIndex: 3 },
            CastMapID: { X: 1, Y: 1 },
            CurrentMapID: function() {
                return { X: this.CastMapID.X, Y: this.CastMapID.Y };
            }
        };
    },

    StripLiveNonRigidBarriers: function() {
        var y;
        var x;
        var cell;
        if (typeof StripNonRigidBarriersFromMap === "function") {
            StripNonRigidBarriersFromMap();
        }
        if (typeof Barrier === "undefined" || !Barrier.Storage) {
            return;
        }
        for (y = 0; y < Barrier.Storage.length; y++) {
            if (!Barrier.Storage[y]) continue;
            for (x = 0; x < Barrier.Storage[y].length; x++) {
                cell = Barrier.Storage[y][x];
                if (!cell) continue;
                if (!IsRigidBarrierNo(cell.No)) {
                    if (cell.Object && typeof cell.Object.Dispose === "function") {
                        cell.Object.Dispose();
                    }
                    Barrier.Storage[y][x] = null;
                }
            }
        }
    },

    FindSafeSpawn: function() {
        var snapshot = BuildThreatSnapshot();
        var attempts = 0;
        while (attempts < 320) {
            var x = Math.floor(Math.random() * 15);
            var y = Math.floor(Math.random() * 13);
            var key = MapKey(x, y);
            if (!IsAIWalkable(x, y)) {
                attempts++;
                continue;
            }
            if (IsTileThreatSoon(key, 200, snapshot)) {
                attempts++;
                continue;
            }
            return { X: x, Y: y };
        }
        if (typeof GetCurrentGameMapSpawn === "function") {
            return GetCurrentGameMapSpawn();
        }
        return { X: 1, Y: 1 };
    },

    FindRandomBubbleTile: function() {
        var attempts = 0;
        while (attempts < 260) {
            var x = Math.floor(Math.random() * 15);
            var y = Math.floor(Math.random() * 13);
            if (!IsAIWalkable(x, y)) {
                attempts++;
                continue;
            }
            if (townBarrierMap[y][x] !== 0) {
                attempts++;
                continue;
            }
            return { X: x, Y: y };
        }
        return null;
    },

    SpawnRandomBubble: function() {
        var tile = this.FindRandomBubbleTile();
        if (!tile || !this.BubbleCaster) {
            return false;
        }
        this.BubbleCaster.CastMapID = tile;
        new Paopao(this.BubbleCaster);
        BNBMLDatasetCollector.OnBubbleSpawned(this.Role);
        BNBMLRuntime.OnBubbleSpawned(this.Role);
        return true;
    },

    BeginAttemptEventTracking: function() {
        this.State.currentAttemptEventMap = {};
        this.State.currentAttemptEventStats = {
            resolved: 0,
            success: 0,
            failed: 0,
            pending: 0,
            reasonCounts: {}
        };
        this.State.currentAttemptFailureLogs = [];
        this.State.traceFrames = [];
        this.State.roundIndex = 0;
        this.State.matchScore = 0;
        this.State.matchSurviveRounds = 0;
        this.State.lastBombReason = "";
        this.State.sameReasonHitStreak = 0;
        this.State.currentAttemptPlannerStats = {
            replanCount: 0,
            fallbackReplanCount: 0,
            newBubbleReplanCount: 0,
            panicEtaReplanCount: 0,
            lowExitReplanCount: 0,
            waitStepsPlanned: 0,
            moveStepsPlanned: 0,
            halfBodyStepsPlanned: 0,
            zeroStepPlanCount: 0,
            forceMovePlanCount: 0,
            preferMovePlanCount: 0,
            blockedFallbackTriggers: 0,
            triggerCounts: {},
            latestSceneRevision: GetThreatSceneRevision(),
            latestPathPreview: "",
            latestPlanMs: 0,
            expandedStates: 0
        };
        this.DecayFailureHeatMap();
    },

    DecayFailureHeatMap: function() {
        var heat = this.State.failureHeatMap || {};
        var key;
        for (key in heat) {
            if (!heat.hasOwnProperty(key)) {
                continue;
            }
            heat[key] = heat[key] * 0.86;
            if (heat[key] < 0.2) {
                delete heat[key];
            }
        }
        this.State.failureHeatMap = heat;
    },

    CaptureTrainingTraceFrame: function() {
        var role = this.Role;
        var currentMap;
        var snapshot;
        var key;
        var nextChoice;
        var trace;

        if (!this.IsRunning || !this.IsAttemptActive || !role || role.IsDeath || role.IsInPaopao) {
            return;
        }
        currentMap = role.CurrentMapID();
        if (!currentMap) {
            return;
        }

        snapshot = LastThreatSnapshot || BuildThreatSnapshot();
        key = MapKey(currentMap.X, currentMap.Y);
        nextChoice = PickNextFrameMovementChoice(role, currentMap, snapshot);
        trace = {
            t: Date.now(),
            x: currentMap.X,
            y: currentMap.Y,
            eta: snapshot && snapshot.dangerEtaMap ? snapshot.dangerEtaMap[key] : null,
            endEta: snapshot && snapshot.dangerEndMap ? snapshot.dangerEndMap[key] : null,
            safeNeighbors: CountSafeNeighborTiles(currentMap.X, currentMap.Y, snapshot),
            unsafeNeighborCount: CountUpcomingUnsafeNeighbors(currentMap.X, currentMap.Y, snapshot),
            escapeSteps: EstimateEscapeSteps(currentMap.X, currentMap.Y, snapshot, 2, 12),
            activeBombs: CountActiveBombs(),
            nearestBombDist: DistanceToNearestBomb(currentMap.X, currentMap.Y),
            nextSafeRank: nextChoice ? nextChoice.safeRank : 9,
            nextHalfOk: nextChoice ? !!nextChoice.halfBodyBufferOk : false,
            state: this.Monster ? this.Monster.State : "",
            stuckCount: this.Monster ? this.Monster.StuckCount : 0
        };

        this.State.traceFrames.push(trace);
        if (this.State.traceFrames.length > 160) {
            this.State.traceFrames = this.State.traceFrames.slice(this.State.traceFrames.length - 160);
        }
    },

    BuildPlannerStatsSnapshot: function() {
        var src = this.State.currentAttemptPlannerStats || {};
        var counts = src.triggerCounts || {};
        var total = 0;
        var key;
        var triggerShare = {};
        for (key in counts) {
            if (!counts.hasOwnProperty(key)) {
                continue;
            }
            total += counts[key];
        }
        for (key in counts) {
            if (!counts.hasOwnProperty(key)) {
                continue;
            }
            triggerShare[key] = total > 0 ? counts[key] / total : 0;
        }
        return {
            replanCount: src.replanCount || 0,
            fallbackReplanCount: src.fallbackReplanCount || 0,
            newBubbleReplanCount: src.newBubbleReplanCount || 0,
            panicEtaReplanCount: src.panicEtaReplanCount || 0,
            lowExitReplanCount: src.lowExitReplanCount || 0,
            waitStepsPlanned: src.waitStepsPlanned || 0,
            moveStepsPlanned: src.moveStepsPlanned || 0,
            halfBodyStepsPlanned: src.halfBodyStepsPlanned || 0,
            zeroStepPlanCount: src.zeroStepPlanCount || 0,
            forceMovePlanCount: src.forceMovePlanCount || 0,
            preferMovePlanCount: src.preferMovePlanCount || 0,
            blockedFallbackTriggers: src.blockedFallbackTriggers || 0,
            latestSceneRevision: src.latestSceneRevision || 0,
            latestPathPreview: src.latestPathPreview || "",
            latestPlanMs: src.latestPlanMs || 0,
            expandedStates: src.expandedStates || 0,
            triggerCounts: JSON.parse(JSON.stringify(counts)),
            triggerShare: triggerShare
        };
    },

    OnTemporalPlanBuilt: function(payload) {
        var info = payload || {};
        var plan = info.plan || null;
        var reason = info.reason || "unknown";
        var bubble = info.bubbleEvent || null;
        var stats = this.State.currentAttemptPlannerStats;
        var start;
        var target;
        var pathText;
        var triggerText;

        if (!stats) {
            this.State.currentAttemptPlannerStats = {
                replanCount: 0,
                fallbackReplanCount: 0,
                newBubbleReplanCount: 0,
                panicEtaReplanCount: 0,
                lowExitReplanCount: 0,
                waitStepsPlanned: 0,
                moveStepsPlanned: 0,
                halfBodyStepsPlanned: 0,
                zeroStepPlanCount: 0,
                forceMovePlanCount: 0,
                preferMovePlanCount: 0,
                blockedFallbackTriggers: 0,
                triggerCounts: {},
                latestSceneRevision: 0,
                latestPathPreview: "",
                latestPlanMs: 0,
                expandedStates: 0
            };
            stats = this.State.currentAttemptPlannerStats;
        }

        stats.replanCount += 1;
        stats.triggerCounts[reason] = (stats.triggerCounts[reason] || 0) + 1;
        if (reason === "new_bubble_event") {
            stats.newBubbleReplanCount += 1;
        }
        if (reason === "path_blocked_fallback") {
            stats.fallbackReplanCount += 1;
        }
        if (reason === "panic_eta_replan") {
            stats.panicEtaReplanCount += 1;
        }
        if (reason === "low_exit_replan") {
            stats.lowExitReplanCount += 1;
        }
        stats.latestSceneRevision = typeof info.sceneRevision === "number" ? info.sceneRevision : stats.latestSceneRevision;
        stats.blockedFallbackTriggers = this.Monster ? (this.Monster.TemporalFallbackTriggerCount || 0) : stats.blockedFallbackTriggers;

        if (plan) {
            stats.waitStepsPlanned += plan.waitSteps || 0;
            stats.moveStepsPlanned += plan.moveSteps || 0;
            stats.halfBodyStepsPlanned += plan.halfBodySteps || 0;
            if ((plan.totalSteps || 0) === 0) {
                stats.zeroStepPlanCount += 1;
            }
            if (plan.forceMove) {
                stats.forceMovePlanCount += 1;
            }
            if (plan.preferMove) {
                stats.preferMovePlanCount += 1;
            }
            stats.latestPlanMs = plan.estimatedMs || 0;
            stats.expandedStates = plan.expandedStates || 0;
            stats.latestPathPreview = info.preview || "";
        }

        start = plan && plan.start ? "(" + plan.start.x + "," + plan.start.y + ")" : "(na,na)";
        target = plan && plan.target ? "(" + plan.target.x + "," + plan.target.y + ")" : "(na,na)";
        pathText = info.preview || "无可行路径";
        triggerText = reason;
        if (bubble && reason === "new_bubble_event") {
            triggerText += "@" + (bubble.eventId || "unknown")
                + " (" + bubble.x + "," + bubble.y + ")"
                + " [" + (bubble.startAt || "-") + "->" + (bubble.endAt || "-") + "]"
                + " cluster=" + (bubble.clusterId || "");
        }

        this.AddLog("时序重规划：触发=" + triggerText
            + "｜rev=" + (typeof info.sceneRevision === "number" ? info.sceneRevision : "na")
            + "｜起点" + start
            + "｜目标" + target
            + "｜总步数=" + (plan ? plan.totalSteps : 0)
            + "｜耗时≈" + (plan ? plan.estimatedMs : 0) + "ms"
            + "｜半身步=" + (plan ? plan.halfBodySteps : 0)
            + "｜等待步=" + (plan ? plan.waitSteps : 0)
            + "｜模式="
            + (plan
                ? ((plan.panicMode ? "panic_" : "")
                    + (plan.forceMove ? "force_move" : (plan.preferMove ? "prefer_move" : "normal")))
                : "no_plan")
            + "｜起点安全={rank:"
            + (plan && plan.planContext ? plan.planContext.startSafeRank : "na")
            + ",dur:"
            + (plan && plan.planContext ? plan.planContext.startSafeDurationMs : "na")
            + ",eta:"
            + (plan && plan.planContext && typeof plan.planContext.currentEta === "number"
                ? plan.planContext.currentEta
                : "na")
            + ",minExits:"
            + (plan && plan.planContext ? plan.planContext.goalMinExits : "na")
            + ",minMove:"
            + (plan && plan.planContext ? plan.planContext.minGoalMoveSteps : "na")
            + ",minDisp:"
            + (plan && plan.planContext ? plan.planContext.minGoalDisplacement : "na")
            + "}"
            + "｜路径=" + pathText);
    },

    BuildBombFailureContext: function(victim, eventId) {
        var now = Date.now();
        var currentMap = victim && typeof victim.CurrentMapID === "function" ? victim.CurrentMapID() : null;
        var snapshot = LastThreatSnapshot || BuildThreatSnapshot();
        var key = currentMap ? MapKey(currentMap.X, currentMap.Y) : "";
        var eta = key && snapshot && snapshot.dangerEtaMap ? snapshot.dangerEtaMap[key] : null;
        var endEta = key && snapshot && snapshot.dangerEndMap ? snapshot.dangerEndMap[key] : null;
        var nextChoice = currentMap ? PickNextFrameMovementChoice(victim, currentMap, snapshot) : null;
        var eventOverlap = victim && victim.LastUnsafeExplosionEventIds ? victim.LastUnsafeExplosionEventIds.length : 0;
        var recentTrace = this.State.traceFrames.slice(Math.max(0, this.State.traceFrames.length - 8));
        var stayFrames = 0;
        var i;

        if (currentMap) {
            for (i = recentTrace.length - 1; i >= 0; i--) {
                if (recentTrace[i].x === currentMap.X && recentTrace[i].y === currentMap.Y) {
                    stayFrames++;
                }
                else {
                    break;
                }
            }
        }

        return {
            ts: now,
            eventId: eventId || "",
            x: currentMap ? currentMap.X : -1,
            y: currentMap ? currentMap.Y : -1,
            eta: typeof eta === "number" ? eta : null,
            endEta: typeof endEta === "number" ? endEta : null,
            activeBombs: currentMap ? CountActiveBombs() : 0,
            nearestBombDist: currentMap ? DistanceToNearestBomb(currentMap.X, currentMap.Y) : 99,
            safeNeighbors: currentMap ? CountSafeNeighborTiles(currentMap.X, currentMap.Y, snapshot) : 0,
            stuckCount: this.Monster ? this.Monster.StuckCount : 0,
            state: this.Monster ? this.Monster.State : "",
            eventOverlap: eventOverlap,
            nextSafeRank: nextChoice ? nextChoice.safeRank : 9,
            nextHalfOk: nextChoice ? !!nextChoice.halfBodyBufferOk : false,
            nextEta: nextChoice && typeof nextChoice.unsafeEtaPenalty === "number" ? nextChoice.unsafeEtaPenalty : null,
            nextSafeNeighbors: nextChoice && typeof nextChoice.safeNeighbors === "number" ? nextChoice.safeNeighbors : 0,
            escapeSteps: currentMap ? EstimateEscapeSteps(currentMap.X, currentMap.Y, snapshot, 2, 14) : null,
            unsafeNeighborCount: currentMap ? CountUpcomingUnsafeNeighbors(currentMap.X, currentMap.Y, snapshot) : 0,
            stayFrames: stayFrames,
            recentTrace: recentTrace
        };
    },

    GetExplosionSnapshot: function() {
        if (typeof GetBNBExplosionEventSnapshot === "function") {
            return GetBNBExplosionEventSnapshot(Date.now());
        }
        return { now: Date.now(), bombs: [], clusters: [], activeWindows: [] };
    },

    EnsureEventRecord: function(bombEvent) {
        var map = this.State.currentAttemptEventMap;
        var rec = map[bombEvent.eventId];
        if (!rec) {
            rec = {
                eventId: bombEvent.eventId,
                clusterId: bombEvent.clusterId,
                startAt: bombEvent.startAt,
                endAt: bombEvent.endAt,
                coverageCount: bombEvent.coverageMapIds.length,
                status: "pending",
                reason: "",
                resolvedAt: 0
            };
            map[bombEvent.eventId] = rec;
        }
        else {
            rec.clusterId = bombEvent.clusterId;
            rec.startAt = bombEvent.startAt;
            rec.endAt = bombEvent.endAt;
            rec.coverageCount = bombEvent.coverageMapIds.length;
        }
        return rec;
    },

    MarkEventFailed: function(eventId, reason) {
        var rec = this.State.currentAttemptEventMap[eventId];
        if (!rec || rec.status === "failed" || rec.status === "success") {
            return;
        }
        rec.status = "failed";
        rec.reason = reason || "未知";
    },

    ResolveEventResultIfNeeded: function(rec, now) {
        var stats = this.State.currentAttemptEventStats;
        if (!rec || rec.status === "resolved") {
            return;
        }
        if (rec.status === "pending" && now < rec.endAt) {
            return;
        }
        if (rec.status === "pending") {
            rec.status = "success";
            rec.reason = "窗口结束前未触发被炸";
        }
        stats.resolved++;
        this.State.roundIndex = stats.resolved;
        if (rec.status === "success") {
            stats.success++;
            this.State.matchScore = stats.success;
            this.State.matchSurviveRounds = stats.success;
            this.State.totalScore++;
            this.AddLog(
                "事件 " + rec.eventId + "（簇 " + rec.clusterId
                + "｜" + rec.startAt + "->" + rec.endAt
                + "｜覆盖 " + rec.coverageCount + " 格）判定：安全。原因：" + rec.reason
            );
        }
        else if (rec.status === "failed") {
            stats.failed++;
            stats.reasonCounts[rec.reason] = (stats.reasonCounts[rec.reason] || 0) + 1;
            this.AddLog(
                "事件 " + rec.eventId + "（簇 " + rec.clusterId
                + "｜" + rec.startAt + "->" + rec.endAt
                + "｜覆盖 " + rec.coverageCount + " 格）判定：被炸。原因：" + rec.reason
            );
        }
        rec.status = "resolved";
        rec.resolvedAt = now;
    },

    SyncExplosionEventProgress: function() {
        var snapshot = this.GetExplosionSnapshot();
        var now = snapshot.now || Date.now();
        var i;
        var map = this.State.currentAttemptEventMap;
        var key;
        var pendingCount = 0;

        for (i = 0; i < snapshot.bombs.length; i++) {
            this.EnsureEventRecord(snapshot.bombs[i]);
        }
        for (key in map) {
            if (!map.hasOwnProperty(key)) {
                continue;
            }
            this.ResolveEventResultIfNeeded(map[key], now);
            if (map[key].status === "pending" || map[key].status === "failed" || map[key].status === "success") {
                if (map[key].status !== "resolved") {
                    pendingCount++;
                }
            }
        }
        this.State.currentAttemptEventStats.pending = pendingCount;
    },

    InferBombFailureReason: function(context) {
        var ctx = context || {};
        var lateRetreatThreshold = Math.max(80, AIDodgePolicy.safeBufferMs - 20);

        if (ctx.state === "half_body_training" && (!ctx.nextHalfOk || (typeof ctx.nextEta === "number" && ctx.nextEta < AIDodgePolicy.halfBodyMinEtaMs))) {
            return "半身失败";
        }
        if (ctx.stuckCount >= 2
            || (ctx.safeNeighbors <= 1 && (ctx.nextSafeRank >= 2 || ctx.escapeSteps == null || ctx.escapeSteps >= 5))
            || (ctx.safeNeighbors <= 1 && ctx.unsafeNeighborCount >= 3)) {
            return "路径卡死";
        }
        if (ctx.eventOverlap >= 2
            || ctx.activeBombs >= 5
            || (ctx.activeBombs >= 4 && (ctx.escapeSteps == null || ctx.escapeSteps >= 4))) {
            return "高密度连泡覆盖";
        }
        if (typeof ctx.eta === "number"
            && (ctx.eta <= lateRetreatThreshold
                || (ctx.stayFrames >= 3 && ctx.eta <= lateRetreatThreshold + 90))) {
            return "晚撤离";
        }
        return "随机爆线覆盖";
    },

    SetReasonGuard: function(reason, repeatLevel, phase, context) {
        var lvl = repeatLevel > 0 ? repeatLevel : 1;
        var now = Date.now();
        var baseMs = phase === "in_attempt" ? 9000 : (phase === "attempt_end" ? 22000 : 28000);
        var durationMs = Math.round(baseMs + (lvl - 1) * baseMs * 0.7);
        var guards = this.State.reasonGuards || {};
        var guard = {
            reason: reason,
            level: lvl,
            until: now + durationMs,
            phase: phase || "attempt",
            createdAt: now,
            directives: {
                panicEtaBoost: 0,
                disableHalfBody: false,
                minSafeNeighbors: 1,
                disallowStay: false,
                preferFarSafe: false
            }
        };

        switch (reason) {
            case "晚撤离":
                guard.directives.panicEtaBoost = 120 + 70 * lvl;
                guard.directives.disallowStay = true;
                break;
            case "半身失败":
                guard.directives.disableHalfBody = true;
                guard.directives.panicEtaBoost = 80 + 40 * lvl;
                break;
            case "路径卡死":
                guard.directives.minSafeNeighbors = Math.min(3, 2 + Math.floor((lvl - 1) / 2));
                guard.directives.disallowStay = true;
                break;
            case "高密度连泡覆盖":
                guard.directives.panicEtaBoost = 200 + 80 * lvl;
                guard.directives.minSafeNeighbors = 2;
                guard.directives.preferFarSafe = true;
                guard.directives.disallowStay = true;
                break;
            default:
                guard.directives.panicEtaBoost = 60 + 30 * lvl;
                break;
        }
        if (context && typeof context.safeNeighbors === "number" && context.safeNeighbors <= 1) {
            guard.directives.minSafeNeighbors = Math.max(guard.directives.minSafeNeighbors, 2);
        }
        guards[reason] = guard;
        this.State.reasonGuards = guards;
        return guard;
    },

    GetActiveGuardDirectives: function() {
        var guards = this.State.reasonGuards || {};
        var now = Date.now();
        var reason;
        var guard;
        var merged = {
            panicEtaBoost: 0,
            disableHalfBody: false,
            minSafeNeighbors: 1,
            disallowStay: false,
            preferFarSafe: false
        };
        var hasActive = false;

        for (reason in guards) {
            if (!guards.hasOwnProperty(reason)) {
                continue;
            }
            guard = guards[reason];
            if (!guard || guard.until <= now) {
                delete guards[reason];
                continue;
            }
            hasActive = true;
            merged.panicEtaBoost = Math.max(merged.panicEtaBoost, guard.directives.panicEtaBoost || 0);
            merged.minSafeNeighbors = Math.max(merged.minSafeNeighbors, guard.directives.minSafeNeighbors || 1);
            merged.disableHalfBody = merged.disableHalfBody || !!guard.directives.disableHalfBody;
            merged.disallowStay = merged.disallowStay || !!guard.directives.disallowStay;
            merged.preferFarSafe = merged.preferFarSafe || !!guard.directives.preferFarSafe;
        }
        this.State.reasonGuards = guards;
        return hasActive ? merged : null;
    },

    ApplyAntiRepeatAdjustment: function(reason, repeatLevel, context, phase) {
        var lvl = repeatLevel > 0 ? repeatLevel : 1;
        var severity = 1 + (lvl - 1) * 0.6;
        var actions = [];
        var ctx = context || {};

        if (IsOfflineMLExpertFreezeEnabled()) {
            return {
                phase: phase || "attempt",
                reason: reason,
                repeatLevel: lvl,
                actions: ["专家策略冻结（跳过在线调参）"]
            };
        }

        switch (reason) {
            case "晚撤离":
                AIDodgePolicy.forecastMs += Math.round(70 * severity);
                AIDodgePolicy.safeBufferMs += Math.round(22 * severity);
                AIDodgePolicy.repathMs -= Math.round(24 * severity);
                AIDodgePolicy.halfBodyMinEtaMs += Math.round(12 * severity);
                actions.push("提前撤离窗口");
                break;
            case "半身失败":
                AIDodgePolicy.halfBodyMinEtaMs += Math.round(42 * severity);
                AIDodgePolicy.safeBufferMs += Math.round(18 * severity);
                AIDodgePolicy.repathMs -= Math.round(20 * severity);
                AIEvolution.tacticWeights.halfBody = Math.max(0.25, AIEvolution.tacticWeights.halfBody - 0.03 * severity);
                actions.push("收紧半身触发阈值");
                break;
            case "路径卡死":
                AIDodgePolicy.stuckTimeoutMs -= Math.round(120 * severity);
                AIDodgePolicy.repathMs -= Math.round(36 * severity);
                AIDodgePolicy.roamRadius += Math.round(1 * severity);
                actions.push("加快解卡与重规划");
                break;
            case "高密度连泡覆盖":
                AIDodgePolicy.safeBufferMs += Math.round(36 * severity);
                AIDodgePolicy.forecastMs += Math.round(130 * severity);
                AIDodgePolicy.halfBodyMinEtaMs += Math.round(24 * severity);
                AIDodgePolicy.roamRadius += Math.round(1 * severity);
                actions.push("提升连泡高压保守度");
                break;
            default:
                AIDodgePolicy.forecastMs += Math.round(30 * severity);
                AIDodgePolicy.safeBufferMs += Math.round(12 * severity);
                actions.push("小幅增强兜底预判");
                break;
        }

        if (ctx.safeNeighbors <= 1) {
            AIDodgePolicy.roamRadius += 1;
            AIDodgePolicy.repathMs -= 10;
            actions.push("远离低出口区域");
        }

        NormalizeDodgePolicyValue(AIDodgePolicy);
        SaveAIDodgePolicy();
        AIEvolution.normalizeTactics();
        AIEvolution.refreshBombCooldown();
        AIEvolution.save();
        this.SetReasonGuard(reason, lvl, phase, ctx);
        actions.push("启用防重复护栏");
        this.State.adaptiveTuningVersion += 1;

        return {
            phase: phase || "attempt",
            reason: reason,
            repeatLevel: lvl,
            actions: actions
        };
    },

    OnRoleBombed: function(victim) {
        var eventId;
        var reason;
        var now;
        var key;
        var rec;
        var context;
        var mapKey;
        var repeatInAttempt = 0;
        var i;
        var adjust;
        var reasonTail;
        if (!this.IsRunning || !this.IsAttemptActive || victim !== this.Role) {
            return;
        }
        BNBMLDatasetCollector.OnBombed();
        BNBMLRuntime.OnBombed();
        eventId = victim.LastUnsafeExplosionEventId;
        if (!eventId && victim.LastUnsafeExplosionEventIds && victim.LastUnsafeExplosionEventIds.length > 0) {
            eventId = victim.LastUnsafeExplosionEventIds[0];
        }
        if (!eventId) {
            now = Date.now();
            for (key in this.State.currentAttemptEventMap) {
                if (!this.State.currentAttemptEventMap.hasOwnProperty(key)) {
                    continue;
                }
                rec = this.State.currentAttemptEventMap[key];
                if (rec.status === "pending" && rec.startAt <= now && rec.endAt >= now) {
                    eventId = rec.eventId;
                    break;
                }
            }
        }
        context = this.BuildBombFailureContext(victim, eventId);
        reason = this.InferBombFailureReason(context);
        context.reason = reason;
        if (this.State.lastBombReason === reason) {
            this.State.sameReasonHitStreak += 1;
        }
        else {
            this.State.lastBombReason = reason;
            this.State.sameReasonHitStreak = 1;
        }
        context.sameReasonHitStreak = this.State.sameReasonHitStreak;
        this.State.currentAttemptFailureLogs.push(context);
        if (this.State.currentAttemptFailureLogs.length > 240) {
            this.State.currentAttemptFailureLogs = this.State.currentAttemptFailureLogs.slice(this.State.currentAttemptFailureLogs.length - 240);
        }
        if (eventId) {
            this.MarkEventFailed(eventId, reason);
        }
        if (context.x >= 0 && context.y >= 0) {
            mapKey = MapKey(context.x, context.y);
            this.State.failureHeatMap[mapKey] = (this.State.failureHeatMap[mapKey] || 0) + 1 + (context.activeBombs >= 4 ? 0.8 : 0);
        }
        for (i = this.State.currentAttemptFailureLogs.length - 1; i >= 0; i--) {
            if (this.State.currentAttemptFailureLogs[i].reason !== reason) {
                continue;
            }
            repeatInAttempt++;
            if (repeatInAttempt >= 3) {
                break;
            }
        }
        if (repeatInAttempt >= 2) {
            adjust = this.ApplyAntiRepeatAdjustment(reason, repeatInAttempt, context, "in_attempt");
            this.AddLog("反复原因抑制(" + reason + " x" + repeatInAttempt + ")："
                + adjust.actions.join("、"));
        }
        if (this.State.sameReasonHitStreak >= 2) {
            adjust = this.ApplyAntiRepeatAdjustment(reason, this.State.sameReasonHitStreak + 1, context, "in_attempt");
            this.AddLog("连续同因加码(" + reason + " 连续" + this.State.sameReasonHitStreak + "次)："
                + adjust.actions.join("、"));
        }

        this.State.latestDeathSummary = reason;
        this.State.matchDeaths++;
        this.State.totalDeaths++;
        this.State.deathReasonCounts[reason] = (this.State.deathReasonCounts[reason] || 0) + 1;
        reasonTail = "eta=" + (typeof context.eta === "number" ? context.eta : "na")
            + "ms, 邻接安全格=" + context.safeNeighbors
            + ", 活跃泡=" + context.activeBombs
            + ", 下一步等级=" + context.nextSafeRank
            + ", 逃生步数=" + (typeof context.escapeSteps === "number" ? context.escapeSteps : "na")
            + ", 邻接危险=" + context.unsafeNeighborCount
            + ", 原地停滞帧=" + context.stayFrames;
        this.AddLog("第" + this.State.matchIndex + "轮-尝试" + this.State.matchAttempt + "：AI被炸第 "
            + this.State.matchDeaths + "/" + this.Config.deathLimitPerMatch + " 次。原因：" + reason
            + "（" + reasonTail + "）");
        if (this.State.matchDeaths >= this.Config.deathLimitPerMatch) {
            this.RequestFinishAttempt();
        }
    },

    OnRoleDeath: function(victim) {
        var self = this;
        if (!this.IsRunning || !this.IsAttemptActive || victim !== this.Role) {
            return;
        }
        if (this.State.matchDeaths >= this.Config.deathLimitPerMatch) {
            return;
        }

        if (this.RespawnTimeout) {
            clearTimeout(this.RespawnTimeout);
        }
        this.RespawnTimeout = setTimeout(function() {
            var spawn = self.FindSafeSpawn();
            if (!self.IsRunning || !self.Role) {
                return;
            }
            self.Role.RespawnAt(spawn.X, spawn.Y, 700);
            self.Monster.Start();
            self.StripLiveNonRigidBarriers();
        }, this.Config.respawnDelayMs);
    },

    RequestFinishAttempt: function() {
        var self = this;
        if (this.AttemptEndTimeout) {
            return;
        }
        this.AttemptEndTimeout = setTimeout(function() {
            self.FinishAttempt();
        }, 120);
    },

    AnalyzeFailureContexts: function(contexts) {
        var list = contexts || [];
        var i;
        var c;
        var etaSum = 0;
        var etaCount = 0;
        var summary = {
            total: list.length,
            reasonCounts: {},
            dominantReason: "",
            dominantCount: 0,
            avgEta: null,
            avgSafeNeighbors: 0,
            avgActiveBombs: 0,
            avgUnsafeNeighbors: 0,
            avgEscapeSteps: null,
            stuckRate: 0,
            lowExitRate: 0,
            riskyHalfBodyRate: 0
        };
        var safeNeighborSum = 0;
        var activeBombSum = 0;
        var unsafeNeighborSum = 0;
        var stuckHits = 0;
        var lowExitHits = 0;
        var halfRiskHits = 0;
        var escapeStepSum = 0;
        var escapeStepCount = 0;
        var reason;

        for (i = 0; i < list.length; i++) {
            c = list[i] || {};
            reason = c.reason || "未知";
            summary.reasonCounts[reason] = (summary.reasonCounts[reason] || 0) + 1;
            if (summary.reasonCounts[reason] > summary.dominantCount) {
                summary.dominantCount = summary.reasonCounts[reason];
                summary.dominantReason = reason;
            }
            if (typeof c.eta === "number") {
                etaSum += c.eta;
                etaCount++;
            }
            safeNeighborSum += typeof c.safeNeighbors === "number" ? c.safeNeighbors : 0;
            activeBombSum += typeof c.activeBombs === "number" ? c.activeBombs : 0;
            unsafeNeighborSum += typeof c.unsafeNeighborCount === "number" ? c.unsafeNeighborCount : 0;
            if ((c.stuckCount || 0) >= 2) {
                stuckHits++;
            }
            if ((c.safeNeighbors || 0) <= 1) {
                lowExitHits++;
            }
            if (c.nextSafeRank === 1 && !c.nextHalfOk) {
                halfRiskHits++;
            }
            if (typeof c.escapeSteps === "number") {
                escapeStepSum += c.escapeSteps;
                escapeStepCount++;
            }
        }

        if (list.length > 0) {
            summary.avgSafeNeighbors = safeNeighborSum / list.length;
            summary.avgActiveBombs = activeBombSum / list.length;
            summary.avgUnsafeNeighbors = unsafeNeighborSum / list.length;
            summary.stuckRate = stuckHits / list.length;
            summary.lowExitRate = lowExitHits / list.length;
            summary.riskyHalfBodyRate = halfRiskHits / list.length;
        }
        if (etaCount > 0) {
            summary.avgEta = etaSum / etaCount;
        }
        if (escapeStepCount > 0) {
            summary.avgEscapeSteps = escapeStepSum / escapeStepCount;
        }

        return summary;
    },

    UpdateReasonMemoryByAttempt: function(reasonCounts, dominantReason) {
        var memory = this.State.reasonMemory || {};
        var reason;
        var entry;
        var knownReasons = ["晚撤离", "半身失败", "路径卡死", "高密度连泡覆盖", "随机爆线覆盖"];
        var seen = {};

        for (var i = 0; i < knownReasons.length; i++) {
            reason = knownReasons[i];
            seen[reason] = true;
            entry = memory[reason] || {
                total: 0,
                consecutiveAttempts: 0,
                lastMatch: 0,
                lastAttempt: 0
            };
            if ((reasonCounts[reason] || 0) > 0) {
                entry.total += reasonCounts[reason];
                entry.consecutiveAttempts += 1;
                entry.lastMatch = this.State.matchIndex;
                entry.lastAttempt = this.State.matchAttempt;
            }
            else {
                entry.consecutiveAttempts = 0;
            }
            memory[reason] = entry;
        }
        for (reason in reasonCounts) {
            if (!reasonCounts.hasOwnProperty(reason) || seen[reason]) {
                continue;
            }
            entry = memory[reason] || {
                total: 0,
                consecutiveAttempts: 0,
                lastMatch: 0,
                lastAttempt: 0
            };
            if ((reasonCounts[reason] || 0) > 0) {
                entry.total += reasonCounts[reason];
                entry.consecutiveAttempts += 1;
                entry.lastMatch = this.State.matchIndex;
                entry.lastAttempt = this.State.matchAttempt;
            }
            else {
                entry.consecutiveAttempts = 0;
            }
            memory[reason] = entry;
        }

        if (dominantReason) {
            if (this.State.lastDominantReason === dominantReason) {
                this.State.dominantReasonStreak += 1;
            }
            else {
                this.State.lastDominantReason = dominantReason;
                this.State.dominantReasonStreak = 1;
            }
        }

        this.State.reasonMemory = memory;
        return memory;
    },

    LearnFromAttempt: function(attemptSummary) {
        var contextAnalysis = this.AnalyzeFailureContexts(this.State.currentAttemptFailureLogs);
        var counts = contextAnalysis.reasonCounts;
        var memory = this.UpdateReasonMemoryByAttempt(counts, contextAnalysis.dominantReason);
        var actions = [];
        var reason;
        var repeatLevel;
        var adjust;
        var reasonCount;
        var freezeAdjust = IsOfflineMLExpertFreezeEnabled();

        if (!freezeAdjust) {
            for (reason in counts) {
                if (!counts.hasOwnProperty(reason)) {
                    continue;
                }
                reasonCount = counts[reason] || 0;
                if (reasonCount <= 0) {
                    continue;
                }
                repeatLevel = memory[reason] ? memory[reason].consecutiveAttempts : 1;
                adjust = this.ApplyAntiRepeatAdjustment(reason, repeatLevel, contextAnalysis, "attempt_end");
                actions.push(reason + "x" + reasonCount + " -> " + adjust.actions.join("、"));
            }

            if (contextAnalysis.total === 0) {
                adjust = this.ApplyAntiRepeatAdjustment("随机爆线覆盖", 1, contextAnalysis, "attempt_end");
                actions.push("无失败样本 -> " + adjust.actions.join("、"));
            }

            if (contextAnalysis.lowExitRate >= 0.45) {
                AIDodgePolicy.roamRadius += 1;
                AIDodgePolicy.stuckTimeoutMs -= 40;
                NormalizeDodgePolicyValue(AIDodgePolicy);
                SaveAIDodgePolicy();
                actions.push("低出口区域占比高，增大巡逻半径并提前解卡");
            }
            if (contextAnalysis.riskyHalfBodyRate >= 0.35) {
                AIDodgePolicy.halfBodyMinEtaMs += 25;
                NormalizeDodgePolicyValue(AIDodgePolicy);
                SaveAIDodgePolicy();
                actions.push("半身高风险样本偏多，抬高半身缓冲阈值");
            }
        }
        else {
            actions.push("专家策略冻结（仅采集，不在线调参）");
        }

        this.State.latestDeathSummary = contextAnalysis.dominantReason
            ? (contextAnalysis.dominantReason + "（占比" + Math.round((contextAnalysis.dominantCount / Math.max(1, contextAnalysis.total)) * 100) + "%）")
            : "暂无";
        this.State.latestLesson = "主因=" + (contextAnalysis.dominantReason || "无")
            + "，均值eta=" + (contextAnalysis.avgEta == null ? "na" : Math.round(contextAnalysis.avgEta) + "ms")
            + "，均值逃生步数=" + (contextAnalysis.avgEscapeSteps == null ? "na" : contextAnalysis.avgEscapeSteps.toFixed(2))
            + "，均值邻接危险=" + contextAnalysis.avgUnsafeNeighbors.toFixed(2)
            + "，低出口率=" + Math.round(contextAnalysis.lowExitRate * 100) + "%"
            + "；调整：" + actions.join("；");
        this.AddLog("深度分析：主因=" + (contextAnalysis.dominantReason || "无")
            + "，卡死率=" + Math.round(contextAnalysis.stuckRate * 100) + "%"
            + "，低出口率=" + Math.round(contextAnalysis.lowExitRate * 100) + "%"
            + "，均值逃生步数=" + (contextAnalysis.avgEscapeSteps == null ? "na" : contextAnalysis.avgEscapeSteps.toFixed(2)) + "。");
        return this.State.latestLesson;
    },

    BuildMatchDeepAnalysis: function(attempts) {
        var list = attempts || [];
        var i;
        var reason;
        var reasonCounts = {};
        var totalFails = 0;
        var dominantReason = "";
        var dominantCount = 0;
        var attemptScores = [];

        for (i = 0; i < list.length; i++) {
            attemptScores.push(list[i].successCount || 0);
            if (!list[i].eventStats || !list[i].eventStats.reasonCounts) {
                continue;
            }
            for (reason in list[i].eventStats.reasonCounts) {
                if (!list[i].eventStats.reasonCounts.hasOwnProperty(reason)) {
                    continue;
                }
                reasonCounts[reason] = (reasonCounts[reason] || 0) + (list[i].eventStats.reasonCounts[reason] || 0);
                totalFails += list[i].eventStats.reasonCounts[reason] || 0;
                if (reasonCounts[reason] > dominantCount) {
                    dominantCount = reasonCounts[reason];
                    dominantReason = reason;
                }
            }
        }

        return {
            totalFails: totalFails,
            reasonCounts: reasonCounts,
            dominantReason: dominantReason,
            dominantCount: dominantCount,
            attemptScores: attemptScores
        };
    },

    BuildRepeatReasonChains: function(contexts) {
        var list = contexts || [];
        var longest = {};
        var currentReason = "";
        var streak = 0;
        var i;
        var reason;
        var rows = [];

        for (i = 0; i < list.length; i++) {
            reason = list[i] && list[i].reason ? list[i].reason : "未知";
            if (reason === currentReason) {
                streak++;
            }
            else {
                if (currentReason) {
                    longest[currentReason] = Math.max(longest[currentReason] || 0, streak);
                }
                currentReason = reason;
                streak = 1;
            }
        }
        if (currentReason) {
            longest[currentReason] = Math.max(longest[currentReason] || 0, streak);
        }
        for (reason in longest) {
            if (!longest.hasOwnProperty(reason)) {
                continue;
            }
            rows.push({
                reason: reason,
                streak: longest[reason]
            });
        }
        rows.sort(function(a, b) {
            return b.streak - a.streak;
        });
        return rows.slice(0, 5);
    },

    BuildHotZones: function(contexts) {
        var list = contexts || [];
        var heat = {};
        var i;
        var c;
        var key;
        var parts;
        var rows = [];

        for (i = 0; i < list.length; i++) {
            c = list[i] || {};
            if (typeof c.x !== "number" || typeof c.y !== "number" || c.x < 0 || c.y < 0) {
                continue;
            }
            key = c.x + "_" + c.y;
            heat[key] = (heat[key] || 0) + 1;
        }
        for (key in heat) {
            if (!heat.hasOwnProperty(key)) {
                continue;
            }
            parts = key.split("_");
            rows.push({
                x: parseInt(parts[0], 10),
                y: parseInt(parts[1], 10),
                hits: heat[key]
            });
        }
        rows.sort(function(a, b) {
            return b.hits - a.hits;
        });
        return rows.slice(0, 5);
    },

    BuildRoundReview: function(attemptSummary, policyBefore, policyAfter) {
        var summary = attemptSummary || {};
        var contextAnalysis = summary.contextAnalysis || this.AnalyzeFailureContexts(this.State.currentAttemptFailureLogs);
        var reasonCounts = summary.eventStats && summary.eventStats.reasonCounts ? summary.eventStats.reasonCounts : {};
        var topReasons = [];
        var reason;

        for (reason in reasonCounts) {
            if (!reasonCounts.hasOwnProperty(reason)) {
                continue;
            }
            topReasons.push({
                reason: reason,
                count: reasonCounts[reason]
            });
        }
        topReasons.sort(function(a, b) {
            return b.count - a.count;
        });

        return {
            round: this.State.matchIndex,
            attempt: this.State.matchAttempt,
            successCount: summary.successCount || 0,
            failCount: summary.failCount || 0,
            dominantReason: contextAnalysis.dominantReason || "",
            topReasons: topReasons.slice(0, 4),
            repeatReasonChains: this.BuildRepeatReasonChains(this.State.currentAttemptFailureLogs),
            hotZones: this.BuildHotZones(this.State.currentAttemptFailureLogs),
            congestion: {
                avgEta: contextAnalysis.avgEta,
                avgSafeNeighbors: contextAnalysis.avgSafeNeighbors,
                avgUnsafeNeighbors: contextAnalysis.avgUnsafeNeighbors,
                avgActiveBombs: contextAnalysis.avgActiveBombs,
                avgEscapeSteps: contextAnalysis.avgEscapeSteps
            },
            plannerStats: summary.pathPlanStats || this.BuildPlannerStatsSnapshot(),
            policyDelta: this.DiffPolicySnapshot(policyBefore, policyAfter),
            lesson: this.State.latestLesson
        };
    },

    OptimizeForGap: function(delta, review) {
        var needed = this.Config.minIncrementPerRound;
        var gap = Math.max(0, needed - delta);
        var severity = 1 + gap / Math.max(10, needed);
        var dominantReason = review && review.dominantReason ? review.dominantReason : "";
        var before;
        var after;
        var diff;
        var reasonAdjust;

        if (gap <= 0) {
            return null;
        }
        if (IsOfflineMLExpertFreezeEnabled()) {
            return {
                gap: gap,
                delta: delta,
                policyDelta: {
                    forecastMs: 0,
                    safeBufferMs: 0,
                    repathMs: 0,
                    stuckTimeoutMs: 0,
                    roamRadius: 0,
                    halfBodyMinEtaMs: 0
                }
            };
        }

        before = this.CapturePolicySnapshot();
        if (dominantReason) {
            reasonAdjust = this.ApplyAntiRepeatAdjustment(
                dominantReason,
                Math.max(2, Math.round(severity)),
                review || {},
                "gap_retry"
            );
        }

        AIDodgePolicy.forecastMs += Math.round(45 * severity);
        AIDodgePolicy.safeBufferMs += Math.round(18 * severity);
        AIDodgePolicy.repathMs -= Math.round(18 * severity);
        AIDodgePolicy.stuckTimeoutMs -= Math.round(60 * severity);
        AIDodgePolicy.halfBodyMinEtaMs += Math.round(12 * severity);
        AIDodgePolicy.roamRadius += 1;
        NormalizeDodgePolicyValue(AIDodgePolicy);
        SaveAIDodgePolicy();
        AIEvolution.normalizeTactics();
        AIEvolution.refreshBombCooldown();
        AIEvolution.save();
        this.State.adaptiveTuningVersion += 1;
        after = this.CapturePolicySnapshot();
        diff = this.DiffPolicySnapshot(before, after);
        this.AddLog("增量检查未达标：本次较基线仅 +" + delta + "，低于 +" + needed
            + "；强制优化差值=预判" + (diff.forecastMs >= 0 ? "+" : "") + diff.forecastMs
            + "ms, 缓冲" + (diff.safeBufferMs >= 0 ? "+" : "") + diff.safeBufferMs
            + "ms, 重规划" + (diff.repathMs >= 0 ? "+" : "") + diff.repathMs
            + "ms, 解卡" + (diff.stuckTimeoutMs >= 0 ? "+" : "") + diff.stuckTimeoutMs
            + "ms, 半身阈值" + (diff.halfBodyMinEtaMs >= 0 ? "+" : "") + diff.halfBodyMinEtaMs + "ms。");
        if (reasonAdjust) {
            this.AddLog("增量补偿附加策略(" + dominantReason + ")：" + reasonAdjust.actions.join("、"));
        }
        return {
            gap: gap,
            delta: delta,
            policyDelta: diff
        };
    },

    StopTraining: function(reason, attemptSummary, roundReview) {
        var finalSummary = attemptSummary || null;
        var finalReview = roundReview || null;
        var finalResult;

        this.IsRunning = false;
        this.IsAttemptActive = false;
        this.ResetTimers();
        this.State.stopReason = reason || "manual_stop";
        this.State.stopAtRound = this.State.matchIndex;
        if (this.Monster) {
            this.Monster.Stop();
        }
        if (finalReview) {
            this.State.latestReview = finalReview;
        }
        this.AddLog("训练停止：原因=" + this.State.stopReason + "。");

        finalResult = {
            completedMatches: this.State.completedMatches,
            totalAttempts: this.State.totalAttempts,
            totalScore: this.State.totalScore,
            baselineScore: this.State.baselineScore,
            targetScore: this.State.currentMatchTargetScore,
            stopReason: this.State.stopReason,
            stopAtRound: this.State.stopAtRound,
            latestAttempt: finalSummary,
            roundReviews: this.State.roundReviews.slice(0),
            matches: this.State.matchHistory.map(function(item) {
                return {
                    matchIndex: item.matchIndex,
                    attempts: item.attempts.slice(0),
                    bestScore: item.bestScore,
                    targetScore: item.targetScore,
                    metTarget: item.metTarget,
                    eventStats: item.eventStats,
                    deepAnalysis: item.deepAnalysis
                };
            })
        };

        if (typeof window !== "undefined") {
            window.BNBTrainingCompleted = true;
            window.BNBLatestTrainingResult = finalResult;
        }
        this.PublishRuntimeState();
        this.RenderPanel();
    },

    LoadBootstrapState: function() {
        var boot;
        if (typeof window === "undefined" || !window.BNBTrainingBootstrap) {
            return false;
        }
        boot = window.BNBTrainingBootstrap || {};
        if (typeof boot.matchIndex === "number" && boot.matchIndex >= 1) {
            this.State.matchIndex = parseInt(boot.matchIndex, 10);
        }
        if (typeof boot.matchAttempt === "number" && boot.matchAttempt >= 1) {
            this.State.matchAttempt = parseInt(boot.matchAttempt, 10);
        }
        if (typeof boot.completedMatches === "number" && boot.completedMatches >= 0) {
            this.State.completedMatches = parseInt(boot.completedMatches, 10);
        }
        if (typeof boot.totalAttempts === "number" && boot.totalAttempts >= 0) {
            this.State.totalAttempts = parseInt(boot.totalAttempts, 10);
        }
        if (typeof boot.totalScore === "number" && boot.totalScore >= 0) {
            this.State.totalScore = parseInt(boot.totalScore, 10);
        }
        if (typeof boot.baselineScore === "number" && boot.baselineScore >= 0) {
            this.State.baselineScore = parseInt(boot.baselineScore, 10);
        }
        if (typeof boot.targetScore === "number" && boot.targetScore > 0) {
            this.State.currentMatchTargetScore = parseInt(boot.targetScore, 10);
            this.State.targetScore = this.State.currentMatchTargetScore;
        }
        if (Array.isArray(boot.acceptedResults)) {
            this.State.acceptedResults = boot.acceptedResults.slice(0);
        }
        if (Array.isArray(boot.roundReviews)) {
            this.State.roundReviews = boot.roundReviews.slice(0);
        }
        if (Array.isArray(boot.matchHistory)) {
            this.State.matchHistory = boot.matchHistory.slice(0);
        }
        window.BNBTrainingBootstrap = null;
        this.AddLog("已恢复训练上下文：第" + this.State.matchIndex + "轮，目标="
            + (this.State.currentMatchTargetScore == null ? "基线建立中" : this.State.currentMatchTargetScore)
            + "，基线=" + (this.State.baselineScore == null ? "na" : this.State.baselineScore) + "。");
        return true;
    },

    StartMatchAttempt: function(matchIndex, attemptNo) {
        var spawn;
        var self = this;
        var thinkMs;
        this.ResetTimers();
        this.State.matchIndex = matchIndex;
        this.State.matchAttempt = attemptNo;
        this.State.matchDeaths = 0;
        this.State.latestDeathSummary = "";
        this.State.totalAttempts++;
        this.State.deathReasonCounts = {};
        this.BeginAttemptEventTracking();
        this.IsAttemptActive = true;
        this.EnforceHighPressurePolicyFloor("attempt_start");

        spawn = this.FindSafeSpawn();
        this.Role.RespawnAt(spawn.X, spawn.Y, 900);
        this.Monster.Start();
        if (this.Monster && typeof this.Monster.ResetTemporalPlanState === "function") {
            this.Monster.ResetTemporalPlanState(GetThreatSceneRevision());
        }
        this.StripLiveNonRigidBarriers();
        thinkMs = this.Monster && this.Monster.ActiveThinkIntervalMs
            ? this.Monster.ActiveThinkIntervalMs
            : MonsterThinkInterval;
        this.AddLog("开始第" + matchIndex + "轮（尝试 " + attemptNo + "）：持续0.2s随机满威力水泡训练。"
            + "思考间隔=" + thinkMs + "ms。");

        this.SpawnInterval = setInterval(function() {
            if (!self.IsRunning) {
                return;
            }
            self.SpawnRandomBubble();
        }, this.Config.bubbleIntervalMs);

        this.TraceInterval = setInterval(function() {
            if (!self.IsRunning) {
                return;
            }
            self.CaptureTrainingTraceFrame();
            self.SyncExplosionEventProgress();
            if (self.State.matchDeaths >= self.Config.deathLimitPerMatch) {
                self.RequestFinishAttempt();
            }
        }, this.Config.traceSampleMs);
    },

    BuildAttemptSummary: function() {
        var stats = this.State.currentAttemptEventStats || { success: 0, failed: 0, pending: 0, reasonCounts: {} };
        var contextAnalysis = this.AnalyzeFailureContexts(this.State.currentAttemptFailureLogs);
        var mergedReasonCounts = JSON.parse(JSON.stringify(stats.reasonCounts || {}));
        var reason;

        for (reason in contextAnalysis.reasonCounts) {
            if (!contextAnalysis.reasonCounts.hasOwnProperty(reason)) {
                continue;
            }
            mergedReasonCounts[reason] = Math.max(
                mergedReasonCounts[reason] || 0,
                contextAnalysis.reasonCounts[reason] || 0
            );
        }
        return {
            attempt: this.State.matchAttempt,
            successCount: stats.success || 0,
            failCount: stats.failed || 0,
            pendingCount: stats.pending || 0,
            deaths: this.State.matchDeaths,
            eventStats: {
                success: stats.success || 0,
                failed: stats.failed || 0,
                pending: stats.pending || 0,
                reasonCounts: mergedReasonCounts
            },
            contextAnalysis: contextAnalysis,
            pathPlanStats: this.BuildPlannerStatsSnapshot()
        };
    },

    FinishAttempt: function() {
        var attemptSummary;
        var lesson;
        var matchIdx = this.State.matchIndex - 1;
        var currentMatch;
        var bestScore;
        var metTarget;
        var matchDeep;
        var memoryEntry;
        var matchAdjust;
        var roundReview;
        var policyBefore;
        var policyAfter;
        var deltaFromBaseline;
        var nextRound;
        var nextTarget;
        var self = this;

        if (!this.IsRunning) {
            return;
        }
        this.IsAttemptActive = false;
        this.ResetTimers();
        this.SyncExplosionEventProgress();
        policyBefore = this.CapturePolicySnapshot();
        attemptSummary = this.BuildAttemptSummary();
        lesson = this.LearnFromAttempt(attemptSummary);
        policyAfter = this.CapturePolicySnapshot();
        roundReview = this.BuildRoundReview(attemptSummary, policyBefore, policyAfter);
        this.State.latestReview = roundReview;
        this.State.roundReviews.push(roundReview);
        if (this.State.roundReviews.length > 2000) {
            this.State.roundReviews = this.State.roundReviews.slice(this.State.roundReviews.length - 2000);
        }

        this.State.currentMatchAttempts.push(attemptSummary);
        this.AddLog(
            "第" + this.State.matchIndex + "轮-尝试" + this.State.matchAttempt
            + " 结束：躲泡成功 " + attemptSummary.successCount
            + "，失败 " + attemptSummary.failCount
            + "，被炸 " + attemptSummary.deaths + " 次。"
        );
        this.AddLog("经验总结：" + lesson);
        this.AddLog("本轮复盘：主因=" + (roundReview.dominantReason || "无")
            + "，Top原因="
            + (roundReview.topReasons.length > 0
                ? roundReview.topReasons.map(function(item) { return item.reason + "x" + item.count; }).join("/")
                : "无")
            + "，热点="
            + (roundReview.hotZones.length > 0
                ? roundReview.hotZones.map(function(item) { return "(" + item.x + "," + item.y + ")x" + item.hits; }).join("/")
                : "无")
            + "。");
        this.AddLog("规划复盘：重规划=" + (roundReview.plannerStats ? roundReview.plannerStats.replanCount : 0)
            + "（新泡=" + (roundReview.plannerStats ? roundReview.plannerStats.newBubbleReplanCount : 0)
            + "，兜底=" + (roundReview.plannerStats ? roundReview.plannerStats.fallbackReplanCount : 0)
            + "，急迫ETA=" + (roundReview.plannerStats ? roundReview.plannerStats.panicEtaReplanCount : 0)
            + "），阻塞触发=" + (roundReview.plannerStats ? roundReview.plannerStats.blockedFallbackTriggers : 0)
            + "，等待步=" + (roundReview.plannerStats ? roundReview.plannerStats.waitStepsPlanned : 0)
            + "，半身步=" + (roundReview.plannerStats ? roundReview.plannerStats.halfBodyStepsPlanned : 0)
            + "，0步计划=" + (roundReview.plannerStats ? roundReview.plannerStats.zeroStepPlanCount : 0)
            + "，force_move=" + (roundReview.plannerStats ? roundReview.plannerStats.forceMovePlanCount : 0)
            + "，触发占比="
            + (roundReview.plannerStats && roundReview.plannerStats.triggerShare
                ? JSON.stringify(roundReview.plannerStats.triggerShare)
                : "{}")
            + "。");

        currentMatch = this.State.matchHistory[matchIdx];
        if (!currentMatch) {
            currentMatch = {
                matchIndex: this.State.matchIndex,
                attempts: [],
                targetScore: this.State.currentMatchTargetScore,
                bestScore: 0,
                metTarget: false,
                eventStats: null,
                deepAnalysis: null
            };
            this.State.matchHistory[matchIdx] = currentMatch;
        }
        currentMatch.targetScore = this.State.currentMatchTargetScore;
        currentMatch.attempts = this.State.currentMatchAttempts.slice(0);
        bestScore = 0;
        currentMatch.attempts.forEach(function(a) {
            if (a.successCount > bestScore) {
                bestScore = a.successCount;
                currentMatch.eventStats = a.eventStats;
            }
        });
        currentMatch.bestScore = bestScore;
        matchDeep = this.BuildMatchDeepAnalysis(currentMatch.attempts);
        currentMatch.deepAnalysis = matchDeep;
        if (matchDeep.totalFails > 0) {
            this.AddLog("本轮深度复盘：主因=" + (matchDeep.dominantReason || "无")
                + "，失败总数=" + matchDeep.totalFails
                + "，各尝试得分=" + matchDeep.attemptScores.join("/"));
            if (matchDeep.dominantReason) {
                memoryEntry = this.State.reasonMemory[matchDeep.dominantReason];
                matchAdjust = this.ApplyAntiRepeatAdjustment(
                    matchDeep.dominantReason,
                    memoryEntry ? Math.max(1, memoryEntry.consecutiveAttempts) : 1,
                    {
                        safeNeighbors: this.State.currentAttemptFailureLogs.length > 0
                            ? this.State.currentAttemptFailureLogs[this.State.currentAttemptFailureLogs.length - 1].safeNeighbors
                            : 2
                    },
                    "match_end"
                );
                this.AddLog("轮级抑制调参(" + matchDeep.dominantReason + ")："
                    + matchAdjust.actions.join("、"));
            }
        }

        if (attemptSummary.deaths >= this.Config.deathLimitPerMatch
            && attemptSummary.successCount >= this.Config.stopSuccessThreshold) {
            this.StopTraining(
                "success_gt_" + this.Config.stopSuccessThreshold,
                attemptSummary,
                roundReview
            );
            return;
        }

        metTarget = this.State.currentMatchTargetScore == null
            ? true
            : (attemptSummary.successCount >= this.State.currentMatchTargetScore);
        currentMatch.metTarget = metTarget;

        if (metTarget) {
            this.State.completedMatches = this.State.matchIndex;
            this.State.baselineScore = attemptSummary.successCount;
            this.State.acceptedResults.push({ matchScore: attemptSummary.successCount });
            this.AddLog(
                "第" + this.State.matchIndex + "轮达标：目标 "
                + (this.State.currentMatchTargetScore == null ? "基线建立" : this.State.currentMatchTargetScore)
                + "，本次 " + attemptSummary.successCount + "。"
            );
            if (typeof window !== "undefined") {
                if (!window.BNBTrainingMatchLogs) {
                    window.BNBTrainingMatchLogs = [];
                }
                window.BNBTrainingMatchLogs.push({
                    matchIndex: this.State.matchIndex,
                    targetScore: this.State.currentMatchTargetScore,
                    bestScore: bestScore,
                    metTarget: true,
                    attempts: currentMatch.attempts.slice(0),
                    eventStats: currentMatch.eventStats,
                    deepAnalysis: currentMatch.deepAnalysis,
                    roundReview: roundReview
                });
            }
            nextRound = this.State.matchIndex + 1;
            nextTarget = this.State.baselineScore + this.Config.minIncrementPerRound;
            this.State.matchIndex = nextRound;
            this.State.matchAttempt = 1;
            this.State.currentMatchAttempts = [];
            this.State.currentMatchTargetScore = nextTarget;
            this.State.targetScore = nextTarget;
            this.AddLog("进入第" + nextRound + "轮：目标 >= " + nextTarget + "（基线+"
                + this.Config.minIncrementPerRound + "）。");
            this.NextStepTimeout = setTimeout(function() {
                self.StartMatchAttempt(self.State.matchIndex, self.State.matchAttempt);
            }, this.Config.matchGapMs);
            return;
        }

        deltaFromBaseline = this.State.baselineScore == null
            ? 0
            : (attemptSummary.successCount - this.State.baselineScore);
        this.AddLog("未达标：本次较基线 +" + deltaFromBaseline + "，要求至少 +"
            + this.Config.minIncrementPerRound + "；立即进入深度调参并继续重试当前目标。");
        this.OptimizeForGap(deltaFromBaseline, roundReview);
        this.State.matchAttempt += 1;
        this.AddLog("继续第" + this.State.matchIndex + "轮（尝试 " + this.State.matchAttempt + "）：目标仍为 >= "
            + this.State.currentMatchTargetScore + "。");
        this.NextStepTimeout = setTimeout(function() {
            self.StartMatchAttempt(self.State.matchIndex, self.State.matchAttempt);
        }, this.Config.matchGapMs);
        return;
    },

    StartNextMatchOrFinish: function() {
        var nextMatchIndex;
        var nextTarget;
        if (!this.IsRunning) {
            return;
        }
        if (!this.Config.unboundedRounds && this.State.matchIndex >= this.Config.targetIterations) {
            this.StopTraining("target_round_reached", null, null);
            return;
        }

        nextMatchIndex = this.State.matchIndex + 1;
        nextTarget = this.State.baselineScore == null
            ? null
            : (this.State.baselineScore + this.Config.minIncrementPerRound);
        this.State.matchIndex = nextMatchIndex;
        this.State.matchAttempt = 1;
        this.State.currentMatchAttempts = [];
        this.State.currentMatchTargetScore = nextTarget;
        this.State.targetScore = nextTarget;
        this.AddLog("进入第" + nextMatchIndex + "轮：目标="
            + (nextTarget == null ? "基线建立" : nextTarget) + "。");
        this.StartMatchAttempt(nextMatchIndex, 1);
    },

    Start: function(monster) {
        var restored = false;
        this.ResetTimers();
        this.Monster = monster;
        this.Role = monster ? monster.Role : null;
        this.IsRunning = true;
        this.State.matchIndex = 1;
        this.State.matchAttempt = 1;
        this.State.roundIndex = 0;
        this.State.matchScore = 0;
        this.State.matchDeaths = 0;
        this.State.matchSurviveRounds = 0;
        this.State.totalScore = 0;
        this.State.totalDeaths = 0;
        this.State.totalAttempts = 0;
        this.State.completedMatches = 0;
        this.State.baselineScore = null;
        this.State.targetScore = null;
        this.State.latestLesson = "";
        this.State.latestDeathSummary = "";
        this.State.acceptedResults = [];
        this.State.matchHistory = [];
        this.State.roundReviews = [];
        this.State.logs = [];
        this.State.deathReasonCounts = {};
        this.State.currentMatchAttempts = [];
        this.State.currentMatchTargetScore = null;
        this.State.targetScore = null;
        this.State.currentAttemptFailureLogs = [];
        this.State.traceFrames = [];
        this.State.failureHeatMap = {};
        this.State.reasonMemory = {};
        this.State.lastDominantReason = "";
        this.State.dominantReasonStreak = 0;
        this.State.adaptiveTuningVersion = 0;
        this.State.reasonGuards = {};
        this.State.lastBombReason = "";
        this.State.sameReasonHitStreak = 0;
        this.State.stopReason = "";
        this.State.stopAtRound = 0;
        this.State.latestReview = null;
        this.State.currentAttemptPlannerStats = null;
        this.IsAttemptActive = false;

        this.EnsurePanel();
        this.RenderPanel();
        this.BuildBubbleCaster();
        window.BNBPaopaoFuseMs = this.Config.bubbleFuseMs;
        if (typeof window !== "undefined") {
            window.BNBTrainingCompleted = false;
            window.BNBTrainingMatchLogs = [];
            window.BNBLatestTrainingResult = null;
            window.BNBTrainingRuntimeState = null;
        }

        if (this.Role) {
            var self = this;
            this.Role.OnDeath = function(victim) {
                self.OnRoleDeath(victim);
            };
            this.Role.OnBombed = function(victim) {
                self.OnRoleBombed(victim);
            };
        }

        restored = this.LoadBootstrapState();
        this.EnforceHighPressurePolicyFloor(restored ? "resume_start" : "fresh_start");
        if (!restored) {
            this.AddLog("训练目标：无限迭代；每次被炸10次结算，单次成功>="
                + this.Config.stopSuccessThreshold + "即停止；未达到基线+"
                + this.Config.minIncrementPerRound + "则自动优化并继续重试。");
        }
        if (IsOfflineMLExpertFreezeEnabled()) {
            this.AddLog("Offline ML采集模式：专家策略冻结，禁用在线自调参。");
        }
        if (this.State.currentMatchTargetScore == null && this.State.baselineScore != null) {
            this.State.currentMatchTargetScore = this.State.baselineScore + this.Config.minIncrementPerRound;
            this.State.targetScore = this.State.currentMatchTargetScore;
        }
        this.StartMatchAttempt(this.State.matchIndex || 1, this.State.matchAttempt || 1);
    }
};

function StartAIDodgeTraining() {
    var spawn = { X: 1, Y: 1 };
    var monster;
    RefreshBNBMLConfigFromQuery();
    BNBMLDatasetCollector.Init();
    BNBMLRuntime.Init();
    if (AIDodgeTrainer && AIDodgeTrainer.IsRunning) {
        return AIDodgeTrainer;
    }
    if (typeof window !== "undefined") {
        window.BNBTrainingStripNonRigid = true;
    }

    if (typeof SetCurrentGameMap === "function") {
        SetCurrentGameMap("windmill-heart");
    }
    if (typeof SaveSelectedGameMap === "function") {
        SaveSelectedGameMap("windmill-heart");
    }
    if (typeof StripNonRigidBarriersFromMap === "function") {
        StripNonRigidBarriersFromMap();
    }

    InitGame();

    if (typeof GetCurrentGameMapSpawn === "function") {
        spawn = GetCurrentGameMapSpawn();
    }
    monster = new Monster();
    monster.SetMap(spawn.X, spawn.Y);

    AIDodgeTrainer.Start(monster);
    return AIDodgeTrainer;
}
