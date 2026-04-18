var resPrefix = 'game/';

//泡泡
var PaopaoArray = [];
var DefaultPaopaoImage = resPrefix + "Pic/Popo.png";
var PlayerPaopaoSkinImageMap = {
    football: resPrefix + "Pic/PopoFootball.png",
    basketball: resPrefix + "Pic/PopoBasketball.png"
};
var DefaultPlayerPaopaoSkin = "football";
var DefaultPaopaoFuseMs = 3000;
var ExplosionAnimStartFrame = 6;
var ExplosionAnimEndFrame = 13;
var ExplosionAnimFrameIntervalMs = 50;
var ExplosionForceDisposeMs = 900;
var ExplosionHitRequiredFrames = 2;
var ActiveExplosionZones = [];
var ExplosionSafetyDebugLogs = [];
var ExplosionSafetyDebugMaxLogs = 180;
var ExplosionSafetyDebugPanelId = "explosion-safety-debug-panel";
var ExplosionSafetyDebugListId = "explosion-safety-debug-list";
var PaopaoEventIdSeed = 0;
var ExplosionZoneIdSeed = 0;
var BNBThreatSceneRevision = 0;
var BNBLastBubbleSpawnEvent = null;
var ExplosionSafetyDebugModeCache = null;

function ParseExplosionDebugBool(raw) {
    if (raw == null) {
        return null;
    }
    if (typeof raw === "boolean") {
        return raw;
    }
    raw = String(raw).toLowerCase();
    if (raw === "1" || raw === "true" || raw === "yes" || raw === "on") {
        return true;
    }
    if (raw === "0" || raw === "false" || raw === "no" || raw === "off") {
        return false;
    }
    return null;
}

function ResolveExplosionSafetyDebugMode() {
    var query;
    var trainMode = false;
    var battleMode = false;
    var panelEnabled;
    var consoleEnabled;
    var panelQuery;
    var consoleQuery;
    var panelFlag;
    var consoleFlag;
    if (ExplosionSafetyDebugModeCache) {
        return ExplosionSafetyDebugModeCache;
    }
    if (typeof window === "undefined" || !window.location) {
        ExplosionSafetyDebugModeCache = { panel: false, console: false };
        return ExplosionSafetyDebugModeCache;
    }
    query = new URLSearchParams(window.location.search || "");
    trainMode = query.get("train") === "1";
    battleMode = query.get("mode") === "battle";
    panelQuery = ParseExplosionDebugBool(query.get("safety_debug"));
    consoleQuery = ParseExplosionDebugBool(query.get("safety_debug_console"));
    panelFlag = ParseExplosionDebugBool(window.BNBEnableExplosionSafetyDebugPanel);
    if (panelFlag == null) {
        panelFlag = ParseExplosionDebugBool(window.BNBEnableExplosionSafetyDebug);
    }
    consoleFlag = ParseExplosionDebugBool(window.BNBEnableExplosionSafetyConsole);

    panelEnabled = panelQuery;
    if (panelEnabled == null) {
        panelEnabled = panelFlag;
    }
    if (panelEnabled == null) {
        panelEnabled = trainMode && !battleMode;
    }

    consoleEnabled = consoleQuery;
    if (consoleEnabled == null) {
        consoleEnabled = consoleFlag;
    }
    if (consoleEnabled == null) {
        consoleEnabled = false;
    }

    ExplosionSafetyDebugModeCache = {
        panel: !!panelEnabled,
        console: !!consoleEnabled
    };
    return ExplosionSafetyDebugModeCache;
}

function IsExplosionSafetyDebugPanelEnabled() {
    return ResolveExplosionSafetyDebugMode().panel;
}

function IsExplosionSafetyDebugConsoleEnabled() {
    return ResolveExplosionSafetyDebugMode().console;
}

function ShouldOutputExplosionSafetyDebug() {
    var mode = ResolveExplosionSafetyDebugMode();
    return mode.panel || mode.console;
}

function PublishThreatSceneMeta() {
    if (typeof window === "undefined") {
        return;
    }
    window.BNBThreatSceneRevision = BNBThreatSceneRevision;
    if (!BNBLastBubbleSpawnEvent) {
        window.BNBLastBubbleSpawnEvent = null;
        return;
    }
    window.BNBLastBubbleSpawnEvent = {
        eventId: BNBLastBubbleSpawnEvent.eventId,
        x: BNBLastBubbleSpawnEvent.x,
        y: BNBLastBubbleSpawnEvent.y,
        startAt: BNBLastBubbleSpawnEvent.startAt,
        endAt: BNBLastBubbleSpawnEvent.endAt,
        clusterId: BNBLastBubbleSpawnEvent.clusterId || ""
    };
}

function BuildLatestBubbleSpawnEventFromBomb(bomb, now) {
    var current = typeof now === "number" ? now : Date.now();
    var startAt;
    var endAt;
    var eventId;
    var snapshot;
    var i;
    var match;

    if (!bomb || typeof bomb.BombId !== "number") {
        return null;
    }
    startAt = typeof bomb.ExplodeAt === "number" ? bomb.ExplodeAt : current + GetPaopaoFuseMs();
    endAt = startAt + GetExplosionUnsafeWindowMs();
    eventId = "B" + bomb.BombId;

    match = {
        eventId: eventId,
        x: bomb.CurrentMapID ? bomb.CurrentMapID.X : -1,
        y: bomb.CurrentMapID ? bomb.CurrentMapID.Y : -1,
        startAt: startAt,
        endAt: endAt,
        clusterId: ""
    };
    snapshot = BuildBNBExplosionEventSnapshot(current);
    for (i = 0; i < snapshot.bombs.length; i++) {
        if (snapshot.bombs[i].eventId === eventId) {
            match.clusterId = snapshot.bombs[i].clusterId || "";
            match.startAt = snapshot.bombs[i].startAt;
            match.endAt = snapshot.bombs[i].endAt;
            break;
        }
    }
    return match;
}

function TouchThreatSceneRevisionOnBubbleSpawn(bomb) {
    var spawnEvent = BuildLatestBubbleSpawnEventFromBomb(bomb, Date.now());
    BNBThreatSceneRevision += 1;
    BNBLastBubbleSpawnEvent = spawnEvent;
    PublishThreatSceneMeta();
}

function ResetExplosionUnsafeZones() {
    ActiveExplosionZones = [];
    ExplosionSafetyDebugLogs = [];
    ExplosionZoneIdSeed = 0;
    PaopaoEventIdSeed = 0;
    BNBThreatSceneRevision = 0;
    BNBLastBubbleSpawnEvent = null;
    PublishThreatSceneMeta();
    if (typeof document !== "undefined") {
        var listNode = document.getElementById(ExplosionSafetyDebugListId);
        if (listNode) {
            listNode.innerHTML = "";
        }
    }
}

function GetPaopaoFuseMs() {
    var fuseMs = DefaultPaopaoFuseMs;
    if (typeof window !== "undefined" && typeof window.BNBPaopaoFuseMs === "number") {
        fuseMs = window.BNBPaopaoFuseMs;
    }
    fuseMs = parseInt(fuseMs, 10);
    if (isNaN(fuseMs) || fuseMs < 300) {
        fuseMs = DefaultPaopaoFuseMs;
    }
    return fuseMs;
}

function GetCurrentPlayerPaopaoImage() {
    var skin = DefaultPlayerPaopaoSkin;

    if (typeof window !== "undefined" && window.PlayerBubbleSkin) {
        skin = window.PlayerBubbleSkin;
    }
    if (!PlayerPaopaoSkinImageMap[skin]) {
        skin = DefaultPlayerPaopaoSkin;
    }

    return PlayerPaopaoSkinImageMap[skin];
}

function GetExplosionUnsafeWindowMs() {
    // 当前动画逻辑下：第一个判定帧到判定清除帧之间一共是 (end-start+2) 个 interval
    var windowMs = (ExplosionAnimEndFrame - ExplosionAnimStartFrame + 2) * ExplosionAnimFrameIntervalMs;
    if (windowMs <= 0) {
        windowMs = ExplosionAnimFrameIntervalMs;
    }
    return Math.min(windowMs, ExplosionForceDisposeMs);
}

function BuildUniqueSortedMapIds(mapIds) {
    var lookup = {};
    var unique = [];
    var i;
    var id;

    for (i = 0; i < mapIds.length; i++) {
        id = parseInt(mapIds[i], 10);
        if (isNaN(id) || lookup[id]) {
            continue;
        }
        lookup[id] = true;
        unique.push(id);
    }
    unique.sort(function(a, b) {
        return a - b;
    });
    return unique;
}

function BuildMapLookup(mapIds) {
    var lookup = {};
    for (var i = 0; i < mapIds.length; i++) {
        lookup[mapIds[i]] = true;
    }
    return lookup;
}

function IsMapNoNeighborConnected(mapNo, otherLookup) {
    var xy = MapNoToXY(mapNo);
    var left;
    var right;
    var up;
    var down;

    if (!xy) {
        return false;
    }
    left = xy.X > 0 ? mapNo - 1 : null;
    right = xy.X < 14 ? mapNo + 1 : null;
    up = xy.Y > 0 ? mapNo - 15 : null;
    down = xy.Y < 12 ? mapNo + 15 : null;

    return (left != null && !!otherLookup[left])
        || (right != null && !!otherLookup[right])
        || (up != null && !!otherLookup[up])
        || (down != null && !!otherLookup[down]);
}

function AreMapCollectionsConnected(aMapIds, bMapIds) {
    var aLookup = BuildMapLookup(aMapIds);
    var bLookup = BuildMapLookup(bMapIds);
    var i;
    var mapNo;

    for (i = 0; i < aMapIds.length; i++) {
        mapNo = aMapIds[i];
        if (bLookup[mapNo] || IsMapNoNeighborConnected(mapNo, bLookup)) {
            return true;
        }
    }
    for (i = 0; i < bMapIds.length; i++) {
        mapNo = bMapIds[i];
        if (aLookup[mapNo] || IsMapNoNeighborConnected(mapNo, aLookup)) {
            return true;
        }
    }
    return false;
}

function BuildConnectedClusters(items, getMapIds, idPrefix) {
    var parent = [];
    var i;
    var j;
    var rootToMembers = {};
    var roots = [];
    var root;
    var clusters = [];
    var clusterIndex = 1;

    function find(x) {
        while (parent[x] !== x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    function union(a, b) {
        var ra = find(a);
        var rb = find(b);
        if (ra !== rb) {
            parent[rb] = ra;
        }
    }

    for (i = 0; i < items.length; i++) {
        parent[i] = i;
    }

    for (i = 0; i < items.length; i++) {
        for (j = i + 1; j < items.length; j++) {
            if (AreMapCollectionsConnected(getMapIds(items[i]), getMapIds(items[j]))) {
                union(i, j);
            }
        }
    }

    for (i = 0; i < items.length; i++) {
        root = find(i);
        if (!rootToMembers[root]) {
            rootToMembers[root] = [];
            roots.push(root);
        }
        rootToMembers[root].push(i);
    }

    roots.sort(function(a, b) {
        return a - b;
    });

    for (i = 0; i < roots.length; i++) {
        clusters.push({
            ClusterId: idPrefix + clusterIndex,
            MemberIndexes: rootToMembers[roots[i]]
        });
        clusterIndex++;
    }

    return clusters;
}

function GetBombCoverageMapIdsByCenterMapNo(centerMapNo, strong) {
    var blast = FindPaopaoBombXY(centerMapNo, strong);
    var all = blast.X.concat(blast.Y);
    all.push(centerMapNo);
    return BuildUniqueSortedMapIds(all);
}

function BuildPendingBombEvents(now) {
    var events = [];
    var y;
    var x;
    var bomb;
    var centerMapNo;
    var startAt;
    var coverageMapIds;
    var windowMs = GetExplosionUnsafeWindowMs();

    for (y = 0; y < PaopaoArray.length; y++) {
        if (!PaopaoArray[y]) {
            continue;
        }
        for (x = 0; x < PaopaoArray[y].length; x++) {
            bomb = PaopaoArray[y][x];
            if (!bomb || bomb.IsExploded) {
                continue;
            }
            if (typeof bomb.BombId !== "number") {
                bomb.BombId = ++PaopaoEventIdSeed;
            }

            centerMapNo = y * 15 + x;
            startAt = typeof bomb.ExplodeAt === "number" ? bomb.ExplodeAt : now + GetPaopaoFuseMs();
            coverageMapIds = GetBombCoverageMapIdsByCenterMapNo(centerMapNo, bomb.PaopaoStrong);
            events.push({
                EventId: "B" + bomb.BombId,
                BombId: bomb.BombId,
                CenterMapNo: centerMapNo,
                X: x,
                Y: y,
                Strong: bomb.PaopaoStrong,
                StartAt: startAt,
                EndAt: startAt + windowMs,
                CoverageMapIds: coverageMapIds,
                ClusterId: ""
            });
        }
    }

    return events;
}

function BuildPendingBombClusters(now, bombEvents) {
    var clusters = BuildConnectedClusters(bombEvents, function(item) {
        return item.CoverageMapIds;
    }, "PC");
    var i;
    var j;
    var members;
    var mergedMapIds;
    var earliestStart;
    var cluster;
    var member;
    var windowMs = GetExplosionUnsafeWindowMs();
    var result = [];

    for (i = 0; i < clusters.length; i++) {
        members = clusters[i].MemberIndexes;
        mergedMapIds = [];
        earliestStart = null;
        for (j = 0; j < members.length; j++) {
            member = bombEvents[members[j]];
            mergedMapIds = mergedMapIds.concat(member.CoverageMapIds);
            if (earliestStart == null || member.StartAt < earliestStart) {
                earliestStart = member.StartAt;
            }
        }
        mergedMapIds = BuildUniqueSortedMapIds(mergedMapIds);
        cluster = {
            ClusterId: clusters[i].ClusterId,
            StartAt: earliestStart == null ? now : earliestStart,
            EndAt: (earliestStart == null ? now : earliestStart) + windowMs,
            CoverageMapIds: mergedMapIds,
            BombEventIds: []
        };
        for (j = 0; j < members.length; j++) {
            member = bombEvents[members[j]];
            member.ClusterId = cluster.ClusterId;
            member.StartAt = cluster.StartAt;
            member.EndAt = cluster.EndAt;
            cluster.BombEventIds.push(member.EventId);
        }
        result.push(cluster);
    }

    return result;
}

function BuildActiveExplosionWindows(now) {
    var activeZones = [];
    var i;
    var zone;
    var clusters;
    var activeClusters = [];
    var j;
    var members;
    var member;
    var mergedMapIds;
    var startAt;
    var item;
    var windowMs = GetExplosionUnsafeWindowMs();

    PruneExpiredExplosionZones(now);
    for (i = 0; i < ActiveExplosionZones.length; i++) {
        zone = ActiveExplosionZones[i];
        if (zone.ExpiresAt <= now) {
            continue;
        }
        activeZones.push(zone);
    }

    clusters = BuildConnectedClusters(activeZones, function(z) {
        return z.MapIds;
    }, "AC");

    for (i = 0; i < clusters.length; i++) {
        members = clusters[i].MemberIndexes;
        mergedMapIds = [];
        startAt = null;
        item = {
            ClusterId: clusters[i].ClusterId,
            StartAt: now,
            EndAt: now,
            CoverageMapIds: [],
            EventIds: [],
            ZoneIds: []
        };

        for (j = 0; j < members.length; j++) {
            member = activeZones[members[j]];
            mergedMapIds = mergedMapIds.concat(member.MapIds);
            startAt = startAt == null || member.StartAt < startAt ? member.StartAt : startAt;
            item.EventIds.push(member.EventId);
            item.ZoneIds.push(member.ZoneId);
        }

        item.StartAt = startAt == null ? now : startAt;
        // 连片同窗：统一以最早爆炸时间 + 固定水柱窗口作为活动时长
        item.EndAt = item.StartAt + windowMs;
        if (item.EndAt <= now) {
            continue;
        }
        item.CoverageMapIds = BuildUniqueSortedMapIds(mergedMapIds);
        for (j = 0; j < members.length; j++) {
            activeZones[members[j]].ActiveClusterId = item.ClusterId;
        }
        activeClusters.push(item);
    }

    return {
        ActiveZones: activeZones,
        ActiveClusters: activeClusters
    };
}

function BuildBNBExplosionEventSnapshot(now) {
    var timestamp = typeof now === "number" ? now : Date.now();
    var pendingBombs = BuildPendingBombEvents(timestamp);
    var pendingClusters = BuildPendingBombClusters(timestamp, pendingBombs);
    var active = BuildActiveExplosionWindows(timestamp);

    return {
        now: timestamp,
        unsafeWindowMs: GetExplosionUnsafeWindowMs(),
        bombs: pendingBombs.map(function(item) {
            return {
                eventId: item.EventId,
                bombId: item.BombId,
                clusterId: item.ClusterId,
                x: item.X,
                y: item.Y,
                startAt: item.StartAt,
                endAt: item.EndAt,
                coverageMapIds: item.CoverageMapIds.slice(0)
            };
        }),
        clusters: pendingClusters.map(function(item) {
            return {
                clusterId: item.ClusterId,
                startAt: item.StartAt,
                endAt: item.EndAt,
                coverageMapIds: item.CoverageMapIds.slice(0),
                bombEventIds: item.BombEventIds.slice(0)
            };
        }),
        activeWindows: active.ActiveClusters.map(function(item) {
            return {
                clusterId: item.ClusterId,
                startAt: item.StartAt,
                endAt: item.EndAt,
                coverageMapIds: item.CoverageMapIds.slice(0),
                eventIds: item.EventIds.slice(0),
                zoneIds: item.ZoneIds.slice(0)
            };
        })
    };
}

function RegisterExplosionUnsafeZone(allMapIds, centerMapId, attacker, sourceBomb) {
    var unsafeLookup;
    var now = Date.now();
    var zoneId = ++ExplosionZoneIdSeed;
    var eventId;
    var sourceBombId = sourceBomb && typeof sourceBomb.BombId === "number" ? sourceBomb.BombId : null;
    var mapIds = allMapIds.concat([centerMapId]);
    var uniqueMapIds = BuildUniqueSortedMapIds(mapIds);
    var zone;

    unsafeLookup = BuildMapLookup(uniqueMapIds);
    eventId = sourceBombId != null ? ("B" + sourceBombId) : ("Z" + zoneId);

    zone = {
        ZoneId: zoneId,
        EventId: eventId,
        SourceBombId: sourceBombId,
        CenterMapId: centerMapId,
        MapIds: uniqueMapIds,
        UnsafeLookup: unsafeLookup,
        Attacker: attacker || null,
        StartAt: now,
        // 先给兜底上限，正常会在水柱销毁时提前结束
        ExpiresAt: now + ExplosionForceDisposeMs,
        ActiveClusterId: ""
    };
    ActiveExplosionZones.push(zone);
    return zone;
}

function ExpireExplosionUnsafeZone(zone) {
    if (!zone) {
        return;
    }
    zone.ExpiresAt = Date.now();
}

function PruneExpiredExplosionZones(now) {
    for (var i = ActiveExplosionZones.length - 1; i >= 0; i--) {
        if (ActiveExplosionZones[i].ExpiresAt <= now) {
            ActiveExplosionZones.splice(i, 1);
        }
    }
}

function BuildUnsafeZoneSnapshot(now) {
    var unsafeLookup = {};
    var unsafeAttackerLookup = {};
    var unsafeEventLookup = {};
    var unsafeEventListLookup = {};
    var i;
    var zone;
    var mapNo;
    var eventList;

    PruneExpiredExplosionZones(now);
    for (i = 0; i < ActiveExplosionZones.length; i++) {
        zone = ActiveExplosionZones[i];
        for (mapNo in zone.UnsafeLookup) {
            if (!zone.UnsafeLookup.hasOwnProperty(mapNo)) {
                continue;
            }
            unsafeLookup[mapNo] = true;
            if (zone.Attacker) {
                unsafeAttackerLookup[mapNo] = zone.Attacker;
            }
            if (!unsafeEventLookup[mapNo]) {
                unsafeEventLookup[mapNo] = zone.EventId;
            }
            eventList = unsafeEventListLookup[mapNo];
            if (!eventList) {
                eventList = [];
                unsafeEventListLookup[mapNo] = eventList;
            }
            if (eventList.indexOf(zone.EventId) === -1) {
                eventList.push(zone.EventId);
            }
        }
    }

    return {
        UnsafeLookup: unsafeLookup,
        UnsafeAttackerLookup: unsafeAttackerLookup,
        UnsafeEventLookup: unsafeEventLookup,
        UnsafeEventListLookup: unsafeEventListLookup
    };
}

function MapNoToXY(mapNo) {
    if (typeof mapNo !== "number" || mapNo < 0) {
        return null;
    }
    return {
        X: mapNo % 15,
        Y: parseInt(mapNo / 15, 10)
    };
}

function FormatMapNo(mapNo) {
    var xy = MapNoToXY(mapNo);
    if (!xy) {
        return "越界";
    }
    return "(" + xy.X + "," + xy.Y + ")";
}

function GetRoleDebugName(role, index) {
    if (!role) {
        return "未知角色";
    }
    if (role.RoleNumber == 1) {
        return "玩家";
    }
    if (typeof index !== "number" || index < 0) {
        return "AI";
    }
    return "AI#" + (index + 1);
}

function EnsureExplosionSafetyDebugPanel() {
    var panel;
    var title;
    var list;
    var host;

    if (typeof document === "undefined") {
        return null;
    }

    panel = document.getElementById(ExplosionSafetyDebugPanelId);
    if (panel) {
        return panel;
    }

    panel = document.createElement("aside");
    panel.id = ExplosionSafetyDebugPanelId;
    panel.style.position = "relative";
    panel.style.width = "100%";
    panel.style.maxWidth = "800px";
    panel.style.maxHeight = "220px";
    panel.style.overflow = "hidden";
    panel.style.marginTop = "10px";
    panel.style.padding = "10px 12px";
    panel.style.boxSizing = "border-box";
    panel.style.borderRadius = "10px";
    panel.style.background = "rgba(8, 12, 22, 0.92)";
    panel.style.border = "1px solid rgba(255,255,255,0.15)";
    panel.style.color = "#e8f1ff";
    panel.style.zIndex = "1";

    title = document.createElement("div");
    title.style.fontSize = "12px";
    title.style.fontWeight = "bold";
    title.style.marginBottom = "6px";
    title.textContent = "安全区判定日志";

    list = document.createElement("ol");
    list.id = ExplosionSafetyDebugListId;
    list.style.margin = "0";
    list.style.paddingLeft = "16px";
    // 最多显示 8 条日志的可视高度，超出后滚动
    list.style.maxHeight = "156px";
    list.style.overflowY = "auto";
    list.style.overflowX = "hidden";
    list.style.fontSize = "12px";
    list.style.lineHeight = "1.45";
    list.style.color = "#d8e6ff";

    panel.appendChild(title);
    panel.appendChild(list);
    host = GetExplosionSafetyDebugHost();
    host.appendChild(panel);
    SyncExplosionSafetyDebugPanelLayout(panel);
    BindExplosionSafetyDebugPanelResize();

    return panel;
}

function GetExplosionSafetyDebugHost() {
    var gameRoot = document.getElementById("game-root");
    var canvas;

    if (gameRoot) {
        gameRoot.style.display = "flex";
        gameRoot.style.flexDirection = "column";
        gameRoot.style.alignItems = "center";
        gameRoot.style.justifyContent = "flex-start";
        return gameRoot;
    }

    canvas = document.querySelector("canvas");
    if (canvas && canvas.parentNode) {
        return canvas.parentNode;
    }
    return document.body;
}

function SyncExplosionSafetyDebugPanelLayout(panel) {
    var gameRoot;
    var canvas;
    var widthPx;

    if (!panel || typeof document === "undefined") {
        return;
    }

    gameRoot = document.getElementById("game-root");
    canvas = document.querySelector("canvas");
    widthPx = 800;
    if (canvas && canvas.clientWidth > 0) {
        widthPx = canvas.clientWidth;
    }
    else if (gameRoot && gameRoot.clientWidth > 0) {
        widthPx = gameRoot.clientWidth;
    }

    if (gameRoot) {
        panel.style.width = "100%";
        panel.style.maxWidth = widthPx + "px";
    }
    else {
        panel.style.width = widthPx + "px";
        panel.style.maxWidth = widthPx + "px";
    }
}

function BindExplosionSafetyDebugPanelResize() {
    if (typeof window === "undefined") {
        return;
    }
    if (window.__bnbExplosionDebugPanelResizeBound) {
        return;
    }

    window.addEventListener("resize", function() {
        var panel = document.getElementById(ExplosionSafetyDebugPanelId);
        SyncExplosionSafetyDebugPanelLayout(panel);
    });
    window.__bnbExplosionDebugPanelResizeBound = true;
}

function PushExplosionSafetyDebugLog(message) {
    var panel;
    var list;
    var li;
    var time;
    var text;
    var listChildren;
    var mode;

    if (!message) {
        return;
    }
    mode = ResolveExplosionSafetyDebugMode();
    if (!mode.panel && !mode.console) {
        return;
    }

    time = new Date();
    text = "[" + time.toLocaleTimeString() + "] " + message;
    ExplosionSafetyDebugLogs.push(text);
    if (ExplosionSafetyDebugLogs.length > ExplosionSafetyDebugMaxLogs) {
        ExplosionSafetyDebugLogs = ExplosionSafetyDebugLogs.slice(ExplosionSafetyDebugLogs.length - ExplosionSafetyDebugMaxLogs);
    }

    if (mode.console && typeof console !== "undefined" && typeof console.log === "function") {
        console.log("[安全判定] " + text);
    }

    if (!mode.panel) {
        return;
    }
    panel = EnsureExplosionSafetyDebugPanel();
    if (!panel) {
        return;
    }
    SyncExplosionSafetyDebugPanelLayout(panel);
    list = document.getElementById(ExplosionSafetyDebugListId);
    if (!list) {
        return;
    }

    li = document.createElement("li");
    li.textContent = text;
    // 倒序：最新日志放在最上面
    list.insertBefore(li, list.firstChild);
    listChildren = list.children;
    while (listChildren.length > ExplosionSafetyDebugMaxLogs) {
        list.removeChild(list.lastChild);
    }
    list.scrollTop = 0;
}

function HasAnyPaopaoOnMap() {
    var row;
    var x;
    var y;
    for (y = 0; y < PaopaoArray.length; y++) {
        row = PaopaoArray[y];
        if (!row) {
            continue;
        }
        for (x = 0; x < row.length; x++) {
            if (row[x]) {
                return true;
            }
        }
    }
    return false;
}

function EmitRoleSafetyLog(role, index, key, message) {
    if (!role) {
        return;
    }
    if (!ShouldOutputExplosionSafetyDebug()) {
        return;
    }
    if (role.__ExplosionSafetyDebugKey === key) {
        return;
    }
    role.__ExplosionSafetyDebugKey = key;
    PushExplosionSafetyDebugLog(GetRoleDebugName(role, index) + "：" + message);
}

function TickExplosionUnsafeByFrame() {
    var now = Date.now();
    var snapshot = BuildUnsafeZoneSnapshot(now);
    var hasPaopao = HasAnyPaopaoOnMap();
    var hasActiveExplosionZone = ActiveExplosionZones.length > 0;
    var role;
    var state;
    var attacker;
    var safeReason;
    var attackerName;

    // 仅在地图存在泡泡或仍有爆炸窗口时检测
    if (!hasPaopao && !hasActiveExplosionZone) {
        for (var r = 0; r < RoleStorage.length; r++) {
            role = RoleStorage[r];
            if (!role) {
                continue;
            }
            role.ResetExplosionUnsafeFrameCount();
            role.__ExplosionSafetyDebugKey = "";
        }
        return;
    }

    for (var i = 0; i < RoleStorage.length; i++) {
        role = RoleStorage[i];
        if (!role) {
            continue;
        }

        if (role.IsDeath || role.IsInPaopao || role.IsBombImmune) {
            role.ResetExplosionUnsafeFrameCount();
            if (!role.IsDeath && role.IsInPaopao) {
                EmitRoleSafetyLog(role, i, "safe-trap", "已在困泡状态，跳过安全区判定。");
            }
            continue;
        }
        if (role.DismountProtectionUntil > now || role.ExplosionImmuneUntil > now) {
            role.ResetExplosionUnsafeFrameCount();
            EmitRoleSafetyLog(role, i, "safe-immune", "处于无敌保护时间，判定为安全。");
            continue;
        }

        if (!hasActiveExplosionZone) {
            role.ResetExplosionUnsafeFrameCount();
            EmitRoleSafetyLog(role, i, "safe-no-water-column", "地图有泡泡但尚未形成水柱覆盖，判定为安全。");
            continue;
        }

        state = role.ResolveExplosionUnsafeState(
            snapshot.UnsafeLookup,
            snapshot.UnsafeAttackerLookup,
            snapshot.UnsafeEventLookup,
            snapshot.UnsafeEventListLookup
        );
        if (!state.IsUnsafe) {
            role.ResetExplosionUnsafeFrameCount();
            if (state.LeftUnsafe !== state.RightUnsafe) {
                safeReason = "单脚在非安全区（左脚" + (state.LeftUnsafe ? "非安全" : "安全")
                    + "@" + FormatMapNo(state.LeftMapNo)
                    + "，右脚" + (state.RightUnsafe ? "非安全" : "安全")
                    + "@" + FormatMapNo(state.RightMapNo)
                    + "），触发半身安全规则。";
                EmitRoleSafetyLog(role, i, "safe-half-" + state.LeftMapNo + "-" + state.RightMapNo, safeReason);
            }
            else {
                safeReason = "双脚都在安全区（左脚@" + FormatMapNo(state.LeftMapNo)
                    + "，右脚@" + FormatMapNo(state.RightMapNo) + "）。";
                EmitRoleSafetyLog(role, i, "safe-full-" + state.LeftMapNo + "-" + state.RightMapNo, safeReason);
            }
            continue;
        }

        role.ExplosionUnsafeFrameCount += 1;
        if (state.Attacker) {
            role.LastUnsafeExplosionAttacker = state.Attacker;
        }
        if (state.EventId) {
            role.LastUnsafeExplosionEventId = state.EventId;
        }
        role.LastUnsafeExplosionEventIds = state.EventIds ? state.EventIds.slice(0) : [];
        role.LastUnsafeExplosionMapNo = state.LeftMapNo;

        EmitRoleSafetyLog(
            role,
            i,
            "unsafe-" + role.ExplosionUnsafeFrameCount + "-" + state.LeftMapNo + "-" + state.RightMapNo,
            "双脚都在非安全区（左脚@" + FormatMapNo(state.LeftMapNo)
                + "，右脚@" + FormatMapNo(state.RightMapNo)
                + "），连续命中计数=" + role.ExplosionUnsafeFrameCount + "。"
        );

        if (role.ExplosionUnsafeFrameCount >= ExplosionHitRequiredFrames) {
            attacker = state.Attacker || role.LastUnsafeExplosionAttacker || null;
            attackerName = attacker ? GetRoleDebugName(attacker, RoleStorage.indexOf(attacker)) : "未知";
            PushExplosionSafetyDebugLog(
                GetRoleDebugName(role, i)
                + "：被炸到（连续" + role.ExplosionUnsafeFrameCount
                + "帧双脚处于非安全区，攻击者=" + attackerName + "）。"
            );
            role.ResetExplosionUnsafeFrameCount();
            role.Bomb(attacker, true);
        }
    }
}

function BindExplosionFrameHook() {
    var previousOnGameFrame;

    if (typeof window === "undefined") {
        return;
    }
    if (window.__bnbExplosionFrameHookBound) {
        return;
    }

    previousOnGameFrame = window.OnGameFrame;
    window.OnGameFrame = function(game) {
        if (typeof previousOnGameFrame === "function") {
            previousOnGameFrame(game);
        }
        TickExplosionUnsafeByFrame();
    };
    window.__bnbExplosionFrameHookBound = true;
    window.BNBExplosionUnsafeWindowMs = GetExplosionUnsafeWindowMs();
    PublishThreatSceneMeta();
    window.GetBNBExplosionEventSnapshot = function(now) {
        return BuildBNBExplosionEventSnapshot(typeof now === "number" ? now : Date.now());
    };
    window.BNBGetExplosionEventSnapshot = window.GetBNBExplosionEventSnapshot;
}

BindExplosionFrameHook();

//泡泡
var Paopao = function(role) {
    this.Master = role;
    this.BombId = ++PaopaoEventIdSeed;
    this.PaopaoStrong = role.PaopaoStrong;
    this.CurrentMapID = role.CurrentMapID();
    this.SpawnAt = Date.now();
    this.ExplodeAt = this.SpawnAt + GetPaopaoFuseMs();
    this.IsPlaced = false;

    if (townBarrierMap[this.CurrentMapID.Y][this.CurrentMapID.X] == 0) {
        townBarrierMap[this.CurrentMapID.Y][this.CurrentMapID.X] = 100;
        this.Object = new Bitmap(this.Master.RoleNumber == 1 ? GetCurrentPlayerPaopaoImage() : DefaultPaopaoImage);

        //初始化
        {
            this.Master.PaopaoCount++;
            this.Object.ZIndex = this.Master.Object.ZIndex - 1;

            //设置位置
            this.Object.Position = new Point(this.CurrentMapID.X * 40 + 20 - 2, this.CurrentMapID.Y * 40 + 40 - 5);

            //声音
            SystemSound.Play(SoundType.Appear);

            this.Object.Size = new Size(44, 41);

            var poponumber = 0;

            var t = this;
            var popoInterval = setInterval(function() {
                if (poponumber >= 3) {
                    poponumber = 0;
                }
                t.Object.StartPoint = new Point(poponumber * 44, 0);
                poponumber++;
            }, 200);

            //发生爆炸
            var popoTimeout = setTimeout(function() {
                t.Bomb();
            }, GetPaopaoFuseMs());
            
            if(!PaopaoArray[this.CurrentMapID.Y]){
                PaopaoArray[this.CurrentMapID.Y] = [];
            }
            //加入泡泡集合
            PaopaoArray[this.CurrentMapID.Y][this.CurrentMapID.X] = this;
            this.IsPlaced = true;
            TouchThreatSceneRevisionOnBubbleSpawn(this);
        }

        this.IsExploded = false;

        //泡泡爆炸
        this.Bomb = function() {
            if (this.IsExploded) {
                return;
            }
            this.IsExploded = true;
        
            clearInterval(popoInterval);
            this.Object.Dispose();
            clearTimeout(popoTimeout);
            PaopaoArray[this.CurrentMapID.Y][this.CurrentMapID.X] = null;
            townBarrierMap[this.CurrentMapID.Y][this.CurrentMapID.X] = 0;
            this.Master.PaopaoCount--;
            PopoBang(this.CurrentMapID, this.PaopaoStrong, this.Master, this);
        }
    }
}


//泡泡爆炸
function PopoBang(mapid, strong, role, sourceBomb){
    var explosionimage = resPrefix + "Pic/Explosion.png";
    var xymapidarray = FindPaopaoBombXY(mapid.X + mapid.Y * 15, strong);
    //X轴方向
    var xmaparray = xymapidarray.X;
    //Y轴方向
    var ymaparray = xymapidarray.Y;

    //泡泡位置
    var point = new Point(mapid.X * 40 + 20, mapid.Y * 40 + 40);
    SystemSound.Play(SoundType.Explode);
    
    var BombXUnits = [];
    for(var i = 0; i < xmaparray.length; i++){
        BombXUnits[i] = new Bitmap(explosionimage);
        BombXUnits[i].Size = new Size(40, 40);
        BombXUnits[i].ZIndex = 3;
        BombXUnits[i].Position = new Point((xmaparray[i] % 15) * 40 + 20, point.Y);
        
        //第一个
        if(i == 0 && xmaparray[i] < mapid){
            BombXUnits[i].StartPoint = new Point(200, 80);
        }
        //最后一个
        else if(i == xmaparray.length - 1 && xmaparray[i] > mapid){
            BombXUnits[i].StartPoint = new Point(200, 120);
        }
        //左边
        else if(xmaparray[i] < mapid){
            BombXUnits[i].StartPoint = new Point(120, 80);
        }
        //右边
        else{
            BombXUnits[i].StartPoint = new Point(120, 120);
        }
    }
    
    var BombYUnits = [];
    for(var i = 0; i < ymaparray.length; i++){
        BombYUnits[i] = new Bitmap(explosionimage);
        BombYUnits[i].Size = new Size(40, 40);
        BombYUnits[i].Position = new Point(point.X, parseInt(ymaparray[i] / 15, 10) * 40 + 40);
        BombYUnits[i].ZIndex = 3;
        
        //第一个
        if(i == 0 && ymaparray[i] < mapid){
            BombYUnits[i].StartPoint = new Point(200, 0);
        }
        //最后一个
        else if(i == ymaparray.length - 1 && ymaparray[i] > mapid){
            BombYUnits[i].StartPoint = new Point(200, 40);
        }
        //上边
        else if(ymaparray[i] < mapid){
            BombYUnits[i].StartPoint = new Point(120, 0);
        }
        //下边
        else{
            BombYUnits[i].StartPoint = new Point(120, 40);
        }
    }
    var bongbongCenter = new Bitmap(explosionimage);
    bongbongCenter.StartPoint = new Point(0, 160);
    bongbongCenter.Size = new Size(40, 40);
    bongbongCenter.Position = point;
    bongbongCenter.ZIndex = 3;

    //爆炸区域立即结算：先登记危险区窗口，再处理障碍破坏和连锁引爆
    var explosionZone = RegisterExplosionUnsafeZone(
        xmaparray.concat(ymaparray),
        mapid.Y * 15 + mapid.X,
        role,
        sourceBomb
    );
    ResolveExplosion(xmaparray.concat(ymaparray), mapid.Y * 15 + mapid.X);

    var bongbongpicnumber = ExplosionAnimStartFrame;
    var bongbongpiccenternumber = 1;
    var isExplosionDisposed = false;
    function DisposeExplosionSprites() {
        if (isExplosionDisposed) {
            return;
        }
        isExplosionDisposed = true;
        for (var xunit = 0; xunit < BombXUnits.length; xunit++) {
            if (BombXUnits[xunit]) {
                BombXUnits[xunit].Dispose();
            }
        }
        for (var yunit = 0; yunit < BombYUnits.length; yunit++) {
            if (BombYUnits[yunit]) {
                BombYUnits[yunit].Dispose();
            }
        }
        bongbongCenter.Dispose();
        ExpireExplosionUnsafeZone(explosionZone);
        clearInterval(bongbongInterval);
        clearTimeout(bongbongForceDisposeTimeout);
    }

    var bongbongInterval = setInterval(function() {
        if (bongbongpicnumber > ExplosionAnimEndFrame) {
            DisposeExplosionSprites();
        }
        else {
            if (bongbongpiccenternumber > 3) {
                bongbongpiccenternumber = 0;
            }
            
            for(var i = 0; i < xmaparray.length; i++){
                if(i == 0 || i == xmaparray.length - 1){
                    BombXUnits[i].StartPoint.X = 40 * bongbongpicnumber;
                }
            }
            for(var i = 0; i < ymaparray.length; i++){
                if(i == 0 || i == ymaparray.length - 1){
                    BombYUnits[i].StartPoint.X = 40 * bongbongpicnumber;
                }
            }
            bongbongCenter.StartPoint = new Point(bongbongpiccenternumber * 40, 160);
            bongbongpicnumber++;
            bongbongpiccenternumber++;
        }
    }, ExplosionAnimFrameIntervalMs);
    var bongbongForceDisposeTimeout = setTimeout(function() {
        DisposeExplosionSprites();
    }, ExplosionForceDisposeMs);
}

function ResolveExplosion(allmapidarray, centerMapId) {
    var chainBombs = [];

    allmapidarray.push(centerMapId);
    for (var i = 0; i < allmapidarray.length; i++) {
        var mapid = allmapidarray[i];
        var x = mapid % 15;
        var y = parseInt(mapid / 15, 10);
        var bomb = (PaopaoArray[y] && PaopaoArray[y][x]) ? PaopaoArray[y][x] : null;

        if (bomb && chainBombs.indexOf(bomb) === -1) {
            chainBombs.push(bomb);
        }

        Barrier.Bomb(x, y);
    }

    for (var j = 0; j < chainBombs.length; j++) {
        chainBombs[j].Bomb();
    }
}

//找出爆炸的MapID集合
function FindPaopaoBombXY(mapid, strong){
    var centerX = mapid % 15;
    var centerY = parseInt(mapid / 15, 10);
    var maxStrongCap = (typeof RoleConstant !== "undefined" && typeof RoleConstant.MaxPaopaoStrong === "number")
        ? RoleConstant.MaxPaopaoStrong
        : 10;
    var baseStrong = Math.min(parseInt(strong, 10) || 0, maxStrongCap);
    //X轴方向
    var xmaparray = [];
    //Y轴方向
    var ymaparray = [];

    function InMap(x, y) {
        return x >= 0 && y >= 0 && x < 15 && y < 13;
    }

    function GetBarrierNo(x, y) {
        return townBarrierMap[y][x];
    }

    function IsDestructibleBarrier(no) {
        return no === 3 || no === 8;
    }

    function IsSolidBarrier(no) {
        return no > 0 && no < 100 && !IsDestructibleBarrier(no);
    }

    function ScanDirection(dx, dy, targetArray) {
        // 每个方向单独计算边界限制，避免某一侧靠边影响其他方向的威力
        var edgeLimit;
        if (dx === 1) edgeLimit = 14 - centerX;
        else if (dx === -1) edgeLimit = centerX;
        else if (dy === 1) edgeLimit = 12 - centerY;
        else edgeLimit = centerY;
        var maxStrong = Math.min(baseStrong, edgeLimit);
        for (var step = 1; step <= maxStrong; step++) {
            var x = centerX + dx * step;
            var y = centerY + dy * step;
            var no;

            if (!InMap(x, y)) {
                break;
            }

            no = GetBarrierNo(x, y);

            // 不可穿透障碍，直接阻断（不包含该格）
            if (IsSolidBarrier(no)) {
                break;
            }

            // 可炸障碍，包含该格后阻断
            if (IsDestructibleBarrier(no)) {
                targetArray.push(y * 15 + x);
                break;
            }

            // 空地或道具可继续
            targetArray.push(y * 15 + x);
        }
    }

    ScanDirection(1, 0, xmaparray);   // Right
    ScanDirection(-1, 0, xmaparray);  // Left
    ScanDirection(0, 1, ymaparray);   // Down
    ScanDirection(0, -1, ymaparray);  // Up

    xmaparray.sort(function(a, b){
        return +(a) - +(b);
    });
    ymaparray.sort(function(a, b){
        return +(a) - +(b);
    });
    
    return {X: xmaparray, Y: ymaparray};
}
