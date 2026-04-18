var resPrefix = 'game/';
var MapColumnCount = 15;
var MapRowCount = 13;
var GameMapStorageKey = "bnb_selected_map";
var DefaultGameMapId = "classic";
var currentGameMapId = DefaultGameMapId;
var currentGameMapLabel = "当前地图（经典）";
var currentGroundMode = "town-ground";
var currentMapSpawn = { X: 0, Y: 0 };
var currentMapDecorations = [];
var WindmillBaseCellCode = 9;
var WindmillTopSolidPixels = 14;
var WindmillPassFrontRoleZIndex = 95;

function IsWindmillBaseCell(no) {
    return no === WindmillBaseCellCode;
}

function IsWindmillFloorPassByLocalY(localY) {
    return localY >= WindmillTopSolidPixels;
}

function IsMapBarrierBlockingByLocalY(no, localY) {
    if (no <= 0 || no > 100) {
        return false;
    }
    if (IsWindmillBaseCell(no)) {
        return !IsWindmillFloorPassByLocalY(localY);
    }
    // 其他障碍格默认整格阻挡
    return true;
}

function IsMapBarrierBlockingAtCanvasPoint(no, mapID, point) {
    if (!mapID || !point) {
        return no > 0 && no < 100;
    }
    return IsMapBarrierBlockingByLocalY(no, point.Y - (40 + mapID.Y * 40));
}

function IsMapBarrierBlockingAtRelativePoint(no, mapID, point) {
    if (!mapID || !point) {
        return no > 0 && no < 100;
    }
    return IsMapBarrierBlockingByLocalY(no, point.Y - mapID.Y * 40);
}

function IsWindmillFloorPassByCanvasPoint(mapID, point) {
    if (!mapID || !point || !townBarrierMap[mapID.Y]) {
        return false;
    }
    if (!IsWindmillBaseCell(townBarrierMap[mapID.Y][mapID.X])) {
        return false;
    }
    return IsWindmillFloorPassByLocalY(point.Y - (40 + mapID.Y * 40));
}

function GetWindmillFrontRoleZIndex() {
    return WindmillPassFrontRoleZIndex;
}

var ClassicGroundMapTemplate = [
    [1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1],
    [2, 2, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 2, 1],
    [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 2, 2, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1],
    [1, 2, 2, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1],
    [1, 2, 2, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1],
    [1, 2, 2, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 2],
    [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 2, 2, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1]
];

var ClassicBarrierMapTemplate = [
    [0, 3, 1, 5, 1, 7, 0, 7, 0, 7, 1, 4, 1, 4, 0],
    [0, 0, 3, 0, 0, 1, 0, 0, 0, 1, 2, 1, 2, 0, 0],
    [3, 5, 3, 5, 1, 7, 3, 7, 3, 7, 1, 4, 1, 4, 0],
    [0, 3, 2, 1, 7, 3, 3, 1, 3, 3, 7, 1, 2, 1, 2],
    [1, 7, 1, 7, 3, 3, 7, 0, 7, 3, 3, 7, 1, 7, 1],
    [2, 0, 3, 0, 0, 7, 1, 1, 1, 7, 0, 0, 3, 0, 2],
    [2, 7, 1, 7, 0, 2, 3, 3, 3, 2, 0, 7, 1, 7, 2],
    [2, 0, 3, 0, 0, 7, 1, 1, 1, 7, 0, 0, 3, 0, 2],
    [1, 7, 1, 7, 3, 3, 7, 2, 7, 3, 3, 7, 1, 7, 1],
    [2, 1, 2, 1, 7, 3, 3, 1, 3, 3, 7, 1, 2, 1, 2],
    [0, 4, 1, 4, 1, 7, 3, 7, 3, 7, 1, 6, 1, 6, 0],
    [0, 0, 2, 1, 2, 1, 0, 0, 0, 1, 2, 1, 2, 0, 0],
    [0, 4, 1, 4, 1, 7, 0, 7, 0, 7, 1, 6, 1, 6, 0]
];

function CloneMapMatrix(matrix) {
    var clone = [];
    for (var y = 0; y < matrix.length; y++) {
        clone[y] = matrix[y].slice(0);
    }
    return clone;
}

function CreateUniformGroundMap(unit) {
    var map = [];
    for (var y = 0; y < MapRowCount; y++) {
        map[y] = [];
        for (var x = 0; x < MapColumnCount; x++) {
            map[y][x] = unit;
        }
    }
    return map;
}

function CreateWindmillHeartBarrierMap() {
    var map = CreateUniformGroundMap(0);
    var y;
    var x;
    var heartShape = [
        "...............",
        "..#...###...#..",
        ".....#####.....",
        "....#.....#....",
        "..##.......##..",
        ".###.......###.",
        ".###.......###.",
        "..##.......##..",
        "..###.....###..",
        "....##...##....",
        ".....#####.....",
        "......###......",
        "..............."
    ];

    for (y = 0; y < MapRowCount; y++) {
        for (x = 0; x < MapColumnCount; x++) {
            if (heartShape[y].charAt(x) === "#") {
                map[y][x] = 3;
            }
        }
    }

    for (x = 0; x < MapColumnCount; x++) {
        map[0][x] = 8;
        map[MapRowCount - 1][x] = 8;
    }
    for (y = 0; y < MapRowCount; y++) {
        map[y][0] = 8;
        map[y][MapColumnCount - 1] = 8;
    }

    map[1][1] = 0;
    map[1][2] = 0;
    map[2][1] = 0;

    // 中央风车占地为不可穿越、不可炸开的刚体（9）
    // 风车整体下移一格后，底座刚体行也同步下移到 y=6。
    MarkSolidArea(map, 6, 6, 3, 1);

    return map;
}

function MarkSolidArea(map, startX, startY, width, height) {
    for (var y = startY; y < startY + height; y++) {
        for (var x = startX; x < startX + width; x++) {
            if (y >= 0 && y < MapRowCount && x >= 0 && x < MapColumnCount) {
                map[y][x] = 9;
            }
        }
    }
}

var GameMapDefinitions = {
    classic: {
        Id: "classic",
        Label: "当前地图（经典）",
        GroundMode: "town-ground",
        GroundMap: ClassicGroundMapTemplate,
        BarrierMap: ClassicBarrierMapTemplate,
        Spawn: { X: 0, Y: 0 },
        Decorations: []
    },
    "windmill-heart": {
        Id: "windmill-heart",
        Label: "风车爱心地图",
        GroundMode: "maptype2-first-tile",
        GroundMap: CreateUniformGroundMap(1),
        BarrierMap: CreateWindmillHeartBarrierMap(),
        Spawn: { X: 1, Y: 1 },
        Decorations: [
            { Type: "Windmill", MapX: 6, MapY: 3 }
        ]
    }
};

function NormalizeGameMapId(mapId) {
    return GameMapDefinitions[mapId] ? mapId : DefaultGameMapId;
}

function GetGameMapOptionList() {
    return [
        { id: "classic", label: GameMapDefinitions.classic.Label },
        { id: "windmill-heart", label: GameMapDefinitions["windmill-heart"].Label }
    ];
}

function GetCurrentGameMapId() {
    return currentGameMapId;
}

function GetCurrentGameMapLabel() {
    return currentGameMapLabel;
}

function GetCurrentGameMapSpawn() {
    return { X: currentMapSpawn.X, Y: currentMapSpawn.Y };
}

function GetStoredGameMapId() {
    try {
        return NormalizeGameMapId(localStorage.getItem(GameMapStorageKey) || DefaultGameMapId);
    } catch (e) {
        return DefaultGameMapId;
    }
}

function SaveSelectedGameMap(mapId) {
    var normalized = NormalizeGameMapId(mapId);
    try {
        localStorage.setItem(GameMapStorageKey, normalized);
    } catch (e) {}
}

//背景地图
var townGroundMap = [];
//障碍物地图
var townBarrierMap = [];

function SetCurrentGameMap(mapId) {
    var normalized = NormalizeGameMapId(mapId);
    var mapConfig = GameMapDefinitions[normalized];

    currentGameMapId = normalized;
    currentGameMapLabel = mapConfig.Label;
    currentGroundMode = mapConfig.GroundMode;
    currentMapSpawn = { X: mapConfig.Spawn.X, Y: mapConfig.Spawn.Y };
    currentMapDecorations = mapConfig.Decorations.slice(0);

    townGroundMap = CloneMapMatrix(mapConfig.GroundMap);
    townBarrierMap = CloneMapMatrix(mapConfig.BarrierMap);

    return currentGameMapId;
}

function CreateScaledBitmap(imageUrl, sourceX, sourceY, sourceW, sourceH, targetX, targetY, targetW, targetH, zIndex) {
    var sprite = {
        Type: "ScaledBitmap",
        image: new Image(),
        SourcePoint: new Point(sourceX, sourceY),
        SourceSize: new Size(sourceW, sourceH),
        Position: new Point(targetX, targetY),
        DrawSize: new Size(targetW, targetH),
        ZIndex: zIndex || 0,
        Visible: true,
        Click: null,
        Drag: null,
        EndPosition: function () {
            return new Point(this.Position.X + this.DrawSize.Width, this.Position.Y + this.DrawSize.Height);
        },
        Show: function (context) {
            context.drawImage(
                this.image,
                this.SourcePoint.X,
                this.SourcePoint.Y,
                this.SourceSize.Width,
                this.SourceSize.Height,
                this.Position.X,
                this.Position.Y,
                this.DrawSize.Width,
                this.DrawSize.Height
            );
        },
        Dispose: function () {
            this.Visible = false;
        }
    };

    Game.SpriteArray.push(sprite);
    Game.NeedLoadObjectsCount++;
    sprite.image.onload = function () {
        Game.LoadedObjectsCount++;
    };
    sprite.image.src = imageUrl;
    return sprite;
}

function DrawGameBackground() {
    var i;
    var j;

    if (currentGroundMode === "maptype2-first-tile") {
        for (i = 0; i < townGroundMap.length; i++) {
            for (j = 0; j < townGroundMap[i].length; j++) {
                CreateScaledBitmap(
                    resPrefix + "Pic/MapType2.png",
                    1,
                    1,
                    16,
                    16,
                    20 + 40 * j,
                    40 + 40 * i,
                    40,
                    40,
                    1
                );
            }
        }
        return;
    }

    for (i = 0; i < townGroundMap.length; i++) {
        for (j = 0; j < townGroundMap[i].length; j++) {
            var townGroundUnit = new Bitmap(resPrefix + "Pic/TownGround.png");
            townGroundUnit.Size = new Size(40, 40);
            townGroundUnit.StartPoint = new Point((townGroundMap[i][j] - 1) * 40, 0);
            townGroundUnit.Position = new Point(20 + 40 * j, 40 + 40 * i);
            townGroundUnit.ZIndex = 1;
        }
    }
}

function CreateWindmillDecoration(mapX, mapY) {
    var windmillX = 20 + 40 * mapX;
    // 风车头和底座整体下移一格
    var fanY = 40 + 40 * mapY;
    // 风车头使用左半部分完整高度，避免第三行被裁掉
    var fanHeight = 118;
    // 底座与风车头底部无缝贴合
    var baseY = fanY + fanHeight;
    // 层级：地板(1) < 底座(2) < 水泡/人物
    var baseZIndex = 2;
    var fanTopZIndex = 90;

    var base = new Bitmap(resPrefix + "Pic/TownWindmill.png");
    base.Size = new Size(120, 62);
    base.Position = new Point(windmillX, baseY);
    base.ZIndex = baseZIndex;

    // 风车头使用 TownWindmillAni 左半部分，并放到前景层；与底座做上下拼接（不重叠）。
    CreateScaledBitmap(
        resPrefix + "Pic/TownWindmillAni.png",
        0,
        0,
        120,
        fanHeight,
        windmillX,
        fanY,
        120,
        fanHeight,
        fanTopZIndex
    );
}

function DrawMapDecorations() {
    for (var i = 0; i < currentMapDecorations.length; i++) {
        var decoration = currentMapDecorations[i];
        if (decoration.Type === "Windmill") {
            CreateWindmillDecoration(decoration.MapX, decoration.MapY);
        }
    }
}

function ReplaceTreeAndHouseWithBoxes() {
    var boxReplacements = {4: true, 5: true, 6: true, 7: true};

    for (var y = 0; y < townBarrierMap.length; y++) {
        for (var x = 0; x < townBarrierMap[y].length; x++) {
            if (boxReplacements[townBarrierMap[y][x]]) {
                townBarrierMap[y][x] = 3;
            }
        }
    }
}

function IsRigidBarrierNo(no) {
    return no > 0 && no < 100 && no !== 3 && no !== 8;
}

function StripNonRigidBarriersFromMap() {
    for (var y = 0; y < townBarrierMap.length; y++) {
        for (var x = 0; x < townBarrierMap[y].length; x++) {
            if (!IsRigidBarrierNo(townBarrierMap[y][x])) {
                townBarrierMap[y][x] = 0;
            }
        }
    }
}

//所处的地图区块
function FindMapID(point) {
    point.X = point.X - 20;
    point.Y = point.Y - 40;
    //坐标范围
    if (point.X >= 0 && point.Y >= 0 && point.X < MapColumnCount * 40 && point.Y < MapRowCount * 40) {
        var xunitNumber = parseInt(point.X / 40, 10);
        var yunitNumber = parseInt(point.Y / 40, 10);

        return {X : xunitNumber, Y : yunitNumber};
    }
    return null;
}

SetCurrentGameMap(GetStoredGameMapId());
