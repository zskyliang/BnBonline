
var backgroundMusic;
var resPrefix = 'game/';
var gameRunning;
var singlePlayerState;
var roundTimerThread = null;
var respawnDelayMs = 2400;
var respawnInvincibleMs = 1000;
var roundDurationSeconds = 300;
var maxMoveStepInputId = "max-move-step-input";
var maxMoveStepApplyId = "apply-max-move-step";
var maxMoveStepHintId = "max-move-step-hint";
var maxMoveStepConfigBound = false;
var maxBubbleCountInputId = "max-bubble-count-input";
var maxBubbleCountApplyId = "apply-max-bubble-count";
var maxBubbleCountHintId = "max-bubble-count-hint";
var maxBubbleCountConfigBound = false;
var maxPowerInputId = "max-power-input";
var maxPowerApplyId = "apply-max-power";
var maxPowerHintId = "max-power-hint";
var maxPowerConfigBound = false;
var bubbleSkinSelectId = "bubble-skin-select";
var bubbleSkinHintId = "bubble-skin-hint";
var bubbleSkinStorageKey = "bnb_player_bubble_skin";
var bubbleSkinDefaultValue = "football";
var bubbleSkinConfigBound = false;
var mapSelectId = "map-select";
var mapSelectHintId = "map-select-hint";
var mapSelectConfigBound = false;
var aiEnemyCountSelectId = "ai-enemy-count-select";
var aiEnemyCountHintId = "ai-enemy-count-hint";
var aiEnemyCountStorageKey = "bnb_ai_enemy_count";
var aiEnemyCountDefaultValue = 3;
var aiEnemyCountConfigBound = false;
var bubbleSkinOptionList = [
    { value: "football", label: "足球" },
    { value: "basketball", label: "篮球" }
];
var gameFrameObserverDispatcherBound = false;
var gameFrameObservers = [];
var expertDuelFrameLogLimit = 72;

function EnsureGameFrameObserverDispatcher() {
    var previousOnGameFrame;
    if (gameFrameObserverDispatcherBound || typeof window === "undefined") {
        return;
    }
    previousOnGameFrame = window.OnGameFrame;
    window.OnGameFrame = function(game) {
        if (typeof previousOnGameFrame === "function") {
            previousOnGameFrame(game);
        }
        for (var i = 0; i < gameFrameObservers.length; i++) {
            try {
                gameFrameObservers[i](game);
            }
            catch (err) {
                // 观测回调异常不影响主循环
            }
        }
    };
    gameFrameObserverDispatcherBound = true;
}

function AddGameFrameObserver(observer) {
    if (typeof observer !== "function") {
        return;
    }
    EnsureGameFrameObserverDispatcher();
    if (gameFrameObservers.indexOf(observer) === -1) {
        gameFrameObservers.push(observer);
    }
}

function RemoveGameFrameObserver(observer) {
    var index = gameFrameObservers.indexOf(observer);
    if (index !== -1) {
        gameFrameObservers.splice(index, 1);
    }
}

function EnsureMaxMoveStepConfigDom() {
    var panel = document.getElementById("match-panel");
    var inputNode = document.getElementById(maxMoveStepInputId);
    var scoreList;
    var configBlock;
    var label;
    var row;
    var applyButton;
    var hintNode;

    if (!panel) {
        panel = document.createElement("aside");
        panel.id = "match-panel";
        panel.style.position = "fixed";
        panel.style.top = "16px";
        panel.style.right = "16px";
        panel.style.width = "220px";
        panel.style.padding = "12px";
        panel.style.borderRadius = "10px";
        panel.style.background = "rgba(8, 12, 22, 0.92)";
        panel.style.border = "1px solid rgba(255,255,255,0.15)";
        panel.style.color = "#e8f1ff";
        panel.style.zIndex = "9999";
        document.body.appendChild(panel);
    }

    panel.style.display = "block";

    if (!inputNode) {
        configBlock = document.createElement("div");
        configBlock.className = "config-block";

        label = document.createElement("label");
        label.className = "config-label";
        label.setAttribute("for", maxMoveStepInputId);
        label.textContent = "人物最大速度 (px/s)";

        row = document.createElement("div");
        row.className = "config-row";

        inputNode = document.createElement("input");
        inputNode.id = maxMoveStepInputId;
        inputNode.className = "config-input";
        inputNode.type = "number";
        inputNode.min = "50";
        inputNode.max = "1000";
        inputNode.step = "25";
        inputNode.value = RoleBalanceConfig.MaxSpeedPxPerSec;

        applyButton = document.createElement("button");
        applyButton.id = maxMoveStepApplyId;
        applyButton.className = "config-button";
        applyButton.type = "button";
        applyButton.textContent = "应用";

        hintNode = document.createElement("div");
        hintNode.id = maxMoveStepHintId;
        hintNode.className = "config-hint";
        hintNode.textContent = "当前上限：" + RoleBalanceConfig.MaxSpeedPxPerSec + " px/s";

        row.appendChild(inputNode);
        row.appendChild(applyButton);
        configBlock.appendChild(label);
        configBlock.appendChild(row);
        configBlock.appendChild(hintNode);

        scoreList = document.getElementById("score-list");
        if (scoreList && scoreList.parentNode === panel) {
            panel.insertBefore(configBlock, scoreList);
        }
        else {
            panel.appendChild(configBlock);
        }
    }
}

function EnsureMaxBubbleCountConfigDom() {
    var panel = document.getElementById("match-panel");
    var inputNode = document.getElementById(maxBubbleCountInputId);
    var scoreList;
    var configBlock;
    var label;
    var row;
    var applyButton;
    var hintNode;

    if (!panel) {
        return;
    }

    if (!inputNode) {
        configBlock = document.createElement("div");
        configBlock.className = "config-block";

        label = document.createElement("label");
        label.className = "config-label";
        label.setAttribute("for", maxBubbleCountInputId);
        label.textContent = "人物最大水泡数";

        row = document.createElement("div");
        row.className = "config-row";

        inputNode = document.createElement("input");
        inputNode.id = maxBubbleCountInputId;
        inputNode.className = "config-input";
        inputNode.type = "number";
        inputNode.min = String(RoleBalanceConfig.InitialBubbleCount);
        inputNode.max = "20";
        inputNode.step = "1";
        inputNode.value = RoleBalanceConfig.MaxBubbleCount;

        applyButton = document.createElement("button");
        applyButton.id = maxBubbleCountApplyId;
        applyButton.className = "config-button";
        applyButton.type = "button";
        applyButton.textContent = "应用";

        hintNode = document.createElement("div");
        hintNode.id = maxBubbleCountHintId;
        hintNode.className = "config-hint";
        hintNode.textContent = "当前上限：" + RoleBalanceConfig.MaxBubbleCount;

        row.appendChild(inputNode);
        row.appendChild(applyButton);
        configBlock.appendChild(label);
        configBlock.appendChild(row);
        configBlock.appendChild(hintNode);

        scoreList = document.getElementById("score-list");
        if (scoreList && scoreList.parentNode === panel) {
            panel.insertBefore(configBlock, scoreList);
        }
        else {
            panel.appendChild(configBlock);
        }
    }
}

function EnsureMaxPowerConfigDom() {
    var panel = document.getElementById("match-panel");
    var inputNode = document.getElementById(maxPowerInputId);
    var scoreList;
    var configBlock;
    var label;
    var row;
    var applyButton;
    var hintNode;

    if (!panel) {
        return;
    }

    if (!inputNode) {
        configBlock = document.createElement("div");
        configBlock.className = "config-block";

        label = document.createElement("label");
        label.className = "config-label";
        label.setAttribute("for", maxPowerInputId);
        label.textContent = "人物最大威力 (覆盖格)";

        row = document.createElement("div");
        row.className = "config-row";

        inputNode = document.createElement("input");
        inputNode.id = maxPowerInputId;
        inputNode.className = "config-input";
        inputNode.type = "number";
        inputNode.min = String(RoleBalanceConfig.InitialPower);
        inputNode.max = "20";
        inputNode.step = "1";
        inputNode.value = RoleBalanceConfig.MaxPower;

        applyButton = document.createElement("button");
        applyButton.id = maxPowerApplyId;
        applyButton.className = "config-button";
        applyButton.type = "button";
        applyButton.textContent = "应用";

        hintNode = document.createElement("div");
        hintNode.id = maxPowerHintId;
        hintNode.className = "config-hint";
        hintNode.textContent = "当前上限：" + RoleBalanceConfig.MaxPower + " 格";

        row.appendChild(inputNode);
        row.appendChild(applyButton);
        configBlock.appendChild(label);
        configBlock.appendChild(row);
        configBlock.appendChild(hintNode);

        scoreList = document.getElementById("score-list");
        if (scoreList && scoreList.parentNode === panel) {
            panel.insertBefore(configBlock, scoreList);
        }
        else {
            panel.appendChild(configBlock);
        }
    }
}

function EnsureBubbleSkinConfigDom() {
    var panel = document.getElementById("match-panel");
    var selectNode = document.getElementById(bubbleSkinSelectId);
    var scoreList;
    var configBlock;
    var label;
    var row;
    var hintNode;
    var i;
    var optionNode;

    if (!panel) {
        return;
    }

    if (!selectNode) {
        configBlock = document.createElement("div");
        configBlock.className = "config-block";

        label = document.createElement("label");
        label.className = "config-label";
        label.setAttribute("for", bubbleSkinSelectId);
        label.textContent = "玩家水泡皮肤";

        row = document.createElement("div");
        row.className = "config-row";

        selectNode = document.createElement("select");
        selectNode.id = bubbleSkinSelectId;
        selectNode.className = "config-input";

        for (i = 0; i < bubbleSkinOptionList.length; i++) {
            optionNode = document.createElement("option");
            optionNode.value = bubbleSkinOptionList[i].value;
            optionNode.textContent = bubbleSkinOptionList[i].label;
            selectNode.appendChild(optionNode);
        }

        hintNode = document.createElement("div");
        hintNode.id = bubbleSkinHintId;
        hintNode.className = "config-hint";
        hintNode.textContent = "当前皮肤：足球";

        row.appendChild(selectNode);
        configBlock.appendChild(label);
        configBlock.appendChild(row);
        configBlock.appendChild(hintNode);

        scoreList = document.getElementById("score-list");
        if (scoreList && scoreList.parentNode === panel) {
            panel.insertBefore(configBlock, scoreList);
        }
        else {
            panel.appendChild(configBlock);
        }
    }
}

function NormalizePlayerBubbleSkin(value) {
    if (value === "basketball") {
        return "basketball";
    }
    return "football";
}

function GetBubbleSkinLabel(value) {
    if (value === "basketball") {
        return "篮球";
    }
    return "足球";
}

function SetPlayerBubbleSkin(nextSkin) {
    var normalizedSkin = NormalizePlayerBubbleSkin(nextSkin);
    var selectNode = document.getElementById(bubbleSkinSelectId);
    var hintNode = document.getElementById(bubbleSkinHintId);

    window.PlayerBubbleSkin = normalizedSkin;

    try {
        localStorage.setItem(bubbleSkinStorageKey, normalizedSkin);
    } catch (e) {}

    if (selectNode) {
        selectNode.value = normalizedSkin;
    }
    if (hintNode) {
        hintNode.textContent = "当前皮肤：" + GetBubbleSkinLabel(normalizedSkin);
    }
}

function ReadStoredPlayerBubbleSkin() {
    try {
        return NormalizePlayerBubbleSkin(localStorage.getItem(bubbleSkinStorageKey) || bubbleSkinDefaultValue);
    } catch (e) {
        return bubbleSkinDefaultValue;
    }
}

function NormalizeMaxMoveStep(rawValue) {
    var maxSpeedPxPerSec = parseInt(rawValue, 10);

    if (isNaN(maxSpeedPxPerSec)) {
        maxSpeedPxPerSec = RoleBalanceConfig.MaxSpeedPxPerSec;
    }
    if (maxSpeedPxPerSec < RoleBalanceConfig.InitialSpeedPxPerSec) {
        maxSpeedPxPerSec = RoleBalanceConfig.InitialSpeedPxPerSec;
    }
    if (maxSpeedPxPerSec > 1000) {
        maxSpeedPxPerSec = 1000;
    }

    return maxSpeedPxPerSec;
}

function SyncRoleSpeedByMaxMoveStep() {
    for (var i = 0; i < RoleStorage.length; i++) {
        var role = RoleStorage[i];
        if (!role) {
            continue;
        }

        if (role.RawSpeed > RoleConstant.MaxMoveStep) {
            role.RawSpeed = RoleConstant.MaxMoveStep;
        }
        if (role.PreDeathMoveStep != null && role.PreDeathMoveStep > RoleConstant.MaxMoveStep) {
            role.PreDeathMoveStep = RoleConstant.MaxMoveStep;
        }
        if (!role.IsInPaopao && role.MoveStep > RoleConstant.MaxMoveStep) {
            role.MoveStep = RoleConstant.MaxMoveStep;
        }
    }
}

function SetRoleMaxMoveStep(nextMaxMoveStep) {
    var inputNode = document.getElementById(maxMoveStepInputId);
    var hintNode = document.getElementById(maxMoveStepHintId);
    var normalizedMaxSpeedPxPerSec = NormalizeMaxMoveStep(nextMaxMoveStep);
    var normalizedMaxMoveStep = SpeedPxPerSecToMoveStep(normalizedMaxSpeedPxPerSec);

    RoleBalanceConfig.MaxSpeedPxPerSec = normalizedMaxSpeedPxPerSec;
    RoleConstant.MaxMoveStep = normalizedMaxMoveStep;
    SyncRoleSpeedByMaxMoveStep();

    if (inputNode) {
        inputNode.value = normalizedMaxSpeedPxPerSec;
    }
    if (hintNode) {
        hintNode.textContent = "当前上限：" + normalizedMaxSpeedPxPerSec + " px/s";
    }
}

function InitRoleMaxMoveStepConfig() {
    EnsureMaxMoveStepConfigDom();

    var inputNode = document.getElementById(maxMoveStepInputId);
    var applyButton = document.getElementById(maxMoveStepApplyId);

    if (!inputNode || !applyButton) {
        return;
    }

    if (!maxMoveStepConfigBound) {
        applyButton.addEventListener("click", function () {
            SetRoleMaxMoveStep(inputNode.value);
        });

        inputNode.addEventListener("keydown", function (e) {
            var key = window.event ? e.keyCode : e.which;
            if (key === 13) {
                e.preventDefault();
                SetRoleMaxMoveStep(inputNode.value);
            }
        });

        maxMoveStepConfigBound = true;
    }

    SetRoleMaxMoveStep(RoleBalanceConfig.MaxSpeedPxPerSec);
}

function NormalizeMaxBubbleCount(rawValue) {
    var maxBubbleCount = parseInt(rawValue, 10);
    if (isNaN(maxBubbleCount)) {
        maxBubbleCount = RoleBalanceConfig.MaxBubbleCount;
    }
    if (maxBubbleCount < RoleBalanceConfig.InitialBubbleCount) {
        maxBubbleCount = RoleBalanceConfig.InitialBubbleCount;
    }
    if (maxBubbleCount > 20) {
        maxBubbleCount = 20;
    }
    return maxBubbleCount;
}

function SyncRoleBubbleCountByMax() {
    for (var i = 0; i < RoleStorage.length; i++) {
        var role = RoleStorage[i];
        if (!role) {
            continue;
        }
        role.CanPaopaoLength = ClampRoleBubbleCount(role.CanPaopaoLength);
    }
}

function SetRoleMaxBubbleCount(nextMaxBubbleCount) {
    var inputNode = document.getElementById(maxBubbleCountInputId);
    var hintNode = document.getElementById(maxBubbleCountHintId);
    var normalizedMaxBubbleCount = NormalizeMaxBubbleCount(nextMaxBubbleCount);

    RoleBalanceConfig.MaxBubbleCount = normalizedMaxBubbleCount;
    if (typeof MonsterMaxPaopaoLength === "number") {
        MonsterMaxPaopaoLength = normalizedMaxBubbleCount;
    }
    SyncRoleBubbleCountByMax();

    if (inputNode) {
        inputNode.value = normalizedMaxBubbleCount;
    }
    if (hintNode) {
        hintNode.textContent = "当前上限：" + normalizedMaxBubbleCount;
    }
}

function InitRoleMaxBubbleCountConfig() {
    EnsureMaxBubbleCountConfigDom();

    var inputNode = document.getElementById(maxBubbleCountInputId);
    var applyButton = document.getElementById(maxBubbleCountApplyId);

    if (!inputNode || !applyButton) {
        return;
    }

    if (!maxBubbleCountConfigBound) {
        applyButton.addEventListener("click", function () {
            SetRoleMaxBubbleCount(inputNode.value);
        });

        inputNode.addEventListener("keydown", function (e) {
            var key = window.event ? e.keyCode : e.which;
            if (key === 13) {
                e.preventDefault();
                SetRoleMaxBubbleCount(inputNode.value);
            }
        });

        maxBubbleCountConfigBound = true;
    }

    SetRoleMaxBubbleCount(RoleBalanceConfig.MaxBubbleCount);
}

function NormalizeMaxPower(rawValue) {
    var maxPower = parseInt(rawValue, 10);
    if (isNaN(maxPower)) {
        maxPower = RoleBalanceConfig.MaxPower;
    }
    if (maxPower < RoleBalanceConfig.InitialPower) {
        maxPower = RoleBalanceConfig.InitialPower;
    }
    if (maxPower > 20) {
        maxPower = 20;
    }
    return maxPower;
}

function SyncRolePowerByMax() {
    for (var i = 0; i < RoleStorage.length; i++) {
        var role = RoleStorage[i];
        if (!role) {
            continue;
        }
        role.PaopaoStrong = ClampRolePower(role.PaopaoStrong);
    }
}

function SetRoleMaxPower(nextMaxPower) {
    var inputNode = document.getElementById(maxPowerInputId);
    var hintNode = document.getElementById(maxPowerHintId);
    var normalizedMaxPower = NormalizeMaxPower(nextMaxPower);

    RoleBalanceConfig.MaxPower = normalizedMaxPower;
    RoleConstant.MaxPaopaoStrong = normalizedMaxPower;
    SyncRolePowerByMax();

    if (inputNode) {
        inputNode.value = normalizedMaxPower;
    }
    if (hintNode) {
        hintNode.textContent = "当前上限：" + normalizedMaxPower + " 格";
    }
}

function InitRoleMaxPowerConfig() {
    EnsureMaxPowerConfigDom();

    var inputNode = document.getElementById(maxPowerInputId);
    var applyButton = document.getElementById(maxPowerApplyId);

    if (!inputNode || !applyButton) {
        return;
    }

    if (!maxPowerConfigBound) {
        applyButton.addEventListener("click", function () {
            SetRoleMaxPower(inputNode.value);
        });

        inputNode.addEventListener("keydown", function (e) {
            var key = window.event ? e.keyCode : e.which;
            if (key === 13) {
                e.preventDefault();
                SetRoleMaxPower(inputNode.value);
            }
        });

        maxPowerConfigBound = true;
    }

    SetRoleMaxPower(RoleBalanceConfig.MaxPower);
}

function InitPlayerBubbleSkinConfig() {
    EnsureBubbleSkinConfigDom();

    var selectNode = document.getElementById(bubbleSkinSelectId);
    if (!selectNode) {
        return;
    }

    if (!bubbleSkinConfigBound) {
        selectNode.addEventListener("change", function () {
            SetPlayerBubbleSkin(selectNode.value);
        });
        bubbleSkinConfigBound = true;
    }

    SetPlayerBubbleSkin(ReadStoredPlayerBubbleSkin());
}

function NormalizeAIEnemyCount(rawValue) {
    var enemyCount = parseInt(rawValue, 10);
    if (isNaN(enemyCount)) {
        enemyCount = aiEnemyCountDefaultValue;
    }
    if (enemyCount < 0) {
        enemyCount = 0;
    }
    if (enemyCount > 4) {
        enemyCount = 4;
    }
    return enemyCount;
}

function ReadStoredAIEnemyCount() {
    try {
        return NormalizeAIEnemyCount(localStorage.getItem(aiEnemyCountStorageKey));
    } catch (e) {
        return aiEnemyCountDefaultValue;
    }
}

function BuildAIEnemyCountOptions(selectNode) {
    var i;
    var optionNode;
    if (!selectNode) {
        return;
    }
    selectNode.innerHTML = "";
    for (i = 0; i <= 4; i++) {
        optionNode = document.createElement("option");
        optionNode.value = String(i);
        optionNode.textContent = String(i);
        selectNode.appendChild(optionNode);
    }
}

function UpdateAIEnemyCountHint(enemyCount) {
    var hintNode = document.getElementById(aiEnemyCountHintId);
    if (hintNode) {
        hintNode.textContent = "当前敌人数：" + enemyCount;
    }
}

function SetAIEnemyCount(nextEnemyCount, shouldRestart) {
    var normalizedEnemyCount = NormalizeAIEnemyCount(nextEnemyCount);
    var selectNode = document.getElementById(aiEnemyCountSelectId);
    try {
        localStorage.setItem(aiEnemyCountStorageKey, String(normalizedEnemyCount));
    } catch (e) {}

    if (selectNode) {
        selectNode.value = String(normalizedEnemyCount);
    }
    UpdateAIEnemyCountHint(normalizedEnemyCount);

    if (shouldRestart) {
        window.location.reload();
    }
}

function EnsureAIEnemyCountConfigDom() {
    var panel = document.getElementById("match-panel");
    var selectNode = document.getElementById(aiEnemyCountSelectId);
    var scoreList;
    var configBlock;
    var label;
    var row;
    var hintNode;

    if (!panel) {
        return;
    }
    if (!selectNode) {
        configBlock = document.createElement("div");
        configBlock.className = "config-block";

        label = document.createElement("label");
        label.className = "config-label";
        label.setAttribute("for", aiEnemyCountSelectId);
        label.textContent = "AI敌人数";

        row = document.createElement("div");
        row.className = "config-row";

        selectNode = document.createElement("select");
        selectNode.id = aiEnemyCountSelectId;
        selectNode.className = "config-input";
        BuildAIEnemyCountOptions(selectNode);

        hintNode = document.createElement("div");
        hintNode.id = aiEnemyCountHintId;
        hintNode.className = "config-hint";
        hintNode.textContent = "当前敌人数：" + aiEnemyCountDefaultValue;

        row.appendChild(selectNode);
        configBlock.appendChild(label);
        configBlock.appendChild(row);
        configBlock.appendChild(hintNode);

        scoreList = document.getElementById("score-list");
        if (scoreList && scoreList.parentNode === panel) {
            panel.insertBefore(configBlock, scoreList);
        }
        else {
            panel.appendChild(configBlock);
        }
    }
}

function InitAIEnemyCountConfig() {
    var selectNode;

    EnsureAIEnemyCountConfigDom();
    selectNode = document.getElementById(aiEnemyCountSelectId);
    if (!selectNode) {
        return;
    }

    BuildAIEnemyCountOptions(selectNode);
    if (!aiEnemyCountConfigBound) {
        selectNode.addEventListener("change", function () {
            SetAIEnemyCount(selectNode.value, true);
        });
        aiEnemyCountConfigBound = true;
    }

    SetAIEnemyCount(ReadStoredAIEnemyCount(), false);
}

function EnsureMapSelectConfigDom() {
    var panel = document.getElementById("match-panel");
    var selectNode = document.getElementById(mapSelectId);
    var scoreList;
    var configBlock;
    var label;
    var row;
    var hintNode;

    if (!panel) {
        return;
    }

    if (!selectNode) {
        configBlock = document.createElement("div");
        configBlock.className = "config-block";

        label = document.createElement("label");
        label.className = "config-label";
        label.setAttribute("for", mapSelectId);
        label.textContent = "游戏地图";

        row = document.createElement("div");
        row.className = "config-row";

        selectNode = document.createElement("select");
        selectNode.id = mapSelectId;
        selectNode.className = "config-input";

        hintNode = document.createElement("div");
        hintNode.id = mapSelectHintId;
        hintNode.className = "config-hint";
        hintNode.textContent = "当前地图：当前地图（经典）";

        row.appendChild(selectNode);
        configBlock.appendChild(label);
        configBlock.appendChild(row);
        configBlock.appendChild(hintNode);

        scoreList = document.getElementById("score-list");
        if (scoreList && scoreList.parentNode === panel) {
            panel.insertBefore(configBlock, scoreList);
        }
        else {
            panel.appendChild(configBlock);
        }
    }
}

function BuildMapSelectOptions(selectNode) {
    var mapList = typeof GetGameMapOptionList === "function" ? GetGameMapOptionList() : [{ id: "classic", label: "当前地图（经典）" }];
    var i;
    var optionNode;

    if (!selectNode) {
        return;
    }

    selectNode.innerHTML = "";
    for (i = 0; i < mapList.length; i++) {
        optionNode = document.createElement("option");
        optionNode.value = mapList[i].id;
        optionNode.textContent = mapList[i].label;
        selectNode.appendChild(optionNode);
    }
}

function UpdateMapSelectHint(mapId) {
    var hintNode = document.getElementById(mapSelectHintId);
    var mapLabel = typeof GetCurrentGameMapLabel === "function" ? GetCurrentGameMapLabel() : mapId;

    if (hintNode) {
        hintNode.textContent = "当前地图：" + mapLabel;
    }
}

function ApplySelectedGameMap(mapId, shouldRestart) {
    var normalizedMapId = mapId;
    var selectNode = document.getElementById(mapSelectId);

    if (typeof SetCurrentGameMap === "function") {
        normalizedMapId = SetCurrentGameMap(mapId);
    }

    if (typeof SaveSelectedGameMap === "function") {
        SaveSelectedGameMap(normalizedMapId);
    }

    if (selectNode) {
        selectNode.value = normalizedMapId;
    }
    UpdateMapSelectHint(normalizedMapId);

    if (shouldRestart) {
        window.location.reload();
    }
}

function InitGameMapConfig() {
    var selectNode;
    var initialMapId;

    EnsureMapSelectConfigDom();
    selectNode = document.getElementById(mapSelectId);
    if (!selectNode) {
        return;
    }

    BuildMapSelectOptions(selectNode);

    if (!mapSelectConfigBound) {
        selectNode.addEventListener("change", function () {
            ApplySelectedGameMap(selectNode.value, true);
        });
        mapSelectConfigBound = true;
    }

    initialMapId = typeof GetStoredGameMapId === "function" ? GetStoredGameMapId() : selectNode.value;
    ApplySelectedGameMap(initialMapId, false);
}

function InitGame(){
    if (typeof ResetExplosionUnsafeZones === "function") {
        ResetExplosionUnsafeZones();
    }
    InitGameMapConfig();
    InitAIEnemyCountConfig();
    InitRoleMaxMoveStepConfig();
    InitRoleMaxBubbleCountConfig();
    InitRoleMaxPowerConfig();
    InitPlayerBubbleSkinConfig();

    if (typeof ReplaceTreeAndHouseWithBoxes === "function") {
        ReplaceTreeAndHouseWithBoxes();
    }
    if (typeof window !== "undefined" && window.BNBTrainingStripNonRigid && typeof StripNonRigidBarriersFromMap === "function") {
        StripNonRigidBarriersFromMap();
    }

    var game = new Game(800, 600);
    game.ExceptFPS = 60;
    //游戏背景
    var background = new Bitmap(resPrefix + 'Pic/BG.png');
    background.ZIndex = 0;

    //背景音乐
    SystemSound.Play(SoundType.Start, false);

    setTimeout(function () {
        backgroundMusic = SystemSound.Play(SoundType.Background, true);
    }, 300);

    //时间显示
    var timesecondsCount = 0;
    var timetenMinutes = new Bitmap(resPrefix + "Pic/Number.png");
    timetenMinutes.Size = new Size(12, 10);
    timetenMinutes.Position = new Point(708, 43);

    var timenumberMinutes = new Bitmap(resPrefix + "Pic/Number.png");
    timenumberMinutes.Size = new Size(12, 10);
    timenumberMinutes.Position = new Point(720, 43);

    var timetenSeconds = new Bitmap(resPrefix + "Pic/Number.png");
    timetenSeconds.Size = new Size(12, 10);
    timetenSeconds.Position = new Point(742, 43);

    var timenumberSeconds = new Bitmap(resPrefix + "Pic/Number.png");
    timenumberSeconds.Size = new Size(12, 10);
    timenumberSeconds.Position = new Point(754, 43);

    var timeControlInterval = new Thread(function() {
        timesecondsCount++;

        var minutestemp = parseInt(timesecondsCount / 60, 10) % 60;
        timetenMinutes.StartPoint = new Point(parseInt(minutestemp / 10, 10) * 12, 0);
        timenumberMinutes.StartPoint = new Point((minutestemp % 10) * 12, 0);

        var secondstemp = timesecondsCount % 60;
        timetenSeconds.StartPoint = new Point(parseInt(secondstemp / 10, 10) * 12, 0);
        timenumberSeconds.StartPoint = new Point((secondstemp % 10) * 12, 0);

    }, 1000).Start();


    //文字显示
    var fpsText = new Label("FPS: " + game.FPS);
    fpsText.Position = new Point(700, 20);
    fpsText.Color = "White";

    //显示FPS
    new Thread(function() {
        fpsText.Text = 'FPS: ' + game.FPS;
    }, 500).Start();

    //游戏开始
    game.Start();
    DrawGameBackground();
    if (typeof DrawMapDecorations === "function") {
        DrawMapDecorations();
    }
    DrawBarrierMap();
    gameRunning = true;
}

function CreateRole(number, x, y){
    //任务角色
    var role1 = new Role(number);
    if(number == 1){
        role1.AniSize = new Size(48, 64);
        role1.DieSize = new Size(48, 100);
        role1.Offset = new Size(0, 12);
        role1.RideSize = new Size(48, 56);
        role1.Object.Size = new Size(48, 64);
    }
    else{
        role1.Offset = new Size(0, 17);
        role1.RideSize = new Size(56, 60);
        role1.Offset = new Size(0, 17);
        role1.Object.Size = new Size(56, 67);
        role1.AniSize = new Size(56, 70);
        role1.DieSize = new Size(56, 98);
    }
    role1.SetMoveSpeedPxPerSec(RoleBalanceConfig.InitialSpeedPxPerSec);
    role1.PaopaoStrong = ClampRolePower(RoleBalanceConfig.InitialPower);
    role1.CanPaopaoLength = ClampRoleBubbleCount(RoleBalanceConfig.InitialBubbleCount);
    role1.SetToMap(x, y);

    role1.isKeyup = false;
    role1.currentKeyCode = 0;
    return role1;
}

var isKeyup = false;
var currentKeyCode = 0;
var controlKeys = {37: true, 38: true, 39: true, 40: true, 32: true};

function CreateUserEvent(role, socket){
    //按键事件
    document.addEventListener("keydown", RoleEvent, true);
    document.addEventListener("keyup", RoleEventEnd, true);

    function RoleEvent(e) {
        if (gameRunning) {
            var key = window.event ? e.keyCode : e.which;

            if (controlKeys[key]) {
                e.preventDefault();
                RoleKeyEvent(key, role);
            }
            else if (key === 49) {
                RoleKeyEvent(key, role);
            }
            if (socket) {
                socket.emit("KeyUp", key);
            }
        }
    }

    //KeyPress结束事件
    function RoleEventEnd(e) {
        if (gameRunning) {
            var key = window.event ? e.keyCode : e.which;
            if (controlKeys[key]) {
                e.preventDefault();
            }
            if (controlKeys[key] || key === 49) {
                RoleKeyEventEnd(key, role);
            }
            if (socket) {
                socket.emit("KeyDown", key);
            }
        }
    }
}

function StopGameWithMessage(message, soundType) {
    if (!gameRunning) {
        return;
    }

    gameRunning = false;

    if (backgroundMusic) {
        SystemSound.Stop(backgroundMusic);
    }

    if (soundType) {
        SystemSound.Play(soundType, false);
    }

    setTimeout(function () {
        alert(message);
    }, 150);
}

function FormatRoundTime(seconds) {
    var m = parseInt(seconds / 60, 10);
    var s = seconds % 60;
    return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
}

function BuildFighterList(player, monsters) {
    var fighters = [{ id: "player", name: "玩家", role: player, kills: 0, deaths: 0 }];

    for (var i = 0; i < monsters.length; i++) {
        fighters.push({
            id: "ai_" + (i + 1),
            name: "AI " + (i + 1),
            role: monsters[i].Role,
            kills: 0,
            deaths: 0
        });
    }

    return fighters;
}

function BuildExpertDuelFighterList(monsters) {
    var fighters = [];
    for (var i = 0; i < monsters.length; i++) {
        fighters.push({
            id: "expert_ai_" + (i + 1),
            name: "专家AI " + (i + 1),
            role: monsters[i].Role,
            kills: 0,
            deaths: 0
        });
    }
    return fighters;
}

function FindFarthestWalkablePointFrom(startPoint) {
    var bestPoint = null;
    var bestDistance = -1;
    var x;
    var y;
    var distance;
    var safeStart = startPoint || { X: 0, Y: 0 };
    var rows = townBarrierMap ? townBarrierMap.length : 0;

    for (y = 0; y < rows; y++) {
        if (!townBarrierMap[y]) {
            continue;
        }
        for (x = 0; x < townBarrierMap[y].length; x++) {
            if (townBarrierMap[y][x] !== 0) {
                continue;
            }
            distance = Math.abs(x - safeStart.X) + Math.abs(y - safeStart.Y);
            if (distance > bestDistance) {
                bestDistance = distance;
                bestPoint = { X: x, Y: y };
            }
        }
    }

    if (!bestPoint && typeof GetCurrentGameMapSpawn === "function") {
        bestPoint = GetCurrentGameMapSpawn();
    }
    return bestPoint || { X: 14, Y: 12 };
}

function FindMonsterByRole(role) {
    if (!singlePlayerState || !singlePlayerState.Monsters) {
        return null;
    }
    for (var i = 0; i < singlePlayerState.Monsters.length; i++) {
        if (singlePlayerState.Monsters[i].Role === role) {
            return singlePlayerState.Monsters[i];
        }
    }
    return null;
}

function FormatMapPointText(mapPoint) {
    if (!mapPoint) {
        return "--";
    }
    return mapPoint.X + "," + mapPoint.Y;
}

function BuildRoleFrameText(role, fighterName) {
    var map = role ? role.CurrentMapID() : null;
    var feet = role && typeof role.GetFootMapIDPair === "function" ? role.GetFootMapIDPair() : null;
    var monster = FindMonsterByRole(role);
    var stateTag = monster ? monster.State : "idle";
    return fighterName
        + " @(" + FormatMapPointText(map) + ")"
        + " foot(" + FormatMapPointText(feet ? feet.Left : null) + " | " + FormatMapPointText(feet ? feet.Right : null) + ")"
        + " unsafeFrames=" + (role ? (role.ExplosionUnsafeFrameCount || 0) : 0)
        + " trapped=" + (role ? !!role.IsInPaopao : false)
        + " state=" + stateTag;
}

function EnsureExpertDuelFramePanel() {
    var panel = document.getElementById("match-panel");
    var scoreList = document.getElementById("score-list");
    var block = document.getElementById("expert-duel-frame-block");

    if (!panel) {
        return null;
    }
    if (!block) {
        block = document.createElement("div");
        block.id = "expert-duel-frame-block";
        block.className = "config-block";
        block.style.display = "none";
        block.innerHTML = ""
            + "<div class=\"config-label\" style=\"font-weight:700;\">专家AI 1v1 逐帧观测</div>"
            + "<div id=\"expert-duel-frame-head\" class=\"config-hint\" style=\"color:#ffe38b;\"></div>"
            + "<pre id=\"expert-duel-frame-log\" style=\"margin:6px 0 0 0;padding:8px;border-radius:6px;background:rgba(0,0,0,0.26);color:#d8e6ff;font-size:11px;line-height:1.4;max-height:180px;overflow:auto;white-space:pre-wrap;\"></pre>";
        if (scoreList && scoreList.parentNode === panel) {
            panel.insertBefore(block, scoreList);
        }
        else {
            panel.appendChild(block);
        }
    }
    return block;
}

function SetExpertDuelFramePanelVisible(visible) {
    var block = EnsureExpertDuelFramePanel();
    if (!block) {
        return;
    }
    block.style.display = visible ? "block" : "none";
}

function ResetExpertDuelFrameViewState() {
    var headNode = document.getElementById("expert-duel-frame-head");
    var logNode = document.getElementById("expert-duel-frame-log");
    if (!singlePlayerState) {
        return;
    }
    singlePlayerState.FrameView = {
        frameCount: 0,
        startAt: Date.now(),
        lines: []
    };
    if (headNode) {
        headNode.textContent = "帧 #0";
    }
    if (logNode) {
        logNode.textContent = "";
    }
}

function RenderExpertDuelFrameView(game) {
    var state;
    var headNode;
    var logNode;
    var fighters;
    var line;
    var elapsedMs;

    if (!singlePlayerState || singlePlayerState.Mode !== "expert_duel_1v1") {
        return;
    }

    EnsureExpertDuelFramePanel();
    state = singlePlayerState.FrameView;
    if (!state) {
        ResetExpertDuelFrameViewState();
        state = singlePlayerState.FrameView;
    }

    state.frameCount += 1;
    elapsedMs = Date.now() - state.startAt;
    fighters = singlePlayerState.Fighters || [];

    if (fighters.length < 2) {
        return;
    }

    line = "#"
        + state.frameCount
        + " t=" + (elapsedMs / 1000).toFixed(2) + "s"
        + " fps=" + (game && typeof game.FPS === "number" ? game.FPS : 0)
        + "\n"
        + BuildRoleFrameText(fighters[0].role, fighters[0].name)
        + "\n"
        + BuildRoleFrameText(fighters[1].role, fighters[1].name);

    state.lines.push(line);
    if (state.lines.length > expertDuelFrameLogLimit) {
        state.lines = state.lines.slice(state.lines.length - expertDuelFrameLogLimit);
    }

    headNode = document.getElementById("expert-duel-frame-head");
    logNode = document.getElementById("expert-duel-frame-log");
    if (headNode) {
        headNode.textContent = "帧 #" + state.frameCount + "｜用时 " + (elapsedMs / 1000).toFixed(2) + "s";
    }
    if (logNode) {
        logNode.textContent = state.lines.join("\n\n");
        logNode.scrollTop = logNode.scrollHeight;
    }
}

function FindFighterByRole(role) {
    if (!singlePlayerState || !singlePlayerState.Fighters) {
        return null;
    }

    for (var i = 0; i < singlePlayerState.Fighters.length; i++) {
        if (singlePlayerState.Fighters[i].role === role) {
            return singlePlayerState.Fighters[i];
        }
    }

    return null;
}

function RenderMatchPanel() {
    var panel = document.getElementById("match-panel");
    var timerNode = document.getElementById("round-timer");
    var scoreList = document.getElementById("score-list");
    var fighters;

    if (!panel || !timerNode || !scoreList || !singlePlayerState) {
        return;
    }

    panel.style.display = "block";
    timerNode.textContent = FormatRoundTime(singlePlayerState.RemainingSeconds);

    fighters = singlePlayerState.Fighters.slice(0);
    fighters.sort(function(a, b) {
        return b.kills - a.kills;
    });

    scoreList.innerHTML = "";
    for (var i = 0; i < fighters.length; i++) {
        var item = document.createElement("li");
        item.className = "score-item";
        item.textContent = fighters[i].name + "  击败: " + fighters[i].kills;
        scoreList.appendChild(item);
    }
}

function EndRoundByTime() {
    var winner = null;

    if (!singlePlayerState || !gameRunning) {
        return;
    }

    gameRunning = false;
    if (roundTimerThread != null) {
        clearInterval(roundTimerThread);
        roundTimerThread = null;
    }

    singlePlayerState.Monsters.forEach(function(monster) {
        monster.Stop();
    });
    singlePlayerState.Fighters.forEach(function(unit) {
        unit.role.Stop();
    });

    singlePlayerState.Fighters.forEach(function(unit) {
        if (winner == null || unit.kills > winner.kills) {
            winner = unit;
        }
    });

    if (typeof AIEvolution !== "undefined" && typeof AIEvolution.finalizeMatch === "function") {
        AIEvolution.finalizeMatch();
    }

    RenderMatchPanel();
    setTimeout(function() {
        alert("时间到！本局最高击败：" + (winner ? winner.name + "（" + winner.kills + "）" : "无"));
    }, 120);
}

function FindRandomRespawnPoint(role) {
    var attempts = 0;

    while (attempts < 500) {
        var x = Math.floor(Math.random() * 15);
        var y = Math.floor(Math.random() * 13);
        var blocked = townBarrierMap[y][x] !== 0;

        if (!blocked) {
            for (var i = 0; i < RoleStorage.length; i++) {
                var other = RoleStorage[i];
                var m;
                if (!other || other === role || other.IsDeath) {
                    continue;
                }
                m = other.CurrentMapID();
                if (m && m.X === x && m.Y === y) {
                    blocked = true;
                    break;
                }
            }
        }

        if (!blocked) {
            return { X: x, Y: y };
        }
        attempts++;
    }

    if (typeof GetCurrentGameMapSpawn === "function") {
        return GetCurrentGameMapSpawn();
    }
    return { X: 0, Y: 0 };
}

function HandleRoleDeath(victim, attacker) {
    var victimFighter = FindFighterByRole(victim);
    var killerFighter = attacker ? FindFighterByRole(attacker) : null;
    var monsterUnit = null;
    var monsterKiller = null;

    if (victimFighter) {
        victimFighter.deaths++;
    }
    if (killerFighter && killerFighter !== victimFighter) {
        killerFighter.kills++;
    }

    // 通知 AI 进化系统
    if (typeof AIEvolution !== "undefined" && singlePlayerState) {
        var victimPos = victim.CurrentMapID();
        for (var mi = 0; mi < singlePlayerState.Monsters.length; mi++) {
            if (singlePlayerState.Monsters[mi].Role === victim) {
                monsterUnit = singlePlayerState.Monsters[mi];
            }
            if (attacker && singlePlayerState.Monsters[mi].Role === attacker) {
                monsterKiller = singlePlayerState.Monsters[mi];
            }
        }
        if (monsterUnit && victimPos) {
            monsterUnit.OnDeath(victimPos);
        }
        if (monsterKiller) {
            monsterKiller.OnKill();
        }
    }

    RenderMatchPanel();

    if (!gameRunning) {
        return;
    }

    monsterUnit = null;
    setTimeout(function() {
        var spawn;
        for (var i = 0; i < singlePlayerState.Monsters.length; i++) {
            if (singlePlayerState.Monsters[i].Role === victim) {
                monsterUnit = singlePlayerState.Monsters[i];
                break;
            }
        }
        if (!gameRunning) {
            return;
        }
        spawn = FindRandomRespawnPoint(victim);
        victim.RespawnAt(spawn.X, spawn.Y, respawnInvincibleMs);
        if (monsterUnit != null) {
            monsterUnit.Start();
        }
    }, respawnDelayMs);
}

function StartSinglePlayerGame(monsterCount) {
    var monsters;
    var player;
    var fighters;
    var playerSpawn = { X: 0, Y: 0 };
    var resolvedMonsterCount;

    if (typeof window !== "undefined" && typeof window.BNBMLRefreshConfig === "function") {
        window.BNBMLRefreshConfig();
    }

    if (typeof GetStoredGameMapId === "function" && typeof SetCurrentGameMap === "function") {
        SetCurrentGameMap(GetStoredGameMapId());
    }

    InitGame();
    if (typeof AIEvolution !== "undefined" && typeof AIEvolution.startMatch === "function") {
        AIEvolution.startMatch();
    }
    if (typeof GetCurrentGameMapSpawn === "function") {
        playerSpawn = GetCurrentGameMapSpawn();
    }
    player = CreateRole(1, playerSpawn.X, playerSpawn.Y);
    CreateUserEvent(player);

    resolvedMonsterCount = NormalizeAIEnemyCount(
        typeof monsterCount === "number" ? monsterCount : ReadStoredAIEnemyCount()
    );
    SetAIEnemyCount(resolvedMonsterCount, false);
    MonsterCount = resolvedMonsterCount;

    monsters = typeof StartMonsters === "function" ? StartMonsters(resolvedMonsterCount) : [];
    fighters = BuildFighterList(player, monsters);
    RemoveGameFrameObserver(RenderExpertDuelFrameView);
    SetExpertDuelFramePanelVisible(false);

    fighters.forEach(function(unit) {
        unit.role.OnDeath = function(victim, killer) {
            HandleRoleDeath(victim, killer);
        };
    });

    singlePlayerState = {
        Mode: "battle",
        Player: player,
        Monsters: monsters,
        Fighters: fighters,
        RemainingSeconds: roundDurationSeconds
    };

    if (roundTimerThread != null) {
        clearInterval(roundTimerThread);
    }
    roundTimerThread = setInterval(function() {
        if (!gameRunning || !singlePlayerState) {
            return;
        }
        singlePlayerState.RemainingSeconds--;
        if (singlePlayerState.RemainingSeconds <= 0) {
            singlePlayerState.RemainingSeconds = 0;
            EndRoundByTime();
        }
        RenderMatchPanel();
    }, 1000);

    RenderMatchPanel();

    return singlePlayerState;
}

function StartExpertDuelGame() {
    var aiA;
    var aiB;
    var roleA;
    var roleB;
    var fighters;
    var spawnA = { X: 0, Y: 0 };
    var spawnB = { X: 14, Y: 12 };

    if (typeof window !== "undefined" && typeof window.BNBMLRefreshConfig === "function") {
        window.BNBMLRefreshConfig();
    }

    if (typeof GetStoredGameMapId === "function" && typeof SetCurrentGameMap === "function") {
        SetCurrentGameMap(GetStoredGameMapId());
    }

    InitGame();
    if (typeof AIEvolution !== "undefined" && typeof AIEvolution.startMatch === "function") {
        AIEvolution.startMatch();
    }

    if (typeof GetCurrentGameMapSpawn === "function") {
        spawnA = GetCurrentGameMapSpawn();
    }
    spawnB = FindFarthestWalkablePointFrom(spawnA);

    roleA = CreateRole(1, spawnA.X, spawnA.Y);
    roleB = CreateRole(2, spawnB.X, spawnB.Y);

    aiA = new Monster(roleA);
    aiB = new Monster(roleB);
    aiA.Start();
    aiB.Start();

    fighters = BuildExpertDuelFighterList([aiA, aiB]);
    fighters.forEach(function(unit) {
        unit.role.OnDeath = function(victim, killer) {
            HandleRoleDeath(victim, killer);
        };
    });

    singlePlayerState = {
        Mode: "expert_duel_1v1",
        Player: roleA,
        Monsters: [aiA, aiB],
        Fighters: fighters,
        RemainingSeconds: roundDurationSeconds
    };

    if (roundTimerThread != null) {
        clearInterval(roundTimerThread);
    }
    roundTimerThread = setInterval(function() {
        if (!gameRunning || !singlePlayerState) {
            return;
        }
        singlePlayerState.RemainingSeconds--;
        if (singlePlayerState.RemainingSeconds <= 0) {
            singlePlayerState.RemainingSeconds = 0;
            EndRoundByTime();
        }
        RenderMatchPanel();
    }, 1000);

    SetExpertDuelFramePanelVisible(true);
    ResetExpertDuelFrameViewState();
    AddGameFrameObserver(RenderExpertDuelFrameView);
    RenderMatchPanel();

    return singlePlayerState;
}

function RoleKeyEventEnd(key, role) {
    if (key == role.currentKeyCode) {
        role.isKeyup = false;
        role.Stop();
    }
}

function RoleKeyEvent(key, role) {
    if(key in {38 : 'Up', 37 : 'Left', 40 : 'Down', 39 : 'Right'}){
        //另一个键按下后
        if (key != role.currentKeyCode) {
            role.isKeyup = false;
            role.Stop();
        }
        if (!role.isKeyup) {
            role.currentKeyCode = key;
            role.isKeyup = true;
            
            switch (key) {
                //上箭头
                case 38:
                    role.Move(Direction.Up);
                    break;
                //左箭头
                case 37:
                    role.Move(Direction.Left);
                    break;
                //下箭头
                case 40:
                    role.Move(Direction.Down);
                    break;
                //右箭头
                case 39:
                    role.Move(Direction.Right);
                    break;
            }
        }
    }
    //空格键
    else if(key == 32){
        role.PaoPao();
    }
    //数字1：被困泡时自救
    else if (key == 49 && role.RoleNumber == 1) {
        if (typeof role.TrySelfRescue === "function") {
            role.TrySelfRescue();
        }
    }
}
