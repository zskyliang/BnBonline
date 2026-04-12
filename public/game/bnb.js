
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
        label.textContent = "人物最大速度";

        row = document.createElement("div");
        row.className = "config-row";

        inputNode = document.createElement("input");
        inputNode.id = maxMoveStepInputId;
        inputNode.className = "config-input";
        inputNode.type = "number";
        inputNode.min = "2";
        inputNode.max = "12";
        inputNode.step = "1";
        inputNode.value = RoleConstant.MaxMoveStep;

        applyButton = document.createElement("button");
        applyButton.id = maxMoveStepApplyId;
        applyButton.className = "config-button";
        applyButton.type = "button";
        applyButton.textContent = "应用";

        hintNode = document.createElement("div");
        hintNode.id = maxMoveStepHintId;
        hintNode.className = "config-hint";
        hintNode.textContent = "当前上限：" + RoleConstant.MaxMoveStep;

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

function NormalizeMaxMoveStep(rawValue) {
    var maxMoveStep = parseInt(rawValue, 10);

    if (isNaN(maxMoveStep)) {
        maxMoveStep = RoleConstant.MaxMoveStep;
    }
    if (maxMoveStep < RoleConstant.MinMoveStep) {
        maxMoveStep = RoleConstant.MinMoveStep;
    }
    if (maxMoveStep > 12) {
        maxMoveStep = 12;
    }

    return maxMoveStep;
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
    var normalizedMaxMoveStep = NormalizeMaxMoveStep(nextMaxMoveStep);

    RoleConstant.MaxMoveStep = normalizedMaxMoveStep;
    SyncRoleSpeedByMaxMoveStep();

    if (inputNode) {
        inputNode.value = normalizedMaxMoveStep;
    }
    if (hintNode) {
        hintNode.textContent = "当前上限：" + normalizedMaxMoveStep;
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

    SetRoleMaxMoveStep(inputNode.value || RoleConstant.MaxMoveStep);
}

function InitGame(){
    InitRoleMaxMoveStepConfig();

    if (typeof ReplaceTreeAndHouseWithBoxes === "function") {
        ReplaceTreeAndHouseWithBoxes();
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
    role1.SetRawSpeed(4);
    role1.PaopaoStrong = 2;
    role1.CanPaopaoLength = 2;
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
            RoleKeyEventEnd(key, role);
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

    InitGame();
    player = CreateRole(1, 0, 0);
    CreateUserEvent(player);

    monsters = typeof StartMonsters === "function" ? StartMonsters(monsterCount || 3) : [];
    fighters = BuildFighterList(player, monsters);

    fighters.forEach(function(unit) {
        unit.role.OnDeath = function(victim, killer) {
            HandleRoleDeath(victim, killer);
        };
    });

    singlePlayerState = {
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
}
