var resPrefix = 'game/';

//物体移动方向枚举
var Direction = {
    Up: 0,
    Down: 1,
    Left: 2,
    Right: 3
}

var RoleMoveTickMs = 20;
var RoleMoveTicksPerSecond = 1000 / RoleMoveTickMs;

// 角色平衡配置：统一初始值、增量与上限（玩家与 AI 共用）
var RoleBalanceConfig = {
    InitialBubbleCount: 2,
    InitialSpeedPxPerSec: 150,
    InitialPower: 2,
    BubblePerItem: 1,
    SpeedPerItemPxPerSec: 25,
    PowerPerItem: 1,
    MaxBubbleCount: 8,
    MaxSpeedPxPerSec: 300,
    MaxPower: 10
};

function SpeedPxPerSecToMoveStep(speedPxPerSec) {
    var speed = parseFloat(speedPxPerSec);
    if (isNaN(speed) || speed < 0) {
        speed = 0;
    }
    return speed / RoleMoveTicksPerSecond;
}

function MoveStepToSpeedPxPerSec(moveStep) {
    var step = parseFloat(moveStep);
    if (isNaN(step) || step < 0) {
        step = 0;
    }
    return step * RoleMoveTicksPerSecond;
}

function ClampRoleBubbleCount(length) {
    var normalized = parseInt(length, 10);
    if (isNaN(normalized)) {
        normalized = RoleBalanceConfig.InitialBubbleCount;
    }
    if (normalized < 1) {
        normalized = 1;
    }
    if (normalized > RoleBalanceConfig.MaxBubbleCount) {
        normalized = RoleBalanceConfig.MaxBubbleCount;
    }
    return normalized;
}

function ClampRolePower(power) {
    var normalized = parseInt(power, 10);
    if (isNaN(normalized)) {
        normalized = RoleBalanceConfig.InitialPower;
    }
    if (normalized < 1) {
        normalized = 1;
    }
    if (normalized > RoleBalanceConfig.MaxPower) {
        normalized = RoleBalanceConfig.MaxPower;
    }
    return normalized;
}

function GetRoleInitialMoveStep() {
    return SpeedPxPerSecToMoveStep(RoleBalanceConfig.InitialSpeedPxPerSec);
}

//角色的属性值
var RoleConstant = {
    MinMoveStep: 2,
    //最大速度
    MaxMoveStep: SpeedPxPerSecToMoveStep(RoleBalanceConfig.MaxSpeedPxPerSec),

    //泡泡最大强度
    MaxPaopaoStrong: RoleBalanceConfig.MaxPower
}
var RolePushBoxHoldMs = 300;

var RoleStorage = [];

//角色对象
var Role = function(number) {
    this.GUID = "";
    this.RoleNumber = number;
    this.Object = new Bitmap(resPrefix + "Pic/Role" + number + ".png");

    RoleStorage.push(this);
    
    //初始层次
    this.Object.ZIndex = 3;

    //是否死亡
    this.IsDeath = false;

    //是否免疫泡泡爆炸
    this.IsBombImmune = false;

    //偏移
    this.Offset = new Size(0, 0);

    //原始偏移
    this.RawOffset = null;

    //行动方向，默认向下
    this.Direction = Direction.Down;

    //原始速度
    this.RawSpeed = 0;

    //移动速度
    this.MoveStep = GetRoleInitialMoveStep();

    //坐骑类型
    this.MoveHorse = MoveHorseObject.None;

    //是否可以踢泡泡
    this.IsCanMovePaopao = false;

    //连续可放泡泡次数
    this.CanPaopaoLength = ClampRoleBubbleCount(RoleBalanceConfig.InitialBubbleCount);

    //已经放还未爆炸的泡泡数
    this.PaopaoCount = 0;

    //泡泡爆炸强度
    this.PaopaoStrong = ClampRolePower(RoleBalanceConfig.InitialPower);

    //是否在泡泡中
    this.IsInPaopao = false;

    //坐骑被炸掉后的短暂无敌（毫秒时间戳）
    this.DismountProtectionUntil = 0;

    //复活无敌时间（毫秒时间戳）
    this.ExplosionImmuneUntil = 0;

    //死亡前速度（用于复活恢复）
    this.PreDeathMoveStep = null;

    //最近一次击杀者
    this.LastAttacker = null;

    //半身接触爆炸水柱的时间记录（毫秒时间戳），任意两次半身命中在窗口内即致死
    this.LastHalfHitTime = 0;

    //连续处于爆炸非安全区的帧计数（每帧检测）
    this.ExplosionUnsafeFrameCount = 0;

    //最近一次所在非安全区对应的攻击者
    this.LastUnsafeExplosionAttacker = null;

    //最近一次命中的爆炸事件ID（用于训练统计）
    this.LastUnsafeExplosionEventId = "";
    this.LastUnsafeExplosionEventIds = [];
    this.LastUnsafeExplosionMapNo = -1;

    //被炸回调（仅用于观察/训练，不参与规则判定）
    this.OnBombed = function() {};

    //困泡状态中的大泡泡对象
    this.TrapBubbleObject = null;

    //困泡状态动画、接触检测、死亡倒计时句柄
    this.TrapFloatInterval = 0;
    this.TrapTouchInterval = 0;
    this.TrapDieTimeout = 0;

    //推箱子蓄力：同一方向持续推动至少 RolePushBoxHoldMs 才会位移
    this.PushBoxHoldKey = "";
    this.PushBoxHoldStartAt = 0;

    this.ResetPushBoxHold = function() {
        this.PushBoxHoldKey = "";
        this.PushBoxHoldStartAt = 0;
    }

    //坐骑对象
    this.RideHorseObject = null;

    //坐骑时的大小
    this.RideSize = null;

    //角色原始大小
    this.RawSize = null;

    //AniSize
    this.AniSize = null;

    //Die Size
    this.DieSize = null;

    //设置初始速度
    this.SetRawSpeed = function(speed) {
        var limitedSpeed = speed;
        if (limitedSpeed > RoleConstant.MaxMoveStep) {
            limitedSpeed = RoleConstant.MaxMoveStep;
        }
        if (limitedSpeed < 0) {
            limitedSpeed = 0;
        }
        this.RawSpeed = limitedSpeed;
        this.MoveStep = limitedSpeed;
    }

    this.SetMoveSpeedPxPerSec = function(speedPxPerSec) {
        this.SetRawSpeed(SpeedPxPerSecToMoveStep(speedPxPerSec));
    }

    //角色坐标重新设置
    this.ResetPosition = function() {
        this.Object.Position.X = this.Object.Position.X - this.Offset.Width;
        this.Object.Position.Y = this.Object.Position.Y - this.Offset.Height;
        //console.log(this.Object.Position.X, this.Object.Position.Y);
    }

    //设置位置坐标，中心坐标，MAP中心内坐标
    this.SetPosition = function(x, y) {
        this.Object.Position = new Point(x + 20 - this.Object.Size.Width / 2 - this.Offset.Width, y + 40 - this.Object.Size.Height / 2 - this.Offset.Height);
    }

    //设置到Map区块
    this.SetToMap = function(x, y) {
        //获取MapID的中心坐标
        var mapx = x * 40 + 20;
        var mapy = y * 40 + 20;
        this.Object.Position = new Point(mapx + 20 - this.Object.Size.Width / 2 - this.Offset.Width, mapy + 40 - this.Object.Size.Height / 2 - this.Offset.Height);
        this.Object.ZIndex = (y + 2) * 2;
    }

    //中心坐标
    this.CenterPoint = function() {
        return new Point(this.Object.Position.X + this.Object.Size.Width / 2 + this.Offset.Width
                            , this.Object.Position.Y + this.Object.Size.Height / 2 + this.Offset.Height);
    }

    //地图的相对坐标
    this.MapPoint = function() {
        var cp = this.CenterPoint();
        return new Point(cp.X - 20, cp.Y - 40);
    }

    //获取当前的MapID
    this.CurrentMapID = function() {
        return FindMapID(this.CenterPoint());
    }

    var animateInterval = 0;
    var moveInterval = 0;

    //角色移动函数
    this.Move = function(directionnum) {
        if (directionnum < 0 || directionnum > 3) return;
        this.Direction = directionnum;
        if (this.RideHorseObject != null) {
            this.RideHorseObject.SetDirection(directionnum);
        }

        var t = this;
        var number = 0;

        if (!this.IsInPaopao) {
            //如果有坐骑
            if (this.MoveHorse != MoveHorseObject.None && this.RideHorseObject != null) {
                this.Object.StartPoint = new Point(this.Object.Size.Width * directionnum, 0);
            }
            else {
                this.Object.StartPoint = new Point(0, this.Object.Size.Height * directionnum);
                //动画线程
                animateInterval = setInterval(function() {
                    if (!t.IsInPaopao) {
                        if (t.MoveHorse != MoveHorseObject.None) {
                            t.Object.StartPoint = new Point(t.Object.Size.Width * directionnum, 0);
                            clearInterval(animateInterval);
                        }
                        else {
                            if (number >= 5) {
                                number = 0;
                            }
                            else {
                                number++;
                            }
                            t.Object.StartPoint = new Point(number * t.Object.Size.Width, t.Object.Size.Height * directionnum);
                        }
                    }
                    else {
                        clearInterval(animateInterval);
                    }
                }, 60);
            }
        }

        //移动线程
        moveInterval = setInterval(function() {
            var roleActualPoint = t.CenterPoint();
            switch (directionnum) {
                case Direction.Up:
                    if (t.IsCanPass(new Point(roleActualPoint.X, roleActualPoint.Y - t.MoveStep - 20))) {
                        t.Object.Position.Y -= t.MoveStep;
                        t.RoleOffset();
                        if (t.RideHorseObject != null) {
                            t.RideHorseObject.ResetPosition(t);
                        }
                    }
                    break;
                case Direction.Down:
                    if (t.IsCanPass(new Point(roleActualPoint.X, roleActualPoint.Y + t.MoveStep + 20))) {
                        t.Object.Position.Y += t.MoveStep;
                        t.RoleOffset();
                        if (t.RideHorseObject != null) {
                            t.RideHorseObject.ResetPosition(t);
                        }
                    }
                    break;
                case Direction.Left:
                    if (t.IsCanPass(new Point(roleActualPoint.X - t.MoveStep - 20, roleActualPoint.Y))) {
                        t.Object.Position.X -= t.MoveStep;
                        t.RoleOffset();
                        if (t.RideHorseObject != null) {
                            t.RideHorseObject.ResetPosition(t);
                        }
                    }
                    break;
                case Direction.Right:
                    if (t.IsCanPass(new Point(roleActualPoint.X + t.MoveStep + 20, roleActualPoint.Y))) {
                        t.Object.Position.X += t.MoveStep;
                        t.RoleOffset();
                        if (t.RideHorseObject != null) {
                            t.RideHorseObject.ResetPosition(t);
                        }
                    }
                    break;
            }
        }, RoleMoveTickMs);
    }
    
    //增加移动速度
    this.AddMoveStep = function(addNum) {
        this.MoveStep += addNum;
        if (this.MoveStep > RoleConstant.MaxMoveStep) {
            this.MoveStep = RoleConstant.MaxMoveStep;
        }
        if (this.MoveStep < 0) {
            this.MoveStep = 0;
        }
    }

    this.AddMoveSpeedPxPerSec = function(addPxPerSec) {
        this.AddMoveStep(SpeedPxPerSecToMoveStep(addPxPerSec));
    }

    this.AddPaopaoLength = function(addNum) {
        this.CanPaopaoLength = ClampRoleBubbleCount(this.CanPaopaoLength + addNum);
    }

    //增加泡泡强度
    this.AddPaopaoStrong = function(addNum) {
        this.PaopaoStrong = ClampRolePower(this.PaopaoStrong + addNum);
    }

    //下一个区块是否可以通过
    this.IsCanMoveNext = function(diretion) {
        var currentMapID = FindMapID(this.CenterPoint());
        var nextmapID = null;
        var currentNo;
        var currentPassable;
        switch (diretion) {
            case Direction.Up:
                nextmapID = currentMapID.Y - 1;
                break;
            case Direction.Down:
                nextmapID = currentMapID.Y + 1;
                break;
            case Direction.Left:
                nextmapID = currentMapID.X - 1;
                break;
            case Direction.Right:
                nextmapID = currentMapID.X + 1;
                break;
        }
        currentNo = townBarrierMap[currentMapID.Y][currentMapID.X];
        currentPassable = currentNo == 0 || currentNo > 100;
        if (!currentPassable && typeof IsWindmillFloorPassByCanvasPoint === "function") {
            currentPassable = IsWindmillFloorPassByCanvasPoint(currentMapID, this.CenterPoint());
        }
        return nextmapID != null && currentPassable;
    }

    //坐标所属区块是否可以通过
    this.IsCanPass = function(point) {
        //去掉边框的像素
        var nextmap = FindMapID(point);

        //坐标范围
        if (point.X >= 0 && point.Y >= 0 && point.X < 600 && point.Y < 520) {
            var currentMapID = this.CurrentMapID();
            var nextNo = townBarrierMap[nextmap.Y][nextmap.X];
            var mapBlocked;
            var shouldUseFrontZIndex = false;
            
            if(nextNo == 100 && currentMapID.X == nextmap.X && currentMapID.Y == nextmap.Y){
                return true;
            }

            var result = false;
            if (nextNo == 100) {
                // 只能穿过自己脚下的泡泡，其他泡泡格不可通过
                result = false;
            }
            else {
                mapBlocked = typeof IsMapBarrierBlockingAtCanvasPoint === "function"
                    ? IsMapBarrierBlockingAtCanvasPoint(nextNo, nextmap, point)
                    : (nextNo > 0 && nextNo < 100);

                if (this.MoveHorse == MoveHorseObject.UFO) {
                    //飞碟可以飞越能炸掉的障碍物
                    result = !mapBlocked || nextNo == 3 || nextNo == 8;
                }
                else {
                    result = !mapBlocked;
                }
            }

            // 推箱子：需要对同一箱子同一方向持续推动一段时间才会位移
            if (!result && this.MoveHorse != MoveHorseObject.UFO && townBarrierMap[nextmap.Y][nextmap.X] == 3) {
                var mapPoint = this.MapPoint();
                var aligned = true;
                var holdKey;
                var now;
                if (this.Direction == Direction.Up || this.Direction == Direction.Down) {
                    aligned = Math.abs((mapPoint.X % 40) - 20) <= 2;
                }
                else if (this.Direction == Direction.Left || this.Direction == Direction.Right) {
                    aligned = Math.abs((mapPoint.Y % 40) - 20) <= 2;
                }

                if (aligned) {
                    holdKey = nextmap.X + "_" + nextmap.Y + "_" + this.Direction;
                    now = Date.now();
                    if (this.PushBoxHoldKey != holdKey) {
                        this.PushBoxHoldKey = holdKey;
                        this.PushBoxHoldStartAt = now;
                    }
                    if (now - this.PushBoxHoldStartAt >= RolePushBoxHoldMs) {
                        result = Barrier.PushBox(nextmap.X, nextmap.Y, this.Direction);
                        if (result) {
                            this.ResetPushBoxHold();
                        }
                    }
                }
                else {
                    this.ResetPushBoxHold();
                }
            }
            else {
                this.ResetPushBoxHold();
            }

            if (result) {
                var zindex = nextmap.Y;
                //zindex += nextmap.X > 0 ? 1 : 0;
                this.Object.ZIndex = zindex * 2 + 2;
                if (this.MoveHorse == MoveHorseObject.UFO) {
                    this.Object.ZIndex += 3;
                }
                // 仍在泡泡所在格子时，角色必须始终显示在泡泡前面
                if (townBarrierMap[currentMapID.Y][currentMapID.X] == 100
                    && typeof PaopaoArray !== "undefined"
                    && PaopaoArray[currentMapID.Y]
                    && PaopaoArray[currentMapID.Y][currentMapID.X]
                    && PaopaoArray[currentMapID.Y][currentMapID.X].Object
                    && this.Object.ZIndex <= PaopaoArray[currentMapID.Y][currentMapID.X].Object.ZIndex) {
                    this.Object.ZIndex = PaopaoArray[currentMapID.Y][currentMapID.X].Object.ZIndex + 1;
                }

                if (typeof IsWindmillFloorPassByCanvasPoint === "function") {
                    shouldUseFrontZIndex = IsWindmillFloorPassByCanvasPoint(nextmap, point)
                        || IsWindmillFloorPassByCanvasPoint(currentMapID, this.CenterPoint());
                }
                if (shouldUseFrontZIndex) {
                    var frontZ = typeof GetWindmillFrontRoleZIndex === "function" ? GetWindmillFrontRoleZIndex() : 95;
                    if (this.Object.ZIndex < frontZ) {
                        this.Object.ZIndex = frontZ;
                    }
                }

                if (this.MoveHorse != MoveHorseObject.UFO) {
                    //捡宝物
                    if (townBarrierMap[currentMapID.Y][currentMapID.X] > 100) {
                        SystemSound.Play(SoundType.Get);
                        Barrier.Storage[currentMapID.Y][currentMapID.X].Object.Dispose();

                        //捡宝物后的属性
                        switch (townBarrierMap[currentMapID.Y][currentMapID.X]) {
                            //加泡泡次数                                         
                            case 101:
                                this.AddPaopaoLength(RoleBalanceConfig.BubblePerItem);
                                break;
                            //速度 +25px/s                                         
                            case 102:
                                this.AddMoveSpeedPxPerSec(RoleBalanceConfig.SpeedPerItemPxPerSec);
                                break;
                            //泡泡强度 +1 格                                         
                            case 103:
                                this.AddPaopaoStrong(RoleBalanceConfig.PowerPerItem);
                                break;
                            default:
                                break;
                        }
                        townBarrierMap[currentMapID.Y][currentMapID.X] = 0;
                    }
                }
            }
            return result;
        }
        return false;
    }

    //停止移动
    this.Stop = function() {
        clearInterval(animateInterval);
        clearInterval(moveInterval);
        this.ResetPushBoxHold();
        if (!this.IsInPaopao) {
            if (this.MoveHorse != MoveHorseObject.None) {
                this.Object.StartPoint = new Point(this.Object.Size.Width * this.Direction, 0);
                /*********************解决吃坐骑道具后方向问题*******************/
                //console.log(this.Object.StartPoint.X, this.Object.StartPoint.Y);
            }
            else {
                this.Object.StartPoint = new Point(0, this.Object.Size.Height * this.Direction);
            }
        }
    }

    //对象角色的偏移
    this.RoleOffset = function() {
        var mappoint = this.MapPoint();

        switch (this.Direction) {
            //向上,判断左右区块                                              
            case Direction.Up:
                this.CheckOffset(mappoint, 1, true);
                this.CheckOffset(mappoint, 2, true);
                break;
            case Direction.Down:
                this.CheckOffset(mappoint, 3, true);
                this.CheckOffset(mappoint, 4, true);
                break;
            case Direction.Left:
                this.CheckOffset(mappoint, 1, false);
                this.CheckOffset(mappoint, 3, false);
                break;
            case Direction.Right:
                this.CheckOffset(mappoint, 2, false);
                this.CheckOffset(mappoint, 4, false);
                break;
        }
    }

    //物体碰撞偏移
    this.CheckOffset = function(mappoint, direction, isxline) {
        var newPoint = new Point(mappoint.X, mappoint.Y);
        switch (direction) {
            //左上顶点                                              
            case 1:
                newPoint.X -= 20;
                newPoint.Y -= 20;
                break;
            //右上顶点                                              
            case 2:
                newPoint.X += 20;
                newPoint.Y -= 20;
                break;
            //左下顶点                                              
            case 3:
                newPoint.X -= 20;
                newPoint.Y += 20;
                break;
            //右下顶点                                              
            case 4:
                newPoint.X += 20;
                newPoint.Y += 20;
                break;
        }
        var lefttopmapID = GetMapIDByRelativePoint(newPoint.X, newPoint.Y);
        if (lefttopmapID != null) {
            var currentMapID = this.CurrentMapID();
            var unitNo = townBarrierMap[lefttopmapID.Y][lefttopmapID.X];
            var isCurrentBubbleTile = unitNo == 100
                && currentMapID != null
                && currentMapID.X == lefttopmapID.X
                && currentMapID.Y == lefttopmapID.Y;
            var isBlocking = false;

            if (unitNo == 100) {
                isBlocking = !isCurrentBubbleTile;
            }
            else if (typeof IsMapBarrierBlockingAtRelativePoint === "function") {
                isBlocking = IsMapBarrierBlockingAtRelativePoint(unitNo, lefttopmapID, newPoint);
            }
            else {
                isBlocking = unitNo > 0 && unitNo < 100;
            }

            if (!isBlocking) {
                return;
            }

            if (isxline) {
                var xunitNumber = parseInt(mappoint.X / 40, 10);
                this.SetPosition(xunitNumber * 40 + 20, mappoint.Y);
            }
            else {
                var yunitNumber = parseInt(mappoint.Y / 40, 10);
                this.SetPosition(mappoint.X, yunitNumber * 40 + 20);
            }

            if (this.MoveHorse != MoveHorseObject.None) {
                this.RideHorseObject.ResetPosition(this);
            }
        }
    }
}

//根据相对位置获取区块ID
function GetMapIDByRelativePoint(x, y) {
    if (x >= 0 && y >= 0 && x < 600 && y < 520) {
        var xunitNumber = parseInt(x / 40, 10);
        var yunitNumber = parseInt(y / 40, 10);

        return {X: xunitNumber, Y : yunitNumber};
    }
    return null;
}

//角色放泡泡
Role.prototype.PaoPao = function() {
    if(!this.IsDeath && !this.IsInPaopao){
        //判断是否还可以放
        if (this.CanPaopaoLength > this.PaopaoCount && !this.IsInPaopao && !this.IsDeath) {
            new Paopao(this);
        }
    }
}

//角色被炸到
Role.prototype.Bomb = function(attacker, forceTrap){
    if (this.DismountProtectionUntil > Date.now()) {
        return;
    }
    if (this.ExplosionImmuneUntil > Date.now()) {
        return;
    }

    if(!this.IsDeath && !this.IsInPaopao){
        if (attacker != null) {
            this.LastAttacker = attacker;
        }
        if (typeof this.OnBombed === "function") {
            this.OnBombed(this, attacker, !!forceTrap);
        }
        if(this.MoveHorse != MoveHorseObject.None){
            if (forceTrap) {
                this.OutRide(false);
                this.InPaoPao();
            }
            else {
                this.OutRide(true);
            }
        }
        else{
            this.InPaoPao();
        }
    }
}

Role.prototype.IsExplosionHit = function(mapid) {
    var roleMapID = this.CurrentMapID();
    if (!roleMapID) {
        return false;
    }
    return roleMapID.Y * 15 + roleMapID.X == mapid;
}

Role.prototype.GetFootMapIDPair = function() {
    var mapPoint = this.MapPoint();
    // 脚点采样略低于中心但不压在格子分界线上，避免横向移动时误判到下一行
    var footSampleYOffset = 16;
    var footSampleXOffset = 12;
    var leftFootMapID = GetMapIDByRelativePoint(mapPoint.X - footSampleXOffset, mapPoint.Y + footSampleYOffset);
    var rightFootMapID = GetMapIDByRelativePoint(mapPoint.X + footSampleXOffset, mapPoint.Y + footSampleYOffset);

    return {
        Left: leftFootMapID,
        Right: rightFootMapID
    };
}

Role.prototype.ResetExplosionUnsafeFrameCount = function() {
    this.ExplosionUnsafeFrameCount = 0;
    this.LastUnsafeExplosionAttacker = null;
    this.LastUnsafeExplosionEventId = "";
    this.LastUnsafeExplosionEventIds = [];
    this.LastUnsafeExplosionMapNo = -1;
}

Role.prototype.ResolveExplosionUnsafeState = function(unsafeMapLookup, unsafeAttackerLookup, unsafeEventLookup, unsafeEventListLookup) {
    var feet = this.GetFootMapIDPair();
    var leftMapNo = feet.Left ? feet.Left.Y * 15 + feet.Left.X : -1;
    var rightMapNo = feet.Right ? feet.Right.Y * 15 + feet.Right.X : -1;
    var leftUnsafe = leftMapNo >= 0 && !!unsafeMapLookup[leftMapNo];
    var rightUnsafe = rightMapNo >= 0 && !!unsafeMapLookup[rightMapNo];
    var attacker = null;
    var eventId = "";
    var eventIds = [];
    var leftEventIds = unsafeEventListLookup && unsafeEventListLookup[leftMapNo] ? unsafeEventListLookup[leftMapNo] : [];
    var rightEventIds = unsafeEventListLookup && unsafeEventListLookup[rightMapNo] ? unsafeEventListLookup[rightMapNo] : [];

    // 半身安全：只有单脚在非安全区时，整体视为安全
    if (!(leftUnsafe && rightUnsafe)) {
        return {
            IsUnsafe: false,
            Attacker: null,
            LeftUnsafe: leftUnsafe,
            RightUnsafe: rightUnsafe,
            LeftMapNo: leftMapNo,
            RightMapNo: rightMapNo,
            EventId: "",
            EventIds: []
        };
    }

    if (leftUnsafe && unsafeAttackerLookup[leftMapNo]) {
        attacker = unsafeAttackerLookup[leftMapNo];
    }
    if (!attacker && rightUnsafe && unsafeAttackerLookup[rightMapNo]) {
        attacker = unsafeAttackerLookup[rightMapNo];
    }

    if (leftUnsafe && unsafeEventLookup && unsafeEventLookup[leftMapNo]) {
        eventId = unsafeEventLookup[leftMapNo];
    }
    if (!eventId && rightUnsafe && unsafeEventLookup && unsafeEventLookup[rightMapNo]) {
        eventId = unsafeEventLookup[rightMapNo];
    }

    for (var i = 0; i < leftEventIds.length; i++) {
        if (eventIds.indexOf(leftEventIds[i]) === -1) {
            eventIds.push(leftEventIds[i]);
        }
    }
    for (i = 0; i < rightEventIds.length; i++) {
        if (eventIds.indexOf(rightEventIds[i]) === -1) {
            eventIds.push(rightEventIds[i]);
        }
    }

    return {
        IsUnsafe: true,
        Attacker: attacker,
        LeftUnsafe: leftUnsafe,
        RightUnsafe: rightUnsafe,
        LeftMapNo: leftMapNo,
        RightMapNo: rightMapNo,
        EventId: eventId,
        EventIds: eventIds
    };
}

Role.prototype.IsFriendlyWith = function(otherRole) {
    return otherRole != null && this.RoleNumber == otherRole.RoleNumber;
}

Role.prototype.IsTouchingRole = function(otherRole) {
    var selfCenter = this.CenterPoint();
    var otherCenter = otherRole.CenterPoint();
    return Math.abs(selfCenter.X - otherCenter.X) <= 24 && Math.abs(selfCenter.Y - otherCenter.Y) <= 24;
}

Role.prototype.ClearTrapStateTimers = function() {
    if (this.TrapFloatInterval) {
        clearInterval(this.TrapFloatInterval);
        this.TrapFloatInterval = 0;
    }
    if (this.TrapTouchInterval) {
        clearInterval(this.TrapTouchInterval);
        this.TrapTouchInterval = 0;
    }
    if (this.TrapDieTimeout) {
        clearTimeout(this.TrapDieTimeout);
        this.TrapDieTimeout = 0;
    }
}

Role.prototype.ReleaseFromPaoPao = function(rescuerRole) {
    var restoreMoveStep;

    if (!this.IsInPaopao || this.IsDeath) {
        return;
    }

    this.ClearTrapStateTimers();
    if (this.TrapBubbleObject != null) {
        this.TrapBubbleObject.Dispose();
        this.TrapBubbleObject = null;
    }

    this.IsInPaopao = false;
    restoreMoveStep = this.PreDeathMoveStep || this.RawSpeed;
    if (restoreMoveStep > RoleConstant.MaxMoveStep) {
        restoreMoveStep = RoleConstant.MaxMoveStep;
    }
    this.MoveStep = restoreMoveStep;

    this.Object.SetImage(resPrefix + "Pic/Role" + this.RoleNumber + ".png");
    if (this.RoleNumber == 1) {
        this.Object.Size = new Size(48, 64);
        this.Offset = new Size(0, 12);
    }
    else {
        this.Object.Size = new Size(56, 67);
        this.Offset = new Size(0, 17);
    }
    this.Object.StartPoint = new Point(0, this.Object.Size.Height * this.Direction);

    if (rescuerRole != null) {
        SystemSound.Play(SoundType.Save, false);
    }
}

Role.prototype.TrySelfRescue = function() {
    if (this.IsDeath || !this.IsInPaopao) {
        return false;
    }
    this.LastAttacker = null;
    this.ReleaseFromPaoPao(this);
    return true;
}

Role.prototype.ResolveTrapTouchResult = function() {
    for (var i = 0; i < RoleStorage.length; i++) {
        var otherRole = RoleStorage[i];
        var trapBubble;

        if (!otherRole || otherRole === this || otherRole.IsDeath || otherRole.IsInPaopao) {
            continue;
        }
        if (!this.IsTouchingRole(otherRole)) {
            continue;
        }

        if (this.IsFriendlyWith(otherRole)) {
            this.ReleaseFromPaoPao(otherRole);
        }
        else {
            this.LastAttacker = otherRole;
            this.ClearTrapStateTimers();
            trapBubble = this.TrapBubbleObject;
            this.TrapBubbleObject = null;
            this.Die(trapBubble);
        }
        return;
    }
}

//进入了泡泡
Role.prototype.InPaoPao = function() {
    if(!this.IsInPaopao){
        this.ClearTrapStateTimers();
        this.PreDeathMoveStep = this.MoveStep;
        this.MoveStep = 0.1;
        this.IsInPaopao = true;

        this.Object.SetImage(resPrefix + "Pic/Role" + this.RoleNumber + "Ani.png");
        this.Object.StartPoint.Y = 0;
        this.Object.Size = this.AniSize;

        var paopaoimage = resPrefix + "Pic/BigPopo.png";
        var bigPaopao = new Bitmap(paopaoimage);
        this.TrapBubbleObject = bigPaopao;
        bigPaopao.Size = new Size(72, 72);
        var centerpoint = this.CenterPoint();
        bigPaopao.Position = new Point(centerpoint.X - bigPaopao.Size.Width / 2, centerpoint.Y - bigPaopao.Size.Height / 2 - this.Offset.Height);
        bigPaopao.ZIndex = this.Object.ZIndex + 1;

        var picnumber = 0;
        var t = this;
        this.TrapFloatInterval = setInterval(function() {
            if (picnumber < 3) {
                picnumber++;
                bigPaopao.StartPoint = new Point(72 * picnumber, 0);
            }
            centerpoint = t.CenterPoint();
            if (t.Object.StartPoint.X == 0) {
                t.Object.StartPoint.X = t.Object.Size.Width;
            }
            else {
                t.Object.StartPoint.X = 0;
            }
            bigPaopao.Position = new Point(centerpoint.X - bigPaopao.Size.Width / 2, centerpoint.Y - bigPaopao.Size.Height / 2 - t.Offset.Height);
            bigPaopao.ZIndex = t.Object.ZIndex + 1;
        }, 100);

        this.TrapTouchInterval = setInterval(function() {
            if (!t.IsDeath && t.IsInPaopao) {
                t.ResolveTrapTouchResult();
            }
        }, 50);

        //死亡倒计时
        this.TrapDieTimeout = setTimeout(function() {
            t.ClearTrapStateTimers();
            t.TrapBubbleObject = null;
            t.Die(bigPaopao);
        }, 3000);
    }
}

//角色死亡
Role.prototype.Die = function (bigPaopao) {
    this.ClearTrapStateTimers();
    this.TrapBubbleObject = null;
    this.Object.SetImage(resPrefix + "Pic/Role" + this.RoleNumber + "Die.png");
    this.Object.Size = this.DieSize;

    var dienumber = 0;
    var t = this;
    var dieinterval = setInterval(function () {
        if (dienumber < 11) {
            t.Object.StartPoint.X = t.Object.Size.Width * dienumber;
            if (bigPaopao != null) {
                if (dienumber + 3 < 8) {
                    bigPaopao.StartPoint.X = 72 * (dienumber + 3);
                }
                else {
                    bigPaopao.Dispose();
                }
            }
            dienumber++;
        }
        else {
            clearInterval(dieinterval);
            t.Object.Dispose();
            t.Stop();
            t = null;
        }
    }, 200);
    if (this.RoleNumber == 1) {
        SystemSound.Stop(backgroundMusic);
        SystemSound.Play(SoundType.Die, false);
    }
    this.IsDeath = true;
    this.OnDeath(this, this.LastAttacker);
}
// 死亡时回调
Role.prototype.OnDeath = function () {

}

//角色骑上坐骑
Role.prototype.Ride = function() {
    if (!this.IsDeath && !this.IsInPaopao && this.MoveHorse != MoveHorseObject.None) {
        if(this.RawSize == null){
            this.RawSize = new Size(this.Object.Size.Width, this.Object.Size.Height);
        }
        if(this.RawOffset == null){
            this.RawOffset = new Size(this.Offset.Width, this.Offset.Height);
        }
        this.Object.Size = this.RideSize;
        if (this.RideHorseObject == null) {
            this.RideHorseObject = new RideHorse(this, this.MoveHorse);
            this.RideHorseObject.RoleOffset = this.Offset;
        }
        else {
            this.RideHorseObject.SetRideType(this.MoveHorse);
        }
        this.Object.SetImage(resPrefix + "Pic/Role" + this.RoleNumber + "Ride.png");
        this.RideHorseObject.SetDirection(this.Direction);
        switch (this.MoveHorse) {
            case MoveHorseObject.Owl:
                this.Offset.Height = this.MoveHorse.Size.Height - 10;
                break;
            case MoveHorseObject.Turtle:
                this.Offset.Height = this.MoveHorse.Size.Height;
                break;
            case MoveHorseObject.UFO:
                this.Offset.Height = this.MoveHorse.Size.Height;
                break;
        }
        //this.ResetPosition();
        this.RideHorseObject.ResetPosition(this);
    }
}

//坐骑被炸死
Role.prototype.OutRide = function(isFromBomb){
    if(this.MoveHorse !=  MoveHorseObject.None){
        this.Object.Size = new Size(this.RawSize.Width, this.RawSize.Height);
        this.Offset = new Size(this.RawOffset.Width, this.RawOffset.Height);
        this.MoveHorse =  MoveHorseObject.None;
        this.MoveStep = this.RawSpeed > RoleConstant.MaxMoveStep ? RoleConstant.MaxMoveStep : this.RawSpeed;
        this.Object.SetImage(resPrefix + "Pic/Role" + this.RoleNumber + ".png");
        if (this.RideHorseObject != null) {
            this.RideHorseObject.Die();
            this.RideHorseObject = null;
        }
        if (isFromBomb) {
            this.DismountProtectionUntil = Date.now() + 700;
        }
    }
}

Role.prototype.RespawnAt = function(x, y, invincibleMs) {
    var restoreMoveStep = this.PreDeathMoveStep || this.RawSpeed;
    if (restoreMoveStep > RoleConstant.MaxMoveStep) {
        restoreMoveStep = RoleConstant.MaxMoveStep;
    }

    this.Stop();
    clearInterval(this.movetoInterval);
    this.ClearTrapStateTimers();
    if (this.TrapBubbleObject != null) {
        this.TrapBubbleObject.Dispose();
        this.TrapBubbleObject = null;
    }

    if (this.RideHorseObject != null) {
        this.RideHorseObject.Die();
        this.RideHorseObject = null;
    }

    this.IsDeath = false;
    this.IsInPaopao = false;
    this.LastAttacker = null;
    this.DismountProtectionUntil = 0;
    this.ExplosionImmuneUntil = Date.now() + (invincibleMs || 0);
    this.LastHalfHitTime = 0;
    this.ResetExplosionUnsafeFrameCount();

    this.Object.Visible = true;
    if (Game.SpriteArray.indexOf(this.Object) === -1) {
        Game.SpriteArray.push(this.Object);
    }
    this.Object.SetImage(resPrefix + "Pic/Role" + this.RoleNumber + ".png");

    if (this.RoleNumber == 1) {
        this.Object.Size = new Size(48, 64);
        this.Offset = new Size(0, 12);
    }
    else {
        this.Object.Size = new Size(56, 67);
        this.Offset = new Size(0, 17);
    }

    this.MoveHorse = MoveHorseObject.None;
    this.MoveStep = restoreMoveStep;
    this.Direction = Direction.Down;
    this.Object.StartPoint = new Point(0, this.Object.Size.Height * this.Direction);
    this.SetToMap(x, y);
}

this.movetoInterval = 0;

//去任意点
Role.prototype.MoveTo = function(x, y) {
    this.Stop();
    clearInterval(this.movetoInterval);
    
    var astar = new Astar(townBarrierMap);
    var current = this.CurrentMapID();
    var stallTicks = 0;
    var lastMapKey;
    var directionTemp;
    if (!current) {
        return false;
    }
    var paths = astar.getPath(current.Y, current.X, y, x);
    //console.log("Start:(%s, %s)  End:(%s, %s)", current.X, current.Y, x, y)
    //console.log(paths);
    
    if(paths.length > 0){
        if (paths.length <= 1) {
            return true;
        }
        var t = this;
        var currentnum = 0;
        var movedone = true;
        var direction;
        lastMapKey = current.X + "_" + current.Y;
        this.movetoInterval = setInterval(function(){
            if(movedone){
                currentnum++;
            }
            if(currentnum < paths.length){
                var currentxy = t.CurrentMapID();
                var currentMapKey;
                if (!currentxy) {
                    clearInterval(t.movetoInterval);
                    return;
                }
                currentMapKey = currentxy.X + "_" + currentxy.Y;
                if (currentMapKey === lastMapKey) {
                    stallTicks++;
                }
                else {
                    lastMapKey = currentMapKey;
                    stallTicks = 0;
                }
                if (stallTicks > 90) {
                    // 卡在原地，结束当前路径，允许上层重新规划
                    t.Stop();
                    clearInterval(t.movetoInterval);
                    return;
                }

                directionTemp = GetDirection(currentxy.X, currentxy.Y, paths[currentnum]);
                
                if(movedone){
                    movedone = false;
                    direction = directionTemp;
                    if (direction == null) {
                        movedone = true;
                        return;
                    }
                    //console.log("Start:(%s, %s)  End:(%s, %s)", currentxy.X, currentxy.Y, paths[currentnum][1], paths[currentnum][0])
                    t.Move(direction);
                }
                else{
                    //console.log(currentxy.X, currentxy.Y,paths[currentnum][1], paths[currentnum][0])
                    var maprelativepoint = t.MapPoint();
                    if(currentxy.X == paths[currentnum][1] && currentxy.Y == paths[currentnum][0] 
                        && maprelativepoint.X % 40 > 0 && maprelativepoint.X % 40 < 40
                        && maprelativepoint.Y % 40 > 0 && maprelativepoint.Y % 40 < 40){
                        movedone = true;
                        t.Stop();
                    }
                }
            }
            else{
                clearInterval(t.movetoInterval);
            }
        }, 10);
        return true;
    }
    return false;
}

//获取相对位置的方向
function GetDirection(x, y, pathxy){
    //console.log(x, y, pathxy);
    var direct;
    //0是y, 1是x
    if(pathxy[1] - x > 0){
        direct = Direction.Right;
    }
    else if(pathxy[1] - x < 0){
        direct = Direction.Left
    }
    else if(pathxy[0] - y > 0){
        direct = Direction.Down;
    }
    else if(pathxy[0] - y < 0){
        direct = Direction.Up;
    }
    return direct;
}

//获取地图点的相对坐标
function GetMapPointXY(mapid){
    return {X : (mapid % 15), Y : parseInt(mapid / 15, 10) };
}
