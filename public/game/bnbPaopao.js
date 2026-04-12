var resPrefix = 'game/';

//泡泡
var PaopaoArray = [];

//泡泡
var Paopao = function(role) {
    this.Master = role;
    this.PaopaoStrong = role.PaopaoStrong;
    this.CurrentMapID = role.CurrentMapID();

    if (townBarrierMap[this.CurrentMapID.Y][this.CurrentMapID.X] == 0) {
        townBarrierMap[this.CurrentMapID.Y][this.CurrentMapID.X] = 100;
        this.Object = new Bitmap(resPrefix + "Pic/Popo.png");

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
            }, 3000);
            
            if(!PaopaoArray[this.CurrentMapID.Y]){
                PaopaoArray[this.CurrentMapID.Y] = [];
            }
            //加入泡泡集合
            PaopaoArray[this.CurrentMapID.Y][this.CurrentMapID.X] = this;
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
            PopoBang(this.CurrentMapID, this.PaopaoStrong, this.Master);
        }
    }
}


//泡泡爆炸
function PopoBang(mapid, strong, role){
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

    //爆炸区域立即结算：人物受伤、障碍破坏、连锁引爆同时发生
    ResolveExplosion(xmaparray.concat(ymaparray), mapid.Y * 15 + mapid.X, role);

    var bongbongpicnumber = 6;
    var bongbongpiccenternumber = 1;
    var bongbongInterval = setInterval(function() {
        if (bongbongpicnumber > 13) {
            for(var xunit in BombXUnits){
                BombXUnits[xunit].Dispose();
            }
            for(var yunit in BombYUnits){
                BombYUnits[yunit].Dispose();
            }
            bongbongCenter.Dispose();
            clearInterval(bongbongInterval);
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
    }, 50);
}

function ResolveExplosion(allmapidarray, centerMapId, attacker) {
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

        for (var m = 0; m < RoleStorage.length; m++) {
            var role = RoleStorage[m];
            if (!role.IsDeath && !role.IsBombImmune && role.IsExplosionHit(mapid)) {
                role.Bomb(attacker);
            }
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
    var baseStrong = Math.min(parseInt(strong, 10) || 0, 10);
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
            if (no > 3 && no < 100) {
                break;
            }

            // 可炸障碍，包含该格后阻断
            if (no > 0 && no <= 3) {
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
