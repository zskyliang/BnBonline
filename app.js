var express = require('express'),
    bodyParser = require('body-parser');
var fs = require('fs');
var path = require('path');
var app = express();
var server = require('http').Server(app);
var io = require('socket.io')(server);

var swig = require('swig');

app.engine('html', swig.renderFile);

app.set('view engine', 'html');
app.set('views', __dirname + '/templates');
app.use(express.static('public'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({     // to support URL-encoded bodies
    extended: true
}));

var rooms = {};
var TRAINING_OUTPUT_DIR = path.join(__dirname, 'output', 'web-game');
var TRAINING_LOCK_FILE = path.join(TRAINING_OUTPUT_DIR, 'training-runtime.lock.json');
var TRAINING_STATE_FILE = path.join(TRAINING_OUTPUT_DIR, 'training-runtime-state.json');
var TRAINING_FRAME_FILE = path.join(TRAINING_OUTPUT_DIR, 'training-iter5-live.png');
var TRAINING_LOCK_STALE_MS = 15000;

function safeReadJSON(filepath) {
    try {
        if (!fs.existsSync(filepath)) {
            return null;
        }
        return JSON.parse(fs.readFileSync(filepath, 'utf8'));
    } catch (err) {
        return null;
    }
}

function readTrainingLock() {
    var lock = safeReadJSON(TRAINING_LOCK_FILE) || { active: false };
    var now = Date.now();
    var heartbeat = typeof lock.heartbeat === 'number' ? lock.heartbeat : 0;
    var age = heartbeat > 0 ? (now - heartbeat) : null;
    var alive = false;
    if (typeof lock.pid === 'number' && lock.pid > 0) {
        try {
            process.kill(lock.pid, 0);
            alive = true;
        } catch (err) {
            alive = false;
        }
    }
    lock.pidAlive = alive;
    lock.heartbeatAgeMs = age;
    lock.stale = !!(age != null && age > TRAINING_LOCK_STALE_MS);
    lock.active = !!lock.active && alive && !lock.stale;
    return lock;
}

app.get('/', function (req, res) {
    var lock;
    if (req.query && (req.query.train === '1' || req.query.mode === 'battle')) {
        res.render('index');
        return;
    }
    lock = readTrainingLock();
    if (lock && lock.active) {
        res.redirect('/viewer');
        return;
    }
    res.render('index');
});

app.get('/viewer', function(req, res) {
    res.render('viewer');
});

app.get('/api/training/status', function(req, res) {
    var lock = readTrainingLock();
    var raw = safeReadJSON(TRAINING_STATE_FILE) || {};
    var state = raw && raw.state ? raw.state : raw;
    res.set('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.json({
        ts: Date.now(),
        active: !!lock.active,
        lock: lock,
        state: state,
        raw: raw
    });
});

app.get('/api/training/frame', function(req, res) {
    if (!fs.existsSync(TRAINING_FRAME_FILE)) {
        res.status(404).json({ ok: false, error: 'frame_not_found' });
        return;
    }
    res.set('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.sendFile(TRAINING_FRAME_FILE);
});

io.on('connection', function (socket) {
    var clientIp = socket.request.connection.remoteAddress;
    console.log('New connection from ' + clientIp);

    socket.on('joinRoom', function (roomname) {
        var room = rooms[roomname];
        if (!room) {
            socket.emit('joinRoom', {ret: 0, err: 'no such room'});
        } else {
            socket.roomname = roomname;
            socket.role = 'challenger';
            room.challenger = socket;
            var RandomSeed = Math.random();
            room.master.emit("start", {role: "master", seed: RandomSeed});
            room.challenger.emit("start", {role: "challenger", seed: RandomSeed} )
        }

    });
    socket.on('getRooms', function(data) {
        var msg = {'ret': 1, 'data': Object.keys(rooms)};
        socket.emit('getRooms', msg);
    });
    socket.on('newRoom', function(data) {
        var roomname = data['name'];
        var msg;
        if (roomname in rooms) {
            msg = {'ret': 0, 'err': 'room already existed'}
        } else {
            rooms[roomname] = {master: socket, challenger: null, winner: null};
            msg = {'ret': 1};
            socket.roomname = roomname;
            socket.role = 'master';
        }
        socket.emit('newRooms', msg);
    });
    socket.on('KeyUp', function (data) {
        var room = rooms[socket.roomname];
        if(room){
            if (socket.role === 'master') {
                room.challenger.emit("KU", data);
            } else {
                room.master.emit("KU", data);
            }
        }
    });
    socket.on('KeyDown', function (data) {
        var room = rooms[socket.roomname];
        if (room) {
            if (socket.role === 'master') {
                room.challenger.emit("KD", data);
            } else {
                room.master.emit("KD", data);
            }
        }
    });
    socket.on('end', function (data) {
        var room = rooms[socket.roomname];
        var winner = data;
        if (room.winner == null ) {
            room.winner = winner;
        } else if (room.winner != winner) {
            socket.emit('end', {ret: 0, err: "result don't match"})
        } else {
            room.master.emit('end', {ret: 1, data: winner});
            room.challenger.emit('end', {ret: 1, data: winner});
            delete rooms[socket.roomname];
        }
    });
    socket.on('disconnect', function(){
        var room = rooms[socket.roomname];
        if (room) {
            var other = (room.challenger == socket)?room.master:room.challenger;
            other.emit('err', "Other Player Disconnected!");
        }
    })

});

server.listen(4000, function(){
    var host = server.address().address;
    var port = server.address().port;

    console.log('App listening at http://%s:%s', host, port);
});
