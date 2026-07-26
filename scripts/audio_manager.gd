extends Node
## Thin global audio service. Gameplay remains owned by the match scene.

const STREAMS: Dictionary = {
	&"start": preload("res://assets/audio/sfx/start.ogg"),
	&"appear": preload("res://assets/audio/sfx/appear.wav"),
	&"lay": preload("res://assets/audio/sfx/lay.wav"),
	&"explode": preload("res://assets/audio/sfx/explode.wav"),
	&"get": preload("res://assets/audio/sfx/get.ogg"),
	&"save": preload("res://assets/audio/sfx/save.ogg"),
	&"die": preload("res://assets/audio/sfx/die.ogg"),
	&"win": preload("res://assets/audio/sfx/win.ogg"),
	&"draw": preload("res://assets/audio/sfx/draw.ogg"),
}
const MUSIC: AudioStream = preload("res://assets/audio/music/battle_loop.ogg")
const SFX_VOLUME_OFFSETS_DB: Dictionary = {
	&"start": -2.0,
	&"appear": -1.0,
	&"lay": 0.0,
	&"explode": -8.0,
	&"get": -7.0,
	&"save": -2.0,
	&"die": -1.0,
	&"win": 1.0,
	&"draw": -1.0,
}
const SFX_PITCH_RANGES: Dictionary = {
	&"appear": Vector2(0.98, 1.04),
	&"lay": Vector2(0.98, 1.06),
	&"explode": Vector2(1.04, 1.12),
	&"get": Vector2(0.98, 1.08),
}
const SFX_POOL_SIZE: int = 12

var _music_player: AudioStreamPlayer
var _sfx_players: Array[AudioStreamPlayer] = []
var _next_sfx_player: int = 0
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	_rng.randomize()
	var ogg_loop_stream := MUSIC as AudioStreamOggVorbis
	if ogg_loop_stream != null:
		ogg_loop_stream.loop = true
	_music_player = AudioStreamPlayer.new()
	_music_player.stream = MUSIC
	_music_player.volume_db = -5.0
	_music_player.process_mode = Node.PROCESS_MODE_ALWAYS
	_music_player.finished.connect(play_music)
	add_child(_music_player)
	for _index: int in range(SFX_POOL_SIZE):
		var player := AudioStreamPlayer.new()
		player.process_mode = Node.PROCESS_MODE_ALWAYS
		add_child(player)
		_sfx_players.append(player)

func play_music() -> void:
	if not _music_player.playing:
		_music_player.play()

func stop_music() -> void:
	_music_player.stop()

func stop_all() -> void:
	stop_music()
	for player: AudioStreamPlayer in _sfx_players:
		if is_instance_valid(player):
			player.stop()

func play_sfx(sound_name: StringName, volume_db: float = -3.0) -> void:
	if not STREAMS.has(sound_name):
		return
	var player: AudioStreamPlayer = _acquire_sfx_player()
	player.stream = STREAMS[sound_name] as AudioStream
	player.volume_db = volume_db + get_sfx_volume_offset_db(sound_name)
	var pitch_range: Vector2 = SFX_PITCH_RANGES.get(sound_name, Vector2.ONE) as Vector2
	player.pitch_scale = _rng.randf_range(pitch_range.x, pitch_range.y)
	player.play()


func get_sfx_volume_offset_db(sound_name: StringName) -> float:
	return float(SFX_VOLUME_OFFSETS_DB.get(sound_name, 0.0))


func _acquire_sfx_player() -> AudioStreamPlayer:
	for offset: int in range(_sfx_players.size()):
		var index: int = (_next_sfx_player + offset) % _sfx_players.size()
		var candidate: AudioStreamPlayer = _sfx_players[index]
		if not candidate.playing:
			_next_sfx_player = (index + 1) % _sfx_players.size()
			return candidate
	var player: AudioStreamPlayer = _sfx_players[_next_sfx_player]
	_next_sfx_player = (_next_sfx_player + 1) % _sfx_players.size()
	player.stop()
	return player

func _exit_tree() -> void:
	stop_all()
