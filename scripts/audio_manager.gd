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
const MENU_MUSIC: AudioStream = preload(
	"res://assets/audio/music/tiptoe_through_the_ferns.mp3"
)
const BATTLE_MUSIC: AudioStream = preload(
	"res://assets/audio/music/puddle_jumpers_loop.ogg"
)
const RAIN_THUNDER_AMBIENCE: AudioStream = preload(
	"res://assets/audio/ambience/gentle_rain_thunder_loop.ogg"
)
const MENU_MUSIC_VOLUME_DB: float = -5.0
const BATTLE_MUSIC_VOLUME_DB: float = -5.0
const RAIN_THUNDER_VOLUME_DB: float = -19.0
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
var _ambience_player: AudioStreamPlayer
var _sfx_players: Array[AudioStreamPlayer] = []
var _next_sfx_player: int = 0
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	_rng.randomize()
	var menu_loop_stream := MENU_MUSIC as AudioStreamMP3
	if menu_loop_stream != null:
		menu_loop_stream.loop = true
	var battle_loop_stream := BATTLE_MUSIC as AudioStreamOggVorbis
	if battle_loop_stream != null:
		battle_loop_stream.loop = true
	_music_player = AudioStreamPlayer.new()
	_music_player.name = "MusicPlayer"
	_music_player.process_mode = Node.PROCESS_MODE_ALWAYS
	_music_player.finished.connect(_restart_current_music)
	add_child(_music_player)
	var ambience_loop_stream := RAIN_THUNDER_AMBIENCE as AudioStreamOggVorbis
	if ambience_loop_stream != null:
		ambience_loop_stream.loop = true
	_ambience_player = AudioStreamPlayer.new()
	_ambience_player.name = "GentleRainThunderAmbiencePlayer"
	_ambience_player.stream = RAIN_THUNDER_AMBIENCE
	_ambience_player.volume_db = RAIN_THUNDER_VOLUME_DB
	_ambience_player.process_mode = Node.PROCESS_MODE_ALWAYS
	_ambience_player.finished.connect(_play_rain_thunder_ambience)
	add_child(_ambience_player)
	for _index: int in range(SFX_POOL_SIZE):
		var player := AudioStreamPlayer.new()
		player.process_mode = Node.PROCESS_MODE_ALWAYS
		add_child(player)
		_sfx_players.append(player)

func play_menu_music() -> void:
	if is_instance_valid(_ambience_player):
		_ambience_player.stop()
	_play_music_stream(MENU_MUSIC, MENU_MUSIC_VOLUME_DB)


func play_battle_music() -> void:
	_play_music_stream(BATTLE_MUSIC, BATTLE_MUSIC_VOLUME_DB)
	_play_rain_thunder_ambience()


## Backward-compatible battle-music entry point used by visual test helpers.
func play_music() -> void:
	play_battle_music()


func stop_music() -> void:
	_music_player.stop()
	if is_instance_valid(_ambience_player):
		_ambience_player.stop()


func is_rain_thunder_ambience_playing() -> bool:
	return is_instance_valid(_ambience_player) and _ambience_player.playing


func get_rain_thunder_volume_db() -> float:
	return RAIN_THUNDER_VOLUME_DB

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


func _play_music_stream(stream: AudioStream, volume_db: float) -> void:
	if _music_player.stream == stream:
		_music_player.volume_db = volume_db
		if not _music_player.playing:
			_music_player.play()
		return
	_music_player.stop()
	_music_player.stream = stream
	_music_player.volume_db = volume_db
	_music_player.play()


func _restart_current_music() -> void:
	if is_instance_valid(_music_player) \
			and _music_player.stream != null \
			and not _music_player.playing:
		_music_player.play()


func _play_rain_thunder_ambience() -> void:
	if is_instance_valid(_ambience_player) and not _ambience_player.playing:
		_ambience_player.play()

func _exit_tree() -> void:
	stop_all()
