extends Node
## Thin global audio service. Gameplay remains owned by the match scene.

const STREAMS: Dictionary = {
	&"start": preload("res://assets/audio/start.wav"),
	&"appear": preload("res://assets/audio/appear.wav"),
	&"lay": preload("res://assets/audio/lay.wav"),
	&"explode": preload("res://assets/audio/explode.wav"),
	&"get": preload("res://assets/audio/get.wav"),
	&"save": preload("res://assets/audio/save.wav"),
	&"die": preload("res://assets/audio/die.wav"),
	&"win": preload("res://assets/audio/win.wav"),
	&"draw": preload("res://assets/audio/draw.wav"),
}
const MUSIC: AudioStream = preload("res://assets/audio/bg.wav")

var _music_player: AudioStreamPlayer
var _sfx_players: Array[AudioStreamPlayer] = []

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	_music_player = AudioStreamPlayer.new()
	_music_player.stream = MUSIC
	_music_player.volume_db = -9.0
	_music_player.process_mode = Node.PROCESS_MODE_ALWAYS
	_music_player.finished.connect(play_music)
	add_child(_music_player)

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
			player.queue_free()
	_sfx_players.clear()

func play_sfx(sound_name: StringName, volume_db: float = -3.0) -> void:
	if not STREAMS.has(sound_name):
		return
	var player := AudioStreamPlayer.new()
	player.stream = STREAMS[sound_name] as AudioStream
	player.volume_db = volume_db
	player.finished.connect(_on_sfx_finished.bind(player))
	add_child(player)
	_sfx_players.append(player)
	player.play()

func _on_sfx_finished(player: AudioStreamPlayer) -> void:
	_sfx_players.erase(player)
	player.queue_free()

func _exit_tree() -> void:
	stop_all()
