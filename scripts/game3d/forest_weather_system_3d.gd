class_name ForestWeatherSystem3D
extends Node3D
## Visual-only, deterministic forest weather for the fixed orthographic arena.
## It never reads or mutates gameplay state.

signal weather_changed(weather: int)
signal wind_strength_changed(strength: float)

enum Weather {
	CLEAR,
	RAIN,
}

const RAIN_STREAK_SHADER := preload(
	"res://assets/materials/storybook_rain_streak.gdshader"
)
const RAIN_RIPPLE_SHADER := preload(
	"res://assets/materials/storybook_rain_ripple.gdshader"
)
const DROP_COUNT := 84
const RIPPLE_COUNT := 36
const RIPPLE_LIFETIME := 0.82
const RAIN_SECONDS := 22.0
const CLEAR_SECONDS := 16.0
const BOARD_MIN_X := -7.45
const BOARD_MAX_X := 7.45
const BOARD_MIN_Z := -6.45
const BOARD_MAX_Z := 6.45

var _weather: int = Weather.RAIN
var _weather_seconds_left := RAIN_SECONDS
var _rain_intensity := 1.0
var _target_rain_intensity := 1.0
var _last_emitted_wind := -1.0
var _rng := RandomNumberGenerator.new()
var _rain_streaks: MultiMeshInstance3D
var _rain_ripples: MultiMeshInstance3D
var _drop_positions: Array[Vector3] = []
var _drop_speeds: Array[float] = []
var _drop_lengths: Array[float] = []
var _ripple_cells: Array[Vector2i] = []
var _ripple_ages: Array[float] = []
var _next_ripple_index := 0


func _ready() -> void:
	_rng.seed = 0xF025D
	_build_rain_streaks()
	_build_ripple_pool()
	set_weather(Weather.RAIN, true)


func _process(delta: float) -> void:
	_weather_seconds_left -= delta
	if _weather_seconds_left <= 0.0:
		set_weather(
			Weather.CLEAR if _weather == Weather.RAIN else Weather.RAIN
		)
	_rain_intensity = move_toward(
		_rain_intensity,
		_target_rain_intensity,
		delta * (1.35 if _target_rain_intensity > 0.0 else 0.75)
	)
	var wind_strength := lerpf(0.52, 1.0, _rain_intensity)
	if absf(wind_strength - _last_emitted_wind) >= 0.015:
		_last_emitted_wind = wind_strength
		wind_strength_changed.emit(wind_strength)
	_update_rain_streaks(delta)
	_update_ripples(delta)


func set_weather(next_weather: int, immediate: bool = false) -> void:
	_weather = clampi(next_weather, Weather.CLEAR, Weather.RAIN)
	_weather_seconds_left = RAIN_SECONDS if _weather == Weather.RAIN else CLEAR_SECONDS
	_target_rain_intensity = 1.0 if _weather == Weather.RAIN else 0.0
	if immediate:
		_rain_intensity = _target_rain_intensity
	weather_changed.emit(_weather)


func get_weather_state() -> int:
	return _weather


func get_weather_name() -> String:
	return "rain" if _weather == Weather.RAIN else "clear"


func get_rain_intensity() -> float:
	return _rain_intensity


func get_active_ripple_count() -> int:
	var active := 0
	for age: float in _ripple_ages:
		if age >= 0.0 and age < RIPPLE_LIFETIME:
			active += 1
	return active


func debug_spawn_ripple(cell: Vector2i) -> void:
	if GameConstants.is_inside(cell):
		_spawn_ripple(cell)


func _build_rain_streaks() -> void:
	var mesh := QuadMesh.new()
	mesh.size = Vector2.ONE
	var multi_mesh := MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	multi_mesh.use_colors = true
	multi_mesh.instance_count = DROP_COUNT
	multi_mesh.visible_instance_count = DROP_COUNT
	multi_mesh.mesh = mesh
	_drop_positions.clear()
	_drop_speeds.clear()
	_drop_lengths.clear()
	for index: int in range(DROP_COUNT):
		_drop_positions.append(_new_drop_position(index))
		_drop_speeds.append(_rng.randf_range(6.8, 9.6))
		_drop_lengths.append(_rng.randf_range(0.38, 0.68))
		multi_mesh.set_instance_transform(index, Transform3D.IDENTITY)
		multi_mesh.set_instance_color(index, Color.TRANSPARENT)
	_rain_streaks = MultiMeshInstance3D.new()
	_rain_streaks.name = "WatercolorRainStreaks"
	_rain_streaks.multimesh = multi_mesh
	var material := ShaderMaterial.new()
	material.shader = RAIN_STREAK_SHADER
	_rain_streaks.material_override = material
	_rain_streaks.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(_rain_streaks)


func _build_ripple_pool() -> void:
	var mesh := QuadMesh.new()
	mesh.size = Vector2.ONE
	var multi_mesh := MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	multi_mesh.use_colors = true
	multi_mesh.instance_count = RIPPLE_COUNT
	multi_mesh.visible_instance_count = RIPPLE_COUNT
	multi_mesh.mesh = mesh
	_ripple_cells.clear()
	_ripple_ages.clear()
	for index: int in range(RIPPLE_COUNT):
		_ripple_cells.append(Vector2i.ZERO)
		_ripple_ages.append(-1.0)
		multi_mesh.set_instance_transform(
			index,
			Transform3D(Basis().scaled(Vector3.ZERO), Vector3.ZERO)
		)
		multi_mesh.set_instance_color(index, Color.TRANSPARENT)
	_rain_ripples = MultiMeshInstance3D.new()
	_rain_ripples.name = "BoardRainRipples"
	_rain_ripples.multimesh = multi_mesh
	var material := ShaderMaterial.new()
	material.shader = RAIN_RIPPLE_SHADER
	_rain_ripples.material_override = material
	_rain_ripples.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(_rain_ripples)


func _update_rain_streaks(delta: float) -> void:
	if not is_instance_valid(_rain_streaks) or _rain_streaks.multimesh == null:
		return
	_rain_streaks.visible = _rain_intensity > 0.015
	var screen_facing_basis := Basis(Vector3.RIGHT, deg_to_rad(-54.0))
	for index: int in range(DROP_COUNT):
		var position := _drop_positions[index]
		position += Vector3(0.42, -_drop_speeds[index], 0.22) * delta
		if position.y <= 0.12:
			if position.x >= BOARD_MIN_X and position.x <= BOARD_MAX_X \
					and position.z >= BOARD_MIN_Z and position.z <= BOARD_MAX_Z \
					and _rain_intensity > 0.2:
				_spawn_ripple(Vector2i(
					clampi(roundi(position.x + 7.0), 0, GameConstants.GRID_COLUMNS - 1),
					clampi(roundi(position.z + 6.0), 0, GameConstants.GRID_ROWS - 1)
				))
			position = _new_drop_position(index)
		_drop_positions[index] = position
		var basis := screen_facing_basis.scaled(
			Vector3(0.035, _drop_lengths[index], 1.0)
		)
		_rain_streaks.multimesh.set_instance_transform(
			index,
			Transform3D(basis, position)
		)
		var depth_variation := 0.72 + float(index % 7) * 0.035
		_rain_streaks.multimesh.set_instance_color(
			index,
			Color(0.68, 0.86, 0.92, _rain_intensity * depth_variation * 0.62)
		)


func _update_ripples(delta: float) -> void:
	if not is_instance_valid(_rain_ripples) or _rain_ripples.multimesh == null:
		return
	var flat_basis := Basis(Vector3.RIGHT, -PI * 0.5)
	for index: int in range(RIPPLE_COUNT):
		var age := _ripple_ages[index]
		if age < 0.0:
			continue
		age += delta
		if age >= RIPPLE_LIFETIME:
			_ripple_ages[index] = -1.0
			_rain_ripples.multimesh.set_instance_transform(
				index,
				Transform3D(Basis().scaled(Vector3.ZERO), Vector3.ZERO)
			)
			_rain_ripples.multimesh.set_instance_color(index, Color.TRANSPARENT)
			continue
		_ripple_ages[index] = age
		var progress := age / RIPPLE_LIFETIME
		var scale := lerpf(0.18, 0.92, ease(progress, -0.55))
		var basis := flat_basis.scaled(Vector3(scale, scale, scale))
		var position := GameConstants.grid_to_world_3d(_ripple_cells[index], 0.098)
		_rain_ripples.multimesh.set_instance_transform(
			index,
			Transform3D(basis, position)
		)
		_rain_ripples.multimesh.set_instance_color(
			index,
			Color(0.42, 0.72, 0.82, (1.0 - progress) * 0.58)
		)


func _new_drop_position(index: int) -> Vector3:
	var board_biased := index % 5 != 0
	return Vector3(
		_rng.randf_range(
			BOARD_MIN_X,
			BOARD_MAX_X
		) if board_biased else _rng.randf_range(-9.4, 9.4),
		_rng.randf_range(2.8, 8.5),
		_rng.randf_range(
			BOARD_MIN_Z,
			BOARD_MAX_Z
		) if board_biased else _rng.randf_range(-8.0, 8.0)
	)


func _spawn_ripple(cell: Vector2i) -> void:
	var index := _next_ripple_index
	_next_ripple_index = (_next_ripple_index + 1) % RIPPLE_COUNT
	_ripple_cells[index] = cell
	_ripple_ages[index] = 0.0
	var basis := Basis(Vector3.RIGHT, -PI * 0.5).scaled(Vector3.ONE * 0.18)
	_rain_ripples.multimesh.set_instance_transform(
		index,
		Transform3D(basis, GameConstants.grid_to_world_3d(cell, 0.098))
	)
	_rain_ripples.multimesh.set_instance_color(
		index,
		Color(0.42, 0.72, 0.82, 0.58)
	)
