class_name ArenaView3D
extends Node3D

signal zoom_changed(percent: int)

# A slight horizontal skew keeps the clay buildings visibly three-dimensional
# while logical directions remain close to the screen's vertical/horizontal axes.
const CAMERA_AZIMUTH_DEGREES := -5.0
const CAMERA_ELEVATION_DEGREES := 38.0
const CAMERA_DISTANCE := 28.0
const CAMERA_MARGIN := 1.14
const DEFAULT_ZOOM := 1.1
const MIN_ZOOM := 0.85
const MAX_ZOOM := 1.5
const ZOOM_STEP := 0.1

var board_view: BoardView3D
var actor_root: Node3D
var bubble_root: Node3D
var explosion_root: Node3D
var camera: Camera3D
var roof_occlusion_controller: RoofOcclusionController

var _current_map_data: MapData
var _fitted_camera_size: float = 20.0
var _zoom: float = DEFAULT_ZOOM


func _ready() -> void:
	_build_environment()
	board_view = BoardView3D.new()
	board_view.name = "BoardView3D"
	add_child(board_view)
	bubble_root = Node3D.new()
	bubble_root.name = "BubbleViews3D"
	add_child(bubble_root)
	actor_root = Node3D.new()
	actor_root.name = "ActorViews3D"
	add_child(actor_root)
	explosion_root = ClayExplosionPool.new()
	explosion_root.name = "ClayExplosionPool"
	add_child(explosion_root)
	roof_occlusion_controller = RoofOcclusionController.new()
	roof_occlusion_controller.name = "RoofOcclusionController"
	add_child(roof_occlusion_controller)
	roof_occlusion_controller.bind(camera, actor_root)
	get_viewport().size_changed.connect(_on_viewport_size_changed)


func bind_board(board: GameBoard) -> void:
	board_view.bind_board(board)
	_current_map_data = board.map_data
	fit_camera(_current_map_data)


func clear_entities() -> void:
	for root in [actor_root, bubble_root]:
		for child in root.get_children():
			child.queue_free()
	if explosion_root is ClayExplosionPool:
		(explosion_root as ClayExplosionPool).release_all()


func add_actor(actor: GameActor, character_id: String) -> ActorView3D:
	var view := ActorView3D.new()
	actor_root.add_child(view)
	view.bind_actor(actor, character_id)
	return view


func add_bubble(bubble: GameBubble) -> BubbleView3D:
	var view := BubbleView3D.new()
	bubble_root.add_child(view)
	view.bind_bubble(bubble)
	return view


func add_explosion(effect: ExplosionEffect) -> ExplosionView3D:
	return (explosion_root as ClayExplosionPool).spawn(effect)


func zoom_in() -> void:
	set_zoom(_zoom + ZOOM_STEP)


func zoom_out() -> void:
	set_zoom(_zoom - ZOOM_STEP)


func reset_zoom() -> void:
	set_zoom(DEFAULT_ZOOM)


func set_zoom(value: float) -> void:
	var next_zoom := clampf(value, MIN_ZOOM, MAX_ZOOM)
	if is_equal_approx(next_zoom, _zoom) and is_instance_valid(camera):
		_apply_zoom()
		return
	_zoom = next_zoom
	_apply_zoom()
	zoom_changed.emit(get_zoom_percent())


func get_zoom() -> float:
	return _zoom


func get_zoom_percent() -> int:
	return roundi(_zoom * 100.0)


func fit_camera(map_data: MapData, viewport_size_override: Vector2 = Vector2.ZERO) -> void:
	if not is_instance_valid(camera) or map_data == null:
		return
	_current_map_data = map_data
	var azimuth := deg_to_rad(CAMERA_AZIMUTH_DEGREES)
	var elevation := deg_to_rad(CAMERA_ELEVATION_DEGREES)
	var target := Vector3(0.0, 0.7, 0.0)
	var target_to_camera := Vector3(
		sin(azimuth) * cos(elevation),
		sin(elevation),
		cos(azimuth) * cos(elevation)
	).normalized()
	var camera_position := target + target_to_camera * CAMERA_DISTANCE
	camera.look_at_from_position(camera_position, target, Vector3.UP)
	var viewport_size := (
		viewport_size_override
		if viewport_size_override.x > 0.0 and viewport_size_override.y > 0.0
		else get_viewport().get_visible_rect().size
	)
	var aspect := viewport_size.x / maxf(1.0, viewport_size.y)
	var right := camera.global_transform.basis.x.normalized()
	var up := camera.global_transform.basis.y.normalized()
	var maximum_x := 0.0
	var maximum_y := 0.0
	for index: int in range(8):
		var relative := map_data.camera_bounds.get_endpoint(index) - target
		maximum_x = maxf(maximum_x, absf(relative.dot(right)))
		maximum_y = maxf(maximum_y, absf(relative.dot(up)))
	var required_height := maxf(maximum_y * 2.0, maximum_x * 2.0 / aspect)
	_fitted_camera_size = required_height * CAMERA_MARGIN
	_apply_zoom()
	zoom_changed.emit(get_zoom_percent())


func _apply_zoom() -> void:
	if is_instance_valid(camera):
		camera.size = _fitted_camera_size / _zoom


func _on_viewport_size_changed() -> void:
	fit_camera(_current_map_data)


func _build_environment() -> void:
	var environment_node := WorldEnvironment.new()
	environment_node.name = "ClayWorldEnvironment"
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color("#4faed0")
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = ClayMaterialLibrary.CREAM
	environment.ambient_light_energy = 0.13
	environment.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	environment_node.environment = environment
	add_child(environment_node)

	var sun := DirectionalLight3D.new()
	sun.name = "Sun"
	sun.rotation_degrees = Vector3(-58.0, -32.0, 0.0)
	sun.light_color = Color("#ffe2ba")
	sun.light_energy = 0.92
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = 26.0
	add_child(sun)

	camera = Camera3D.new()
	camera.name = "ArenaCamera3D"
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = 20.0
	camera.current = true
	camera.keep_aspect = Camera3D.KEEP_HEIGHT
	add_child(camera)
