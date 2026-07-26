class_name ArenaView3D
extends Node3D

signal camera_pose_changed(azimuth: float, elevation: float, zoom: float)
signal camera_adjustment_finished(azimuth: float, elevation: float, zoom: float)
signal zoom_changed(percent: int)

const CAMERA_DISTANCE := 28.0
const CAMERA_MARGIN := 1.14
const ZOOM_STEP := 0.1
const ORBIT_SENSITIVITY_DEGREES := 0.22

var board_view: BoardView3D
var actor_root: Node3D
var bubble_root: Node3D
var explosion_root: Node3D
var item_root: Node3D
var camera: Camera3D

var _current_map_data: MapData
var _fitted_camera_size: float = 20.0
var _azimuth: float = MatchSettings.DEFAULT_CAMERA_AZIMUTH
var _elevation: float = MatchSettings.DEFAULT_CAMERA_ELEVATION
var _zoom: float = MatchSettings.DEFAULT_CAMERA_ZOOM
var _orbit_dragging: bool = false
var _item_views: Dictionary = {}


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
	item_root = Node3D.new()
	item_root.name = "ArenaItemViews3D"
	add_child(item_root)
	get_viewport().size_changed.connect(_on_viewport_size_changed)


func bind_board(board: GameBoard) -> void:
	var previous_board: GameBoard = board_view.board if is_instance_valid(board_view) else null
	if is_instance_valid(previous_board):
		if previous_board.item_spawned.is_connected(_on_item_spawned):
			previous_board.item_spawned.disconnect(_on_item_spawned)
		if previous_board.item_collected.is_connected(_on_item_collected):
			previous_board.item_collected.disconnect(_on_item_collected)
		if previous_board.items_cleared.is_connected(_clear_item_views):
			previous_board.items_cleared.disconnect(_clear_item_views)
	board_view.bind_board(board)
	_current_map_data = board.map_data
	board.item_spawned.connect(_on_item_spawned)
	board.item_collected.connect(_on_item_collected)
	board.items_cleared.connect(_clear_item_views)
	_clear_item_views()
	for item: ArenaItemState in board.get_item_states():
		add_item(item)
	fit_camera(_current_map_data)


func clear_entities() -> void:
	for root: Node3D in [actor_root, bubble_root]:
		for child in root.get_children():
			child.queue_free()
	_clear_item_views()
	if explosion_root is ClayExplosionPool:
		(explosion_root as ClayExplosionPool).release_all()


func add_actor(
		actor: GameActor,
		character_id: String,
		color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID
	) -> ActorView3D:
	var view := ActorView3D.new()
	actor_root.add_child(view)
	view.bind_actor(actor, character_id, color_id)
	return view


func add_bubble(bubble: GameBubble) -> BubbleView3D:
	if not is_instance_valid(bubble) or bubble.has_exploded:
		return null
	var view := BubbleView3D.new()
	bubble_root.add_child(view)
	view.bind_bubble(bubble)
	return view


func add_item(item: ArenaItemState) -> ItemView3D:
	if item == null or _item_views.has(item.item_id):
		return null
	var view := ItemView3D.new()
	item_root.add_child(view)
	view.setup(item)
	_item_views[item.item_id] = view
	return view


func add_explosion(effect: ExplosionEffect) -> ExplosionView3D:
	return (explosion_root as ClayExplosionPool).spawn(effect)


func play_defeat_burst(center_cell: Vector2i, color_id: String, raw_cells: Array) -> void:
	var burst_cells: Array[Vector2i] = []
	for value: Variant in raw_cells:
		if value is Vector2i:
			burst_cells.append(value as Vector2i)
	if burst_cells.is_empty():
		burst_cells.append(center_cell)
	var effect := ExplosionEffect.new()
	add_child(effect)
	effect.setup(burst_cells, center_cell, null)
	effect.color_id = color_id
	add_explosion(effect)


func zoom_in() -> void:
	set_zoom(_zoom + ZOOM_STEP)
	_emit_adjustment_finished()


func zoom_out() -> void:
	set_zoom(_zoom - ZOOM_STEP)
	_emit_adjustment_finished()


func reset_zoom() -> void:
	set_zoom(MatchSettings.DEFAULT_CAMERA_ZOOM)
	_emit_adjustment_finished()


func reset_camera() -> void:
	set_camera_pose(
		MatchSettings.DEFAULT_CAMERA_AZIMUTH,
		MatchSettings.DEFAULT_CAMERA_ELEVATION,
		MatchSettings.DEFAULT_CAMERA_ZOOM
	)
	_emit_adjustment_finished()


func set_zoom(value: float) -> void:
	var next_zoom := clampf(
		value,
		MatchSettings.MIN_CAMERA_ZOOM,
		MatchSettings.MAX_CAMERA_ZOOM
	)
	if is_equal_approx(next_zoom, _zoom) and is_instance_valid(camera):
		_apply_zoom()
		return
	_zoom = next_zoom
	_apply_zoom()
	zoom_changed.emit(get_zoom_percent())
	_emit_pose_changed()


func set_camera_pose(azimuth: float, elevation: float, zoom: float) -> void:
	_azimuth = clampf(
		azimuth,
		MatchSettings.MIN_CAMERA_AZIMUTH,
		MatchSettings.MAX_CAMERA_AZIMUTH
	)
	_elevation = clampf(
		elevation,
		MatchSettings.MIN_CAMERA_ELEVATION,
		MatchSettings.MAX_CAMERA_ELEVATION
	)
	_zoom = clampf(
		zoom,
		MatchSettings.MIN_CAMERA_ZOOM,
		MatchSettings.MAX_CAMERA_ZOOM
	)
	fit_camera(_current_map_data)
	_emit_pose_changed()


func apply_camera_settings(settings: MatchSettings) -> void:
	if settings == null:
		return
	set_camera_pose(
		settings.camera_azimuth,
		settings.camera_elevation,
		settings.camera_zoom
	)


func finish_camera_adjustment() -> void:
	_emit_adjustment_finished()


func get_zoom() -> float:
	return _zoom


func get_zoom_percent() -> int:
	return roundi(_zoom * 100.0)


func get_azimuth() -> float:
	return _azimuth


func get_elevation() -> float:
	return _elevation


func fit_camera(map_data: MapData, viewport_size_override: Vector2 = Vector2.ZERO) -> void:
	if not is_instance_valid(camera) or map_data == null:
		return
	_current_map_data = map_data
	var azimuth := deg_to_rad(_azimuth)
	var elevation := deg_to_rad(_elevation)
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


func _input(event: InputEvent) -> void:
	if not visible:
		return
	if event is InputEventMouseButton:
		var mouse_button := event as InputEventMouseButton
		if mouse_button.button_index == MOUSE_BUTTON_RIGHT:
			_orbit_dragging = mouse_button.pressed
			if not _orbit_dragging:
				_emit_adjustment_finished()
			get_viewport().set_input_as_handled()
	elif event is InputEventMouseMotion and _orbit_dragging:
		var motion := event as InputEventMouseMotion
		set_camera_pose(
			_azimuth - motion.relative.x * ORBIT_SENSITIVITY_DEGREES,
			_elevation + motion.relative.y * ORBIT_SENSITIVITY_DEGREES,
			_zoom
		)
		get_viewport().set_input_as_handled()


func _on_item_spawned(item: ArenaItemState) -> void:
	add_item(item)


func _on_item_collected(item: ArenaItemState, _actor_id: int) -> void:
	var view: ItemView3D = _item_views.get(item.item_id) as ItemView3D
	_item_views.erase(item.item_id)
	if is_instance_valid(view):
		view.queue_free()


func _clear_item_views() -> void:
	_item_views.clear()
	if not is_instance_valid(item_root):
		return
	for child: Node in item_root.get_children():
		child.queue_free()


func _emit_pose_changed() -> void:
	camera_pose_changed.emit(_azimuth, _elevation, _zoom)


func _emit_adjustment_finished() -> void:
	camera_adjustment_finished.emit(_azimuth, _elevation, _zoom)


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
