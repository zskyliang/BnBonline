class_name BoardView3D
extends Node3D
## Batched 15x13 watercolor board with Sprite3D forest decoration.

const TILE_SHADER := preload("res://assets/materials/storybook_tile.gdshader")
const ART_ROOT := "res://assets/art/storybook25d/environment/"
const WIND_ART_ROOT := ART_ROOT + "wind/"
const WIND_FRAME_COUNT := 4
const WIND_PLAYBACK_SEQUENCE: Array[int] = [0, 2, 1, 2, 3, 2]
const WIND_SPEED_SCALE := 1.0 / 3.0
const WIND_CALM_FRAME_RATE := 3.2 * WIND_SPEED_SCALE
const WIND_RAIN_FRAME_RATE := 6.4 * WIND_SPEED_SCALE
const WIND_ANIMATED_ASSETS: Array[String] = [
	"conifer",
	"bush",
	"mushrooms",
	"flowers",
	"lavender",
]

var board: GameBoard
var _world_root: Node3D
var _floor_instance: MultiMeshInstance3D
var _lock_instance: MultiMeshInstance3D
var _displayed_colors: Dictionary = {}
var _wind_sprites: Array[Sprite3D] = []
var _wind_texture_sets: Array[Array] = []
var _wind_phase_offsets: Array[float] = []
var _wind_frame_indices: Array[int] = []
var _wind_speeds: Array[float] = []
var _wind_elapsed := 0.0
var _wind_strength := 0.52
var _target_wind_strength := 0.52


func _process(delta: float) -> void:
	_wind_elapsed += minf(delta, 0.1)
	_wind_strength = move_toward(_wind_strength, _target_wind_strength, delta * 0.8)
	for index: int in range(_wind_sprites.size()):
		var sprite := _wind_sprites[index]
		if not is_instance_valid(sprite):
			continue
		var animation_rate := get_wind_animation_rate()
		var playback_index := posmod(
			floori(
				(_wind_elapsed + _wind_phase_offsets[index])
				* animation_rate
				* _wind_speeds[index]
			),
			WIND_PLAYBACK_SEQUENCE.size()
		)
		var frame_index := WIND_PLAYBACK_SEQUENCE[playback_index]
		if frame_index == _wind_frame_indices[index]:
			continue
		_wind_frame_indices[index] = frame_index
		var textures := _wind_texture_sets[index]
		sprite.texture = textures[frame_index] as Texture2D


func set_wind_strength(strength: float) -> void:
	_target_wind_strength = clampf(strength, 0.0, 1.35)


func get_wind_animation_rate() -> float:
	return lerpf(WIND_CALM_FRAME_RATE, WIND_RAIN_FRAME_RATE, _wind_strength)


func get_wind_pose_snapshot() -> Dictionary:
	var angles: Array[float] = []
	for sprite: Sprite3D in _wind_sprites:
		if is_instance_valid(sprite):
			angles.append(rad_to_deg(sprite.rotation.z))
	return {
		"plant_count": _wind_sprites.size(),
		"angles_degrees": angles,
		"frame_indices": _wind_frame_indices.duplicate(),
		"phase_offsets_seconds": _wind_phase_offsets.duplicate(),
		"animation_rate": get_wind_animation_rate(),
		"strength": _wind_strength,
	}


func bind_board(logic_board: GameBoard) -> void:
	if is_instance_valid(board):
		if board.paint_changed.is_connected(_on_paint_changed):
			board.paint_changed.disconnect(_on_paint_changed)
		if board.board_reset.is_connected(_on_board_reset):
			board.board_reset.disconnect(_on_board_reset)
	board = logic_board
	if not is_instance_valid(board):
		return
	board.paint_changed.connect(_on_paint_changed)
	board.board_reset.connect(_on_board_reset)
	rebuild()


func rebuild() -> void:
	_displayed_colors.clear()
	_wind_sprites.clear()
	_wind_texture_sets.clear()
	_wind_phase_offsets.clear()
	_wind_frame_indices.clear()
	_wind_speeds.clear()
	if is_instance_valid(_world_root):
		_world_root.queue_free()
	_world_root = Node3D.new()
	_world_root.name = "ImageGenPaintArena"
	add_child(_world_root)
	if not is_instance_valid(board) or board.paint_owners.is_empty():
		return
	_build_grass_ground()
	_build_board_frame()
	_build_floor_tiles()
	_build_lock_shadows()
	_build_forest_sprites()


func _on_board_reset() -> void:
	rebuild()


func _on_paint_changed(cell: Vector2i, owner_team: int, is_locked: bool) -> void:
	if not is_instance_valid(_floor_instance) or _floor_instance.multimesh == null:
		return
	var index := _cell_index(cell)
	var color := _team_color(owner_team)
	_floor_instance.multimesh.set_instance_color(index, color)
	_floor_instance.multimesh.set_instance_custom_data(
		index,
		_tile_custom_data(cell, owner_team)
	)
	_displayed_colors[cell] = color
	if is_locked and is_instance_valid(_lock_instance):
		_lock_instance.multimesh.set_instance_transform(index, _lock_transform(cell, true))


func _build_grass_ground() -> void:
	var ground := MeshInstance3D.new()
	ground.name = "WatercolorGrassGround"
	var mesh := PlaneMesh.new()
	mesh.size = Vector2(22.0, 19.5)
	ground.mesh = mesh
	ground.position.y = -0.13
	var texture := load(ART_ROOT + "grass.png") as Texture2D
	var material := StorybookMaterialLibrary.make_textured(texture, Color("#dce09b"), false)
	material.uv1_scale = Vector3(2.6, 2.3, 1.0)
	ground.material_override = material
	ground.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_world_root.add_child(ground)


func _build_board_frame() -> void:
	var rail_texture := load(ART_ROOT + "wood_rail.png") as Texture2D
	var corner_texture := load(ART_ROOT + "wood_corner.png") as Texture2D
	_add_flat_art_mesh(
		"WoodFrameNorth",
		rail_texture,
		Vector2(15.9, 0.72),
		Vector3(0.0, 0.014, -6.68),
		0.0
	)
	_add_flat_art_mesh(
		"WoodFrameSouth",
		rail_texture,
		Vector2(15.9, 0.72),
		Vector3(0.0, 0.014, 6.68),
		PI
	)
	_add_flat_art_mesh(
		"WoodFrameWest",
		rail_texture,
		Vector2(13.9, 0.72),
		Vector3(-7.68, 0.014, 0.0),
		PI * 0.5
	)
	_add_flat_art_mesh(
		"WoodFrameEast",
		rail_texture,
		Vector2(13.9, 0.72),
		Vector3(7.68, 0.014, 0.0),
		-PI * 0.5
	)
	var corner_positions: Array[Vector3] = [
		Vector3(-7.68, 0.019, -6.68),
		Vector3(7.68, 0.019, -6.68),
		Vector3(7.68, 0.019, 6.68),
		Vector3(-7.68, 0.019, 6.68),
	]
	for index: int in range(corner_positions.size()):
		_add_flat_art_mesh(
			"WoodFrameCorner%d" % index,
			corner_texture,
			Vector2(1.0, 1.0),
			corner_positions[index],
			float(index) * PI * 0.5
		)


func _add_flat_art_mesh(
		node_name: String,
		texture: Texture2D,
		mesh_size: Vector2,
		location: Vector3,
		y_rotation: float
	) -> void:
	var instance := MeshInstance3D.new()
	instance.name = node_name
	var mesh := PlaneMesh.new()
	mesh.size = mesh_size
	instance.mesh = mesh
	instance.position = location
	instance.rotation.y = y_rotation
	instance.material_override = StorybookMaterialLibrary.make_textured(texture)
	instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_world_root.add_child(instance)


func _build_floor_tiles() -> void:
	var tile_mesh := QuadMesh.new()
	tile_mesh.size = Vector2(0.965, 0.965)
	var multi_mesh := MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	multi_mesh.use_colors = true
	multi_mesh.use_custom_data = true
	multi_mesh.instance_count = GameConstants.GRID_COLUMNS * GameConstants.GRID_ROWS
	multi_mesh.visible_instance_count = multi_mesh.instance_count
	multi_mesh.mesh = tile_mesh
	var flat_basis := Basis(Vector3.RIGHT, -PI * 0.5)
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			var index := _cell_index(cell)
			var owner := board.paint_owner(cell)
			multi_mesh.set_instance_transform(
				index,
				Transform3D(flat_basis, GameConstants.grid_to_world_3d(cell, 0.045))
			)
			multi_mesh.set_instance_color(index, _team_color(owner))
			multi_mesh.set_instance_custom_data(index, _tile_custom_data(cell, owner))
			_displayed_colors[cell] = _team_color(owner)
	_floor_instance = MultiMeshInstance3D.new()
	_floor_instance.name = "PaintFloorTiles"
	_floor_instance.multimesh = multi_mesh
	var material := ShaderMaterial.new()
	material.shader = TILE_SHADER
	material.set_shader_parameter(
		"tile_atlas",
		load(ART_ROOT + "floor_tile_atlas.png") as Texture2D
	)
	_floor_instance.material_override = material
	_floor_instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_world_root.add_child(_floor_instance)


func _tile_custom_data(cell: Vector2i, owner_team: int) -> Color:
	var variant := float(absi(cell.x * 17 + cell.y * 31) % 6) / 6.0
	return Color(variant, 0.0 if owner_team == PaintPalette.TEAM_NEUTRAL else 1.0, 0.0, 1.0)


func _build_lock_shadows() -> void:
	var shadow_mesh := QuadMesh.new()
	shadow_mesh.size = Vector2(0.9, 0.9)
	var multi_mesh := MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	multi_mesh.instance_count = GameConstants.GRID_COLUMNS * GameConstants.GRID_ROWS
	multi_mesh.visible_instance_count = multi_mesh.instance_count
	multi_mesh.mesh = shadow_mesh
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			multi_mesh.set_instance_transform(
				_cell_index(cell),
				_lock_transform(cell, board.is_locked(cell))
			)
	_lock_instance = MultiMeshInstance3D.new()
	_lock_instance.name = "LockedTileShadows"
	_lock_instance.multimesh = multi_mesh
	var shadow_material := StandardMaterial3D.new()
	shadow_material.albedo_color = Color(0.13, 0.1, 0.08, 0.2)
	shadow_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	shadow_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	shadow_material.cull_mode = BaseMaterial3D.CULL_DISABLED
	_lock_instance.material_override = shadow_material
	_lock_instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_world_root.add_child(_lock_instance)


func _lock_transform(cell: Vector2i, is_visible: bool) -> Transform3D:
	var basis := Basis(Vector3.RIGHT, -PI * 0.5)
	if not is_visible:
		basis = basis.scaled(Vector3.ZERO)
	return Transform3D(basis, GameConstants.grid_to_world_3d(cell, 0.072))


func _build_forest_sprites() -> void:
	var specs: Array[Dictionary] = [
		{"asset": "conifer", "position": Vector3(-8.75, 0.0, -5.4), "size": 0.0072, "speed": 1.15, "phase": 0.0},
		{"asset": "conifer", "position": Vector3(8.7, 0.0, -5.1), "size": 0.0066, "speed": 1.08, "phase": 1.25},
		{"asset": "conifer", "position": Vector3(-9.0, 0.0, 4.8), "size": 0.0063, "speed": 1.22, "phase": 2.65},
		{"asset": "conifer", "position": Vector3(8.85, 0.0, 4.9), "size": 0.0068, "speed": 1.11, "phase": 4.15},
		{"asset": "bush", "position": Vector3(-8.35, 0.0, -1.2), "size": 0.0054, "speed": 1.45, "phase": 0.72},
		{"asset": "bush", "position": Vector3(8.2, 0.0, 0.9), "size": 0.0051, "speed": 1.38, "phase": 3.42},
		{"asset": "mushrooms", "position": Vector3(-6.0, 0.0, -7.45), "size": 0.0042, "speed": 0.92, "phase": 2.05},
		{"asset": "mushrooms", "position": Vector3(5.7, 0.0, 7.45), "size": 0.0039, "speed": 0.86, "phase": 4.8},
		{"asset": "stump", "position": Vector3(-2.3, 0.0, -7.55), "size": 0.0042},
		{"asset": "wood_sign", "position": Vector3(3.9, 0.0, -7.55), "size": 0.0040},
		{"asset": "rocks", "position": Vector3(8.55, 0.0, -2.5), "size": 0.0032},
		{"asset": "flowers", "position": Vector3(-4.8, 0.0, 7.45), "size": 0.0032, "speed": 1.72, "phase": 1.63},
		{"asset": "lavender", "position": Vector3(2.1, 0.0, 7.5), "size": 0.0030, "speed": 1.64, "phase": 3.83},
	]
	for index: int in range(specs.size()):
		var spec := specs[index]
		var asset_id := str(spec["asset"])
		var texture := load(ART_ROOT + asset_id + ".png") as Texture2D
		if texture == null:
			continue
		var pivot := Node3D.new()
		pivot.name = "%sPlantAnchor%d" % [asset_id.capitalize(), index]
		pivot.position = spec["position"] as Vector3
		_world_root.add_child(pivot)
		var sprite := Sprite3D.new()
		sprite.name = "%sSprite%d" % [asset_id.capitalize(), index]
		sprite.texture = texture
		sprite.pixel_size = float(spec["size"])
		sprite.position.y = float(texture.get_height()) * sprite.pixel_size * 0.48
		StorybookMaterialLibrary.configure_billboard(sprite)
		pivot.add_child(sprite)
		if asset_id not in WIND_ANIMATED_ASSETS:
			continue
		var textures: Array[Texture2D] = []
		for frame_index: int in range(WIND_FRAME_COUNT):
			var frame_texture := load(
				WIND_ART_ROOT + asset_id + "_%d.png" % frame_index
			) as Texture2D
			if frame_texture != null:
				textures.append(frame_texture)
		if textures.size() != WIND_FRAME_COUNT:
			continue
		sprite.texture = textures[0]
		_wind_sprites.append(sprite)
		_wind_texture_sets.append(textures)
		_wind_phase_offsets.append(float(spec.get("phase", 0.0)))
		_wind_frame_indices.append(0)
		_wind_speeds.append(float(spec.get("speed", 1.0)))


func _team_color(owner_team: int) -> Color:
	match owner_team:
		PaintPalette.TEAM_PLAYER:
			return PaintPalette.get_color(board.player_color_id)
		PaintPalette.TEAM_AI:
			return PaintPalette.get_color(board.ai_color_id)
		_:
			return PaintPalette.NEUTRAL_COLOR


func _cell_index(cell: Vector2i) -> int:
	return cell.y * GameConstants.GRID_COLUMNS + cell.x


func displayed_color(cell: Vector2i) -> Color:
	return _displayed_colors.get(cell, PaintPalette.NEUTRAL_COLOR) as Color
