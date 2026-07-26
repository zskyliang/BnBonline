class_name BoardView3D
extends Node3D
## Batched 15x13 paint floor with incremental ownership and lock updates.

var board: GameBoard
var _world_root: Node3D
var _floor_instance: MultiMeshInstance3D
var _lock_instance: MultiMeshInstance3D
var _displayed_colors: Dictionary = {}


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
	if is_instance_valid(_world_root):
		_world_root.queue_free()
	_world_root = Node3D.new()
	_world_root.name = "PaintArena"
	add_child(_world_root)
	if not is_instance_valid(board) or board.paint_owners.is_empty():
		return
	_build_floor_base()
	_build_floor_tiles()
	_build_lock_shadows()


func get_building_views() -> Array[ClayBuildingView3D]:
	return []


func _on_board_reset() -> void:
	rebuild()


func _on_paint_changed(cell: Vector2i, owner_team: int, is_locked: bool) -> void:
	if not is_instance_valid(_floor_instance) or _floor_instance.multimesh == null:
		return
	var index: int = _cell_index(cell)
	var color: Color = _team_color(owner_team)
	_floor_instance.multimesh.set_instance_color(index, color)
	_displayed_colors[cell] = color
	if is_locked and is_instance_valid(_lock_instance):
		_lock_instance.multimesh.set_instance_transform(index, _lock_transform(cell, true))


func _build_floor_base() -> void:
	var base := MeshInstance3D.new()
	base.name = "NeutralFloorBase"
	var mesh := BoxMesh.new()
	mesh.size = Vector3(
		float(GameConstants.GRID_COLUMNS) + 0.45,
		0.18,
		float(GameConstants.GRID_ROWS) + 0.45
	)
	base.mesh = mesh
	base.position.y = -0.08
	base.material_override = ClayMaterialLibrary.make(Color("#a98268"), 0.94)
	_world_root.add_child(base)


func _build_floor_tiles() -> void:
	var tile_mesh := BoxMesh.new()
	tile_mesh.size = Vector3(0.965, 0.08, 0.965)
	var multi_mesh := MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	multi_mesh.use_colors = true
	multi_mesh.instance_count = GameConstants.GRID_COLUMNS * GameConstants.GRID_ROWS
	multi_mesh.visible_instance_count = multi_mesh.instance_count
	multi_mesh.mesh = tile_mesh
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			var index: int = _cell_index(cell)
			multi_mesh.set_instance_transform(
				index,
				Transform3D(Basis.IDENTITY, GameConstants.grid_to_world_3d(cell, 0.055))
			)
			multi_mesh.set_instance_color(index, _team_color(board.paint_owner(cell)))
			_displayed_colors[cell] = _team_color(board.paint_owner(cell))
	_floor_instance = MultiMeshInstance3D.new()
	_floor_instance.name = "PaintFloorTiles"
	_floor_instance.multimesh = multi_mesh
	var material := ClayMaterialLibrary.make(Color.WHITE, 0.94).duplicate() as StandardMaterial3D
	material.vertex_color_use_as_albedo = true
	_floor_instance.material_override = material
	_floor_instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_world_root.add_child(_floor_instance)


func _build_lock_shadows() -> void:
	var shadow_mesh := BoxMesh.new()
	shadow_mesh.size = Vector3(0.91, 0.014, 0.91)
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
	shadow_material.albedo_color = Color(0.09, 0.07, 0.1, 0.24)
	shadow_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	shadow_material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	shadow_material.cull_mode = BaseMaterial3D.CULL_DISABLED
	_lock_instance.material_override = shadow_material
	_lock_instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_world_root.add_child(_lock_instance)


func _lock_transform(cell: Vector2i, visible: bool) -> Transform3D:
	var basis := Basis.IDENTITY
	if not visible:
		basis = basis.scaled(Vector3.ZERO)
	var position := GameConstants.grid_to_world_3d(cell, 0.108)
	return Transform3D(basis, position)


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
