class_name GameBoard
extends Node2D
## Owns mutable map cells, map rendering, items, bombs, and pathfinding grids.

signal cell_changed(cell: Vector2i, new_code: int)
signal hazard_changed
signal board_reset

var map_data: MapData
var cells: Array[PackedInt32Array] = []
var bombs: Dictionary = {}

## Compatibility-only depth metadata used by legacy rule tests. These nodes
## have no texture and live under the hidden 2D logic world.
var _visual_root: Node2D
var _cell_sprites: Dictionary = {}
var _rng := RandomNumberGenerator.new()

func _init() -> void:
	_rng.randomize()

func reset(new_map_data: MapData) -> void:
	map_data = new_map_data
	cells = MapCatalog.clone_matrix(map_data.barrier_cells)
	bombs.clear()
	_cell_sprites.clear()
	if is_instance_valid(_visual_root):
		_visual_root.queue_free()
	_visual_root = Node2D.new()
	_visual_root.name = "DepthMetadata"
	add_child(_visual_root)
	for y in range(GameConstants.GRID_ROWS):
		for x in range(GameConstants.GRID_COLUMNS):
			var code: int = cells[y][x]
			if code <= 0 or code >= 100 or code == 9:
				continue
			var cell := Vector2i(x, y)
			var depth_marker := Sprite2D.new()
			depth_marker.z_index = 40 + int(GameConstants.grid_to_world(cell).y)
			_visual_root.add_child(depth_marker)
			_cell_sprites[cell] = depth_marker
	board_reset.emit()
	hazard_changed.emit()

func cell_code(cell: Vector2i) -> int:
	if not GameConstants.is_inside(cell):
		return 9
	return cells[cell.y][cell.x]

func is_cell_walkable(cell: Vector2i) -> bool:
	return GameConstants.is_inside(cell) and GameRules.is_walkable(cell_code(cell))

func can_actor_occupy(world_position: Vector2, actor: GameActor) -> bool:
	if not GameConstants.is_actor_center_inside_arena(world_position):
		return false
	var occupied_cells: Array[Vector2i] = GameRules.body_cells(world_position)
	var current_cells: Array[Vector2i] = GameRules.body_cells(actor.position)
	for cell: Vector2i in occupied_cells:
		if not is_cell_walkable(cell):
			return false
		if bombs.has(cell):
			var bubble: GameBubble = bombs[cell] as GameBubble
			var can_finish_exiting: bool = is_instance_valid(bubble) \
				and bubble.can_actor_finish_exiting(actor, current_cells)
			if not can_finish_exiting:
				return false
	return true

func can_actor_move(
		from_position: Vector2,
		to_position: Vector2,
		actor: GameActor
	) -> bool:
	if not can_actor_occupy(to_position, actor):
		return false
	var motion: Vector2 = to_position - from_position
	if motion.x != 0.0:
		return _has_horizontal_rigid_clearance(from_position, to_position, motion.x)
	if motion.y != 0.0:
		return _has_vertical_rigid_clearance(from_position, to_position, motion.y)
	return true

func _has_horizontal_rigid_clearance(
		from_position: Vector2,
		to_position: Vector2,
		direction: float
	) -> bool:
	var row: int = GameConstants.world_to_grid(to_position).y
	if row < 0 or row >= GameConstants.GRID_ROWS:
		return false
	var nearby_columns: Vector2i = _nearby_axis_range(
		from_position.x,
		to_position.x,
		GameConstants.GRID_ORIGIN.x,
		GameConstants.GRID_COLUMNS
	)
	for x: int in range(nearby_columns.x, nearby_columns.y + 1):
		var cell := Vector2i(x, row)
		if not _uses_rigid_center_boundary(cell):
			continue
		var left: float = GameConstants.grid_to_top_left(cell).x
		var right: float = left + GameConstants.CELL_SIZE
		if direction > 0.0 \
				and from_position.x <= left \
				and to_position.x > left - GameRules.RIGID_CENTER_CLEARANCE:
			return false
		if direction < 0.0 \
				and from_position.x >= right \
				and to_position.x < right + GameRules.RIGID_CENTER_CLEARANCE:
			return false
	return true

func _has_vertical_rigid_clearance(
		from_position: Vector2,
		to_position: Vector2,
		direction: float
	) -> bool:
	var column: int = GameConstants.world_to_grid(to_position).x
	if column < 0 or column >= GameConstants.GRID_COLUMNS:
		return false
	var nearby_rows: Vector2i = _nearby_axis_range(
		from_position.y,
		to_position.y,
		GameConstants.GRID_ORIGIN.y,
		GameConstants.GRID_ROWS
	)
	for y: int in range(nearby_rows.x, nearby_rows.y + 1):
		var cell := Vector2i(column, y)
		if not _uses_rigid_center_boundary(cell):
			continue
		var top: float = GameConstants.grid_to_top_left(cell).y
		var bottom: float = top + GameConstants.CELL_SIZE
		if direction > 0.0 \
				and from_position.y <= top \
				and to_position.y > top - GameRules.RIGID_CENTER_CLEARANCE:
			return false
		if direction < 0.0 \
				and from_position.y >= bottom \
				and to_position.y < bottom + GameRules.RIGID_CENTER_CLEARANCE:
			return false
	return true

func _uses_rigid_center_boundary(cell: Vector2i) -> bool:
	var code: int = cell_code(cell)
	return code > 0 and code < 100

func _nearby_axis_range(
		from_axis: float,
		to_axis: float,
		origin_axis: float,
		cell_count: int
	) -> Vector2i:
	var minimum_axis: float = minf(from_axis, to_axis) \
		- GameRules.RIGID_CENTER_CLEARANCE \
		- GameConstants.CELL_SIZE
	var maximum_axis: float = maxf(from_axis, to_axis) + GameRules.RIGID_CENTER_CLEARANCE
	return Vector2i(
		clampi(floori((minimum_axis - origin_axis) / GameConstants.CELL_SIZE), 0, cell_count - 1),
		clampi(floori((maximum_axis - origin_axis) / GameConstants.CELL_SIZE), 0, cell_count - 1)
	)

func register_bubble(bubble: GameBubble) -> void:
	bombs[bubble.cell] = bubble
	hazard_changed.emit()

func unregister_bubble(bubble: GameBubble) -> void:
	if bombs.get(bubble.cell) == bubble:
		bombs.erase(bubble.cell)
		hazard_changed.emit()

func can_place_bubble(cell: Vector2i) -> bool:
	return GameConstants.is_inside(cell) and cell_code(cell) == 0 and not bombs.has(cell)

func take_item(cell: Vector2i) -> int:
	var code: int = cell_code(cell)
	if code < 101:
		return 0
	cells[cell.y][cell.x] = 0
	cell_changed.emit(cell, 0)
	return code

func destroy_cell(cell: Vector2i) -> int:
	var code: int = cell_code(cell)
	if not GameRules.is_destructible(code):
		return 0
	var item_codes: PackedInt32Array = PackedInt32Array([
		GameConstants.ITEM_BUBBLE, GameConstants.ITEM_SPEED, GameConstants.ITEM_POWER,
	])
	var item_code: int = item_codes[_rng.randi_range(0, item_codes.size() - 1)]
	cells[cell.y][cell.x] = item_code
	cell_changed.emit(cell, item_code)
	hazard_changed.emit()
	return item_code

func get_open_cells() -> Array[Vector2i]:
	var result: Array[Vector2i] = []
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			if is_cell_walkable(cell) and not bombs.has(cell):
				result.append(cell)
	return result

func get_item_cells() -> Array[Vector2i]:
	var result: Array[Vector2i] = []
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			if cells[y][x] >= 101:
				result.append(Vector2i(x, y))
	return result

func get_astar(actor: GameActor = null) -> AStarGrid2D:
	var astar := AStarGrid2D.new()
	astar.region = Rect2i(0, 0, GameConstants.GRID_COLUMNS, GameConstants.GRID_ROWS)
	astar.cell_size = Vector2.ONE
	astar.diagonal_mode = AStarGrid2D.DIAGONAL_MODE_NEVER
	astar.update()
	var actor_cell: Vector2i = actor.current_cell() if is_instance_valid(actor) else Vector2i(-1, -1)
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			var solid: bool = not is_cell_walkable(cell)
			if bombs.has(cell) and cell != actor_cell:
				solid = true
			astar.set_point_solid(cell, solid)
	return astar

func find_path(from_cell: Vector2i, to_cell: Vector2i, actor: GameActor = null) -> Array[Vector2i]:
	if not GameConstants.is_inside(from_cell) or not GameConstants.is_inside(to_cell):
		return []
	var astar: AStarGrid2D = get_astar(actor)
	if astar.is_point_solid(to_cell):
		return []
	return astar.get_id_path(from_cell, to_cell)

func predicted_blast(cell: Vector2i, power: int) -> Array[Vector2i]:
	return GameRules.blast_cells(cell, power, cells)
