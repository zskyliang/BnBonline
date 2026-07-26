class_name GameBoard
extends Node2D
## Owns the open arena, paint state, bubbles, and pathfinding grid.

signal paint_changed(cell: Vector2i, owner_team: int, is_locked: bool)
signal territory_changed(counts: Dictionary)
signal item_spawned(item: ArenaItemState)
signal item_collected(item: ArenaItemState, actor_id: int)
signal items_cleared
signal hazard_changed
signal board_reset

var map_data: MapData
var cells: Array[PackedInt32Array] = []
var paint_owners: Array[PackedInt32Array] = []
var locked_cells: Array[PackedByteArray] = []
var bombs: Dictionary = {}
var items_by_cell: Dictionary = {}
var player_color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID
var ai_color_id: String = "blue"
var _next_item_id: int = 1

func reset(new_map_data: MapData) -> void:
	map_data = new_map_data
	cells = MapCatalog.clone_matrix(map_data.barrier_cells)
	paint_owners.clear()
	locked_cells.clear()
	for _y: int in range(GameConstants.GRID_ROWS):
		var owner_row := PackedInt32Array()
		owner_row.resize(GameConstants.GRID_COLUMNS)
		owner_row.fill(PaintPalette.TEAM_NEUTRAL)
		paint_owners.append(owner_row)
		var locked_row := PackedByteArray()
		locked_row.resize(GameConstants.GRID_COLUMNS)
		locked_row.fill(0)
		locked_cells.append(locked_row)
	bombs.clear()
	clear_items()
	_next_item_id = 1
	board_reset.emit()
	territory_changed.emit(get_territory_counts())
	hazard_changed.emit()


func configure_team_colors(new_player_color_id: String, new_ai_color_id: String) -> void:
	player_color_id = new_player_color_id
	ai_color_id = new_ai_color_id

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
	return GameConstants.is_inside(cell) \
		and cell_code(cell) == 0 \
		and not bombs.has(cell) \
		and not items_by_cell.has(cell)


func spawn_item(item_type: int, cell: Vector2i, spawned_ms: int = 0) -> int:
	if not ArenaItemType.is_valid(item_type) \
		or not is_cell_walkable(cell) \
		or bombs.has(cell) \
		or items_by_cell.has(cell):
		return 0
	var item := ArenaItemState.new(_next_item_id, item_type, cell, spawned_ms)
	_next_item_id += 1
	items_by_cell[cell] = item
	item_spawned.emit(item.duplicate_state())
	hazard_changed.emit()
	return item.item_id


func take_item(cell: Vector2i, actor_id: int) -> ArenaItemState:
	var item: ArenaItemState = items_by_cell.get(cell) as ArenaItemState
	if item == null:
		return null
	items_by_cell.erase(cell)
	var result: ArenaItemState = item.duplicate_state()
	item_collected.emit(result, actor_id)
	hazard_changed.emit()
	return result


func item_at(cell: Vector2i) -> ArenaItemState:
	var item: ArenaItemState = items_by_cell.get(cell) as ArenaItemState
	return item.duplicate_state() if item != null else null


func item_by_id(item_id: int) -> ArenaItemState:
	for value: Variant in items_by_cell.values():
		var item: ArenaItemState = value as ArenaItemState
		if item != null and item.item_id == item_id:
			return item.duplicate_state()
	return null


func get_item_states() -> Array[ArenaItemState]:
	var result: Array[ArenaItemState] = []
	for value: Variant in items_by_cell.values():
		var item: ArenaItemState = value as ArenaItemState
		if item != null:
			result.append(item.duplicate_state())
	result.sort_custom(func(left: ArenaItemState, right: ArenaItemState) -> bool:
		return left.item_id < right.item_id
	)
	return result


func clear_items() -> void:
	if items_by_cell.is_empty():
		return
	items_by_cell.clear()
	items_cleared.emit()
	hazard_changed.emit()


func paint_cells(
		target_cells: Array[Vector2i],
		owner_team: int,
		lock_painted_cells: bool = false
	) -> Dictionary:
	if owner_team not in [PaintPalette.TEAM_PLAYER, PaintPalette.TEAM_AI]:
		return {}
	var changed: Dictionary = {}
	for cell: Vector2i in target_cells:
		if not GameConstants.is_inside(cell) or is_locked(cell) or changed.has(cell):
			continue
		var previous_owner: int = paint_owner(cell)
		if previous_owner == owner_team and not lock_painted_cells:
			continue
		paint_owners[cell.y][cell.x] = owner_team
		if lock_painted_cells:
			locked_cells[cell.y][cell.x] = 1
		changed[cell] = {
			"previous_owner": previous_owner,
			"owner_team": owner_team,
			"locked": lock_painted_cells,
		}
		paint_changed.emit(cell, owner_team, lock_painted_cells)
	if not changed.is_empty():
		territory_changed.emit(get_territory_counts())
	return changed


func lock_neighborhood(center: Vector2i, owner_team: int) -> Dictionary:
	var target_cells: Array[Vector2i] = []
	for offset_y: int in range(-1, 2):
		for offset_x: int in range(-1, 2):
			var cell := center + Vector2i(offset_x, offset_y)
			if GameConstants.is_inside(cell):
				target_cells.append(cell)
	return paint_cells(target_cells, owner_team, true)


func paint_owner(cell: Vector2i) -> int:
	if not GameConstants.is_inside(cell) or paint_owners.is_empty():
		return PaintPalette.TEAM_NEUTRAL
	return paint_owners[cell.y][cell.x]


func is_locked(cell: Vector2i) -> bool:
	return GameConstants.is_inside(cell) \
		and not locked_cells.is_empty() \
		and locked_cells[cell.y][cell.x] != 0


func territory_swing(target_cells: Array[Vector2i], owner_team: int) -> int:
	var result: int = 0
	var visited: Dictionary = {}
	for cell: Vector2i in target_cells:
		if not GameConstants.is_inside(cell) or is_locked(cell) or visited.has(cell):
			continue
		visited[cell] = true
		var current_owner: int = paint_owner(cell)
		if current_owner == PaintPalette.TEAM_NEUTRAL:
			result += 1
		elif current_owner != owner_team:
			result += 2
	return result


func get_territory_counts() -> Dictionary:
	var player_count: int = 0
	var ai_count: int = 0
	var neutral_count: int = 0
	var player_locked: int = 0
	var ai_locked: int = 0
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var owner: int = paint_owners[y][x] if not paint_owners.is_empty() \
				else PaintPalette.TEAM_NEUTRAL
			match owner:
				PaintPalette.TEAM_PLAYER:
					player_count += 1
					if locked_cells[y][x] != 0:
						player_locked += 1
				PaintPalette.TEAM_AI:
					ai_count += 1
					if locked_cells[y][x] != 0:
						ai_locked += 1
				_:
					neutral_count += 1
	return {
		"player": player_count,
		"ai": ai_count,
		"neutral": neutral_count,
		"player_locked": player_locked,
		"ai_locked": ai_locked,
	}

func get_open_cells() -> Array[Vector2i]:
	var result: Array[Vector2i] = []
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			if is_cell_walkable(cell) and not bombs.has(cell):
				result.append(cell)
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
