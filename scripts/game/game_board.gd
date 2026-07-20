class_name GameBoard
extends Node2D
## Owns mutable map cells, map rendering, items, bombs, and pathfinding grids.

signal cell_changed(cell: Vector2i, new_code: int)
signal hazard_changed

const BACKGROUND_TEXTURE: Texture2D = preload("res://assets/sprites/BG.png")
const TOWN_GROUND_TEXTURE: Texture2D = preload("res://assets/sprites/TownGround.png")
const HEART_GROUND_TEXTURE: Texture2D = preload("res://assets/sprites/MapType2.png")
const BLOCK_RED_TEXTURE: Texture2D = preload("res://assets/sprites/TownBlockRed.png")
const BLOCK_YELLOW_TEXTURE: Texture2D = preload("res://assets/sprites/TownBlockYellow.png")
const BOX_TEXTURE: Texture2D = preload("res://assets/sprites/TownBox.png")
const SAND_BLOCK_TEXTURE: Texture2D = preload("res://assets/sprites/SandBlockYellow.png")
const WINDMILL_BASE_TEXTURE: Texture2D = preload("res://assets/sprites/TownWindmill.png")
const WINDMILL_FAN_TEXTURE: Texture2D = preload("res://assets/sprites/TownWindmillAni.png")
const WINDMILL_COLLISION_ROW_OFFSET: int = 3
const GIFT_TEXTURES: Dictionary = {
	GameConstants.ITEM_BUBBLE: preload("res://assets/sprites/Gift1.png"),
	GameConstants.ITEM_SPEED: preload("res://assets/sprites/Gift2.png"),
	GameConstants.ITEM_POWER: preload("res://assets/sprites/Gift3.png"),
}

var map_data: MapData
var cells: Array[PackedInt32Array] = []
var bombs: Dictionary = {}

var _visual_root: Node2D
var _cell_sprites: Dictionary = {}
var _item_sprites: Dictionary = {}
var _rng := RandomNumberGenerator.new()

func _init() -> void:
	_rng.randomize()
	texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST

func reset(new_map_data: MapData) -> void:
	map_data = new_map_data
	cells = MapCatalog.clone_matrix(map_data.barrier_cells)
	bombs.clear()
	_cell_sprites.clear()
	_item_sprites.clear()
	if is_instance_valid(_visual_root):
		_visual_root.queue_free()
	_visual_root = Node2D.new()
	_visual_root.name = "MapVisuals"
	add_child(_visual_root)
	queue_redraw()
	_draw_barriers()
	_draw_decorations()
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
	if _item_sprites.has(cell):
		var sprite: Node = _item_sprites[cell] as Node
		if is_instance_valid(sprite):
			sprite.queue_free()
		_item_sprites.erase(cell)
	cell_changed.emit(cell, 0)
	return code

func destroy_cell(cell: Vector2i) -> int:
	var code: int = cell_code(cell)
	if not GameRules.is_destructible(code):
		return 0
	_remove_cell_sprite(cell)
	var item_codes: PackedInt32Array = PackedInt32Array([
		GameConstants.ITEM_BUBBLE, GameConstants.ITEM_SPEED, GameConstants.ITEM_POWER,
	])
	var item_code: int = item_codes[_rng.randi_range(0, item_codes.size() - 1)]
	cells[cell.y][cell.x] = item_code
	_draw_item(cell, item_code)
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

func _draw() -> void:
	draw_texture(BACKGROUND_TEXTURE, Vector2.ZERO)
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			var rect: Rect2
			var texture: Texture2D
			if map_data.ground_mode == "maptype2":
				texture = HEART_GROUND_TEXTURE
				rect = Rect2(1, 1, 16, 16)
			else:
				texture = TOWN_GROUND_TEXTURE
				rect = Rect2((map_data.ground_cells[y][x] - 1) * 40, 0, 40, 40)
			draw_texture_rect_region(
				texture,
				Rect2(GameConstants.grid_to_top_left(cell), Vector2(40, 40)),
				rect
			)

func _draw_barriers() -> void:
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			var code: int = cells[y][x]
			if code > 0 and code < 100 and code != 9:
				_draw_barrier(cell, code)

func _draw_barrier(cell: Vector2i, code: int) -> void:
	var texture: Texture2D = BOX_TEXTURE
	if code == 1:
		texture = BLOCK_RED_TEXTURE
	elif code == 2:
		texture = BLOCK_YELLOW_TEXTURE
	elif code == 8:
		texture = SAND_BLOCK_TEXTURE
	var sprite := Sprite2D.new()
	sprite.texture = texture
	sprite.centered = false
	sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	sprite.position = GameConstants.grid_to_top_left(cell) + Vector2(0, -4)
	sprite.z_index = 40 + int(GameConstants.grid_to_world(cell).y)
	_visual_root.add_child(sprite)
	_cell_sprites[cell] = sprite

func _draw_item(cell: Vector2i, code: int) -> void:
	if not GIFT_TEXTURES.has(code):
		return
	var texture: Texture2D = GIFT_TEXTURES[code] as Texture2D
	var sprite: Sprite2D = _make_region_sprite(
		texture, Rect2(0, 0, 42, 45),
		GameConstants.grid_to_top_left(cell) + Vector2(-1, -7), Vector2(42, 45)
	)
	sprite.z_index = 25 + cell.y * 2
	_visual_root.add_child(sprite)
	_item_sprites[cell] = sprite

func _draw_decorations() -> void:
	for decoration: Dictionary in map_data.decorations:
		if decoration.get("type", "") != "windmill":
			continue
		var cell: Vector2i = decoration.get("cell", Vector2i.ZERO)
		var top_left: Vector2 = GameConstants.grid_to_top_left(cell)
		var collision_row: int = mini(GameConstants.GRID_ROWS - 1, cell.y + WINDMILL_COLLISION_ROW_OFFSET)
		var decoration_depth: int = 40 + int(GameConstants.grid_to_world(Vector2i(cell.x, collision_row)).y)
		var fan: Sprite2D = _make_region_sprite(
			WINDMILL_FAN_TEXTURE, Rect2(0, 0, 120, 118), top_left, Vector2(120, 118)
		)
		fan.z_index = decoration_depth
		_visual_root.add_child(fan)
		var base := Sprite2D.new()
		base.texture = WINDMILL_BASE_TEXTURE
		base.centered = false
		base.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
		base.position = top_left + Vector2(0, 118)
		base.z_index = decoration_depth
		_visual_root.add_child(base)

func _remove_cell_sprite(cell: Vector2i) -> void:
	if not _cell_sprites.has(cell):
		return
	var sprite: Node = _cell_sprites[cell] as Node
	if is_instance_valid(sprite):
		sprite.queue_free()
	_cell_sprites.erase(cell)

func _make_region_sprite(
		texture: Texture2D,
		region: Rect2,
		top_left: Vector2,
		draw_size: Vector2
	) -> Sprite2D:
	var sprite := Sprite2D.new()
	sprite.texture = texture
	sprite.region_enabled = true
	sprite.region_rect = region
	sprite.centered = false
	sprite.position = top_left
	sprite.scale = draw_size / region.size
	sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	return sprite
