class_name AIBattleSnapshot
extends RefCounted
## Immutable, renderer-free view of the arena used by rule AI and headless tests.


class BombState extends RefCounted:
	var cell: Vector2i
	var power: int
	var remaining_ms: int
	var owner_id: int
	var serial: int
	var owner_team: int

	func _init(
			new_cell: Vector2i,
			new_power: int,
			new_remaining_ms: int,
			new_owner_id: int = 0,
			new_serial: int = 0,
			new_owner_team: int = PaintPalette.TEAM_NEUTRAL
		) -> void:
		cell = new_cell
		power = new_power
		remaining_ms = maxi(0, new_remaining_ms)
		owner_id = new_owner_id
		serial = new_serial
		owner_team = new_owner_team


class ExplosionState extends RefCounted:
	var cells: Array[Vector2i]
	var remaining_ms: int
	var attacker_id: int
	var attacker_team: int

	func _init(
			new_cells: Array[Vector2i],
			new_remaining_ms: int,
			new_attacker_id: int = 0,
			new_attacker_team: int = PaintPalette.TEAM_NEUTRAL
		) -> void:
		cells = new_cells.duplicate()
		remaining_ms = maxi(0, new_remaining_ms)
		attacker_id = new_attacker_id
		attacker_team = new_attacker_team


class ItemState extends RefCounted:
	var item_id: int
	var item_type: int
	var cell: Vector2i
	var spawned_ms: int

	func _init(
			new_item_id: int,
			new_item_type: int,
			new_cell: Vector2i,
			new_spawned_ms: int = 0
		) -> void:
		item_id = new_item_id
		item_type = new_item_type
		cell = new_cell
		spawned_ms = maxi(0, new_spawned_ms)


class ActorState extends RefCounted:
	var instance_id: int
	var team_id: int
	var cell: Vector2i
	var world_position: Vector2
	var move_speed: float
	var bubble_capacity: int
	var active_bubbles: int
	var power: int
	var is_dead: bool
	var is_trapped: bool
	var is_player: bool
	var stage_speed_items: int
	var stage_bubble_items: int
	var stage_power_items: int

	func _init(
			new_instance_id: int,
			new_team_id: int,
			new_cell: Vector2i,
			new_world_position: Vector2,
			new_move_speed: float,
			new_bubble_capacity: int,
			new_active_bubbles: int,
			new_power: int,
			new_is_dead: bool,
			new_is_trapped: bool,
			new_is_player: bool,
			new_stage_speed_items: int = 0,
			new_stage_bubble_items: int = 0,
			new_stage_power_items: int = 0
		) -> void:
		instance_id = new_instance_id
		team_id = new_team_id
		cell = new_cell
		world_position = new_world_position
		move_speed = new_move_speed
		bubble_capacity = new_bubble_capacity
		active_bubbles = new_active_bubbles
		power = new_power
		is_dead = new_is_dead
		is_trapped = new_is_trapped
		is_player = new_is_player
		stage_speed_items = maxi(0, new_stage_speed_items)
		stage_bubble_items = maxi(0, new_stage_bubble_items)
		stage_power_items = maxi(0, new_stage_power_items)


var cells: Array[PackedInt32Array] = []
var paint_owners: Array[PackedInt32Array] = []
var locked_cells: Array[PackedByteArray] = []
var bombs: Array[BombState] = []
var explosions: Array[ExplosionState] = []
var items: Array[ItemState] = []
var actors: Array[ActorState] = []
var remaining_round_ms: int = 0


func paint_owner(cell: Vector2i) -> int:
	if not GameConstants.is_inside(cell) or paint_owners.is_empty():
		return PaintPalette.TEAM_NEUTRAL
	return paint_owners[cell.y][cell.x]


func is_paint_locked(cell: Vector2i) -> bool:
	return GameConstants.is_inside(cell) \
		and not locked_cells.is_empty() \
		and locked_cells[cell.y][cell.x] != 0


func territory_swing(target_cells: Array[Vector2i], owner_team: int) -> int:
	var result: int = 0
	var visited: Dictionary = {}
	for cell: Vector2i in target_cells:
		if not GameConstants.is_inside(cell) or is_paint_locked(cell) or visited.has(cell):
			continue
		visited[cell] = true
		var current_owner: int = paint_owner(cell)
		if current_owner == PaintPalette.TEAM_NEUTRAL:
			result += 1
		elif current_owner != owner_team:
			result += 2
	return result


func actor_by_id(instance_id: int) -> ActorState:
	for actor_state: ActorState in actors:
		if actor_state.instance_id == instance_id:
			return actor_state
	return null


func item_by_id(item_id: int) -> ItemState:
	for item: ItemState in items:
		if item.item_id == item_id:
			return item
	return null


func clone_cells() -> Array[PackedInt32Array]:
	var result: Array[PackedInt32Array] = []
	for row: PackedInt32Array in cells:
		result.append(row.duplicate())
	return result
