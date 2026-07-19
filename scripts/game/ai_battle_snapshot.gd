class_name AIBattleSnapshot
extends RefCounted
## Immutable, renderer-free view of the arena used by rule AI and headless tests.


class BombState extends RefCounted:
	var cell: Vector2i
	var power: int
	var remaining_ms: int
	var owner_id: int
	var serial: int

	func _init(
			new_cell: Vector2i,
			new_power: int,
			new_remaining_ms: int,
			new_owner_id: int = 0,
			new_serial: int = 0
		) -> void:
		cell = new_cell
		power = new_power
		remaining_ms = maxi(0, new_remaining_ms)
		owner_id = new_owner_id
		serial = new_serial


class ExplosionState extends RefCounted:
	var cells: Array[Vector2i]
	var remaining_ms: int

	func _init(new_cells: Array[Vector2i], new_remaining_ms: int) -> void:
		cells = new_cells.duplicate()
		remaining_ms = maxi(0, new_remaining_ms)


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
			new_is_player: bool
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


var cells: Array[PackedInt32Array] = []
var bombs: Array[BombState] = []
var explosions: Array[ExplosionState] = []
var actors: Array[ActorState] = []
var item_cells: Array[Vector2i] = []


func actor_by_id(instance_id: int) -> ActorState:
	for actor_state: ActorState in actors:
		if actor_state.instance_id == instance_id:
			return actor_state
	return null


func clone_cells() -> Array[PackedInt32Array]:
	var result: Array[PackedInt32Array] = []
	for row: PackedInt32Array in cells:
		result.append(row.duplicate())
	return result

