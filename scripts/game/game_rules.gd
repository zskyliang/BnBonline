class_name GameRules
extends RefCounted
## Deterministic rules shared by gameplay and the headless tests.

const BODY_HALF_WIDTH: float = 11.5
const BODY_TOP_OFFSET: float = -7.5
const BODY_BOTTOM_OFFSET: float = 11.5
const RIGID_CENTER_CLEARANCE: float = 20.0
const BLAST_DIRECTIONS: Array[Vector2i] = [
	Vector2i.RIGHT, Vector2i.LEFT, Vector2i.DOWN, Vector2i.UP,
]

static func is_destructible(code: int) -> bool:
	return code == 3 or code == 8

static func is_rigid(code: int) -> bool:
	return code > 0 and code < 100 and not is_destructible(code)

static func is_walkable(code: int) -> bool:
	return code == 0 or code >= 101

static func blast_cells(
		center: Vector2i,
		power: int,
		cells: Array[PackedInt32Array]
	) -> Array[Vector2i]:
	var result: Array[Vector2i] = [center]
	for direction: Vector2i in BLAST_DIRECTIONS:
		for distance: int in range(1, power + 1):
			var cell: Vector2i = center + direction * distance
			if not GameConstants.is_inside(cell):
				break
			var code: int = cells[cell.y][cell.x]
			if is_rigid(code):
				break
			result.append(cell)
			if is_destructible(code):
				break
	return result

static func foot_cells(world_position: Vector2) -> Array[Vector2i]:
	return [
		GameConstants.world_to_grid(world_position + Vector2(-12.0, 8.0)),
		GameConstants.world_to_grid(world_position + Vector2(12.0, 8.0)),
	]

static func body_cells(world_position: Vector2) -> Array[Vector2i]:
	return [
		GameConstants.world_to_grid(world_position + Vector2(-BODY_HALF_WIDTH, BODY_TOP_OFFSET)),
		GameConstants.world_to_grid(world_position + Vector2(BODY_HALF_WIDTH, BODY_TOP_OFFSET)),
		GameConstants.world_to_grid(world_position + Vector2(-BODY_HALF_WIDTH, BODY_BOTTOM_OFFSET)),
		GameConstants.world_to_grid(world_position + Vector2(BODY_HALF_WIDTH, BODY_BOTTOM_OFFSET)),
	]

static func both_feet_unsafe(world_position: Vector2, unsafe_cells: Dictionary) -> bool:
	var feet: Array[Vector2i] = foot_cells(world_position)
	return unsafe_cells.has(feet[0]) and unsafe_cells.has(feet[1])
