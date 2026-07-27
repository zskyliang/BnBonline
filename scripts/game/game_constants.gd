class_name GameConstants
extends RefCounted
## Shared gameplay constants and grid conversion helpers.

const GRID_COLUMNS: int = 15
const GRID_ROWS: int = 13
const CELL_SIZE: float = 40.0
const GRID_ORIGIN: Vector2 = Vector2(20.0, 40.0)
const GAME_VIEW_SIZE: Vector2 = Vector2(800.0, 600.0)

const BASE_MOVE_SPEED: float = 100.0
const SPEED_PER_POINT: float = 25.0
const DEFAULT_INITIAL_SPEED_POINTS: int = 2
const DEFAULT_INITIAL_BUBBLE_POINTS: int = 2
const DEFAULT_INITIAL_POWER_POINTS: int = 2
const INITIAL_SPEED: float = 150.0
const INITIAL_BUBBLES: int = 2
const INITIAL_POWER: int = 2
const MAX_SPEED: float = 300.0
const MAX_SPEED_POINTS: int = 8
const MAX_BUBBLES: int = 10
const MAX_POWER: int = 10
const SPEED_PER_SKILL_POINT: float = SPEED_PER_POINT
const SPEED_PER_STAGE_ITEM: float = SPEED_PER_POINT
const MAX_SPEED_SKILL_POINTS: int = 7
const MAX_BUBBLE_SKILL_POINTS: int = 9
const MAX_POWER_SKILL_POINTS: int = 9
const ITEM_SPAWN_INTERVAL_SECONDS: float = 10.0
const ITEMS_PER_SPAWN: int = 3
const BUBBLE_FUSE_SECONDS: float = 3.0
const EXPLOSION_SECONDS: float = 0.45
const TRAP_SECONDS: float = 10.0
const RESPAWN_SECONDS: float = 2.4
const RESPAWN_INVINCIBLE_SECONDS: float = 1.0
const ROUND_SECONDS: float = 120.0
const AI_THINK_SECONDS: float = 0.15


static func speed_from_points(points: int) -> float:
	return clampf(
		BASE_MOVE_SPEED + float(maxi(0, points)) * SPEED_PER_POINT,
		0.0,
		MAX_SPEED
	)


static func speed_points_from_pixels(speed: float) -> int:
	return clampi(
		roundi((speed - BASE_MOVE_SPEED) / SPEED_PER_POINT),
		0,
		MAX_SPEED_POINTS
	)


static func grid_to_world(cell: Vector2i) -> Vector2:
	return GRID_ORIGIN + Vector2(cell) * CELL_SIZE + Vector2.ONE * CELL_SIZE * 0.5


static func logic_to_world_3d(logic_position: Vector2, height: float = 0.0) -> Vector3:
	var first_cell_center := GRID_ORIGIN + Vector2.ONE * CELL_SIZE * 0.5
	var cell_position := (logic_position - first_cell_center) / CELL_SIZE
	return Vector3(
		cell_position.x - float(GRID_COLUMNS - 1) * 0.5,
		height,
		cell_position.y - float(GRID_ROWS - 1) * 0.5
	)


static func grid_to_world_3d(cell: Vector2i, height: float = 0.0) -> Vector3:
	return Vector3(
		float(cell.x) - float(GRID_COLUMNS - 1) * 0.5,
		height,
		float(cell.y) - float(GRID_ROWS - 1) * 0.5
	)

static func grid_to_top_left(cell: Vector2i) -> Vector2:
	return GRID_ORIGIN + Vector2(cell) * CELL_SIZE

static func world_to_grid(world_position: Vector2) -> Vector2i:
	var local_position: Vector2 = world_position - GRID_ORIGIN
	return Vector2i(floori(local_position.x / CELL_SIZE), floori(local_position.y / CELL_SIZE))

static func is_inside(cell: Vector2i) -> bool:
	return cell.x >= 0 and cell.y >= 0 and cell.x < GRID_COLUMNS and cell.y < GRID_ROWS

static func is_actor_center_inside_arena(world_position: Vector2) -> bool:
	var first_center: Vector2 = grid_to_world(Vector2i.ZERO)
	var last_center: Vector2 = grid_to_world(Vector2i(GRID_COLUMNS - 1, GRID_ROWS - 1))
	return world_position.x >= first_center.x \
		and world_position.x <= last_center.x \
		and world_position.y >= first_center.y \
		and world_position.y <= last_center.y
