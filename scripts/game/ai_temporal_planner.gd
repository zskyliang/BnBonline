class_name AITemporalPlanner
extends RefCounted
## Time-expanded grid search over (cell, arrival tick), including wait actions.

const WAIT_STEP_MS: int = 150
const SAFETY_MARGIN_MS: int = 100
const SAFE_TAIL_MS: int = 450
const CELL_COUNT: int = GameConstants.GRID_COLUMNS * GameConstants.GRID_ROWS
const CARDINAL_DIRECTIONS: Array[Vector2i] = [
	Vector2i.UP, Vector2i.DOWN, Vector2i.LEFT, Vector2i.RIGHT,
]


class TimedPlan extends RefCounted:
	var cells: Array[Vector2i] = []
	var arrival_ms: PackedInt32Array = PackedInt32Array()
	var valid: bool = false

	func travel_ms() -> int:
		return arrival_ms[-1] if not arrival_ms.is_empty() else 0

	func target_cell() -> Vector2i:
		return cells[-1] if not cells.is_empty() else Vector2i(-1, -1)


static func find_path(
		snapshot: AIBattleSnapshot,
		forecast: AIHazardForecast,
		start: Vector2i,
		goal: Vector2i,
		move_speed: float,
		max_time_ms: int = 8000,
		safe_tail_ms: int = SAFE_TAIL_MS,
		start_time_ms: int = 0
	) -> TimedPlan:
	if not _is_walkable(snapshot, start) or not _is_walkable(snapshot, goal):
		return TimedPlan.new()
	return _search(snapshot, forecast, start, move_speed, max_time_ms, start_time_ms, func(cell: Vector2i, time_ms: int) -> bool:
		return cell == goal and not forecast.is_unsafe(
			cell, time_ms, time_ms + safe_tail_ms, SAFETY_MARGIN_MS
		)
	)


static func find_escape_plan(
		snapshot: AIBattleSnapshot,
		forecast: AIHazardForecast,
		start: Vector2i,
		move_speed: float,
		max_time_ms: int = 5000,
		start_time_ms: int = 0
	) -> TimedPlan:
	var stable_until: int = maxi(forecast.latest_danger_end_ms() + SAFETY_MARGIN_MS, SAFE_TAIL_MS)
	return _search(snapshot, forecast, start, move_speed, max_time_ms, start_time_ms, func(cell: Vector2i, time_ms: int) -> bool:
		return not forecast.is_unsafe(cell, time_ms, stable_until, SAFETY_MARGIN_MS)
	)


static func find_direct_escape_plan(
		snapshot: AIBattleSnapshot,
		forecast: AIHazardForecast,
		start: Vector2i,
		move_speed: float,
		max_time_ms: int = 5000,
		start_time_ms: int = 0
	) -> TimedPlan:
	var empty_plan := TimedPlan.new()
	if not _is_walkable(snapshot, start):
		return empty_plan
	var move_ticks: int = maxi(1, ceili(_move_duration_ms(move_speed) / float(WAIT_STEP_MS)))
	var move_ms: int = move_ticks * WAIT_STEP_MS
	var max_steps: int = maxi(0, max_time_ms / move_ms)
	var stable_until_ms: int = maxi(
		forecast.latest_danger_end_ms() + SAFETY_MARGIN_MS,
		start_time_ms + SAFE_TAIL_MS
	)
	var queue: Array[Vector2i] = [start]
	var queue_index: int = 0
	var steps_by_cell: Dictionary = {start: 0}
	var parents: Dictionary = {start: Vector2i(-1, -1)}
	while queue_index < queue.size():
		var cell: Vector2i = queue[queue_index]
		queue_index += 1
		var steps: int = int(steps_by_cell[cell])
		var arrival_ms: int = steps * move_ms
		if steps > 0 and not forecast.is_unsafe(
				cell,
				start_time_ms + arrival_ms,
				stable_until_ms,
				SAFETY_MARGIN_MS
			):
			return _reconstruct_direct_plan(cell, parents, steps_by_cell, move_ms)
		if steps >= max_steps:
			continue
		for direction: Vector2i in CARDINAL_DIRECTIONS:
			var neighbor: Vector2i = cell + direction
			if steps_by_cell.has(neighbor) or not _is_walkable(snapshot, neighbor):
				continue
			var next_arrival_ms: int = arrival_ms + move_ms
			if forecast.is_unsafe(
					cell,
					start_time_ms + arrival_ms,
					start_time_ms + next_arrival_ms,
					SAFETY_MARGIN_MS
				) or forecast.is_unsafe(
					neighbor,
					start_time_ms + arrival_ms,
					start_time_ms + next_arrival_ms,
					SAFETY_MARGIN_MS
				):
				continue
			if forecast.is_bomb_blocked(
					neighbor,
					start_time_ms + arrival_ms,
					start_time_ms + next_arrival_ms
				):
				continue
			steps_by_cell[neighbor] = steps + 1
			parents[neighbor] = cell
			queue.append(neighbor)
	return empty_plan


static func reachable_cells(
		snapshot: AIBattleSnapshot,
		forecast: AIHazardForecast,
		start: Vector2i,
		move_speed: float,
		max_time_ms: int,
		start_time_ms: int = 0,
		terminal_only: bool = false
	) -> Dictionary:
	var result: Dictionary = {}
	if not _is_walkable(snapshot, start):
		return result
	var max_tick: int = maxi(0, ceili(float(max_time_ms) / WAIT_STEP_MS))
	var move_ticks: int = maxi(1, ceili(_move_duration_ms(move_speed) / float(WAIT_STEP_MS)))
	var buckets: Array = []
	buckets.resize(max_tick + 1)
	for tick: int in range(max_tick + 1):
		buckets[tick] = []
	var visited: Dictionary = {}
	var start_key: int = _state_key(start, 0)
	visited[start_key] = true
	(buckets[0] as Array).append(start)
	for tick: int in range(max_tick + 1):
		var time_ms: int = tick * WAIT_STEP_MS
		for cell: Vector2i in buckets[tick] as Array:
			var absolute_time_ms: int = start_time_ms + time_ms
			if not terminal_only or tick == max_tick:
				result[cell] = mini(int(result.get(cell, absolute_time_ms)), absolute_time_ms)
			_push_reachable_state(
				snapshot, forecast, buckets, visited, cell, cell,
				tick, tick + 1, max_tick, start_time_ms
			)
			for direction: Vector2i in CARDINAL_DIRECTIONS:
				_push_reachable_state(
					snapshot, forecast, buckets, visited, cell, cell + direction,
					tick, tick + move_ticks, max_tick, start_time_ms
				)
	return result


static func reachable_cells_fast(
		snapshot: AIBattleSnapshot,
		forecast: AIHazardForecast,
		start: Vector2i,
		move_speed: float,
		max_time_ms: int
	) -> Dictionary:
	var result: Dictionary = {}
	if not _is_walkable(snapshot, start):
		return result
	var move_ms: int = _move_duration_ms(move_speed)
	var max_steps: int = maxi(0, max_time_ms / maxi(1, move_ms))
	var queue: Array[Vector2i] = [start]
	var queue_index: int = 0
	var distances: Dictionary = {start: 0}
	var bomb_cells: Dictionary = {}
	for bomb: AIBattleSnapshot.BombState in snapshot.bombs:
		bomb_cells[bomb.cell] = true
	while queue_index < queue.size():
		var cell: Vector2i = queue[queue_index]
		queue_index += 1
		var steps: int = int(distances[cell])
		var arrival_ms: int = steps * move_ms
		result[cell] = arrival_ms
		if steps >= max_steps:
			continue
		for direction: Vector2i in CARDINAL_DIRECTIONS:
			var neighbor: Vector2i = cell + direction
			if distances.has(neighbor) or not _is_walkable(snapshot, neighbor):
				continue
			if bomb_cells.has(neighbor) and neighbor != start:
				continue
			distances[neighbor] = steps + 1
			queue.append(neighbor)
	return result


static func _search(
		snapshot: AIBattleSnapshot,
		forecast: AIHazardForecast,
		start: Vector2i,
		move_speed: float,
		max_time_ms: int,
		start_time_ms: int,
		goal_test: Callable
	) -> TimedPlan:
	var max_tick: int = maxi(0, ceili(float(max_time_ms) / WAIT_STEP_MS))
	var move_ticks: int = maxi(1, ceili(_move_duration_ms(move_speed) / float(WAIT_STEP_MS)))
	var buckets: Array = []
	buckets.resize(max_tick + 1)
	for tick: int in range(max_tick + 1):
		buckets[tick] = []
	var parents: Dictionary = {}
	var visited: Dictionary = {}
	var start_key: int = _state_key(start, 0)
	visited[start_key] = true
	parents[start_key] = -1
	(buckets[0] as Array).append(start)
	for tick: int in range(max_tick + 1):
		var time_ms: int = tick * WAIT_STEP_MS
		for cell: Vector2i in buckets[tick] as Array:
			var state_key: int = _state_key(cell, tick)
			if goal_test.call(cell, start_time_ms + time_ms):
				return _reconstruct(state_key, parents)
			_push_search_state(
				snapshot, forecast, buckets, visited, parents,
				cell, cell, tick, tick + 1, max_tick, state_key
				, start_time_ms
			)
			for direction: Vector2i in CARDINAL_DIRECTIONS:
				_push_search_state(
					snapshot, forecast, buckets, visited, parents,
					cell, cell + direction, tick, tick + move_ticks, max_tick, state_key
					, start_time_ms
				)
	return TimedPlan.new()


static func _push_search_state(
		snapshot: AIBattleSnapshot,
		forecast: AIHazardForecast,
		buckets: Array,
		visited: Dictionary,
		parents: Dictionary,
		from_cell: Vector2i,
		to_cell: Vector2i,
		from_tick: int,
		to_tick: int,
		max_tick: int,
		parent_key: int,
		start_time_ms: int
	) -> void:
	if not _transition_is_safe(
			snapshot, forecast, from_cell, to_cell, from_tick, to_tick, max_tick, start_time_ms
		):
		return
	var key: int = _state_key(to_cell, to_tick)
	if visited.has(key):
		return
	visited[key] = true
	parents[key] = parent_key
	(buckets[to_tick] as Array).append(to_cell)


static func _push_reachable_state(
		snapshot: AIBattleSnapshot,
		forecast: AIHazardForecast,
		buckets: Array,
		visited: Dictionary,
		from_cell: Vector2i,
		to_cell: Vector2i,
		from_tick: int,
		to_tick: int,
		max_tick: int,
		start_time_ms: int
	) -> void:
	if not _transition_is_safe(
			snapshot, forecast, from_cell, to_cell, from_tick, to_tick, max_tick, start_time_ms
		):
		return
	var key: int = _state_key(to_cell, to_tick)
	if visited.has(key):
		return
	visited[key] = true
	(buckets[to_tick] as Array).append(to_cell)


static func _transition_is_safe(
		snapshot: AIBattleSnapshot,
		forecast: AIHazardForecast,
		from_cell: Vector2i,
		to_cell: Vector2i,
		from_tick: int,
		to_tick: int,
		max_tick: int,
		start_time_ms: int
	) -> bool:
	if to_tick > max_tick or not _is_walkable(snapshot, to_cell):
		return false
	var from_ms: int = start_time_ms + from_tick * WAIT_STEP_MS
	var to_ms: int = start_time_ms + to_tick * WAIT_STEP_MS
	if forecast.is_unsafe(from_cell, from_ms, to_ms, SAFETY_MARGIN_MS):
		return false
	if forecast.is_unsafe(to_cell, from_ms, to_ms, SAFETY_MARGIN_MS):
		return false
	if to_cell != from_cell and forecast.is_bomb_blocked(to_cell, from_ms, to_ms):
		return false
	return true


static func _reconstruct(last_key: int, parents: Dictionary) -> TimedPlan:
	var reverse_cells: Array[Vector2i] = []
	var reverse_times: PackedInt32Array = PackedInt32Array()
	var key: int = last_key
	while key >= 0:
		var tick: int = key / CELL_COUNT
		var cell_index: int = key % CELL_COUNT
		reverse_cells.append(Vector2i(
			cell_index % GameConstants.GRID_COLUMNS,
			cell_index / GameConstants.GRID_COLUMNS
		))
		reverse_times.append(tick * WAIT_STEP_MS)
		key = int(parents.get(key, -1))
	reverse_cells.reverse()
	reverse_times.reverse()
	var plan := TimedPlan.new()
	plan.cells = reverse_cells
	plan.arrival_ms = reverse_times
	plan.valid = true
	return plan


static func _reconstruct_direct_plan(
		last_cell: Vector2i,
		parents: Dictionary,
		steps_by_cell: Dictionary,
		move_ms: int
	) -> TimedPlan:
	var reverse_cells: Array[Vector2i] = []
	var cell: Vector2i = last_cell
	while cell != Vector2i(-1, -1):
		reverse_cells.append(cell)
		cell = parents.get(cell, Vector2i(-1, -1)) as Vector2i
	reverse_cells.reverse()
	var plan := TimedPlan.new()
	for path_cell: Vector2i in reverse_cells:
		plan.cells.append(path_cell)
		plan.arrival_ms.append(int(steps_by_cell[path_cell]) * move_ms)
	plan.valid = not plan.cells.is_empty()
	return plan


static func _move_duration_ms(move_speed: float) -> int:
	return ceili(GameConstants.CELL_SIZE / maxf(1.0, move_speed) * 1000.0)


static func _state_key(cell: Vector2i, tick: int) -> int:
	return tick * CELL_COUNT + cell.y * GameConstants.GRID_COLUMNS + cell.x


static func _is_walkable(snapshot: AIBattleSnapshot, cell: Vector2i) -> bool:
	return GameConstants.is_inside(cell) and GameRules.is_walkable(snapshot.cells[cell.y][cell.x])
