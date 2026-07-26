class_name AIHazardForecast
extends RefCounted
## Simulates timed blast intervals, chain reactions, and bomb blocking windows.

const NO_DANGER_MS: int = 999999


class BombBlast extends RefCounted:
	var bomb_cell: Vector2i
	var owner_id: int
	var owner_team: int
	var explode_ms: int
	var cells: Array[Vector2i]

	func _init(
			new_bomb_cell: Vector2i,
			new_owner_id: int,
			new_owner_team: int,
			new_explode_ms: int,
			new_cells: Array[Vector2i]
		) -> void:
		bomb_cell = new_bomb_cell
		owner_id = new_owner_id
		owner_team = new_owner_team
		explode_ms = new_explode_ms
		cells = new_cells.duplicate()


var horizon_ms: int = 5000
var unsafe_intervals: Dictionary = {}
var bomb_block_intervals: Dictionary = {}
var blast_events: Array[BombBlast] = []
var destroyed_at_ms: Dictionary = {}
var initial_cells: Array[PackedInt32Array] = []
var predicted_cells: Array[PackedInt32Array] = []
var _latest_danger_end_ms: int = 0
var _time_offset_ms: int = 0


static func build(
		snapshot: AIBattleSnapshot,
		new_horizon_ms: int = 5000,
		virtual_cell: Vector2i = Vector2i(-1, -1),
		virtual_power: int = 0,
		virtual_place_ms: int = 0,
		virtual_fuse_ms: int = 0,
		virtual_owner_id: int = 0,
		virtual_owner_team: int = PaintPalette.TEAM_NEUTRAL
	) -> AIHazardForecast:
	var forecast := AIHazardForecast.new()
	forecast.horizon_ms = maxi(0, new_horizon_ms)
	forecast.initial_cells = snapshot.clone_cells()
	forecast.predicted_cells = snapshot.clone_cells()
	for explosion: AIBattleSnapshot.ExplosionState in snapshot.explosions:
		for cell: Vector2i in explosion.cells:
			forecast._add_unsafe_interval(cell, 0, mini(forecast.horizon_ms, explosion.remaining_ms))
	var simulated_bombs: Array[Dictionary] = []
	for bomb: AIBattleSnapshot.BombState in snapshot.bombs:
		simulated_bombs.append({
			"cell": bomb.cell,
			"power": bomb.power,
			"start_ms": 0,
			"explode_ms": bomb.remaining_ms,
			"actual_explode_ms": -1,
			"serial": bomb.serial,
			"owner_id": bomb.owner_id,
			"owner_team": bomb.owner_team,
			"exploded": false,
		})
	if GameConstants.is_inside(virtual_cell) and virtual_power > 0 and virtual_fuse_ms > 0:
		simulated_bombs.append({
			"cell": virtual_cell,
			"power": virtual_power,
			"start_ms": maxi(0, virtual_place_ms),
			"explode_ms": maxi(0, virtual_place_ms) + virtual_fuse_ms,
			"actual_explode_ms": -1,
			"serial": 1000000,
			"owner_id": virtual_owner_id,
			"owner_team": virtual_owner_team,
			"exploded": false,
		})
	forecast._simulate_bombs(simulated_bombs)
	forecast._record_bomb_blocks(simulated_bombs)
	forecast._merge_intervals()
	return forecast


func danger_eta_ms(cell: Vector2i) -> int:
	if not unsafe_intervals.has(cell):
		return NO_DANGER_MS
	var best: int = NO_DANGER_MS
	for interval: Vector2i in unsafe_intervals[cell] as Array:
		if interval.y < _time_offset_ms:
			continue
		if interval.x <= _time_offset_ms:
			return 0
		best = mini(best, interval.x - _time_offset_ms)
	return best


func is_unsafe(cell: Vector2i, from_ms: int, to_ms: int, margin_ms: int = 0) -> bool:
	if not unsafe_intervals.has(cell):
		return false
	var interval_start: int = mini(from_ms, to_ms) + _time_offset_ms
	var interval_end: int = maxi(from_ms, to_ms) + _time_offset_ms
	for unsafe: Vector2i in unsafe_intervals[cell] as Array:
		if interval_start <= unsafe.y + margin_ms and interval_end >= unsafe.x - margin_ms:
			return true
	return false


func is_bomb_blocked(cell: Vector2i, from_ms: int, to_ms: int) -> bool:
	if not bomb_block_intervals.has(cell):
		return false
	var interval_start: int = mini(from_ms, to_ms) + _time_offset_ms
	var interval_end: int = maxi(from_ms, to_ms) + _time_offset_ms
	for blocked: Vector2i in bomb_block_intervals[cell] as Array:
		if interval_start < blocked.y and interval_end > blocked.x:
			return true
	return false


func latest_danger_end_ms() -> int:
	return maxi(0, _latest_danger_end_ms - _time_offset_ms)


func is_predicted_walkable(cell: Vector2i, at_ms: int) -> bool:
	if not GameConstants.is_inside(cell):
		return false
	var initial_code: int = initial_cells[cell.y][cell.x]
	if GameRules.is_walkable(initial_code):
		return true
	return GameRules.is_destructible(initial_code) \
		and destroyed_at_ms.has(cell) \
		and int(destroyed_at_ms[cell]) <= at_ms + _time_offset_ms


func set_time_offset_ms(offset_ms: int) -> void:
	_time_offset_ms = maxi(0, offset_ms)


func blast_time_ms(blast_event: BombBlast) -> int:
	return maxi(0, blast_event.explode_ms - _time_offset_ms)


func _simulate_bombs(simulated_bombs: Array[Dictionary]) -> void:
	while true:
		var event_ms: int = NO_DANGER_MS
		for bomb: Dictionary in simulated_bombs:
			if bool(bomb["exploded"]):
				continue
			event_ms = mini(event_ms, int(bomb["explode_ms"]))
		if event_ms == NO_DANGER_MS or event_ms > horizon_ms:
			break
		var queue: Array[Dictionary] = []
		for bomb: Dictionary in simulated_bombs:
			if not bool(bomb["exploded"]) and int(bomb["explode_ms"]) == event_ms:
				queue.append(bomb)
		queue.sort_custom(func(left: Dictionary, right: Dictionary) -> bool:
			return int(left["serial"]) < int(right["serial"])
		)
		var queue_index: int = 0
		while queue_index < queue.size():
			var exploding: Dictionary = queue[queue_index]
			queue_index += 1
			if bool(exploding["exploded"]) or int(exploding["start_ms"]) > event_ms:
				continue
			exploding["exploded"] = true
			exploding["actual_explode_ms"] = event_ms
			var blast: Array[Vector2i] = GameRules.blast_cells(
				exploding["cell"] as Vector2i,
				int(exploding["power"]),
				predicted_cells
			)
			blast_events.append(BombBlast.new(
				exploding["cell"] as Vector2i,
				int(exploding["owner_id"]),
				int(exploding["owner_team"]),
				event_ms,
				blast
			))
			for cell: Vector2i in blast:
				_add_unsafe_interval(
					cell,
					event_ms,
					mini(horizon_ms, event_ms + int(GameConstants.EXPLOSION_SECONDS * 1000.0))
				)
			for chained: Dictionary in simulated_bombs:
				if bool(chained["exploded"]) or int(chained["start_ms"]) > event_ms:
					continue
				if chained["cell"] as Vector2i in blast:
					chained["explode_ms"] = event_ms
					if chained not in queue:
						queue.append(chained)
			for cell: Vector2i in blast:
				var code: int = predicted_cells[cell.y][cell.x]
				if GameRules.is_destructible(code) or code >= 101:
					if GameRules.is_destructible(code) and not destroyed_at_ms.has(cell):
						destroyed_at_ms[cell] = event_ms
					predicted_cells[cell.y][cell.x] = 0


func _record_bomb_blocks(simulated_bombs: Array[Dictionary]) -> void:
	for bomb: Dictionary in simulated_bombs:
		var end_ms: int = int(bomb["actual_explode_ms"])
		if end_ms < 0:
			end_ms = int(bomb["explode_ms"])
		_add_bomb_block(
			bomb["cell"] as Vector2i,
			int(bomb["start_ms"]),
			mini(horizon_ms + 1, end_ms)
		)


func _add_unsafe_interval(cell: Vector2i, start_ms: int, end_ms: int) -> void:
	if end_ms < start_ms:
		return
	_latest_danger_end_ms = maxi(_latest_danger_end_ms, end_ms)
	if not unsafe_intervals.has(cell):
		unsafe_intervals[cell] = []
	var intervals: Array = unsafe_intervals[cell] as Array
	intervals.append(Vector2i(start_ms, end_ms))


func _add_bomb_block(cell: Vector2i, start_ms: int, end_ms: int) -> void:
	if end_ms <= start_ms:
		return
	if not bomb_block_intervals.has(cell):
		bomb_block_intervals[cell] = []
	var intervals: Array = bomb_block_intervals[cell] as Array
	intervals.append(Vector2i(start_ms, end_ms))


func _merge_intervals() -> void:
	for cell: Vector2i in unsafe_intervals.keys():
		unsafe_intervals[cell] = _merged(unsafe_intervals[cell] as Array)
	for cell: Vector2i in bomb_block_intervals.keys():
		bomb_block_intervals[cell] = _merged(bomb_block_intervals[cell] as Array)


func _merged(source: Array) -> Array[Vector2i]:
	var sorted: Array[Vector2i] = []
	for interval: Vector2i in source:
		sorted.append(interval)
	sorted.sort_custom(func(left: Vector2i, right: Vector2i) -> bool:
		return left.x < right.x
	)
	var result: Array[Vector2i] = []
	for interval: Vector2i in sorted:
		if result.is_empty() or interval.x > result[-1].y:
			result.append(interval)
		else:
			var previous: Vector2i = result[-1]
			previous.y = maxi(previous.y, interval.y)
			result[-1] = previous
	return result
