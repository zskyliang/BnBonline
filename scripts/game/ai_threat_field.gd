class_name AIThreatField
extends RefCounted
## Weighted spatial pressure produced by one actor's predicted bubble blasts.

const MAX_RING_DISTANCE: int = 3
const RING_WEIGHTS: Array[float] = [1.0, 0.5, 0.25, 0.125]

var weights: Dictionary = {}
var total_weight: float = 0.0
var source_bomb_count: int = 0


static func build(forecast: AIHazardForecast, owner_id: int) -> AIThreatField:
	var field := AIThreatField.new()
	for blast_event: AIHazardForecast.BombBlast in forecast.blast_events:
		if blast_event.owner_id != owner_id:
			continue
		field._add_bomb_blast(blast_event, forecast)
		field.source_bomb_count += 1
	field._recalculate_total()
	return field


static func build_for_blast(
		forecast: AIHazardForecast,
		blast_event: AIHazardForecast.BombBlast
	) -> AIThreatField:
	var field := AIThreatField.new()
	field._add_bomb_blast(blast_event, forecast)
	field.source_bomb_count = 1
	field._recalculate_total()
	return field


static func estimate_blast_weight(
		blast_cells: Array[Vector2i],
		cells: Array[PackedInt32Array],
		relevant_cells: Dictionary = {}
	) -> float:
	var bomb_weights: Dictionary = _spread_from_blast(
		blast_cells,
		func(cell: Vector2i) -> bool:
			return GameConstants.is_inside(cell) and GameRules.is_walkable(cells[cell.y][cell.x])
	)
	return _sum_relevant(bomb_weights, relevant_cells)


static func estimate_marginal_blast_weight(
		blast_cells: Array[Vector2i],
		cells: Array[PackedInt32Array],
		previous: AIThreatField,
		relevant_cells: Dictionary
	) -> float:
	var bomb_weights: Dictionary = _spread_from_blast(
		blast_cells,
		func(cell: Vector2i) -> bool:
			return GameConstants.is_inside(cell) and GameRules.is_walkable(cells[cell.y][cell.x])
	)
	var marginal: float = 0.0
	for cell: Vector2i in relevant_cells.keys():
		var previous_weight: float = previous.weight_at(cell)
		var added_weight: float = float(bomb_weights.get(cell, 0.0))
		marginal += (1.0 - previous_weight) * added_weight
	return marginal


func weight_at(cell: Vector2i) -> float:
	return float(weights.get(cell, 0.0))


func weight_sum(relevant_cells: Dictionary = {}) -> float:
	return _sum_relevant(weights, relevant_cells)


func coverage_ratio(relevant_cells: Dictionary) -> float:
	if relevant_cells.is_empty():
		return 0.0
	return clampf(weight_sum(relevant_cells) / float(relevant_cells.size()), 0.0, 1.0)


func marginal_weight(previous: AIThreatField, relevant_cells: Dictionary = {}) -> float:
	return maxf(0.0, weight_sum(relevant_cells) - previous.weight_sum(relevant_cells))


func _add_bomb_blast(
		blast_event: AIHazardForecast.BombBlast,
		forecast: AIHazardForecast
	) -> void:
	var bomb_weights: Dictionary = _spread_from_blast(
		blast_event.cells,
		func(cell: Vector2i) -> bool:
			return forecast.is_predicted_walkable(cell, blast_event.explode_ms)
	)
	for cell: Vector2i in bomb_weights.keys():
		var old_weight: float = float(weights.get(cell, 0.0))
		var added_weight: float = float(bomb_weights[cell])
		weights[cell] = 1.0 - (1.0 - old_weight) * (1.0 - added_weight)


func _recalculate_total() -> void:
	total_weight = 0.0
	for value: Variant in weights.values():
		total_weight += float(value)


static func _spread_from_blast(blast_cells: Array[Vector2i], can_expand: Callable) -> Dictionary:
	var distances: Dictionary = {}
	var queue: Array[Vector2i] = []
	var queue_index: int = 0
	for cell: Vector2i in blast_cells:
		if not GameConstants.is_inside(cell) or distances.has(cell):
			continue
		distances[cell] = 0
		queue.append(cell)
	while queue_index < queue.size():
		var cell: Vector2i = queue[queue_index]
		queue_index += 1
		var distance: int = int(distances[cell])
		if distance >= MAX_RING_DISTANCE:
			continue
		for direction: Vector2i in AITemporalPlanner.CARDINAL_DIRECTIONS:
			var neighbor: Vector2i = cell + direction
			if distances.has(neighbor) or not can_expand.call(neighbor):
				continue
			distances[neighbor] = distance + 1
			queue.append(neighbor)
	var result: Dictionary = {}
	for cell: Vector2i in distances.keys():
		result[cell] = RING_WEIGHTS[int(distances[cell])]
	return result


static func _sum_relevant(source: Dictionary, relevant_cells: Dictionary) -> float:
	var total: float = 0.0
	if relevant_cells.is_empty():
		for value: Variant in source.values():
			total += float(value)
		return total
	for cell: Vector2i in relevant_cells.keys():
		total += float(source.get(cell, 0.0))
	return total
