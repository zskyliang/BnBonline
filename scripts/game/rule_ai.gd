class_name RuleAI
extends Node
## Godot-native rule AI using AStarGrid2D paths and a 150 ms priority loop.

var actor: GameActor
var board: GameBoard
var match_controller: Node

var _think_timer: Timer
var _path: Array[Vector2i] = []
var _rng := RandomNumberGenerator.new()
var _last_bomb_ms: int = -10000

func setup(new_actor: GameActor, new_board: GameBoard, new_match_controller: Node) -> void:
	actor = new_actor
	board = new_board
	match_controller = new_match_controller
	_rng.seed = hash(actor.actor_name) + Time.get_ticks_msec()
	_think_timer = Timer.new()
	_think_timer.wait_time = GameConstants.AI_THINK_SECONDS
	_think_timer.timeout.connect(_think)
	add_child(_think_timer)
	_think_timer.start()
	_think()

func _process(_delta: float) -> void:
	if not is_instance_valid(actor) or actor.stats.is_dead or actor.stats.is_trapped:
		if is_instance_valid(actor):
			actor.set_ai_direction(Vector2.ZERO)
		return
	_follow_path()

func _think() -> void:
	if not is_instance_valid(actor) or actor.stats.is_dead or actor.stats.is_trapped:
		_path.clear()
		return
	var current: Vector2i = actor.current_cell()
	if match_controller.danger_eta_ms(current) <= 950 or board.bombs.has(current):
		_retreat(current)
		return
	var interaction_target: GameActor = _find_trapped_interaction_target()
	if is_instance_valid(interaction_target):
		_move_to(interaction_target.current_cell())
		return
	var item_target: Vector2i = _find_best_item(current)
	if item_target.x >= 0:
		_move_to(item_target)
		return
	var player: GameActor = match_controller.get_player()
	if is_instance_valid(player) and not player.stats.is_dead:
		if _can_blast_target_from(current, player.current_cell()) and _can_drop_safely(current):
			_drop_and_retreat(current)
			return
		var attack_cell: Vector2i = _find_attack_cell(current, player.current_cell())
		if attack_cell.x >= 0:
			_move_to(attack_cell)
			return
	if _has_adjacent_box(current) and _can_drop_safely(current):
		_drop_and_retreat(current)
		return
	_patrol(current)

func _follow_path() -> void:
	if _path.is_empty():
		actor.set_ai_direction(Vector2.ZERO)
		return
	var next_cell: Vector2i = _path[0]
	var target: Vector2 = GameConstants.grid_to_world(next_cell)
	var difference: Vector2 = target - actor.position
	if difference.length() < 4.0:
		actor.position = target
		_path.pop_front()
		_follow_path()
		return
	actor.set_ai_direction(difference)

func _move_to(target: Vector2i) -> bool:
	var current: Vector2i = actor.current_cell()
	var new_path: Array[Vector2i] = board.find_path(current, target, actor)
	if new_path.size() <= 1:
		return false
	new_path.pop_front()
	_path = new_path
	return true

func _retreat(current: Vector2i) -> void:
	var safe_cell: Vector2i = _find_safe_cell(current)
	if safe_cell.x >= 0:
		_move_to(safe_cell)
	else:
		actor.set_ai_direction(Vector2.ZERO)

func _drop_and_retreat(current: Vector2i) -> void:
	_last_bomb_ms = Time.get_ticks_msec()
	actor.request_ai_bomb()
	_retreat(current)

func _can_drop_safely(current: Vector2i) -> bool:
	if Time.get_ticks_msec() - _last_bomb_ms < 700:
		return false
	if actor.stats.active_bubbles >= actor.stats.bubble_capacity or not board.can_place_bubble(current):
		return false
	var predicted: Array[Vector2i] = board.predicted_blast(current, actor.stats.power)
	for candidate: Vector2i in board.get_open_cells():
		if candidate in predicted:
			continue
		var path: Array[Vector2i] = board.find_path(current, candidate, actor)
		if path.size() >= 2 and path.size() <= 6:
			return true
	return false

func _find_safe_cell(current: Vector2i) -> Vector2i:
	var best := Vector2i(-1, -1)
	var best_cost: int = 99999
	for candidate: Vector2i in board.get_open_cells():
		var path: Array[Vector2i] = board.find_path(current, candidate, actor)
		if path.is_empty():
			continue
		var travel_ms: int = int(maxi(0, path.size() - 1) * GameConstants.CELL_SIZE / actor.stats.move_speed * 1000.0)
		if match_controller.danger_eta_ms(candidate) <= travel_ms + 750:
			continue
		if path.size() < best_cost:
			best = candidate
			best_cost = path.size()
	return best

func _find_trapped_interaction_target() -> GameActor:
	var best: GameActor
	var best_distance: int = 99999
	for other: GameActor in match_controller.get_actors():
		if other == actor or other.stats.is_dead or not other.stats.is_trapped:
			continue
		if other.team_id != actor.team_id and not other.is_player:
			continue
		var distance: int = _manhattan(actor.current_cell(), other.current_cell())
		if distance < best_distance:
			best = other
			best_distance = distance
	return best

func _find_best_item(current: Vector2i) -> Vector2i:
	if actor.stats.bubble_capacity >= actor.settings.max_bubbles \
			and actor.stats.move_speed >= actor.settings.max_speed \
			and actor.stats.power >= actor.settings.max_power:
		return Vector2i(-1, -1)
	var best := Vector2i(-1, -1)
	var best_length: int = 99999
	for item: Vector2i in board.get_item_cells():
		if match_controller.danger_eta_ms(item) <= 900:
			continue
		var path: Array[Vector2i] = board.find_path(current, item, actor)
		if not path.is_empty() and path.size() < best_length:
			best = item
			best_length = path.size()
	return best

func _find_attack_cell(current: Vector2i, target: Vector2i) -> Vector2i:
	var best := Vector2i(-1, -1)
	var best_length: int = 99999
	for candidate: Vector2i in board.get_open_cells():
		if not _can_blast_target_from(candidate, target):
			continue
		if match_controller.danger_eta_ms(candidate) <= 1000:
			continue
		var path: Array[Vector2i] = board.find_path(current, candidate, actor)
		if not path.is_empty() and path.size() < best_length:
			best = candidate
			best_length = path.size()
	return best

func _can_blast_target_from(origin: Vector2i, target: Vector2i) -> bool:
	return target in board.predicted_blast(origin, actor.stats.power)

func _has_adjacent_box(cell: Vector2i) -> bool:
	for direction: Vector2i in [Vector2i.UP, Vector2i.DOWN, Vector2i.LEFT, Vector2i.RIGHT]:
		if GameRules.is_destructible(board.cell_code(cell + direction)):
			return true
	return false

func _patrol(current: Vector2i) -> void:
	var candidates: Array[Vector2i] = board.get_open_cells()
	if candidates.is_empty():
		return
	for _attempt: int in range(12):
		var candidate: Vector2i = candidates[_rng.randi_range(0, candidates.size() - 1)]
		if _manhattan(current, candidate) >= 3 and match_controller.danger_eta_ms(candidate) > 1200:
			if _move_to(candidate):
				return

func _manhattan(a: Vector2i, b: Vector2i) -> int:
	return absi(a.x - b.x) + absi(a.y - b.y)
