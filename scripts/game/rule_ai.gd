class_name RuleAI
extends Node
## Rule AI with timed hazard prediction, safe routing, item racing, and pressure scoring.

signal decision_made(mode: int, target_cell: Vector2i, score: float, elapsed_usec: int)

enum Mode { EVADING, INTERACTING, COLLECTING, PRESSURING, CLEARING, PATROLLING }

const INVALID_CELL: Vector2i = Vector2i(-1, -1)
const PLANNING_HORIZON_MS: int = 5500
const PRESSURE_HORIZON_MS: int = 3000
const ATTACK_APPROACH_MS: int = 1800
const DANGER_REACTION_MS: int = 1200
const BOMB_COOLDOWN_MS: int = 700
const TARGET_LOCK_MS: int = 1200
const PRESSURE_TARGET_LOCK_MS: int = 5200
const MAX_PRESSURE_CANDIDATES: int = 2
const MAX_SAFE_PRESSURE_CHOICES: int = 1
const MAX_STATIC_THREAT_CANDIDATES: int = 24
const MIN_MARGINAL_THREAT_CELLS: float = 2.0
const MIN_MARGINAL_REACHABLE_RATIO: float = 0.10

var actor: GameActor
var board: GameBoard
var match_controller: MatchController
var current_mode: Mode = Mode.PATROLLING
var decision_target: Vector2i = INVALID_CELL
var last_decision_score: float = 0.0
var last_decision_usec: int = 0
var last_pressure_reduction: float = 0.0
var pressure_target_id: int:
	get:
		return _pressure_target_id
var pressure_target_cell: Vector2i:
	get:
		return _pressure_target_cell
var active_pressure_bubbles: int:
	get:
		return _active_pressure_bubbles
var peak_pressure_bubbles: int:
	get:
		return _peak_pressure_bubbles
var total_threat_cells: float:
	get:
		return _total_threat_cells
var target_threat_coverage: float:
	get:
		return _target_threat_coverage
var last_marginal_threat: float:
	get:
		return _last_marginal_threat
var last_pressure_score: float:
	get:
		return _last_pressure_score

var _think_timer: Timer
var _plan: AITemporalPlanner.TimedPlan
var _plan_index: int = 0
var _plan_started_ms: int = 0
var _rng := RandomNumberGenerator.new()
var _last_bomb_ms: int = -10000
var _locked_target: Vector2i = INVALID_CELL
var _target_locked_until_ms: int = 0
var _last_enemy_positions: Dictionary = {}
var _enemy_directions: Dictionary = {}
var _pressure_target_id: int = 0
var _pressure_target_cell: Vector2i = INVALID_CELL
var _pressure_locked_until_ms: int = 0
var _active_pressure_bubbles: int = 0
var _peak_pressure_bubbles: int = 0
var _total_threat_cells: float = 0.0
var _target_threat_coverage: float = 0.0
var _last_marginal_threat: float = 0.0
var _last_pressure_score: float = 0.0


func setup(
		new_actor: GameActor,
		new_board: GameBoard,
		new_match_controller: MatchController,
		decision_seed: int = -1
	) -> void:
	actor = new_actor
	board = new_board
	match_controller = new_match_controller
	_rng.seed = decision_seed if decision_seed >= 0 else hash(actor.actor_name) + Time.get_ticks_msec()
	_think_timer = Timer.new()
	_think_timer.wait_time = GameConstants.AI_THINK_SECONDS
	_think_timer.process_callback = Timer.TIMER_PROCESS_PHYSICS
	_think_timer.timeout.connect(_think)
	add_child(_think_timer)
	_think_timer.start()
	_think()


func _physics_process(_delta: float) -> void:
	if not is_instance_valid(actor) or actor.stats.is_dead or actor.stats.is_trapped:
		if is_instance_valid(actor):
			actor.set_ai_direction(Vector2.ZERO)
		return
	_follow_plan()


func mode_name() -> String:
	return Mode.keys()[current_mode].to_lower()


func set_decision_seed(seed: int) -> void:
	_rng.seed = seed


func reconsider_now() -> void:
	_think()


func reset_for_scenario(seed: int) -> void:
	set_decision_seed(seed)
	_clear_plan()
	current_mode = Mode.PATROLLING
	decision_target = INVALID_CELL
	last_decision_score = 0.0
	last_pressure_reduction = 0.0
	_pressure_target_id = 0
	_pressure_target_cell = INVALID_CELL
	_pressure_locked_until_ms = 0
	_active_pressure_bubbles = 0
	_peak_pressure_bubbles = 0
	_total_threat_cells = 0.0
	_target_threat_coverage = 0.0
	_last_marginal_threat = 0.0
	_last_pressure_score = 0.0
	_last_bomb_ms = -10000
	_locked_target = INVALID_CELL
	_target_locked_until_ms = 0
	_last_enemy_positions.clear()
	_enemy_directions.clear()


func stop_thinking() -> void:
	if is_instance_valid(_think_timer):
		_think_timer.stop()
	_clear_plan()


func _think() -> void:
	var started_usec: int = Time.get_ticks_usec()
	if not is_instance_valid(actor) or actor.stats.is_dead or actor.stats.is_trapped:
		_clear_plan()
		_finish_decision(started_usec)
		return
	match_controller.release_ai_item_claims(actor.get_instance_id())
	var snapshot: AIBattleSnapshot = match_controller.build_ai_snapshot()
	var self_state: AIBattleSnapshot.ActorState = snapshot.actor_by_id(actor.get_instance_id())
	if self_state == null:
		_clear_plan()
		_finish_decision(started_usec)
		return
	_update_enemy_motion(snapshot, self_state.team_id)
	var forecast: AIHazardForecast = AIHazardForecast.build(snapshot, PLANNING_HORIZON_MS)
	_refresh_pressure_metrics(snapshot, self_state, forecast)
	var current: Vector2i = self_state.cell
	if _needs_escape(current, forecast):
		if current_mode == Mode.EVADING and _remaining_plan_is_safe(forecast, true):
			_set_debug(Mode.EVADING, _plan.target_cell(), last_decision_score)
			_finish_decision(started_usec)
			return
		var escape: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_escape_plan(
			snapshot, forecast, current, self_state.move_speed, PLANNING_HORIZON_MS
		)
		if escape.valid:
			_commit_plan(escape, Mode.EVADING, escape.target_cell(), 1000.0)
		else:
			_clear_plan()
			_set_debug(Mode.EVADING, current, -1000.0)
		_finish_decision(started_usec)
		return
	if current_mode == Mode.EVADING and _remaining_plan_is_safe(forecast, true):
		_set_debug(Mode.EVADING, _plan.target_cell(), last_decision_score)
		_finish_decision(started_usec)
		return
	if _locked_plan_can_continue(snapshot, forecast):
		if current_mode == Mode.COLLECTING:
			match_controller.claim_ai_item(actor.get_instance_id(), decision_target)
		_finish_decision(started_usec)
		return
	var interaction: Dictionary = _find_interaction_decision(snapshot, self_state, forecast)
	if not interaction.is_empty():
		_commit_decision(interaction, Mode.INTERACTING)
		_finish_decision(started_usec)
		return
	var target: AIBattleSnapshot.ActorState = _find_enemy(snapshot, self_state)
	var continuing_pressure: bool = target != null and _pressure_intent_is_active(self_state)
	if continuing_pressure:
		var barrage: Dictionary = _find_pressure_decision(snapshot, self_state, target, forecast)
		if not barrage.is_empty():
			_commit_pressure_decision(barrage, snapshot, self_state)
		else:
			var reposition: Dictionary = _find_pressure_reposition(
				snapshot, self_state, target, forecast
			)
			if not reposition.is_empty():
				_commit_decision(reposition, Mode.PRESSURING)
			else:
				_clear_plan()
				_set_debug(Mode.PRESSURING, current, _last_pressure_score)
		_finish_decision(started_usec)
		return
	var item_decision: Dictionary = _find_item_decision(snapshot, self_state, forecast)
	if not item_decision.is_empty():
		var item_cell: Vector2i = item_decision["cell"] as Vector2i
		match_controller.claim_ai_item(actor.get_instance_id(), item_cell)
		_commit_decision(item_decision, Mode.COLLECTING)
		_finish_decision(started_usec)
		return
	if current_mode == Mode.EVADING \
		and forecast.latest_danger_end_ms() > 0 \
		and not continuing_pressure \
		and not forecast.is_unsafe(
				current, 0, forecast.latest_danger_end_ms(), AITemporalPlanner.SAFETY_MARGIN_MS
			):
		_clear_plan()
		_set_debug(Mode.EVADING, current, 900.0)
		_finish_decision(started_usec)
		return
	if target != null and not continuing_pressure:
		var pressure: Dictionary = _find_pressure_decision(snapshot, self_state, target, forecast)
		if not pressure.is_empty():
			_commit_pressure_decision(pressure, snapshot, self_state)
			_finish_decision(started_usec)
			return
	var clearing: Dictionary = _find_clearing_decision(snapshot, self_state, forecast)
	if not clearing.is_empty():
		if bool(clearing.get("drop", false)):
			_drop_bomb_and_escape(snapshot, self_state, Mode.CLEARING, float(clearing["score"]))
		else:
			_commit_decision(clearing, Mode.CLEARING)
		_finish_decision(started_usec)
		return
	var patrol: Dictionary = _find_patrol_decision(snapshot, self_state, forecast, target)
	if not patrol.is_empty():
		_commit_decision(patrol, Mode.PATROLLING)
	else:
		_clear_plan()
		_set_debug(Mode.PATROLLING, current, 0.0)
	_finish_decision(started_usec)


func _follow_plan() -> void:
	if _plan == null or not _plan.valid or _plan_index >= _plan.cells.size():
		actor.set_ai_direction(Vector2.ZERO)
		return
	var elapsed_ms: int = _now_ms() - _plan_started_ms
	while _plan_index < _plan.cells.size():
		var next_cell: Vector2i = _plan.cells[_plan_index]
		var target_position: Vector2 = GameConstants.grid_to_world(next_cell)
		var difference: Vector2 = target_position - actor.position
		if difference.length() >= 4.0:
			actor.set_ai_direction(difference)
			return
		actor.position = target_position
		var previous_planned_cell: Vector2i = _plan.cells[_plan_index - 1]
		var is_wait_step: bool = next_cell == previous_planned_cell
		var early_allowance_ms: int = 0 if is_wait_step else AITemporalPlanner.SAFETY_MARGIN_MS
		if elapsed_ms + early_allowance_ms < _plan.arrival_ms[_plan_index]:
			actor.set_ai_direction(Vector2.ZERO)
			return
		_plan_index += 1
	actor.set_ai_direction(Vector2.ZERO)


func _needs_escape(current: Vector2i, forecast: AIHazardForecast) -> bool:
	return board.bombs.has(current) or forecast.danger_eta_ms(current) <= DANGER_REACTION_MS


func _locked_plan_can_continue(
		snapshot: AIBattleSnapshot,
		forecast: AIHazardForecast
	) -> bool:
	if _now_ms() >= _target_locked_until_ms or _plan == null or _plan_index >= _plan.cells.size():
		return false
	if current_mode == Mode.COLLECTING and decision_target not in snapshot.item_cells:
		return false
	return _remaining_plan_is_safe(forecast, false)


func _remaining_plan_is_safe(forecast: AIHazardForecast, require_stable_target: bool) -> bool:
	if _plan == null or not _plan.valid or _plan_index >= _plan.cells.size():
		return false
	var elapsed_ms: int = maxi(0, _now_ms() - _plan_started_ms)
	var previous_cell: Vector2i = actor.current_cell()
	var previous_time_ms: int = 0
	for index: int in range(_plan_index, _plan.cells.size()):
		var next_cell: Vector2i = _plan.cells[index]
		var next_time_ms: int = maxi(previous_time_ms, _plan.arrival_ms[index] - elapsed_ms)
		if forecast.is_unsafe(
				previous_cell, previous_time_ms, next_time_ms, AITemporalPlanner.SAFETY_MARGIN_MS
			):
			return false
		if forecast.is_unsafe(
				next_cell, previous_time_ms, next_time_ms, AITemporalPlanner.SAFETY_MARGIN_MS
			):
			return false
		if next_cell != previous_cell \
				and forecast.is_bomb_blocked(next_cell, previous_time_ms, next_time_ms):
			return false
		previous_cell = next_cell
		previous_time_ms = next_time_ms
	var safe_until_ms: int = forecast.latest_danger_end_ms() \
		if require_stable_target else previous_time_ms + AITemporalPlanner.SAFE_TAIL_MS
	return not forecast.is_unsafe(
		previous_cell, previous_time_ms, safe_until_ms, AITemporalPlanner.SAFETY_MARGIN_MS
	)


func _find_interaction_decision(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast
	) -> Dictionary:
	var best: Dictionary = {}
	var best_score: float = -INF
	for other: AIBattleSnapshot.ActorState in snapshot.actors:
		if other.instance_id == self_state.instance_id or other.is_dead or not other.is_trapped:
			continue
		var plan: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_path(
			snapshot, forecast, self_state.cell, other.cell, self_state.move_speed, 5000, 750
		)
		if not plan.valid:
			continue
		var score: float = 300.0 - float(plan.travel_ms()) / 10.0
		if other.team_id != self_state.team_id:
			score += 80.0
		if score > best_score:
			best_score = score
			best = {"plan": plan, "cell": other.cell, "score": score}
	return best


func _find_item_decision(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast
	) -> Dictionary:
	var best: Dictionary = {}
	var best_score: float = -INF
	var safe_tail_ms: int = maxi(900, forecast.latest_danger_end_ms() + 100)
	for item_cell: Vector2i in snapshot.item_cells:
		var item_code: int = snapshot.cells[item_cell.y][item_cell.x]
		var value: float = _item_value(item_code)
		if value <= 0.0:
			continue
		var plan: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_path(
			snapshot, forecast, self_state.cell, item_cell,
			self_state.move_speed, 8000, safe_tail_ms
		)
		if not plan.valid:
			continue
		var score: float = value - float(plan.travel_ms()) / 90.0
		var competitor_eta: int = _best_competitor_eta(snapshot, self_state, item_cell)
		if plan.travel_ms() + AITemporalPlanner.WAIT_STEP_MS < competitor_eta:
			score += 22.0
		elif competitor_eta + AITemporalPlanner.WAIT_STEP_MS < plan.travel_ms():
			score -= 42.0
		if match_controller.is_ai_item_claimed_by_other(self_state.instance_id, item_cell):
			score -= 70.0
		if item_cell == _locked_target and _now_ms() < _target_locked_until_ms:
			score += 18.0
		if score > best_score:
			best_score = score
			best = {"plan": plan, "cell": item_cell, "score": score}
	return best


func _find_pressure_decision(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		target: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast
	) -> Dictionary:
	if self_state.active_bubbles >= self_state.bubble_capacity:
		return {}
	if self_state.active_bubbles > 0 and not _bomb_cooldown_ready():
		return {}
	var base_reachable: Dictionary = AITemporalPlanner.reachable_cells_fast(
		snapshot, forecast, target.cell, target.move_speed, PRESSURE_HORIZON_MS
	)
	if base_reachable.is_empty():
		return {}
	var base_threat: AIThreatField = AIThreatField.build(forecast, self_state.instance_id)
	var predicted_target_cells: Array[Vector2i] = _predicted_target_cells(snapshot, target)
	var self_reachable: Dictionary = AITemporalPlanner.reachable_cells_fast(
		snapshot, forecast, self_state.cell, self_state.move_speed, ATTACK_APPROACH_MS
	)
	var rough_candidates: Array[Dictionary] = []
	var minimum_marginal: float = maxf(
		MIN_MARGINAL_THREAT_CELLS,
		float(base_reachable.size()) * MIN_MARGINAL_REACHABLE_RATIO
	)
	for cell: Vector2i in self_reachable.keys():
		var place_ms: int = int(self_reachable[cell])
		var blast: Array[Vector2i] = GameRules.blast_cells(cell, self_state.power, snapshot.cells)
		var predicted_hits: int = 0
		for predicted: Vector2i in predicted_target_cells:
			if predicted in blast:
				predicted_hits += 1
		var exit_hits: int = _covered_target_exits(snapshot, target.cell, blast)
		var nearest_owned_bomb: int = 8
		for existing_bomb: AIBattleSnapshot.BombState in snapshot.bombs:
			if existing_bomb.owner_id == self_state.instance_id:
				nearest_owned_bomb = mini(nearest_owned_bomb, _manhattan(cell, existing_bomb.cell))
		var rough_score: float = predicted_hits * 35.0 \
			+ exit_hits * 18.0 \
			+ blast.size() * 1.5 \
			+ nearest_owned_bomb * 6.0 \
			- _manhattan(cell, target.cell) * 4.0 \
			- place_ms / 180.0
		if cell == self_state.cell:
			rough_score += 100.0 if self_state.active_bubbles == 0 else 50.0
		rough_candidates.append({
			"cell": cell,
			"place_ms": place_ms,
			"blast": blast,
			"predicted_hits": predicted_hits,
			"exit_hits": exit_hits,
			"rough_score": rough_score,
		})
	rough_candidates.sort_custom(func(left: Dictionary, right: Dictionary) -> bool:
		return float(left["rough_score"]) > float(right["rough_score"])
	)
	var selected_rough_candidates: Array[Dictionary] = []
	for rough_index: int in range(mini(MAX_STATIC_THREAT_CANDIDATES, rough_candidates.size())):
		selected_rough_candidates.append(rough_candidates[rough_index])
	var includes_current: bool = false
	for selected: Dictionary in selected_rough_candidates:
		if selected["cell"] as Vector2i == self_state.cell:
			includes_current = true
			break
	if not includes_current:
		for rough_candidate: Dictionary in rough_candidates:
			if rough_candidate["cell"] as Vector2i == self_state.cell:
				selected_rough_candidates.append(rough_candidate)
				break
	var candidates: Array[Dictionary] = []
	for rough_candidate: Dictionary in selected_rough_candidates:
		var blast: Array[Vector2i] = rough_candidate["blast"] as Array[Vector2i]
		var estimated_threat: float = AIThreatField.estimate_marginal_blast_weight(
			blast, snapshot.cells, base_threat, base_reachable
		) if self_state.active_bubbles > 0 else AIThreatField.estimate_blast_weight(
			blast, snapshot.cells, base_reachable
		)
		if estimated_threat <= 0.0 \
				or (self_state.active_bubbles > 0 and estimated_threat + 0.001 < minimum_marginal):
			continue
		var static_score: float = estimated_threat * 8.0 \
			+ int(rough_candidate["predicted_hits"]) * 20.0 \
			+ int(rough_candidate["exit_hits"]) * 10.0 \
			- float(rough_candidate["place_ms"]) / 180.0
		if rough_candidate["cell"] as Vector2i == self_state.cell:
			static_score += 100.0 if self_state.active_bubbles == 0 else 50.0
		candidates.append({
			"cell": rough_candidate["cell"],
			"place_ms": rough_candidate["place_ms"],
			"blast": blast,
			"predicted_hits": rough_candidate["predicted_hits"],
			"exit_hits": rough_candidate["exit_hits"],
			"estimated_threat": estimated_threat,
			"static_score": static_score,
		})
	candidates.sort_custom(func(left: Dictionary, right: Dictionary) -> bool:
		return float(left["static_score"]) > float(right["static_score"])
	)
	for candidate_index: int in range(candidates.size()):
		if candidates[candidate_index]["cell"] as Vector2i == self_state.cell:
			var current_candidate: Dictionary = candidates.pop_at(candidate_index)
			candidates.push_front(current_candidate)
			break
	var best: Dictionary = {}
	var best_score: float = -INF
	var best_reduction: float = 0.0
	var best_coverage: float = base_threat.coverage_ratio(base_reachable)
	var best_marginal: float = 0.0
	var safe_choices: int = 0
	for index: int in range(mini(MAX_PRESSURE_CANDIDATES, candidates.size())):
		var candidate: Dictionary = candidates[index]
		var candidate_cell: Vector2i = candidate["cell"] as Vector2i
		var place_ms: int = int(candidate["place_ms"])
		var virtual_forecast: AIHazardForecast = AIHazardForecast.build(
			snapshot,
			maxi(PLANNING_HORIZON_MS, place_ms + 4000),
			candidate_cell,
			self_state.power,
			place_ms,
			int(GameConstants.BUBBLE_FUSE_SECONDS * 1000.0),
			self_state.instance_id
		)
		var virtual_route: AITemporalPlanner.TimedPlan = _stationary_plan(self_state.cell)
		if candidate_cell != self_state.cell:
			virtual_route = _timed_plan_from_static_path(
				board.find_path(self_state.cell, candidate_cell, actor), self_state.move_speed
			)
		if not virtual_route.valid or not _timed_plan_is_safe(virtual_route, virtual_forecast):
			continue
		var escape: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_direct_escape_plan(
			snapshot, virtual_forecast, candidate_cell, self_state.move_speed,
			PLANNING_HORIZON_MS - place_ms, place_ms
		)
		if not escape.valid:
			continue
		var candidate_blast: Array[Vector2i] = _virtual_candidate_blast(
			virtual_forecast, self_state.instance_id, candidate_cell, candidate["blast"] as Array[Vector2i]
		)
		if not _allies_can_escape(snapshot, self_state, virtual_forecast, candidate_blast):
			continue
		var combined_threat: AIThreatField = AIThreatField.build(
			virtual_forecast, self_state.instance_id
		)
		var marginal_threat: float = combined_threat.marginal_weight(
			base_threat, base_reachable
		)
		if self_state.active_bubbles > 0 and marginal_threat + 0.001 < minimum_marginal:
			continue
		var coverage: float = combined_threat.coverage_ratio(base_reachable)
		var score: float = coverage * 120.0 \
			+ marginal_threat * 12.0 \
			+ int(candidate["predicted_hits"]) * 16.0 \
			+ int(candidate["exit_hits"]) * 10.0 \
			- float(place_ms) / 120.0
		if target.cell in candidate_blast:
			score += 28.0
		if candidate_cell == self_state.cell:
			score += 12.0
		safe_choices += 1
		if score > best_score:
			best_score = score
			best_coverage = coverage
			best_marginal = marginal_threat
			best = {
				"plan": virtual_route,
				"cell": candidate_cell,
				"score": score,
				"drop": candidate_cell == self_state.cell and _bomb_cooldown_ready(),
				"forecast": virtual_forecast,
				"escape": escape,
			}
		if safe_choices >= MAX_SAFE_PRESSURE_CHOICES:
			break
	if not best.is_empty():
		var selected_forecast: AIHazardForecast = best["forecast"] as AIHazardForecast
		var pressured_safe_count: int = 0
		for reachable_cell: Vector2i in base_reachable.keys():
			if not selected_forecast.is_unsafe(
					reachable_cell,
					PRESSURE_HORIZON_MS - AITemporalPlanner.SAFETY_MARGIN_MS,
					PRESSURE_HORIZON_MS + AITemporalPlanner.SAFETY_MARGIN_MS,
					AITemporalPlanner.SAFETY_MARGIN_MS
				):
				pressured_safe_count += 1
		best_reduction = clampf(
			1.0 - float(pressured_safe_count) / maxf(1.0, float(base_reachable.size())),
			0.0,
			1.0
		)
		best_score += best_reduction * 150.0
		best["score"] = best_score
		best.erase("forecast")
		last_pressure_reduction = best_reduction
		_target_threat_coverage = best_coverage
		_last_marginal_threat = best_marginal
		_last_pressure_score = best_score
		_lock_pressure_target(target)
	return best


func _find_pressure_reposition(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		target: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast
	) -> Dictionary:
	var reachable: Dictionary = AITemporalPlanner.reachable_cells_fast(
		snapshot, forecast, self_state.cell, self_state.move_speed, 1200
	)
	var candidates: Array[Dictionary] = []
	for cell: Vector2i in reachable.keys():
		if cell == self_state.cell:
			continue
		var nearest_owned_bomb: int = 999
		for bomb: AIBattleSnapshot.BombState in snapshot.bombs:
			if bomb.owner_id == self_state.instance_id:
				nearest_owned_bomb = mini(nearest_owned_bomb, _manhattan(cell, bomb.cell))
		var score: float = nearest_owned_bomb * 5.0 \
			- _manhattan(cell, target.cell) * 11.0 \
			- float(reachable[cell]) / 180.0
		candidates.append({"cell": cell, "score": score})
	candidates.sort_custom(func(left: Dictionary, right: Dictionary) -> bool:
		return float(left["score"]) > float(right["score"])
	)
	for index: int in range(mini(4, candidates.size())):
		var candidate: Dictionary = candidates[index]
		var candidate_cell: Vector2i = candidate["cell"] as Vector2i
		var path: Array[Vector2i] = board.find_path(self_state.cell, candidate_cell, actor)
		var plan: AITemporalPlanner.TimedPlan = _timed_plan_from_static_path(
			path, self_state.move_speed
		)
		if not plan.valid or not _timed_plan_is_safe(plan, forecast):
			continue
		if forecast.is_unsafe(
				candidate_cell,
				plan.travel_ms(),
				plan.travel_ms() + 750,
				AITemporalPlanner.SAFETY_MARGIN_MS
			):
			continue
		return {"plan": plan, "cell": candidate_cell, "score": float(candidate["score"])}
	return {}


func _find_clearing_decision(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast
	) -> Dictionary:
	if self_state.active_bubbles >= self_state.bubble_capacity:
		return {}
	if not _snapshot_has_boxes(snapshot):
		return {}
	var reachable: Dictionary = AITemporalPlanner.reachable_cells(
		snapshot, forecast, self_state.cell, self_state.move_speed, 5000
	)
	var best_cell: Vector2i = INVALID_CELL
	var best_score: float = -INF
	for cell: Vector2i in reachable.keys():
		if not _has_adjacent_box(snapshot, cell):
			continue
		var score: float = 90.0 - float(reachable[cell]) / 80.0
		if cell == self_state.cell:
			score += 20.0
		if score > best_score:
			best_score = score
			best_cell = cell
	if best_cell == INVALID_CELL:
		return {}
	var plan: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_path(
		snapshot, forecast, self_state.cell, best_cell, self_state.move_speed, 5000, 600
	)
	if not plan.valid:
		return {}
	var can_drop_here: bool = best_cell == self_state.cell \
		and _bomb_cooldown_ready() \
		and _virtual_drop_is_safe(snapshot, self_state, best_cell)
	return {
		"plan": plan,
		"cell": best_cell,
		"score": best_score,
		"drop": can_drop_here,
	}


func _find_patrol_decision(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast,
		target: AIBattleSnapshot.ActorState
	) -> Dictionary:
	var candidates: Array[Vector2i] = _open_cells(snapshot)
	if candidates.is_empty():
		return {}
	if forecast.latest_danger_end_ms() == 0:
		return _find_static_patrol_decision(candidates, self_state, target)
	var reachable: Dictionary = AITemporalPlanner.reachable_cells(
		snapshot, forecast, self_state.cell, self_state.move_speed, 6000
	)
	var best_cell: Vector2i = INVALID_CELL
	var best_score: float = -INF
	for _attempt: int in range(mini(24, candidates.size())):
		var cell: Vector2i = candidates[_rng.randi_range(0, candidates.size() - 1)]
		if _manhattan(self_state.cell, cell) < 2:
			continue
		if not reachable.has(cell):
			continue
		var score: float = -float(reachable[cell]) / 200.0
		if target != null:
			score -= _manhattan(cell, target.cell) * 2.0
		if score > best_score:
			best_score = score
			best_cell = cell
	if best_cell == INVALID_CELL:
		return {}
	var plan: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_path(
		snapshot, forecast, self_state.cell, best_cell, self_state.move_speed, 6000, 600
	)
	return {"plan": plan, "cell": best_cell, "score": best_score} if plan.valid else {}


func _find_static_patrol_decision(
		candidates: Array[Vector2i],
		self_state: AIBattleSnapshot.ActorState,
		target: AIBattleSnapshot.ActorState
	) -> Dictionary:
	var best_cell: Vector2i = INVALID_CELL
	var best_score: float = -INF
	for _attempt: int in range(mini(24, candidates.size())):
		var cell: Vector2i = candidates[_rng.randi_range(0, candidates.size() - 1)]
		var distance: int = _manhattan(self_state.cell, cell)
		if distance < 2:
			continue
		var score: float = -distance * 1.5
		if target != null:
			score -= _manhattan(cell, target.cell) * 2.0
		if score > best_score:
			best_score = score
			best_cell = cell
	if best_cell == INVALID_CELL:
		return {}
	var path: Array[Vector2i] = board.find_path(self_state.cell, best_cell, actor)
	var plan: AITemporalPlanner.TimedPlan = _timed_plan_from_static_path(path, self_state.move_speed)
	return {"plan": plan, "cell": best_cell, "score": best_score} if plan.valid else {}


func _commit_pressure_decision(
		decision: Dictionary,
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState
	) -> void:
	if bool(decision.get("drop", false)):
		var prechecked_escape: AITemporalPlanner.TimedPlan = decision.get("escape") as AITemporalPlanner.TimedPlan
		if _drop_bomb_and_escape(
				snapshot,
				self_state,
				Mode.PRESSURING,
				float(decision["score"]),
				prechecked_escape
			):
			return
	_commit_decision(decision, Mode.PRESSURING)


func _drop_bomb_and_escape(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		mode: Mode,
		score: float,
		prechecked_escape: AITemporalPlanner.TimedPlan = null
	) -> bool:
	var current: Vector2i = self_state.cell
	if not _bomb_cooldown_ready() or not board.can_place_bubble(current):
		return false
	if prechecked_escape == null and not _virtual_drop_is_safe(snapshot, self_state, current):
		return false
	var previous_active: int = actor.stats.active_bubbles
	actor.request_ai_bomb()
	if actor.stats.active_bubbles <= previous_active:
		return false
	_last_bomb_ms = _now_ms()
	if mode == Mode.PRESSURING:
		_pressure_locked_until_ms = _now_ms() + PRESSURE_TARGET_LOCK_MS
	var updated_snapshot: AIBattleSnapshot = match_controller.build_ai_snapshot()
	var updated_forecast: AIHazardForecast = AIHazardForecast.build(updated_snapshot, PLANNING_HORIZON_MS)
	var updated_self: AIBattleSnapshot.ActorState = updated_snapshot.actor_by_id(self_state.instance_id)
	if updated_self != null:
		_refresh_pressure_metrics(updated_snapshot, updated_self, updated_forecast)
	var escape: AITemporalPlanner.TimedPlan = prechecked_escape
	if escape == null or not escape.valid:
		escape = AITemporalPlanner.find_escape_plan(
			updated_snapshot, updated_forecast, current, self_state.move_speed, PLANNING_HORIZON_MS
		)
	if escape.valid:
		_commit_plan(escape, Mode.EVADING, escape.target_cell(), score + 100.0)
	else:
		_clear_plan()
		_set_debug(mode, current, score)
	return true


func _virtual_drop_is_safe(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		cell: Vector2i
	) -> bool:
	if self_state.active_bubbles >= self_state.bubble_capacity or not board.can_place_bubble(cell):
		return false
	var virtual_forecast: AIHazardForecast = AIHazardForecast.build(
		snapshot,
		PLANNING_HORIZON_MS,
		cell,
		self_state.power,
		0,
		int(GameConstants.BUBBLE_FUSE_SECONDS * 1000.0),
		self_state.instance_id
	)
	var escape: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_escape_plan(
		snapshot, virtual_forecast, cell, self_state.move_speed, PLANNING_HORIZON_MS
	)
	return escape.valid and escape.cells.size() > 1


func _allies_can_escape(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast,
		blast: Array[Vector2i]
	) -> bool:
	for ally: AIBattleSnapshot.ActorState in snapshot.actors:
		if ally.instance_id == self_state.instance_id or ally.team_id != self_state.team_id:
			continue
		if ally.is_dead or ally.is_trapped or ally.cell not in blast:
			continue
		var escape: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_escape_plan(
			snapshot, forecast, ally.cell, ally.move_speed, PLANNING_HORIZON_MS
		)
		if not escape.valid:
			return false
	return true


func _virtual_candidate_blast(
		forecast: AIHazardForecast,
		owner_id: int,
		candidate_cell: Vector2i,
		fallback: Array[Vector2i]
	) -> Array[Vector2i]:
	var result: Array[Vector2i] = fallback
	for blast_event: AIHazardForecast.BombBlast in forecast.blast_events:
		if blast_event.owner_id == owner_id and blast_event.bomb_cell == candidate_cell:
			result = blast_event.cells
	return result


func _predicted_target_cells(
		snapshot: AIBattleSnapshot,
		target: AIBattleSnapshot.ActorState
	) -> Array[Vector2i]:
	var result: Array[Vector2i] = [target.cell]
	var direction: Vector2i = _enemy_directions.get(target.instance_id, Vector2i.ZERO) as Vector2i
	for distance: int in range(1, 3):
		var predicted: Vector2i = target.cell + direction * distance
		if direction == Vector2i.ZERO or not _snapshot_walkable(snapshot, predicted):
			break
		result.append(predicted)
	return result


func _covered_target_exits(
		snapshot: AIBattleSnapshot,
		target_cell: Vector2i,
		blast: Array[Vector2i]
	) -> int:
	var covered: int = 0
	for direction: Vector2i in AITemporalPlanner.CARDINAL_DIRECTIONS:
		var exit_cell: Vector2i = target_cell + direction
		if _snapshot_walkable(snapshot, exit_cell) and exit_cell in blast:
			covered += 1
	return covered


func _find_enemy(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState
	) -> AIBattleSnapshot.ActorState:
	if _pressure_target_id != 0:
		var locked_target: AIBattleSnapshot.ActorState = snapshot.actor_by_id(_pressure_target_id)
		if locked_target != null \
				and locked_target.team_id != self_state.team_id \
				and not locked_target.is_dead \
				and not locked_target.is_trapped \
				and _pressure_intent_is_active(self_state):
			return locked_target
		if locked_target == null or locked_target.is_dead or locked_target.is_trapped:
			_clear_pressure_target()
	var best: AIBattleSnapshot.ActorState
	var best_distance: int = 99999
	for other: AIBattleSnapshot.ActorState in snapshot.actors:
		if other.instance_id == self_state.instance_id or other.team_id == self_state.team_id:
			continue
		if other.is_dead or other.is_trapped:
			continue
		var distance: int = _manhattan(self_state.cell, other.cell)
		if distance < best_distance:
			best_distance = distance
			best = other
	return best


func _pressure_intent_is_active(self_state: AIBattleSnapshot.ActorState) -> bool:
	return _pressure_target_id != 0 \
		and (_active_pressure_bubbles > 0 \
			or self_state.active_bubbles > 0 \
			or _now_ms() < _pressure_locked_until_ms)


func _lock_pressure_target(target: AIBattleSnapshot.ActorState) -> void:
	_pressure_target_id = target.instance_id
	_pressure_target_cell = target.cell
	_pressure_locked_until_ms = _now_ms() + PRESSURE_TARGET_LOCK_MS


func _clear_pressure_target() -> void:
	_pressure_target_id = 0
	_pressure_target_cell = INVALID_CELL
	_pressure_locked_until_ms = 0
	_target_threat_coverage = 0.0
	_last_marginal_threat = 0.0
	_last_pressure_score = 0.0


func _refresh_pressure_metrics(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast
	) -> void:
	_active_pressure_bubbles = 0
	for bomb: AIBattleSnapshot.BombState in snapshot.bombs:
		if bomb.owner_id == self_state.instance_id:
			_active_pressure_bubbles += 1
	_peak_pressure_bubbles = maxi(_peak_pressure_bubbles, _active_pressure_bubbles)
	var active_threat: AIThreatField = AIThreatField.build(forecast, self_state.instance_id)
	_total_threat_cells = active_threat.total_weight
	if _pressure_target_id == 0:
		return
	var target: AIBattleSnapshot.ActorState = snapshot.actor_by_id(_pressure_target_id)
	if target == null or target.is_dead or target.is_trapped:
		_clear_pressure_target()
		return
	_pressure_target_cell = target.cell
	if _active_pressure_bubbles == 0 and _now_ms() >= _pressure_locked_until_ms:
		_clear_pressure_target()


func _best_competitor_eta(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		item_cell: Vector2i
	) -> int:
	var best: int = 999999
	for other: AIBattleSnapshot.ActorState in snapshot.actors:
		if other.instance_id == self_state.instance_id or other.is_dead or other.is_trapped:
			continue
		var travel_ms: int = ceili(
			_manhattan(other.cell, item_cell) * GameConstants.CELL_SIZE \
			/ maxf(1.0, other.move_speed) * 1000.0
		)
		best = mini(best, travel_ms)
	return best


func _item_value(item_code: int) -> float:
	match item_code:
		GameConstants.ITEM_BUBBLE:
			if actor.stats.bubble_capacity < actor.settings.max_bubbles:
				return 74.0 + (actor.settings.max_bubbles - actor.stats.bubble_capacity) * 2.0
		GameConstants.ITEM_SPEED:
			if actor.stats.move_speed < actor.settings.max_speed:
				return 78.0 + (actor.settings.max_speed - actor.stats.move_speed) / 25.0
		GameConstants.ITEM_POWER:
			if actor.stats.power < actor.settings.max_power:
				return 84.0 + (actor.settings.max_power - actor.stats.power) * 2.0
	return 0.0


func _update_enemy_motion(snapshot: AIBattleSnapshot, team_id: int) -> void:
	for other: AIBattleSnapshot.ActorState in snapshot.actors:
		if other.team_id == team_id or other.is_dead:
			continue
		var previous: Vector2 = _last_enemy_positions.get(other.instance_id, other.world_position) as Vector2
		var delta: Vector2 = other.world_position - previous
		var direction := Vector2i.ZERO
		if absf(delta.x) >= absf(delta.y) and absf(delta.x) > 0.5:
			direction.x = signi(int(delta.x))
		elif absf(delta.y) > 0.5:
			direction.y = signi(int(delta.y))
		_enemy_directions[other.instance_id] = direction
		_last_enemy_positions[other.instance_id] = other.world_position


func _has_adjacent_box(snapshot: AIBattleSnapshot, cell: Vector2i) -> bool:
	for direction: Vector2i in AITemporalPlanner.CARDINAL_DIRECTIONS:
		var neighbor: Vector2i = cell + direction
		if GameConstants.is_inside(neighbor) \
				and GameRules.is_destructible(snapshot.cells[neighbor.y][neighbor.x]):
			return true
	return false


func _open_cells(snapshot: AIBattleSnapshot) -> Array[Vector2i]:
	var result: Array[Vector2i] = []
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			if _snapshot_walkable(snapshot, cell):
				result.append(cell)
	return result


func _snapshot_has_boxes(snapshot: AIBattleSnapshot) -> bool:
	for row: PackedInt32Array in snapshot.cells:
		for code: int in row:
			if GameRules.is_destructible(code):
				return true
	return false


func _snapshot_walkable(snapshot: AIBattleSnapshot, cell: Vector2i) -> bool:
	return GameConstants.is_inside(cell) and GameRules.is_walkable(snapshot.cells[cell.y][cell.x])


func _commit_decision(decision: Dictionary, mode: Mode) -> void:
	_commit_plan(
		decision["plan"] as AITemporalPlanner.TimedPlan,
		mode,
		decision["cell"] as Vector2i,
		float(decision["score"])
	)


func _stationary_plan(cell: Vector2i) -> AITemporalPlanner.TimedPlan:
	var plan := AITemporalPlanner.TimedPlan.new()
	plan.cells = [cell]
	plan.arrival_ms = PackedInt32Array([0])
	plan.valid = true
	return plan


func _timed_plan_from_static_path(
		path: Array[Vector2i],
		move_speed: float
	) -> AITemporalPlanner.TimedPlan:
	var plan := AITemporalPlanner.TimedPlan.new()
	if path.is_empty():
		return plan
	var move_ms: int = ceili(
		GameConstants.CELL_SIZE / maxf(1.0, move_speed) * 1000.0 \
		/ AITemporalPlanner.WAIT_STEP_MS
	) * AITemporalPlanner.WAIT_STEP_MS
	for index: int in range(path.size()):
		plan.cells.append(path[index])
		plan.arrival_ms.append(index * move_ms)
	plan.valid = true
	return plan


func _timed_plan_is_safe(
		plan: AITemporalPlanner.TimedPlan,
		forecast: AIHazardForecast
	) -> bool:
	if not plan.valid or plan.cells.is_empty():
		return false
	var previous_cell: Vector2i = plan.cells[0]
	var previous_ms: int = 0
	for index: int in range(1, plan.cells.size()):
		var next_cell: Vector2i = plan.cells[index]
		var next_ms: int = plan.arrival_ms[index]
		if forecast.is_unsafe(
				previous_cell, previous_ms, next_ms, AITemporalPlanner.SAFETY_MARGIN_MS
			) or forecast.is_unsafe(
				next_cell, previous_ms, next_ms, AITemporalPlanner.SAFETY_MARGIN_MS
			):
			return false
		if next_cell != previous_cell \
				and forecast.is_bomb_blocked(next_cell, previous_ms, next_ms):
			return false
		previous_cell = next_cell
		previous_ms = next_ms
	return true


func _commit_plan(
		new_plan: AITemporalPlanner.TimedPlan,
		mode: Mode,
		target_cell: Vector2i,
		score: float
	) -> void:
	_plan = new_plan
	_plan_index = 1 if new_plan.cells.size() > 1 else new_plan.cells.size()
	_plan_started_ms = _now_ms()
	_set_debug(mode, target_cell, score)
	if mode != Mode.EVADING:
		_locked_target = target_cell
		_target_locked_until_ms = _now_ms() + TARGET_LOCK_MS


func _clear_plan() -> void:
	_plan = null
	_plan_index = 0
	if is_instance_valid(actor):
		actor.set_ai_direction(Vector2.ZERO)


func _set_debug(mode: Mode, target_cell: Vector2i, score: float) -> void:
	current_mode = mode
	decision_target = target_cell
	last_decision_score = score


func _finish_decision(started_usec: int) -> void:
	last_decision_usec = Time.get_ticks_usec() - started_usec
	decision_made.emit(current_mode, decision_target, last_decision_score, last_decision_usec)


func _bomb_cooldown_ready() -> bool:
	return _now_ms() - _last_bomb_ms >= BOMB_COOLDOWN_MS


func _now_ms() -> int:
	return match_controller.get_simulation_time_ms() \
		if is_instance_valid(match_controller) else Time.get_ticks_msec()


func _manhattan(left: Vector2i, right: Vector2i) -> int:
	return absi(left.x - right.x) + absi(left.y - right.y)
