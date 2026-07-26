class_name RuleAI
extends Node
## Rule AI with timed hazard prediction, paint scoring, and safe routing.

signal decision_made(mode: int, target_cell: Vector2i, score: float, elapsed_usec: int)

enum Mode { EVADING, PAINTING, INTERACTING, PRESSURING, PATROLLING, COLLECTING }

const INVALID_CELL: Vector2i = Vector2i(-1, -1)
const PLANNING_HORIZON_MS: int = 5500
const PRESSURE_HORIZON_MS: int = 3000
const ATTACK_APPROACH_MS: int = 1800
const PAINT_APPROACH_MS: int = 1200
const DANGER_REACTION_MS: int = 1200
const BOMB_COOLDOWN_MS: int = AITemporalPlanner.WAIT_STEP_MS * 2
const TARGET_LOCK_MS: int = 1200
const PRESSURE_TARGET_LOCK_MS: int = 5200
const MAX_PRESSURE_CANDIDATES: int = 2
const MAX_SAFE_PRESSURE_CHOICES: int = 1
const MAX_STATIC_THREAT_CANDIDATES: int = 24
const MIN_MARGINAL_THREAT_CELLS: float = 2.0
const MIN_MARGINAL_REACHABLE_RATIO: float = 0.10
const PATROL_MIN_FLOW_DISTANCE: int = 3
const PATROL_DESIRED_FLOW_DISTANCE: int = 9
const RECENT_NAVIGATION_CELL_LIMIT: int = 7

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
var _continuous_plan_motion: bool = false
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
var _thinking_enabled: bool = true
var _last_navigation_cell: Vector2i = INVALID_CELL
var _navigation_heading: Vector2i = Vector2i.ZERO
var _recent_navigation_cells: Array[Vector2i] = []
var _claimed_item_id: int = 0


func setup(
		new_actor: GameActor,
		new_board: GameBoard,
		new_match_controller: MatchController,
		decision_seed: int = -1,
		use_internal_timer: bool = true
	) -> void:
	actor = new_actor
	board = new_board
	match_controller = new_match_controller
	_thinking_enabled = true
	_rng.seed = decision_seed if decision_seed >= 0 else hash(actor.actor_name) + Time.get_ticks_msec()
	if use_internal_timer:
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
	if _thinking_enabled:
		_think()


func reset_for_scenario(seed: int) -> void:
	_thinking_enabled = true
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
	_last_navigation_cell = INVALID_CELL
	_navigation_heading = Vector2i.ZERO
	_recent_navigation_cells.clear()
	_release_item_claim()


func stop_thinking() -> void:
	_thinking_enabled = false
	if is_instance_valid(_think_timer):
		_think_timer.stop()
	_clear_plan()


func _think() -> void:
	var started_usec: int = Time.get_ticks_usec()
	if not is_instance_valid(actor) or actor.stats.is_dead or actor.stats.is_trapped:
		_clear_plan()
		_finish_decision(started_usec)
		return
	var snapshot: AIBattleSnapshot = match_controller.build_ai_snapshot()
	var self_state: AIBattleSnapshot.ActorState = snapshot.actor_by_id(actor.get_instance_id())
	if self_state == null:
		_clear_plan()
		_finish_decision(started_usec)
		return
	_record_navigation_cell(self_state.cell)
	_update_enemy_motion(snapshot, self_state.team_id)
	var forecast: AIHazardForecast = match_controller.get_shared_ai_forecast(
		snapshot, PLANNING_HORIZON_MS
	)
	var current: Vector2i = self_state.cell
	if _needs_escape(current, forecast):
		if current_mode == Mode.EVADING and _remaining_plan_is_safe(forecast, true):
			_set_debug(Mode.EVADING, _plan.target_cell(), last_decision_score)
			_finish_decision(started_usec)
			return
		var escape: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_direct_escape_plan(
			snapshot, forecast, current, self_state.move_speed, PLANNING_HORIZON_MS
		)
		if not escape.valid:
			escape = AITemporalPlanner.find_escape_plan(
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
	_refresh_pressure_metrics(snapshot, self_state, forecast)
	if current_mode == Mode.COLLECTING and _locked_plan_can_continue(snapshot, forecast):
		_finish_decision(started_usec)
		return
	var item_decision: Dictionary = _find_item_decision(
		snapshot,
		self_state,
		forecast,
		match_controller.ai_profile.minimum_item_priority_score
	)
	if not item_decision.is_empty() and _commit_item_decision(item_decision):
		_finish_decision(started_usec)
		return
	if _locked_plan_can_continue(snapshot, forecast):
		_finish_decision(started_usec)
		return
	var paint_decision: Dictionary = _find_paint_decision(snapshot, self_state, forecast)
	var interaction: Dictionary = _find_interaction_decision(snapshot, self_state, forecast)
	var selected_kind: String = ""
	var selected_score: float = -INF
	for candidate: Dictionary in [
		{"kind": "paint", "decision": paint_decision},
		{"kind": "interaction", "decision": interaction},
	]:
		var decision: Dictionary = candidate["decision"] as Dictionary
		if decision.is_empty() or float(decision.get("score", -INF)) <= selected_score:
			continue
		selected_kind = str(candidate["kind"])
		selected_score = float(decision["score"])
	match selected_kind:
		"interaction":
			_commit_decision(interaction, Mode.INTERACTING)
			_finish_decision(started_usec)
			return
		"paint":
			if bool(paint_decision.get("drop", false)):
				_drop_bomb_and_escape(
					snapshot,
					self_state,
					Mode.PAINTING,
					float(paint_decision["score"]),
					paint_decision.get("escape") as AITemporalPlanner.TimedPlan
				)
			else:
				_commit_decision(paint_decision, Mode.PAINTING)
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
		if not (_continuous_plan_motion and not is_wait_step) \
				and elapsed_ms + early_allowance_ms < _plan.arrival_ms[_plan_index]:
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
	if current_mode == Mode.COLLECTING:
		var elapsed_ms: int = maxi(0, _now_ms() - _plan_started_ms)
		var remaining_travel_ms: int = maxi(
			0,
			_plan.travel_ms() - elapsed_ms
		)
		if _claimed_item_id == 0 \
				or snapshot.item_by_id(_claimed_item_id) == null \
				or not match_controller.claim_item(
					actor.get_instance_id(),
					_claimed_item_id,
					remaining_travel_ms
				):
			_release_item_claim()
			_clear_plan()
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
		if other.instance_id == self_state.instance_id \
			or other.team_id == self_state.team_id \
			or other.is_dead \
			or not other.is_trapped:
			continue
		var plan: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_path(
			snapshot, forecast, self_state.cell, other.cell, self_state.move_speed, 5000, 750
		)
		if not plan.valid:
			continue
		var score: float = 300.0 - float(plan.travel_ms()) / 10.0
		score += 80.0
		if score > best_score:
			best_score = score
			best = {"plan": plan, "cell": other.cell, "score": score}
	return best


func _find_item_decision(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast,
		minimum_score: float = -INF
	) -> Dictionary:
	if snapshot.items.is_empty() or snapshot.remaining_round_ms <= 1000:
		return {}
	var profile: AIBehaviorProfile = match_controller.ai_profile
	var maximum_travel_ms: int = mini(
		profile.maximum_item_travel_ms,
		snapshot.remaining_round_ms - 500
	)
	if maximum_travel_ms <= 0:
		return {}
	var move_ms_per_cell: int = ceili(
		GameConstants.CELL_SIZE / maxf(1.0, self_state.move_speed) * 1000.0 \
		/ AITemporalPlanner.WAIT_STEP_MS
	) * AITemporalPlanner.WAIT_STEP_MS
	var ranked: Array[Dictionary] = []
	for item: AIBattleSnapshot.ItemState in snapshot.items:
		var travel_ms: int = _manhattan(self_state.cell, item.cell) * move_ms_per_cell
		if travel_ms > maximum_travel_ms:
			continue
		if not match_controller.can_claim_item(
				self_state.instance_id,
				item.item_id,
				travel_ms
			):
			continue
		var score: float = profile.item_base_value \
			+ _item_marginal_value(item, snapshot, self_state, profile) \
			- float(travel_ms) / profile.item_travel_divisor
		var payback_ms: int = travel_ms \
			+ int(GameConstants.BUBBLE_FUSE_SECONDS * 1000.0)
		if snapshot.remaining_round_ms \
				<= payback_ms + profile.minimum_item_use_window_ms:
			continue
		var enemy_eta_ms: int = _nearest_enemy_item_eta(snapshot, self_state, item.cell)
		if enemy_eta_ms <= travel_ms + profile.item_competition_window_ms:
			score -= profile.item_competition_penalty
		if snapshot.remaining_round_ms < payback_ms + profile.item_payback_buffer_ms:
			var late_ratio: float = 1.0 - clampf(
				float(snapshot.remaining_round_ms - travel_ms) / 5500.0,
				0.0,
				1.0
			)
			score -= profile.item_late_round_penalty * late_ratio
		if item.cell == self_state.cell:
			score += 80.0
		ranked.append({
			"cell": item.cell,
			"score": score,
			"item_id": item.item_id,
			"travel_ms": travel_ms,
		})
	ranked.sort_custom(func(left: Dictionary, right: Dictionary) -> bool:
		return float(left["score"]) > float(right["score"])
	)
	if ranked.is_empty() or float(ranked[0]["score"]) <= minimum_score:
		return {}
	for index: int in range(mini(3, ranked.size())):
		var candidate: Dictionary = ranked[index]
		var candidate_cell: Vector2i = candidate["cell"] as Vector2i
		var path: Array[Vector2i] = board.find_path(self_state.cell, candidate_cell, actor)
		var plan: AITemporalPlanner.TimedPlan = _timed_plan_from_static_path(
			path,
			self_state.move_speed
		)
		if not plan.valid or not _timed_plan_is_safe(plan, forecast):
			continue
		if forecast.is_unsafe(
				candidate_cell,
				plan.travel_ms(),
				plan.travel_ms() + 500,
				AITemporalPlanner.SAFETY_MARGIN_MS
			):
			continue
		candidate["plan"] = plan
		candidate["travel_ms"] = plan.travel_ms()
		return candidate
	return {}


func _item_marginal_value(
		item: AIBattleSnapshot.ItemState,
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		profile: AIBehaviorProfile
	) -> float:
	var remaining_cycles: float = clampf(
		float(snapshot.remaining_round_ms) / 9000.0,
		1.0,
		5.0
	)
	match item.item_type:
		ArenaItemType.Value.SPEED:
			var eta_gain_ms: float = _speed_target_eta_gain_ms(
				item.cell,
				snapshot,
				self_state
			)
			return profile.speed_item_value \
				+ eta_gain_ms \
				/ profile.speed_eta_divisor_ms \
				* remaining_cycles \
				* profile.speed_payback_weight
		ArenaItemType.Value.BUBBLE:
			var capacity_pressure: float = (
				1.0
				if self_state.active_bubbles >= self_state.bubble_capacity - 1
				else 0.35
			)
			return profile.bubble_item_value \
				+ capacity_pressure \
				* remaining_cycles \
				* profile.bubble_capacity_pressure_weight
		ArenaItemType.Value.POWER:
			var swing_delta: int = _best_power_swing_delta(
				item.cell,
				snapshot,
				self_state
			)
			return profile.power_item_value \
				+ float(swing_delta) * remaining_cycles * profile.power_swing_weight
		_:
			return 0.0


func _speed_target_eta_gain_ms(
		origin: Vector2i,
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState
	) -> float:
	var current_cell_ms: float = GameConstants.CELL_SIZE \
		/ maxf(1.0, self_state.move_speed) * 1000.0
	var upgraded_cell_ms: float = GameConstants.CELL_SIZE \
		/ maxf(
			1.0,
			self_state.move_speed + GameConstants.SPEED_PER_STAGE_ITEM
		) * 1000.0
	var gain_per_cell_ms: float = current_cell_ms - upgraded_cell_ms
	var best_gain_ms: float = 0.0
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			if snapshot.is_paint_locked(cell):
				continue
			var owner_team: int = snapshot.paint_owner(cell)
			if owner_team == self_state.team_id:
				continue
			var target_value: float = (
				2.0
				if owner_team != PaintPalette.TEAM_NEUTRAL
				else 1.0
			)
			var nearby_value: float = target_value
			for direction: Vector2i in AITemporalPlanner.CARDINAL_DIRECTIONS:
				var neighbor: Vector2i = cell + direction
				if not GameConstants.is_inside(neighbor) \
						or snapshot.is_paint_locked(neighbor):
					continue
				var neighbor_owner: int = snapshot.paint_owner(neighbor)
				if neighbor_owner != self_state.team_id:
					nearby_value += (
						2.0
						if neighbor_owner != PaintPalette.TEAM_NEUTRAL
						else 1.0
					)
			var weighted_distance: float = float(_manhattan(origin, cell)) \
				* clampf(nearby_value / 2.0, 1.0, 4.0)
			best_gain_ms = maxf(
				best_gain_ms,
				weighted_distance * gain_per_cell_ms
			)
	for other_item: AIBattleSnapshot.ItemState in snapshot.items:
		if other_item.cell == origin:
			continue
		best_gain_ms = maxf(
			best_gain_ms,
			float(_manhattan(origin, other_item.cell)) * gain_per_cell_ms * 1.5
		)
	return minf(best_gain_ms, 2000.0)


func _best_power_swing_delta(
		origin: Vector2i,
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState
	) -> int:
	var best_delta: int = 0
	var candidates: Array[Vector2i] = [origin]
	for direction: Vector2i in AITemporalPlanner.CARDINAL_DIRECTIONS:
		var candidate: Vector2i = origin + direction
		if _snapshot_walkable(snapshot, candidate):
			candidates.append(candidate)
	for cell: Vector2i in candidates:
		var current_blast: Array[Vector2i] = GameRules.blast_cells(
			cell,
			self_state.power,
			snapshot.cells
		)
		var upgraded_blast: Array[Vector2i] = GameRules.blast_cells(
			cell,
			self_state.power + 1,
			snapshot.cells
		)
		best_delta = maxi(
			best_delta,
			snapshot.territory_swing(upgraded_blast, self_state.team_id) \
				- snapshot.territory_swing(current_blast, self_state.team_id)
		)
	return best_delta


func _nearest_enemy_item_eta(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		item_cell: Vector2i
	) -> int:
	var result: int = 999999
	for other: AIBattleSnapshot.ActorState in snapshot.actors:
		if other.team_id == self_state.team_id or other.is_dead or other.is_trapped:
			continue
		var cell_ms: float = GameConstants.CELL_SIZE / maxf(1.0, other.move_speed) * 1000.0
		result = mini(result, roundi(float(_manhattan(other.cell, item_cell)) * cell_ms))
	return result


func _commit_item_decision(decision: Dictionary) -> bool:
	var item_id: int = int(decision.get("item_id", 0))
	var travel_ms: int = int(decision.get("travel_ms", 0))
	if item_id == 0 or not match_controller.claim_item(
			actor.get_instance_id(),
			item_id,
			travel_ms
		):
		return false
	if _claimed_item_id != 0 and _claimed_item_id != item_id:
		match_controller.release_item_claim(
			_claimed_item_id,
			actor.get_instance_id()
		)
	_claimed_item_id = item_id
	_commit_decision(decision, Mode.COLLECTING)
	return true


func _find_paint_decision(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast
	) -> Dictionary:
	if self_state.active_bubbles >= self_state.bubble_capacity:
		return {}
	if self_state.active_bubbles > 0 and not _bomb_cooldown_ready():
		return {}
	var profile: AIBehaviorProfile = match_controller.ai_profile
	var available_slots: int = self_state.bubble_capacity - self_state.active_bubbles
	var slot_pressure: float = float(available_slots) \
		/ maxf(1.0, float(self_state.bubble_capacity))
	# The end of a verified escape leg is the overwhelmingly preferred barrage
	# placement. Validate it before scanning every reachable tile; with several
	# live bubbles this also avoids rebuilding a large candidate set every tick.
	if self_state.active_bubbles > 0 \
			and not snapshot.is_paint_locked(self_state.cell) \
			and board.can_place_bubble(self_state.cell):
		var current_blast: Array[Vector2i] = GameRules.blast_cells(
			self_state.cell, self_state.power, snapshot.cells
		)
		var current_swing: int = snapshot.territory_swing(
			current_blast, self_state.team_id
		)
		if current_swing > 0:
			var current_escape: AITemporalPlanner.TimedPlan = _virtual_drop_escape(
				snapshot, self_state, forecast, self_state.cell
			)
			if current_escape.valid:
				return {
					"plan": _stationary_plan(self_state.cell),
					"cell": self_state.cell,
					"score": float(current_swing) * profile.paint_swing_weight \
						+ slot_pressure * profile.bubble_slot_fill_weight \
						+ profile.active_barrage_current_cell_bonus,
					"drop": true,
					"escape": current_escape,
				}
	var reachable: Dictionary = AITemporalPlanner.reachable_cells_fast(
		snapshot,
		forecast,
		self_state.cell,
		self_state.move_speed,
		PAINT_APPROACH_MS
	)
	var pending_paint_cells: Dictionary = {}
	for bomb: AIBattleSnapshot.BombState in snapshot.bombs:
		var pending_blast: Array[Vector2i] = GameRules.blast_cells(
			bomb.cell,
			bomb.power,
			snapshot.cells
		)
		for pending_cell: Vector2i in pending_blast:
			pending_paint_cells[pending_cell] = true
	var ranked: Array[Dictionary] = []
	for cell: Vector2i in reachable.keys():
		if snapshot.is_paint_locked(cell) or not board.can_place_bubble(cell):
			continue
		var blast: Array[Vector2i] = GameRules.blast_cells(cell, self_state.power, snapshot.cells)
		var swing: int = snapshot.territory_swing(blast, self_state.team_id)
		if swing <= 0:
			continue
		var overlap_penalty: int = 0
		for blast_cell: Vector2i in blast:
			if pending_paint_cells.has(blast_cell):
				overlap_penalty += 1
		var travel_ms: int = int(reachable[cell])
		var overlap_scale: float = (
			profile.active_barrage_overlap_scale
			if self_state.active_bubbles > 0
			else 1.0
		)
		var score: float = float(swing) * profile.paint_swing_weight \
			- float(overlap_penalty) * profile.pending_overlap_penalty * overlap_scale \
			- float(travel_ms) / profile.paint_travel_divisor \
			+ slot_pressure * profile.bubble_slot_fill_weight \
			+ _rng.randf_range(0.0, 3.0)
		if cell == self_state.cell:
			score += (
				profile.active_barrage_current_cell_bonus
				if self_state.active_bubbles > 0
				else 14.0
			)
		ranked.append({
			"cell": cell,
			"score": score,
			"swing": swing,
		})
	ranked.sort_custom(func(left: Dictionary, right: Dictionary) -> bool:
		return float(left["score"]) > float(right["score"])
	)
	# Once a safe opening bubble is active, prefer dropping again at the end of
	# each verified escape leg. This turns extra capacity into an actual barrage
	# instead of spending the whole fuse walking toward a slightly better tile.
	if self_state.active_bubbles > 0:
		for candidate_index: int in range(ranked.size()):
			if ranked[candidate_index]["cell"] as Vector2i == self_state.cell:
				var current_candidate: Dictionary = ranked.pop_at(candidate_index)
				ranked.push_front(current_candidate)
				break
	for index: int in range(mini(8, ranked.size())):
		var candidate: Dictionary = ranked[index]
		var candidate_cell: Vector2i = candidate["cell"] as Vector2i
		if candidate_cell == self_state.cell:
			var escape: AITemporalPlanner.TimedPlan = _virtual_drop_escape(
				snapshot, self_state, forecast, candidate_cell
			)
			if _bomb_cooldown_ready() and escape.valid:
				return {
					"plan": _stationary_plan(candidate_cell),
					"cell": candidate_cell,
					"score": float(candidate["score"]),
					"drop": true,
					"escape": escape,
				}
			continue
		var path: Array[Vector2i] = board.find_path(self_state.cell, candidate_cell, actor)
		var plan: AITemporalPlanner.TimedPlan = _timed_plan_from_static_path(
			path,
			self_state.move_speed
		)
		if plan.valid and _timed_plan_is_safe(plan, forecast):
			return {
				"plan": plan,
				"cell": candidate_cell,
				"score": float(candidate["score"]),
				"drop": false,
			}
	return {}


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
			self_state.instance_id,
			self_state.team_id
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
	# Patrol is interruptible and low priority. Use the cheap spatial reachability
	# pass to rank destinations, then validate only a few concrete routes against
	# the full hazard timeline.
	var reachable: Dictionary = AITemporalPlanner.reachable_cells_fast(
		snapshot, forecast, self_state.cell, self_state.move_speed, 6000
	)
	var spacious: bool = reachable.size() >= 12
	var ranked: Array[Dictionary] = []
	for cell: Vector2i in candidates:
		var distance: int = _manhattan(self_state.cell, cell)
		if distance < (PATROL_MIN_FLOW_DISTANCE if spacious else 2):
			continue
		if not reachable.has(cell):
			continue
		var score: float = _patrol_flow_score(cell, self_state, target, spacious) \
			- float(reachable[cell]) / 500.0
		ranked.append({"cell": cell, "score": score})
	ranked.sort_custom(func(left: Dictionary, right: Dictionary) -> bool:
		return float(left["score"]) > float(right["score"])
	)
	var reverse_fallback: Dictionary = {}
	for index: int in range(mini(8, ranked.size())):
		var choice: Dictionary = ranked[index]
		var choice_cell: Vector2i = choice["cell"] as Vector2i
		var path: Array[Vector2i] = board.find_path(self_state.cell, choice_cell, actor)
		var plan: AITemporalPlanner.TimedPlan = _timed_plan_from_static_path(
			path, self_state.move_speed
		)
		if not plan.valid or not _timed_plan_is_safe(plan, forecast):
			continue
		if forecast.is_unsafe(
				choice_cell,
				plan.travel_ms(),
				plan.travel_ms() + 600,
				AITemporalPlanner.SAFETY_MARGIN_MS
			):
			continue
		var result: Dictionary = {
			"plan": plan, "cell": choice_cell, "score": float(choice["score"]),
		}
		if _path_reverses_navigation_heading(path):
			if reverse_fallback.is_empty():
				reverse_fallback = result
			continue
		return result
	return reverse_fallback


func _find_static_patrol_decision(
		candidates: Array[Vector2i],
		self_state: AIBattleSnapshot.ActorState,
		target: AIBattleSnapshot.ActorState
	) -> Dictionary:
	var spacious: bool = candidates.size() >= 12
	var ranked: Array[Dictionary] = []
	for cell: Vector2i in candidates:
		var distance: int = _manhattan(self_state.cell, cell)
		if distance < (PATROL_MIN_FLOW_DISTANCE if spacious else 2):
			continue
		ranked.append({
			"cell": cell,
			"score": _patrol_flow_score(cell, self_state, target, spacious),
		})
	ranked.sort_custom(func(left: Dictionary, right: Dictionary) -> bool:
		return float(left["score"]) > float(right["score"])
	)
	var reverse_fallback: Dictionary = {}
	for index: int in range(mini(8, ranked.size())):
		var choice: Dictionary = ranked[index]
		var choice_cell: Vector2i = choice["cell"] as Vector2i
		var path: Array[Vector2i] = board.find_path(self_state.cell, choice_cell, actor)
		var plan: AITemporalPlanner.TimedPlan = _timed_plan_from_static_path(
			path, self_state.move_speed
		)
		if not plan.valid:
			continue
		var result: Dictionary = {
			"plan": plan,
			"cell": choice_cell,
			"score": float(choice["score"]),
			"continuous": true,
		}
		if _path_reverses_navigation_heading(path):
			if reverse_fallback.is_empty():
				reverse_fallback = result
			continue
		return result
	return reverse_fallback


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
	var escape: AITemporalPlanner.TimedPlan = prechecked_escape
	if escape == null or not escape.valid:
		var forecast: AIHazardForecast = match_controller.get_shared_ai_forecast(
			snapshot, PLANNING_HORIZON_MS
		)
		escape = _virtual_drop_escape(snapshot, self_state, forecast, current)
	if not escape.valid:
		return false
	var previous_active: int = actor.stats.active_bubbles
	actor.request_ai_bomb()
	if actor.stats.active_bubbles <= previous_active:
		return false
	_last_bomb_ms = _now_ms()
	if mode == Mode.PRESSURING:
		_pressure_locked_until_ms = _now_ms() + PRESSURE_TARGET_LOCK_MS
	# The route was checked against a virtual bubble with the same owner, fuse,
	# and power, so it remains valid after the real placement.
	_commit_plan(escape, Mode.EVADING, escape.target_cell(), score + 100.0)
	return true


func _virtual_drop_escape(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast,
		cell: Vector2i
	) -> AITemporalPlanner.TimedPlan:
	if self_state.active_bubbles >= self_state.bubble_capacity or not board.can_place_bubble(cell):
		return AITemporalPlanner.TimedPlan.new()
	var detonation_ms: int = int(GameConstants.BUBBLE_FUSE_SECONDS * 1000.0)
	for blast_event: AIHazardForecast.BombBlast in forecast.blast_events:
		if cell in blast_event.cells:
			detonation_ms = mini(detonation_ms, forecast.blast_time_ms(blast_event))
	var blast: Array[Vector2i] = GameRules.blast_cells(
		cell, self_state.power, snapshot.cells
	)
	var escape: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_blast_escape_plan(
		snapshot,
		forecast,
		cell,
		blast,
		self_state.move_speed,
		detonation_ms
	)
	if escape.cells.size() <= 1:
		escape.valid = false
	return escape


func _allies_can_escape(
		snapshot: AIBattleSnapshot,
		self_state: AIBattleSnapshot.ActorState,
		forecast: AIHazardForecast,
		blast: Array[Vector2i]
	) -> bool:
	# AI teammates share paint and do not take friendly-fire damage.
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
	var active_threat_cells: Dictionary = {}
	for bomb: AIBattleSnapshot.BombState in snapshot.bombs:
		if bomb.owner_id == self_state.instance_id:
			_active_pressure_bubbles += 1
	for blast_event: AIHazardForecast.BombBlast in forecast.blast_events:
		if blast_event.owner_id != self_state.instance_id:
			continue
		for threatened_cell: Vector2i in blast_event.cells:
			active_threat_cells[threatened_cell] = true
	_peak_pressure_bubbles = maxi(_peak_pressure_bubbles, _active_pressure_bubbles)
	# Live metrics only need the actually threatened cell count. Building the
	# full three-ring pressure field here duplicated the expensive field that
	# pressure candidate evaluation creates later and scaled poorly in barrages.
	_total_threat_cells = float(active_threat_cells.size())
	if _pressure_target_id == 0:
		return
	var target: AIBattleSnapshot.ActorState = snapshot.actor_by_id(_pressure_target_id)
	if target == null or target.is_dead or target.is_trapped:
		_clear_pressure_target()
		return
	_pressure_target_cell = target.cell
	if _active_pressure_bubbles == 0 and _now_ms() >= _pressure_locked_until_ms:
		_clear_pressure_target()


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


func _record_navigation_cell(current_cell: Vector2i) -> void:
	if current_cell == _last_navigation_cell:
		return
	if _last_navigation_cell != INVALID_CELL:
		var movement: Vector2i = current_cell - _last_navigation_cell
		if absi(movement.x) >= absi(movement.y) and movement.x != 0:
			_navigation_heading = Vector2i(signi(movement.x), 0)
		elif movement.y != 0:
			_navigation_heading = Vector2i(0, signi(movement.y))
	_recent_navigation_cells.append(current_cell)
	while _recent_navigation_cells.size() > RECENT_NAVIGATION_CELL_LIMIT:
		_recent_navigation_cells.pop_front()
	_last_navigation_cell = current_cell


func _patrol_flow_score(
		cell: Vector2i,
		self_state: AIBattleSnapshot.ActorState,
		target: AIBattleSnapshot.ActorState,
		spacious: bool
	) -> float:
	var distance: int = _manhattan(self_state.cell, cell)
	var desired_distance: int = PATROL_DESIRED_FLOW_DISTANCE if spacious else 3
	var score: float = -absf(float(distance - desired_distance)) * 5.0
	if target != null:
		score -= _manhattan(cell, target.cell) * 1.5
	if _navigation_heading != Vector2i.ZERO:
		var offset: Vector2i = cell - self_state.cell
		var forward_amount: int = offset.x * _navigation_heading.x \
			+ offset.y * _navigation_heading.y
		if forward_amount > 0:
			score += 18.0
		elif forward_amount < 0:
			score -= 32.0
	var recent_index: int = _recent_navigation_cells.find(cell)
	if recent_index >= 0:
		score -= 70.0 + recent_index * 4.0
	return score


func _path_reverses_navigation_heading(path: Array[Vector2i]) -> bool:
	if _navigation_heading == Vector2i.ZERO or path.size() < 2:
		return false
	var first_step: Vector2i = path[1] - path[0]
	return first_step == -_navigation_heading


func _open_cells(snapshot: AIBattleSnapshot) -> Array[Vector2i]:
	var result: Array[Vector2i] = []
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			if _snapshot_walkable(snapshot, cell):
				result.append(cell)
	return result


func _snapshot_walkable(snapshot: AIBattleSnapshot, cell: Vector2i) -> bool:
	return GameConstants.is_inside(cell) and GameRules.is_walkable(snapshot.cells[cell.y][cell.x])


func _commit_decision(decision: Dictionary, mode: Mode) -> void:
	_commit_plan(
		decision["plan"] as AITemporalPlanner.TimedPlan,
		mode,
		decision["cell"] as Vector2i,
		float(decision["score"]),
		bool(decision.get("continuous", false))
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
		score: float,
		continuous_motion: bool = false
	) -> void:
	if mode != Mode.COLLECTING:
		_release_item_claim()
	_plan = new_plan
	_plan_index = 1 if new_plan.cells.size() > 1 else new_plan.cells.size()
	_plan_started_ms = _now_ms()
	_continuous_plan_motion = continuous_motion
	_set_debug(mode, target_cell, score)
	if mode != Mode.EVADING:
		_locked_target = target_cell
		var plan_lock_ms: int = TARGET_LOCK_MS
		if mode == Mode.PATROLLING:
			plan_lock_ms = maxi(
				TARGET_LOCK_MS,
				new_plan.travel_ms() + AITemporalPlanner.WAIT_STEP_MS * 2
			)
		_target_locked_until_ms = _now_ms() + plan_lock_ms


func _clear_plan() -> void:
	_release_item_claim()
	_plan = null
	_plan_index = 0
	_continuous_plan_motion = false
	if is_instance_valid(actor):
		actor.set_ai_direction(Vector2.ZERO)


func _set_debug(mode: Mode, target_cell: Vector2i, score: float) -> void:
	current_mode = mode
	decision_target = target_cell
	last_decision_score = score


func _finish_decision(started_usec: int) -> void:
	last_decision_usec = Time.get_ticks_usec() - started_usec
	decision_made.emit(current_mode, decision_target, last_decision_score, last_decision_usec)


func _release_item_claim() -> void:
	if _claimed_item_id == 0:
		return
	if is_instance_valid(match_controller) and is_instance_valid(actor):
		match_controller.release_item_claim(
			_claimed_item_id,
			actor.get_instance_id()
		)
	_claimed_item_id = 0


func _bomb_cooldown_ready() -> bool:
	return _now_ms() - _last_bomb_ms >= BOMB_COOLDOWN_MS


func _now_ms() -> int:
	return match_controller.get_simulation_time_ms() \
		if is_instance_valid(match_controller) else Time.get_ticks_msec()


func _manhattan(left: Vector2i, right: Vector2i) -> int:
	return absi(left.x - right.x) + absi(left.y - right.y)
