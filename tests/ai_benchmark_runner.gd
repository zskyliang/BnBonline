extends SceneTree
## Seeded live-match benchmark for survival, item racing, pressure, and decision latency.

const SURVIVAL_SCENARIOS: int = 40
const ITEM_SCENARIOS: int = 30
const PRESSURE_SCENARIOS: int = 30
const SURVIVAL_GATE: float = 0.95
const ITEM_GATE: float = 0.70
const PRESSURE_GATE: float = 0.80
const ATTACK_SURVIVAL_GATE: float = 0.90
const MULTI_BUBBLE_GATE: float = 0.85
const THREAT_UPLIFT_GATE: float = 0.80
const THREAT_UPLIFT_MULTIPLIER: float = 1.50
const PRESSURE_COVERAGE_GATE: float = 0.45
const BENCHMARK_TIME_SCALE: float = 12.0
const BENCHMARK_PHYSICS_TICKS: int = 720

var _match: MatchController
var _decision_times_usec: PackedInt32Array = PackedInt32Array()
var _decision_times_by_mode: Dictionary = {}


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	Engine.time_scale = BENCHMARK_TIME_SCALE
	Engine.physics_ticks_per_second = BENCHMARK_PHYSICS_TICKS
	var packed: PackedScene = load("res://scenes/main.tscn") as PackedScene
	_match = packed.instantiate() as MatchController
	root.add_child(_match)
	await process_frame
	_match.settings.ai_count = 1
	_match.settings.max_speed = 300
	_match.settings.max_bubbles = 8
	_match.settings.max_power = 10
	var survival_passed: int = await _run_survival_suite()
	var item_passed: int = await _run_item_suite()
	var pressure_result: Dictionary = await _run_pressure_suite()
	_release_movement()
	for battle_actor: GameActor in _match.get_actors():
		if battle_actor.stats.is_trapped:
			battle_actor.rescue()
		battle_actor.stats.invincible_until_ms = Time.get_ticks_msec() + 10000
		for child: Node in battle_actor.get_children():
			if child is RuleAI:
				(child as RuleAI).stop_thinking()
	await _wait_simulation_ms(2800)
	Engine.time_scale = 1.0
	Engine.physics_ticks_per_second = 60
	var survival_rate: float = float(survival_passed) / SURVIVAL_SCENARIOS
	var item_rate: float = float(item_passed) / ITEM_SCENARIOS
	var pressure_rate: float = float(pressure_result["passed"]) / PRESSURE_SCENARIOS
	var trap_rate: float = float(pressure_result["trapped"]) / PRESSURE_SCENARIOS
	var attack_survival_rate: float = float(pressure_result["survived"]) / PRESSURE_SCENARIOS
	var multi_bubble_rate: float = float(pressure_result["multi_bubble"]) / PRESSURE_SCENARIOS
	var threat_uplift_rate: float = float(pressure_result["threat_uplift"]) / PRESSURE_SCENARIOS
	var average_usec: float = _average_decision_usec()
	var p95_usec: int = _percentile_decision_usec(0.95)
	print("AI benchmark survival: %d/%d = %.1f%% (gate %.0f%%)" % [
		survival_passed, SURVIVAL_SCENARIOS, survival_rate * 100.0, SURVIVAL_GATE * 100.0,
	])
	print("AI benchmark item race: %d/%d = %.1f%% (gate %.0f%%)" % [
		item_passed, ITEM_SCENARIOS, item_rate * 100.0, ITEM_GATE * 100.0,
	])
	print("AI benchmark pressure: %d/%d = %.1f%% (gate %.0f%%), trap/kill %.1f%%, attack survival %.1f%%" % [
		int(pressure_result["passed"]), PRESSURE_SCENARIOS, pressure_rate * 100.0,
		PRESSURE_GATE * 100.0, trap_rate * 100.0, attack_survival_rate * 100.0,
	])
	print("AI benchmark barrage: multi-bubble %d/%d = %.1f%% (gate %.0f%%), threat uplift %d/%d = %.1f%% (gate %.0f%%)" % [
		int(pressure_result["multi_bubble"]), PRESSURE_SCENARIOS, multi_bubble_rate * 100.0,
		MULTI_BUBBLE_GATE * 100.0,
		int(pressure_result["threat_uplift"]), PRESSURE_SCENARIOS, threat_uplift_rate * 100.0,
		THREAT_UPLIFT_GATE * 100.0,
	])
	print("AI benchmark pressure peaks: bubbles %.2f, threat %.2f cells, target coverage %.1f%%, single-bubble uplift %.2fx" % [
		float(pressure_result["average_peak_bubbles"]),
		float(pressure_result["average_peak_threat"]),
		float(pressure_result["average_peak_coverage"]) * 100.0,
		float(pressure_result["average_peak_uplift"]),
	])
	print("AI benchmark decisions: %d samples, mean %.2f ms, p95 %.2f ms" % [
		_decision_times_usec.size(), average_usec / 1000.0, p95_usec / 1000.0,
	])
	for mode: int in _decision_times_by_mode.keys():
		var mode_times: PackedInt32Array = _decision_times_by_mode[mode] as PackedInt32Array
		print("  %s: %d samples, mean %.2f ms, p95 %.2f ms" % [
			RuleAI.Mode.keys()[mode], mode_times.size(),
			_average_of(mode_times) / 1000.0, _percentile_of(mode_times, 0.95) / 1000.0,
		])
	var failed: bool = survival_rate < SURVIVAL_GATE \
		or item_rate < ITEM_GATE \
		or pressure_rate < PRESSURE_GATE \
		or attack_survival_rate < ATTACK_SURVIVAL_GATE \
		or multi_bubble_rate < MULTI_BUBBLE_GATE \
		or threat_uplift_rate < THREAT_UPLIFT_GATE \
		or average_usec >= 5000.0 \
		or p95_usec >= 10000
	var audio_manager: Node = root.get_node_or_null("AudioManager")
	if is_instance_valid(audio_manager):
		audio_manager.call("stop_all")
	_match.queue_free()
	await process_frame
	await process_frame
	print("AI benchmark: %s" % ("FAILED" if failed else "PASS"))
	quit(1 if failed else 0)


func _run_survival_suite() -> int:
	var passed: int = 0
	for scenario: int in range(SURVIVAL_SCENARIOS):
		var context: Dictionary = await _prepare_empty_scenario(1000 + scenario)
		var ai_actor: GameActor = context["ai"] as GameActor
		var player: GameActor = context["player"] as GameActor
		var controller: RuleAI = context["controller"] as RuleAI
		player.stats.is_dead = true
		player.visible = false
		ai_actor.stats.bubble_capacity = 8
		ai_actor.position = GameConstants.grid_to_world(Vector2i(7, 6))
		var state: Dictionary = {"failed": false, "failed_at": -1, "failed_cell": Vector2i(-1, -1)}
		ai_actor.trapped.connect(func(_victim: GameActor, _attacker: GameActor) -> void:
			state["failed"] = true
			state["failed_at"] = _match.get_simulation_time_ms()
			state["failed_cell"] = ai_actor.current_cell()
		)
		ai_actor.died.connect(func(_victim: GameActor, _attacker: GameActor) -> void:
			state["failed"] = true
			if int(state["failed_at"]) < 0:
				state["failed_at"] = _match.get_simulation_time_ms()
				state["failed_cell"] = ai_actor.current_cell()
		)
		var offsets: Array[Vector2i] = [
			Vector2i(0, -3), Vector2i(4, 0), Vector2i.ZERO,
			Vector2i(-4, 2), Vector2i(3, 3), Vector2i(-3, -3),
		]
		var fuses: Array[float] = [0.85, 1.25, 2.20, 1.70, 2.60, 2.90]
		var powers: PackedInt32Array = PackedInt32Array([3, 4, 3, 3, 2, 2])
		var bubble_count: int = 3 + scenario % 4
		for index: int in range(bubble_count):
			var cell: Vector2i = Vector2i(7, 6) + _rotate(offsets[index], scenario % 4)
			var owner: GameActor = ai_actor if offsets[index] == Vector2i.ZERO else player
			_spawn_test_bubble(owner, cell, powers[index], fuses[index] + (scenario % 3) * 0.03)
		controller.reconsider_now()
		await _wait_simulation_ms(3700)
		if not bool(state["failed"]) and not ai_actor.stats.is_trapped and not ai_actor.stats.is_dead:
			passed += 1
		elif scenario < 4:
			print("Survival diagnostic #%d: cell=%s mode=%s target=%s failed=%s at=%d hit_cell=%s" % [
				scenario, ai_actor.current_cell(), controller.mode_name(),
				controller.decision_target, state["failed"], state["failed_at"], state["failed_cell"],
			])
	return passed


func _run_item_suite() -> int:
	var passed: int = 0
	for scenario: int in range(ITEM_SCENARIOS):
		var context: Dictionary = await _prepare_empty_scenario(2000 + scenario)
		var ai_actor: GameActor = context["ai"] as GameActor
		var player: GameActor = context["player"] as GameActor
		var controller: RuleAI = context["controller"] as RuleAI
		var item_cell := Vector2i(7, 6)
		var axis: Vector2i = Vector2i.RIGHT if scenario % 2 == 0 else Vector2i.DOWN
		var side: int = -1 if scenario % 4 < 2 else 1
		ai_actor.position = GameConstants.grid_to_world(item_cell + axis * 3 * side)
		player.position = GameConstants.grid_to_world(item_cell - axis * 4 * side)
		if scenario % 2 == 0:
			ai_actor.process_physics_priority = -1
			player.process_physics_priority = 0
		else:
			ai_actor.process_physics_priority = 0
			player.process_physics_priority = -1
		var item_codes: PackedInt32Array = PackedInt32Array([
			GameConstants.ITEM_BUBBLE, GameConstants.ITEM_SPEED, GameConstants.ITEM_POWER,
		])
		_match.board.cells[item_cell.y][item_cell.x] = item_codes[scenario % item_codes.size()]
		var state: Dictionary = {"collector": 0, "ai_failed": false}
		ai_actor.item_collected.connect(func(_collector: GameActor, _code: int) -> void:
			state["collector"] = ai_actor.get_instance_id()
		)
		player.item_collected.connect(func(_collector: GameActor, _code: int) -> void:
			state["collector"] = player.get_instance_id()
		)
		ai_actor.trapped.connect(func(_victim: GameActor, _attacker: GameActor) -> void:
			state["ai_failed"] = true
		)
		controller.reconsider_now()
		var deadline: int = _match.get_simulation_time_ms() + 2200
		while _match.get_simulation_time_ms() < deadline and int(state["collector"]) == 0:
			_drive_player_toward(player, item_cell)
			await physics_frame
		_release_movement()
		if int(state["collector"]) == ai_actor.get_instance_id() and not bool(state["ai_failed"]):
			passed += 1
		elif scenario < 5:
			print("Item diagnostic #%d: collector=%d ai_cell=%s player_cell=%s mode=%s target=%s" % [
				scenario, state["collector"], ai_actor.current_cell(), player.current_cell(),
				controller.mode_name(), controller.decision_target,
			])
	return passed


func _run_pressure_suite() -> Dictionary:
	var passed: int = 0
	var trapped: int = 0
	var survived: int = 0
	var multi_bubble: int = 0
	var threat_uplift: int = 0
	var peak_bubbles_total: int = 0
	var peak_threat_total: float = 0.0
	var peak_coverage_total: float = 0.0
	var peak_uplift_total: float = 0.0
	for scenario: int in range(PRESSURE_SCENARIOS):
		var context: Dictionary = await _prepare_empty_scenario(3000 + scenario)
		var ai_actor: GameActor = context["ai"] as GameActor
		var player: GameActor = context["player"] as GameActor
		var controller: RuleAI = context["controller"] as RuleAI
		_configure_pressure_layout(scenario, ai_actor, player)
		ai_actor.stats.move_speed = 275.0
		player.stats.move_speed = 275.0
		ai_actor.stats.power = 3
		ai_actor.stats.bubble_capacity = 4
		var state: Dictionary = {"ai_failed": false, "target_harmed": false, "placed": false}
		ai_actor.trapped.connect(func(_victim: GameActor, _attacker: GameActor) -> void:
			state["ai_failed"] = true
		)
		ai_actor.died.connect(func(_victim: GameActor, _attacker: GameActor) -> void:
			state["ai_failed"] = true
		)
		player.trapped.connect(func(_victim: GameActor, _attacker: GameActor) -> void:
			state["target_harmed"] = true
		)
		player.died.connect(func(_victim: GameActor, _attacker: GameActor) -> void:
			state["target_harmed"] = true
		)
		controller.reconsider_now()
		var max_reduction: float = 0.0
		var max_coverage: float = 0.0
		var max_active_bubbles: int = 0
		var max_threat_cells: float = 0.0
		var max_uplift: float = 1.0
		var next_metric_sample_ms: int = _match.get_simulation_time_ms()
		var deadline: int = _match.get_simulation_time_ms() + 8000
		while _match.get_simulation_time_ms() < deadline:
			_drive_player_to_safety(player)
			if controller.active_pressure_bubbles > 0:
				state["placed"] = true
			max_active_bubbles = maxi(max_active_bubbles, controller.active_pressure_bubbles)
			max_threat_cells = maxf(max_threat_cells, controller.total_threat_cells)
			max_coverage = maxf(max_coverage, controller.target_threat_coverage)
			max_reduction = maxf(max_reduction, controller.last_pressure_reduction)
			if _match.get_simulation_time_ms() >= next_metric_sample_ms:
				max_uplift = maxf(max_uplift, _current_threat_uplift(ai_actor, player))
				next_metric_sample_ms += AITemporalPlanner.WAIT_STEP_MS
			await physics_frame
		_release_movement()
		if bool(state["target_harmed"]):
			trapped += 1
		if not bool(state["ai_failed"]):
			survived += 1
		if max_active_bubbles >= 2:
			multi_bubble += 1
		if max_uplift >= THREAT_UPLIFT_MULTIPLIER:
			threat_uplift += 1
		peak_bubbles_total += max_active_bubbles
		peak_threat_total += max_threat_cells
		peak_coverage_total += max_coverage
		peak_uplift_total += max_uplift
		var spatially_pressured: bool = max_reduction >= 0.30 \
			and max_coverage >= PRESSURE_COVERAGE_GATE
		if bool(state["placed"]) \
				and not bool(state["ai_failed"]) \
				and (bool(state["target_harmed"]) or spatially_pressured):
			passed += 1
		elif scenario < 6:
			print("Pressure diagnostic #%d: layout=%d placed=%s failed=%s harmed=%s peak_bombs=%d reduction=%.2f coverage=%.2f threat=%.2f uplift=%.2f" % [
				scenario, scenario % 3, state["placed"], state["ai_failed"], state["target_harmed"],
				max_active_bubbles, max_reduction, max_coverage, max_threat_cells, max_uplift,
			])
		if max_uplift < THREAT_UPLIFT_MULTIPLIER and scenario < 9:
			print("Barrage diagnostic #%d: layout=%d peak_bombs=%d threat=%.2f coverage=%.2f uplift=%.2f" % [
				scenario, scenario % 3, max_active_bubbles, max_threat_cells, max_coverage, max_uplift,
			])
	return {
		"passed": passed,
		"trapped": trapped,
		"survived": survived,
		"multi_bubble": multi_bubble,
		"threat_uplift": threat_uplift,
		"average_peak_bubbles": float(peak_bubbles_total) / PRESSURE_SCENARIOS,
		"average_peak_threat": peak_threat_total / PRESSURE_SCENARIOS,
		"average_peak_coverage": peak_coverage_total / PRESSURE_SCENARIOS,
		"average_peak_uplift": peak_uplift_total / PRESSURE_SCENARIOS,
	}


func _configure_pressure_layout(
		scenario: int,
		ai_actor: GameActor,
		player: GameActor
	) -> void:
	var center := Vector2i(7, 6)
	var layout: int = scenario % 3
	var turn: int = (scenario / 3) % 4
	if layout == 0:
		for y: int in range(GameConstants.GRID_ROWS):
			for x: int in range(GameConstants.GRID_COLUMNS):
				_match.board.cells[y][x] = 0
		var islands: Array[Vector2i] = [
			Vector2i(0, -2), Vector2i(0, 2), Vector2i(2, 2), Vector2i(-2, -2),
		]
		for index: int in range(islands.size()):
			if index == scenario % islands.size():
				continue
			var island: Vector2i = center + _rotate(islands[index], turn)
			_match.board.cells[island.y][island.x] = 1
	else:
		for y: int in range(GameConstants.GRID_ROWS):
			for x: int in range(GameConstants.GRID_COLUMNS):
				_match.board.cells[y][x] = 1
		var open_offsets: Array[Vector2i] = []
		if layout == 1:
			for x: int in range(-4, 5):
				for y: int in range(-2, 3):
					open_offsets.append(Vector2i(x, y))
			for y: int in range(-3, 4):
				for x: int in range(-2, 3):
					open_offsets.append(Vector2i(x, y))
		else:
			for x: int in range(-4, 5):
				for y: int in range(-2, 3):
					open_offsets.append(Vector2i(x, y))
			for branch_x: int in [-3, 3]:
				for y: int in range(-3, 4):
					open_offsets.append(Vector2i(branch_x, y))
		for offset: Vector2i in open_offsets:
			var open_cell: Vector2i = center + _rotate(offset, turn)
			_match.board.cells[open_cell.y][open_cell.x] = 0
	var ai_offset: Vector2i = Vector2i(-4, 0) if layout < 2 else Vector2i(-3, 0)
	var player_offset: Vector2i = Vector2i(4, 0) if layout < 2 else Vector2i(3, 0)
	ai_actor.position = GameConstants.grid_to_world(center + _rotate(ai_offset, turn))
	player.position = GameConstants.grid_to_world(center + _rotate(player_offset, turn))


func _current_threat_uplift(ai_actor: GameActor, player: GameActor) -> float:
	var snapshot: AIBattleSnapshot = _match.build_ai_snapshot()
	var forecast: AIHazardForecast = AIHazardForecast.build(snapshot, 5500)
	var owned_events: Array[AIHazardForecast.BombBlast] = []
	for blast_event: AIHazardForecast.BombBlast in forecast.blast_events:
		if blast_event.owner_id == ai_actor.get_instance_id():
			owned_events.append(blast_event)
	if owned_events.size() < 2:
		return 1.0
	var base_snapshot: AIBattleSnapshot = _snapshot_without_owner_bombs(
		snapshot, ai_actor.get_instance_id()
	)
	var base_forecast: AIHazardForecast = AIHazardForecast.build(base_snapshot, 5500)
	var target_state: AIBattleSnapshot.ActorState = base_snapshot.actor_by_id(player.get_instance_id())
	if target_state == null:
		return 1.0
	var relevant_cells: Dictionary = AITemporalPlanner.reachable_cells_fast(
		base_snapshot,
		base_forecast,
		target_state.cell,
		target_state.move_speed,
		RuleAI.PRESSURE_HORIZON_MS
	)
	if relevant_cells.is_empty():
		return 1.0
	var combined: AIThreatField = AIThreatField.build(forecast, ai_actor.get_instance_id())
	var combined_weight: float = combined.weight_sum(relevant_cells)
	var best_single_weight: float = 0.0
	for owned_event: AIHazardForecast.BombBlast in owned_events:
		var single: AIThreatField = AIThreatField.build_for_blast(forecast, owned_event)
		best_single_weight = maxf(best_single_weight, single.weight_sum(relevant_cells))
	if best_single_weight <= 0.001:
		return 1.0
	return combined_weight / best_single_weight


func _snapshot_without_owner_bombs(
		snapshot: AIBattleSnapshot,
		owner_id: int
	) -> AIBattleSnapshot:
	var result := AIBattleSnapshot.new()
	result.cells = snapshot.clone_cells()
	for bomb: AIBattleSnapshot.BombState in snapshot.bombs:
		if bomb.owner_id != owner_id:
			result.bombs.append(bomb)
	for explosion: AIBattleSnapshot.ExplosionState in snapshot.explosions:
		result.explosions.append(explosion)
	for actor_state: AIBattleSnapshot.ActorState in snapshot.actors:
		result.actors.append(actor_state)
	for item_cell: Vector2i in snapshot.item_cells:
		result.item_cells.append(item_cell)
	return result


func _prepare_empty_scenario(seed: int) -> Dictionary:
	_release_movement()
	_match.settings.map_id = "classic"
	_match.settings.ai_count = 1
	_match.start_match()
	for value: Variant in _match.board.bombs.values().duplicate():
		var existing_bubble: GameBubble = value as GameBubble
		if not is_instance_valid(existing_bubble):
			continue
		_match.board.unregister_bubble(existing_bubble)
		if is_instance_valid(existing_bubble.bubble_owner):
			existing_bubble.bubble_owner.stats.active_bubbles = maxi(
				0, existing_bubble.bubble_owner.stats.active_bubbles - 1
			)
		existing_bubble.queue_free()
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			_match.board.cells[y][x] = 0
	var player: GameActor = _match.get_player()
	var ai_actor: GameActor
	for battle_actor: GameActor in _match.get_actors():
		if not battle_actor.is_player:
			ai_actor = battle_actor
			break
	var controller: RuleAI
	for child: Node in ai_actor.get_children():
		if child is RuleAI:
			controller = child as RuleAI
			break
	await process_frame
	controller.reset_for_scenario(seed)
	controller.decision_made.connect(func(
			mode: int, _target: Vector2i, _score: float, elapsed_usec: int
		) -> void:
		_decision_times_usec.append(elapsed_usec)
		var mode_times: PackedInt32Array = _decision_times_by_mode.get(mode, PackedInt32Array()) as PackedInt32Array
		mode_times.append(elapsed_usec)
		_decision_times_by_mode[mode] = mode_times
	)
	return {"player": player, "ai": ai_actor, "controller": controller}


func _spawn_test_bubble(
		owner: GameActor,
		cell: Vector2i,
		power: int,
		fuse_seconds: float
	) -> void:
	var previous_power: int = owner.stats.power
	owner.stats.power = power
	var bubble := GameBubble.new()
	_match._entity_root.add_child(bubble)
	bubble.setup(owner, cell, "football", fuse_seconds)
	bubble.exploded.connect(_match._on_bubble_exploded)
	_match.board.register_bubble(bubble)
	owner.stats.active_bubbles += 1
	owner.stats.power = previous_power


func _wait_simulation_ms(duration_ms: int) -> void:
	var deadline: int = _match.get_simulation_time_ms() + duration_ms
	while _match.get_simulation_time_ms() < deadline:
		await physics_frame


func _drive_player_toward(player: GameActor, target_cell: Vector2i) -> void:
	var difference: Vector2i = target_cell - player.current_cell()
	var direction := Vector2i.ZERO
	if absi(difference.x) >= absi(difference.y) and difference.x != 0:
		direction.x = signi(difference.x)
	elif difference.y != 0:
		direction.y = signi(difference.y)
	_set_player_direction(direction)


func _drive_player_to_safety(player: GameActor) -> void:
	if player.stats.is_dead or player.stats.is_trapped:
		_release_movement()
		return
	var snapshot: AIBattleSnapshot = _match.build_ai_snapshot()
	var forecast: AIHazardForecast = AIHazardForecast.build(snapshot, 4000)
	var best_direction := Vector2i.ZERO
	var best_score: float = -INF
	var directions: Array[Vector2i] = [
		Vector2i.ZERO, Vector2i.UP, Vector2i.DOWN, Vector2i.LEFT, Vector2i.RIGHT,
	]
	for direction: Vector2i in directions:
		var candidate: Vector2i = player.current_cell() + direction
		if not GameConstants.is_inside(candidate) \
				or not GameRules.is_walkable(snapshot.cells[candidate.y][candidate.x]):
			continue
		if direction != Vector2i.ZERO and forecast.is_bomb_blocked(candidate, 0, 350):
			continue
		var eta: int = mini(5000, forecast.danger_eta_ms(candidate))
		var open_neighbors: int = 0
		for neighbor_direction: Vector2i in AITemporalPlanner.CARDINAL_DIRECTIONS:
			var neighbor: Vector2i = candidate + neighbor_direction
			if GameConstants.is_inside(neighbor) \
					and GameRules.is_walkable(snapshot.cells[neighbor.y][neighbor.x]):
				open_neighbors += 1
		var score: float = eta + open_neighbors * 80.0
		if score > best_score:
			best_score = score
			best_direction = direction
	_set_player_direction(best_direction)


func _set_player_direction(direction: Vector2i) -> void:
	_release_movement()
	if direction == Vector2i.LEFT:
		Input.action_press("move_left")
	elif direction == Vector2i.RIGHT:
		Input.action_press("move_right")
	elif direction == Vector2i.UP:
		Input.action_press("move_up")
	elif direction == Vector2i.DOWN:
		Input.action_press("move_down")


func _release_movement() -> void:
	for action: StringName in [&"move_left", &"move_right", &"move_up", &"move_down"]:
		if InputMap.has_action(action):
			Input.action_release(action)


func _rotate(offset: Vector2i, quarter_turns: int) -> Vector2i:
	var result: Vector2i = offset
	for _turn: int in range(posmod(quarter_turns, 4)):
		result = Vector2i(-result.y, result.x)
	return result


func _average_decision_usec() -> float:
	return _average_of(_decision_times_usec)


func _average_of(values: PackedInt32Array) -> float:
	if values.is_empty():
		return 999999.0
	var total: int = 0
	for elapsed_usec: int in values:
		total += elapsed_usec
	return float(total) / values.size()


func _percentile_decision_usec(percentile: float) -> int:
	return _percentile_of(_decision_times_usec, percentile)


func _percentile_of(values: PackedInt32Array, percentile: float) -> int:
	if values.is_empty():
		return 999999
	var sorted: PackedInt32Array = values.duplicate()
	sorted.sort()
	var index: int = clampi(ceili(sorted.size() * percentile) - 1, 0, sorted.size() - 1)
	return sorted[index]
