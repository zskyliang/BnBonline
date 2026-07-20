extends SceneTree
## Four-AI scheduling and main-thread decision burst regression benchmark.

const STRESS_AI_COUNT: int = 4
const STRESS_DURATION_MS: int = 7000
const WARMUP_MS: int = 700
const BENCHMARK_TIME_SCALE: float = 12.0
const BENCHMARK_PHYSICS_TICKS: int = 720
const MEAN_DECISION_GATE_USEC: float = 5000.0
const P95_DECISION_GATE_USEC: int = 10000
const P99_FRAME_BURST_GATE_USEC: int = 16667
const MOVEMENT_RATIO_GATE: float = 0.78

var _match: MatchController
var _sample_started_ms: int = 0
var _decision_times_usec: PackedInt32Array = PackedInt32Array()
var _frame_decision_usec: Dictionary = {}
var _frame_decision_count: Dictionary = {}
var _controller_decision_count: Dictionary = {}
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
	_match.settings.ai_count = STRESS_AI_COUNT
	_match.settings.max_speed = 300
	_match.settings.max_bubbles = 8
	_match.settings.max_power = 10
	_match.start_match()
	_configure_stress_arena()
	_connect_decision_metrics()
	await _wait_simulation_ms(WARMUP_MS)
	_sample_started_ms = _match.get_simulation_time_ms()
	await _wait_simulation_ms(STRESS_DURATION_MS)
	var average_usec: float = _average_of(_decision_times_usec)
	var p95_usec: int = _percentile_of(_decision_times_usec, 0.95)
	var frame_bursts: PackedInt32Array = PackedInt32Array()
	var max_decisions_in_frame: int = 0
	var clustered_frames: int = 0
	for frame: int in _frame_decision_usec.keys():
		frame_bursts.append(int(_frame_decision_usec[frame]))
		var count: int = int(_frame_decision_count.get(frame, 0))
		max_decisions_in_frame = maxi(max_decisions_in_frame, count)
		if count > 1:
			clustered_frames += 1
	var p95_frame_burst: int = _percentile_of(frame_bursts, 0.95)
	var p99_frame_burst: int = _percentile_of(frame_bursts, 0.99)
	var minimum_controller_decisions: int = 999999
	var maximum_controller_decisions: int = 0
	for actor_id: int in _controller_decision_count.keys():
		var count: int = int(_controller_decision_count[actor_id])
		minimum_controller_decisions = mini(minimum_controller_decisions, count)
		maximum_controller_decisions = maxi(maximum_controller_decisions, count)
	print("AI stress decisions: %d samples, mean %.2f ms, p95 %.2f ms" % [
		_decision_times_usec.size(), average_usec / 1000.0, p95_usec / 1000.0,
	])
	print("AI stress frame bursts: %d active frames, p95 %.2f ms, p99 %.2f ms, max decisions/frame %d, clustered frames %d" % [
		frame_bursts.size(), p95_frame_burst / 1000.0, p99_frame_burst / 1000.0,
		max_decisions_in_frame, clustered_frames,
	])
	print("AI stress scheduling fairness: min %d, max %d decisions/controller" % [
		minimum_controller_decisions, maximum_controller_decisions,
	])
	for mode: int in _decision_times_by_mode.keys():
		var mode_times: PackedInt32Array = _decision_times_by_mode[mode] as PackedInt32Array
		print("  %s: %d samples, mean %.2f ms, p95 %.2f ms" % [
			RuleAI.Mode.keys()[mode], mode_times.size(),
			_average_of(mode_times) / 1000.0, _percentile_of(mode_times, 0.95) / 1000.0,
		])
	var movement_result: Dictionary = await _run_safe_movement_probe()
	print("AI stress safe movement: %.1f%% moving frames, %d visited cells, %d immediate reversals" % [
		float(movement_result["moving_ratio"]) * 100.0,
		int(movement_result["visited_cells"]),
		int(movement_result["immediate_reversals"]),
	])
	print("AI stress safe movement idle: max %d frames, direction active %.1f%%" % [
		int(movement_result["max_idle_frames"]),
		float(movement_result["direction_active_ratio"]) * 100.0,
	])
	var failed: bool = _decision_times_usec.size() < STRESS_AI_COUNT * 30 \
		or average_usec >= MEAN_DECISION_GATE_USEC \
		or p95_usec >= P95_DECISION_GATE_USEC \
		or p99_frame_burst >= P99_FRAME_BURST_GATE_USEC \
		or max_decisions_in_frame > 1 \
		or maximum_controller_decisions - minimum_controller_decisions > 1 \
		or float(movement_result["moving_ratio"]) < MOVEMENT_RATIO_GATE \
		or int(movement_result["immediate_reversals"]) > 0
	Engine.time_scale = 1.0
	Engine.physics_ticks_per_second = 60
	var audio_manager: Node = root.get_node_or_null("AudioManager")
	if is_instance_valid(audio_manager):
		audio_manager.call("stop_all")
	_match.queue_free()
	await process_frame
	await process_frame
	print("AI stress benchmark: %s" % ("FAILED" if failed else "PASS"))
	quit(1 if failed else 0)


func _configure_stress_arena() -> void:
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			_match.board.cells[y][x] = 0
	var center := Vector2i(7, 6)
	var ai_offsets: Array[Vector2i] = [
		Vector2i(-4, 0), Vector2i(4, 0), Vector2i(0, -4), Vector2i(0, 4),
	]
	var ai_index: int = 0
	for battle_actor: GameActor in _match.get_actors():
		battle_actor.stats.move_speed = 275.0
		battle_actor.stats.power = 3
		battle_actor.stats.bubble_capacity = 4
		if battle_actor.is_player:
			battle_actor.position = GameConstants.grid_to_world(center)
			continue
		battle_actor.position = GameConstants.grid_to_world(center + ai_offsets[ai_index])
		ai_index += 1


func _connect_decision_metrics() -> void:
	for battle_actor: GameActor in _match.get_actors():
		if battle_actor.is_player:
			continue
		var actor_id: int = battle_actor.get_instance_id()
		_controller_decision_count[actor_id] = 0
		for child: Node in battle_actor.get_children():
			if child is RuleAI:
				(child as RuleAI).decision_made.connect(_on_decision_made.bind(actor_id))


func _run_safe_movement_probe() -> Dictionary:
	_match.settings.map_id = "classic"
	_match.settings.ai_count = 1
	_match.start_match()
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			_match.board.cells[y][x] = 0
	var ai_actor: GameActor
	var controller: RuleAI
	for battle_actor: GameActor in _match.get_actors():
		if battle_actor.is_player:
			battle_actor.stats.is_dead = true
			battle_actor.visible = false
			continue
		ai_actor = battle_actor
		ai_actor.position = GameConstants.grid_to_world(Vector2i(7, 6))
		ai_actor.stats.move_speed = 200.0
		for child: Node in ai_actor.get_children():
			if child is RuleAI:
				controller = child as RuleAI
				break
	controller.reset_for_scenario(9001)
	controller.reconsider_now()
	await _wait_simulation_ms(300)
	var previous_position: Vector2 = ai_actor.position
	var transition_cells: Array[Vector2i] = [ai_actor.current_cell()]
	var visited_cells: Dictionary = {ai_actor.current_cell(): true}
	var moving_frames: int = 0
	var direction_active_frames: int = 0
	var total_frames: int = 0
	var immediate_reversals: int = 0
	var current_idle_frames: int = 0
	var max_idle_frames: int = 0
	var deadline: int = _match.get_simulation_time_ms() + 4000
	while _match.get_simulation_time_ms() < deadline:
		await physics_frame
		total_frames += 1
		if ai_actor.position.distance_to(previous_position) > 0.05:
			moving_frames += 1
			current_idle_frames = 0
		else:
			current_idle_frames += 1
			max_idle_frames = maxi(max_idle_frames, current_idle_frames)
		if ai_actor._desired_direction != Vector2.ZERO:
			direction_active_frames += 1
		previous_position = ai_actor.position
		var current_cell: Vector2i = ai_actor.current_cell()
		visited_cells[current_cell] = true
		if current_cell != transition_cells[-1]:
			if transition_cells.size() >= 2 and current_cell == transition_cells[-2]:
				immediate_reversals += 1
			transition_cells.append(current_cell)
	return {
		"moving_ratio": float(moving_frames) / maxf(1.0, float(total_frames)),
		"visited_cells": visited_cells.size(),
		"immediate_reversals": immediate_reversals,
		"max_idle_frames": max_idle_frames,
		"direction_active_ratio": float(direction_active_frames) / maxf(1.0, float(total_frames)),
	}


func _on_decision_made(
		mode: int,
		_target: Vector2i,
		_score: float,
		elapsed_usec: int,
		actor_id: int
	) -> void:
	if _sample_started_ms <= 0 or _match.get_simulation_time_ms() < _sample_started_ms:
		return
	_decision_times_usec.append(elapsed_usec)
	var mode_times: PackedInt32Array = _decision_times_by_mode.get(
		mode, PackedInt32Array()
	) as PackedInt32Array
	mode_times.append(elapsed_usec)
	_decision_times_by_mode[mode] = mode_times
	var physics_frame: int = Engine.get_physics_frames()
	_frame_decision_usec[physics_frame] = int(_frame_decision_usec.get(physics_frame, 0)) + elapsed_usec
	_frame_decision_count[physics_frame] = int(_frame_decision_count.get(physics_frame, 0)) + 1
	_controller_decision_count[actor_id] = int(_controller_decision_count.get(actor_id, 0)) + 1


func _wait_simulation_ms(duration_ms: int) -> void:
	var deadline: int = _match.get_simulation_time_ms() + duration_ms
	while _match.get_simulation_time_ms() < deadline:
		await physics_frame


func _average_of(values: PackedInt32Array) -> float:
	if values.is_empty():
		return 0.0
	var total: int = 0
	for value: int in values:
		total += value
	return float(total) / values.size()


func _percentile_of(values: PackedInt32Array, ratio: float) -> int:
	if values.is_empty():
		return 0
	var sorted_values: PackedInt32Array = values.duplicate()
	sorted_values.sort()
	var index: int = clampi(ceili(sorted_values.size() * ratio) - 1, 0, sorted_values.size() - 1)
	return sorted_values[index]
