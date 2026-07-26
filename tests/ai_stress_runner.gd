extends SceneTree
## Four-AI paint-campaign scheduler, fairness, and latency stress check.

const SIMULATION_FRAMES: int = 360
const MIN_DECISIONS_PER_AI: int = 24
const P95_GATE_USEC: int = 10000

var _failed: bool = false
var _decision_counts: Dictionary = {}
var _frame_decisions: Dictionary = {}
var _decision_times: PackedInt64Array = PackedInt64Array()
var _decision_times_by_mode: Dictionary = {}
var _evasion_times_by_kind: Dictionary = {}
var _peak_active_bubbles: Dictionary = {}


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	var packed := load("res://scenes/main.tscn") as PackedScene
	var match_node := packed.instantiate() as MatchController
	root.add_child(match_node)
	await process_frame
	match_node.call("_enter_lobby")
	match_node.settings.character_id = "cat"
	match_node.settings.player_color_id = "red"
	match_node._rng.seed = 20260726
	match_node.call("_begin_new_run")
	match_node.run_progress.advance_with_skill(RunProgress.SKILL_SPEED, match_node._rng)
	match_node.run_progress.advance_with_skill(RunProgress.SKILL_BUBBLE, match_node._rng)
	match_node.run_progress.advance_with_skill(RunProgress.SKILL_POWER, match_node._rng)
	match_node.start_match()
	await process_frame
	var ai_actors: Array[GameActor] = []
	for actor_index: int in range(match_node.get_actors().size()):
		var actor: GameActor = match_node.get_actors()[actor_index]
		if actor.is_player:
			continue
		ai_actors.append(actor)
		_check(actor.color_id == match_node.run_progress.ai_color_id, "all AI share one team color")
		var assigned_points: int = roundi(
			(actor.stats.move_speed - GameConstants.INITIAL_SPEED) \
			/ GameConstants.SPEED_PER_SKILL_POINT
		) + actor.stats.bubble_capacity - GameConstants.INITIAL_BUBBLES \
			+ actor.stats.power - GameConstants.INITIAL_POWER
		_check(assigned_points == 3, "each stage-four AI owns three skill points")
		var controller: RuleAI = _controller_for(actor)
		controller.set_decision_seed(20260726 + actor_index * 101)
		_decision_counts[actor.get_instance_id()] = 0
		_peak_active_bubbles[actor.get_instance_id()] = 0
		controller.decision_made.connect(_on_decision.bind(actor.get_instance_id()))
	_check(ai_actors.size() == 4, "stage four runs four AI")
	for index: int in range(6):
		match_node.board.spawn_item(
			ArenaItemType.ALL[index % ArenaItemType.ALL.size()],
			Vector2i(index + 1, 3 + index % 4),
			index
		)
	for _frame: int in range(SIMULATION_FRAMES):
		await physics_frame
		for actor: GameActor in ai_actors:
			_peak_active_bubbles[actor.get_instance_id()] = maxi(
				int(_peak_active_bubbles.get(actor.get_instance_id(), 0)),
				actor.stats.active_bubbles
			)
	var minimum_count: int = 99999
	var maximum_count: int = 0
	for count_value: Variant in _decision_counts.values():
		var count: int = int(count_value)
		minimum_count = mini(minimum_count, count)
		maximum_count = maxi(maximum_count, count)
		_check(count >= MIN_DECISIONS_PER_AI, "every AI receives regular scheduler decisions")
	_check(maximum_count - minimum_count <= 2, "round-robin scheduler remains fair")
	var maximum_same_frame: int = 0
	for count_value: Variant in _frame_decisions.values():
		maximum_same_frame = maxi(maximum_same_frame, int(count_value))
	_check(maximum_same_frame <= 1, "normal scheduler never dispatches two AI on one physics frame")
	_decision_times.sort()
	var p95_index: int = clampi(
		ceili(float(_decision_times.size()) * 0.95) - 1,
		0,
		_decision_times.size() - 1
	)
	var p95_usec: int = int(_decision_times[p95_index])
	_check(p95_usec < P95_GATE_USEC, "four-AI P95 decision latency stays below 10 ms")
	var counts: Dictionary = match_node.board.get_territory_counts()
	_check(int(counts["ai"]) > 0, "four-AI match paints shared AI territory")
	var collected_item_count: int = 0
	var total_peak_unused_slots: int = 0
	for actor: GameActor in ai_actors:
		collected_item_count += actor.stats.stage_speed_items \
			+ actor.stats.stage_bubble_items \
			+ actor.stats.stage_power_items
		var peak_active: int = int(_peak_active_bubbles.get(actor.get_instance_id(), 0))
		var peak_unused: int = maxi(0, actor.stats.bubble_capacity - peak_active)
		total_peak_unused_slots += peak_unused
		_check(
			peak_unused <= 1,
			"each AI drives safe live bubble capacity to within one free slot; %s peaked %d/%d"
			% [actor.actor_name, peak_active, actor.stats.bubble_capacity]
		)
	_check(collected_item_count > 0, "four-AI live battle successfully collects useful items")
	_check(
		float(total_peak_unused_slots) / maxf(1.0, float(ai_actors.size())) <= 0.5,
		"four-AI average peak free slots stays at or below 0.5"
	)
	var distinct_item_targets: Dictionary = {}
	var collecting_count: int = 0
	for actor: GameActor in ai_actors:
		var controller: RuleAI = _controller_for(actor)
		if controller.current_mode == RuleAI.Mode.COLLECTING:
			collecting_count += 1
			distinct_item_targets[controller.decision_target] = true
	_check(
		distinct_item_targets.size() == collecting_count,
		"four-AI item coordination prevents duplicate live pickup targets"
	)
	print(
		"BnBonline AI paint stress: decisions %d-%d, same-frame max %d, p95 %.2f ms, AI cells %d, avg peak free slots %.2f"
		% [
			minimum_count,
			maximum_count,
			maximum_same_frame,
			float(p95_usec) / 1000.0,
			int(counts["ai"]),
			float(total_peak_unused_slots) / maxf(1.0, float(ai_actors.size())),
		]
	)
	for mode: Variant in _decision_times_by_mode:
		var mode_times: PackedInt64Array = _decision_times_by_mode[mode]
		mode_times.sort()
		var mode_p95_index: int = clampi(
			ceili(float(mode_times.size()) * 0.95) - 1,
			0,
			mode_times.size() - 1
		)
		print(
			"  %s: n=%d p95=%.2f ms max=%.2f ms"
			% [
				RuleAI.Mode.keys()[int(mode)],
				mode_times.size(),
				float(mode_times[mode_p95_index]) / 1000.0,
				float(mode_times[-1]) / 1000.0,
			]
		)
	for kind: Variant in _evasion_times_by_kind:
		var kind_times: PackedInt64Array = _evasion_times_by_kind[kind]
		kind_times.sort()
		var kind_p95_index: int = clampi(
			ceili(float(kind_times.size()) * 0.95) - 1,
			0,
			kind_times.size() - 1
		)
		print(
			"  evasion/%s: n=%d p95=%.2f ms max=%.2f ms"
			% [
				str(kind),
				kind_times.size(),
				float(kind_times[kind_p95_index]) / 1000.0,
				float(kind_times[-1]) / 1000.0,
			]
		)
	match_node.call("_enter_lobby")
	match_node.queue_free()
	await process_frame
	quit(1 if _failed else 0)


func _on_decision(
		_mode: int,
		_target_cell: Vector2i,
		_score: float,
		elapsed_usec: int,
		actor_id: int
	) -> void:
	_decision_counts[actor_id] = int(_decision_counts.get(actor_id, 0)) + 1
	var frame: int = Engine.get_physics_frames()
	_frame_decisions[frame] = int(_frame_decisions.get(frame, 0)) + 1
	_decision_times.append(elapsed_usec)
	if not _decision_times_by_mode.has(_mode):
		_decision_times_by_mode[_mode] = PackedInt64Array()
	var mode_times: PackedInt64Array = _decision_times_by_mode[_mode]
	mode_times.append(elapsed_usec)
	_decision_times_by_mode[_mode] = mode_times
	if _mode == RuleAI.Mode.EVADING:
		var kind := "hazard" if is_equal_approx(_score, 1000.0) else "drop"
		if not _evasion_times_by_kind.has(kind):
			_evasion_times_by_kind[kind] = PackedInt64Array()
		var kind_times: PackedInt64Array = _evasion_times_by_kind[kind]
		kind_times.append(elapsed_usec)
		_evasion_times_by_kind[kind] = kind_times


func _controller_for(actor: GameActor) -> RuleAI:
	for child: Node in actor.get_children():
		if child is RuleAI:
			return child as RuleAI
	return null


func _check(condition: bool, message: String) -> void:
	if condition:
		return
	_failed = true
	push_error("AI STRESS FAILED: %s" % message)
