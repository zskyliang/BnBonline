extends SceneTree
## Deterministic offline search/holdout harness for item-aware rule-AI weights.

const TRAINING_SCENARIOS: int = 48
const HOLDOUT_SCENARIOS: int = 48
const MIN_HOLDOUT_PARTICIPATION: float = 0.70
const MIN_UTILITY_IMPROVEMENT: float = 0.08

var _failed: bool = false


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	var packed := load("res://scenes/main.tscn") as PackedScene
	var match_node := packed.instantiate() as MatchController
	root.add_child(match_node)
	await process_frame
	var ai_actor: GameActor = _first_ai(match_node)
	var controller: RuleAI = _controller_for(ai_actor)
	controller.stop_thinking()
	var source_profile: AIBehaviorProfile = AIBehaviorProfile.search_baseline()
	var best_multiplier: float = 1.0
	var best_training_utility: float = -INF
	for multiplier: float in [0.85, 1.0, 1.15]:
		var candidate: AIBehaviorProfile = _scaled_item_profile(source_profile, multiplier)
		var training: Dictionary = _evaluate_profile(
			match_node,
			ai_actor,
			controller,
			candidate,
			202607260,
			TRAINING_SCENARIOS
		)
		var training_utility: float = float(training["aware_utility"])
		if training_utility > best_training_utility + 0.001 \
				or (
					is_equal_approx(training_utility, best_training_utility)
					and absf(multiplier - 1.0) < absf(best_multiplier - 1.0)
				):
			best_training_utility = float(training["aware_utility"])
			best_multiplier = multiplier
		print(
			"AI item training x%.2f: participation %.1f%%, utility %.2f"
			% [
				multiplier,
				float(training["participation"]) * 100.0,
				float(training["aware_utility"]),
			]
		)
	var selected_profile: AIBehaviorProfile = _scaled_item_profile(
		source_profile,
		best_multiplier
	)
	var tuned_profile: AIBehaviorProfile = MatchController.DEFAULT_AI_PROFILE.duplicate(
		true
	) as AIBehaviorProfile
	_check(
		_profiles_match(selected_profile, tuned_profile),
		"serialized runtime profile matches the best training candidate"
	)
	var holdout: Dictionary = _evaluate_profile(
		match_node,
		ai_actor,
		controller,
		tuned_profile,
		202607900,
		HOLDOUT_SCENARIOS,
		true
	)
	var baseline_utility: float = float(holdout["blind_utility"])
	var aware_utility: float = float(holdout["aware_utility"])
	var improvement: float = (aware_utility - baseline_utility) \
		/ maxf(1.0, baseline_utility)
	var participation: float = float(holdout["participation"])
	print(
		"AI item holdout: selected x%.2f, participation %.1f%%, utility improvement %.1f%%"
		% [best_multiplier, participation * 100.0, improvement * 100.0]
	)
	_check(
		participation >= MIN_HOLDOUT_PARTICIPATION,
		"safe positive item participation stays at or above 70%"
	)
	_check(
		improvement >= MIN_UTILITY_IMPROVEMENT,
		"item-aware expected territory utility improves by at least 8%"
	)
	match_node.call("_enter_lobby")
	match_node.queue_free()
	await process_frame
	quit(1 if _failed else 0)


func _evaluate_profile(
		match_node: MatchController,
		ai_actor: GameActor,
		controller: RuleAI,
		profile: AIBehaviorProfile,
		first_seed: int,
		scenario_count: int,
		runtime_item_priority: bool = false
	) -> Dictionary:
	match_node.ai_profile = profile
	var aware_utility: float = 0.0
	var blind_utility: float = 0.0
	var item_selections: int = 0
	var positive_opportunities: int = 0
	for index: int in range(scenario_count):
		var rng := RandomNumberGenerator.new()
		rng.seed = first_seed + index
		var scenario: Dictionary = _prepare_scenario(match_node, ai_actor, rng, index)
		controller.reset_for_scenario(first_seed + index)
		var snapshot: AIBattleSnapshot = match_node.build_ai_snapshot()
		var self_state: AIBattleSnapshot.ActorState = snapshot.actor_by_id(
			ai_actor.get_instance_id()
		)
		var forecast: AIHazardForecast = AIHazardForecast.build(
			snapshot,
			RuleAI.PLANNING_HORIZON_MS
		)
		var paint: Dictionary = controller.call(
			"_find_paint_decision",
			snapshot,
			self_state,
			forecast
		) as Dictionary
		var paint_score: float = maxf(0.0, float(paint.get("score", 0.0)))
		var paint_utility: float = maxf(1.0, float(paint.get("swing", 0)))
		var item: Dictionary = controller.call(
			"_find_item_decision",
			snapshot,
			self_state,
			forecast,
			(
				profile.minimum_item_priority_score
				if runtime_item_priority
				else paint_score
			)
		) as Dictionary
		var selected_item: bool = not item.is_empty()
		var scenario_utility: float = paint_utility
		if bool(scenario["positive_item"]):
			positive_opportunities += 1
			if selected_item:
				item_selections += 1
				scenario_utility += float(scenario["item_territory_gain"])
		elif selected_item:
			# Chasing a low-payback item sacrifices the strong immediate paint play.
			scenario_utility = maxf(0.0, paint_utility - 4.0)
		if not item.is_empty():
			# Exercise the same claim path used by the live controller, then release
			# it so seeded scenarios remain independent.
			match_node.claim_item(
				self_state.instance_id,
				int(item["item_id"]),
				int(item["travel_ms"])
			)
			match_node.release_item_claim(int(item["item_id"]))
		blind_utility += paint_utility
		aware_utility += scenario_utility
	return {
		"aware_utility": aware_utility,
		"blind_utility": blind_utility,
		"participation": float(item_selections) / maxf(1.0, float(positive_opportunities)),
	}


func _prepare_scenario(
		match_node: MatchController,
		ai_actor: GameActor,
		rng: RandomNumberGenerator,
		index: int
	) -> Dictionary:
	var board: GameBoard = match_node.board
	for value: Variant in board.bombs.values():
		var bubble: GameBubble = value as GameBubble
		if is_instance_valid(bubble):
			board.unregister_bubble(bubble)
			bubble.queue_free()
	board.clear_items()
	var positive_item: bool = index % 4 != 0
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			board.paint_owners[y][x] = PaintPalette.TEAM_AI
			board.locked_cells[y][x] = 0
	var actor_cell := Vector2i(rng.randi_range(5, 9), rng.randi_range(4, 8))
	ai_actor.respawn(actor_cell)
	ai_actor.stats.invincible_until_ms = 0
	var stage_number: int = [1, 2, 4][index % 3]
	var allocation: Array[int] = [0, 0, 0]
	for _point: int in range(stage_number - 1):
		allocation[rng.randi_range(0, allocation.size() - 1)] += 1
	ai_actor.stats.apply_skill_points(
		allocation[0],
		allocation[1],
		allocation[2]
	)
	if positive_item:
		# A small distant patch keeps an immediate paint action available while
		# making the nearby upgrade the higher expected multi-cycle territory gain.
		for patch_index: int in range(2 + index % 2):
			var patch_cell := Vector2i(
				1 + patch_index % 2,
				1 + int(patch_index / 2)
			)
			board.paint_owners[patch_cell.y][patch_cell.x] = (
				PaintPalette.TEAM_PLAYER
				if patch_index % 2 == 0
				else PaintPalette.TEAM_NEUTRAL
			)
	else:
		# Control scenarios contain a dense, immediately paintable opponent patch;
		# correct tuning should reject the lower-payback detour.
		for patch_y: int in range(3, 8):
			for patch_x: int in range(3, 8):
				var patch_cell := Vector2i(patch_x, patch_y)
				if patch_cell != actor_cell:
					board.paint_owners[patch_y][patch_x] = PaintPalette.TEAM_PLAYER
	var offset: Vector2i = AITemporalPlanner.CARDINAL_DIRECTIONS[
		rng.randi_range(0, AITemporalPlanner.CARDINAL_DIRECTIONS.size() - 1)
	] * (rng.randi_range(1, 2) if positive_item else rng.randi_range(4, 5))
	var item_cell: Vector2i = actor_cell + offset
	if not GameConstants.is_inside(item_cell):
		item_cell = actor_cell + Vector2i.RIGHT
	board.spawn_item(ArenaItemType.ALL[index % ArenaItemType.ALL.size()], item_cell, 10000)
	return {
		"positive_item": positive_item,
		"item_territory_gain": 4.0 + float(index % 3),
		"stage_number": stage_number,
	}


func _scaled_item_profile(
		source: AIBehaviorProfile,
		multiplier: float
	) -> AIBehaviorProfile:
	var result: AIBehaviorProfile = source.duplicate(true) as AIBehaviorProfile
	result.item_base_value *= multiplier
	result.speed_item_value *= multiplier
	result.bubble_item_value *= multiplier
	result.power_item_value *= multiplier
	result.power_swing_weight *= multiplier
	result.bubble_capacity_pressure_weight *= multiplier
	result.speed_payback_weight *= multiplier
	return result


func _profiles_match(left: AIBehaviorProfile, right: AIBehaviorProfile) -> bool:
	return is_equal_approx(left.item_base_value, right.item_base_value) \
		and is_equal_approx(left.speed_item_value, right.speed_item_value) \
		and is_equal_approx(left.bubble_item_value, right.bubble_item_value) \
		and is_equal_approx(left.power_item_value, right.power_item_value) \
		and is_equal_approx(left.power_swing_weight, right.power_swing_weight) \
		and is_equal_approx(
			left.bubble_capacity_pressure_weight,
			right.bubble_capacity_pressure_weight
		) \
		and is_equal_approx(left.speed_payback_weight, right.speed_payback_weight)


func _first_ai(match_node: MatchController) -> GameActor:
	for actor: GameActor in match_node.get_actors():
		if not actor.is_player:
			return actor
	return null


func _controller_for(actor: GameActor) -> RuleAI:
	for child: Node in actor.get_children():
		if child is RuleAI:
			return child as RuleAI
	return null


func _check(condition: bool, message: String) -> void:
	if condition:
		return
	_failed = true
	push_error("AI ITEM TRAINING FAILED: %s" % message)
