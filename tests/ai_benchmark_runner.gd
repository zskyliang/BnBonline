extends SceneTree
## Seeded benchmark for paint-first AI decisions and safety constraints.

const DECISION_SAMPLES: int = 120
const AVERAGE_GATE_USEC: float = 5000.0
const P95_GATE_USEC: int = 10000

var _failed: bool = false


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	var packed := load("res://scenes/main.tscn") as PackedScene
	var match_node := packed.instantiate() as MatchController
	root.add_child(match_node)
	await process_frame
	_stop_all_ai(match_node)
	var ai_actor: GameActor = _first_ai(match_node)
	var controller := _controller_for(ai_actor)
	controller.set_decision_seed(20260725)
	ai_actor.position = GameConstants.grid_to_world(Vector2i(7, 6))
	controller.reset_for_scenario(20260725)
	controller.reconsider_now()
	_check(
		not match_node.board.bombs.is_empty(),
		"paint-first AI immediately drops a safe bubble on neutral territory"
	)
	var first_bubble: GameBubble = match_node.board.bombs.values()[0] as GameBubble
	_check(first_bubble.owner_team == PaintPalette.TEAM_AI, "AI bubble belongs to shared AI paint team")
	ai_actor.stats.bubble_capacity = 4
	ai_actor.stats.move_speed = 225.0
	var barrage_peak_active: int = ai_actor.stats.active_bubbles
	for frame: int in range(150):
		await physics_frame
		if frame % 9 == 0:
			controller.reconsider_now()
		barrage_peak_active = maxi(barrage_peak_active, ai_actor.stats.active_bubbles)
	_check(
		barrage_peak_active >= ai_actor.stats.bubble_capacity - 1,
		"safe barrage keeps at most one bubble slot unused; peak %d/%d"
		% [barrage_peak_active, ai_actor.stats.bubble_capacity]
	)
	var barrage_bubbles: Array = match_node.board.bombs.values().duplicate()
	for bubble_value: Variant in barrage_bubbles:
		var barrage_bubble := bubble_value as GameBubble
		if is_instance_valid(barrage_bubble):
			barrage_bubble.explode_now()
	await process_frame
	var counts: Dictionary = match_node.board.get_territory_counts()
	_check(int(counts["ai"]) > 0, "AI explosion converts neutral floor to AI territory")
	_clear_effects(match_node)
	ai_actor.respawn(Vector2i(7, 6))
	ai_actor.stats.invincible_until_ms = 0
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			match_node.board.paint_owners[y][x] = PaintPalette.TEAM_AI
	var nearby_item_cell := Vector2i(8, 6)
	var nearby_item_id: int = match_node.board.spawn_item(
		ArenaItemType.Value.POWER,
		nearby_item_cell,
		10000
	)
	var item_decision: Dictionary = _item_decision(match_node, ai_actor, controller)
	_check(
		not item_decision.is_empty() \
			and int(item_decision.get("item_id", 0)) == nearby_item_id,
		"item-aware AI finds a safe, positive-power pickup"
	)
	match_node._remaining_seconds = 2.0
	_check(
		_item_decision(match_node, ai_actor, controller).is_empty(),
		"AI rejects an item that cannot pay back before the round ends"
	)
	match_node._remaining_seconds = GameConstants.ROUND_SECONDS
	var dangerous_item_effect := ExplosionEffect.new()
	match_node._effect_root.add_child(dangerous_item_effect)
	dangerous_item_effect.setup(
		[nearby_item_cell],
		nearby_item_cell,
		match_node.get_player()
	)
	match_node.call("_register_explosion_effect", dangerous_item_effect)
	_check(
		_item_decision(match_node, ai_actor, controller).is_empty(),
		"AI abandons a positive item whose route is actively dangerous"
	)
	match_node.call("_unregister_explosion_effect", dangerous_item_effect)
	dangerous_item_effect.queue_free()
	_reset_neutral_paint(match_node.board)
	var high_value_paint: Dictionary = _paint_decision(
		match_node,
		ai_actor,
		controller
	)
	_check(
		float(high_value_paint.get("score", 0.0)) > 250.0,
		"priority scenario also contains an immediately valuable paint action"
	)
	controller.reset_for_scenario(20260726)
	controller.reconsider_now()
	_check(
		controller.current_mode == RuleAI.Mode.COLLECTING \
			and controller.decision_target == nearby_item_cell,
		"safe positive-payback item preempts even a high-value paint action; got %s at %s score %.1f dead=%s trapped=%s"
			% [
				controller.mode_name(),
				controller.decision_target,
				controller.last_decision_score,
				ai_actor.stats.is_dead,
				ai_actor.stats.is_trapped,
			]
	)
	match_node.board.clear_items()
	match_node.release_item_claims_for_actor(ai_actor.get_instance_id())
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			match_node.board.paint_owners[y][x] = PaintPalette.TEAM_AI
			match_node.board.locked_cells[y][x] = 0
	var player_patch: Array[Vector2i] = [
		Vector2i(5, 6), Vector2i(6, 6), Vector2i(7, 6),
		Vector2i(8, 6), Vector2i(9, 6),
	]
	for cell: Vector2i in player_patch:
		match_node.board.paint_owners[cell.y][cell.x] = PaintPalette.TEAM_PLAYER
	var decision: Dictionary = _paint_decision(match_node, ai_actor, controller)
	_check(not decision.is_empty(), "AI finds a repaint decision when player territory exists")
	if not decision.is_empty():
		var target: Vector2i = decision["cell"] as Vector2i
		var blast: Array[Vector2i] = GameRules.blast_cells(
			target,
			ai_actor.stats.power,
			match_node.board.cells
		)
		_check(
			match_node.board.territory_swing(blast, PaintPalette.TEAM_AI) >= 4,
			"AI targets a high net-swing repaint area"
		)
	match_node.board.lock_neighborhood(Vector2i(7, 6), PaintPalette.TEAM_PLAYER)
	var locked_snapshot: AIBattleSnapshot = match_node.build_ai_snapshot()
	var locked_blast: Array[Vector2i] = GameRules.blast_cells(
		Vector2i(7, 6),
		ai_actor.stats.power,
		locked_snapshot.cells
	)
	_check(
		locked_snapshot.territory_swing(locked_blast, PaintPalette.TEAM_AI) \
			< locked_blast.size() * 2,
		"AI paint utility excludes permanent locked cells"
	)
	var locked_decision: Dictionary = _paint_decision(match_node, ai_actor, controller)
	if not locked_decision.is_empty():
		_check(
			not locked_snapshot.is_paint_locked(locked_decision["cell"] as Vector2i),
			"AI never selects a locked tile as its placement target"
		)
	_reset_neutral_paint(match_node.board)
	var timings: PackedInt64Array = PackedInt64Array()
	for sample: int in range(DECISION_SAMPLES):
		controller.set_decision_seed(9000 + sample)
		var started_usec: int = Time.get_ticks_usec()
		var sampled: Dictionary = _paint_decision(match_node, ai_actor, controller)
		timings.append(Time.get_ticks_usec() - started_usec)
		if sampled.is_empty():
			_failed = true
			push_error("AI BENCHMARK: neutral board produced no paint decision")
			break
	timings.sort()
	var total_usec: int = 0
	for elapsed_usec: int in timings:
		total_usec += elapsed_usec
	var average_usec: float = float(total_usec) / maxf(1.0, float(timings.size()))
	var p95_index: int = clampi(ceili(float(timings.size()) * 0.95) - 1, 0, timings.size() - 1)
	var p95_usec: int = int(timings[p95_index])
	print("BnBonline AI paint benchmark: avg %.2f ms, p95 %.2f ms" % [
		average_usec / 1000.0,
		float(p95_usec) / 1000.0,
	])
	_check(average_usec < AVERAGE_GATE_USEC, "average paint decision stays below 5 ms")
	_check(p95_usec < P95_GATE_USEC, "P95 paint decision stays below 10 ms")
	match_node.board.clear_items()
	for index: int in range(12):
		var item_cell := Vector2i(index + 1, 2 + index % 3)
		match_node.board.spawn_item(
			ArenaItemType.ALL[index % ArenaItemType.ALL.size()],
			item_cell,
			10000 + index
		)
	var item_timings: PackedInt64Array = PackedInt64Array()
	for sample: int in range(DECISION_SAMPLES):
		controller.set_decision_seed(12000 + sample)
		var item_started_usec: int = Time.get_ticks_usec()
		var sampled_item: Dictionary = _item_decision(match_node, ai_actor, controller)
		item_timings.append(Time.get_ticks_usec() - item_started_usec)
		if sampled_item.is_empty():
			_failed = true
			push_error("AI BENCHMARK: populated item board produced no safe item decision")
			break
	item_timings.sort()
	var item_total_usec: int = 0
	for elapsed_usec: int in item_timings:
		item_total_usec += elapsed_usec
	var item_average_usec: float = float(item_total_usec) / maxf(1.0, float(item_timings.size()))
	var item_p95_index: int = clampi(
		ceili(float(item_timings.size()) * 0.95) - 1,
		0,
		item_timings.size() - 1
	)
	var item_p95_usec: int = int(item_timings[item_p95_index])
	print("BnBonline AI item benchmark: avg %.2f ms, p95 %.2f ms" % [
		item_average_usec / 1000.0,
		float(item_p95_usec) / 1000.0,
	])
	_check(item_average_usec < AVERAGE_GATE_USEC, "average item decision stays below 5 ms")
	_check(item_p95_usec < P95_GATE_USEC, "P95 item decision stays below 10 ms")
	match_node.call("_enter_lobby")
	match_node.queue_free()
	await process_frame
	quit(1 if _failed else 0)


func _paint_decision(
		match_node: MatchController,
		ai_actor: GameActor,
		controller: RuleAI
	) -> Dictionary:
	var snapshot: AIBattleSnapshot = match_node.build_ai_snapshot()
	var self_state: AIBattleSnapshot.ActorState = snapshot.actor_by_id(ai_actor.get_instance_id())
	var forecast: AIHazardForecast = AIHazardForecast.build(snapshot, RuleAI.PLANNING_HORIZON_MS)
	return controller.call("_find_paint_decision", snapshot, self_state, forecast) as Dictionary


func _item_decision(
		match_node: MatchController,
		ai_actor: GameActor,
		controller: RuleAI
	) -> Dictionary:
	var snapshot: AIBattleSnapshot = match_node.build_ai_snapshot()
	var self_state: AIBattleSnapshot.ActorState = snapshot.actor_by_id(ai_actor.get_instance_id())
	var forecast: AIHazardForecast = AIHazardForecast.build(snapshot, RuleAI.PLANNING_HORIZON_MS)
	return controller.call("_find_item_decision", snapshot, self_state, forecast) as Dictionary


func _reset_neutral_paint(board: GameBoard) -> void:
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			board.paint_owners[y][x] = PaintPalette.TEAM_NEUTRAL
			board.locked_cells[y][x] = 0


func _clear_effects(match_node: MatchController) -> void:
	var effects: Array[ExplosionEffect] = match_node._active_explosions.duplicate()
	for effect: ExplosionEffect in effects:
		match_node.call("_unregister_explosion_effect", effect)
		effect.queue_free()


func _stop_all_ai(match_node: MatchController) -> void:
	for actor: GameActor in match_node.get_actors():
		var controller: RuleAI = _controller_for(actor)
		if is_instance_valid(controller):
			controller.stop_thinking()


func _first_ai(match_node: MatchController) -> GameActor:
	for actor: GameActor in match_node.get_actors():
		if not actor.is_player:
			return actor
	return null


func _controller_for(actor: GameActor) -> RuleAI:
	if not is_instance_valid(actor):
		return null
	for child: Node in actor.get_children():
		if child is RuleAI:
			return child as RuleAI
	return null


func _check(condition: bool, message: String) -> void:
	if condition:
		return
	_failed = true
	push_error("AI BENCHMARK FAILED: %s" % message)
