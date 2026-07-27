extends SceneTree
## Exercises the real paint campaign, UI, combat attribution, and progression flow.

var _failed: bool = false


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	var packed := load("res://scenes/main.tscn") as PackedScene
	var match_node := packed.instantiate() as MatchController
	root.add_child(match_node)
	await process_frame
	match_node.call("_enter_lobby")
	match_node.call("_on_language_changed", "en")
	await process_frame
	var lobby_settings_button := match_node.hud.find_child(
		"LobbySettingsButton",
		true,
		false
	) as Button
	_assert(
		is_instance_valid(lobby_settings_button) \
			and tr(lobby_settings_button.text) == "Settings",
		"lobby exposes the default-English settings entry"
	)
	match_node.call("_open_settings")
	var language_selector := match_node.hud.find_child(
		"LanguageSelector",
		true,
		false
	) as OptionButton
	_assert(
		match_node.hud.is_settings_visible() and not paused \
			and is_instance_valid(language_selector) \
			and language_selector.item_count == 2,
		"lobby settings offers English and Chinese without pausing"
	)
	match_node.call("_on_language_changed", "zh")
	await process_frame
	_assert(
		tr(lobby_settings_button.text) == "设置" \
			and match_node.settings.language_code == "zh",
		"language selection immediately switches and persists Chinese"
	)
	match_node.call("_on_language_changed", "en")
	await process_frame
	match_node.call("_close_settings")
	match_node.settings.character_id = "cat"
	match_node.settings.player_color_id = "orange"
	match_node.call("_begin_new_run")
	match_node.start_match()
	await process_frame
	_assert(match_node.run_progress.stage_number == 1, "campaign starts at stage one")
	_assert(match_node.get_actors().size() == 2, "stage one spawns player and one AI")
	_assert(
		match_node._remaining_seconds <= 120.0 and match_node._remaining_seconds > 119.0,
		"live stage starts at two minutes"
	)
	_assert(match_node.hud.timer_label.text == "02:00", "live timer begins at 02:00")
	_assert(match_node.board.get_open_cells().size() == 195, "live arena exposes all 195 cells")
	_assert(not InputMap.has_action("self_rescue"), "self-rescue input has been removed")
	var floor := match_node._arena_view.board_view.find_child(
		"PaintFloorTiles",
		true,
		false
	) as MultiMeshInstance3D
	_assert(
		is_instance_valid(floor) and floor.multimesh.instance_count == 195,
		"live renderer batches all floor tiles"
	)
	var player: GameActor = match_node.get_player()
	var settings_button := match_node.hud.find_child(
		"SettingsButton",
		true,
		false
	) as Button
	var scores_panel := match_node.hud.find_child(
		"LiveScoresPanel",
		true,
		false
	) as PanelContainer
	var item_panel := match_node.hud.find_child(
		"LiveItemPanel",
		true,
		false
	) as PanelContainer
	var scores_style := scores_panel.get_theme_stylebox("panel") as StyleBoxFlat
	var item_style := item_panel.get_theme_stylebox("panel") as StyleBoxFlat
	_assert(is_instance_valid(settings_button), "live HUD exposes one dedicated settings button")
	_assert(
		scores_style.bg_color.a < 0.8 and item_style.bg_color.a < 0.8,
		"live score and item panels stay translucent over the battle"
	)
	match_node.call("_open_settings")
	_assert(
		paused and match_node.hud.is_settings_visible(),
		"settings button modal pauses and blocks the live match"
	)
	_assert(
		match_node.hud.find_children("*", "HSlider", true, false).is_empty(),
		"fixed-camera settings contain no azimuth or elevation sliders"
	)
	match_node.call("_close_settings")
	_assert(
		not paused and not match_node.hud.is_settings_visible(),
		"closing settings returns to the same live match"
	)
	_stop_ai(match_node)
	match_node._simulation_time_ms = 10000.0
	match_node.call("_spawn_due_items")
	_assert(
		match_node.board.get_item_states().size() == 3,
		"three random items spawn together at ten seconds"
	)
	var timed_item: ArenaItemState = match_node.board.get_item_states()[0]
	_assert(
		match_node.build_ai_snapshot().item_by_id(timed_item.item_id) != null,
		"AI snapshot exposes live item state"
	)
	match_node._simulation_time_ms = 119000.0
	match_node.call("_spawn_due_items")
	_assert(
		match_node.board.get_item_states().size() == 33,
		"a full two-minute stage schedules eleven groups of three item drops"
	)
	match_node.board.clear_items()
	match_node._item_rng.seed = 20260726
	for _index: int in range(6):
		match_node.call("_spawn_random_item")
	var first_item_sequence: Array[String] = _item_signature(match_node.board)
	match_node.board.clear_items()
	match_node._item_rng.seed = 20260726
	for _index: int in range(6):
		match_node.call("_spawn_random_item")
	_assert(
		_item_signature(match_node.board) == first_item_sequence,
		"dedicated item RNG reproduces the same seeded type and cell sequence"
	)
	match_node.board.clear_items()
	var player_cell: Vector2i = player.current_cell()
	var spawn_candidates: Array[Vector2i] = match_node.call(
		"_item_spawn_candidates"
	) as Array[Vector2i]
	_assert(
		player_cell not in spawn_candidates,
		"item generation excludes cells occupied by an active actor"
	)
	var blocked_cell := Vector2i(4, 4)
	var test_bubble := GameBubble.new()
	test_bubble.cell = blocked_cell
	match_node._entity_root.add_child(test_bubble)
	match_node.board.register_bubble(test_bubble)
	spawn_candidates = match_node.call("_item_spawn_candidates") as Array[Vector2i]
	_assert(blocked_cell not in spawn_candidates, "item generation excludes bubble cells")
	match_node.board.unregister_bubble(test_bubble)
	test_bubble.queue_free()
	var blast_cell := Vector2i(5, 4)
	var test_effect := ExplosionEffect.new()
	match_node._effect_root.add_child(test_effect)
	test_effect.setup([blast_cell], blast_cell, null)
	match_node.call("_register_explosion_effect", test_effect)
	spawn_candidates = match_node.call("_item_spawn_candidates") as Array[Vector2i]
	_assert(
		blast_cell not in spawn_candidates,
		"item generation excludes active explosion cells"
	)
	match_node.call("_unregister_explosion_effect", test_effect)
	test_effect.queue_free()
	var pickup_cell := Vector2i(6, 6)
	player.position = GameConstants.grid_to_world(pickup_cell)
	match_node.board.spawn_item(ArenaItemType.Value.SPEED, pickup_cell, 10000)
	match_node.call("_resolve_item_pickups")
	_assert(
		player.stats.stage_speed_items == 1 and player.stats.move_speed == 200.0,
		"player pickup applies a stacking stage-only speed bonus"
	)
	player.respawn(Vector2i(6, 6))
	_assert(player.stats.move_speed == 200.0, "respawn preserves current-stage item bonuses")
	player.position = GameConstants.grid_to_world(Vector2i(7, 6))
	var player_view := match_node._arena_view.actor_view_for(player)
	var action_before_placement: StringName = player_view.get_current_action()
	_assert(match_node.request_bomb(player), "player can place a colored bubble")
	_assert(
		is_instance_valid(player_view) \
			and player_view.get_current_action() == action_before_placement \
			and not player_view.has_action(&"PlaceBubble"),
		"successful placement does not interrupt the directional movement state"
	)
	var bubble := match_node.board.bombs.get(Vector2i(7, 6)) as GameBubble
	_assert(is_instance_valid(bubble) and bubble.color_id == "orange", "bubble uses player color")
	bubble.explode_now()
	await process_frame
	var first_counts: Dictionary = match_node.board.get_territory_counts()
	_assert(int(first_counts["player"]) == 5, "cat's one starting power point paints five open cells")
	var ai_actor: GameActor = _first_ai(match_node)
	ai_actor.position = GameConstants.grid_to_world(Vector2i(10, 9))
	ai_actor.trap(player)
	ai_actor.finish_by_touch(player)
	_assert(match_node.board.is_locked(Vector2i(10, 9)), "opponent defeat locks center tile")
	_assert(
		match_node.board.paint_owner(Vector2i(10, 9)) == PaintPalette.TEAM_PLAYER,
		"defeat neighborhood belongs to finishing player"
	)
	var lock_counts: Dictionary = match_node.board.get_territory_counts()
	_assert(int(lock_counts["player_locked"]) == 9, "center defeat locks full nine-cell neighborhood")
	var timeout_cell := Vector2i(2, 10)
	ai_actor.respawn(timeout_cell)
	ai_actor.stats.invincible_until_ms = 0
	ai_actor.trap(player)
	ai_actor.call("_on_trap_timeout")
	_assert(
		not ai_actor.stats.is_dead \
			and not ai_actor.stats.is_trapped \
			and not match_node.board.is_locked(timeout_cell),
		"trap timeout releases the actor and never creates permanent territory"
	)
	var self_cell := Vector2i(2, 2)
	player.position = GameConstants.grid_to_world(self_cell)
	player.trap(player)
	player.call("_on_trap_timeout")
	_assert(
		not player.stats.is_dead \
			and not player.stats.is_trapped \
			and not match_node.board.is_locked(self_cell),
		"self trap timeout also releases without a defeat or permanent territory"
	)
	match_node.run_progress.advance_with_skill(RunProgress.SKILL_SPEED, match_node._rng)
	match_node.start_match()
	await process_frame
	_stop_ai(match_node)
	_assert(match_node.run_progress.stage_number == 2, "campaign advances to stage two")
	_assert(match_node.get_actors().size() == 3, "stage two spawns two AI")
	var ai_actors: Array[GameActor] = _all_ai(match_node)
	var claim_item_id: int = match_node.board.spawn_item(
		ArenaItemType.Value.BUBBLE,
		Vector2i(4, 4),
		0
	)
	_assert(
		match_node.claim_item(ai_actors[0].get_instance_id(), claim_item_id, 2000),
		"first AI can claim an available item"
	)
	_assert(
		not match_node.can_claim_item(ai_actors[1].get_instance_id(), claim_item_id, 1800),
		"teammate respects a comparable existing item ETA"
	)
	_assert(
		match_node.can_claim_item(ai_actors[1].get_instance_id(), claim_item_id, 1000),
		"meaningfully faster teammate may take over an item claim"
	)
	_assert(
		match_node.claim_item(ai_actors[1].get_instance_id(), claim_item_id, 1000),
		"faster teammate atomically takes over the item claim"
	)
	match_node.release_item_claim(claim_item_id, ai_actors[0].get_instance_id())
	_assert(
		not match_node.can_claim_item(
			ai_actors[0].get_instance_id(),
			claim_item_id,
			2000
		),
		"former claimant cannot release the faster teammate's replacement claim"
	)
	match_node.board.clear_items()
	var friendly_cell := Vector2i(8, 7)
	ai_actors[1].position = GameConstants.grid_to_world(friendly_cell)
	var friendly_effect := ExplosionEffect.new()
	match_node._effect_root.add_child(friendly_effect)
	friendly_effect.setup([friendly_cell], friendly_cell, ai_actors[0])
	match_node.call("_register_explosion_effect", friendly_effect)
	var source_snapshot: AIBattleSnapshot = match_node.build_ai_snapshot()
	var source_state: AIBattleSnapshot.ExplosionState = source_snapshot.explosions[0]
	_assert(
		source_state.attacker_team == PaintPalette.TEAM_AI \
			and source_state.attacker_id == ai_actors[0].get_instance_id(),
		"AI snapshot retains the stable explosion source"
	)
	match_node.call("_resolve_explosion_hits")
	match_node.call("_resolve_explosion_hits")
	_assert(not ai_actors[1].stats.is_trapped, "AI teammate ignores friendly explosion")
	player = match_node.get_player()
	player_view = match_node._arena_view.actor_view_for(player)
	player.position = GameConstants.grid_to_world(friendly_cell)
	match_node.call("_resolve_explosion_hits")
	match_node.call("_resolve_explosion_hits")
	_assert(player.stats.is_trapped, "player is trapped by AI explosion")
	match_node.call("_unregister_explosion_effect", friendly_effect)
	friendly_effect.queue_free()
	var victory_cell := Vector2i(1, 1)
	match_node.board.paint_cells([victory_cell], PaintPalette.TEAM_PLAYER)
	match_node.call("_end_round")
	_assert(match_node._last_round_won, "strictly higher player territory wins")
	_assert(
		player_view.get_current_action() == &"Idle" \
			and not player_view.has_action(&"Victory"),
		"winning settlement keeps the last-facing Idle without a victory action"
	)
	_assert(
		player.stats.stage_speed_items == 0 and player.stats.move_speed == 200.0,
		"round settlement removes temporary item bonuses"
	)
	_assert(paused, "result pauses live simulation")
	match_node.call("_advance_stage", RunProgress.SKILL_BUBBLE)
	await process_frame
	_assert(match_node.run_progress.stage_number == 3, "skill confirmation enters next stage")
	_assert(match_node.get_player().stats.move_speed == 200.0, "earlier speed point carries forward")
	_assert(
		match_node.get_player().stats.stage_speed_items == 0,
		"next stage starts without previous item bonuses"
	)
	_assert(match_node.get_player().stats.bubble_capacity == 3, "new bubble point applies next stage")
	var points_before_loss: int = match_node.run_progress.total_skill_points()
	var retry_allocations: Array[Dictionary] = match_node.run_progress.ai_allocations.duplicate(true)
	match_node.call("_end_round")
	_assert(not match_node._last_round_won, "a neutral zero-to-zero tie fails the stage")
	match_node.start_match()
	await process_frame
	_stop_ai(match_node)
	_assert(
		match_node.run_progress.total_skill_points() == points_before_loss,
		"tie grants no skill point"
	)
	match_node.board.paint_cells(
		[Vector2i(1, 1), Vector2i(2, 1)],
		PaintPalette.TEAM_AI
	)
	match_node.call("_end_round")
	_assert(not match_node._last_round_won, "AI lead fails the stage")
	match_node.start_match()
	await process_frame
	_assert(match_node.run_progress.stage_number == 3, "failure retries the same stage")
	_assert(
		match_node.run_progress.total_skill_points() == points_before_loss,
		"failure grants no skill point"
	)
	_assert(
		match_node.run_progress.ai_allocations == retry_allocations,
		"retry keeps the stage AI allocation"
	)
	match_node.call("_enter_lobby")
	_assert(match_node.run_progress.ai_character_ids.is_empty(), "returning to lobby clears campaign growth")
	match_node.queue_free()
	await process_frame
	await process_frame
	print("BnBonline paint smoke: %s" % ("FAILED" if _failed else "PASS"))
	quit(1 if _failed else 0)


func _stop_ai(match_node: MatchController) -> void:
	for actor: GameActor in match_node.get_actors():
		for child: Node in actor.get_children():
			if child is RuleAI:
				(child as RuleAI).stop_thinking()


func _first_ai(match_node: MatchController) -> GameActor:
	for actor: GameActor in match_node.get_actors():
		if not actor.is_player:
			return actor
	return null


func _all_ai(match_node: MatchController) -> Array[GameActor]:
	var result: Array[GameActor] = []
	for actor: GameActor in match_node.get_actors():
		if not actor.is_player:
			result.append(actor)
	return result


func _assert(condition: bool, message: String) -> void:
	if condition:
		return
	_failed = true
	push_error("SMOKE FAILED: %s" % message)


func _item_signature(board: GameBoard) -> Array[String]:
	var result: Array[String] = []
	for item: ArenaItemState in board.get_item_states():
		result.append("%d@%d,%d" % [item.item_type, item.cell.x, item.cell.y])
	return result
