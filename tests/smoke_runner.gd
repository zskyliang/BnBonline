extends SceneTree
## Runs the real main scene long enough to exercise AI, bubble fuse, explosion, and trap flow.

var _failed: bool = false

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	var packed: PackedScene = load("res://scenes/main.tscn") as PackedScene
	var match_node: MatchController = packed.instantiate() as MatchController
	root.add_child(match_node)
	await process_frame
	# Persisted player settings must not make the smoke scenario nondeterministic.
	match_node.settings.map_id = MapCatalog.HARBOR_MARKET
	match_node.settings.ai_count = 1
	match_node.start_match()
	await process_frame
	_assert(match_node.get_actors().size() == match_node.settings.ai_count + 1, "configured fighters spawned")
	_assert(match_node.board.cells.size() == 13, "map initialized")
	_assert(
		match_node._arena_view.board_view.get_building_views().size() \
			== match_node.board.map_data.building_units.size(),
		"every rigid map footprint creates a clay building"
	)
	_assert(
		match_node.board._visual_root.get_child_count() < GameConstants.GRID_COLUMNS * GameConstants.GRID_ROWS,
		"static ground uses batched drawing instead of one node per tile"
	)
	_assert(
		match_node._arena_timer_label.position.y <= 34.0 \
			and match_node._arena_timer_label.position.y + match_node._arena_timer_label.size.y <= 58.0,
		"arena timer stays inside the time header"
	)
	_assert(
		match_node._fps_label.position.y <= 4.0 and match_node._fps_label.text.begins_with("FPS:"),
		"FPS counter is shown on the map-name row"
	)
	var player: GameActor = match_node.get_player()
	_assert(is_instance_valid(player), "player spawned")
	_assert(
		InputMap.has_action("move_left") and InputMap.has_action("place_bomb") \
			and InputMap.has_action("zoom_in") and InputMap.has_action("zoom_out"),
		"movement, bubble and camera zoom input actions registered"
	)
	var default_camera_size := match_node._arena_view.camera.size
	var zoom_in_event := InputEventAction.new()
	zoom_in_event.action = &"zoom_in"
	zoom_in_event.pressed = true
	match_node.call("_unhandled_input", zoom_in_event)
	_assert(
		match_node._arena_view.camera.size < default_camera_size,
		"live match zoom-in input enlarges the map"
	)
	var zoom_reset_event := InputEventAction.new()
	zoom_reset_event.action = &"zoom_reset"
	zoom_reset_event.pressed = true
	match_node.call("_unhandled_input", zoom_reset_event)
	_assert(
		match_node._arena_view.get_zoom_percent() == 110,
		"live match zoom reset restores the recommended scale"
	)
	var zoom_in_button := match_node.hud.find_child("ZoomInButton", true, false) as Button
	_assert(is_instance_valid(zoom_in_button), "match HUD exposes a clickable map zoom control")
	if is_instance_valid(zoom_in_button):
		zoom_in_button.pressed.emit()
		_assert(
			match_node._arena_view.get_zoom_percent() == 120,
			"HUD zoom button updates the live orthographic view"
		)
		match_node._arena_view.reset_zoom()
	match_node.call("_pause_match")
	_assert(paused, "pause suspends the scene tree")
	match_node.call("_resume_match")
	_assert(not paused, "resume restores the scene tree")
	var first_center: Vector2 = GameConstants.grid_to_world(Vector2i.ZERO)
	var last_center: Vector2 = GameConstants.grid_to_world(
		Vector2i(GameConstants.GRID_COLUMNS - 1, GameConstants.GRID_ROWS - 1)
	)
	var arena_center: Vector2 = (first_center + last_center) * 0.5
	var boundary_inputs: Array[Dictionary] = [
		{"start": Vector2(first_center.x, arena_center.y), "action": "move_left", "side": "left"},
		{"start": Vector2(last_center.x, arena_center.y), "action": "move_right", "side": "right"},
		{"start": Vector2(arena_center.x, first_center.y), "action": "move_up", "side": "top"},
		{"start": Vector2(arena_center.x, last_center.y), "action": "move_down", "side": "bottom"},
	]
	for boundary_input: Dictionary in boundary_inputs:
		var boundary_start: Vector2 = boundary_input["start"] as Vector2
		var action: StringName = StringName(boundary_input["action"])
		player.position = boundary_start
		Input.action_press(action)
		for _frame: int in range(3):
			await physics_frame
		Input.action_release(action)
		_assert(
			player.position.is_equal_approx(boundary_start),
			"live %s boundary blocks held outward input" % str(boundary_input["side"])
		)
	var rigid_cell := Vector2i(5, 5)
	for neighbor: Vector2i in [Vector2i.UP, Vector2i.DOWN, Vector2i.LEFT, Vector2i.RIGHT]:
		match_node.board.cells[rigid_cell.y + neighbor.y][rigid_cell.x + neighbor.x] = 0
	match_node.board.cells[rigid_cell.y][rigid_cell.x] = 1
	var rigid_top_left: Vector2 = GameConstants.grid_to_top_left(rigid_cell)
	var rigid_center: Vector2 = GameConstants.grid_to_world(rigid_cell)
	var rigid_inputs: Array[Dictionary] = [
		{"start": Vector2(rigid_top_left.x - 20.0, rigid_center.y), "action": "move_right", "side": "left"},
		{"start": Vector2(rigid_top_left.x + GameConstants.CELL_SIZE + 20.0, rigid_center.y), "action": "move_left", "side": "right"},
		{"start": Vector2(rigid_center.x, rigid_top_left.y - 20.0), "action": "move_down", "side": "top"},
		{"start": Vector2(rigid_center.x, rigid_top_left.y + GameConstants.CELL_SIZE + 20.0), "action": "move_up", "side": "bottom"},
	]
	for rigid_input: Dictionary in rigid_inputs:
		var rigid_start: Vector2 = rigid_input["start"] as Vector2
		var action: StringName = StringName(rigid_input["action"])
		player.position = rigid_start
		Input.action_press(action)
		for _frame: int in range(3):
			await physics_frame
		Input.action_release(action)
		_assert(
			player.position.is_equal_approx(rigid_start),
			"live %s rigid boundary blocks held input" % str(rigid_input["side"])
		)
	var boundary_position := Vector2(
		GameConstants.GRID_ORIGIN.x + GameConstants.CELL_SIZE * 2.0,
		GameConstants.grid_to_world(Vector2i(1, 10)).y
	)
	player.position = boundary_position
	var half_safe_effect := ExplosionEffect.new()
	match_node._effect_root.add_child(half_safe_effect)
	half_safe_effect.setup([Vector2i(1, 10)], Vector2i(1, 10), player)
	_assert(half_safe_effect.get_child_count() == 0, "explosion cells use batched drawing")
	match_node.call("_register_explosion_effect", half_safe_effect)
	match_node.call("_resolve_explosion_hits")
	match_node.call("_resolve_explosion_hits")
	_assert(not player.stats.is_trapped, "one-foot explosion coverage keeps the player half-body safe")
	match_node.call("_unregister_explosion_effect", half_safe_effect)
	half_safe_effect.queue_free()
	var full_hit_effect := ExplosionEffect.new()
	match_node._effect_root.add_child(full_hit_effect)
	full_hit_effect.setup([Vector2i(1, 10), Vector2i(2, 10)], Vector2i(1, 10), player)
	match_node.call("_register_explosion_effect", full_hit_effect)
	match_node.call("_resolve_explosion_hits")
	_assert(not player.stats.is_trapped, "first full-body unsafe frame does not trap")
	match_node.call("_resolve_explosion_hits")
	_assert(player.stats.is_trapped, "two full-body unsafe frames trap in the live match")
	player.rescue()
	match_node.call("_unregister_explosion_effect", full_hit_effect)
	full_hit_effect.queue_free()
	var ai_actor: GameActor
	for battle_actor: GameActor in match_node.get_actors():
		if not battle_actor.is_player:
			ai_actor = battle_actor
			break
	_assert(is_instance_valid(ai_actor), "AI actor available for overlapping-bubble collision test")
	if is_instance_valid(ai_actor):
		for child: Node in ai_actor.get_children():
			if child is RuleAI:
				(child as RuleAI).stop_thinking()
		var overlap_cell := Vector2i(7, 6)
		match_node.board.cells[overlap_cell.y][overlap_cell.x] = 0
		match_node.board.cells[overlap_cell.y + 1][overlap_cell.x] = 0
		player.position = GameConstants.grid_to_world(overlap_cell)
		ai_actor.position = GameConstants.grid_to_world(overlap_cell)
		_assert(match_node.request_bomb(ai_actor), "AI can place a bubble under another actor")
		var overlap_bubble: GameBubble = match_node.board.bombs.get(overlap_cell) as GameBubble
		player.position += Vector2(0.0, 21.0)
		_assert(
			match_node.board.can_actor_occupy(player.position + Vector2(0.0, 3.0), player),
			"non-owner can finish leaving a bubble placed underfoot"
		)
		player.position = GameConstants.grid_to_world(overlap_cell + Vector2i.DOWN)
		_assert(
			not match_node.board.can_actor_occupy(player.position + Vector2(0.0, -14.0), player),
			"non-owner cannot re-enter after fully clearing the underfoot bubble"
		)
		match_node.board.unregister_bubble(overlap_bubble)
		ai_actor.stats.active_bubbles = maxi(0, ai_actor.stats.active_bubbles - 1)
		overlap_bubble.queue_free()
	var player_bubble_cell := Vector2i(1, 11)
	player.position = GameConstants.grid_to_world(player_bubble_cell)
	_assert(match_node.request_bomb(player), "player bubble accepted")
	_assert(match_node.board.bombs.size() >= 1, "bubble registered on board")
	Input.action_press("move_up")
	for _step: int in range(24):
		await physics_frame
	Input.action_release("move_up")
	_assert(
		player.position.y <= GameConstants.grid_to_world(Vector2i(1, 10)).y,
		"player can completely leave a newly placed bubble"
	)
	Input.action_press("move_down")
	for _step: int in range(24):
		await physics_frame
	Input.action_release("move_down")
	_assert(player.current_cell() == Vector2i(1, 10), "player cannot walk back through the bubble after leaving")
	await create_timer(2.0).timeout
	var chained_cell := Vector2i(1, 10)
	player.position = GameConstants.grid_to_world(chained_cell)
	_assert(match_node.request_bomb(player), "second bubble accepted")
	_assert(match_node.board.bombs.has(chained_cell), "second bubble registered")
	await create_timer(1.2).timeout
	_assert(not match_node.board.bombs.has(chained_cell), "first explosion chained the later bubble")
	await create_timer(0.35).timeout
	_assert(player.stats.is_trapped or player.stats.is_dead, "explosion applied combat state")
	if player.stats.is_trapped:
		player.rescue()
		_assert(not player.stats.is_trapped, "self rescue works in live match")
	match_node.settings.map_id = MapCatalog.BELL_GARDEN
	match_node.start_match()
	await process_frame
	_assert(match_node.board.map_data.map_id == MapCatalog.BELL_GARDEN, "second map restarts successfully")
	_assert(match_node.get_actors().size() == match_node.settings.ai_count + 1, "fighters respawn after map change")
	var ai_characters_before_restart: Array[String] = match_node._current_ai_character_ids.duplicate()
	match_node.start_match()
	await process_frame
	_assert(
		match_node._current_ai_character_ids == ai_characters_before_restart,
		"restart preserves this round's AI character assignment"
	)
	match_node.call("_enter_lobby")
	_assert(
		match_node._current_ai_character_ids.is_empty(),
		"returning to the lobby clears AI characters for the next random draw"
	)
	var audio_manager: Node = root.get_node_or_null("AudioManager")
	if is_instance_valid(audio_manager):
		audio_manager.call("stop_all")
	match_node.queue_free()
	await process_frame
	await process_frame
	print("BnBonline smoke test: %s" % ("FAILED" if _failed else "PASS"))
	quit(1 if _failed else 0)

func _assert(condition: bool, message: String) -> void:
	if condition:
		return
	_failed = true
	push_error("SMOKE FAILED: %s" % message)
