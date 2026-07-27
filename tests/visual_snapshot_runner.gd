extends SceneTree
## Manual visual regression helper. Run with a graphical renderer, not --headless.


func _initialize() -> void:
	call_deferred("_capture")


func _capture() -> void:
	var options := _parse_options(OS.get_cmdline_user_args())
	var requested_size := Vector2i(
		int(options.get("width", "1280")),
		int(options.get("height", "720"))
	)
	DisplayServer.window_set_size(requested_size)
	var packed := load("res://scenes/main.tscn") as PackedScene
	var match_node := packed.instantiate() as MatchController
	root.add_child(match_node)
	await process_frame
	if options.has("language"):
		match_node.call("_on_language_changed", str(options["language"]))
		await process_frame
	var state := str(options.get("state", "lobby"))
	if state == "setup":
		match_node.call("_enter_setup")
	elif state.begins_with("match") or state.begins_with("result"):
		match_node.call("_enter_lobby")
		match_node.settings.character_id = str(options.get("character", "cat"))
		match_node.settings.player_color_id = "orange"
		match_node.settings.camera_zoom = float(options.get(
			"zoom",
			MatchSettings.DEFAULT_CAMERA_ZOOM
		))
		match_node.call("_begin_new_run")
		match_node.run_progress.ai_character_ids = [
			"bear", "dog", "rabbit", "fox",
		]
		for skill_id: String in [
			RunProgress.SKILL_SPEED,
			RunProgress.SKILL_BUBBLE,
			RunProgress.SKILL_POWER,
		]:
			match_node.run_progress.advance_with_skill(skill_id, match_node._rng)
		match_node.start_match()
		if state in [
			"match-roster",
			"match-walk-up",
			"match-walk-down",
			"match-walk-left",
			"match-walk-right",
			"match-walk-mixed",
			"match-trapped",
		]:
			var roster_cells: Dictionary = {
				"cat": Vector2i(6, 6),
				"bear": Vector2i(8, 6),
				"dog": Vector2i(1, 1),
				"rabbit": Vector2i(13, 1),
				"fox": Vector2i(13, 11),
			}
			roster_cells[match_node.settings.character_id] = Vector2i(6, 6)
			for actor: GameActor in match_node.get_actors():
				if roster_cells.has(actor.character_id):
					actor.position = GameConstants.grid_to_world(
						roster_cells[actor.character_id] as Vector2i
					)
					actor.velocity = _snapshot_velocity(
						state,
						actor.character_id,
						actor.stats.move_speed
					)
					if actor.velocity.length_squared() > 0.0:
						actor.call("_update_facing", actor.velocity)
			if state == "match-trapped":
				var victim: GameActor = match_node.get_player()
				victim.stats.is_trapped = true
				victim.last_attacker = match_node.get_actors()[1]
			match_node._is_paused = true
			match_node.get_tree().paused = true
		if options.has("zoom"):
			match_node._arena_view.set_camera_pose(
				MatchSettings.DEFAULT_CAMERA_AZIMUTH,
				MatchSettings.DEFAULT_CAMERA_ELEVATION,
				float(options.get("zoom", match_node._arena_view.get_zoom()))
			)
		match_node.board.paint_cells(
			[
				Vector2i(2, 2), Vector2i(3, 2), Vector2i(4, 2),
				Vector2i(2, 3), Vector2i(3, 3), Vector2i(4, 3),
			],
			PaintPalette.TEAM_PLAYER
		)
		match_node.board.paint_cells(
			[
				Vector2i(9, 7), Vector2i(10, 7), Vector2i(11, 7),
				Vector2i(9, 8), Vector2i(10, 8), Vector2i(11, 8),
			],
			PaintPalette.TEAM_AI
		)
		match_node.board.lock_neighborhood(Vector2i(7, 6), PaintPalette.TEAM_PLAYER)
		if state in ["match-items", "match-explosion"]:
			for row: int in range(3):
				for item_type: int in ArenaItemType.ALL:
					match_node.board.spawn_item(
						item_type,
						Vector2i(4 + item_type * 3, 3 + row * 3),
						10000 + row * 100 + item_type
					)
		if state == "match-bubbles":
			var bubble_cells: Array[Vector2i] = [
				Vector2i(3, 3),
				Vector2i(6, 3),
				Vector2i(9, 3),
				Vector2i(12, 3),
				Vector2i(3, 9),
				Vector2i(6, 9),
				Vector2i(9, 9),
				Vector2i(12, 9),
			]
			var bubble_colors: Array[String] = [
				"red", "orange", "yellow", "green",
				"cyan", "blue", "purple", "red",
			]
			for character_index: int in range(CharacterCatalog.IDS.size()):
				var owner := GameActor.new()
				match_node._entity_root.add_child(owner)
				owner.setup(
					"BubbleOwner%d" % character_index,
					PaintPalette.TEAM_PLAYER,
					false,
					match_node.board,
					match_node.settings,
					bubble_cells[character_index],
					bubble_colors[character_index]
				)
				owner.character_id = CharacterCatalog.IDS[character_index]
				owner.visible = false
				var bubble := GameBubble.new()
				match_node._entity_root.add_child(bubble)
				bubble.setup(
					owner,
					bubble_cells[character_index],
					30.0,
					[owner]
				)
				match_node._arena_view.add_bubble(bubble)
			match_node._is_paused = true
			match_node.get_tree().paused = true
		if state == "match-settings":
			match_node.call("_open_settings")
		if state == "match-rain":
			match_node._arena_view.set_weather(
				ForestWeatherSystem3D.Weather.RAIN,
				true
			)
		if state == "match-explosion":
			for center: Vector2i in [
				Vector2i(4, 9),
				Vector2i(7, 7),
				Vector2i(10, 5),
			]:
				var effect := ExplosionEffect.new()
				match_node._effect_root.add_child(effect)
				var effect_cells := GameRules.blast_cells(
					center,
					3,
					match_node.board.map_data.barrier_cells
				)
				effect.setup(effect_cells, center, match_node.get_player())
				effect.set_process(false)
				var effect_view := match_node._arena_view.add_explosion(effect)
				effect_view.call("_process", 0.09)
				effect_view.set_process(false)
		if state == "result-loss":
			match_node.board.paint_cells(
				match_node.board.get_open_cells(),
				PaintPalette.TEAM_AI
			)
			match_node.call("_end_round")
		elif state == "result-win":
			match_node.call("_end_round")
	var default_warmup := "80" if state in [
		"match-roster",
		"match-walk-up",
		"match-walk-down",
		"match-walk-left",
		"match-walk-right",
		"match-walk-mixed",
		"match-trapped",
		"match-bubbles",
		"match-rain",
	] else "18"
	var warmup_frames := int(options.get("warmup", default_warmup))
	for frame_index: int in range(warmup_frames):
		if state.begins_with("match-walk"):
			for actor: GameActor in match_node.get_actors():
				actor.position += actor.velocity / 60.0
		await process_frame
	if state == "match-rain":
		for ripple_cell: Vector2i in [
			Vector2i(2, 2),
			Vector2i(5, 4),
			Vector2i(7, 6),
			Vector2i(10, 8),
			Vector2i(12, 10),
		]:
			match_node._arena_view.weather_system.debug_spawn_ripple(ripple_cell)
		match_node._arena_view.weather_system.call("_process", 0.12)
	var image := root.get_texture().get_image()
	var output_path := str(options.get("output", "/tmp/forest-bubble-%s.png" % state))
	var error := image.save_png(output_path)
	if error != OK:
		push_error("Unable to save visual snapshot: %s" % error_string(error))
		quit(1)
		return
	print("Saved visual snapshot: %s (%dx%d)" % [output_path, image.get_width(), image.get_height()])
	paused = false
	match_node.queue_free()
	for cleanup_frame: int in range(8):
		await process_frame
	quit()


func _snapshot_velocity(
		state: String,
		character_id: String,
		speed: float
	) -> Vector2:
	var direction := Vector2.ZERO
	match state:
		"match-walk-up":
			direction = Vector2.UP
		"match-walk-down":
			direction = Vector2.DOWN
		"match-walk-left":
			direction = Vector2.LEFT
		"match-walk-right":
			direction = Vector2.RIGHT
		"match-walk-mixed":
			var directions: Dictionary = {
				"cat": Vector2.UP,
				"bear": Vector2.DOWN,
				"dog": Vector2.LEFT,
				"rabbit": Vector2.RIGHT,
				"fox": Vector2.UP,
			}
			direction = directions.get(character_id, Vector2.DOWN) as Vector2
	return direction * speed


func _parse_options(arguments: PackedStringArray) -> Dictionary:
	var result: Dictionary = {}
	for argument in arguments:
		if not argument.begins_with("--") or not argument.contains("="):
			continue
		var parts := argument.trim_prefix("--").split("=", true, 1)
		result[parts[0]] = parts[1]
	return result
