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
		if state in ["match-roster", "match-waddle", "match-trapped"]:
			var roster_cells: Dictionary = {
				"cat": Vector2i(6, 6),
				"bear": Vector2i(8, 6),
				"dog": Vector2i(1, 1),
				"rabbit": Vector2i(13, 1),
				"fox": Vector2i(13, 11),
			}
			for actor: GameActor in match_node.get_actors():
				if roster_cells.has(actor.character_id):
					actor.position = GameConstants.grid_to_world(
						roster_cells[actor.character_id] as Vector2i
					)
					actor.velocity = (
						Vector2.RIGHT * actor.stats.move_speed
						if state == "match-waddle"
						else Vector2.ZERO
					)
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
		if state == "match-settings":
			match_node.call("_open_settings")
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
		"match-waddle",
		"match-trapped",
	] else "18"
	var warmup_frames := int(options.get("warmup", default_warmup))
	for frame_index: int in range(warmup_frames):
		if state == "match-waddle":
			for actor: GameActor in match_node.get_actors():
				if frame_index == warmup_frames / 2:
					actor.velocity *= -1.0
				actor.position += actor.velocity / 60.0
		await process_frame
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
	await process_frame
	await process_frame
	quit()


func _parse_options(arguments: PackedStringArray) -> Dictionary:
	var result: Dictionary = {}
	for argument in arguments:
		if not argument.begins_with("--") or not argument.contains("="):
			continue
		var parts := argument.trim_prefix("--").split("=", true, 1)
		result[parts[0]] = parts[1]
	return result
