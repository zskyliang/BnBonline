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
		match_node.settings.character_id = "builder"
		match_node.settings.player_color_id = "orange"
		match_node.settings.camera_azimuth = float(options.get(
			"azimuth",
			MatchSettings.DEFAULT_CAMERA_AZIMUTH
		))
		match_node.settings.camera_elevation = float(options.get(
			"elevation",
			MatchSettings.DEFAULT_CAMERA_ELEVATION
		))
		match_node.settings.camera_zoom = float(options.get(
			"zoom",
			MatchSettings.DEFAULT_CAMERA_ZOOM
		))
		match_node.call("_begin_new_run")
		for skill_id: String in [
			RunProgress.SKILL_SPEED,
			RunProgress.SKILL_BUBBLE,
			RunProgress.SKILL_POWER,
		]:
			match_node.run_progress.advance_with_skill(skill_id, match_node._rng)
		match_node.start_match()
		if options.has("azimuth") \
				or options.has("elevation") \
				or options.has("zoom"):
			match_node._arena_view.set_camera_pose(
				float(options.get("azimuth", match_node._arena_view.get_azimuth())),
				float(options.get("elevation", match_node._arena_view.get_elevation())),
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
		for item_type: int in ArenaItemType.ALL:
			match_node.board.spawn_item(
				item_type,
				Vector2i(5 + item_type * 2, 4),
				10000
			)
		if state == "match-settings":
			match_node.call("_open_settings")
		if state == "match-explosion":
			var effect := ExplosionEffect.new()
			match_node._effect_root.add_child(effect)
			effect.setup(
				[
					Vector2i(4, 9), Vector2i(3, 9), Vector2i(2, 9), Vector2i(1, 9),
					Vector2i(5, 9), Vector2i(6, 9), Vector2i(4, 8), Vector2i(4, 7),
					Vector2i(4, 10), Vector2i(4, 11),
				],
				Vector2i(4, 9),
				match_node.get_player()
			)
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
	for _frame in range(18):
		await process_frame
	var image := root.get_texture().get_image()
	var output_path := str(options.get("output", "/tmp/bnb-clay-%s.png" % state))
	var error := image.save_png(output_path)
	if error != OK:
		push_error("Unable to save visual snapshot: %s" % error_string(error))
		quit(1)
		return
	print("Saved visual snapshot: %s (%dx%d)" % [output_path, image.get_width(), image.get_height()])
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
