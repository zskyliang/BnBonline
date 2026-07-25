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
	elif state.begins_with("match"):
		match_node.settings.ai_count = 4
		match_node.settings.character_id = "builder"
		match_node.settings.map_id = (
			MapCatalog.BELL_GARDEN if state == "match-garden" else MapCatalog.HARBOR_MARKET
		)
		match_node.start_match()
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
