extends SceneTree
## Headless visual-structure checks for character palettes and painted floor batches.

var _checks: int = 0
var _failures: int = 0


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	_test_coordinate_mapping()
	_test_catalog_and_clothing_materials()
	_test_clay_material_constraints()
	_test_audio_assets()
	await _test_character_palette_views()
	await _test_painted_board_view()
	await _test_colored_bubbles_and_explosions()
	await _test_item_views_and_camera()
	print("BnBonline paint visuals: %d checks, %d failures" % [_checks, _failures])
	quit(1 if _failures > 0 else 0)


func _test_coordinate_mapping() -> void:
	_check(
		GameConstants.grid_to_world_3d(Vector2i.ZERO).is_equal_approx(Vector3(-7.0, 0.0, -6.0)),
		"top-left floor maps to centered XZ arena"
	)
	_check(
		GameConstants.grid_to_world_3d(Vector2i(7, 6), 0.5).is_equal_approx(Vector3(0.0, 0.5, 0.0)),
		"center floor maps to world origin"
	)
	var logic_position := GameConstants.grid_to_world(Vector2i(11, 9)) + Vector2(10.0, -5.0)
	_check(
		GameConstants.logic_to_world_3d(logic_position, 0.25).is_equal_approx(
			Vector3(4.25, 0.25, 2.875)
		),
		"continuous actor motion preserves sub-cell offsets"
	)


func _test_catalog_and_clothing_materials() -> void:
	var definitions: Array[CharacterDefinition] = CharacterCatalog.get_all()
	_check(definitions.size() == 8, "catalog keeps all eight characters")
	for definition: CharacterDefinition in definitions:
		var scene: PackedScene = definition.load_model_scene()
		_check(scene != null, "%s model is available" % definition.id)
		_check(
			not definition.clothing_material_names.is_empty(),
			"%s declares clothing material names" % definition.id
		)
		if scene == null:
			continue
		var instance := scene.instantiate() as Node3D
		var available_names: Dictionary = {}
		for node: Node in instance.find_children("*", "MeshInstance3D", true, false):
			var mesh_instance := node as MeshInstance3D
			if mesh_instance.mesh == null:
				continue
			for surface: int in range(mesh_instance.mesh.get_surface_count()):
				var material := mesh_instance.mesh.surface_get_material(surface)
				if material != null:
					available_names[material.resource_name] = true
		var matched: bool = false
		for clothing_name: String in definition.clothing_material_names:
			matched = matched or available_names.has(clothing_name)
		_check(matched, "%s clothing whitelist matches imported materials" % definition.id)
		instance.free()


func _test_clay_material_constraints() -> void:
	var material := ClayMaterialLibrary.make(Color("#d94f45"))
	_check(material.roughness >= 0.86, "paint material remains matte")
	_check(material.metallic == 0.0, "paint material is non-metallic")
	_check(material.normal_enabled, "paint material keeps clay micro-normal")
	var font := load("res://assets/fonts/NotoSansSC-BnB-Subset.ttf") as Font
	_check(is_instance_valid(font), "runtime Chinese font loads")


func _test_audio_assets() -> void:
	var music := load("res://assets/audio/music/battle_loop.ogg") as AudioStreamOggVorbis
	_check(is_instance_valid(music), "replacement CC0 battle loop loads")
	if is_instance_valid(music):
		_check(music.get_length() > 40.0, "cute battle loop has a full-length gameplay phrase")
		_check(
			music.loop \
				or FileAccess.get_file_as_string(
					"res://assets/audio/music/battle_loop.ogg.import"
				).contains("loop=true"),
			"battle music is configured as a seamless forward loop"
		)
	var cue_paths: Dictionary = {
		"start": "res://assets/audio/sfx/start.ogg",
		"appear": "res://assets/audio/sfx/appear.wav",
		"lay": "res://assets/audio/sfx/lay.wav",
		"explode": "res://assets/audio/sfx/explode.wav",
		"get": "res://assets/audio/sfx/get.ogg",
		"save": "res://assets/audio/sfx/save.ogg",
		"die": "res://assets/audio/sfx/die.ogg",
		"win": "res://assets/audio/sfx/win.ogg",
		"draw": "res://assets/audio/sfx/draw.ogg",
	}
	for cue_name: String in cue_paths:
		var cue := load(str(cue_paths[cue_name])) as AudioStream
		_check(is_instance_valid(cue), "%s replacement sound loads" % cue_name)
		if is_instance_valid(cue):
			_check(cue.get_length() >= 0.04, "%s replacement sound is non-empty" % cue_name)
	var audio_manager: Node = root.get_node_or_null("AudioManager")
	_check(is_instance_valid(audio_manager), "audio manager is available for mix validation")
	if is_instance_valid(audio_manager):
		_check(
			float(audio_manager.call("get_sfx_volume_offset_db", &"explode")) <= -8.0,
			"cute bubble pop receives at least 8 dB extra attenuation"
		)
	_check(
		not FileAccess.file_exists("res://assets/audio/music/battle_loop.wav") \
			and not FileAccess.file_exists("res://assets/audio/sfx/explode.ogg"),
		"mechanical battle loop and impact explosion are no longer shipped"
	)


func _test_character_palette_views() -> void:
	var board := GameBoard.new()
	root.add_child(board)
	board.reset(MapCatalog.get_map())
	var settings := MatchSettings.new()
	var actor := GameActor.new()
	root.add_child(actor)
	actor.setup(
		"测试",
		PaintPalette.TEAM_PLAYER,
		false,
		board,
		settings,
		Vector2i(3, 4),
		"purple"
	)
	actor.character_id = "wizard"
	var view_root := Node3D.new()
	root.add_child(view_root)
	var actor_view := ActorView3D.new()
	view_root.add_child(actor_view)
	actor_view.bind_actor(actor, "wizard", "purple")
	await process_frame
	_check(actor_view.color_id == "purple", "actor view stores selected clothing color")
	var clothing_found: bool = _has_surface_color(
		actor_view,
		CharacterCatalog.get_definition("wizard").clothing_material_names,
		PaintPalette.get_color("purple")
	)
	_check(clothing_found, "wizard clothing surfaces receive selected purple palette")
	actor.position += Vector2(13.0, 7.0)
	actor.velocity = Vector2.RIGHT * actor.stats.move_speed
	actor_view.call("_process", 0.09)
	_check(
		actor_view.position.is_equal_approx(GameConstants.logic_to_world_3d(actor.position, 0.04)),
		"3D actor follows logic movement"
	)
	var preview := CharacterPreview3D.new()
	root.add_child(preview)
	preview.setup(CharacterCatalog.get_definition("builder"), "red")
	await process_frame
	preview.set_color_id("cyan")
	_check(preview.color_id == "cyan", "lobby preview updates when a new color is selected")
	preview.queue_free()
	actor_view.queue_free()
	actor.queue_free()
	board.queue_free()
	view_root.queue_free()
	await process_frame


func _test_painted_board_view() -> void:
	var board := GameBoard.new()
	root.add_child(board)
	board.configure_team_colors("orange", "blue")
	board.reset(MapCatalog.get_map())
	var view := BoardView3D.new()
	root.add_child(view)
	view.bind_board(board)
	await process_frame
	var floor := view.find_child("PaintFloorTiles", true, false) as MultiMeshInstance3D
	var locks := view.find_child("LockedTileShadows", true, false) as MultiMeshInstance3D
	_check(is_instance_valid(floor), "paint floor is rendered as a MultiMesh")
	_check(
		is_instance_valid(floor) and floor.multimesh.instance_count == 195,
		"floor batch contains exactly 195 tile instances"
	)
	_check(view.get_building_views().is_empty(), "paint arena has no building views")
	var target := Vector2i(5, 5)
	board.paint_cells([target], PaintPalette.TEAM_PLAYER)
	var index: int = target.y * GameConstants.GRID_COLUMNS + target.x
	_check(
		_colors_close(
			view.displayed_color(target),
			PaintPalette.get_color("orange")
		),
		"paint signal incrementally updates one tile color"
	)
	board.lock_neighborhood(target, PaintPalette.TEAM_PLAYER)
	var lock_transform: Transform3D = locks.multimesh.get_instance_transform(index)
	_check(
		lock_transform.basis.get_scale().length() > 1.0,
		"locked tile exposes a full-cell shadow layer"
	)
	var shadow_material := locks.material_override as StandardMaterial3D
	_check(
		shadow_material != null \
			and shadow_material.transparency == BaseMaterial3D.TRANSPARENCY_ALPHA \
			and shadow_material.albedo_color.a < 0.5,
		"permanent territory uses only a translucent shadow overlay"
	)
	board.paint_cells([target], PaintPalette.TEAM_AI)
	_check(
		_colors_close(
			view.displayed_color(target),
			PaintPalette.get_color("orange")
		),
		"locked visual remains the earliest owner color"
	)
	view.queue_free()
	board.queue_free()
	await process_frame


func _test_colored_bubbles_and_explosions() -> void:
	var board := GameBoard.new()
	root.add_child(board)
	board.reset(MapCatalog.get_map())
	var actor := GameActor.new()
	root.add_child(actor)
	actor.setup(
		"测试",
		PaintPalette.TEAM_PLAYER,
		false,
		board,
		MatchSettings.new(),
		Vector2i(7, 6),
		"green"
	)
	var bubble := GameBubble.new()
	root.add_child(bubble)
	bubble.setup(actor, Vector2i(7, 6), 10.0, [actor])
	var bubble_view := BubbleView3D.new()
	root.add_child(bubble_view)
	bubble_view.bind_bubble(bubble)
	_check(
		(bubble_view.get("_skin_color") as Color).is_equal_approx(PaintPalette.get_color("green")),
		"bubble visual inherits owner green"
	)
	var effect := ExplosionEffect.new()
	root.add_child(effect)
	effect.setup([Vector2i(7, 6), Vector2i(8, 6)], Vector2i(7, 6), actor)
	var explosion_view := ExplosionView3D.new()
	root.add_child(explosion_view)
	explosion_view.activate(effect, 2)
	await process_frame
	var splash := explosion_view.find_child("ClayWaterSplashCells", true, false) as MultiMeshInstance3D
	var splash_material := splash.material_override as StandardMaterial3D
	_check(
		splash_material.albedo_color.is_equal_approx(PaintPalette.get_color("green")),
		"explosion splash inherits owner green"
	)
	explosion_view.call("_process", 0.09)
	_check(explosion_view.visual_stage() == 1, "colored explosion keeps staged animation")
	var arena := ArenaView3D.new()
	root.add_child(arena)
	await process_frame
	_check(arena.find_child("RoofOcclusionController", true, false) == null, "flat arena removes roof occlusion")
	arena.play_defeat_burst(Vector2i(7, 6), "purple", [Vector2i(7, 6)])
	_check(
		(arena.explosion_root as ClayExplosionPool).get_stats()["in_use"] >= 1,
		"defeat burst reuses pooled explosion visuals"
	)
	arena.queue_free()
	explosion_view.queue_free()
	effect.queue_free()
	bubble_view.queue_free()
	bubble.queue_free()
	actor.queue_free()
	board.queue_free()
	await process_frame


func _test_item_views_and_camera() -> void:
	var root_3d := Node3D.new()
	root.add_child(root_3d)
	for item_type: int in ArenaItemType.ALL:
		var packed: PackedScene = load(ArenaItemType.model_path(item_type)) as PackedScene
		_check(packed != null, "%s pickup GLB imports" % ArenaItemType.display_name(item_type))
		var view := ItemView3D.new()
		root_3d.add_child(view)
		view.setup(ArenaItemState.new(item_type + 1, item_type, Vector2i(item_type, 1)))
		_check(
			view.find_child("ImportedModel", true, false) != null \
				and view.find_child("PickupRing", true, false) != null,
			"%s pickup uses imported model and clay identification ring"
				% ArenaItemType.display_name(item_type)
		)
		if item_type == ArenaItemType.Value.BUBBLE:
			_check(
				view.find_child("BubblePlusOne", true, false) != null,
				"bubble-count pickup has an explicit plus-one marker"
			)
	var arena := ArenaView3D.new()
	root.add_child(arena)
	await process_frame
	arena.set_camera_pose(999.0, -999.0, 9.0)
	_check(
		arena.get_azimuth() == MatchSettings.MAX_CAMERA_AZIMUTH \
			and arena.get_elevation() == MatchSettings.MIN_CAMERA_ELEVATION \
			and arena.get_zoom() == MatchSettings.MAX_CAMERA_ZOOM,
		"runtime camera pose clamps to configured limits"
	)
	arena.set_camera_pose(12.0, 52.0, 1.42)
	arena.reset_zoom()
	_check(
		arena.get_azimuth() == 12.0 \
			and arena.get_elevation() == 52.0 \
			and arena.get_zoom() == MatchSettings.DEFAULT_CAMERA_ZOOM,
		"zoom reset preserves the two configured camera angles"
	)
	arena.reset_camera()
	_check(
		arena.get_azimuth() == MatchSettings.DEFAULT_CAMERA_AZIMUTH \
			and arena.get_elevation() == MatchSettings.DEFAULT_CAMERA_ELEVATION \
			and arena.get_zoom() == MatchSettings.DEFAULT_CAMERA_ZOOM,
		"camera reset restores all three persisted defaults"
	)
	var adjustment_finished: Array[bool] = []
	arena.camera_adjustment_finished.connect(
		func(_azimuth: float, _elevation: float, _zoom: float) -> void:
			adjustment_finished.append(true)
	)
	var press := InputEventMouseButton.new()
	press.button_index = MOUSE_BUTTON_RIGHT
	press.pressed = true
	arena.call("_input", press)
	var motion := InputEventMouseMotion.new()
	motion.relative = Vector2(20.0, 10.0)
	arena.call("_input", motion)
	var release := InputEventMouseButton.new()
	release.button_index = MOUSE_BUTTON_RIGHT
	release.pressed = false
	arena.call("_input", release)
	_check(
		arena.get_azimuth() < MatchSettings.DEFAULT_CAMERA_AZIMUTH \
			and arena.get_elevation() > MatchSettings.DEFAULT_CAMERA_ELEVATION,
		"right-drag changes horizontal and overhead camera angles"
	)
	_check(
		adjustment_finished.size() == 1,
		"right-drag persists only when the adjustment finishes"
	)
	arena.queue_free()
	root_3d.queue_free()
	await process_frame


func _has_surface_color(root_node: Node, names: Array[String], target: Color) -> bool:
	for node: Node in root_node.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := node as MeshInstance3D
		if mesh_instance.mesh == null:
			continue
		for surface: int in range(mesh_instance.mesh.get_surface_count()):
			var source := mesh_instance.mesh.surface_get_material(surface)
			if source == null or source.resource_name not in names:
				continue
			var override := mesh_instance.get_surface_override_material(surface) as StandardMaterial3D
			if override == null:
				continue
			var distance: float = Vector3(
				override.albedo_color.r - target.r,
				override.albedo_color.g - target.g,
				override.albedo_color.b - target.b
			).length()
			if distance < 0.22:
				return true
	return false


func _colors_close(left: Color, right: Color) -> bool:
	return Vector3(left.r - right.r, left.g - right.g, left.b - right.b).length() < 0.025


func _check(condition: bool, description: String) -> void:
	_checks += 1
	if condition:
		return
	_failures += 1
	push_error("VISUAL TEST FAILED: %s" % description)
