extends SceneTree
## Headless structural checks for the ImageGen + Sprite3D presentation layer.

var _checks: int = 0
var _failures: int = 0


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	_test_coordinate_mapping()
	_test_catalog_and_sprite_materials()
	_test_storybook_material_constraints()
	_test_audio_assets()
	await _test_setup_character_radars()
	await _test_lobby_pad_alignment()
	await _test_directional_character_sprite_views()
	await _test_painted_board_view()
	await _test_weather_and_ripples()
	await _test_colored_bubbles_and_explosions()
	await _test_item_views_and_camera()
	print("BnBonline Sprite3D visuals: %d checks, %d failures" % [_checks, _failures])
	quit(1 if _failures > 0 else 0)


func _test_coordinate_mapping() -> void:
	_check(
		GameConstants.grid_to_world_3d(Vector2i.ZERO).is_equal_approx(
			Vector3(-7.0, 0.0, -6.0)
		),
		"top-left floor maps to centered XZ arena"
	)
	_check(
		GameConstants.grid_to_world_3d(Vector2i(7, 6), 0.5).is_equal_approx(
			Vector3(0.0, 0.5, 0.0)
		),
		"center floor maps to world origin"
	)
	var logic_position := GameConstants.grid_to_world(Vector2i(11, 9)) \
		+ Vector2(10.0, -5.0)
	_check(
		GameConstants.logic_to_world_3d(logic_position, 0.25).is_equal_approx(
			Vector3(4.25, 0.25, 2.875)
		),
		"continuous actor motion preserves sub-cell offsets"
	)


func _test_catalog_and_sprite_materials() -> void:
	var definitions: Array[CharacterDefinition] = CharacterCatalog.get_all()
	_check(definitions.size() == 8, "catalog keeps all eight animal identities")
	for definition: CharacterDefinition in definitions:
		var sprite_set := definition.load_sprite_set()
		_check(sprite_set != null, "%s SpriteSet is available" % definition.id)
		_check(
			definition.sprite_set_path.begins_with(
				"res://assets/art/storybook25d/characters/"
			),
			"%s uses only runtime ImageGen artwork" % definition.id
		)
		for action: StringName in CharacterSpriteSet.ACTIONS:
			_check(
				sprite_set.has_action(action),
				"%s exposes %s paper-puppet action" % [definition.id, action]
			)
	var base := load(
		"res://assets/art/storybook25d/characters/cat/idle_down.png"
	) as Texture2D
	var mask := load(
		"res://assets/art/storybook25d/characters/cat/idle_down_mask.png"
	) as Texture2D
	var tinted := StorybookMaterialLibrary.make_sprite_material(
		base,
		mask,
		PaintPalette.get_color("purple")
	)
	_check(tinted.shader != null, "character team tint uses a Compatibility shader")
	_check(
		(tinted.get_shader_parameter("team_color") as Color).is_equal_approx(
			PaintPalette.get_color("purple")
		),
		"local mask receives the selected team color"
	)


func _test_storybook_material_constraints() -> void:
	var material := StorybookMaterialLibrary.make(Color("#d94f45"))
	_check(
		material.shading_mode == BaseMaterial3D.SHADING_MODE_UNSHADED,
		"primitive materials preserve ImageGen colors with unlit shading"
	)
	_check(
		material.cull_mode == BaseMaterial3D.CULL_DISABLED,
		"paper primitives are visible from both sides"
	)
	var texture := load(
		"res://assets/art/storybook25d/environment/conifer.png"
	) as Texture2D
	var textured := StorybookMaterialLibrary.make_textured(texture)
	_check(
		textured.transparency == BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR,
		"forest cutouts use depth-safe alpha scissor"
	)
	var font := load("res://assets/fonts/NotoSansSC-BnB-Subset.ttf") as Font
	_check(is_instance_valid(font), "runtime Chinese font loads")


func _test_audio_assets() -> void:
	var music := load(
		"res://assets/audio/music/puddle_jumpers_loop.ogg"
	) as AudioStreamOggVorbis
	_check(is_instance_valid(music), "user-provided Puddle Jumpers battle loop loads")
	if is_instance_valid(music):
		_check(
			music.get_length() > 55.0,
			"Puddle Jumpers keeps a full-length crossfaded loop"
		)
	var ambience := load(
		"res://assets/audio/ambience/gentle_rain_thunder_loop.ogg"
	) as AudioStreamOggVorbis
	_check(is_instance_valid(ambience), "CC0 gentle rain and thunder ambience loads")
	if is_instance_valid(ambience):
		_check(
			ambience.get_length() > 50.0,
			"rain and thunder ambience has a long loop without repetition"
		)
	var audio_manager: Node = root.get_node_or_null("AudioManager")
	if is_instance_valid(audio_manager):
		audio_manager.call("play_music")
		_check(
			bool(audio_manager.call("is_rain_thunder_ambience_playing")),
			"battle music starts the soft storm ambience as a second layer"
		)
		_check(
			float(audio_manager.call("get_rain_thunder_volume_db")) <= -18.0,
			"storm ambience stays gently below the music mix"
		)
		audio_manager.call("stop_music")
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


func _test_setup_character_radars() -> void:
	TranslationServer.set_locale("en")
	var hud := GameHud.new()
	root.add_child(hud)
	await process_frame
	hud.show_setup(MatchSettings.new())
	await process_frame
	var radars := hud.find_children(
		"*InitialStatsRadar",
		"CharacterStatRadar",
		true,
		false
	)
	_check(radars.size() == 8, "character selection renders eight initial-stat radar charts")
	for radar_node: Node in radars:
		var radar := radar_node as CharacterStatRadar
		var snapshot := radar.attribute_snapshot()
		_check(
			int(snapshot["total"]) == 6,
			"%s radar displays all six starting points" % radar.definition.id
		)
	var speed_reward := hud.find_child(
		"SpeedSkillButton",
		true,
		false
	) as Button
	_check(
		speed_reward != null and tr(speed_reward.text).contains("Speed +1"),
		"default-English reward UI expresses speed growth as plus one point"
	)
	var language_selector := hud.find_child(
		"LanguageSelector",
		true,
		false
	) as OptionButton
	_check(
		language_selector != null and language_selector.item_count == 2 \
			and str(language_selector.get_item_metadata(0)) == "en" \
			and str(language_selector.get_item_metadata(1)) == "zh",
		"settings exposes exactly the English and Chinese language choices"
	)
	hud.queue_free()
	await process_frame


func _test_lobby_pad_alignment() -> void:
	var lobby := StorybookLobbyDiorama3D.new()
	lobby.size = Vector2i(1280, 720)
	root.add_child(lobby)
	await process_frame
	var report := lobby.get_pad_alignment_report()
	_check(report.size() == 8, "lobby places all eight animals on painted pads")
	var anchors: Array[Vector3] = []
	for entry: Dictionary in report:
		var anchor := entry["feet_anchor"] as Vector3
		var expected := entry["expected_anchor"] as Vector3
		anchors.append(anchor)
		_check(
			anchor.is_equal_approx(expected),
			"%s feet are locked to its background pad"
				% str(entry["character_id"])
		)
		_check(
			absf(anchor.x) <= 1.5 and anchor.y >= 1.55 and anchor.y <= 3.0,
			"%s remains inside the eight-pad clearing"
				% str(entry["character_id"])
		)
	var unique_anchors: Dictionary = {}
	for anchor: Vector3 in anchors:
		unique_anchors["%.2f,%.2f" % [anchor.x, anchor.y]] = true
	_check(unique_anchors.size() == 8, "no two lobby animals share one pad")
	lobby.queue_free()
	await process_frame


func _test_directional_character_sprite_views() -> void:
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
		Vector2i(3, 4),
		"purple"
	)
	actor.character_id = "cat"
	var actor_view := ActorView3D.new()
	root.add_child(actor_view)
	actor_view.bind_actor(actor, "cat", "purple")
	await process_frame
	var dog_size_view := ActorView3D.new()
	root.add_child(dog_size_view)
	dog_size_view.bind_actor(actor, "dog", "purple")
	var dog_pixel_size := dog_size_view.get_battle_pixel_size()
	var dog_idle_height := dog_size_view.get_max_idle_cell_height()
	dog_size_view.queue_free()
	await process_frame
	for character_id: String in CharacterCatalog.IDS:
		var size_view := ActorView3D.new()
		root.add_child(size_view)
		size_view.bind_actor(actor, character_id, "purple")
		_check(
			is_equal_approx(
				size_view.get_battle_pixel_size(),
				dog_pixel_size
			),
			"%s uses the approved dog battle pixel scale"
				% character_id
		)
		_check(
			absf(
				size_view.get_max_idle_cell_height()
				- dog_idle_height
			) <= 0.01,
			"%s matches the dog's battle character height"
				% character_id
		)
		_check(
			size_view.get_character_feet_clearance() >= 0.02,
			"%s keeps its complete feet above the board after battle scaling"
				% character_id
		)
		size_view.queue_free()
	await process_frame
	var penguin_view := ActorView3D.new()
	root.add_child(penguin_view)
	penguin_view.bind_actor(actor, "penguin", "purple")
	penguin_view.set_process(false)
	var penguin_up_heights: Array[float] = []
	actor.velocity = Vector2.UP * actor.stats.move_speed
	actor.call("_update_facing", actor.velocity)
	for sample_index: int in range(8):
		actor.position += Vector2.UP * 10.0
		penguin_view.call("_process", 0.13)
		penguin_up_heights.append(
			penguin_view.get_visible_frame_cell_height()
		)
	_check(
		_maximum(penguin_up_heights) - _minimum(penguin_up_heights) <= 0.01,
		"penguin WalkUp frames keep one stable back-view character height"
	)
	penguin_view.queue_free()
	actor.velocity = Vector2.ZERO
	await process_frame
	actor_view.set_process(false)
	var sprite := actor_view.find_child(
		"ImageGenDirectionalCharacterSprite",
		true,
		false
	) as Sprite3D
	_check(sprite != null, "actor view is rendered by an ImageGen Sprite3D")
	_check(
		sprite != null and sprite.billboard == BaseMaterial3D.BILLBOARD_ENABLED,
		"character paper puppet always faces the fixed camera"
	)
	var tint := sprite.material_override as ShaderMaterial
	_check(
		tint != null and (
			tint.get_shader_parameter("team_color") as Color
		).is_equal_approx(PaintPalette.get_color("purple")),
		"cat team-color mask receives purple without tinting the whole animal"
	)
	for action: StringName in CharacterSpriteSet.ACTIONS:
		_check(actor_view.has_action(action), "cat has %s directional artwork" % action)
	_check(
		not actor_view.has_method("play_action") \
			and not actor_view.has_action(&"PlaceBubble") \
			and not actor_view.has_action(&"Defeat") \
			and not actor_view.has_action(&"Victory"),
		"old one-shot action API and artwork are absent"
	)
	var direction_cases: Array[Dictionary] = [
		{"facing": &"up", "velocity": Vector2.UP, "action": &"WalkUp"},
		{"facing": &"down", "velocity": Vector2.DOWN, "action": &"WalkDown"},
		{"facing": &"left", "velocity": Vector2.LEFT, "action": &"WalkLeft"},
		{"facing": &"right", "velocity": Vector2.RIGHT, "action": &"WalkRight"},
	]
	for direction_case: Dictionary in direction_cases:
		var direction_velocity := (
			direction_case["velocity"] as Vector2
		) * actor.stats.move_speed
		actor.velocity = direction_velocity
		actor.call("_update_facing", direction_velocity)
		var sampled_frames: Array[int] = []
		var planted_feet: Array[StringName] = []
		var body_rolls: Array[float] = []
		for sample_index: int in range(8):
			actor.position += direction_velocity.normalized() * 10.0
			actor_view.call("_process", 0.13)
			var pose: Dictionary = actor_view.get_walk_pose_snapshot()
			sampled_frames.append(int(pose["frame"]))
			planted_feet.append(pose["planted_foot"] as StringName)
			body_rolls.append(float(pose["body_roll_degrees"]))
		_check(
			actor_view.get_current_action() == direction_case["action"] \
				and actor_view.get_current_facing() == direction_case["facing"],
			"%s movement selects its independent action and artwork"
				% direction_case["facing"]
		)
		_check(
			0 in sampled_frames and 1 in sampled_frames \
				and 2 in sampled_frames and 3 in sampled_frames,
			"%s walk cycles through all four 8 FPS frames"
				% direction_case["facing"]
		)
		_check(
			&"left" in planted_feet and &"right" in planted_feet,
			"%s walk alternates left and right foot contact"
				% direction_case["facing"]
		)
		_check(
			_has_positive_and_negative(body_rolls, 5.5),
			"%s walk shifts body weight by about six degrees"
				% direction_case["facing"]
		)
	_check(
		actor_view.position.is_equal_approx(
			GameConstants.logic_to_world_3d(actor.position, 0.04)
		),
		"Sprite3D actor follows authoritative 2D movement"
	)
	actor.velocity = Vector2.ZERO
	actor_view.call("_process", 0.13)
	_check(
		actor_view.get_current_action() == &"Idle" \
			and actor_view.get_current_facing() == &"right",
		"stopping returns to same-direction Idle within one frame"
	)
	actor.stats.is_trapped = true
	actor_view.call("_process", 0.13)
	_check(
		actor_view.get_current_action() == &"Trapped",
		"front-facing Trapped immediately overrides movement"
	)
	actor.stats.is_trapped = false
	actor.stats.is_dead = true
	actor_view.call("_process", 0.13)
	_check(
		actor_view.get_current_action() == &"Idle",
		"death holds Idle and never requests a Defeat action"
	)
	var preview := CharacterPreview3D.new()
	root.add_child(preview)
	preview.setup(CharacterCatalog.get_definition("cat"), "red")
	await process_frame
	preview.set_color_id("cyan")
	var preview_sprite := preview.find_child(
		"ImageGenPreviewSprite",
		true,
		false
	) as Sprite3D
	_check(preview_sprite != null, "lobby preview reuses the production Sprite3D")
	_check(preview.color_id == "cyan", "preview updates its selected team color")
	preview.queue_free()
	actor_view.queue_free()
	actor.queue_free()
	board.queue_free()
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
	_check(is_instance_valid(floor), "watercolor floor is rendered as a MultiMesh")
	_check(
		is_instance_valid(floor) and floor.multimesh.instance_count == 195,
		"floor contains exactly 195 tile instances"
	)
	_check(
		floor.multimesh.use_colors and floor.multimesh.use_custom_data,
		"floor instances carry paint colors and six atlas variants"
	)
	_check(
		floor.material_override is ShaderMaterial,
		"floor atlas selection uses a Compatibility-safe shader"
	)
	_check(
		view.find_child("ConiferSprite0", true, false) is Sprite3D,
		"tall ImageGen forest sprites stay outside the board"
	)
	var first_wind_pose: Dictionary = view.get_wind_pose_snapshot()
	view.set_wind_strength(1.0)
	for sample: int in range(8):
		view.call("_process", 0.21)
	var next_wind_pose: Dictionary = view.get_wind_pose_snapshot()
	_check(
		int(next_wind_pose["plant_count"]) == 10,
		"every tree, bush, mushroom and flower exposes wind frames"
	)
	_check(
		first_wind_pose["frame_indices"] != next_wind_pose["frame_indices"],
		"forest plants advance through phase-offset redrawn wind frames"
	)
	var phase_offsets := (
		next_wind_pose["phase_offsets_seconds"] as Array[float]
	)
	var unique_phase_offsets: Dictionary = {}
	for phase_offset: float in phase_offsets:
		unique_phase_offsets[phase_offset] = true
	_check(
		unique_phase_offsets.size() == int(next_wind_pose["plant_count"]),
		"every animated plant starts at a distinct wind phase"
	)
	_check(
		float(next_wind_pose["animation_rate"]) <= 6.4 / 3.0 + 0.001,
		"plant wind frame rate is exactly one third of the previous maximum"
	)
	_check(
		_float_arrays_close(
			next_wind_pose["angles_degrees"] as Array[float],
			[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		),
		"wind animation no longer rotates rigid plant cutouts"
	)
	var target := Vector2i(5, 5)
	board.paint_cells([target], PaintPalette.TEAM_PLAYER)
	var index: int = target.y * GameConstants.GRID_COLUMNS + target.x
	_check(
		_colors_close(view.displayed_color(target), PaintPalette.get_color("orange")),
		"paint signal incrementally updates one tile color"
	)
	board.lock_neighborhood(target, PaintPalette.TEAM_PLAYER)
	var lock_transform := locks.multimesh.get_instance_transform(index)
	_check(
		lock_transform.basis.get_scale().length() > 1.0,
		"locked tile exposes a full-cell hand-painted shadow"
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
		_colors_close(view.displayed_color(target), PaintPalette.get_color("orange")),
		"locked visual remains the earliest owner color"
	)
	view.queue_free()
	board.queue_free()
	await process_frame


func _test_weather_and_ripples() -> void:
	var weather := ForestWeatherSystem3D.new()
	root.add_child(weather)
	await process_frame
	_check(
		weather.get_weather_state() == ForestWeatherSystem3D.Weather.RAIN \
			and weather.get_rain_intensity() >= 0.99,
		"arena weather starts with an immediately visible rain phase"
	)
	var streaks := weather.find_child(
		"WatercolorRainStreaks",
		true,
		false
	) as MultiMeshInstance3D
	var ripples := weather.find_child(
		"BoardRainRipples",
		true,
		false
	) as MultiMeshInstance3D
	_check(
		streaks != null and streaks.multimesh.instance_count == 84,
		"rain uses one Compatibility-safe batched streak field"
	)
	_check(
		ripples != null and ripples.multimesh.instance_count == 36,
		"tile impacts reuse a bounded ripple pool"
	)
	weather.debug_spawn_ripple(Vector2i(7, 6))
	weather.call("_process", 0.12)
	_check(
		weather.get_active_ripple_count() >= 1,
		"rain impact creates an expanding watercolor ripple on a board cell"
	)
	weather.set_weather(ForestWeatherSystem3D.Weather.CLEAR, true)
	_check(
		weather.get_weather_name() == "clear" \
			and is_zero_approx(weather.get_rain_intensity()),
		"weather can transition to a dry clear phase without gameplay state"
	)
	weather.set_weather(ForestWeatherSystem3D.Weather.RAIN, true)
	_check(
		weather.get_weather_name() == "rain",
		"weather can deterministically return to rain for snapshots"
	)
	weather.queue_free()
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
	actor.character_id = "fox"
	var bubble := GameBubble.new()
	root.add_child(bubble)
	bubble.setup(actor, Vector2i(7, 6), 10.0, [actor])
	var bubble_view := BubbleView3D.new()
	root.add_child(bubble_view)
	bubble_view.bind_bubble(bubble)
	var bubble_sprite := bubble_view.find_child(
		"FoxImageGenBubbleSprite",
		true,
		false
	) as Sprite3D
	_check(
		bubble_sprite != null and bubble_view.get_skin_character_id() == "fox",
		"bubble selects the owner's exclusive ImageGen fox skin"
	)
	_check(
		bubble_view.get_visual_cell_width() >= 0.8,
		"bubble visibly occupies at least four fifths of its board cell"
	)
	var bubble_material := bubble_sprite.material_override as ShaderMaterial
	_check(
		bubble_material != null and (
			bubble_material.get_shader_parameter("team_color") as Color
		).is_equal_approx(PaintPalette.get_color("green")),
		"bubble fill inherits the owner's green team color"
	)
	var effect := ExplosionEffect.new()
	root.add_child(effect)
	effect.setup([Vector2i(7, 6), Vector2i(8, 6)], Vector2i(7, 6), actor)
	var explosion_view := ExplosionView3D.new()
	root.add_child(explosion_view)
	explosion_view.activate(effect, 2)
	await process_frame
	var splash := explosion_view.find_child(
		"WatercolorSplashCells",
		true,
		false
	) as MultiMeshInstance3D
	_check(
		splash != null and splash.multimesh.instance_count == 2,
		"explosion keeps the authoritative logical cross cells"
	)
	var splash_material := splash.material_override as ShaderMaterial
	_check(
		splash_material != null and (
			splash_material.get_shader_parameter("team_color") as Color
		).is_equal_approx(PaintPalette.get_color("green")),
		"foam burst inherits the owner's green team color"
	)
	explosion_view.call("_process", 0.09)
	_check(explosion_view.visual_stage() == 1, "explosion keeps staged animation")
	var arena := ArenaView3D.new()
	root.add_child(arena)
	await process_frame
	_check(
		arena.find_child("RoofOcclusionController", true, false) == null,
		"flat forest arena contains no obsolete roof occlusion"
	)
	arena.play_defeat_burst(Vector2i(7, 6), "purple", [Vector2i(7, 6)])
	_check(
		(arena.explosion_root as StorybookExplosionPool).get_stats()["in_use"] >= 1,
		"defeat burst reuses the pooled Sprite3D explosion visuals"
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
		var texture := load(ArenaItemType.texture_path(item_type)) as Texture2D
		_check(
			texture != null,
			"%s ImageGen pickup texture imports" % ArenaItemType.display_name(item_type)
		)
		var view := ItemView3D.new()
		root_3d.add_child(view)
		view.setup(ArenaItemState.new(item_type + 1, item_type, Vector2i(item_type, 1)))
		_check(
			view.find_child("ImageGenPickupSprite", true, false) is Sprite3D \
				and view.find_child("PickupRing", true, false) != null,
			"%s pickup uses a Sprite3D and identification ring"
				% ArenaItemType.display_name(item_type)
		)
		_check(
			view.get_visual_cell_width() >= 0.8,
			"%s pickup occupies at least four fifths of one board cell"
				% ArenaItemType.display_name(item_type)
		)
	var arena := ArenaView3D.new()
	root.add_child(arena)
	await process_frame
	arena.set_camera_pose(999.0, -999.0, 9.0)
	_check(
		arena.get_azimuth() == 0.0 \
			and arena.get_elevation() == 54.0 \
			and arena.get_zoom() == MatchSettings.MAX_CAMERA_ZOOM,
		"runtime fixes the front-horizontal 0/54 camera and clamps zoom to 135%"
	)
	arena.set_camera_pose(12.0, 52.0, 1.32)
	arena.reset_zoom()
	_check(
		arena.get_azimuth() == 0.0 \
			and arena.get_elevation() == 54.0 \
			and arena.get_zoom() == MatchSettings.DEFAULT_CAMERA_ZOOM,
		"zoom reset preserves the fixed 0/54 camera"
	)
	var adjustment_finished: Array[bool] = []
	arena.camera_adjustment_finished.connect(
		func(_azimuth: float, _elevation: float, _zoom: float) -> void:
			adjustment_finished.append(true)
	)
	arena.zoom_in()
	_check(
		arena.get_azimuth() == 0.0 \
			and arena.get_elevation() == 54.0 \
			and not arena.has_method("_input"),
		"fixed camera removes the right-drag orbit handler"
	)
	_check(adjustment_finished.size() == 1, "zoom changes emit persistence once")
	arena.queue_free()
	root_3d.queue_free()
	await process_frame


func _has_positive_and_negative(values: Array[float], magnitude: float) -> bool:
	var has_positive := false
	var has_negative := false
	for value: float in values:
		has_positive = has_positive or value >= magnitude
		has_negative = has_negative or value <= -magnitude
	return has_positive and has_negative


func _minimum(values: Array[float]) -> float:
	var result := INF
	for value: float in values:
		result = minf(result, value)
	return result


func _maximum(values: Array[float]) -> float:
	var result := -INF
	for value: float in values:
		result = maxf(result, value)
	return result


func _colors_close(left: Color, right: Color) -> bool:
	return Vector3(left.r - right.r, left.g - right.g, left.b - right.b).length() < 0.025


func _float_arrays_close(left: Array[float], right: Array[float]) -> bool:
	if left.size() != right.size():
		return false
	for index: int in range(left.size()):
		if not is_equal_approx(left[index], right[index]):
			return false
	return true


func _check(condition: bool, description: String) -> void:
	_checks += 1
	if condition:
		return
	_failures += 1
	push_error("SPRITE3D VISUAL TEST FAILED: %s" % description)
