extends SceneTree

var _checks: int = 0
var _failures: int = 0


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	_test_coordinate_mapping()
	_test_catalog_and_animation_aliases()
	_test_building_catalog()
	_test_settings_migration_and_selection()
	_test_ai_assignments()
	_test_clay_material_constraints()
	await _test_visual_follow_and_maps()
	await _test_camera_occlusion_and_effects()
	print("BnBonline character/3D tests: %d checks, %d failures" % [_checks, _failures])
	quit(1 if _failures > 0 else 0)


func _test_coordinate_mapping() -> void:
	_check(
		GameConstants.grid_to_world_3d(Vector2i.ZERO).is_equal_approx(Vector3(-7.0, 0.0, -6.0)),
		"top-left grid cell maps to the centered XZ arena"
	)
	_check(
		GameConstants.grid_to_world_3d(Vector2i(7, 6), 0.5).is_equal_approx(Vector3(0.0, 0.5, 0.0)),
		"center grid cell maps to the world origin"
	)
	var logic_position := GameConstants.grid_to_world(Vector2i(11, 9)) + Vector2(10.0, -5.0)
	_check(
		GameConstants.logic_to_world_3d(logic_position, 0.25).is_equal_approx(Vector3(4.25, 0.25, 2.875)),
		"continuous 2D positions preserve sub-cell offsets in 3D"
	)


func _test_catalog_and_animation_aliases() -> void:
	var definitions := CharacterCatalog.get_all()
	_check(definitions.size() == 8, "catalog contains the eight clay character definitions")
	var ids: Dictionary = {}
	for definition in definitions:
		ids[definition.id] = true
		var scene := definition.load_model_scene()
		_check(scene != null, "%s GLB is available offline" % definition.id)
		if scene == null:
			continue
		var instance := scene.instantiate()
		var animation_names := _collect_animation_names(instance)
		_check(
			_alias_resolves(animation_names, definition.idle_animation_aliases),
			"%s resolves an idle animation alias" % definition.id
		)
		_check(
			_alias_resolves(animation_names, definition.move_animation_aliases),
			"%s resolves a movement animation alias" % definition.id
		)
		instance.free()
	_check(
		ids.keys() == [
			"builder", "chef", "cowboy", "wizard",
			"ninja", "medic", "viking", "fighter",
		],
		"catalog exposes the stable character IDs in selection order"
	)
	_check(ids.size() == 8, "all character IDs are unique")


func _test_settings_migration_and_selection() -> void:
	var migrated := MatchSettings.new()
	migrated.apply_dictionary({
		"map_id": "windmill-heart",
		"ai_count": 2,
		"bubble_skin": "basketball",
		"zodiac_id": "dragon",
	})
	_check(migrated.character_id == "wizard", "v2 dragon selection migrates to the wizard")
	_check(migrated.map_id == MapCatalog.BELL_GARDEN, "legacy heart map migrates to bell garden")
	_check(migrated.bubble_skin == "coral", "legacy basketball skin migrates to coral")
	var legacy_default := MatchSettings.new()
	legacy_default.apply_dictionary({
		"map_id": "classic",
		"bubble_skin": "football",
		"zodiac_id": "rat",
	})
	_check(legacy_default.character_id == "builder", "v2 rat selection migrates to the builder")
	_check(legacy_default.map_id == MapCatalog.HARBOR_MARKET, "legacy classic map migrates to harbor market")
	_check(legacy_default.bubble_skin == "aqua", "legacy football skin migrates to aqua")
	var selected := MatchSettings.new()
	selected.apply_dictionary({"character_id": "wizard", "max_speed": 425})
	var restored := MatchSettings.new()
	restored.apply_dictionary(selected.to_dictionary())
	_check(restored.character_id == "wizard", "player character survives settings serialization")
	_check(restored.max_speed == 425, "existing numeric settings remain compatible")
	restored.apply_dictionary({"character_id": "not-a-character"})
	_check(restored.character_id == "builder", "invalid character IDs normalize safely")


func _test_building_catalog() -> void:
	for building_id: String in [
		"cottage_red", "shop_blue", "cottage_mustard", "cottage_green",
		"bell_tower", "clinic", "hedge",
	]:
		var definition := BuildingCatalog.get_definition(building_id)
		_check(definition.id == building_id, "%s has a stable building definition" % building_id)
		for model_path: String in definition.module_paths:
			_check(ResourceLoader.exists(model_path), "%s module is available offline" % model_path.get_file())
	for nature_id: String in [
		"bush_flowers", "clover", "common_tree", "flower_group", "round_rock", "twisted_tree",
	]:
		_check(
			ResourceLoader.exists(BuildingCatalog.nature_path(nature_id)),
			"%s nature module is available offline" % nature_id
		)


func _test_ai_assignments() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 20260725
	var assigned := CharacterCatalog.assign_ai_characters("ninja", 4, rng)
	var unique: Dictionary = {}
	for character_id in assigned:
		unique[character_id] = true
	_check(assigned.size() == 4, "four AI character IDs are assigned")
	_check(unique.size() == assigned.size(), "AI character IDs do not repeat")
	_check("ninja" not in assigned, "AI assignments exclude the player character")
	_check(CharacterCatalog.assign_ai_characters("builder", 0, rng).is_empty(), "zero AI needs no character")


func _test_clay_material_constraints() -> void:
	var normal := Image.load_from_file(
		ProjectSettings.globalize_path("res://assets/materials/clay_detail_normal.png")
	)
	var roughness := Image.load_from_file(
		ProjectSettings.globalize_path("res://assets/materials/clay_roughness.png")
	)
	_check(normal.get_size() == Vector2i(256, 256), "shared clay normal texture is 256x256")
	_check(roughness.get_size() == Vector2i(256, 256), "shared roughness texture is 256x256")
	var material := ClayMaterialLibrary.make(Color("#d9664c"))
	_check(material.roughness >= 0.86 and material.roughness <= 0.94, "clay material uses matte roughness")
	_check(material.metallic == 0.0, "clay material is non-metallic")
	_check(material.metallic_specular <= 0.16, "clay material keeps specular response low")
	_check(material.normal_enabled and material.normal_texture != null, "clay material uses shared micro-normal detail")
	var interface_font := load("res://assets/fonts/NotoSansSC-BnB-Subset.ttf") as Font
	var zoom_glyphs_complete := is_instance_valid(interface_font)
	if is_instance_valid(interface_font):
		for character: String in "滚轮缩放复位":
			zoom_glyphs_complete = zoom_glyphs_complete and interface_font.has_char(character.unicode_at(0))
		zoom_glyphs_complete = zoom_glyphs_complete \
			and interface_font.has_char(0x20) \
			and interface_font.has_char(0x3000)
	_check(zoom_glyphs_complete, "Web font subset contains zoom labels and both UI space glyphs")


func _test_visual_follow_and_maps() -> void:
	var logic_root := Node2D.new()
	root.add_child(logic_root)
	var board := GameBoard.new()
	logic_root.add_child(board)
	board.reset(MapCatalog.get_map(MapCatalog.HARBOR_MARKET))
	var settings := MatchSettings.new()
	var actor := GameActor.new()
	logic_root.add_child(actor)
	actor.setup("测试", 1, true, board, settings, Vector2i(3, 4))
	actor.is_player = false
	actor.character_id = "fighter"

	var view_root := Node3D.new()
	root.add_child(view_root)
	var actor_view := ActorView3D.new()
	view_root.add_child(actor_view)
	actor_view.bind_actor(actor, actor.character_id)
	await process_frame
	actor.position += Vector2(13.0, 7.0)
	actor.velocity = Vector2.RIGHT * actor.stats.move_speed
	actor_view.call("_process", 0.016)
	_check(
		actor_view.position.is_equal_approx(GameConstants.logic_to_world_3d(actor.position, 0.04)),
		"3D character follows continuous 2D logic position"
	)
	var right_yaw := actor_view.rotation.y
	actor.velocity = Vector2.UP * actor.stats.move_speed
	actor_view.call("_process", 0.016)
	_check(not is_equal_approx(right_yaw, actor_view.rotation.y), "four-direction movement changes 3D facing")
	var pose_time_before := float(actor_view.get("_stop_motion_time"))
	var walk_phase_before := actor_view.get_walk_cycle_phase()
	actor.position += Vector2(5.0, 0.0)
	actor_view.call("_process", 0.04)
	_check(
		actor_view.position.is_equal_approx(GameConstants.logic_to_world_3d(actor.position, 0.04)),
		"character world position remains continuous between pose ticks"
	)
	_check(
		is_equal_approx(float(actor_view.get("_stop_motion_time")), pose_time_before),
		"skeleton pose holds before the next 12 FPS tick"
	)
	actor_view.call("_process", 0.05)
	_check(
		float(actor_view.get("_stop_motion_time")) > pose_time_before,
		"skeleton pose advances on the 12 FPS visual clock"
	)
	_check(
		actor_view.get_walk_cycle_phase() > walk_phase_before,
		"held movement advances the alternating walk cycle from actual travel"
	)
	var previous_walk_phase := actor_view.get_walk_cycle_phase()
	var sampled_animation_positions: Dictionary = {}
	var animation_player := actor_view.get("_animation_player") as AnimationPlayer
	for step: int in range(6):
		actor.position += Vector2(12.0 + step, 0.0)
		actor.velocity = Vector2.RIGHT * actor.stats.move_speed
		actor_view.call("_process", ActorView3D.STOP_MOTION_STEP)
		_check(
			not is_equal_approx(actor_view.get_walk_cycle_phase(), previous_walk_phase),
			"continuous rightward movement does not freeze on pose %d" % step
		)
		if is_instance_valid(animation_player):
			sampled_animation_positions[
				snappedf(animation_player.current_animation_position, 0.001)
			] = true
		previous_walk_phase = actor_view.get_walk_cycle_phase()
	_check(
		sampled_animation_positions.size() >= 4,
		"held movement samples multiple visible rig poses instead of sliding on one frame"
	)
	actor.velocity = Vector2.LEFT * actor.stats.move_speed
	actor.position += Vector2(-12.0, 0.0)
	actor_view.call("_process", ActorView3D.STOP_MOTION_STEP)
	_check(
		not is_equal_approx(actor_view.get_walk_cycle_phase(), previous_walk_phase),
		"leftward movement uses the same alternating-foot loop"
	)
	actor.stats.is_trapped = true
	actor.velocity = Vector2.ZERO
	actor_view.call("_process", 0.016)
	var trap_sphere := actor_view.get("_trap_sphere") as MeshInstance3D
	_check(is_instance_valid(trap_sphere) and trap_sphere.visible, "trapped actor shows the outer 3D bubble")
	actor.stats.is_trapped = false
	actor.stats.is_dead = true
	actor_view.call("_process", 0.016)
	var visual_pivot := actor_view.get("_visual_pivot") as Node3D
	_check(
		is_instance_valid(visual_pivot) and visual_pivot.scale.y < 0.5,
		"dead actor uses the shared clay squash animation"
	)
	_check(
		is_instance_valid(visual_pivot) and absf(visual_pivot.rotation.z) > 1.0,
		"dead actor visibly tips over without model-specific clips"
	)
	actor.stats.is_dead = false

	var board_view := BoardView3D.new()
	view_root.add_child(board_view)
	board_view.bind_board(board)
	await process_frame
	_check(
		not board_view.find_children("*", "MultiMeshInstance3D", true, false).is_empty(),
		"harbor market uses batched 3D geometry"
	)
	_check(
		board_view.get_building_views().size() == board.map_data.building_units.size(),
		"every harbor rigid footprint has a 3D building wrapper"
	)
	board.reset(MapCatalog.get_map(MapCatalog.BELL_GARDEN))
	await process_frame
	var has_bell_tower := false
	for building: ClayBuildingView3D in board_view.get_building_views():
		if building.definition.id == "bell_tower":
			has_bell_tower = true
			break
	_check(
		has_bell_tower,
		"bell garden builds its three-cell clock-tower landmark"
	)
	_check(
		not board_view.find_children("GardenFountain", "Node3D", true, false).is_empty(),
		"bell garden builds its clay fountain decoration"
	)

	logic_root.queue_free()
	view_root.queue_free()
	await process_frame


func _test_camera_occlusion_and_effects() -> void:
	var logic_root := Node2D.new()
	root.add_child(logic_root)
	var board := GameBoard.new()
	logic_root.add_child(board)
	board.reset(MapCatalog.get_map(MapCatalog.BELL_GARDEN))
	var settings := MatchSettings.new()
	settings.bubble_skin = "coral"
	var actor := GameActor.new()
	logic_root.add_child(actor)
	actor.setup("特效测试", 1, true, board, settings, board.map_data.player_spawn)
	actor.is_player = false

	var arena := ArenaView3D.new()
	root.add_child(arena)
	await process_frame
	arena.bind_board(board)
	await process_frame
	var camera_direction := (Vector3(0.0, 0.7, 0.0) - arena.camera.position).normalized()
	var elevation := rad_to_deg(asin(-camera_direction.y))
	var azimuth := rad_to_deg(atan2(-camera_direction.x, -camera_direction.z))
	_check(absf(elevation - ArenaView3D.CAMERA_ELEVATION_DEGREES) < 0.1, "camera uses the approved 38 degree elevation")
	_check(absf(azimuth - ArenaView3D.CAMERA_AZIMUTH_DEGREES) < 0.1, "camera uses the fixed five-degree horizontal skew")
	var camera_right := arena.camera.global_transform.basis.x.normalized()
	var camera_up := arena.camera.global_transform.basis.y.normalized()
	var screen_up := Vector2(
		Vector3(0.0, 0.0, -1.0).dot(camera_right),
		-Vector3(0.0, 0.0, -1.0).dot(camera_up)
	)
	var screen_right := Vector2(
		Vector3(1.0, 0.0, 0.0).dot(camera_right),
		-Vector3(1.0, 0.0, 0.0).dot(camera_up)
	)
	var screen_down := -screen_up
	var screen_left := -screen_right
	_check(screen_up.x < 0.0 and screen_up.y < 0.0, "up input stays in the upper-left screen quadrant")
	_check(screen_right.x > 0.0 and screen_right.y < 0.0, "right input stays in the upper-right screen quadrant")
	_check(screen_down.x > 0.0 and screen_down.y > 0.0, "down input stays in the lower-right screen quadrant")
	_check(screen_left.x < 0.0 and screen_left.y > 0.0, "left input stays in the lower-left screen quadrant")
	_check(
		absf(rad_to_deg(screen_up.angle_to(Vector2.UP))) <= 9.0 \
			and absf(rad_to_deg(screen_down.angle_to(Vector2.DOWN))) <= 9.0,
		"up and down inputs stay within nine degrees of the screen vertical"
	)
	_check(
		absf(rad_to_deg(screen_right.angle_to(Vector2.RIGHT))) <= 4.0 \
			and absf(rad_to_deg(screen_left.angle_to(Vector2.LEFT))) <= 4.0,
		"left and right inputs stay within four degrees of the screen horizontal"
	)
	for map_id: String in [MapCatalog.HARBOR_MARKET, MapCatalog.BELL_GARDEN]:
		var camera_map_data := MapCatalog.get_map(map_id)
		for viewport_size: Vector2i in [Vector2i(1280, 720), Vector2i(1040, 600)]:
			arena.fit_camera(camera_map_data, Vector2(viewport_size))
			_check(
				_camera_contains_bounds(arena.camera, camera_map_data.camera_bounds, viewport_size),
				"camera contains %s at %s" % [map_id, viewport_size]
			)
			var resized_camera_direction := (
				Vector3(0.0, 0.7, 0.0) - arena.camera.position
			).normalized()
			var resized_azimuth := rad_to_deg(atan2(
				-resized_camera_direction.x,
				-resized_camera_direction.z
			))
			_check(
				absf(resized_azimuth - ArenaView3D.CAMERA_AZIMUTH_DEGREES) < 0.1,
				"viewport fitting preserves the fixed camera direction"
			)
	var default_camera_size := arena.camera.size
	_check(arena.get_zoom_percent() == 110, "arena defaults to a slightly larger 110 percent map view")
	arena.zoom_in()
	_check(arena.camera.size < default_camera_size, "zoom-in reduces orthographic camera size")
	_check(arena.get_zoom_percent() == 120, "zoom-in reports the updated percentage")
	arena.zoom_out()
	_check(is_equal_approx(arena.camera.size, default_camera_size), "zoom-out returns to the previous framing")
	arena.set_zoom(ArenaView3D.MAX_ZOOM + 1.0)
	_check(is_equal_approx(arena.get_zoom(), ArenaView3D.MAX_ZOOM), "camera zoom clamps to the safe maximum")
	arena.reset_zoom()
	_check(arena.get_zoom_percent() == 110, "camera zoom reset restores the recommended framing")

	var building_views := arena.board_view.get_building_views()
	var occludable: ClayBuildingView3D
	for building: ClayBuildingView3D in building_views:
		if building.placement.occludable:
			occludable = building
			break
	_check(is_instance_valid(occludable), "bell garden exposes occludable roof wrappers")
	if is_instance_valid(occludable):
		var proxy := occludable.find_child("RoofOcclusionProxy", true, false) as StaticBody3D
		_check(
			is_instance_valid(proxy) and proxy.collision_layer == ClayBuildingView3D.VISUAL_OCCLUDER_LAYER,
			"roof ray proxies use the isolated visual-only collision layer"
		)
		await physics_frame
		var ray_hits: Dictionary = {}
		arena.roof_occlusion_controller.call(
			"_collect_ray_hits",
			arena.camera.global_position,
			occludable.global_position + Vector3(0.0, 1.28, 0.0),
			ray_hits
		)
		_check(ray_hits.has(occludable), "visual-only camera ray resolves the occluding building wrapper")
		occludable.set_occluded(true)
		occludable.call("_process", ClayBuildingView3D.FADE_DURATION)
		_check(
			absf(float(occludable.get("_visible_ratio")) - ClayBuildingView3D.OCCLUDED_VISIBLE_RATIO) < 0.01,
			"occluded roof reaches 25 percent Bayer visibility in 0.18 seconds"
		)
		occludable.set_occluded(false)
		occludable.call("_process", ClayBuildingView3D.RESTORE_DELAY + ClayBuildingView3D.FADE_DURATION)
		_check(not occludable.is_occluded(), "roof restores after the 0.3 second leave delay")

	var bubble := GameBubble.new()
	logic_root.add_child(bubble)
	bubble.setup(actor, Vector2i(2, 10), "basketball", 2.0, [actor])
	var bubble_view := arena.add_bubble(bubble)
	await process_frame
	_check(bubble_view.get("_skin_color") == BubbleView3D.CORAL, "legacy basketball bubble renders with coral clay")
	_check(bubble_view.find_children("PressureDimple*", "MeshInstance3D", true, false).size() == 3, "clay bubble has three pressure dimples")
	bubble_view.set("_last_progress", 0.7)
	bubble_view.call("_apply_stop_motion_pose")
	_check(bubble_view.fuse_stage() == 2, "bubble countdown exposes the third 12 FPS pressure stage")

	var effect := ExplosionEffect.new()
	logic_root.add_child(effect)
	effect.setup(
		[
			Vector2i(7, 6), Vector2i(6, 6), Vector2i(5, 6),
			Vector2i(8, 6), Vector2i(7, 5), Vector2i(7, 7),
		],
		Vector2i(7, 6),
		actor
	)
	var explosion_view := arena.add_explosion(effect)
	_check(explosion_view.visual_stage() == 0, "explosion begins with the compressed cream core")
	explosion_view.call("_process", 0.09)
	_check(explosion_view.visual_stage() == 1, "explosion advances to clay water columns after 0.083 seconds")
	var splash_stage := explosion_view.find_child("ClayWaterSplashCells", true, false) as MultiMeshInstance3D
	var radial_stage := explosion_view.find_child("HandPinchedRadialSplash", true, false) as MultiMeshInstance3D
	_check(
		is_instance_valid(splash_stage) and splash_stage.visible \
			and is_instance_valid(radial_stage) and radial_stage.visible,
		"water-column and hand-pinched center batches are visible during the splash stage"
	)
	explosion_view.call("_process", 0.17)
	_check(explosion_view.visual_stage() == 2, "explosion becomes dithered foam after 0.25 seconds")
	_check(
		is_instance_valid(splash_stage) and splash_stage.multimesh.instance_count == effect.cells.size(),
		"every dangerous grid cell stays fully represented in the water-column batch"
	)
	var pool := arena.explosion_root as ClayExplosionPool
	_check(pool.get_stats()["decorative_droplets"] <= ClayExplosionPool.MAX_DECORATIVE_DROPLETS, "pool enforces the 48-droplet global cap")
	effect.queue_free()
	await process_frame
	_check(pool.get_stats()["in_use"] == 0, "explosion view returns to the object pool after the 0.45 second logic effect")

	bubble.queue_free()
	actor.queue_free()
	arena.queue_free()
	logic_root.queue_free()
	await process_frame


func _camera_contains_bounds(camera: Camera3D, bounds: AABB, viewport_size: Vector2i) -> bool:
	var target := Vector3(0.0, 0.7, 0.0)
	var half_height := camera.size * 0.5 * 0.99
	var half_width := half_height * float(viewport_size.x) / float(viewport_size.y)
	var right := camera.global_transform.basis.x.normalized()
	var up := camera.global_transform.basis.y.normalized()
	for index: int in range(8):
		var relative := bounds.get_endpoint(index) - target
		if absf(relative.dot(right)) > half_width:
			return false
		if absf(relative.dot(up)) > half_height:
			return false
	return true


func _collect_animation_names(root_node: Node) -> Array[StringName]:
	var result: Array[StringName] = []
	if root_node is AnimationPlayer:
		result.append_array((root_node as AnimationPlayer).get_animation_list())
	for child in root_node.get_children():
		result.append_array(_collect_animation_names(child))
	return result


func _alias_resolves(names: Array[StringName], aliases: Array[String]) -> bool:
	for alias in aliases:
		for animation_name in names:
			var normalized_name := String(animation_name).to_lower()
			if normalized_name.ends_with(alias.to_lower()) or normalized_name.contains(alias.to_lower()):
				return true
	return false


func _check(condition: bool, description: String) -> void:
	_checks += 1
	if condition:
		return
	_failures += 1
	push_error("CHARACTER/CLAY TEST FAILED: %s" % description)
