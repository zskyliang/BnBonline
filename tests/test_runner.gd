extends SceneTree
## Headless rules and progression checks for the paint campaign.

var _failures: int = 0
var _checks: int = 0


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	_test_palette_and_settings()
	_test_flat_map()
	_test_blast_and_half_body_rules()
	_test_board_paint_and_locks()
	_test_items_and_temporary_stats()
	_test_run_progression()
	_test_ai_snapshot_paint_value()
	_test_ai_hazard_forecast()
	await _test_actor_and_bubble_rules()
	print("BnBonline paint rules: %d checks, %d failures" % [_checks, _failures])
	quit(1 if _failures > 0 else 0)


func _test_palette_and_settings() -> void:
	_check(PaintPalette.COLOR_IDS.size() == 7, "palette exposes exactly seven colors")
	var unique_colors: Dictionary = {}
	for color_id: String in PaintPalette.COLOR_IDS:
		_check(PaintPalette.is_valid_color_id(color_id), "%s is a valid paint color" % color_id)
		unique_colors[PaintPalette.get_color(color_id).to_html()] = true
	_check(unique_colors.size() == 7, "all seven paint colors are visually distinct")
	_check(
		MatchSettings.WEB_STORAGE_KEY == "bnb.settings.v6" \
			and "bnb.settings.v5" in MatchSettings.LEGACY_WEB_STORAGE_KEYS \
			and "bnb.settings.v4" in MatchSettings.LEGACY_WEB_STORAGE_KEYS \
			and "bnb.settings.v3" in MatchSettings.LEGACY_WEB_STORAGE_KEYS \
			and "bnb.settings.v2" in MatchSettings.LEGACY_WEB_STORAGE_KEYS,
		"appearance and zoom settings migrate from v5 through v2 into v6"
	)
	var legacy_settings := MatchSettings.new()
	legacy_settings.apply_dictionary({"character_id": "ninja"})
	_check(
		legacy_settings.camera_azimuth == MatchSettings.DEFAULT_CAMERA_AZIMUTH \
			and legacy_settings.camera_elevation == MatchSettings.DEFAULT_CAMERA_ELEVATION \
			and legacy_settings.camera_zoom == MatchSettings.DEFAULT_CAMERA_ZOOM,
		"legacy settings without camera fields receive the current defaults"
	)
	var settings := MatchSettings.new()
	settings.apply_dictionary({
		"character_id": "wizard",
		"player_color_id": "purple",
		"map_id": "bell-garden",
		"bubble_skin": "coral",
		"camera_azimuth": 999.0,
		"camera_elevation": -999.0,
		"camera_zoom": 9.0,
	})
	_check(settings.character_id == "bear", "legacy role IDs migrate by original card order")
	_check(settings.player_color_id == "purple", "paint color survives settings application")
	settings.apply_dictionary({"character_id": "missing", "player_color_id": "missing"})
	_check(settings.character_id == "cat", "invalid character falls back to cat")
	_check(
		settings.player_color_id == PaintPalette.DEFAULT_PLAYER_COLOR_ID,
		"invalid paint color falls back"
	)
	_check(
		settings.camera_azimuth == MatchSettings.DEFAULT_CAMERA_AZIMUTH \
			and settings.camera_elevation == MatchSettings.DEFAULT_CAMERA_ELEVATION \
			and settings.camera_zoom == MatchSettings.MAX_CAMERA_ZOOM,
		"legacy angles are ignored while corrupt zoom is clamped"
	)
	_check(
		settings.to_dictionary().keys().size() == 3 \
			and not settings.to_dictionary().has("camera_azimuth") \
			and not settings.to_dictionary().has("camera_elevation"),
		"only character, team color, and zoom are persisted"
	)


func _test_flat_map() -> void:
	var map_data: MapData = MapCatalog.get_map()
	_check(map_data.map_id == MapCatalog.PAINT_ARENA, "single paint arena has the stable ID")
	_check(map_data.decorations.is_empty(), "paint arena has no decorations")
	_check(
		map_data.camera_bounds.size.x >= 17.0 and map_data.camera_bounds.size.z >= 15.0,
		"fixed camera bounds include the playable grid and exterior forest frame"
	)
	var open_count: int = 0
	for row: PackedInt32Array in map_data.barrier_cells:
		for code: int in row:
			if code == 0:
				open_count += 1
	_check(open_count == 195, "all 195 floor cells are open")
	_check(MapCatalog.get_options().size() == 1, "map selection has been retired")


func _test_blast_and_half_body_rules() -> void:
	var cells: Array[PackedInt32Array] = []
	for _y: int in range(GameConstants.GRID_ROWS):
		var row := PackedInt32Array()
		row.resize(GameConstants.GRID_COLUMNS)
		row.fill(0)
		cells.append(row)
	var blast: Array[Vector2i] = GameRules.blast_cells(Vector2i(7, 6), 2, cells)
	_check(blast.size() == 9, "power two paints center plus two cells in four directions")
	_check(Vector2i(9, 6) in blast, "open arena blast reaches full horizontal power")
	var edge_blast: Array[Vector2i] = GameRules.blast_cells(Vector2i.ZERO, 3, cells)
	_check(edge_blast.size() == 7, "edge blast clips to arena bounds")
	var unsafe: Dictionary = {Vector2i(2, 2): true}
	var boundary_position := Vector2(
		GameConstants.GRID_ORIGIN.x + GameConstants.CELL_SIZE * 3.0,
		GameConstants.grid_to_world(Vector2i(2, 2)).y
	)
	_check(
		not GameRules.both_feet_unsafe(boundary_position, unsafe),
		"one-foot coverage remains half-body safe"
	)
	unsafe[Vector2i(3, 2)] = true
	_check(
		GameRules.both_feet_unsafe(boundary_position, unsafe),
		"both covered feet remain unsafe"
	)
	_check(GameConstants.ROUND_SECONDS == 180.0, "each stage lasts exactly three minutes")


func _test_board_paint_and_locks() -> void:
	var board := GameBoard.new()
	root.add_child(board)
	board.configure_team_colors("red", "blue")
	board.reset(MapCatalog.get_map())
	var counts: Dictionary = board.get_territory_counts()
	_check(int(counts["neutral"]) == 195, "new stage starts with 195 neutral cells")
	var stripe: Array[Vector2i] = [
		Vector2i(4, 4),
		Vector2i(5, 4),
		Vector2i(6, 4),
	]
	var first_paint: Dictionary = board.paint_cells(stripe, PaintPalette.TEAM_PLAYER)
	_check(first_paint.size() == 3, "explosion paints every mutable target once")
	_check(board.paint_owner(Vector2i(5, 4)) == PaintPalette.TEAM_PLAYER, "player owns painted tile")
	_check(
		board.territory_swing(stripe, PaintPalette.TEAM_AI) == 6,
		"covering three opposing tiles produces six points of net swing"
	)
	board.paint_cells(stripe, PaintPalette.TEAM_AI)
	_check(board.paint_owner(Vector2i(5, 4)) == PaintPalette.TEAM_AI, "later paint overwrites mutable tile")
	var locked: Dictionary = board.lock_neighborhood(Vector2i.ZERO, PaintPalette.TEAM_PLAYER)
	_check(locked.size() == 4, "corner defeat clips the locked neighborhood to four cells")
	_check(board.is_locked(Vector2i.ZERO), "defeat tile is marked locked")
	board.paint_cells([Vector2i.ZERO], PaintPalette.TEAM_AI)
	_check(
		board.paint_owner(Vector2i.ZERO) == PaintPalette.TEAM_PLAYER,
		"locked ownership cannot be overwritten"
	)
	var second_lock: Dictionary = board.lock_neighborhood(Vector2i.ZERO, PaintPalette.TEAM_AI)
	_check(second_lock.is_empty(), "earliest lock wins every lock conflict")
	counts = board.get_territory_counts()
	_check(int(counts["player_locked"]) == 4, "locked cells count toward player territory")
	_check(int(counts["player"]) + int(counts["ai"]) + int(counts["neutral"]) == 195, "territory counts conserve all cells")
	board.queue_free()


func _test_items_and_temporary_stats() -> void:
	var board := GameBoard.new()
	root.add_child(board)
	board.reset(MapCatalog.get_map())
	var cell := Vector2i(5, 5)
	var item_id: int = board.spawn_item(ArenaItemType.Value.SPEED, cell, 10000)
	_check(item_id > 0, "board assigns a stable item id")
	_check(board.is_cell_walkable(cell), "item cells remain walkable")
	_check(not board.can_place_bubble(cell), "item cells temporarily reject bubble placement")
	board.paint_cells([cell], PaintPalette.TEAM_PLAYER)
	board.lock_neighborhood(cell, PaintPalette.TEAM_PLAYER)
	_check(board.item_at(cell) != null, "painting and locking do not destroy items")
	var taken: ArenaItemState = board.take_item(cell, 77)
	_check(
		taken != null and taken.item_id == item_id and taken.spawned_ms == 10000,
		"atomic pickup returns stable item state"
	)
	_check(board.take_item(cell, 88) == null, "a collected item cannot be taken twice")
	_check(board.can_place_bubble(cell), "bubble placement reopens after pickup")
	var stats := ActorStats.new()
	stats.apply_skill_points(2, 1, 3)
	_check(stats.move_speed == 170.0, "campaign allocation establishes the stage base")
	stats.apply_stage_item(ArenaItemType.Value.SPEED)
	stats.apply_stage_item(ArenaItemType.Value.SPEED)
	stats.apply_stage_item(ArenaItemType.Value.BUBBLE)
	stats.apply_stage_item(ArenaItemType.Value.POWER)
	_check(stats.move_speed == 220.0, "speed items stack by twenty-five without a cap")
	_check(stats.bubble_capacity == 4, "bubble item stacks on campaign capacity")
	_check(stats.power == 6, "power item stacks on campaign power")
	stats.clear_stage_item_bonuses()
	_check(
		stats.move_speed == 170.0 and stats.bubble_capacity == 3 and stats.power == 5,
		"stage cleanup restores campaign values without removing skill points"
	)
	_check(
		GameConstants.ITEM_SPAWN_INTERVAL_SECONDS == 10.0,
		"items use the required ten-second cadence"
	)
	board.queue_free()


func _test_run_progression() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 20260725
	var progress := RunProgress.new()
	progress.begin("fox", "cyan", rng)
	_check(progress.stage_number == 1, "campaign begins at stage one")
	_check(progress.ai_count() == 1, "stage one has one AI")
	_check(progress.ai_color_id != "cyan", "AI color differs from player color")
	_check(progress.ai_character_ids.size() == 4, "four stable AI identities are prepared")
	_check("fox" not in progress.ai_character_ids, "AI roster excludes player animal")
	var stats := ActorStats.new()
	progress.apply_allocation(stats, progress.player_allocation())
	_check(stats.move_speed == 150.0, "campaign uses base speed")
	_check(stats.bubble_capacity == 2, "campaign uses base bubble count")
	_check(stats.power == 2, "campaign uses base power")
	_check(progress.advance_with_skill(RunProgress.SKILL_SPEED, rng), "speed skill advances campaign")
	progress.apply_allocation(stats, progress.player_allocation())
	_check(stats.move_speed == 160.0, "speed point adds ten pixels per second")
	_check(progress.stage_number == 2 and progress.ai_count() == 2, "stage two adds the second AI")
	var stage_two_allocations: Array[Dictionary] = progress.ai_allocations.duplicate(true)
	_check(_allocation_total(progress.ai_allocation(0)) == 1, "each AI receives all earned points")
	_check(progress.ai_allocations == stage_two_allocations, "reading allocations does not reroll retry state")
	progress.advance_with_skill(RunProgress.SKILL_BUBBLE, rng)
	progress.advance_with_skill(RunProgress.SKILL_POWER, rng)
	_check(progress.stage_number == 4 and progress.ai_count() == 4, "stage four reaches four-AI cap")
	for allocation: Dictionary in progress.ai_allocations:
		_check(_allocation_total(allocation) == 3, "every AI independently owns three points")
	progress.advance_with_skill(RunProgress.SKILL_BUBBLE, rng)
	_check(progress.stage_number == 5 and progress.ai_count() == 4, "later stages remain capped at four AI")
	progress.apply_allocation(stats, progress.player_allocation())
	_check(stats.bubble_capacity == 4, "bubble points stack without the old cap")
	_check(stats.power == 3, "power point carries into later stages")
	var long_run := RunProgress.new()
	long_run.begin("cat", "red", rng)
	for _point: int in range(25):
		long_run.advance_with_skill(RunProgress.SKILL_SPEED, rng)
	long_run.apply_allocation(stats, long_run.player_allocation())
	_check(stats.move_speed == 400.0, "long campaigns keep stacking speed")


func _test_ai_snapshot_paint_value() -> void:
	var snapshot := AIBattleSnapshot.new()
	for _y: int in range(GameConstants.GRID_ROWS):
		var cell_row := PackedInt32Array()
		cell_row.resize(GameConstants.GRID_COLUMNS)
		cell_row.fill(0)
		snapshot.cells.append(cell_row)
		var owner_row := PackedInt32Array()
		owner_row.resize(GameConstants.GRID_COLUMNS)
		owner_row.fill(PaintPalette.TEAM_NEUTRAL)
		snapshot.paint_owners.append(owner_row)
		var lock_row := PackedByteArray()
		lock_row.resize(GameConstants.GRID_COLUMNS)
		lock_row.fill(0)
		snapshot.locked_cells.append(lock_row)
	var targets: Array[Vector2i] = [Vector2i(2, 2), Vector2i(3, 2), Vector2i(4, 2)]
	snapshot.paint_owners[2][3] = PaintPalette.TEAM_PLAYER
	snapshot.paint_owners[2][4] = PaintPalette.TEAM_AI
	_check(
		snapshot.territory_swing(targets, PaintPalette.TEAM_AI) == 3,
		"AI values neutral as one, opposing as two, and own as zero"
	)
	snapshot.locked_cells[2][3] = 1
	_check(snapshot.territory_swing(targets, PaintPalette.TEAM_AI) == 1, "AI excludes locked paint")


func _test_ai_hazard_forecast() -> void:
	var snapshot := AIBattleSnapshot.new()
	for _y: int in range(GameConstants.GRID_ROWS):
		var row := PackedInt32Array()
		row.resize(GameConstants.GRID_COLUMNS)
		row.fill(0)
		snapshot.cells.append(row)
	snapshot.bombs.append(
		AIBattleSnapshot.BombState.new(Vector2i(7, 6), 2, 1000, 1, 0)
	)
	var forecast: AIHazardForecast = AIHazardForecast.build(snapshot, 4000)
	_check(forecast.danger_eta_ms(Vector2i(7, 6)) == 1000, "forecast preserves real fuse")
	_check(forecast.danger_eta_ms(Vector2i(9, 6)) == 1000, "forecast covers open blast arm")
	_check(forecast.danger_eta_ms(Vector2i(10, 6)) > 1000, "forecast stops at configured power")


func _test_actor_and_bubble_rules() -> void:
	var board := GameBoard.new()
	root.add_child(board)
	board.reset(MapCatalog.get_map())
	var settings := MatchSettings.new()
	var player := GameActor.new()
	root.add_child(player)
	player.setup(
		"玩家",
		PaintPalette.TEAM_PLAYER,
		true,
		board,
		settings,
		Vector2i(1, 1),
		"red"
	)
	var enemy := GameActor.new()
	root.add_child(enemy)
	enemy.setup(
		"AI",
		PaintPalette.TEAM_AI,
		false,
		board,
		settings,
		Vector2i(2, 1),
		"blue"
	)
	var bubble := GameBubble.new()
	root.add_child(bubble)
	bubble.setup(player, Vector2i(1, 1), 10.0, [player])
	_check(bubble.owner_team == PaintPalette.TEAM_PLAYER, "bubble stores stable owner team")
	_check(bubble.color_id == "red", "bubble inherits owner color")
	_check(bubble.power == player.stats.power, "bubble snapshots owner power")
	player.register_unsafe_frame(enemy)
	_check(not player.stats.is_trapped, "first unsafe frame does not trap")
	player.register_unsafe_frame(enemy)
	_check(player.stats.is_trapped, "second unsafe frame traps")
	var death_result: Dictionary = {"team": PaintPalette.TEAM_NEUTRAL}
	player.died.connect(func(_victim: GameActor, team: int, _attacker: GameActor) -> void:
		death_result["team"] = team
	)
	player.finish_by_touch(enemy)
	_check(player.stats.is_dead, "opponent contact defeats trapped actor")
	_check(
		int(death_result["team"]) == PaintPalette.TEAM_AI,
		"defeat signal preserves opposing team attribution"
	)
	enemy.trap(enemy)
	enemy.finish_by_touch(enemy)
	_check(not enemy.stats.is_dead, "same-team/self touch cannot finish a trapped actor")
	enemy.call("_on_trap_timeout")
	_check(
		not enemy.stats.is_dead and not enemy.stats.is_trapped,
		"trap timeout releases the actor without counting as a defeat"
	)
	enemy.stats.apply_stage_item(ArenaItemType.Value.SPEED)
	enemy.respawn(Vector2i(3, 1))
	_check(
		enemy.stats.stage_speed_items == 1 and enemy.stats.move_speed == 175.0,
		"temporary item bonuses survive death and respawn"
	)
	bubble.queue_free()
	player.queue_free()
	enemy.queue_free()
	board.queue_free()
	await process_frame


func _allocation_total(allocation: Dictionary) -> int:
	return int(allocation.get(RunProgress.SKILL_SPEED, 0)) \
		+ int(allocation.get(RunProgress.SKILL_BUBBLE, 0)) \
		+ int(allocation.get(RunProgress.SKILL_POWER, 0))


func _check(condition: bool, description: String) -> void:
	_checks += 1
	if condition:
		return
	_failures += 1
	push_error("TEST FAILED: %s" % description)
