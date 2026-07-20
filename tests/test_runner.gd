extends SceneTree
## Lightweight headless rules and integration test runner.

var _failures: int = 0
var _checks: int = 0

func _initialize() -> void:
	call_deferred("_run")

func _run() -> void:
	_test_map_catalog()
	_test_settings_bounds()
	_test_blast_propagation()
	_test_half_body_rule()
	_test_ai_hazard_forecast()
	_test_ai_threat_field()
	_test_temporal_pathfinding()
	await _test_board_actor_and_items()
	await _test_rigid_boundaries_and_depth()
	print("BnBonline tests: %d checks, %d failures" % [_checks, _failures])
	quit(1 if _failures > 0 else 0)

func _test_map_catalog() -> void:
	var classic: MapData = MapCatalog.get_map("classic")
	_check(classic.ground_cells.size() == 13, "classic has 13 rows")
	_check(classic.ground_cells[0].size() == 15, "classic has 15 columns")
	_check(classic.player_spawn == Vector2i.ZERO, "classic spawn retained")
	_check(classic.barrier_cells[0][3] == 3, "classic houses become destructible boxes")
	var heart: MapData = MapCatalog.get_map("windmill-heart")
	_check(heart.player_spawn == Vector2i(1, 1), "heart spawn retained")
	_check(heart.barrier_cells[6][7] == 9, "windmill base is rigid")
	_check(heart.barrier_cells[0][0] == 8, "heart border is destructible")

func _test_settings_bounds() -> void:
	var settings := MatchSettings.new()
	settings.map_id = "missing"
	settings.ai_count = 99
	settings.max_speed = 20
	settings.max_bubbles = 50
	settings.max_power = 1
	settings.bubble_skin = "missing"
	settings.normalize()
	_check(settings.map_id == "classic", "invalid map falls back")
	_check(settings.ai_count == 4, "AI count clamps to four")
	_check(settings.max_speed == 150, "speed cap clamps to initial speed")
	_check(settings.max_bubbles == 20, "bubble cap clamps to twenty")
	_check(settings.max_power == 2, "power cap clamps to initial power")
	_check(settings.bubble_skin == "football", "invalid skin falls back")

func _test_blast_propagation() -> void:
	var cells: Array[PackedInt32Array] = []
	for _y: int in range(GameConstants.GRID_ROWS):
		var row := PackedInt32Array()
		row.resize(GameConstants.GRID_COLUMNS)
		row.fill(0)
		cells.append(row)
	cells[6][9] = 1
	cells[6][5] = 3
	var blast: Array[Vector2i] = GameRules.blast_cells(Vector2i(7, 6), 5, cells)
	_check(Vector2i(8, 6) in blast, "empty cell is included in blast")
	_check(Vector2i(9, 6) not in blast, "rigid barrier blocks and is excluded")
	_check(Vector2i(5, 6) in blast, "destructible barrier is included")
	_check(Vector2i(4, 6) not in blast, "blast stops after destructible barrier")
	_check(Vector2i(7, 1) in blast, "each direction keeps its own edge range")
	# A bomb in a covered empty cell is discoverable for chain detonation.
	_check(Vector2i(7, 4) in blast, "covered bubble cell supports chaining")

func _test_half_body_rule() -> void:
	var boundary_position := Vector2(
		GameConstants.GRID_ORIGIN.x + 6.0 * GameConstants.CELL_SIZE,
		GameConstants.GRID_ORIGIN.y + 5.5 * GameConstants.CELL_SIZE
	)
	var feet: Array[Vector2i] = GameRules.foot_cells(boundary_position)
	_check(feet[0] != feet[1], "feet sample opposite sides of a boundary")
	var unsafe: Dictionary = {feet[0]: true}
	_check(not GameRules.both_feet_unsafe(boundary_position, unsafe), "one unsafe foot remains safe")
	unsafe[feet[1]] = true
	_check(GameRules.both_feet_unsafe(boundary_position, unsafe), "two unsafe feet are unsafe")

func _test_ai_hazard_forecast() -> void:
	var snapshot: AIBattleSnapshot = _empty_ai_snapshot()
	snapshot.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(2, 2), 2, 600, 1, 0))
	snapshot.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(10, 8), 2, 1700, 2, 1))
	var forecast: AIHazardForecast = AIHazardForecast.build(snapshot, 3000)
	_check(forecast.danger_eta_ms(Vector2i(2, 2)) == 600, "first bubble keeps its distinct fuse")
	_check(forecast.danger_eta_ms(Vector2i(10, 8)) == 1700, "later bubble keeps its distinct fuse")
	_check(
		forecast.is_unsafe(Vector2i(2, 3), 600, 1050),
		"blast cell remains unsafe for the full explosion interval"
	)
	forecast.set_time_offset_ms(200)
	_check(forecast.danger_eta_ms(Vector2i(2, 2)) == 400, "cached forecast shifts fuse ETA by its age")
	_check(
		forecast.is_unsafe(Vector2i(2, 3), 400, 850),
		"cached forecast shifts unsafe intervals without rebuilding"
	)
	var shifted_field: AIThreatField = AIThreatField.build(forecast, 1)
	_check(
		is_equal_approx(shifted_field.weight_at(Vector2i(2, 2)), 1.0),
		"cached forecast keeps threat expansion aligned to shifted blast time"
	)
	forecast.set_time_offset_ms(1100)
	_check(
		forecast.danger_eta_ms(Vector2i(2, 2)) == AIHazardForecast.NO_DANGER_MS,
		"expired cached danger is discarded after its full interval"
	)
	var chain_snapshot: AIBattleSnapshot = _empty_ai_snapshot()
	chain_snapshot.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(2, 2), 3, 500, 1, 0))
	chain_snapshot.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(5, 2), 2, 2400, 2, 1))
	var chain_forecast: AIHazardForecast = AIHazardForecast.build(chain_snapshot, 3000)
	_check(
		chain_forecast.danger_eta_ms(Vector2i(6, 2)) == 500,
		"earlier blast pulls a later bubble into the chain timeline"
	)
	var ordered_snapshot: AIBattleSnapshot = _empty_ai_snapshot()
	ordered_snapshot.cells[3][4] = 3
	ordered_snapshot.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(2, 3), 2, 500, 1, 0))
	ordered_snapshot.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(6, 3), 4, 500, 2, 1))
	var ordered_forecast: AIHazardForecast = AIHazardForecast.build(ordered_snapshot, 2000)
	_check(
		ordered_forecast.danger_eta_ms(Vector2i(3, 3)) == 500,
		"same-time later blast propagates through a box destroyed by the first blast"
	)
	var active_snapshot: AIBattleSnapshot = _empty_ai_snapshot()
	active_snapshot.explosions.append(AIBattleSnapshot.ExplosionState.new(
		[Vector2i(4, 4)], 275
	))
	var active_forecast: AIHazardForecast = AIHazardForecast.build(active_snapshot, 1000)
	_check(active_forecast.danger_eta_ms(Vector2i(4, 4)) == 0, "active water column is dangerous now")
	_check(
		not active_forecast.is_unsafe(Vector2i(4, 4), 276, 500),
		"active water column clears after its remaining lifetime"
	)

func _test_temporal_pathfinding() -> void:
	var corridor: AIBattleSnapshot = _empty_ai_snapshot(1)
	for x: int in range(3):
		corridor.cells[0][x] = 0
	corridor.cells[1][1] = 0
	corridor.cells[2][1] = 0
	corridor.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(1, 2), 2, 500, 1, 0))
	var corridor_forecast: AIHazardForecast = AIHazardForecast.build(corridor, 2500)
	var waited_plan: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_path(
		corridor, corridor_forecast, Vector2i(0, 0), Vector2i(2, 0), 150.0, 2500, 300
	)
	_check(waited_plan.valid, "time-expanded path can wait for a blast to clear")
	_check(
		waited_plan.cells.size() > 1 and waited_plan.cells[1] == Vector2i(0, 0),
		"safe plan encodes a wait action instead of entering a future blast"
	)
	_check(
		waited_plan.travel_ms() > corridor_forecast.latest_danger_end_ms(),
		"unsafe item-like target is delayed until its complete danger window clears"
	)
	var open_snapshot: AIBattleSnapshot = _empty_ai_snapshot()
	var virtual_forecast: AIHazardForecast = AIHazardForecast.build(
		open_snapshot, 5000, Vector2i(7, 6), 3, 0, 3000
	)
	var escape: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_escape_plan(
		open_snapshot, virtual_forecast, Vector2i(7, 6), 150.0, 5000
	)
	_check(escape.valid and escape.cells.size() > 1, "newly placed bubble has a complete escape plan")
	_check(
		not virtual_forecast.is_unsafe(escape.target_cell(), 0, virtual_forecast.latest_danger_end_ms(), 100),
		"escape plan ends in a cell safe through all predicted explosions"
	)
	var direct_escape: AITemporalPlanner.TimedPlan = AITemporalPlanner.find_direct_escape_plan(
		open_snapshot, virtual_forecast, Vector2i(7, 6), 150.0, 5000
	)
	_check(
		direct_escape.valid and direct_escape.cells.size() > 1,
		"proactive bombing can use a fast no-wait escape with complete interval checks"
	)
	var reachable_now: Dictionary = AITemporalPlanner.reachable_cells(
		open_snapshot, AIHazardForecast.build(open_snapshot), Vector2i(7, 6), 150.0, 600, 0, true
	)
	_check(reachable_now.size() > 1, "terminal reachable-area query returns future safe positions")

func _test_ai_threat_field() -> void:
	var snapshot: AIBattleSnapshot = _empty_ai_snapshot()
	snapshot.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(7, 6), 1, 1000, 1, 0))
	var forecast: AIHazardForecast = AIHazardForecast.build(snapshot, 4000)
	var field: AIThreatField = AIThreatField.build(forecast, 1)
	_check(is_equal_approx(field.weight_at(Vector2i(7, 6)), 1.0), "direct blast cell has full threat")
	_check(is_equal_approx(field.weight_at(Vector2i(8, 7)), 0.5), "first threat ring decays to one half")
	_check(is_equal_approx(field.weight_at(Vector2i(9, 7)), 0.25), "second threat ring decays to one quarter")
	_check(is_equal_approx(field.weight_at(Vector2i(10, 7)), 0.125), "third threat ring decays to one eighth")
	_check(is_zero_approx(field.weight_at(Vector2i(11, 7))), "threat stops after the third ring")
	var relevant: Dictionary = {
		Vector2i(8, 7): true,
		Vector2i(9, 7): true,
	}
	_check(is_equal_approx(field.coverage_ratio(relevant), 0.375), "reachable-cell threat coverage is normalized")
	var blocked: AIBattleSnapshot = _empty_ai_snapshot()
	blocked.cells[2][3] = 1
	blocked.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(2, 2), 0, 1000, 1, 0))
	var blocked_field: AIThreatField = AIThreatField.build(
		AIHazardForecast.build(blocked, 4000), 1
	)
	_check(is_zero_approx(blocked_field.weight_at(Vector2i(4, 2))), "rigid wall blocks threat-ring expansion")
	var overlapping: AIBattleSnapshot = _empty_ai_snapshot()
	overlapping.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(7, 6), 0, 800, 1, 0))
	overlapping.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(7, 6), 0, 1200, 1, 1))
	var overlapping_field: AIThreatField = AIThreatField.build(
		AIHazardForecast.build(overlapping, 3000), 1
	)
	_check(
		is_equal_approx(overlapping_field.weight_at(Vector2i(8, 6)), 0.75),
		"overlapping bubbles use saturating threat accumulation"
	)
	var single_overlap: AIBattleSnapshot = _empty_ai_snapshot()
	single_overlap.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(7, 6), 0, 800, 1, 0))
	var single_overlap_field: AIThreatField = AIThreatField.build(
		AIHazardForecast.build(single_overlap, 3000), 1
	)
	_check(
		is_equal_approx(
			overlapping_field.marginal_weight(single_overlap_field, {Vector2i(8, 6): true}),
			0.25
		),
		"threat field exposes marginal pressure on a relevant region"
	)
	var timed_boxes: AIBattleSnapshot = _empty_ai_snapshot()
	timed_boxes.cells[2][3] = 3
	timed_boxes.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(2, 2), 1, 500, 2, 0))
	timed_boxes.bombs.append(AIBattleSnapshot.BombState.new(Vector2i(4, 2), 0, 1000, 1, 1))
	var timed_field: AIThreatField = AIThreatField.build(
		AIHazardForecast.build(timed_boxes, 3000), 1
	)
	_check(
		is_equal_approx(timed_field.weight_at(Vector2i(2, 2)), 0.25),
		"threat rings use boxes destroyed before the owning bubble explodes"
	)

func _empty_ai_snapshot(fill_code: int = 0) -> AIBattleSnapshot:
	var snapshot := AIBattleSnapshot.new()
	for _y: int in range(GameConstants.GRID_ROWS):
		var row := PackedInt32Array()
		row.resize(GameConstants.GRID_COLUMNS)
		row.fill(fill_code)
		snapshot.cells.append(row)
	return snapshot

func _test_board_actor_and_items() -> void:
	var board := GameBoard.new()
	root.add_child(board)
	board.reset(MapCatalog.get_map("classic"))
	var settings := MatchSettings.new()
	var actor := GameActor.new()
	root.add_child(actor)
	actor.setup("测试玩家", 1, true, board, settings, Vector2i.ZERO)
	_check(board.can_actor_occupy(GameConstants.grid_to_world(Vector2i.ZERO), actor), "actor can occupy spawn")
	_check(not board.can_actor_occupy(GameConstants.grid_to_world(Vector2i(1, 0)), actor), "actor cannot occupy a box")
	var item_code: int = board.destroy_cell(Vector2i(1, 0))
	_check(item_code in [101, 102, 103], "destroyed box creates a valid upgrade")
	_check(board.take_item(Vector2i(1, 0)) == item_code, "upgrade can be collected")
	_check(board.cell_code(Vector2i(1, 0)) == 0, "collected upgrade clears the cell")
	var bubble := GameBubble.new()
	root.add_child(bubble)
	var overlapping_actor := GameActor.new()
	root.add_child(overlapping_actor)
	overlapping_actor.setup("重叠角色", 2, false, board, settings, Vector2i.ZERO)
	var outside_actor := GameActor.new()
	root.add_child(outside_actor)
	outside_actor.setup("外部角色", 2, false, board, settings, Vector2i(0, 1))
	bubble.setup(actor, Vector2i.ZERO, "football", GameConstants.BUBBLE_FUSE_SECONDS, [actor, overlapping_actor])
	board.register_bubble(bubble)
	overlapping_actor.position = GameConstants.grid_to_world(Vector2i.ZERO) + Vector2(0.0, 21.0)
	_check(
		board.can_actor_occupy(overlapping_actor.position + Vector2(0.0, 3.0), overlapping_actor),
		"another actor overlapping a newly placed bubble can finish exiting"
	)
	overlapping_actor.position = GameConstants.grid_to_world(Vector2i(0, 1))
	_check(
		not board.can_actor_occupy(overlapping_actor.position + Vector2(0.0, -14.0), overlapping_actor),
		"overlapping actor cannot re-enter after fully clearing the bubble cell"
	)
	_check(
		not board.can_actor_occupy(outside_actor.position + Vector2(0.0, -14.0), outside_actor),
		"actor outside at placement is blocked immediately"
	)
	actor.position = GameConstants.grid_to_world(Vector2i.ZERO) + Vector2(0.0, 21.0)
	_check(
		board.can_actor_occupy(actor.position + Vector2(0.0, 3.0), actor),
		"bubble owner can keep moving while its body still overlaps the placed bubble"
	)
	actor.position = GameConstants.grid_to_world(Vector2i(0, 1))
	_check(
		not board.can_actor_occupy(actor.position + Vector2(0.0, -14.0), actor),
		"bubble owner cannot re-enter after fully leaving the placed bubble"
	)
	board.unregister_bubble(bubble)
	bubble.queue_free()
	actor.register_unsafe_frame(null)
	_check(not actor.stats.is_trapped, "one unsafe frame does not trap")
	actor.register_unsafe_frame(null)
	_check(actor.stats.is_trapped, "two consecutive unsafe frames trap")
	actor.rescue()
	_check(not actor.stats.is_trapped, "self rescue clears trap")
	var path: Array[Vector2i] = board.find_path(Vector2i.ZERO, Vector2i(1, 1), actor)
	_check(path.size() >= 2, "AStarGrid2D finds a route through open cells")
	actor.queue_free()
	overlapping_actor.queue_free()
	outside_actor.queue_free()
	board.queue_free()
	await process_frame

func _test_rigid_boundaries_and_depth() -> void:
	var board := GameBoard.new()
	root.add_child(board)
	board.reset(MapCatalog.get_map("classic"))
	var rigid_cell := Vector2i(5, 5)
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			board.cells[y][x] = 0
	var settings := MatchSettings.new()
	var actor := GameActor.new()
	root.add_child(actor)
	actor.setup("碰撞测试", 1, true, board, settings, Vector2i(0, 1))
	var first_center: Vector2 = GameConstants.grid_to_world(Vector2i.ZERO)
	var last_center: Vector2 = GameConstants.grid_to_world(
		Vector2i(GameConstants.GRID_COLUMNS - 1, GameConstants.GRID_ROWS - 1)
	)
	var arena_center: Vector2 = (first_center + last_center) * 0.5
	var boundary_cases: Array[Dictionary] = [
		{"start": Vector2(first_center.x, arena_center.y), "outward": Vector2.LEFT, "side": "left"},
		{"start": Vector2(last_center.x, arena_center.y), "outward": Vector2.RIGHT, "side": "right"},
		{"start": Vector2(arena_center.x, first_center.y), "outward": Vector2.UP, "side": "top"},
		{"start": Vector2(arena_center.x, last_center.y), "outward": Vector2.DOWN, "side": "bottom"},
	]
	var physics_delta: float = 1.0 / 60.0
	var expected_distance: float = actor.stats.move_speed * physics_delta
	for boundary_case: Dictionary in boundary_cases:
		var boundary_start: Vector2 = boundary_case["start"] as Vector2
		var outward: Vector2 = boundary_case["outward"] as Vector2
		actor.position = boundary_start
		actor.velocity = outward * actor.stats.move_speed
		actor.call("_attempt_move", physics_delta)
		_check(
			actor.position.is_equal_approx(boundary_start),
			"%s arena boundary blocks outward movement" % str(boundary_case["side"])
		)
		actor.position = boundary_start
		actor.velocity = -outward * actor.stats.move_speed
		actor.call("_attempt_move", physics_delta)
		_check(
			actor.position.is_equal_approx(boundary_start - outward * expected_distance),
			"%s arena boundary still allows inward movement" % str(boundary_case["side"])
		)
	var free_start := GameConstants.grid_to_world(Vector2i(5, 5)) + Vector2(3.25, -4.5)
	var directions: Array[Vector2] = [Vector2.UP, Vector2.DOWN, Vector2.LEFT, Vector2.RIGHT]
	for direction: Vector2 in directions:
		actor.position = free_start
		actor.velocity = direction * actor.stats.move_speed
		actor.call("_attempt_move", physics_delta)
		_check(
			actor.position.is_equal_approx(free_start + direction * expected_distance),
			"%s movement is straight and exactly speed times delta" % str(direction)
		)
	board.cells[rigid_cell.y][rigid_cell.x] = 1
	var rigid_top_left: Vector2 = GameConstants.grid_to_top_left(rigid_cell)
	var rigid_center: Vector2 = GameConstants.grid_to_world(rigid_cell)
	var rigid_contacts: Array[Dictionary] = [
		{"position": Vector2(rigid_top_left.x - 20.0, rigid_center.y), "toward": Vector2.RIGHT, "side": "left"},
		{"position": Vector2(rigid_top_left.x + GameConstants.CELL_SIZE + 20.0, rigid_center.y), "toward": Vector2.LEFT, "side": "right"},
		{"position": Vector2(rigid_center.x, rigid_top_left.y - 20.0), "toward": Vector2.DOWN, "side": "top"},
		{"position": Vector2(rigid_center.x, rigid_top_left.y + GameConstants.CELL_SIZE + 20.0), "toward": Vector2.UP, "side": "bottom"},
	]
	for rigid_contact: Dictionary in rigid_contacts:
		var contact_position: Vector2 = rigid_contact["position"] as Vector2
		var toward_rigid: Vector2 = rigid_contact["toward"] as Vector2
		actor.position = contact_position
		actor.velocity = toward_rigid * actor.stats.move_speed
		actor.call("_attempt_move", physics_delta)
		_check(
			actor.position.is_equal_approx(contact_position),
			"%s rigid center boundary blocks movement into the obstacle" % str(rigid_contact["side"])
		)
		actor.position = contact_position
		actor.velocity = -toward_rigid * actor.stats.move_speed
		actor.call("_attempt_move", physics_delta)
		_check(
			actor.position.is_equal_approx(contact_position - toward_rigid * expected_distance),
			"%s rigid center boundary allows movement away from the obstacle" % str(rigid_contact["side"])
		)
	for y: int in range(3):
		for x: int in range(4):
			board.cells[y][x] = 0
	for x: int in range(1, 4):
		board.cells[0][x] = 1
		board.cells[2][x] = 1
	actor.position = GameConstants.grid_to_world(Vector2i(0, 1)) + Vector2(0.0, 6.0)
	var corridor_y: float = actor.position.y
	actor.velocity = Vector2.RIGHT * actor.stats.move_speed
	for _step: int in range(20):
		actor.call("_attempt_move", 1.0 / 60.0)
	_check(actor.current_cell().x >= 1, "slightly misaligned actor can enter a one-cell empty corridor")
	_check(
		is_equal_approx(actor.position.y, corridor_y),
		"horizontal corridor entry never changes the vertical position"
	)
	for y: int in range(3):
		for x: int in range(3):
			board.cells[y][x] = 0
	board.cells[rigid_cell.y][rigid_cell.x] = 1
	var rigid_bottom: float = rigid_top_left.y + GameConstants.CELL_SIZE
	actor.position = Vector2(rigid_center.x, rigid_bottom + 22.7)
	actor.velocity = Vector2.UP * actor.stats.move_speed
	actor.call("_attempt_move", 1.0 / 60.0)
	actor.call("_attempt_move", 1.0 / 60.0)
	_check(
		absf(actor.position.y - (rigid_bottom + 20.0)) < 0.05,
		"blocked upward movement resolves to the exact rigid boundary"
	)
	board.reset(MapCatalog.get_map("classic"))
	var depth_cell := Vector2i(0, 2)
	var depth_sprite: Sprite2D = board._cell_sprites[depth_cell] as Sprite2D
	actor.position = GameConstants.grid_to_world(Vector2i(0, 1))
	actor.call("_process", 0.0)
	_check(depth_sprite.z_index > actor.z_index, "rigid body covers an actor standing behind it")
	actor.position = GameConstants.grid_to_world(Vector2i(0, 3))
	actor.call("_process", 0.0)
	_check(actor.z_index > depth_sprite.z_index, "actor covers a rigid body when standing in front of it")
	actor.queue_free()
	board.queue_free()
	await process_frame

func _check(condition: bool, description: String) -> void:
	_checks += 1
	if condition:
		return
	_failures += 1
	push_error("TEST FAILED: %s" % description)
