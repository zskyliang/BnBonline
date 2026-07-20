class_name MatchController
extends Node
## Root match orchestrator for spawning, scoring, explosions, settings, and round flow.

var settings: MatchSettings
var board: GameBoard
var hud: GameHud
var _arena_timer_label: Label
var _fps_label: Label

var _world: Node2D
var _entity_root: Node2D
var _effect_root: Node2D
var _actors: Array[GameActor] = []
var _ai_controllers: Array[RuleAI] = []
var _active_explosions: Array[ExplosionEffect] = []
var _active_unsafe_cells: Dictionary = {}
var _active_explosion_attackers: Dictionary = {}
var _scores: Dictionary = {}
var _player: GameActor
var _remaining_seconds: float = GameConstants.ROUND_SECONDS
var _round_over: bool = false
var _is_paused: bool = false
var _simulation_time_ms: float = 0.0
var _rng := RandomNumberGenerator.new()
var _ai_item_claims: Dictionary = {}
var _ai_schedule_elapsed_seconds: float = 0.0
var _next_ai_index: int = 0
var _displayed_seconds: int = -1
var _last_fps_update_ms: int = -1000
var _cached_ai_forecast: AIHazardForecast
var _cached_ai_forecast_horizon_ms: int = 0
var _cached_ai_forecast_built_ms: int = 0
var _ai_hazard_revision: int = 0
var _cached_ai_hazard_revision: int = -1

const AI_FORECAST_CACHE_MS: int = 60

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	_rng.randomize()
	_ensure_input_actions()
	settings = MatchSettings.load_from_disk()
	_build_scene_tree()
	_connect_hud()
	start_match()

func _physics_process(delta: float) -> void:
	if _round_over or _is_paused:
		return
	_simulation_time_ms += delta * 1000.0
	_process_ai_schedule(delta)
	_remaining_seconds = maxf(0.0, _remaining_seconds - delta)
	_update_time_display()
	_resolve_explosion_hits()
	_resolve_actor_contacts()
	if _remaining_seconds <= 0.0:
		_end_round()

func _process(_delta: float) -> void:
	var now_ms: int = Time.get_ticks_msec()
	if now_ms - _last_fps_update_ms < 250:
		return
	_last_fps_update_ms = now_ms
	if is_instance_valid(_fps_label):
		_fps_label.text = "FPS: %d" % Engine.get_frames_per_second()

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("pause_game") and not _round_over:
		if _is_paused:
			_resume_match()
		else:
			_pause_match()
		get_viewport().set_input_as_handled()

func start_match() -> void:
	get_tree().paused = false
	_is_paused = false
	_round_over = false
	_remaining_seconds = GameConstants.ROUND_SECONDS
	_simulation_time_ms = 0.0
	_displayed_seconds = -1
	_clear_match_nodes()
	board.reset(MapCatalog.get_map(settings.map_id))
	_spawn_fighters()
	hud.sync_settings(settings)
	hud.hide_pause()
	hud.hide_result()
	_update_time_display()
	_update_scoreboard()
	_audio_call(&"play_sfx", [&"start"])
	_audio_call(&"play_music")

func danger_eta_ms(cell: Vector2i) -> int:
	return build_ai_forecast().danger_eta_ms(cell)

func build_ai_snapshot() -> AIBattleSnapshot:
	var snapshot := AIBattleSnapshot.new()
	snapshot.cells = MapCatalog.clone_matrix(board.cells)
	var serial: int = 0
	for value: Variant in board.bombs.values():
		var bubble: GameBubble = value as GameBubble
		if not is_instance_valid(bubble) or bubble.has_exploded:
			continue
		var owner_id: int = 0
		if is_instance_valid(bubble.bubble_owner):
			owner_id = bubble.bubble_owner.get_instance_id()
		snapshot.bombs.append(AIBattleSnapshot.BombState.new(
			bubble.cell, bubble.power, bubble.milliseconds_until_explosion(), owner_id, serial
		))
		serial += 1
	for effect: ExplosionEffect in _active_explosions:
		if is_instance_valid(effect):
			snapshot.explosions.append(AIBattleSnapshot.ExplosionState.new(
				effect.cells, effect.milliseconds_remaining()
			))
	for battle_actor: GameActor in _actors:
		if not is_instance_valid(battle_actor):
			continue
		snapshot.actors.append(AIBattleSnapshot.ActorState.new(
			battle_actor.get_instance_id(),
			battle_actor.team_id,
			battle_actor.current_cell(),
			battle_actor.position,
			battle_actor.stats.move_speed,
			battle_actor.stats.bubble_capacity,
			battle_actor.stats.active_bubbles,
			battle_actor.stats.power,
			battle_actor.stats.is_dead,
			battle_actor.stats.is_trapped,
			battle_actor.is_player
		))
	snapshot.item_cells = board.get_item_cells()
	return snapshot

func build_ai_forecast(horizon_ms: int = 5000) -> AIHazardForecast:
	return get_shared_ai_forecast(build_ai_snapshot(), horizon_ms)

func get_shared_ai_forecast(
		snapshot: AIBattleSnapshot,
		horizon_ms: int
	) -> AIHazardForecast:
	var now_ms: int = get_simulation_time_ms()
	var cache_age_ms: int = maxi(0, now_ms - _cached_ai_forecast_built_ms)
	if _cached_ai_forecast == null \
			or _cached_ai_forecast_horizon_ms != horizon_ms \
			or _cached_ai_hazard_revision != _ai_hazard_revision \
			or cache_age_ms > AI_FORECAST_CACHE_MS:
		_cached_ai_forecast = AIHazardForecast.build(snapshot, horizon_ms)
		_cached_ai_forecast_horizon_ms = horizon_ms
		_cached_ai_forecast_built_ms = now_ms
		_cached_ai_hazard_revision = _ai_hazard_revision
		cache_age_ms = 0
	_cached_ai_forecast.set_time_offset_ms(cache_age_ms)
	return _cached_ai_forecast

func get_simulation_time_ms() -> int:
	return int(_simulation_time_ms)

func claim_ai_item(actor_id: int, cell: Vector2i, ttl_ms: int = 600) -> void:
	_prune_ai_item_claims()
	_ai_item_claims[cell] = {
		"actor_id": actor_id,
		"expires_ms": get_simulation_time_ms() + ttl_ms,
	}

func is_ai_item_claimed_by_other(actor_id: int, cell: Vector2i) -> bool:
	_prune_ai_item_claims()
	if not _ai_item_claims.has(cell):
		return false
	return int((_ai_item_claims[cell] as Dictionary).get("actor_id", 0)) != actor_id

func release_ai_item_claims(actor_id: int) -> void:
	for cell: Vector2i in _ai_item_claims.keys():
		if int((_ai_item_claims[cell] as Dictionary).get("actor_id", 0)) == actor_id:
			_ai_item_claims.erase(cell)

func get_player() -> GameActor:
	return _player

func get_actors() -> Array[GameActor]:
	return _actors

func request_bomb(actor: GameActor) -> bool:
	if _round_over or _is_paused or not is_instance_valid(actor):
		return false
	if actor.stats.is_dead or actor.stats.is_trapped:
		return false
	if actor.stats.active_bubbles >= actor.stats.bubble_capacity:
		return false
	var cell: Vector2i = actor.current_cell()
	if not board.can_place_bubble(cell):
		return false
	var initially_overlapping_actors: Array[GameActor] = []
	for battle_actor: GameActor in _actors:
		if not is_instance_valid(battle_actor) or battle_actor.stats.is_dead:
			continue
		if cell in GameRules.body_cells(battle_actor.position):
			initially_overlapping_actors.append(battle_actor)
	var bubble := GameBubble.new()
	_entity_root.add_child(bubble)
	bubble.setup(
		actor,
		cell,
		settings.bubble_skin,
		GameConstants.BUBBLE_FUSE_SECONDS,
		initially_overlapping_actors
	)
	bubble.exploded.connect(_on_bubble_exploded)
	board.register_bubble(bubble)
	actor.stats.active_bubbles += 1
	_audio_call(&"play_sfx", [&"lay"])
	return true

func _build_scene_tree() -> void:
	_world = Node2D.new()
	_world.name = "GameWorld"
	add_child(_world)
	board = GameBoard.new()
	board.name = "Board"
	_world.add_child(board)
	board.hazard_changed.connect(_invalidate_ai_forecast)
	_entity_root = Node2D.new()
	_entity_root.name = "Entities"
	_world.add_child(_entity_root)
	_effect_root = Node2D.new()
	_effect_root.name = "Effects"
	_world.add_child(_effect_root)
	var chrome := CanvasLayer.new()
	chrome.name = "GameChrome"
	chrome.layer = 5
	add_child(chrome)
	_arena_timer_label = Label.new()
	_arena_timer_label.position = Vector2(694, 33)
	_arena_timer_label.size = Vector2(92, 24)
	_arena_timer_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_arena_timer_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	_arena_timer_label.add_theme_font_size_override("font_size", 16)
	_arena_timer_label.add_theme_color_override("font_color", Color("ffe36e"))
	_arena_timer_label.add_theme_constant_override("outline_size", 3)
	_arena_timer_label.add_theme_color_override("font_outline_color", Color("103657"))
	chrome.add_child(_arena_timer_label)
	_fps_label = Label.new()
	_fps_label.position = Vector2(655, 1)
	_fps_label.size = Vector2(125, 24)
	_fps_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_fps_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	_fps_label.add_theme_font_size_override("font_size", 14)
	_fps_label.add_theme_color_override("font_color", Color("b8f4cf"))
	_fps_label.add_theme_constant_override("outline_size", 3)
	_fps_label.add_theme_color_override("font_outline_color", Color("103657"))
	_fps_label.text = "FPS: 0"
	chrome.add_child(_fps_label)
	var canvas := CanvasLayer.new()
	canvas.name = "Interface"
	canvas.layer = 20
	add_child(canvas)
	hud = GameHud.new()
	hud.name = "HUD"
	canvas.add_child(hud)

func _connect_hud() -> void:
	hud.map_selected.connect(_on_map_selected)
	hud.ai_count_selected.connect(_on_ai_count_selected)
	hud.max_speed_changed.connect(_on_max_speed_changed)
	hud.max_bubbles_changed.connect(_on_max_bubbles_changed)
	hud.max_power_changed.connect(_on_max_power_changed)
	hud.bubble_skin_selected.connect(_on_bubble_skin_selected)
	hud.resume_requested.connect(_resume_match)
	hud.restart_requested.connect(start_match)
	hud.quit_requested.connect(get_tree().quit)

func _clear_match_nodes() -> void:
	_active_explosions.clear()
	_active_unsafe_cells.clear()
	_active_explosion_attackers.clear()
	_actors.clear()
	_ai_controllers.clear()
	_ai_schedule_elapsed_seconds = 0.0
	_next_ai_index = 0
	_scores.clear()
	_ai_item_claims.clear()
	_player = null
	for child: Node in _entity_root.get_children():
		child.queue_free()
	for child: Node in _effect_root.get_children():
		child.queue_free()

func _spawn_fighters() -> void:
	var used_cells: Array[Vector2i] = []
	_player = _spawn_actor("玩家", 1, true, board.map_data.player_spawn)
	used_cells.append(board.map_data.player_spawn)
	for index: int in range(settings.ai_count):
		var spawn: Vector2i = _find_ai_spawn(used_cells)
		used_cells.append(spawn)
		var ai_actor: GameActor = _spawn_actor("AI %d" % (index + 1), 2, false, spawn)
		var controller := RuleAI.new()
		controller.name = "RuleAI%d" % (index + 1)
		ai_actor.add_child(controller)
		# Normal matches use one shared round-robin scheduler so four 150 ms
		# thinkers do not create a main-thread spike on the same physics frame.
		controller.setup(ai_actor, board, self, -1, false)
		_ai_controllers.append(controller)
	if not _ai_controllers.is_empty():
		_ai_schedule_elapsed_seconds = _ai_schedule_slot_seconds()

func _process_ai_schedule(delta: float) -> void:
	if _ai_controllers.is_empty():
		return
	_ai_schedule_elapsed_seconds += delta
	var slot_seconds: float = _ai_schedule_slot_seconds()
	if _ai_schedule_elapsed_seconds + 0.000001 < slot_seconds:
		return
	# Never catch up several thinkers in one frame after a hitch. Keeping at
	# most one slot of debt smooths the next frame without lowering steady-state
	# per-AI cadence.
	_ai_schedule_elapsed_seconds = minf(
		_ai_schedule_elapsed_seconds - slot_seconds,
		slot_seconds
	)
	if _next_ai_index >= _ai_controllers.size():
		_next_ai_index = 0
	var controller: RuleAI = _ai_controllers[_next_ai_index]
	_next_ai_index = (_next_ai_index + 1) % _ai_controllers.size()
	if is_instance_valid(controller):
		controller.reconsider_now()

func _ai_schedule_slot_seconds() -> float:
	return GameConstants.AI_THINK_SECONDS / maxf(1.0, float(_ai_controllers.size()))

func _spawn_actor(
		display_name: String,
		team_id: int,
		is_player_actor: bool,
		spawn_cell: Vector2i
	) -> GameActor:
	var actor := GameActor.new()
	actor.name = display_name
	_entity_root.add_child(actor)
	actor.setup(display_name, team_id, is_player_actor, board, settings, spawn_cell)
	actor.bomb_requested.connect(_on_bomb_requested)
	actor.died.connect(_on_actor_died)
	actor.rescued.connect(_on_actor_rescued)
	actor.item_collected.connect(_on_item_collected)
	_actors.append(actor)
	_scores[actor.get_instance_id()] = {"name": display_name, "kills": 0}
	return actor

func _find_ai_spawn(used_cells: Array[Vector2i]) -> Vector2i:
	var best: Vector2i = board.map_data.player_spawn
	var best_score: int = -1
	for candidate: Vector2i in board.get_open_cells():
		if candidate in used_cells:
			continue
		var nearest: int = 99999
		for used: Vector2i in used_cells:
			nearest = mini(nearest, absi(candidate.x - used.x) + absi(candidate.y - used.y))
		if nearest > best_score:
			best_score = nearest
			best = candidate
	return best

func _on_bomb_requested(actor: GameActor) -> void:
	request_bomb(actor)

func _on_bubble_exploded(bubble: GameBubble) -> void:
	if not is_instance_valid(bubble):
		return
	board.unregister_bubble(bubble)
	if is_instance_valid(bubble.bubble_owner):
		bubble.bubble_owner.stats.active_bubbles = maxi(0, bubble.bubble_owner.stats.active_bubbles - 1)
	var blast: Array[Vector2i] = GameRules.blast_cells(bubble.cell, bubble.power, board.cells)
	var chained: Array[GameBubble] = []
	for cell: Vector2i in blast:
		if board.bombs.has(cell):
			var other: GameBubble = board.bombs[cell] as GameBubble
			if other != bubble and is_instance_valid(other) and other not in chained:
				chained.append(other)
		if GameRules.is_destructible(board.cell_code(cell)):
			board.destroy_cell(cell)
		elif board.cell_code(cell) >= 101:
			board.take_item(cell)
	var effect := ExplosionEffect.new()
	_effect_root.add_child(effect)
	effect.setup(blast, bubble.cell, bubble.bubble_owner)
	effect.finished.connect(_on_explosion_finished)
	_register_explosion_effect(effect)
	_audio_call(&"play_sfx", [&"explode"])
	for chained_bubble: GameBubble in chained:
		chained_bubble.call_deferred("explode_now")

func _on_explosion_finished(effect: ExplosionEffect) -> void:
	_unregister_explosion_effect(effect)

func _register_explosion_effect(effect: ExplosionEffect) -> void:
	_active_explosions.append(effect)
	_rebuild_active_explosion_lookup()
	_invalidate_ai_forecast()

func _unregister_explosion_effect(effect: ExplosionEffect) -> void:
	_active_explosions.erase(effect)
	_rebuild_active_explosion_lookup()
	_invalidate_ai_forecast()

func _resolve_explosion_hits() -> void:
	for actor: GameActor in _actors:
		if not is_instance_valid(actor) or actor.stats.is_dead:
			continue
		var feet: Array[Vector2i] = actor.foot_cells()
		if GameRules.both_feet_unsafe(actor.position, _active_unsafe_cells):
			var attacker: GameActor = _active_explosion_attackers.get(
				feet[0], _active_explosion_attackers.get(feet[1], null)
			) as GameActor
			actor.register_unsafe_frame(attacker)
		else:
			actor.register_safe_frame()

func _rebuild_active_explosion_lookup() -> void:
	_active_unsafe_cells.clear()
	_active_explosion_attackers.clear()
	for effect: ExplosionEffect in _active_explosions:
		if not is_instance_valid(effect):
			continue
		for cell: Vector2i in effect.cells:
			_active_unsafe_cells[cell] = true
			if is_instance_valid(effect.attacker):
				_active_explosion_attackers[cell] = effect.attacker

func _resolve_actor_contacts() -> void:
	for left_index: int in range(_actors.size()):
		var left: GameActor = _actors[left_index]
		if not is_instance_valid(left) or left.stats.is_dead:
			continue
		for right_index: int in range(left_index + 1, _actors.size()):
			var right: GameActor = _actors[right_index]
			if not is_instance_valid(right) or right.stats.is_dead:
				continue
			if left.position.distance_to(right.position) > 24.0:
				continue
			_resolve_touch_pair(left, right)

func _resolve_touch_pair(left: GameActor, right: GameActor) -> void:
	if left.stats.is_trapped and not right.stats.is_trapped:
		if left.team_id == right.team_id:
			left.rescue()
		else:
			left.finish_by_touch(right)
	elif right.stats.is_trapped and not left.stats.is_trapped:
		if left.team_id == right.team_id:
			right.rescue()
		else:
			right.finish_by_touch(left)

func _on_actor_died(victim: GameActor, attacker: GameActor) -> void:
	_audio_call(&"play_sfx", [&"die"])
	if is_instance_valid(attacker) and attacker != victim and _scores.has(attacker.get_instance_id()):
		var score: Dictionary = _scores[attacker.get_instance_id()]
		score["kills"] = int(score.get("kills", 0)) + 1
		_scores[attacker.get_instance_id()] = score
	_update_scoreboard()
	_respawn_later(victim)

func _respawn_later(actor: GameActor) -> void:
	await get_tree().create_timer(GameConstants.RESPAWN_SECONDS, false).timeout
	if _round_over or not is_instance_valid(actor):
		return
	actor.respawn(_find_respawn_cell(actor))

func _find_respawn_cell(actor: GameActor) -> Vector2i:
	var candidates: Array[Vector2i] = board.get_open_cells()
	candidates.shuffle()
	var forecast: AIHazardForecast = build_ai_forecast()
	for cell: Vector2i in candidates:
		if forecast.danger_eta_ms(cell) < 1500:
			continue
		var blocked: bool = false
		for other: GameActor in _actors:
			if other != actor and not other.stats.is_dead and other.current_cell() == cell:
				blocked = true
				break
		if not blocked:
			return cell
	return board.map_data.player_spawn

func _on_actor_rescued(_actor: GameActor) -> void:
	_audio_call(&"play_sfx", [&"save"])

func _on_item_collected(_actor: GameActor, _item_code: int) -> void:
	_audio_call(&"play_sfx", [&"get"])

func _update_scoreboard() -> void:
	var entries: Array[Dictionary] = []
	for value: Variant in _scores.values():
		entries.append((value as Dictionary).duplicate())
	entries.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return int(a.get("kills", 0)) > int(b.get("kills", 0))
	)
	hud.update_scores(entries)

func _end_round() -> void:
	if _round_over:
		return
	_round_over = true
	_is_paused = true
	_audio_call(&"stop_music")
	var best_score: int = -1
	var winners: Array[String] = []
	for value: Variant in _scores.values():
		var entry: Dictionary = value as Dictionary
		var kills: int = int(entry.get("kills", 0))
		if kills > best_score:
			best_score = kills
			winners = [str(entry.get("name", "?"))]
		elif kills == best_score:
			winners.append(str(entry.get("name", "?")))
	if winners.size() == 1:
		hud.show_result("%s 获胜！" % winners[0], "最高击败：%d" % best_score)
		_audio_call(&"play_sfx", [&"win"])
	else:
		hud.show_result("平局", "%s｜击败：%d" % ["、".join(winners), best_score])
		_audio_call(&"play_sfx", [&"draw"])
	get_tree().paused = true

func _pause_match() -> void:
	_is_paused = true
	hud.show_pause()
	get_tree().paused = true

func _resume_match() -> void:
	if _round_over:
		return
	get_tree().paused = false
	_is_paused = false
	hud.hide_pause()

func _on_map_selected(map_id: String) -> void:
	settings.map_id = map_id
	settings.save_to_disk()
	start_match()

func _on_ai_count_selected(count: int) -> void:
	settings.ai_count = count
	settings.save_to_disk()
	start_match()

func _on_max_speed_changed(value: int) -> void:
	settings.max_speed = value
	_apply_caps_and_save()

func _on_max_bubbles_changed(value: int) -> void:
	settings.max_bubbles = value
	_apply_caps_and_save()

func _on_max_power_changed(value: int) -> void:
	settings.max_power = value
	_apply_caps_and_save()

func _on_bubble_skin_selected(skin: String) -> void:
	settings.bubble_skin = skin
	settings.save_to_disk()

func _apply_caps_and_save() -> void:
	settings.normalize()
	for actor: GameActor in _actors:
		if is_instance_valid(actor):
			actor.clamp_stats()
	settings.save_to_disk()

func _ensure_input_actions() -> void:
	_register_action(&"move_left", [KEY_LEFT, KEY_A])
	_register_action(&"move_right", [KEY_RIGHT, KEY_D])
	_register_action(&"move_up", [KEY_UP, KEY_W])
	_register_action(&"move_down", [KEY_DOWN, KEY_S])
	_register_action(&"place_bomb", [KEY_SPACE])
	_register_action(&"self_rescue", [KEY_1])
	_register_action(&"pause_game", [KEY_ESCAPE])

func _register_action(action: StringName, keycodes: Array) -> void:
	if InputMap.has_action(action):
		InputMap.erase_action(action)
	InputMap.add_action(action)
	for keycode: Key in keycodes:
		var event := InputEventKey.new()
		event.keycode = keycode
		InputMap.action_add_event(action, event)

func _format_time(seconds_left: float) -> String:
	var total_seconds: int = maxi(0, ceili(seconds_left))
	return "%02d:%02d" % [total_seconds / 60, total_seconds % 60]

func _update_time_display() -> void:
	var total_seconds: int = maxi(0, ceili(_remaining_seconds))
	if total_seconds == _displayed_seconds:
		return
	_displayed_seconds = total_seconds
	hud.update_timer(float(total_seconds))
	_arena_timer_label.text = _format_time(float(total_seconds))

func _prune_ai_item_claims() -> void:
	var now_ms: int = get_simulation_time_ms()
	for cell: Vector2i in _ai_item_claims.keys():
		if int((_ai_item_claims[cell] as Dictionary).get("expires_ms", 0)) <= now_ms:
			_ai_item_claims.erase(cell)

func _invalidate_ai_forecast() -> void:
	_ai_hazard_revision += 1
	_cached_ai_forecast = null

func _audio_call(method: StringName, arguments: Array = []) -> void:
	if "--mute" in OS.get_cmdline_user_args():
		return
	var audio_manager: Node = get_node_or_null("/root/AudioManager")
	if is_instance_valid(audio_manager):
		audio_manager.callv(method, arguments)
