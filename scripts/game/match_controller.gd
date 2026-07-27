class_name MatchController
extends Node
## Root match orchestrator for spawning, scoring, explosions, settings, and round flow.

const DEFAULT_AI_PROFILE: AIBehaviorProfile = preload(
	"res://assets/config/ai_behavior_profile.tres"
)

signal match_finished(results: Array[Dictionary])
signal scores_changed(entries: Array[Dictionary])
signal time_changed(seconds: int)
signal stage_changed(stage_number: int)

enum AppState { LOBBY, SETUP, MATCH, RESULT }

var settings: MatchSettings
var run_progress: RunProgress
var board: GameBoard
var hud: GameHud
var _arena_timer_label: Label
var _fps_label: Label
var app_state: AppState = AppState.LOBBY

var _world: Node2D
var _entity_root: Node2D
var _effect_root: Node2D
var _arena_view: ArenaView3D
var _actors: Array[GameActor] = []
var _ai_controllers: Array[RuleAI] = []
var _active_explosions: Array[ExplosionEffect] = []
var _active_unsafe_cells: Dictionary = {}
var _active_explosion_sources: Dictionary = {}
var _respawn_timers: Dictionary = {}
var _player: GameActor
var _remaining_seconds: float = GameConstants.ROUND_SECONDS
var _round_over: bool = false
var _is_paused: bool = false
var _simulation_time_ms: float = 0.0
var _rng := RandomNumberGenerator.new()
var _item_rng := RandomNumberGenerator.new()
var _next_item_spawn_ms: int = 10000
var _item_claims: Dictionary = {}
var _ai_schedule_elapsed_seconds: float = 0.0
var _next_ai_index: int = 0
var _displayed_seconds: int = -1
var _last_fps_update_ms: int = -1000
var _cached_ai_forecast: AIHazardForecast
var _cached_ai_forecast_horizon_ms: int = 0
var _cached_ai_forecast_built_ms: int = 0
var _ai_hazard_revision: int = 0
var _cached_ai_hazard_revision: int = -1
var _current_ai_character_ids: Array[String] = []
var _last_round_won: bool = false
var ai_profile: AIBehaviorProfile = DEFAULT_AI_PROFILE.duplicate(true) as AIBehaviorProfile

const AI_FORECAST_CACHE_MS: int = 60

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	_rng.randomize()
	_ensure_input_actions()
	settings = MatchSettings.load_from_disk()
	TranslationServer.set_locale(settings.language_code)
	DisplayServer.window_set_title(tr("森林泡泡染色战"))
	run_progress = RunProgress.new()
	_build_scene_tree()
	var web_adapter := WebPlatformAdapter.new()
	web_adapter.name = "WebPlatformAdapter"
	add_child(web_adapter)
	_connect_hud()
	hud.sync_settings(settings)
	if DisplayServer.get_name() == "headless":
		_begin_new_run()
		start_match()
	else:
		_enter_lobby()

func _physics_process(delta: float) -> void:
	if _round_over or _is_paused:
		return
	_simulation_time_ms += delta * 1000.0
	_spawn_due_items()
	_process_ai_schedule(delta)
	_remaining_seconds = maxf(0.0, _remaining_seconds - delta)
	_update_time_display()
	_resolve_explosion_hits()
	_resolve_actor_contacts()
	_resolve_item_pickups()
	if _remaining_seconds <= 0.0:
		_end_round()

func _process(_delta: float) -> void:
	var now_ms: int = Time.get_ticks_msec()
	if now_ms - _last_fps_update_ms < 250:
		return
	_last_fps_update_ms = now_ms
	if is_instance_valid(hud):
		var fps := Engine.get_frames_per_second()
		hud.update_fps(fps)
		_fps_label.text = "FPS: %d" % fps
		hud.update_player_stats(_player)

func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("pause_game") and not _round_over:
		if is_instance_valid(hud) and hud.is_settings_visible():
			_close_settings()
			get_viewport().set_input_as_handled()
			return
		if _is_paused:
			_resume_match()
		else:
			_pause_match()
		get_viewport().set_input_as_handled()
		return
	if app_state != AppState.MATCH or _round_over or _is_paused:
		return
	if event.is_action_pressed("zoom_in"):
		_arena_view.zoom_in()
		get_viewport().set_input_as_handled()
	elif event.is_action_pressed("zoom_out"):
		_arena_view.zoom_out()
		get_viewport().set_input_as_handled()
	elif event.is_action_pressed("zoom_reset"):
		_arena_view.reset_camera()
		get_viewport().set_input_as_handled()

func start_match(
		new_settings: MatchSettings = null,
		ai_character_ids: Array[String] = []
	) -> void:
	if new_settings != null:
		settings = new_settings
	settings.normalize()
	if run_progress == null or run_progress.ai_character_ids.is_empty():
		_begin_new_run()
	if not ai_character_ids.is_empty():
		run_progress.ai_character_ids = ai_character_ids.duplicate()
	_current_ai_character_ids = run_progress.ai_character_ids.duplicate()
	get_tree().paused = false
	_is_paused = false
	_round_over = false
	app_state = AppState.MATCH
	_remaining_seconds = GameConstants.ROUND_SECONDS
	_last_round_won = false
	_simulation_time_ms = 0.0
	_item_rng.seed = _rng.randi()
	_next_item_spawn_ms = int(GameConstants.ITEM_SPAWN_INTERVAL_SECONDS * 1000.0)
	_item_claims.clear()
	_displayed_seconds = -1
	_clear_match_nodes()
	board.configure_team_colors(run_progress.player_color_id, run_progress.ai_color_id)
	board.reset(MapCatalog.get_map())
	_arena_view.apply_camera_settings(settings)
	_arena_view.fit_camera(board.map_data)
	_spawn_fighters()
	_arena_view.visible = true
	hud.sync_settings(settings)
	hud.update_campaign(run_progress)
	hud.update_item_bonuses(_player)
	hud.show_match()
	hud.hide_pause()
	hud.hide_result()
	_update_time_display()
	_update_scoreboard()
	stage_changed.emit(run_progress.stage_number)
	_audio_call(&"play_sfx", [&"start"])
	_audio_call(&"play_music")


func _enter_lobby() -> void:
	get_tree().paused = false
	_round_over = true
	_is_paused = false
	app_state = AppState.LOBBY
	_clear_match_nodes()
	_current_ai_character_ids.clear()
	run_progress = RunProgress.new()
	_arena_view.visible = false
	hud.show_lobby()
	_audio_call(&"stop_music")


func _enter_setup() -> void:
	get_tree().paused = false
	_round_over = true
	_is_paused = false
	app_state = AppState.SETUP
	_arena_view.visible = false
	hud.show_setup(settings)


func _on_match_requested(configuration: Dictionary) -> void:
	settings.apply_dictionary(configuration)
	settings.save_to_disk()
	_begin_new_run()
	start_match(settings)


func _begin_new_run() -> void:
	run_progress = RunProgress.new()
	run_progress.begin(settings.character_id, settings.player_color_id, _rng)
	_current_ai_character_ids = run_progress.ai_character_ids.duplicate()


func _quit_game() -> void:
	get_tree().quit()

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
			bubble.cell,
			bubble.power,
			bubble.milliseconds_until_explosion(),
			owner_id,
			serial,
			bubble.owner_team
		))
		serial += 1
	for effect: ExplosionEffect in _active_explosions:
		if is_instance_valid(effect):
			var attacker_id: int = (
				effect.attacker.get_instance_id()
				if is_instance_valid(effect.attacker)
				else 0
			)
			snapshot.explosions.append(AIBattleSnapshot.ExplosionState.new(
				effect.cells,
				effect.milliseconds_remaining(),
				attacker_id,
				effect.attacker_team
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
			battle_actor.is_player,
			battle_actor.stats.stage_speed_items,
			battle_actor.stats.stage_bubble_items,
			battle_actor.stats.stage_power_items
		))
	for item: ArenaItemState in board.get_item_states():
		snapshot.items.append(AIBattleSnapshot.ItemState.new(
			item.item_id,
			item.item_type,
			item.cell,
			item.spawned_ms
		))
	snapshot.remaining_round_ms = maxi(0, int(_remaining_seconds * 1000.0))
	snapshot.paint_owners = MapCatalog.clone_matrix(board.paint_owners)
	for row: PackedByteArray in board.locked_cells:
		snapshot.locked_cells.append(row.duplicate())
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


func get_remaining_round_ms() -> int:
	return maxi(0, int(_remaining_seconds * 1000.0))

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
		GameConstants.BUBBLE_FUSE_SECONDS,
		initially_overlapping_actors
	)
	bubble.exploded.connect(_on_bubble_exploded)
	board.register_bubble(bubble)
	# Building the decorative 3D mesh is not part of the authoritative placement
	# transaction. Defer it so AI planning frames stay bounded under four actors.
	_arena_view.call_deferred("add_bubble", bubble)
	actor.stats.active_bubbles += 1
	_audio_call(&"play_sfx", [&"lay"])
	return true

func _build_scene_tree() -> void:
	_arena_view = ArenaView3D.new()
	_arena_view.name = "ArenaView3D"
	add_child(_arena_view)
	_world = Node2D.new()
	_world.name = "LogicWorld2D"
	_world.visible = false
	add_child(_world)
	board = GameBoard.new()
	board.name = "Board"
	_world.add_child(board)
	board.hazard_changed.connect(_invalidate_ai_forecast)
	board.territory_changed.connect(_on_territory_changed)
	board.item_collected.connect(_on_board_item_collected)
	_entity_root = Node2D.new()
	_entity_root.name = "Entities"
	_world.add_child(_entity_root)
	_effect_root = Node2D.new()
	_effect_root.name = "Effects"
	_world.add_child(_effect_root)
	_arena_view.bind_board(board)
	var canvas := CanvasLayer.new()
	canvas.name = "Interface"
	canvas.layer = 20
	add_child(canvas)
	hud = GameHud.new()
	hud.name = "HUD"
	canvas.add_child(hud)
	_arena_timer_label = hud.timer_label
	# Kept as a non-rendered compatibility probe for the existing smoke suite.
	_fps_label = Label.new()
	_fps_label.position = Vector2.ZERO
	_fps_label.text = "FPS: 0"
	_fps_label.visible = false
	canvas.add_child(_fps_label)

func _connect_hud() -> void:
	hud.setup_requested.connect(_enter_setup)
	hud.match_requested.connect(_on_match_requested)
	hud.resume_requested.connect(_resume_match)
	hud.restart_requested.connect(start_match)
	hud.retry_requested.connect(start_match)
	hud.skill_confirmed.connect(_advance_stage)
	hud.lobby_requested.connect(_enter_lobby)
	hud.quit_requested.connect(_quit_game)
	hud.zoom_in_requested.connect(_arena_view.zoom_in)
	hud.zoom_out_requested.connect(_arena_view.zoom_out)
	hud.zoom_reset_requested.connect(_arena_view.reset_zoom)
	hud.camera_reset_requested.connect(_arena_view.reset_camera)
	hud.camera_pose_requested.connect(_arena_view.set_camera_pose)
	hud.camera_adjustment_finished.connect(_arena_view.finish_camera_adjustment)
	hud.settings_open_requested.connect(_open_settings)
	hud.settings_close_requested.connect(_close_settings)
	hud.language_changed.connect(_on_language_changed)
	_arena_view.zoom_changed.connect(hud.update_zoom)
	_arena_view.camera_pose_changed.connect(hud.update_camera_pose)
	_arena_view.camera_adjustment_finished.connect(_on_camera_adjustment_finished)
	hud.update_zoom(_arena_view.get_zoom_percent())
	hud.update_camera_pose(
		_arena_view.get_azimuth(),
		_arena_view.get_elevation(),
		_arena_view.get_zoom()
	)

func _clear_match_nodes() -> void:
	for timer_value: Variant in _respawn_timers.values():
		var timer: Timer = timer_value as Timer
		if is_instance_valid(timer):
			timer.queue_free()
	_respawn_timers.clear()
	_active_explosions.clear()
	_active_unsafe_cells.clear()
	_active_explosion_sources.clear()
	_actors.clear()
	_ai_controllers.clear()
	_item_claims.clear()
	_ai_schedule_elapsed_seconds = 0.0
	_next_ai_index = 0
	_player = null
	_arena_view.clear_entities()
	for child: Node in _entity_root.get_children():
		child.queue_free()
	for child: Node in _effect_root.get_children():
		child.queue_free()

func _spawn_fighters() -> void:
	var used_cells: Array[Vector2i] = []
	var player_character := CharacterCatalog.get_definition(settings.character_id)
	_player = _spawn_actor(
		"玩家",
		PaintPalette.TEAM_PLAYER,
		true,
		board.map_data.player_spawn,
		player_character.id,
		run_progress.player_color_id
	)
	run_progress.apply_character_allocation(
		_player.stats,
		run_progress.player_allocation(),
		player_character.id
	)
	used_cells.append(board.map_data.player_spawn)
	for index: int in range(run_progress.ai_count()):
		var spawn: Vector2i = _find_ai_spawn(used_cells)
		used_cells.append(spawn)
		var ai_character_id: String = run_progress.ai_character_ids[index]
		var ai_actor: GameActor = _spawn_actor(
			"AI %d" % (index + 1),
			PaintPalette.TEAM_AI,
			false,
			spawn,
			ai_character_id,
			run_progress.ai_color_id
		)
		run_progress.apply_character_allocation(
			ai_actor.stats,
			run_progress.ai_allocation(index),
			ai_character_id
		)
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
		spawn_cell: Vector2i,
		character_id: String = "cat",
		color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID
	) -> GameActor:
	var actor := GameActor.new()
	actor.name = display_name
	_entity_root.add_child(actor)
	actor.setup(
		display_name,
		team_id,
		is_player_actor,
		board,
		settings,
		spawn_cell,
		color_id
	)
	actor.character_id = character_id
	actor.bomb_requested.connect(_on_bomb_requested)
	actor.died.connect(_on_actor_died)
	actor.trapped.connect(_on_actor_trapped)
	_actors.append(actor)
	_arena_view.add_actor(actor, character_id, color_id)
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
	board.paint_cells(blast, bubble.owner_team)
	var chained: Array[GameBubble] = []
	for cell: Vector2i in blast:
		if board.bombs.has(cell):
			var other: GameBubble = board.bombs[cell] as GameBubble
			if other != bubble and is_instance_valid(other) and other not in chained:
				chained.append(other)
	var effect := ExplosionEffect.new()
	_effect_root.add_child(effect)
	effect.setup(blast, bubble.cell, bubble.bubble_owner)
	_arena_view.add_explosion(effect)
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
		var left_attacker: GameActor = _harmful_attacker_for_cell(actor, feet[0])
		var right_attacker: GameActor = _harmful_attacker_for_cell(actor, feet[1])
		if is_instance_valid(left_attacker) and is_instance_valid(right_attacker):
			var attacker: GameActor = left_attacker
			if attacker.team_id == actor.team_id and right_attacker.team_id != actor.team_id:
				attacker = right_attacker
			actor.register_unsafe_frame(attacker)
		else:
			actor.register_safe_frame()

func _rebuild_active_explosion_lookup() -> void:
	_active_unsafe_cells.clear()
	_active_explosion_sources.clear()
	for effect: ExplosionEffect in _active_explosions:
		if not is_instance_valid(effect):
			continue
		for cell: Vector2i in effect.cells:
			_active_unsafe_cells[cell] = true
			var sources: Array = _active_explosion_sources.get(cell, []) as Array
			sources.append(effect)
			_active_explosion_sources[cell] = sources


func _harmful_attacker_for_cell(victim: GameActor, cell: Vector2i) -> GameActor:
	var sources: Array = _active_explosion_sources.get(cell, []) as Array
	var self_attacker: GameActor
	for value: Variant in sources:
		var effect: ExplosionEffect = value as ExplosionEffect
		if not is_instance_valid(effect) or not is_instance_valid(effect.attacker):
			continue
		if effect.attacker == victim:
			self_attacker = effect.attacker
			continue
		if effect.attacker_team != victim.team_id:
			return effect.attacker
	return self_attacker

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
	if left.team_id == right.team_id:
		return
	if left.stats.is_trapped and not right.stats.is_trapped:
		left.finish_by_touch(right)
	elif right.stats.is_trapped and not left.stats.is_trapped:
		right.finish_by_touch(left)

func _on_actor_died(victim: GameActor, defeating_team: int, attacker: GameActor) -> void:
	_audio_call(&"play_sfx", [&"die"])
	release_item_claims_for_actor(victim.get_instance_id())
	var has_opposing_defeater: bool = defeating_team in [
		PaintPalette.TEAM_PLAYER,
		PaintPalette.TEAM_AI,
	] and defeating_team != victim.team_id \
		and victim.was_finished_by_enemy_touch
	if has_opposing_defeater:
		var locked: Dictionary = board.lock_neighborhood(victim.current_cell(), defeating_team)
		_arena_view.play_defeat_burst(
			victim.current_cell(),
			_team_color_id(defeating_team),
			locked.keys()
		)
	else:
		_arena_view.play_defeat_burst(victim.current_cell(), victim.color_id, [])
	_update_scoreboard()
	_respawn_later(victim)


func _on_actor_trapped(victim: GameActor, _attacker: GameActor) -> void:
	release_item_claims_for_actor(victim.get_instance_id())

func _respawn_later(actor: GameActor) -> void:
	if not is_instance_valid(actor):
		return
	var actor_id: int = actor.get_instance_id()
	if _respawn_timers.has(actor_id):
		var previous: Timer = _respawn_timers[actor_id] as Timer
		if is_instance_valid(previous):
			previous.queue_free()
	var timer := Timer.new()
	timer.name = "RespawnTimer%d" % actor_id
	timer.one_shot = true
	timer.wait_time = GameConstants.RESPAWN_SECONDS
	timer.timeout.connect(_on_respawn_timeout.bind(actor, timer), CONNECT_ONE_SHOT)
	add_child(timer)
	_respawn_timers[actor_id] = timer
	timer.start()

func _on_respawn_timeout(actor: GameActor, timer: Timer) -> void:
	if is_instance_valid(actor):
		_respawn_timers.erase(actor.get_instance_id())
	if is_instance_valid(timer):
		timer.queue_free()
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

func _update_scoreboard() -> void:
	if board == null or board.paint_owners.is_empty() or run_progress == null:
		return
	var counts: Dictionary = board.get_territory_counts()
	var entries: Array[Dictionary] = [
		{
			"name": "玩家",
			"team_id": PaintPalette.TEAM_PLAYER,
			"color_id": run_progress.player_color_id,
			"cells": int(counts.get("player", 0)),
			"locked": int(counts.get("player_locked", 0)),
			"character_id": settings.character_id,
		},
		{
			"name": "AI 队",
			"team_id": PaintPalette.TEAM_AI,
			"color_id": run_progress.ai_color_id,
			"cells": int(counts.get("ai", 0)),
			"locked": int(counts.get("ai_locked", 0)),
			"character_id": (
				run_progress.ai_character_ids[0]
				if not run_progress.ai_character_ids.is_empty()
				else "cat"
			),
		},
	]
	entries.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return int(a.get("cells", 0)) > int(b.get("cells", 0))
	)
	hud.update_scores(entries)
	scores_changed.emit(entries)


func _on_territory_changed(_counts: Dictionary) -> void:
	_update_scoreboard()

func _end_round() -> void:
	if _round_over:
		return
	_round_over = true
	_is_paused = true
	app_state = AppState.RESULT
	_audio_call(&"stop_music")
	var counts: Dictionary = board.get_territory_counts()
	var player_cells: int = int(counts.get("player", 0))
	var ai_cells: int = int(counts.get("ai", 0))
	_last_round_won = player_cells > ai_cells
	for actor: GameActor in _actors:
		if is_instance_valid(actor):
			actor.stats.clear_stage_item_bonuses()
	hud.update_player_stats(_player)
	hud.update_item_bonuses(_player)
	if _last_round_won:
		hud.show_stage_result(
			true,
			run_progress,
			player_cells,
			ai_cells,
			int(counts.get("player_locked", 0)),
			int(counts.get("ai_locked", 0))
		)
		_audio_call(&"play_sfx", [&"win"])
	else:
		hud.show_stage_result(
			false,
			run_progress,
			player_cells,
			ai_cells,
			int(counts.get("player_locked", 0)),
			int(counts.get("ai_locked", 0))
		)
		_audio_call(&"play_sfx", [&"draw"])
	var results: Array[Dictionary] = [
		{
			"team_id": PaintPalette.TEAM_PLAYER,
			"color_id": run_progress.player_color_id,
			"cells": player_cells,
			"locked": int(counts.get("player_locked", 0)),
			"won": _last_round_won,
		},
		{
			"team_id": PaintPalette.TEAM_AI,
			"color_id": run_progress.ai_color_id,
			"cells": ai_cells,
			"locked": int(counts.get("ai_locked", 0)),
			"won": ai_cells > player_cells,
		},
	]
	match_finished.emit(results)
	get_tree().paused = true


func _advance_stage(skill_id: String) -> void:
	if not _round_over or not _last_round_won:
		return
	if run_progress.advance_with_skill(skill_id, _rng):
		start_match()


func _team_color_id(team_id: int) -> String:
	return (
		run_progress.player_color_id
		if team_id == PaintPalette.TEAM_PLAYER
		else run_progress.ai_color_id
	)

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


func _open_settings() -> void:
	if app_state == AppState.RESULT:
		return
	hud.show_settings()
	if app_state == AppState.MATCH and not _round_over:
		_is_paused = true
		get_tree().paused = true


func _close_settings() -> void:
	hud.hide_settings()
	if app_state != AppState.MATCH or _round_over:
		return
	get_tree().paused = false
	_is_paused = false


func _on_language_changed(language_code: String) -> void:
	settings.language_code = language_code
	settings.normalize()
	TranslationServer.set_locale(settings.language_code)
	DisplayServer.window_set_title(tr("森林泡泡染色战"))
	settings.save_to_disk()
	hud.sync_settings(settings)
	hud.refresh_localized_text(run_progress, _player)
	_update_scoreboard()

func _ensure_input_actions() -> void:
	_register_action(&"move_left", [KEY_LEFT, KEY_A])
	_register_action(&"move_right", [KEY_RIGHT, KEY_D])
	_register_action(&"move_up", [KEY_UP, KEY_W])
	_register_action(&"move_down", [KEY_DOWN, KEY_S])
	_register_action(&"place_bomb", [KEY_SPACE])
	_register_action(&"pause_game", [KEY_ESCAPE])
	_register_action(&"zoom_in", [KEY_EQUAL, KEY_PLUS, KEY_KP_ADD])
	_register_action(&"zoom_out", [KEY_MINUS, KEY_KP_SUBTRACT])
	_register_action(&"zoom_reset", [KEY_0, KEY_KP_0])
	_register_mouse_button_action(&"zoom_in", MOUSE_BUTTON_WHEEL_UP)
	_register_mouse_button_action(&"zoom_out", MOUSE_BUTTON_WHEEL_DOWN)

func _register_action(action: StringName, keycodes: Array) -> void:
	if InputMap.has_action(action):
		InputMap.erase_action(action)
	InputMap.add_action(action)
	for keycode: Key in keycodes:
		var event := InputEventKey.new()
		event.keycode = keycode
		InputMap.action_add_event(action, event)


func _register_mouse_button_action(action: StringName, button_index: MouseButton) -> void:
	var event := InputEventMouseButton.new()
	event.button_index = button_index
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
	hud.update_item_countdown(_seconds_until_next_item())
	time_changed.emit(total_seconds)

func _invalidate_ai_forecast() -> void:
	_ai_hazard_revision += 1
	_cached_ai_forecast = null


func _spawn_due_items() -> void:
	var interval_ms: int = int(GameConstants.ITEM_SPAWN_INTERVAL_SECONDS * 1000.0)
	var round_ms: int = int(GameConstants.ROUND_SECONDS * 1000.0)
	while _next_item_spawn_ms < round_ms \
			and int(_simulation_time_ms) >= _next_item_spawn_ms:
		for _item_index: int in range(GameConstants.ITEMS_PER_SPAWN):
			_spawn_random_item()
		_next_item_spawn_ms += interval_ms


func _spawn_random_item() -> int:
	var candidates: Array[Vector2i] = _item_spawn_candidates()
	if candidates.is_empty():
		return 0
	var cell: Vector2i = candidates[_item_rng.randi_range(0, candidates.size() - 1)]
	var item_type: int = ArenaItemType.ALL[
		_item_rng.randi_range(0, ArenaItemType.ALL.size() - 1)
	]
	return board.spawn_item(item_type, cell, int(_simulation_time_ms))


func _item_spawn_candidates() -> Array[Vector2i]:
	var occupied: Dictionary = {}
	for actor: GameActor in _actors:
		if not is_instance_valid(actor) or actor.stats.is_dead:
			continue
		for cell: Vector2i in GameRules.body_cells(actor.position):
			occupied[cell] = true
	var candidates: Array[Vector2i] = []
	for cell: Vector2i in board.get_open_cells():
		if board.bombs.has(cell) \
				or board.items_by_cell.has(cell) \
				or _active_unsafe_cells.has(cell) \
				or occupied.has(cell):
			continue
		candidates.append(cell)
	return candidates


func _resolve_item_pickups() -> void:
	for actor: GameActor in _actors:
		if not is_instance_valid(actor) or actor.stats.is_dead or actor.stats.is_trapped:
			continue
		var cell: Vector2i = actor.current_cell()
		if not board.items_by_cell.has(cell):
			continue
		var item: ArenaItemState = board.take_item(cell, actor.get_instance_id())
		if item == null or not actor.stats.apply_stage_item(item.item_type):
			continue
		release_item_claim(item.item_id)
		if actor == _player:
			hud.show_item_pickup(item.item_type)
			hud.update_item_bonuses(actor)
			_audio_call(&"play_sfx", [&"get"])


func _seconds_until_next_item() -> int:
	if _next_item_spawn_ms >= int(GameConstants.ROUND_SECONDS * 1000.0):
		return -1
	return maxi(
		0,
		ceili(float(_next_item_spawn_ms - int(_simulation_time_ms)) / 1000.0)
	)


func _on_board_item_collected(item: ArenaItemState, _actor_id: int) -> void:
	release_item_claim(item.item_id)


func _on_camera_adjustment_finished(
		_azimuth: float,
		_elevation: float,
		zoom: float
	) -> void:
	settings.camera_zoom = zoom
	settings.camera_azimuth = MatchSettings.DEFAULT_CAMERA_AZIMUTH
	settings.camera_elevation = MatchSettings.DEFAULT_CAMERA_ELEVATION
	settings.save_to_disk()


func can_claim_item(actor_id: int, item_id: int, travel_ms: int) -> bool:
	_prune_item_claims()
	var claim: Dictionary = _item_claims.get(item_id, {}) as Dictionary
	if claim.is_empty() or int(claim.get("actor_id", 0)) == actor_id:
		return true
	var existing_remaining: int = maxi(
		0,
		int(claim.get("arrival_ms", 0)) - get_simulation_time_ms()
	)
	return travel_ms + ai_profile.item_claim_steal_advantage_ms < existing_remaining


func claim_item(actor_id: int, item_id: int, travel_ms: int) -> bool:
	if not can_claim_item(actor_id, item_id, travel_ms):
		return false
	var now_ms: int = get_simulation_time_ms()
	var lifetime: int = mini(
		travel_ms + ai_profile.item_claim_grace_ms,
		ai_profile.maximum_item_claim_ms
	)
	_item_claims[item_id] = {
		"actor_id": actor_id,
		"arrival_ms": now_ms + travel_ms,
		"expires_ms": now_ms + maxi(ai_profile.item_claim_grace_ms, lifetime),
	}
	return true


func release_item_claim(item_id: int, actor_id: int = 0) -> void:
	if actor_id != 0:
		var claim: Dictionary = _item_claims.get(item_id, {}) as Dictionary
		if int(claim.get("actor_id", 0)) != actor_id:
			return
	_item_claims.erase(item_id)


func release_item_claims_for_actor(actor_id: int) -> void:
	for item_id: Variant in _item_claims.keys():
		var claim: Dictionary = _item_claims[item_id] as Dictionary
		if int(claim.get("actor_id", 0)) == actor_id:
			_item_claims.erase(item_id)


func _prune_item_claims() -> void:
	if _item_claims.is_empty():
		return
	var now_ms: int = get_simulation_time_ms()
	for item_id: Variant in _item_claims.keys():
		var claim: Dictionary = _item_claims[item_id] as Dictionary
		var actor_id: int = int(claim.get("actor_id", 0))
		var actor: GameActor = _actor_by_id(actor_id)
		if now_ms >= int(claim.get("expires_ms", 0)) \
				or board.item_by_id(int(item_id)) == null \
				or not is_instance_valid(actor) \
				or actor.stats.is_dead \
				or actor.stats.is_trapped:
			_item_claims.erase(item_id)


func _actor_by_id(actor_id: int) -> GameActor:
	for actor: GameActor in _actors:
		if is_instance_valid(actor) and actor.get_instance_id() == actor_id:
			return actor
	return null

func _audio_call(method: StringName, arguments: Array = []) -> void:
	if "--mute" in OS.get_cmdline_user_args():
		return
	var audio_manager: Node = get_node_or_null("/root/AudioManager")
	if is_instance_valid(audio_manager):
		audio_manager.callv(method, arguments)
