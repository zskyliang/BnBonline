class_name GameActor
extends CharacterBody2D
## Player or AI-controlled arena actor.

signal bomb_requested(actor: GameActor)
signal died(victim: GameActor, attacker: GameActor)
signal trapped(actor: GameActor, attacker: GameActor)
signal rescued(actor: GameActor)
signal item_collected(actor: GameActor, item_code: int)

const COLLISION_SWEEP_ITERATIONS: int = 10

enum Facing { UP, DOWN, LEFT, RIGHT }

var actor_name: String = "Player"
var team_id: int = 1
var is_player: bool = false
var stats := ActorStats.new()
var board: GameBoard
var settings: MatchSettings
var last_attacker: GameActor
var unsafe_frame_count: int = 0
var character_id: String = "builder"

var _desired_direction: Vector2 = Vector2.ZERO
var _facing: Facing = Facing.DOWN
var _trap_timer: Timer
var _death_time: float = 0.0

func setup(
		new_name: String,
		new_team_id: int,
		player_controlled: bool,
		new_board: GameBoard,
		new_settings: MatchSettings,
		spawn_cell: Vector2i
	) -> void:
	actor_name = new_name
	team_id = new_team_id
	is_player = player_controlled
	board = new_board
	settings = new_settings
	stats.reset_for_match()
	position = GameConstants.grid_to_world(spawn_cell)
	collision_layer = 0
	collision_mask = 0
	_build_visuals()

func _physics_process(delta: float) -> void:
	if stats.is_dead or stats.is_trapped:
		_desired_direction = Vector2.ZERO
		return
	if is_player:
		var focused: Control = get_viewport().gui_get_focus_owner()
		if focused is LineEdit:
			_desired_direction = Vector2.ZERO
		else:
			_desired_direction = Input.get_vector("move_left", "move_right", "move_up", "move_down")
	_desired_direction = _cardinalize(_desired_direction)
	velocity = _desired_direction * stats.move_speed
	_attempt_move(delta)
	_try_pickup()

func _process(delta: float) -> void:
	if stats.is_dead and not visible:
		return
	z_index = 40 + int(position.y)
	if stats.is_dead:
		_death_time += delta
		if _death_time >= 2.2:
			visible = false

func _unhandled_input(event: InputEvent) -> void:
	if not is_player or stats.is_dead:
		return
	if event.is_action_pressed("place_bomb"):
		bomb_requested.emit(self)
		get_viewport().set_input_as_handled()
	elif event.is_action_pressed("self_rescue") and stats.is_trapped:
		rescue()
		get_viewport().set_input_as_handled()

func set_ai_direction(direction: Vector2) -> void:
	if not is_player:
		_desired_direction = _cardinalize(direction)

func request_ai_bomb() -> void:
	if not is_player and not stats.is_dead and not stats.is_trapped:
		bomb_requested.emit(self)

func current_cell() -> Vector2i:
	return GameConstants.world_to_grid(position)

func foot_cells() -> Array[Vector2i]:
	return GameRules.foot_cells(position)

func apply_item(item_code: int) -> void:
	match item_code:
		GameConstants.ITEM_BUBBLE:
			stats.bubble_capacity = mini(settings.max_bubbles, stats.bubble_capacity + 1)
		GameConstants.ITEM_SPEED:
			stats.move_speed = minf(float(settings.max_speed), stats.move_speed + 25.0)
		GameConstants.ITEM_POWER:
			stats.power = mini(settings.max_power, stats.power + 1)
	item_collected.emit(self, item_code)

func clamp_stats() -> void:
	stats.clamp_to(settings)

func register_unsafe_frame(attacker: GameActor) -> void:
	if stats.is_dead or stats.is_trapped or stats.is_invincible():
		unsafe_frame_count = 0
		return
	unsafe_frame_count += 1
	if is_instance_valid(attacker):
		last_attacker = attacker
	if unsafe_frame_count >= 2:
		trap(last_attacker)

func register_safe_frame() -> void:
	unsafe_frame_count = 0

func trap(attacker: GameActor) -> void:
	if stats.is_dead or stats.is_trapped or stats.is_invincible():
		return
	last_attacker = attacker
	stats.is_trapped = true
	velocity = Vector2.ZERO
	_desired_direction = Vector2.ZERO
	unsafe_frame_count = 0
	_trap_timer = Timer.new()
	_trap_timer.one_shot = true
	_trap_timer.wait_time = GameConstants.TRAP_SECONDS
	_trap_timer.timeout.connect(_on_trap_timeout)
	add_child(_trap_timer)
	_trap_timer.start()
	trapped.emit(self, attacker)

func rescue() -> void:
	if stats.is_dead or not stats.is_trapped:
		return
	stats.is_trapped = false
	_clear_trap_visual()
	rescued.emit(self)

func finish_by_touch(attacker: GameActor) -> void:
	if stats.is_trapped and not stats.is_dead:
		last_attacker = attacker
		die(attacker)

func die(attacker: GameActor) -> void:
	if stats.is_dead:
		return
	stats.is_dead = true
	stats.is_trapped = false
	last_attacker = attacker
	_clear_trap_visual()
	_death_time = 0.0
	died.emit(self, attacker)

func respawn(spawn_cell: Vector2i) -> void:
	stats.is_dead = false
	stats.is_trapped = false
	stats.active_bubbles = 0
	stats.invincible_until_ms = Time.get_ticks_msec() + int(GameConstants.RESPAWN_INVINCIBLE_SECONDS * 1000.0)
	last_attacker = null
	unsafe_frame_count = 0
	position = GameConstants.grid_to_world(spawn_cell)
	reset_physics_interpolation()
	visible = true
	_facing = Facing.DOWN

func _attempt_move(delta: float) -> void:
	if velocity == Vector2.ZERO:
		return
	_update_facing(velocity)
	var offset: Vector2 = velocity * delta
	_move_on_axis(Vector2(offset.x, 0.0))
	_move_on_axis(Vector2(0.0, offset.y))

func _move_on_axis(motion: Vector2) -> void:
	if motion == Vector2.ZERO:
		return
	var target_position: Vector2 = position + motion
	if board.can_actor_move(position, target_position, self):
		position = target_position
		return
	var safe_fraction: float = 0.0
	var blocked_fraction: float = 1.0
	for _iteration: int in range(COLLISION_SWEEP_ITERATIONS):
		var candidate_fraction: float = (safe_fraction + blocked_fraction) * 0.5
		var candidate_position: Vector2 = position + motion * candidate_fraction
		if board.can_actor_move(position, candidate_position, self):
			safe_fraction = candidate_fraction
		else:
			blocked_fraction = candidate_fraction
	if safe_fraction > 0.0:
		position += motion * safe_fraction

func _try_pickup() -> void:
	var item_code: int = board.take_item(current_cell())
	if item_code > 0:
		apply_item(item_code)

func _build_visuals() -> void:
	var shape := CollisionShape2D.new()
	var rectangle := RectangleShape2D.new()
	rectangle.size = Vector2(
		GameRules.BODY_HALF_WIDTH * 2.0,
		GameRules.BODY_BOTTOM_OFFSET - GameRules.BODY_TOP_OFFSET
	)
	shape.shape = rectangle
	shape.position.y = (GameRules.BODY_TOP_OFFSET + GameRules.BODY_BOTTOM_OFFSET) * 0.5
	add_child(shape)

func _clear_trap_visual() -> void:
	if is_instance_valid(_trap_timer):
		_trap_timer.stop()
		_trap_timer.queue_free()
	_trap_timer = null

func _on_trap_timeout() -> void:
	die(last_attacker)

func _update_facing(direction: Vector2) -> void:
	if absf(direction.x) > absf(direction.y):
		_facing = Facing.RIGHT if direction.x > 0 else Facing.LEFT
	elif direction.y != 0:
		_facing = Facing.DOWN if direction.y > 0 else Facing.UP

func _cardinalize(direction: Vector2) -> Vector2:
	if direction == Vector2.ZERO:
		return Vector2.ZERO
	if absf(direction.x) >= absf(direction.y):
		return Vector2(signf(direction.x), 0)
	return Vector2(0, signf(direction.y))
