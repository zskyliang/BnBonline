class_name GameBubble
extends Node2D
## Grid-snapped bubble logic with an owned fuse.

signal exploded(bubble: GameBubble)

var bubble_owner: GameActor
var cell: Vector2i
var power: int = 2
var explode_at_ms: int = 0
var has_exploded: bool = false
var skin: String = "aqua"

var _fuse_timer: Timer
var _exit_actor_ids: Dictionary = {}

func setup(
		new_owner: GameActor,
		new_cell: Vector2i,
		skin: String,
		fuse_seconds: float = GameConstants.BUBBLE_FUSE_SECONDS,
		initially_overlapping_actors: Array[GameActor] = []
	) -> void:
	bubble_owner = new_owner
	cell = new_cell
	self.skin = skin
	power = bubble_owner.stats.power
	_exit_actor_ids.clear()
	if is_instance_valid(bubble_owner):
		_exit_actor_ids[bubble_owner.get_instance_id()] = true
	for overlapping_actor: GameActor in initially_overlapping_actors:
		if is_instance_valid(overlapping_actor):
			_exit_actor_ids[overlapping_actor.get_instance_id()] = true
	position = GameConstants.grid_to_world(cell)
	_fuse_timer = Timer.new()
	_fuse_timer.one_shot = true
	_fuse_timer.wait_time = maxf(0.01, fuse_seconds)
	_fuse_timer.timeout.connect(explode_now)
	add_child(_fuse_timer)
	_fuse_timer.start()
	explode_at_ms = Time.get_ticks_msec() + int(_fuse_timer.wait_time * 1000.0)

func explode_now() -> void:
	if has_exploded:
		return
	has_exploded = true
	if is_instance_valid(_fuse_timer):
		_fuse_timer.stop()
	exploded.emit(self)
	queue_free()

func milliseconds_until_explosion() -> int:
	if has_exploded:
		return 0
	if is_instance_valid(_fuse_timer):
		return maxi(0, ceili(_fuse_timer.time_left * 1000.0))
	return maxi(0, explode_at_ms - Time.get_ticks_msec())

func can_actor_finish_exiting(
		actor: GameActor,
		actor_current_cells: Array[Vector2i]
	) -> bool:
	if not is_instance_valid(actor) or not _exit_actor_ids.has(actor.get_instance_id()):
		return false
	return cell in actor_current_cells
