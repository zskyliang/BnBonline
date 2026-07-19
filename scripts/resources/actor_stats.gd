class_name ActorStats
extends Resource
## Mutable combat stats owned by one actor.

var move_speed: float = GameConstants.INITIAL_SPEED
var bubble_capacity: int = GameConstants.INITIAL_BUBBLES
var power: int = GameConstants.INITIAL_POWER
var active_bubbles: int = 0
var is_dead: bool = false
var is_trapped: bool = false
var invincible_until_ms: int = 0

func reset_for_match() -> void:
	move_speed = GameConstants.INITIAL_SPEED
	bubble_capacity = GameConstants.INITIAL_BUBBLES
	power = GameConstants.INITIAL_POWER
	active_bubbles = 0
	is_dead = false
	is_trapped = false
	invincible_until_ms = 0

func clamp_to(settings: MatchSettings) -> void:
	move_speed = minf(move_speed, float(settings.max_speed))
	bubble_capacity = mini(bubble_capacity, settings.max_bubbles)
	power = mini(power, settings.max_power)

func is_invincible() -> bool:
	return Time.get_ticks_msec() < invincible_until_ms

