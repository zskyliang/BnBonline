class_name ActorStats
extends Resource
## Mutable combat stats owned by one actor.

var move_speed: float = GameConstants.INITIAL_SPEED:
	set(value):
		move_speed = clampf(value, 0.0, GameConstants.MAX_SPEED)
var bubble_capacity: int = GameConstants.INITIAL_BUBBLES:
	set(value):
		bubble_capacity = clampi(value, 0, GameConstants.MAX_BUBBLES)
var power: int = GameConstants.INITIAL_POWER:
	set(value):
		power = clampi(value, 0, GameConstants.MAX_POWER)
var active_bubbles: int = 0
var is_dead: bool = false
var is_trapped: bool = false
var invincible_until_ms: int = 0
var stage_speed_items: int = 0
var stage_bubble_items: int = 0
var stage_power_items: int = 0

var _campaign_move_speed: float = GameConstants.INITIAL_SPEED
var _campaign_bubble_capacity: int = GameConstants.INITIAL_BUBBLES
var _campaign_power: int = GameConstants.INITIAL_POWER

func reset_for_match() -> void:
	_campaign_move_speed = GameConstants.INITIAL_SPEED
	_campaign_bubble_capacity = GameConstants.INITIAL_BUBBLES
	_campaign_power = GameConstants.INITIAL_POWER
	stage_speed_items = 0
	stage_bubble_items = 0
	stage_power_items = 0
	_recalculate_effective_stats()
	active_bubbles = 0
	is_dead = false
	is_trapped = false
	invincible_until_ms = 0


func apply_skill_points(speed_points: int, bubble_points: int, power_points: int) -> void:
	apply_character_skill_points(
		speed_points,
		bubble_points,
		power_points,
		GameConstants.DEFAULT_INITIAL_SPEED_POINTS,
		GameConstants.DEFAULT_INITIAL_BUBBLE_POINTS,
		GameConstants.DEFAULT_INITIAL_POWER_POINTS
	)


func apply_character_skill_points(
		speed_points: int,
		bubble_points: int,
		power_points: int,
		initial_speed_points: int,
		initial_bubble_points: int,
		initial_power_points: int
	) -> void:
	_campaign_move_speed = minf(
		GameConstants.speed_from_points(
			maxi(0, initial_speed_points) + maxi(0, speed_points)
		),
		GameConstants.MAX_SPEED
	)
	_campaign_bubble_capacity = mini(
		maxi(0, initial_bubble_points) + maxi(0, bubble_points),
		GameConstants.MAX_BUBBLES
	)
	_campaign_power = mini(
		maxi(0, initial_power_points) + maxi(0, power_points),
		GameConstants.MAX_POWER
	)
	_recalculate_effective_stats()


func apply_stage_item(item_type: int) -> bool:
	if is_at_item_cap(item_type):
		return false
	match item_type:
		ArenaItemType.Value.SPEED:
			stage_speed_items += 1
		ArenaItemType.Value.BUBBLE:
			stage_bubble_items += 1
		ArenaItemType.Value.POWER:
			stage_power_items += 1
		_:
			return false
	_recalculate_effective_stats()
	return true


func is_at_item_cap(item_type: int) -> bool:
	match item_type:
		ArenaItemType.Value.SPEED:
			return move_speed >= GameConstants.MAX_SPEED
		ArenaItemType.Value.BUBBLE:
			return bubble_capacity >= GameConstants.MAX_BUBBLES
		ArenaItemType.Value.POWER:
			return power >= GameConstants.MAX_POWER
		_:
			return true


func clear_stage_item_bonuses() -> void:
	stage_speed_items = 0
	stage_bubble_items = 0
	stage_power_items = 0
	_recalculate_effective_stats()


func stage_item_counts() -> Dictionary:
	return {
		"speed": stage_speed_items,
		"bubble": stage_bubble_items,
		"power": stage_power_items,
	}


func speed_points() -> int:
	return GameConstants.speed_points_from_pixels(move_speed)


func attribute_points() -> Dictionary:
	return {
		"speed": speed_points(),
		"bubble": bubble_capacity,
		"power": power,
	}


func _recalculate_effective_stats() -> void:
	move_speed = minf(
		_campaign_move_speed
			+ float(stage_speed_items) * GameConstants.SPEED_PER_STAGE_ITEM,
		GameConstants.MAX_SPEED
	)
	bubble_capacity = mini(
		_campaign_bubble_capacity + stage_bubble_items,
		GameConstants.MAX_BUBBLES
	)
	power = mini(
		_campaign_power + stage_power_items,
		GameConstants.MAX_POWER
	)

func is_invincible() -> bool:
	return Time.get_ticks_msec() < invincible_until_ms
