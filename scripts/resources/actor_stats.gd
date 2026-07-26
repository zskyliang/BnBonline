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
	_campaign_move_speed = GameConstants.INITIAL_SPEED \
		+ float(maxi(0, speed_points)) * GameConstants.SPEED_PER_SKILL_POINT
	_campaign_bubble_capacity = GameConstants.INITIAL_BUBBLES + maxi(0, bubble_points)
	_campaign_power = GameConstants.INITIAL_POWER + maxi(0, power_points)
	_recalculate_effective_stats()


func apply_stage_item(item_type: int) -> bool:
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


func _recalculate_effective_stats() -> void:
	move_speed = _campaign_move_speed \
		+ float(stage_speed_items) * GameConstants.SPEED_PER_STAGE_ITEM
	bubble_capacity = _campaign_bubble_capacity + stage_bubble_items
	power = _campaign_power + stage_power_items

func is_invincible() -> bool:
	return Time.get_ticks_msec() < invincible_until_ms
