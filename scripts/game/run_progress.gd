class_name RunProgress
extends RefCounted
## In-memory progression for one infinite paint campaign.

const SKILL_SPEED: String = "speed"
const SKILL_BUBBLE: String = "bubble"
const SKILL_POWER: String = "power"
const SKILL_IDS: Array[String] = [SKILL_SPEED, SKILL_BUBBLE, SKILL_POWER]

var stage_number: int = 1
var speed_points: int = 0
var bubble_points: int = 0
var power_points: int = 0
var player_character_id: String = "cat"
var player_color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID
var ai_color_id: String = "blue"
var ai_character_ids: Array[String] = []
var ai_allocations: Array[Dictionary] = []


func begin(
		player_character_id: String,
		selected_color_id: String,
		rng: RandomNumberGenerator
	) -> void:
	stage_number = 1
	speed_points = 0
	bubble_points = 0
	power_points = 0
	self.player_character_id = CharacterCatalog.migrate_legacy_id(
		player_character_id
	)
	player_color_id = (
		selected_color_id
		if PaintPalette.is_valid_color_id(selected_color_id)
		else PaintPalette.DEFAULT_PLAYER_COLOR_ID
	)
	ai_color_id = PaintPalette.random_opponent_color(player_color_id, rng)
	ai_character_ids = CharacterCatalog.assign_ai_characters(player_character_id, 4, rng)
	_roll_ai_allocations(rng)


func ai_count() -> int:
	return mini(stage_number, 4)


func total_skill_points() -> int:
	return speed_points + bubble_points + power_points


func player_allocation() -> Dictionary:
	return {
		SKILL_SPEED: speed_points,
		SKILL_BUBBLE: bubble_points,
		SKILL_POWER: power_points,
	}


func ai_allocation(index: int) -> Dictionary:
	if index < 0 or index >= ai_allocations.size():
		return _empty_allocation()
	return (ai_allocations[index] as Dictionary).duplicate()


func advance_with_skill(skill_id: String, rng: RandomNumberGenerator) -> bool:
	if skill_id not in SKILL_IDS:
		return false
	match skill_id:
		SKILL_SPEED:
			speed_points = mini(
				speed_points + 1,
				maximum_skill_points(SKILL_SPEED)
			)
		SKILL_BUBBLE:
			bubble_points = mini(
				bubble_points + 1,
				maximum_skill_points(SKILL_BUBBLE)
			)
		SKILL_POWER:
			power_points = mini(
				power_points + 1,
				maximum_skill_points(SKILL_POWER)
			)
	stage_number += 1
	_roll_ai_allocations(rng)
	return true


func apply_allocation(stats: ActorStats, allocation: Dictionary) -> void:
	apply_character_allocation(stats, allocation, player_character_id)


func apply_character_allocation(
		stats: ActorStats,
		allocation: Dictionary,
		character_id: String
	) -> void:
	var definition := CharacterCatalog.get_definition(character_id)
	stats.apply_character_skill_points(
		int(allocation.get(SKILL_SPEED, 0)),
		int(allocation.get(SKILL_BUBBLE, 0)),
		int(allocation.get(SKILL_POWER, 0)),
		definition.initial_speed_points,
		definition.initial_bubble_points,
		definition.initial_power_points
	)


func maximum_skill_points(skill_id: String, character_id: String = "") -> int:
	var resolved_id := (
		player_character_id
		if character_id.is_empty()
		else CharacterCatalog.migrate_legacy_id(character_id)
	)
	var definition := CharacterCatalog.get_definition(resolved_id)
	match skill_id:
		SKILL_SPEED:
			return GameConstants.MAX_SPEED_POINTS - definition.initial_speed_points
		SKILL_BUBBLE:
			return GameConstants.MAX_BUBBLES - definition.initial_bubble_points
		SKILL_POWER:
			return GameConstants.MAX_POWER - definition.initial_power_points
		_:
			return 0


func _roll_ai_allocations(rng: RandomNumberGenerator) -> void:
	ai_allocations.clear()
	for _index: int in range(ai_count()):
		var allocation: Dictionary = _empty_allocation()
		for _point: int in range(total_skill_points()):
			var skill_id: String = SKILL_IDS[rng.randi_range(0, SKILL_IDS.size() - 1)]
			allocation[skill_id] = int(allocation[skill_id]) + 1
		ai_allocations.append(allocation)


func _empty_allocation() -> Dictionary:
	return {
		SKILL_SPEED: 0,
		SKILL_BUBBLE: 0,
		SKILL_POWER: 0,
	}
