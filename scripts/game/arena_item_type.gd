class_name ArenaItemType
extends RefCounted
## Stable identifiers and presentation metadata for temporary stage pickups.

enum Value { SPEED, BUBBLE, POWER }

const ALL: Array[Value] = [Value.SPEED, Value.BUBBLE, Value.POWER]
const MODEL_ROOT: String = "res://assets/models/items/kenney_platformer/"


static func is_valid(item_type: int) -> bool:
	return item_type >= Value.SPEED and item_type <= Value.POWER


static func display_name(item_type: int) -> String:
	match item_type:
		Value.SPEED:
			return "速度"
		Value.BUBBLE:
			return "水泡数"
		Value.POWER:
			return "威力"
		_:
			return "未知道具"


static func short_bonus(item_type: int) -> String:
	match item_type:
		Value.SPEED:
			return "速度 +25"
		Value.BUBBLE:
			return "水泡数 +1"
		Value.POWER:
			return "威力 +1"
		_:
			return ""


static func model_path(item_type: int) -> String:
	match item_type:
		Value.SPEED:
			return MODEL_ROOT + "spring.glb"
		Value.BUBBLE:
			return MODEL_ROOT + "bomb.glb"
		Value.POWER:
			return MODEL_ROOT + "star.glb"
		_:
			return ""


static func accent_color(item_type: int) -> Color:
	match item_type:
		Value.SPEED:
			return Color("#39bfc3")
		Value.BUBBLE:
			return Color("#4b72cc")
		Value.POWER:
			return Color("#d94f45")
		_:
			return Color.WHITE
