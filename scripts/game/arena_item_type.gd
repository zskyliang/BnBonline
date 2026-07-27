class_name ArenaItemType
extends RefCounted
## Stable identifiers and presentation metadata for temporary stage pickups.

enum Value { SPEED, BUBBLE, POWER }

const ALL: Array[Value] = [Value.SPEED, Value.BUBBLE, Value.POWER]
const TEXTURE_ROOT: String = "res://assets/art/storybook25d/items/"


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
			return "速度 +1"
		Value.BUBBLE:
			return "水泡数 +1"
		Value.POWER:
			return "威力 +1"
		_:
			return ""


static func texture_path(item_type: int) -> String:
	match item_type:
		Value.SPEED:
			return TEXTURE_ROOT + "leaf_shoes.png"
		Value.BUBBLE:
			return TEXTURE_ROOT + "bubble_gourd.png"
		Value.POWER:
			return TEXTURE_ROOT + "paw_burst.png"
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
