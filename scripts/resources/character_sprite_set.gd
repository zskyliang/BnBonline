class_name CharacterSpriteSet
extends Resource
## Runtime texture contract for one four-direction ImageGen animal.

const ACTIONS: Array[StringName] = [
	&"Idle",
	&"WalkUp",
	&"WalkDown",
	&"WalkLeft",
	&"WalkRight",
	&"Trapped",
]
const WALK_ACTIONS: Array[StringName] = [
	&"WalkUp",
	&"WalkDown",
	&"WalkLeft",
	&"WalkRight",
]
const DIRECTIONS: Array[StringName] = [
	&"down",
	&"up",
	&"left",
	&"right",
]
const ACTION_DIRECTIONS: Dictionary = {
	&"WalkUp": &"up",
	&"WalkDown": &"down",
	&"WalkLeft": &"left",
	&"WalkRight": &"right",
}
const WALK_FRAME_COUNT := 4

@export var character_id: String = "cat"
@export_dir var root_path: String = ""
@export var pixel_size: float = 0.00265
@export var ground_offset: float = 0.64

var _textures: Dictionary = {}
var _masks: Dictionary = {}


func configure(
		new_character_id: String,
		new_root_path: String,
		new_pixel_size: float = 0.00265
	) -> CharacterSpriteSet:
	character_id = new_character_id
	root_path = new_root_path.trim_suffix("/")
	pixel_size = new_pixel_size
	return self


func texture_for(
		action: StringName,
		frame_index: int = 0,
		facing: StringName = &"down"
	) -> Texture2D:
	return _load_frame(action, frame_index, facing, false)


func mask_for(
		action: StringName,
		frame_index: int = 0,
		facing: StringName = &"down"
	) -> Texture2D:
	return _load_frame(action, frame_index, facing, true)


func has_action(action: StringName) -> bool:
	if action not in ACTIONS:
		return false
	return texture_for(action) != null and mask_for(action) != null


func frame_count(action: StringName) -> int:
	return WALK_FRAME_COUNT if action in WALK_ACTIONS else 1


func direction_for_action(action: StringName) -> StringName:
	return ACTION_DIRECTIONS.get(action, &"down") as StringName


func texture_path_for(
		action: StringName,
		frame_index: int = 0,
		facing: StringName = &"down"
	) -> String:
	return "%s/%s.png" % [
		root_path,
		_frame_stem(action, frame_index, facing),
	]


func mask_path_for(
		action: StringName,
		frame_index: int = 0,
		facing: StringName = &"down"
	) -> String:
	return "%s/%s_mask.png" % [
		root_path,
		_frame_stem(action, frame_index, facing),
	]


func _load_frame(
		action: StringName,
		frame_index: int,
		facing: StringName,
		load_mask: bool
	) -> Texture2D:
	var normalized_facing := _normalized_facing(facing)
	var normalized_frame := posmod(frame_index, frame_count(action))
	var cache_key := "%s|%s|%d|%s" % [
		String(action),
		String(normalized_facing),
		normalized_frame,
		"mask" if load_mask else "base",
	]
	var cache := _masks if load_mask else _textures
	if cache.has(cache_key):
		return cache[cache_key] as Texture2D
	var path := (
		mask_path_for(action, normalized_frame, normalized_facing)
		if load_mask
		else texture_path_for(action, normalized_frame, normalized_facing)
	)
	var texture := (
		load(path) as Texture2D
		if ResourceLoader.exists(path)
		else null
	)
	cache[cache_key] = texture
	return texture


func _frame_stem(
		action: StringName,
		frame_index: int,
		facing: StringName
	) -> String:
	match action:
		&"Idle":
			return "idle_%s" % String(_normalized_facing(facing))
		&"WalkUp", &"WalkDown", &"WalkLeft", &"WalkRight":
			return "walk_%s_%d" % [
				String(direction_for_action(action)),
				posmod(frame_index, WALK_FRAME_COUNT),
			]
		&"Trapped":
			return "trapped"
		_:
			return "idle_down"


func _normalized_facing(facing: StringName) -> StringName:
	return facing if facing in DIRECTIONS else &"down"
