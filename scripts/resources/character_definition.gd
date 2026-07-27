class_name CharacterDefinition
extends Resource
## ImageGen identity and deterministic six-point starting attributes.

@export var id: String = "cat"
@export var display_name: String = "猫"
@export_dir var sprite_set_path: String = ""
@export var sprite_pixel_size: float = 0.00265
@export var theme_color: Color = Color.WHITE
@export var accent_color: Color = Color.WHITE
@export var source_url: String = ""
@export_range(1, 4, 1) var initial_speed_points: int = 2
@export_range(1, 4, 1) var initial_bubble_points: int = 2
@export_range(1, 4, 1) var initial_power_points: int = 2

var _sprite_set: CharacterSpriteSet


func configure(
		character_id: String,
		chinese_name: String,
		sprite_path: String,
		color: Color,
		accent: Color,
		starting_speed_points: int,
		starting_bubble_points: int,
		starting_power_points: int,
		pixel_size: float = 0.00265,
		asset_source_url: String = ""
	) -> CharacterDefinition:
	id = character_id
	display_name = chinese_name
	sprite_set_path = sprite_path
	sprite_pixel_size = pixel_size
	theme_color = color
	accent_color = accent
	source_url = asset_source_url
	initial_speed_points = starting_speed_points
	initial_bubble_points = starting_bubble_points
	initial_power_points = starting_power_points
	return self


func initial_attributes() -> Dictionary:
	return {
		"speed": initial_speed_points,
		"bubble": initial_bubble_points,
		"power": initial_power_points,
	}


func initial_attribute_total() -> int:
	return initial_speed_points + initial_bubble_points + initial_power_points


func load_sprite_set() -> CharacterSpriteSet:
	if _sprite_set == null:
		_sprite_set = CharacterSpriteSet.new().configure(
			id,
			sprite_set_path,
			sprite_pixel_size
		)
	return _sprite_set
