class_name CharacterDefinition
extends Resource
## Cosmetic-only definition for one storybook animal character.

@export var id: String = "cat"
@export var display_name: String = "猫"
@export_file("*.gltf", "*.glb", "*.fbx") var model_path: String = ""
@export var scale_multiplier: float = 1.0
@export var yaw_offset_degrees: float = 0.0
@export var head_bone_aliases: Array[String] = []
@export var idle_animation_aliases: Array[String] = []
@export var move_animation_aliases: Array[String] = []
@export var place_bubble_animation_aliases: Array[String] = []
@export var trapped_animation_aliases: Array[String] = []
@export var defeat_animation_aliases: Array[String] = []
@export var victory_animation_aliases: Array[String] = []
@export var theme_color: Color = Color.WHITE
@export var accent_color: Color = Color.WHITE
@export var source_url: String = ""
@export var team_tint_material_names: Array[String] = []


func configure(
		character_id: String,
		chinese_name: String,
		scene_path: String,
		color: Color,
		accent: Color,
		idle_aliases: Array[String],
		move_aliases: Array[String],
		yaw_offset: float = 0.0,
		model_scale: float = 1.0,
		asset_source_url: String = "",
		bone_aliases: Array[String] = []
	) -> CharacterDefinition:
	id = character_id
	display_name = chinese_name
	model_path = scene_path
	theme_color = color
	accent_color = accent
	idle_animation_aliases = idle_aliases
	move_animation_aliases = move_aliases
	yaw_offset_degrees = yaw_offset
	scale_multiplier = model_scale
	source_url = asset_source_url
	head_bone_aliases = bone_aliases
	return self


func load_model_scene() -> PackedScene:
	if model_path.is_empty():
		return null
	return load(model_path) as PackedScene
