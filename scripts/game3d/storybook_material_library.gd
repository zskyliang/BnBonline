class_name StorybookMaterialLibrary
extends RefCounted
## Compatibility-safe unlit materials for ImageGen Sprite3D artwork.

const PAPER := Color("#f6eddc")
const CHARCOAL := Color("#302b2b")
const MOSS := Color("#70845f")
const WOOD := Color("#8c6045")
const SKY := Color("#b9d8d2")
const CREAM := PAPER
const GRASS := Color("#6fae66")
const TERRACOTTA := Color("#d9664c")
const MUSTARD := Color("#e4b84f")
const TEAM_TINT_SHADER := preload(
	"res://assets/materials/storybook_sprite_team_tint.gdshader"
)

static var _material_cache: Dictionary = {}


static func make(
		color: Color,
		_roughness: float = 0.9,
		_with_outline: bool = false,
		emission_strength: float = 0.0
	) -> StandardMaterial3D:
	var key := "%s|%.3f" % [color.to_html(), emission_strength]
	if _material_cache.has(key):
		return _material_cache[key] as StandardMaterial3D
	var material := StandardMaterial3D.new()
	material.resource_name = "Storybook_%s" % color.to_html(false)
	material.albedo_color = color
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	if color.a < 0.999:
		material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	if emission_strength > 0.0:
		material.emission_enabled = true
		material.emission = color
		material.emission_energy_multiplier = emission_strength
	_material_cache[key] = material
	return material


static func make_sprite_material(
		texture: Texture2D,
		mask: Texture2D,
		team_color: Color,
		team_strength: float = 0.92
	) -> ShaderMaterial:
	var material := ShaderMaterial.new()
	material.shader = TEAM_TINT_SHADER
	material.set_shader_parameter("base_texture", texture)
	material.set_shader_parameter("tint_mask", mask)
	material.set_shader_parameter("team_color", team_color)
	material.set_shader_parameter("team_strength", team_strength)
	return material


static func make_textured(
		texture: Texture2D,
		color: Color = Color.WHITE,
		alpha_scissor: bool = true
	) -> StandardMaterial3D:
	var material := StandardMaterial3D.new()
	material.albedo_texture = texture
	material.albedo_color = color
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	material.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS
	material.transparency = (
		BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
		if alpha_scissor
		else BaseMaterial3D.TRANSPARENCY_ALPHA
	)
	material.alpha_scissor_threshold = 0.055
	return material


static func configure_billboard(
		sprite: Sprite3D,
		use_alpha_blend: bool = false
	) -> void:
	sprite.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	sprite.alpha_cut = (
		SpriteBase3D.ALPHA_CUT_DISABLED
		if use_alpha_blend
		else SpriteBase3D.ALPHA_CUT_DISCARD
	)
	sprite.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS
	sprite.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
