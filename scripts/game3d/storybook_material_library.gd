class_name StorybookMaterialLibrary
extends RefCounted
## Compatibility-safe hand-painted materials with local team tint and shell ink.

const PAPER := Color("#f6eddc")
const CHARCOAL := Color("#302b2b")
const MOSS := Color("#70845f")
const WOOD := Color("#8c6045")
const SKY := Color("#b9d8d2")
const CREAM := PAPER
const GRASS := Color("#6fae66")
const TERRACOTTA := Color("#d9664c")
const MUSTARD := Color("#e4b84f")

const CHARACTER_ROUGHNESS := 0.9
const OUTLINE_GROW := 0.012

static var _paper_normal: Texture2D
static var _material_cache: Dictionary = {}


static func make(
		color: Color,
		roughness: float = CHARACTER_ROUGHNESS,
		with_outline: bool = true,
		emission_strength: float = 0.0
	) -> StandardMaterial3D:
	var key := "%s|%.3f|%s|%.3f" % [
		color.to_html(),
		roughness,
		str(with_outline),
		emission_strength,
	]
	if _material_cache.has(key):
		return _material_cache[key] as StandardMaterial3D
	var material := StandardMaterial3D.new()
	material.resource_name = "Storybook_%s" % color.to_html(false)
	material.albedo_color = color
	material.diffuse_mode = BaseMaterial3D.DIFFUSE_TOON
	material.metallic = 0.0
	material.metallic_specular = 0.08
	material.roughness = clampf(roughness, 0.0, 1.0)
	material.normal_enabled = true
	material.normal_scale = 0.11
	material.normal_texture = _get_paper_normal()
	material.uv1_scale = Vector3(5.0, 5.0, 5.0)
	if emission_strength > 0.0:
		material.emission_enabled = true
		material.emission = color
		material.emission_energy_multiplier = emission_strength
	if with_outline:
		material.next_pass = _make_outline()
	_material_cache[key] = material
	return material


static func apply_character_palette(
		root: Node3D,
		team_color: Color,
		team_tint_material_names: Array[String],
		enable_ink_outlines: bool = true
	) -> void:
	for child: Node in root.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := child as MeshInstance3D
		if mesh_instance.mesh == null:
			continue
		mesh_instance.material_override = null
		mesh_instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		for surface: int in range(mesh_instance.mesh.get_surface_count()):
			var source_color := Color.WHITE
			var material_name: String = mesh_instance.mesh.surface_get_name(surface)
			var source := mesh_instance.mesh.surface_get_material(surface) as BaseMaterial3D
			if source != null:
				source_color = source.albedo_color
				if not source.resource_name.is_empty():
					material_name = source.resource_name
			var is_team_surface := material_name in team_tint_material_names
			var output_color := team_color if is_team_surface else source_color
			var is_ring := material_name == "FootRing"
			var is_internal_detail := material_name in [
				"Eyes",
				"Pupils",
				"Belly",
				"Muzzle",
				"Nose",
			]
			mesh_instance.set_surface_override_material(
				surface,
				make(
					output_color,
					0.88 if is_team_surface else CHARACTER_ROUGHNESS,
					enable_ink_outlines and not is_ring and not is_internal_detail,
					0.08 if is_ring else 0.0
				)
			)


static func _make_outline() -> StandardMaterial3D:
	var outline := StandardMaterial3D.new()
	outline.resource_name = "StorybookInkShell"
	outline.albedo_color = CHARCOAL
	outline.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	outline.cull_mode = BaseMaterial3D.CULL_FRONT
	outline.grow = true
	outline.grow_amount = OUTLINE_GROW
	return outline


static func _get_paper_normal() -> Texture2D:
	if _paper_normal == null:
		_paper_normal = load(
			"res://assets/materials/storybook_paper_normal.png"
		) as Texture2D
	return _paper_normal
