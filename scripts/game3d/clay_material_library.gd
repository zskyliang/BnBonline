class_name ClayMaterialLibrary
extends RefCounted
## Shared, cached Compatibility-renderer materials for the clay art direction.

const CREAM := Color("#f4e7d1")
const TERRACOTTA := Color("#d9664c")
const MUSTARD := Color("#e4b84f")
const GRASS := Color("#6fae66")
const SKY := Color("#a9d8e8")
const CHARCOAL := Color("#342f35")

const CHARACTER_ROUGHNESS := 0.9
const CHARACTER_SPECULAR := 0.12
const SCENE_ROUGHNESS := 0.92

static var _normal_texture: Texture2D
static var _roughness_texture: Texture2D
static var _color_cache: Dictionary = {}


static func make(
		color: Color,
		roughness: float = SCENE_ROUGHNESS,
		emission_strength: float = 0.0
	) -> StandardMaterial3D:
	var key := "%s|%.3f|%.3f" % [color.to_html(), roughness, emission_strength]
	if _color_cache.has(key):
		return _color_cache[key] as StandardMaterial3D
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.metallic = 0.0
	material.metallic_specular = 0.12
	material.roughness = clampf(roughness, 0.0, 1.0)
	material.normal_enabled = true
	material.normal_scale = 0.32
	material.normal_texture = _get_normal_texture()
	material.roughness_texture = _get_roughness_texture()
	material.uv1_scale = Vector3(7.5, 7.5, 7.5)
	if emission_strength > 0.0:
		material.emission_enabled = true
		material.emission = color
		material.emission_energy_multiplier = emission_strength
	_color_cache[key] = material
	return material


static func apply_to_model(root: Node3D, tint: Color) -> void:
	for child in root.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := child as MeshInstance3D
		if mesh_instance.mesh == null:
			continue
		mesh_instance.material_override = null
		for surface in range(mesh_instance.mesh.get_surface_count()):
			var source_color := Color.WHITE
			var source := mesh_instance.mesh.surface_get_material(surface) as BaseMaterial3D
			if source != null:
				source_color = source.albedo_color
			var mixed := source_color.lerp(tint, 0.16)
			mesh_instance.set_surface_override_material(
				surface,
				make(mixed, CHARACTER_ROUGHNESS)
			)


static func apply_character_palette(
		root: Node3D,
		clothing_color: Color,
		clothing_material_names: Array[String]
	) -> void:
	for child: Node in root.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := child as MeshInstance3D
		if mesh_instance.mesh == null:
			continue
		mesh_instance.material_override = null
		for surface: int in range(mesh_instance.mesh.get_surface_count()):
			var source_color := Color.WHITE
			var material_name: String = mesh_instance.mesh.surface_get_name(surface)
			var source := mesh_instance.mesh.surface_get_material(surface) as BaseMaterial3D
			if source != null:
				source_color = source.albedo_color
				if not source.resource_name.is_empty():
					material_name = source.resource_name
			var output_color: Color = source_color
			if material_name in clothing_material_names:
				var shade_seed: int = absi(material_name.hash()) % 3
				match shade_seed:
					0:
						output_color = clothing_color.lightened(0.1)
					1:
						output_color = clothing_color
					_:
						output_color = clothing_color.darkened(0.13)
			mesh_instance.set_surface_override_material(
				surface,
				make(output_color, CHARACTER_ROUGHNESS)
			)


static func clayify_imported_model(root: Node3D, tint: Color = Color.WHITE) -> void:
	for child in root.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := child as MeshInstance3D
		if mesh_instance.mesh == null:
			continue
		mesh_instance.material_override = null
		for surface: int in range(mesh_instance.mesh.get_surface_count()):
			var source := mesh_instance.mesh.surface_get_material(surface) as BaseMaterial3D
			if source == null:
				mesh_instance.set_surface_override_material(surface, make(tint, SCENE_ROUGHNESS))
				continue
			var material := source.duplicate() as BaseMaterial3D
			material.albedo_color = material.albedo_color.lerp(tint, 0.1)
			material.metallic = 0.0
			material.metallic_specular = 0.12
			material.roughness = SCENE_ROUGHNESS
			if not material.normal_enabled:
				material.normal_enabled = true
				material.normal_texture = _get_normal_texture()
				material.normal_scale = 0.18
			mesh_instance.set_surface_override_material(surface, material)


static func _get_normal_texture() -> Texture2D:
	if _normal_texture == null:
		_normal_texture = load("res://assets/materials/clay_detail_normal.png") as Texture2D
	return _normal_texture


static func _get_roughness_texture() -> Texture2D:
	if _roughness_texture == null:
		_roughness_texture = load("res://assets/materials/clay_roughness.png") as Texture2D
	return _roughness_texture
