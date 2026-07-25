class_name BuildingDefinition
extends Resource
## Catalog entry for a normalized clay building or scenery unit.

var id: String = ""
var display_name: String = ""
var kind: StringName = &"building"
var module_paths: PackedStringArray = []
var footprint: Vector2i = Vector2i.ONE
var base_color: Color = ClayMaterialLibrary.CREAM
var roof_color: Color = ClayMaterialLibrary.TERRACOTTA
var source_name: String = ""
var source_url: String = ""
var license_name: String = "CC0-1.0"


func configure(
		new_id: String,
		new_display_name: String,
		new_kind: StringName,
		new_module_paths: PackedStringArray,
		new_footprint: Vector2i,
		new_base_color: Color,
		new_roof_color: Color,
		new_source_name: String,
		new_source_url: String
	) -> BuildingDefinition:
	id = new_id
	display_name = new_display_name
	kind = new_kind
	module_paths = new_module_paths
	footprint = new_footprint
	base_color = new_base_color
	roof_color = new_roof_color
	source_name = new_source_name
	source_url = new_source_url
	return self
