class_name BuildingCatalog
extends RefCounted
## Central source of selected CC0 environment modules and clay wrapper metadata.

const KENNEY_ROOT := "res://assets/models/environment/kenney_town/"
const QUATERNIUS_ROOT := "res://assets/models/environment/quaternius_nature/"
const KENNEY_URL := "https://kenney.nl/assets/fantasy-town-kit"
const QUATERNIUS_URL := "https://quaternius.com/packs/stylizednaturemegakit.html"

static var _scene_cache: Dictionary = {}


static func get_definition(id: String) -> BuildingDefinition:
	var data: Dictionary = _definition_data().get(id, _definition_data()["hedge"])
	return BuildingDefinition.new().configure(
		id if _definition_data().has(id) else "hedge",
		str(data["display_name"]),
		StringName(data["kind"]),
		data["module_paths"] as PackedStringArray,
		data["footprint"] as Vector2i,
		data["base_color"] as Color,
		data["roof_color"] as Color,
		str(data["source_name"]),
		str(data["source_url"])
	)


static func get_module_scene(resource_path: String) -> PackedScene:
	if resource_path.is_empty():
		return null
	if _scene_cache.has(resource_path):
		return _scene_cache[resource_path] as PackedScene
	var scene := load(resource_path) as PackedScene
	if scene != null:
		_scene_cache[resource_path] = scene
	return scene


static func instantiate_module(resource_path: String) -> Node3D:
	var scene := get_module_scene(resource_path)
	if scene == null:
		return null
	return scene.instantiate() as Node3D


static func nature_path(asset_name: String) -> String:
	return QUATERNIUS_ROOT + asset_name + ".glb"


static func _definition_data() -> Dictionary:
	return {
		"cottage_red": {
			"display_name": "陶红屋顶小屋",
			"kind": &"house",
			"module_paths": PackedStringArray([
				KENNEY_ROOT + "wall-wood.glb",
				KENNEY_ROOT + "wall-window-shutters.glb",
				KENNEY_ROOT + "wall-door.glb",
				KENNEY_ROOT + "roof-high.glb",
			]),
			"footprint": Vector2i(2, 2),
			"base_color": ClayMaterialLibrary.CREAM,
			"roof_color": ClayMaterialLibrary.TERRACOTTA,
			"source_name": "Kenney Fantasy Town Kit",
			"source_url": KENNEY_URL,
		},
		"shop_blue": {
			"display_name": "天空蓝商店",
			"kind": &"shop",
			"module_paths": PackedStringArray([
				KENNEY_ROOT + "wall.glb",
				KENNEY_ROOT + "wall-window-shutters.glb",
				KENNEY_ROOT + "wall-door.glb",
				KENNEY_ROOT + "roof-high-gable.glb",
			]),
			"footprint": Vector2i(2, 2),
			"base_color": ClayMaterialLibrary.CREAM.lightened(0.05),
			"roof_color": ClayMaterialLibrary.SKY.darkened(0.08),
			"source_name": "Kenney Fantasy Town Kit",
			"source_url": KENNEY_URL,
		},
		"cottage_mustard": {
			"display_name": "芥末黄屋顶小屋",
			"kind": &"house",
			"module_paths": PackedStringArray([
				KENNEY_ROOT + "wall-wood.glb",
				KENNEY_ROOT + "wall-window-shutters.glb",
				KENNEY_ROOT + "wall-door.glb",
				KENNEY_ROOT + "roof-gable.glb",
			]),
			"footprint": Vector2i(2, 2),
			"base_color": ClayMaterialLibrary.CREAM,
			"roof_color": ClayMaterialLibrary.MUSTARD,
			"source_name": "Kenney Fantasy Town Kit",
			"source_url": KENNEY_URL,
		},
		"cottage_green": {
			"display_name": "草绿屋顶小屋",
			"kind": &"house",
			"module_paths": PackedStringArray([
				KENNEY_ROOT + "wall.glb",
				KENNEY_ROOT + "wall-window-shutters.glb",
				KENNEY_ROOT + "wall-door.glb",
				KENNEY_ROOT + "roof-high.glb",
			]),
			"footprint": Vector2i(2, 2),
			"base_color": ClayMaterialLibrary.CREAM.darkened(0.03),
			"roof_color": ClayMaterialLibrary.GRASS,
			"source_name": "Kenney Fantasy Town Kit",
			"source_url": KENNEY_URL,
		},
		"bell_tower": {
			"display_name": "软陶钟楼",
			"kind": &"tower",
			"module_paths": PackedStringArray([
				KENNEY_ROOT + "wall.glb",
				KENNEY_ROOT + "wall-window-shutters.glb",
			]),
			"footprint": Vector2i(3, 3),
			"base_color": ClayMaterialLibrary.CREAM,
			"roof_color": ClayMaterialLibrary.SKY.darkened(0.16),
			"source_name": "Kenney Fantasy Town Kit",
			"source_url": KENNEY_URL,
		},
		"clinic": {
			"display_name": "花园诊所",
			"kind": &"clinic",
			"module_paths": PackedStringArray([
				KENNEY_ROOT + "wall.glb",
				KENNEY_ROOT + "wall-window-shutters.glb",
				KENNEY_ROOT + "wall-door.glb",
				KENNEY_ROOT + "roof-gable.glb",
			]),
			"footprint": Vector2i(2, 1),
			"base_color": ClayMaterialLibrary.CREAM,
			"roof_color": ClayMaterialLibrary.TERRACOTTA.lightened(0.05),
			"source_name": "Kenney Fantasy Town Kit",
			"source_url": KENNEY_URL,
		},
		"hedge": {
			"display_name": "花园树篱",
			"kind": &"hedge",
			"module_paths": PackedStringArray([
				QUATERNIUS_ROOT + "bush_flowers.glb",
			]),
			"footprint": Vector2i.ONE,
			"base_color": ClayMaterialLibrary.GRASS,
			"roof_color": ClayMaterialLibrary.GRASS,
			"source_name": "Quaternius Stylized Nature MegaKit",
			"source_url": QUATERNIUS_URL,
		},
	}
