class_name MapData
extends Resource
## Runtime map resource for the 15 by 13 arena.

var map_id: String = "paint-arena"
var display_name: String = "染色竞技场"
var theme_id: String = "paint"
var ground_mode: String = "paint"
var ground_cells: Array[PackedInt32Array] = []
var barrier_cells: Array[PackedInt32Array] = []
var player_spawn: Vector2i = Vector2i(1, 11)
var camera_bounds: AABB = AABB(Vector3(-9.5, -1.0, -8.5), Vector3(19.0, 7.0, 17.0))
var building_units: Array[BuildingPlacement] = []
var decorations: Array[Dictionary] = []


func configure(
		new_id: String,
		new_name: String,
		new_theme_id: String,
		new_ground_cells: Array[PackedInt32Array],
		new_barrier_cells: Array[PackedInt32Array],
		new_player_spawn: Vector2i,
		new_building_units: Array[BuildingPlacement] = [],
		new_camera_bounds: AABB = AABB(Vector3(-9.5, -1.0, -8.5), Vector3(19.0, 7.0, 17.0)),
		new_decorations: Array[Dictionary] = []
	) -> MapData:
	map_id = new_id
	display_name = new_name
	theme_id = new_theme_id
	ground_mode = new_theme_id
	ground_cells = new_ground_cells
	barrier_cells = new_barrier_cells
	player_spawn = new_player_spawn
	building_units = new_building_units
	camera_bounds = new_camera_bounds
	decorations = new_decorations
	return self
