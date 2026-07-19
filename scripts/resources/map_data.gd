class_name MapData
extends Resource
## Runtime map resource for the 15 by 13 arena.

var map_id: String = "classic"
var display_name: String = "当前地图（经典）"
var ground_mode: String = "town"
var ground_cells: Array[PackedInt32Array] = []
var barrier_cells: Array[PackedInt32Array] = []
var player_spawn: Vector2i = Vector2i.ZERO
var decorations: Array[Dictionary] = []

func configure(
		new_id: String,
		new_name: String,
		new_ground_mode: String,
		new_ground_cells: Array[PackedInt32Array],
		new_barrier_cells: Array[PackedInt32Array],
		new_player_spawn: Vector2i,
		new_decorations: Array[Dictionary] = []
	) -> MapData:
	map_id = new_id
	display_name = new_name
	ground_mode = new_ground_mode
	ground_cells = new_ground_cells
	barrier_cells = new_barrier_cells
	player_spawn = new_player_spawn
	decorations = new_decorations
	return self

