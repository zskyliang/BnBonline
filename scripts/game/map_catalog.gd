class_name MapCatalog
extends RefCounted
## Factory for the single obstacle-free paint arena.

const PAINT_ARENA := "paint-arena"
const HARBOR_MARKET := PAINT_ARENA
const BELL_GARDEN := PAINT_ARENA
const PLAYER_SPAWN := Vector2i(1, 11)


static func get_map(_map_id: String = PAINT_ARENA) -> MapData:
	return _build_paint_arena()


static func get_options() -> Array[Dictionary]:
	return [
		{"id": PAINT_ARENA, "label": "染色竞技场"},
	]


static func is_valid_id(map_id: String) -> bool:
	return map_id == PAINT_ARENA


static func migrate_legacy_id(_map_id: String) -> String:
	return PAINT_ARENA


static func clone_matrix(source: Array[PackedInt32Array]) -> Array[PackedInt32Array]:
	var result: Array[PackedInt32Array] = []
	for row: PackedInt32Array in source:
		result.append(row.duplicate())
	return result


static func count_code(map_data: MapData, code: int) -> int:
	var result := 0
	for row: PackedInt32Array in map_data.barrier_cells:
		for value: int in row:
			if value == code:
				result += 1
	return result


static func _build_paint_arena() -> MapData:
	return MapData.new().configure(
		PAINT_ARENA,
		"染色竞技场",
		"paint",
		_make_matrix(1),
		_make_matrix(0),
		PLAYER_SPAWN,
		AABB(Vector3(-8.5, -0.4, -7.5), Vector3(17.0, 3.4, 15.0)),
		[]
	)


static func _make_matrix(fill_code: int) -> Array[PackedInt32Array]:
	var result: Array[PackedInt32Array] = []
	for _y: int in range(GameConstants.GRID_ROWS):
		var row := PackedInt32Array()
		row.resize(GameConstants.GRID_COLUMNS)
		row.fill(fill_code)
		result.append(row)
	return result
