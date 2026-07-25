class_name MapCatalog
extends RefCounted
## Factory for the two fixed clay-island battle maps.

const HARBOR_MARKET := "harbor-market"
const BELL_GARDEN := "bell-garden"
const PLAYER_SPAWN := Vector2i(1, 11)

const HARBOR_LAYOUT: PackedStringArray = [
	"~~~~~~~~~~~~~~~",
	"~..##..+.##...~",
	"~..##+.+.##...~",
	"~+.+..#..+.+..~",
	"~..#.+.+.#.+..~",
	"~+.#..+..#..+.~",
	"~.+..##.##.+..~",
	"~..+.##.##..+.~",
	"~.+#..+.+..#+.~",
	"~..#.+..+..#..~",
	"~..+..##.+.+..~",
	"~...+...+.....~",
	"~~~~~~~~~~~~~~~",
]

const BELL_LAYOUT: PackedStringArray = [
	"~~~~~~~~~~~~~~~",
	"~...+.###+....~",
	"~..+..###..+..~",
	"~.....###.....~",
	"~.#.+..+..+.#.~",
	"~.#..+...+..#.~",
	"~+..##.+.##..+~",
	"~..+.##.##.+..~",
	"~.+..##.##..+.~",
	"~.#.+..+..+.#.~",
	"~..+.......+..~",
	"~....+...+....~",
	"~~~~~~~~~~~~~~~",
]


static func get_map(map_id: String) -> MapData:
	var migrated_id := migrate_legacy_id(map_id)
	if migrated_id == BELL_GARDEN:
		return _build_bell_garden()
	return _build_harbor_market()


static func get_options() -> Array[Dictionary]:
	return [
		{"id": HARBOR_MARKET, "label": "软陶海岛集市"},
		{"id": BELL_GARDEN, "label": "钟楼花园"},
	]


static func is_valid_id(map_id: String) -> bool:
	return map_id == HARBOR_MARKET or map_id == BELL_GARDEN


static func migrate_legacy_id(map_id: String) -> String:
	if map_id == "windmill-heart" or map_id == BELL_GARDEN:
		return BELL_GARDEN
	return HARBOR_MARKET


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


static func _build_harbor_market() -> MapData:
	var buildings: Array[BuildingPlacement] = [
		_placement("cottage_red", Vector2i(3, 1), Vector2i(2, 2), 0, 0),
		_placement("shop_blue", Vector2i(9, 1), Vector2i(2, 2), 2, 1),
		_placement("cottage_mustard", Vector2i(5, 6), Vector2i(2, 2), 1, 2),
		_placement("cottage_green", Vector2i(8, 6), Vector2i(2, 2), 3, 3),
	]
	for cell: Vector2i in [
		Vector2i(3, 4), Vector2i(3, 5), Vector2i(3, 8), Vector2i(3, 9),
		Vector2i(6, 3), Vector2i(6, 10), Vector2i(7, 10),
		Vector2i(9, 4), Vector2i(9, 5), Vector2i(11, 8), Vector2i(11, 9),
	]:
		buildings.append(_placement("hedge", cell, Vector2i.ONE, cell.x + cell.y, 0, false))
	return MapData.new().configure(
		HARBOR_MARKET,
		"软陶海岛集市",
		"harbor",
		_make_ground("harbor"),
		_parse_layout(HARBOR_LAYOUT),
		PLAYER_SPAWN,
		buildings,
		AABB(Vector3(-9.5, -1.1, -8.5), Vector3(19.0, 6.4, 17.0)),
		[
			{"type": "dock", "cell": Vector2i(2, 12), "rotation": 0},
			{"type": "dock", "cell": Vector2i(12, 0), "rotation": 2},
		]
	)


static func _build_bell_garden() -> MapData:
	var buildings: Array[BuildingPlacement] = [
		_placement("bell_tower", Vector2i(6, 1), Vector2i(3, 3), 0, 0),
		_placement("clinic", Vector2i(4, 6), Vector2i(2, 1), 0, 0),
		_placement("cottage_red", Vector2i(5, 7), Vector2i(2, 2), 1, 1),
		_placement("clinic", Vector2i(9, 6), Vector2i(2, 1), 2, 2),
		_placement("cottage_green", Vector2i(8, 7), Vector2i(2, 2), 3, 3),
	]
	for cell: Vector2i in [
		Vector2i(2, 4), Vector2i(2, 5), Vector2i(2, 9),
		Vector2i(12, 4), Vector2i(12, 5), Vector2i(12, 9),
	]:
		buildings.append(_placement("hedge", cell, Vector2i.ONE, cell.y, 0, false))
	return MapData.new().configure(
		BELL_GARDEN,
		"钟楼花园",
		"garden",
		_make_ground("garden"),
		_parse_layout(BELL_LAYOUT),
		PLAYER_SPAWN,
		buildings,
		AABB(Vector3(-9.5, -1.1, -8.5), Vector3(19.0, 7.2, 17.0)),
		[
			{"type": "fountain", "cell": Vector2i(7, 0), "rotation": 0},
			{"type": "garden_gate", "cell": Vector2i(7, 12), "rotation": 0},
		]
	)


static func _parse_layout(layout: PackedStringArray) -> Array[PackedInt32Array]:
	assert(layout.size() == GameConstants.GRID_ROWS)
	var result: Array[PackedInt32Array] = []
	for row_text: String in layout:
		assert(row_text.length() == GameConstants.GRID_COLUMNS)
		var row := PackedInt32Array()
		for character: String in row_text:
			match character:
				"#":
					row.append(1)
				"+":
					row.append(3)
				"~":
					row.append(9)
				_:
					row.append(0)
		result.append(row)
	return result


static func _make_ground(theme_id: String) -> Array[PackedInt32Array]:
	var result: Array[PackedInt32Array] = []
	for y: int in range(GameConstants.GRID_ROWS):
		var row := PackedInt32Array()
		for x: int in range(GameConstants.GRID_COLUMNS):
			var accent := 2 if (x * 5 + y * 3 + theme_id.length()) % 11 == 0 else 1
			row.append(accent)
		result.append(row)
	return result


static func _placement(
		asset_id: String,
		origin_cell: Vector2i,
		footprint: Vector2i,
		rotation_quadrants: int,
		variant: int,
		occludable: bool = true
	) -> BuildingPlacement:
	return BuildingPlacement.new().configure(
		asset_id,
		origin_cell,
		footprint,
		rotation_quadrants,
		variant,
		occludable
	)
