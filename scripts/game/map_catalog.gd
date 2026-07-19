class_name MapCatalog
extends RefCounted
## Factory for the two maps retained from the H5 version.

static func get_map(map_id: String) -> MapData:
	if map_id == "windmill-heart":
		return _build_windmill_heart()
	return _build_classic()

static func get_options() -> Array[Dictionary]:
	return [
		{"id": "classic", "label": "当前地图（经典）"},
		{"id": "windmill-heart", "label": "风车爱心地图"},
	]

static func clone_matrix(source: Array[PackedInt32Array]) -> Array[PackedInt32Array]:
	var result: Array[PackedInt32Array] = []
	for row: PackedInt32Array in source:
		result.append(row.duplicate())
	return result

static func _build_classic() -> MapData:
	var ground: Array[PackedInt32Array] = [
		PackedInt32Array([1,1,1,1,1,1,3,1,3,1,1,1,1,1,1]),
		PackedInt32Array([2,2,1,1,1,1,3,1,3,1,1,1,1,2,1]),
		PackedInt32Array([2,3,3,3,3,3,3,3,3,3,3,3,3,3,3]),
		PackedInt32Array([3,2,2,1,1,1,1,3,1,3,1,1,1,1,2]),
		PackedInt32Array([2,1,1,1,1,1,1,3,1,3,1,1,1,1,1]),
		PackedInt32Array([1,2,2,1,1,1,1,3,1,3,1,1,1,1,2]),
		PackedInt32Array([2,1,1,1,1,1,1,3,1,3,1,1,1,1,1]),
		PackedInt32Array([1,2,2,1,1,1,1,3,1,3,1,1,1,1,2]),
		PackedInt32Array([2,1,1,1,1,1,1,3,1,3,1,1,1,1,1]),
		PackedInt32Array([1,2,2,1,1,1,1,3,1,3,1,1,1,1,2]),
		PackedInt32Array([2,3,3,3,3,3,3,3,3,3,3,3,3,3,3]),
		PackedInt32Array([3,2,2,1,1,1,1,3,1,3,1,1,1,1,2]),
		PackedInt32Array([2,1,1,1,1,1,1,3,1,3,1,1,1,1,1]),
	]
	var barriers: Array[PackedInt32Array] = [
		PackedInt32Array([0,3,1,5,1,7,0,7,0,7,1,4,1,4,0]),
		PackedInt32Array([0,0,3,0,0,1,0,0,0,1,2,1,2,0,0]),
		PackedInt32Array([3,5,3,5,1,7,3,7,3,7,1,4,1,4,0]),
		PackedInt32Array([0,3,2,1,7,3,3,1,3,3,7,1,2,1,2]),
		PackedInt32Array([1,7,1,7,3,3,7,0,7,3,3,7,1,7,1]),
		PackedInt32Array([2,0,3,0,0,7,1,1,1,7,0,0,3,0,2]),
		PackedInt32Array([2,7,1,7,0,2,3,3,3,2,0,7,1,7,2]),
		PackedInt32Array([2,0,3,0,0,7,1,1,1,7,0,0,3,0,2]),
		PackedInt32Array([1,7,1,7,3,3,7,2,7,3,3,7,1,7,1]),
		PackedInt32Array([2,1,2,1,7,3,3,1,3,3,7,1,2,1,2]),
		PackedInt32Array([0,4,1,4,1,7,3,7,3,7,1,6,1,6,0]),
		PackedInt32Array([0,0,2,1,2,1,0,0,0,1,2,1,2,0,0]),
		PackedInt32Array([0,4,1,4,1,7,0,7,0,7,1,6,1,6,0]),
	]
	# The maintained battle mode replaced trees and houses with boxes.
	for y: int in range(barriers.size()):
		for x: int in range(barriers[y].size()):
			if barriers[y][x] in [4, 5, 6, 7]:
				barriers[y][x] = 3
	return MapData.new().configure("classic", "当前地图（经典）", "town", ground, barriers, Vector2i.ZERO)

static func _build_windmill_heart() -> MapData:
	var ground: Array[PackedInt32Array] = []
	var barriers: Array[PackedInt32Array] = []
	for y: int in range(GameConstants.GRID_ROWS):
		var ground_row := PackedInt32Array()
		var barrier_row := PackedInt32Array()
		ground_row.resize(GameConstants.GRID_COLUMNS)
		barrier_row.resize(GameConstants.GRID_COLUMNS)
		ground_row.fill(1)
		barrier_row.fill(0)
		ground.append(ground_row)
		barriers.append(barrier_row)
	var heart: PackedStringArray = [
		"...............", "..#...###...#..", ".....#####.....",
		"....#.....#....", "..##.......##..", ".###.......###.",
		".###.......###.", "..##.......##..", "..###.....###..",
		"....##...##....", ".....#####.....", "......###......",
		"...............",
	]
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			if heart[y][x] == "#":
				barriers[y][x] = 3
	for x: int in range(GameConstants.GRID_COLUMNS):
		barriers[0][x] = 8
		barriers[GameConstants.GRID_ROWS - 1][x] = 8
	for y: int in range(GameConstants.GRID_ROWS):
		barriers[y][0] = 8
		barriers[y][GameConstants.GRID_COLUMNS - 1] = 8
	barriers[1][1] = 0
	barriers[1][2] = 0
	barriers[2][1] = 0
	for x: int in range(6, 9):
		barriers[6][x] = 9
	var decorations: Array[Dictionary] = [{"type": "windmill", "cell": Vector2i(6, 3)}]
	return MapData.new().configure(
		"windmill-heart", "风车爱心地图", "maptype2", ground, barriers,
		Vector2i(1, 1), decorations
	)

