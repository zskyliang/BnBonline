class_name BuildingPlacement
extends Resource
## Map-owned placement for one static building, landmark, or rigid scenery unit.

var asset_id: String = ""
var origin_cell: Vector2i = Vector2i.ZERO
var footprint: Vector2i = Vector2i.ONE
var rotation_quadrants: int = 0
var variant: int = 0
var occludable: bool = true


func configure(
		new_asset_id: String,
		new_origin_cell: Vector2i,
		new_footprint: Vector2i = Vector2i.ONE,
		new_rotation_quadrants: int = 0,
		new_variant: int = 0,
		new_occludable: bool = true
	) -> BuildingPlacement:
	asset_id = new_asset_id
	origin_cell = new_origin_cell
	footprint = Vector2i(maxi(1, new_footprint.x), maxi(1, new_footprint.y))
	rotation_quadrants = posmod(new_rotation_quadrants, 4)
	variant = maxi(0, new_variant)
	occludable = new_occludable
	return self


func covered_cells() -> Array[Vector2i]:
	var result: Array[Vector2i] = []
	for y: int in range(footprint.y):
		for x: int in range(footprint.x):
			result.append(origin_cell + Vector2i(x, y))
	return result
