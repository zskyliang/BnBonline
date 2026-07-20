class_name ExplosionEffect
extends Node2D
## Visual and unsafe-cell lifetime for one cross explosion.

signal finished(effect: ExplosionEffect)

const EXPLOSION_TEXTURE: Texture2D = preload("res://assets/sprites/Explosion.png")

var cells: Array[Vector2i] = []
var center_cell: Vector2i
var attacker: GameActor
var unsafe_lookup: Dictionary = {}

var _elapsed: float = 0.0
var _ray_ends: Array[bool] = []
var _initial_regions: Array[Rect2] = []
var _animation_frame: int = 6

func setup(new_cells: Array[Vector2i], new_center: Vector2i, new_attacker: GameActor) -> void:
	cells = new_cells
	center_cell = new_center
	attacker = new_attacker
	z_index = 150
	texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	for cell: Vector2i in cells:
		unsafe_lookup[cell] = true
		var is_ray_end: bool = _is_ray_end(cell)
		_ray_ends.append(is_ray_end)
		_initial_regions.append(_initial_region(cell, is_ray_end))
	queue_redraw()

func _draw() -> void:
	for index: int in range(cells.size()):
		var region: Rect2 = _initial_regions[index]
		if cells[index] == center_cell:
			region = Rect2((_animation_frame % 4) * 40, 160, 40, 40)
		elif _ray_ends[index]:
			region.position.x = _animation_frame * 40
		draw_texture_rect_region(
			EXPLOSION_TEXTURE,
			Rect2(GameConstants.grid_to_top_left(cells[index]), Vector2(40, 40)),
			region
		)

func _process(delta: float) -> void:
	_elapsed += delta
	var animation_frame: int = clampi(6 + int(_elapsed / 0.05), 6, 13)
	if animation_frame != _animation_frame:
		_animation_frame = animation_frame
		queue_redraw()
	if _elapsed >= GameConstants.EXPLOSION_SECONDS:
		finished.emit(self)
		queue_free()

func contains(cell: Vector2i) -> bool:
	return unsafe_lookup.has(cell)

func milliseconds_remaining() -> int:
	return maxi(0, ceili((GameConstants.EXPLOSION_SECONDS - _elapsed) * 1000.0))

func _initial_region(cell: Vector2i, is_ray_end: bool) -> Rect2:
	if cell == center_cell:
		return Rect2(0, 160, 40, 40)
	if cell.y == center_cell.y:
		return Rect2(200 if is_ray_end else 120, 80 if cell.x < center_cell.x else 120, 40, 40)
	return Rect2(200 if is_ray_end else 120, 0 if cell.y < center_cell.y else 40, 40, 40)

func _is_ray_end(cell: Vector2i) -> bool:
	var direction := Vector2i(signi(cell.x - center_cell.x), signi(cell.y - center_cell.y))
	return not cells.has(cell + direction)
