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
var _sprites: Array[Sprite2D] = []

func setup(new_cells: Array[Vector2i], new_center: Vector2i, new_attacker: GameActor) -> void:
	cells = new_cells
	center_cell = new_center
	attacker = new_attacker
	z_index = 150
	for cell: Vector2i in cells:
		unsafe_lookup[cell] = true
		var sprite := Sprite2D.new()
		sprite.texture = EXPLOSION_TEXTURE
		sprite.region_enabled = true
		sprite.region_rect = _initial_region(cell)
		sprite.centered = false
		sprite.position = GameConstants.grid_to_top_left(cell)
		sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
		add_child(sprite)
		_sprites.append(sprite)

func _process(delta: float) -> void:
	_elapsed += delta
	var animation_frame: int = clampi(6 + int(_elapsed / 0.05), 6, 13)
	for index: int in range(_sprites.size()):
		var cell: Vector2i = cells[index]
		if cell == center_cell:
			_sprites[index].region_rect = Rect2((animation_frame % 4) * 40, 160, 40, 40)
		elif _is_ray_end(cell):
			_sprites[index].region_rect.position.x = animation_frame * 40
	if _elapsed >= GameConstants.EXPLOSION_SECONDS:
		finished.emit(self)
		queue_free()

func contains(cell: Vector2i) -> bool:
	return unsafe_lookup.has(cell)

func milliseconds_remaining() -> int:
	return maxi(0, ceili((GameConstants.EXPLOSION_SECONDS - _elapsed) * 1000.0))

func _initial_region(cell: Vector2i) -> Rect2:
	if cell == center_cell:
		return Rect2(0, 160, 40, 40)
	if cell.y == center_cell.y:
		return Rect2(200 if _is_ray_end(cell) else 120, 80 if cell.x < center_cell.x else 120, 40, 40)
	return Rect2(200 if _is_ray_end(cell) else 120, 0 if cell.y < center_cell.y else 40, 40, 40)

func _is_ray_end(cell: Vector2i) -> bool:
	var direction := Vector2i(signi(cell.x - center_cell.x), signi(cell.y - center_cell.y))
	return not cells.has(cell + direction)
