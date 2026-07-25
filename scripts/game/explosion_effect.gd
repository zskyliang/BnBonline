class_name ExplosionEffect
extends Node2D
## Unsafe-cell lifetime for one cross explosion.

signal finished(effect: ExplosionEffect)

var cells: Array[Vector2i] = []
var center_cell: Vector2i
var attacker: GameActor
var unsafe_lookup: Dictionary = {}

var _elapsed: float = 0.0

func setup(new_cells: Array[Vector2i], new_center: Vector2i, new_attacker: GameActor) -> void:
	cells = new_cells
	center_cell = new_center
	attacker = new_attacker
	for cell: Vector2i in cells:
		unsafe_lookup[cell] = true

func _process(delta: float) -> void:
	_elapsed += delta
	if _elapsed >= GameConstants.EXPLOSION_SECONDS:
		finished.emit(self)
		queue_free()

func contains(cell: Vector2i) -> bool:
	return unsafe_lookup.has(cell)

func milliseconds_remaining() -> int:
	return maxi(0, ceili((GameConstants.EXPLOSION_SECONDS - _elapsed) * 1000.0))
