class_name ArenaItemState
extends RefCounted
## Renderer-independent state for one pickup on the logical board.

var item_id: int
var item_type: int
var cell: Vector2i
var spawned_ms: int


func _init(
		new_item_id: int,
		new_item_type: int,
		new_cell: Vector2i,
		new_spawned_ms: int = 0
	) -> void:
	item_id = new_item_id
	item_type = new_item_type
	cell = new_cell
	spawned_ms = maxi(0, new_spawned_ms)


func duplicate_state() -> ArenaItemState:
	return ArenaItemState.new(item_id, item_type, cell, spawned_ms)
