class_name ClayExplosionPool
extends Node3D
## Arena-local warm pool that caps decorative droplets while preserving hazard cells.

const INITIAL_SIZE := 8
const MAX_SIZE := 64
const MAX_DECORATIVE_DROPLETS := 48
const DROPLETS_PER_EFFECT := 4

var _available: Array[ExplosionView3D] = []
var _in_use: Array[ExplosionView3D] = []


func _ready() -> void:
	for _index: int in range(INITIAL_SIZE):
		_available.append(_create_view())


func spawn(effect: ExplosionEffect) -> ExplosionView3D:
	var view: ExplosionView3D
	if _available.is_empty():
		if _in_use.size() >= MAX_SIZE:
			push_warning("Clay explosion pool exhausted; reusing the oldest completed slot.")
			view = _in_use.pop_front()
			view.deactivate()
		else:
			view = _create_view()
	else:
		view = _available.pop_back()
	_in_use.append(view)
	var effects_with_droplets := MAX_DECORATIVE_DROPLETS / DROPLETS_PER_EFFECT
	var droplet_count := DROPLETS_PER_EFFECT if _in_use.size() <= effects_with_droplets else 0
	view.activate(effect, droplet_count)
	return view


func release(view: ExplosionView3D) -> void:
	if view not in _in_use:
		return
	_in_use.erase(view)
	view.deactivate()
	_available.append(view)


func release_all() -> void:
	var active := _in_use.duplicate()
	for view: ExplosionView3D in active:
		release(view)


func get_stats() -> Dictionary:
	return {
		"available": _available.size(),
		"in_use": _in_use.size(),
		"total": _available.size() + _in_use.size(),
		"decorative_droplets": mini(
			_in_use.size() * DROPLETS_PER_EFFECT,
			MAX_DECORATIVE_DROPLETS
		),
	}


func _create_view() -> ExplosionView3D:
	var view := ExplosionView3D.new()
	add_child(view)
	view.release_requested.connect(release)
	view.deactivate()
	return view
