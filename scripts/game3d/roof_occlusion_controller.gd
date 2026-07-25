class_name RoofOcclusionController
extends Node
## Unions camera-to-actor roof hits and drives per-building dither visibility.

const CHECK_INTERVAL := 0.1
const MAX_RAY_HITS := 8
const VISUAL_OCCLUDER_MASK := ClayBuildingView3D.VISUAL_OCCLUDER_LAYER

var camera: Camera3D
var actor_root: Node3D
var _accumulator := 0.0


func bind(new_camera: Camera3D, new_actor_root: Node3D) -> void:
	camera = new_camera
	actor_root = new_actor_root


func _process(delta: float) -> void:
	if not is_instance_valid(camera) or not is_instance_valid(actor_root):
		return
	_accumulator += delta
	if _accumulator < CHECK_INTERVAL:
		return
	_accumulator = fmod(_accumulator, CHECK_INTERVAL)
	_update_occlusion()


func _update_occlusion() -> void:
	var buildings: Array[ClayBuildingView3D] = []
	for node: Node in get_tree().get_nodes_in_group("clay_occludable_buildings"):
		if node is ClayBuildingView3D:
			var building := node as ClayBuildingView3D
			buildings.append(building)
			building.set_occluded(false)
	if buildings.is_empty() or camera.get_world_3d() == null:
		return
	var hidden: Dictionary = {}
	for child: Node in actor_root.get_children():
		if not child is ActorView3D:
			continue
		var actor_view := child as ActorView3D
		if not actor_view.visible or not is_instance_valid(actor_view.actor):
			continue
		if actor_view.actor.stats.is_dead:
			continue
		_collect_ray_hits(
			camera.global_position,
			actor_view.global_position + Vector3(0.0, 0.72, 0.0),
			hidden
		)
	for building: ClayBuildingView3D in hidden.keys():
		building.set_occluded(true)


func _collect_ray_hits(origin: Vector3, target: Vector3, result: Dictionary) -> void:
	var exclusions: Array[RID] = []
	var space_state := camera.get_world_3d().direct_space_state
	for _index: int in range(MAX_RAY_HITS):
		var query := PhysicsRayQueryParameters3D.create(origin, target, VISUAL_OCCLUDER_MASK, exclusions)
		query.collide_with_areas = false
		query.collide_with_bodies = true
		var hit := space_state.intersect_ray(query)
		if hit.is_empty():
			break
		var collider := hit.get("collider") as CollisionObject3D
		if not is_instance_valid(collider):
			break
		exclusions.append(collider.get_rid())
		var building := collider.get_meta("clay_building_view", null) as ClayBuildingView3D
		if is_instance_valid(building):
			result[building] = true
