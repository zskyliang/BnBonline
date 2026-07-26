extends SceneTree
## Import, animation, and budget contract for every original Blender MCP asset.

const REQUIRED_ACTIONS: Array[String] = [
	"Idle",
	"Waddle",
	"PlaceBubble",
	"Trapped",
	"Defeat",
	"Victory",
]
const CHARACTER_IDS: Array[String] = [
	"cat",
	"dog",
	"rabbit",
	"bear",
	"fox",
	"raccoon",
	"penguin",
	"capybara",
]
const PROP_PATHS: Array[String] = [
	"res://assets/models/items/storybook/leaf_shoes.glb",
	"res://assets/models/items/storybook/bubble_gourd.glb",
	"res://assets/models/items/storybook/paw_burst.glb",
]
const WORLD_PATHS: Array[String] = [
	"res://assets/models/effects/storybook/bubble_bomb.glb",
	"res://assets/models/effects/storybook/effect_shapes.glb",
	"res://assets/models/environment/storybook/storybook_floor_tile.glb",
	"res://assets/models/environment/storybook/forest_board_decor.glb",
	"res://assets/models/environment/storybook/storybook_lobby.glb",
]

var _checks: int = 0
var _failures: int = 0


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	for character_id: String in CHARACTER_IDS:
		await _validate_character(character_id)
	for prop_path: String in PROP_PATHS:
		await _validate_static_asset(prop_path, 2000)
	for world_path: String in WORLD_PATHS:
		await _validate_static_asset(world_path, 2500)
	print("Forest storybook GLBs: %d checks, %d failures" % [_checks, _failures])
	quit(1 if _failures > 0 else 0)


func _validate_character(character_id: String) -> void:
	var path := "res://assets/models/characters/%s.glb" % character_id
	var packed := load(path) as PackedScene
	_check(packed != null, "%s GLB imports as PackedScene" % character_id)
	if packed == null:
		return
	var instance := packed.instantiate() as Node3D
	_check(instance != null, "%s GLB instantiates as Node3D" % character_id)
	if instance == null:
		return
	root.add_child(instance)
	await process_frame

	var bounds := _calculate_bounds(instance)
	_check(
		bounds.size.y >= 1.20 and bounds.size.y <= 1.40,
		"%s raw height stays near 1.3 Godot units" % character_id
	)
	_check(
		absf(bounds.position.y) <= 0.04,
		"%s origin is centered at the grounded feet" % character_id
	)

	var skeleton := _find_first(instance, "Skeleton3D") as Skeleton3D
	_check(skeleton != null, "%s imports a Skeleton3D" % character_id)
	if skeleton != null:
		_check(
			skeleton.get_bone_count() > 0 and skeleton.get_bone_count() <= 32,
			"%s stays within the 32-bone budget" % character_id
		)

	var animation_player := _find_first(instance, "AnimationPlayer") as AnimationPlayer
	_check(animation_player != null, "%s imports an AnimationPlayer" % character_id)
	if animation_player != null:
		var imported_names: Array[String] = []
		for animation_name: StringName in animation_player.get_animation_list():
			imported_names.append(String(animation_name))
		for required_action: String in REQUIRED_ACTIONS:
			var matched := false
			for imported_name: String in imported_names:
				matched = matched or imported_name.ends_with(required_action) \
					or imported_name.contains(required_action)
			_check(matched, "%s imports %s" % [character_id, required_action])
		_validate_waddle(character_id, animation_player)

	var material_names: Dictionary = {}
	var triangle_count := 0
	for node: Node in instance.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := node as MeshInstance3D
		if mesh_instance.mesh == null:
			continue
		triangle_count += mesh_instance.mesh.get_faces().size() / 3
		for surface: int in range(mesh_instance.mesh.get_surface_count()):
			var material := mesh_instance.mesh.surface_get_material(surface)
			if material != null:
				material_names[material.resource_name] = true
	_check(triangle_count <= 8000, "%s stays within the 8k triangle budget" % character_id)
	_check(
		material_names.has("TeamTint") and material_names.has("FootRing"),
		"%s exposes only the declared local team-tint slots" % character_id
	)
	_check(
		material_names.has("Eyes") \
			and material_names.has("Belly") \
			and material_names.has("Muzzle"),
		"%s keeps eyes, belly, and muzzle in fixed material slots" % character_id
	)

	instance.queue_free()
	await process_frame


func _validate_waddle(character_id: String, player: AnimationPlayer) -> void:
	var waddle: Animation
	for animation_name: StringName in player.get_animation_list():
		if String(animation_name).contains("Waddle"):
			waddle = player.get_animation(animation_name)
			break
	_check(waddle != null, "%s exposes its Waddle clip for gait inspection" % character_id)
	if waddle == null:
		return
	var arm_spread := false
	var leg_positions: Array[Vector3] = []
	var root_positions: Array[Vector3] = []
	for track_index: int in range(waddle.get_track_count()):
		var path := String(waddle.track_get_path(track_index))
		for key_index: int in range(waddle.track_get_key_count(track_index)):
			var value: Variant = waddle.track_get_key_value(track_index, key_index)
			if ("Arm.L" in path or "Arm.R" in path) and value is Quaternion:
				arm_spread = arm_spread or absf((value as Quaternion).get_euler().z) >= 0.65
			if ("Leg.L" in path or "Leg.R" in path) and value is Vector3:
				leg_positions.append(value as Vector3)
			if "Root" in path and value is Vector3:
				root_positions.append(value as Vector3)
	_check(arm_spread, "%s Waddle keeps both arms in a broad toddler balance pose" % character_id)
	_check(
		_max_vector_separation(leg_positions) >= 0.05,
		"%s Waddle visibly alternates lifted and planted feet" % character_id
	)
	_check(
		_max_vector_separation(root_positions) >= 0.06,
		"%s Waddle shifts the body weight instead of sliding rigidly" % character_id
	)


func _validate_static_asset(path: String, triangle_budget: int) -> void:
	var packed := load(path) as PackedScene
	_check(packed != null, "%s imports as PackedScene" % path.get_file())
	if packed == null:
		return
	var instance := packed.instantiate() as Node3D
	_check(instance != null, "%s instantiates as Node3D" % path.get_file())
	if instance == null:
		return
	root.add_child(instance)
	await process_frame
	_check(
		_triangle_count(instance) <= triangle_budget,
		"%s stays within its runtime triangle budget" % path.get_file()
	)
	instance.queue_free()
	await process_frame


func _triangle_count(root_node: Node) -> int:
	var triangle_count := 0
	for node: Node in root_node.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := node as MeshInstance3D
		if mesh_instance.mesh != null:
			triangle_count += mesh_instance.mesh.get_faces().size() / 3
	return triangle_count


func _max_vector_separation(values: Array[Vector3]) -> float:
	var maximum := 0.0
	for left_index: int in range(values.size()):
		for right_index: int in range(left_index + 1, values.size()):
			maximum = maxf(maximum, values[left_index].distance_to(values[right_index]))
	return maximum


func _find_first(root_node: Node, type_name: String) -> Node:
	if root_node.is_class(type_name):
		return root_node
	for child: Node in root_node.get_children():
		var found := _find_first(child, type_name)
		if found != null:
			return found
	return null


func _calculate_bounds(root_node: Node3D) -> AABB:
	var minimum := Vector3(INF, INF, INF)
	var maximum := Vector3(-INF, -INF, -INF)
	var found := false
	var root_inverse := root_node.global_transform.affine_inverse()
	for node: Node in root_node.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := node as MeshInstance3D
		if mesh_instance.mesh == null:
			continue
		var box := mesh_instance.get_aabb()
		var relative := root_inverse * mesh_instance.global_transform
		for x: float in [box.position.x, box.end.x]:
			for y: float in [box.position.y, box.end.y]:
				for z: float in [box.position.z, box.end.z]:
					var point := relative * Vector3(x, y, z)
					minimum = minimum.min(point)
					maximum = maximum.max(point)
					found = true
	if not found:
		return AABB(Vector3.ZERO, Vector3.ZERO)
	return AABB(minimum, maximum - minimum)


func _check(condition: bool, description: String) -> void:
	_checks += 1
	if condition:
		return
	_failures += 1
	push_error("STORYBOOK ASSET FAILED: %s" % description)
