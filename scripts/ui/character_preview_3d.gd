class_name CharacterPreview3D
extends SubViewport

var definition: CharacterDefinition
var _animation_player: AnimationPlayer
var _animation_accumulator := 0.0
const STOP_MOTION_STEP := 1.0 / 12.0


func setup(character_definition: CharacterDefinition) -> void:
	definition = character_definition
	size = Vector2i(256, 192)
	own_world_3d = true
	transparent_bg = true
	render_target_update_mode = SubViewport.UPDATE_ALWAYS
	msaa_3d = Viewport.MSAA_2X

	var world := Node3D.new()
	world.name = "ClayPreviewWorld"
	add_child(world)
	var model_scene := definition.load_model_scene()
	if model_scene != null:
		var model := model_scene.instantiate() as Node3D
		if model != null:
			world.add_child(model)
			_normalize_model(model)
			ClayMaterialLibrary.apply_to_model(model, definition.theme_color)
			_animation_player = _find_animation_player(model)
			_start_idle_animation()
	else:
		_add_placeholder(world)

	var ground := MeshInstance3D.new()
	var ground_mesh := CylinderMesh.new()
	ground_mesh.top_radius = 0.68
	ground_mesh.bottom_radius = 0.72
	ground_mesh.height = 0.08
	ground_mesh.radial_segments = 20
	ground.mesh = ground_mesh
	ground.position.y = -0.04
	ground.material_override = ClayMaterialLibrary.make(definition.theme_color.lightened(0.62), 0.94)
	world.add_child(ground)

	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-48.0, -28.0, 0.0)
	light.light_color = Color("#ffe8c7")
	light.light_energy = 1.18
	light.shadow_enabled = false
	world.add_child(light)

	var environment_node := WorldEnvironment.new()
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color(0, 0, 0, 0)
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color("#f4e7d1")
	environment.ambient_light_energy = 0.42
	environment_node.environment = environment
	world.add_child(environment_node)

	var camera := Camera3D.new()
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = 1.48
	camera.position = Vector3(1.45, 1.15, 3.0)
	camera.look_at_from_position(camera.position, Vector3(0.0, 0.62, 0.0))
	camera.current = true
	world.add_child(camera)


func _process(delta: float) -> void:
	if not is_instance_valid(_animation_player):
		return
	_animation_accumulator += minf(delta, STOP_MOTION_STEP * 3.0)
	while _animation_accumulator >= STOP_MOTION_STEP:
		_animation_accumulator -= STOP_MOTION_STEP
		_animation_player.advance(STOP_MOTION_STEP)


func _normalize_model(model: Node3D) -> void:
	var bounds := _calculate_bounds(model)
	var horizontal := maxf(bounds.size.x, bounds.size.z)
	if bounds.size.y <= 0.001 or horizontal <= 0.001:
		return
	var model_scale := minf(1.25 / (bounds.size.y * 0.9), 0.92 / (horizontal * 1.08))
	model.scale = Vector3(1.08, 0.9, 1.08) * model_scale
	model.position.y = -bounds.position.y * model_scale * 0.9
	model.rotation_degrees.y = definition.yaw_offset_degrees


func _calculate_bounds(root: Node3D) -> AABB:
	var minimum := Vector3(INF, INF, INF)
	var maximum := Vector3(-INF, -INF, -INF)
	var found := false
	var stack: Array[Dictionary] = [{"node": root, "transform": Transform3D.IDENTITY}]
	while not stack.is_empty():
		var item: Dictionary = stack.pop_back()
		var current := item["node"] as Node3D
		var current_transform := item["transform"] as Transform3D
		if current != root:
			current_transform = current_transform * current.transform
		if current is MeshInstance3D and (current as MeshInstance3D).mesh != null:
			var box := (current as MeshInstance3D).get_aabb()
			for x in [box.position.x, box.end.x]:
				for y in [box.position.y, box.end.y]:
					for z in [box.position.z, box.end.z]:
						var point := current_transform * Vector3(x, y, z)
						minimum = minimum.min(point)
						maximum = maximum.max(point)
						found = true
		for child in current.get_children():
			if child is Node3D:
				stack.append({"node": child, "transform": current_transform})
	if not found:
		return AABB(Vector3.ZERO, Vector3.ONE)
	return AABB(minimum, maximum - minimum)


func _find_animation_player(root: Node) -> AnimationPlayer:
	if root is AnimationPlayer:
		return root as AnimationPlayer
	for child in root.get_children():
		var found := _find_animation_player(child)
		if found != null:
			return found
	return null


func _start_idle_animation() -> void:
	if not is_instance_valid(_animation_player):
		return
	for alias in definition.idle_animation_aliases:
		for animation_name in _animation_player.get_animation_list():
			if String(animation_name).to_lower().contains(alias.to_lower()):
				_animation_player.callback_mode_process = AnimationMixer.ANIMATION_CALLBACK_MODE_PROCESS_MANUAL
				_animation_player.play(animation_name, 0.0)
				_animation_player.advance(0.0)
				return


func _add_placeholder(parent: Node3D) -> void:
	var mesh_instance := MeshInstance3D.new()
	var mesh := CapsuleMesh.new()
	mesh.radius = 0.32
	mesh.height = 1.15
	mesh_instance.mesh = mesh
	mesh_instance.position.y = 0.58
	mesh_instance.material_override = ClayMaterialLibrary.make(definition.theme_color, 0.92)
	parent.add_child(mesh_instance)
