class_name ClayLobbyDiorama3D
extends SubViewport
## Live, self-contained clay miniature used behind the lobby UI.

const STOP_MOTION_STEP := 1.0 / 12.0

var _animation_players: Array[AnimationPlayer] = []
var _animation_accumulator := 0.0
var _clock_hand: Node3D


func _ready() -> void:
	own_world_3d = true
	transparent_bg = false
	render_target_update_mode = SubViewport.UPDATE_ALWAYS
	msaa_3d = Viewport.MSAA_2X
	_build_world()


func _process(delta: float) -> void:
	_animation_accumulator += minf(delta, STOP_MOTION_STEP * 3.0)
	while _animation_accumulator >= STOP_MOTION_STEP:
		_animation_accumulator -= STOP_MOTION_STEP
		if is_instance_valid(_clock_hand):
			_clock_hand.rotate_z(deg_to_rad(-1.0))
		for player in _animation_players:
			if is_instance_valid(player):
				player.advance(STOP_MOTION_STEP)


func _build_world() -> void:
	var world := Node3D.new()
	world.name = "ClayLobbyWorld"
	add_child(world)
	_build_environment(world)
	_build_ground(world)
	_build_clock_tower(world)
	_build_garden(world)
	_add_character(world, "builder", Vector3(-3.1, 0.05, 0.5), 22.0)
	_add_character(world, "chef", Vector3(-0.65, 0.05, 0.0), -10.0)
	_add_character(world, "wizard", Vector3(2.05, 0.05, 0.55), -24.0)


func _build_environment(world: Node3D) -> void:
	var environment_node := WorldEnvironment.new()
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = ClayMaterialLibrary.SKY
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = ClayMaterialLibrary.CREAM
	environment.ambient_light_energy = 0.34
	environment.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	environment_node.environment = environment
	world.add_child(environment_node)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-56.0, -34.0, 0.0)
	sun.light_color = Color("#ffe4bd")
	sun.light_energy = 1.08
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = 24.0
	world.add_child(sun)

	var camera := Camera3D.new()
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = 9.4
	camera.position = Vector3(0.4, 6.8, 9.0)
	camera.look_at_from_position(camera.position, Vector3(0.0, 0.8, -0.1))
	camera.current = true
	world.add_child(camera)


func _build_ground(world: Node3D) -> void:
	var island := MeshInstance3D.new()
	var island_mesh := CylinderMesh.new()
	island_mesh.top_radius = 5.7
	island_mesh.bottom_radius = 5.35
	island_mesh.height = 0.46
	island_mesh.radial_segments = 28
	island.mesh = island_mesh
	island.scale.z = 0.68
	island.position.y = -0.22
	island.material_override = ClayMaterialLibrary.make(ClayMaterialLibrary.GRASS, 0.94)
	world.add_child(island)

	var path := MeshInstance3D.new()
	var path_mesh := BoxMesh.new()
	path_mesh.size = Vector3(8.4, 0.09, 1.12)
	path.mesh = path_mesh
	path.position = Vector3(0.0, 0.03, 0.75)
	path.rotation_degrees.y = -4.0
	path.material_override = ClayMaterialLibrary.make(ClayMaterialLibrary.CREAM.darkened(0.06), 0.93)
	world.add_child(path)


func _build_clock_tower(world: Node3D) -> void:
	var root := Node3D.new()
	root.position = Vector3(4.0, 0.0, -0.45)
	root.rotation_degrees.y = -8.0
	world.add_child(root)

	var tower := MeshInstance3D.new()
	var tower_mesh := BoxMesh.new()
	tower_mesh.size = Vector3(1.45, 2.75, 1.45)
	tower.mesh = tower_mesh
	tower.position.y = 1.375
	tower.material_override = ClayMaterialLibrary.make(ClayMaterialLibrary.CREAM, 0.92)
	root.add_child(tower)

	var roof := MeshInstance3D.new()
	var roof_mesh := CylinderMesh.new()
	roof_mesh.top_radius = 0.05
	roof_mesh.bottom_radius = 1.02
	roof_mesh.height = 1.2
	roof_mesh.radial_segments = 8
	roof.mesh = roof_mesh
	roof.position.y = 3.35
	roof.material_override = ClayMaterialLibrary.make(ClayMaterialLibrary.SKY.darkened(0.12), 0.9)
	root.add_child(roof)

	var face := MeshInstance3D.new()
	var face_mesh := CylinderMesh.new()
	face_mesh.top_radius = 0.38
	face_mesh.bottom_radius = 0.38
	face_mesh.height = 0.07
	face_mesh.radial_segments = 18
	face.mesh = face_mesh
	face.position = Vector3(0.0, 2.05, 0.75)
	face.rotation.x = PI * 0.5
	face.material_override = ClayMaterialLibrary.make(ClayMaterialLibrary.CREAM.lightened(0.05), 0.9)
	root.add_child(face)

	_clock_hand = Node3D.new()
	_clock_hand.position = Vector3(0.0, 2.05, 0.8)
	root.add_child(_clock_hand)
	var minute := MeshInstance3D.new()
	var minute_mesh := BoxMesh.new()
	minute_mesh.size = Vector3(0.045, 0.28, 0.045)
	minute.mesh = minute_mesh
	minute.position.y = 0.1
	minute.material_override = ClayMaterialLibrary.make(ClayMaterialLibrary.CHARCOAL, 0.92)
	_clock_hand.add_child(minute)
	var hour := MeshInstance3D.new()
	var hour_mesh := BoxMesh.new()
	hour_mesh.size = Vector3(0.2, 0.045, 0.045)
	hour.mesh = hour_mesh
	hour.position.x = 0.08
	hour.material_override = ClayMaterialLibrary.make(ClayMaterialLibrary.CHARCOAL, 0.92)
	_clock_hand.add_child(hour)


func _build_garden(world: Node3D) -> void:
	for index in range(22):
		var angle := float(index) * TAU / 22.0
		var radius := 4.45 + float((index * 17) % 5) * 0.08
		var shrub := MeshInstance3D.new()
		var shrub_mesh := SphereMesh.new()
		shrub_mesh.radius = 0.28 + float(index % 3) * 0.045
		shrub_mesh.height = shrub_mesh.radius * 1.65
		shrub_mesh.radial_segments = 8
		shrub_mesh.rings = 4
		shrub.mesh = shrub_mesh
		shrub.position = Vector3(cos(angle) * radius, 0.24, sin(angle) * radius * 0.65)
		shrub.scale = Vector3(1.2, 0.82 + float(index % 2) * 0.12, 1.0)
		var color := ClayMaterialLibrary.GRASS.lightened(float(index % 3) * 0.045)
		shrub.material_override = ClayMaterialLibrary.make(color, 0.94)
		world.add_child(shrub)
	for index in range(13):
		var flower := MeshInstance3D.new()
		var flower_mesh := SphereMesh.new()
		flower_mesh.radius = 0.11
		flower_mesh.height = 0.16
		flower_mesh.radial_segments = 8
		flower_mesh.rings = 4
		flower.mesh = flower_mesh
		flower.position = Vector3(-4.2 + float((index * 19) % 77) * 0.11, 0.13, -1.5 + float((index * 23) % 21) * 0.13)
		var flower_color := ClayMaterialLibrary.MUSTARD if index % 2 == 0 else ClayMaterialLibrary.TERRACOTTA
		flower.material_override = ClayMaterialLibrary.make(flower_color, 0.91)
		world.add_child(flower)


func _add_character(world: Node3D, character_id: String, location: Vector3, yaw: float) -> void:
	var definition := CharacterCatalog.get_definition(character_id)
	var scene := definition.load_model_scene()
	if scene == null:
		return
	var model := scene.instantiate() as Node3D
	if model == null:
		return
	world.add_child(model)
	var bounds := _calculate_bounds(model)
	var horizontal := maxf(bounds.size.x, bounds.size.z)
	var model_scale := minf(1.46 / maxf(bounds.size.y * 0.9, 0.001), 1.02 / maxf(horizontal * 1.08, 0.001))
	model.scale = Vector3(1.08, 0.9, 1.08) * model_scale
	model.position = location
	model.position.y -= bounds.position.y * model_scale * 0.9
	model.rotation_degrees.y = definition.yaw_offset_degrees + yaw
	ClayMaterialLibrary.apply_to_model(model, definition.theme_color)
	var player := _find_animation_player(model)
	if player == null:
		return
	for animation_name in player.get_animation_list():
		if String(animation_name).to_lower().contains("idle"):
			player.callback_mode_process = AnimationMixer.ANIMATION_CALLBACK_MODE_PROCESS_MANUAL
			player.play(animation_name, 0.0)
			player.advance(0.0)
			_animation_players.append(player)
			return


func _find_animation_player(root: Node) -> AnimationPlayer:
	if root is AnimationPlayer:
		return root as AnimationPlayer
	for child in root.get_children():
		var found := _find_animation_player(child)
		if found != null:
			return found
	return null


func _calculate_bounds(root: Node3D) -> AABB:
	var minimum := Vector3(INF, INF, INF)
	var maximum := Vector3(-INF, -INF, -INF)
	var found := false
	var root_inverse := root.global_transform.affine_inverse()
	for node in root.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := node as MeshInstance3D
		if mesh_instance.mesh == null:
			continue
		var box := mesh_instance.get_aabb()
		var relative := root_inverse * mesh_instance.global_transform
		for x in [box.position.x, box.end.x]:
			for y in [box.position.y, box.end.y]:
				for z in [box.position.z, box.end.z]:
					var point := relative * Vector3(x, y, z)
					minimum = minimum.min(point)
					maximum = maximum.max(point)
					found = true
	if not found:
		return AABB(Vector3.ZERO, Vector3.ONE)
	return AABB(minimum, maximum - minimum)
