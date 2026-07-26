class_name StorybookLobbyDiorama3D
extends SubViewport
## Live forest clearing using the production Blender environment and all animals.

const STOP_MOTION_STEP := 1.0 / 8.0
const LOBBY_SCENE_PATH := \
	"res://assets/models/environment/storybook/storybook_lobby.glb"
const CHARACTER_POSITIONS: Dictionary = {
	"cat": Vector3(-2.6, 0.13, 1.4),
	"dog": Vector3(-0.9, 0.13, 1.7),
	"rabbit": Vector3(0.9, 0.13, 1.7),
	"bear": Vector3(2.6, 0.13, 1.4),
	"fox": Vector3(-2.6, 0.13, -0.6),
	"raccoon": Vector3(-0.9, 0.13, -0.9),
	"penguin": Vector3(0.9, 0.13, -0.9),
	"capybara": Vector3(2.6, 0.13, -0.6),
}

var _animation_players: Array[AnimationPlayer] = []
var _animation_accumulator: float = 0.0


func _ready() -> void:
	own_world_3d = true
	transparent_bg = false
	render_target_update_mode = SubViewport.UPDATE_ALWAYS
	msaa_3d = Viewport.MSAA_DISABLED if OS.has_feature("web") else Viewport.MSAA_2X
	_build_world()


func _process(delta: float) -> void:
	_animation_accumulator += minf(delta, STOP_MOTION_STEP * 3.0)
	while _animation_accumulator >= STOP_MOTION_STEP:
		_animation_accumulator -= STOP_MOTION_STEP
		for player: AnimationPlayer in _animation_players:
			if is_instance_valid(player):
				player.advance(STOP_MOTION_STEP)


func _build_world() -> void:
	var world := Node3D.new()
	world.name = "ForestLobbyWorld"
	add_child(world)
	_build_environment(world)
	_add_lobby_scene(world)
	for index: int in range(CharacterCatalog.IDS.size()):
		var character_id: String = CharacterCatalog.IDS[index]
		_add_character(
			world,
			character_id,
			CHARACTER_POSITIONS[character_id] as Vector3,
			PaintPalette.COLOR_IDS[index % PaintPalette.COLOR_IDS.size()],
			index % 3
		)


func _build_environment(world: Node3D) -> void:
	var environment_node := WorldEnvironment.new()
	environment_node.name = "StorybookLobbyEnvironment"
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = StorybookMaterialLibrary.SKY
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = StorybookMaterialLibrary.PAPER
	environment.ambient_light_energy = 0.34
	environment.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	environment_node.environment = environment
	world.add_child(environment_node)

	var sun := DirectionalLight3D.new()
	sun.name = "LobbySun"
	sun.rotation_degrees = Vector3(-58.0, -32.0, 0.0)
	sun.light_color = Color("#ffe4bd")
	sun.light_energy = 1.02
	sun.shadow_enabled = false
	sun.directional_shadow_max_distance = 20.0
	world.add_child(sun)

	var camera := Camera3D.new()
	camera.name = "LobbyCamera"
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = 10.8
	var target := Vector3(0.0, 0.75, 0.0)
	var azimuth := deg_to_rad(MatchSettings.DEFAULT_CAMERA_AZIMUTH)
	var elevation := deg_to_rad(MatchSettings.DEFAULT_CAMERA_ELEVATION)
	var direction := Vector3(
		sin(azimuth) * cos(elevation),
		sin(elevation),
		cos(azimuth) * cos(elevation)
	).normalized()
	camera.look_at_from_position(target + direction * 16.0, target, Vector3.UP)
	camera.current = true
	world.add_child(camera)


func _add_lobby_scene(world: Node3D) -> void:
	var packed := load(LOBBY_SCENE_PATH) as PackedScene
	if packed == null:
		push_warning("Storybook lobby GLB is unavailable.")
		return
	var instance := packed.instantiate() as Node3D
	if instance == null:
		return
	instance.name = "StorybookLobbyEnvironmentModel"
	world.add_child(instance)
	StorybookMaterialLibrary.apply_character_palette(instance, Color.WHITE, [], false)


func _add_character(
		world: Node3D,
		character_id: String,
		location: Vector3,
		color_id: String,
		action_index: int
	) -> void:
	var definition := CharacterCatalog.get_definition(character_id)
	var scene := definition.load_model_scene()
	if scene == null:
		return
	var model := scene.instantiate() as Node3D
	if model == null:
		return
	model.name = "%sLobbyModel" % character_id.capitalize()
	world.add_child(model)
	var bounds := _calculate_bounds(model)
	var horizontal := maxf(bounds.size.x, bounds.size.z)
	var model_scale := minf(
		1.32 / maxf(bounds.size.y, 0.001),
		1.05 / maxf(horizontal, 0.001)
	)
	model.scale = Vector3.ONE * model_scale
	model.position = location
	model.position.y -= bounds.position.y * model_scale
	model.rotation_degrees.y = -8.0 + float(action_index - 1) * 10.0
	StorybookMaterialLibrary.apply_character_palette(
		model,
		PaintPalette.get_color(color_id),
		definition.team_tint_material_names
	)
	var player := _find_animation_player(model)
	if player == null:
		return
	var aliases: Array[String] = (
		definition.move_animation_aliases
		if action_index == 1
		else definition.victory_animation_aliases
		if action_index == 2
		else definition.idle_animation_aliases
	)
	var animation_name := _find_animation(player, aliases)
	if animation_name == &"":
		return
	player.callback_mode_process = AnimationMixer.ANIMATION_CALLBACK_MODE_PROCESS_MANUAL
	player.play(animation_name, 0.0)
	player.advance(0.0)
	_animation_players.append(player)


func _find_animation(player: AnimationPlayer, aliases: Array[String]) -> StringName:
	for alias: String in aliases:
		for animation_name: StringName in player.get_animation_list():
			if String(animation_name).to_lower().contains(alias.to_lower()):
				return animation_name
	return &""


func _find_animation_player(root: Node) -> AnimationPlayer:
	if root is AnimationPlayer:
		return root as AnimationPlayer
	for child: Node in root.get_children():
		var found := _find_animation_player(child)
		if found != null:
			return found
	return null


func _calculate_bounds(root: Node3D) -> AABB:
	var minimum := Vector3(INF, INF, INF)
	var maximum := Vector3(-INF, -INF, -INF)
	var found := false
	var root_inverse := root.global_transform.affine_inverse()
	for node: Node in root.find_children("*", "MeshInstance3D", true, false):
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
		return AABB(Vector3.ZERO, Vector3.ONE)
	return AABB(minimum, maximum - minimum)
