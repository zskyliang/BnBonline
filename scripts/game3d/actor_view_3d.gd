class_name ActorView3D
extends Node3D
## Read-only storybook visual and animation coordinator for one 2D logic actor.

const TARGET_HEIGHT := 1.3
const TARGET_WIDTH := 1.05
const STOP_MOTION_STEP := 1.0 / 8.0
const WALK_CYCLE_DISTANCE := 64.0
const MIN_WALK_PHASE_ADVANCE := 0.08
const MAX_WALK_PHASE_ADVANCE := 0.24
const MOVEMENT_GRACE_SECONDS := 0.12
const LOOPING_ACTIONS: Array[StringName] = [&"Idle", &"Waddle", &"Trapped"]
const ONE_SHOT_ACTIONS: Array[StringName] = [&"PlaceBubble", &"Victory"]

var actor: GameActor
var definition: CharacterDefinition
var color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID

var _visual_pivot: Node3D
var _model_root: Node3D
var _animation_player: AnimationPlayer
var _skeleton: Skeleton3D
var _trap_sphere: MeshInstance3D
var _resolved_animations: Dictionary = {}
var _current_animation: StringName = &""
var _current_action: StringName = &"Idle"
var _one_shot_action: StringName = &""
var _animation_accumulator: float = 0.0
var _animation_time: float = 0.0
var _stop_motion_time: float = 0.0
var _walk_cycle_phase: float = 0.0
var _walk_distance_since_pose: float = 0.0
var _last_direction := Vector2.DOWN
var _last_logic_position: Vector2
var _has_last_logic_position: bool = false
var _movement_grace_remaining: float = 0.0
var _is_walking: bool = false


func bind_actor(
		logic_actor: GameActor,
		character_id: String,
		new_color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID
	) -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	actor = logic_actor
	definition = CharacterCatalog.get_definition(character_id)
	color_id = new_color_id
	name = "%sView3D" % definition.id.capitalize()
	_build_visual()
	if is_instance_valid(actor):
		_last_logic_position = actor.position
		_has_last_logic_position = true
		actor.tree_exiting.connect(queue_free, CONNECT_ONE_SHOT)


func play_action(action: StringName) -> bool:
	var animation_name: StringName = _resolved_animations.get(action, &"") as StringName
	if animation_name == &"":
		return false
	if action in ONE_SHOT_ACTIONS:
		_one_shot_action = action
	_current_action = action
	_switch_animation(animation_name)
	return true


func get_current_action() -> StringName:
	return _current_action


func get_walk_cycle_phase() -> float:
	return _walk_cycle_phase


func has_action(action: StringName) -> bool:
	return (_resolved_animations.get(action, &"") as StringName) != &""


func _process(delta: float) -> void:
	if not is_instance_valid(actor):
		queue_free()
		return
	var travelled_distance := 0.0
	if _has_last_logic_position:
		travelled_distance = actor.position.distance_to(_last_logic_position)
	_last_logic_position = actor.position
	_has_last_logic_position = true
	position = GameConstants.logic_to_world_3d(actor.position, 0.04)
	visible = actor.visible
	if not visible:
		return
	var planar_velocity := actor.velocity
	var can_walk := planar_velocity.length_squared() > 1.0 \
		and not actor.stats.is_dead and not actor.stats.is_trapped
	if can_walk and travelled_distance > 0.001:
		_walk_distance_since_pose += travelled_distance
		_movement_grace_remaining = MOVEMENT_GRACE_SECONDS
	else:
		_movement_grace_remaining = maxf(0.0, _movement_grace_remaining - delta)
	_is_walking = can_walk and (
		travelled_distance > 0.001 or _movement_grace_remaining > 0.0
	)
	if _is_walking:
		_last_direction = planar_velocity.normalized()
		rotation.y = atan2(_last_direction.x, _last_direction.y) \
			+ deg_to_rad(definition.yaw_offset_degrees)
	_advance_stop_motion(delta)
	_apply_outer_animation()


func _build_visual() -> void:
	_visual_pivot = Node3D.new()
	_visual_pivot.name = "CharacterPivot"
	add_child(_visual_pivot)
	_build_shadow()
	_build_trap_sphere()
	var scene := definition.load_model_scene()
	if scene == null:
		_build_placeholder()
		return
	var instance := scene.instantiate()
	if not instance is Node3D:
		instance.queue_free()
		_build_placeholder()
		return
	_model_root = instance as Node3D
	_model_root.name = "Model"
	_visual_pivot.add_child(_model_root)
	_normalize_model()
	StorybookMaterialLibrary.apply_character_palette(
		_model_root,
		PaintPalette.get_color(color_id),
		definition.team_tint_material_names
	)
	_find_animation_nodes(_model_root)
	_resolve_animations()


func _build_placeholder() -> void:
	_model_root = Node3D.new()
	_model_root.name = "StorybookPlaceholder"
	_visual_pivot.add_child(_model_root)
	var body := MeshInstance3D.new()
	var body_mesh := CapsuleMesh.new()
	body_mesh.radius = 0.36
	body_mesh.height = 0.88
	body_mesh.radial_segments = 12
	body.mesh = body_mesh
	body.position.y = 0.45
	body.material_override = StorybookMaterialLibrary.make(definition.theme_color)
	_model_root.add_child(body)
	var head := MeshInstance3D.new()
	var head_mesh := SphereMesh.new()
	head_mesh.radius = 0.32
	head_mesh.height = 0.64
	head_mesh.radial_segments = 12
	head_mesh.rings = 6
	head.mesh = head_mesh
	head.position.y = 1.0
	head.scale = Vector3(1.08, 0.95, 1.0)
	head.material_override = StorybookMaterialLibrary.make(definition.accent_color)
	_model_root.add_child(head)


func _build_shadow() -> void:
	var shadow := MeshInstance3D.new()
	shadow.name = "ContactShadow"
	var mesh := CylinderMesh.new()
	mesh.top_radius = 0.4
	mesh.bottom_radius = 0.4
	mesh.height = 0.012
	mesh.radial_segments = 20
	shadow.mesh = mesh
	shadow.position.y = 0.01
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.12, 0.1, 0.11, 0.24)
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	shadow.material_override = material
	_visual_pivot.add_child(shadow)


func _build_trap_sphere() -> void:
	_trap_sphere = MeshInstance3D.new()
	_trap_sphere.name = "TrapBubble"
	var sphere := SphereMesh.new()
	sphere.radius = 0.67
	sphere.height = 1.34
	sphere.radial_segments = 20
	sphere.rings = 10
	_trap_sphere.mesh = sphere
	_trap_sphere.position.y = 0.67
	_trap_sphere.scale = Vector3(1.04, 0.96, 0.98)
	var bubble_material := StandardMaterial3D.new()
	bubble_material.albedo_color = Color(0.62, 0.88, 0.92, 0.23)
	bubble_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	bubble_material.metallic = 0.0
	bubble_material.metallic_specular = 0.12
	bubble_material.roughness = 0.18
	var ink_outline := StandardMaterial3D.new()
	ink_outline.albedo_color = Color(0.16, 0.29, 0.31, 0.58)
	ink_outline.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	ink_outline.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	ink_outline.cull_mode = BaseMaterial3D.CULL_FRONT
	ink_outline.grow = true
	ink_outline.grow_amount = 0.018
	bubble_material.next_pass = ink_outline
	_trap_sphere.material_override = bubble_material
	_trap_sphere.visible = false
	add_child(_trap_sphere)


func _normalize_model() -> void:
	var bounds := _calculate_bounds(_model_root)
	if bounds.size.y <= 0.001:
		return
	var horizontal_size := maxf(bounds.size.x, bounds.size.z)
	var normalization_scale := minf(
		TARGET_HEIGHT / bounds.size.y,
		TARGET_WIDTH / maxf(horizontal_size, 0.001)
	) * definition.scale_multiplier
	_model_root.scale = Vector3.ONE * normalization_scale
	_model_root.position.y = -bounds.position.y * normalization_scale
	_model_root.rotation_degrees.y = definition.yaw_offset_degrees


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
					var corner := relative * Vector3(x, y, z)
					minimum = minimum.min(corner)
					maximum = maximum.max(corner)
					found = true
	if not found:
		return AABB(Vector3.ZERO, Vector3.ONE)
	return AABB(minimum, maximum - minimum)


func _find_animation_nodes(root: Node) -> void:
	if root is AnimationPlayer and not is_instance_valid(_animation_player):
		_animation_player = root as AnimationPlayer
	if root is Skeleton3D and not is_instance_valid(_skeleton):
		_skeleton = root as Skeleton3D
	for child: Node in root.get_children():
		_find_animation_nodes(child)


func _resolve_animations() -> void:
	if not is_instance_valid(_animation_player):
		return
	_resolved_animations = {
		&"Idle": _find_animation(definition.idle_animation_aliases),
		&"Waddle": _find_animation(definition.move_animation_aliases),
		&"PlaceBubble": _find_animation(definition.place_bubble_animation_aliases),
		&"Trapped": _find_animation(definition.trapped_animation_aliases),
		&"Defeat": _find_animation(definition.defeat_animation_aliases),
		&"Victory": _find_animation(definition.victory_animation_aliases),
	}
	if (_resolved_animations[&"Idle"] as StringName) == &"":
		_resolved_animations[&"Idle"] = _resolved_animations[&"Waddle"]
	if (_resolved_animations[&"Waddle"] as StringName) == &"":
		_resolved_animations[&"Waddle"] = _resolved_animations[&"Idle"]
	_animation_player.callback_mode_process = AnimationMixer.ANIMATION_CALLBACK_MODE_PROCESS_MANUAL
	_switch_animation(_resolved_animations[&"Idle"] as StringName)


func _find_animation(aliases: Array[String]) -> StringName:
	for alias: String in aliases:
		for animation_name: StringName in _animation_player.get_animation_list():
			var normalized := String(animation_name).to_lower()
			if normalized.ends_with(alias.to_lower()) or normalized.contains(alias.to_lower()):
				return animation_name
	return &""


func _desired_action() -> StringName:
	if actor.stats.is_dead and has_action(&"Defeat"):
		return &"Defeat"
	if actor.stats.is_trapped and has_action(&"Trapped"):
		return &"Trapped"
	if _one_shot_action != &"":
		return _one_shot_action
	if _is_walking:
		return &"Waddle"
	return &"Idle"


func _advance_stop_motion(delta: float) -> void:
	if not is_instance_valid(_animation_player):
		return
	_animation_accumulator += minf(delta, STOP_MOTION_STEP * 3.0)
	while _animation_accumulator >= STOP_MOTION_STEP:
		_animation_accumulator -= STOP_MOTION_STEP
		_stop_motion_time += STOP_MOTION_STEP
		var desired := _desired_action()
		var desired_animation: StringName = _resolved_animations.get(desired, &"") as StringName
		if desired_animation == &"":
			desired = &"Idle"
			desired_animation = _resolved_animations.get(desired, &"") as StringName
		if desired != _current_action or desired_animation != _current_animation:
			_current_action = desired
			_switch_animation(desired_animation)
		if _current_action == &"Waddle" and _is_walking:
			_advance_walk_pose()
		elif _current_action in LOOPING_ACTIONS:
			_advance_looped_pose(STOP_MOTION_STEP)
		else:
			_advance_one_shot_or_held_pose(STOP_MOTION_STEP)


func _switch_animation(animation_name: StringName) -> void:
	if animation_name == &"" or not is_instance_valid(_animation_player):
		return
	_current_animation = animation_name
	_animation_time = 0.0
	_animation_player.play(animation_name, 0.0)
	_animation_player.seek(0.0, true)
	_animation_player.advance(0.0)


func _advance_walk_pose() -> void:
	if _walk_distance_since_pose > 0.001:
		var phase_advance := clampf(
			_walk_distance_since_pose / WALK_CYCLE_DISTANCE,
			MIN_WALK_PHASE_ADVANCE,
			MAX_WALK_PHASE_ADVANCE
		)
		_walk_cycle_phase = fposmod(_walk_cycle_phase + phase_advance, 1.0)
		_walk_distance_since_pose = 0.0
	var animation := _animation_player.get_animation(_current_animation)
	if animation == null or animation.length <= 0.001:
		return
	_animation_player.seek(_walk_cycle_phase * animation.length, true)
	_animation_player.advance(0.0)


func _advance_looped_pose(step: float) -> void:
	var animation := _animation_player.get_animation(_current_animation)
	if animation == null or animation.length <= 0.001:
		return
	_animation_time = fposmod(_animation_time + step, animation.length)
	_animation_player.seek(_animation_time, true)
	_animation_player.advance(0.0)


func _advance_one_shot_or_held_pose(step: float) -> void:
	var animation := _animation_player.get_animation(_current_animation)
	if animation == null or animation.length <= 0.001:
		_one_shot_action = &""
		return
	_animation_time = minf(_animation_time + step, animation.length)
	_animation_player.seek(_animation_time, true)
	_animation_player.advance(0.0)
	if _current_action in ONE_SHOT_ACTIONS and _animation_time >= animation.length:
		_one_shot_action = &""


func _apply_outer_animation() -> void:
	_trap_sphere.visible = actor.stats.is_trapped
	if actor.stats.is_trapped:
		var trap_color_id: String = actor.color_id
		if is_instance_valid(actor.last_attacker):
			trap_color_id = actor.last_attacker.color_id
		var trap_color := PaintPalette.get_color(trap_color_id)
		var trap_material := _trap_sphere.material_override as StandardMaterial3D
		if trap_material != null:
			trap_material.albedo_color = Color(trap_color, 0.24)
	var bob := 0.0
	if actor.stats.is_trapped:
		bob = sin(_stop_motion_time * 5.2) * 0.06 + 0.12
	elif _is_walking:
		bob = absf(sin(_walk_cycle_phase * TAU)) * 0.018
	elif _one_shot_action == &"" and not actor.stats.is_dead:
		bob = sin(_stop_motion_time * 2.4) * 0.012
	_visual_pivot.position.y = bob
	if actor.stats.is_invincible() and not actor.stats.is_dead:
		_visual_pivot.visible = int(Time.get_ticks_msec() / 90) % 2 == 0
	else:
		_visual_pivot.visible = true
