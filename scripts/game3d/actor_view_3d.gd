class_name ActorView3D
extends Node3D
## Read-only four-direction ImageGen Sprite3D coordinator.

const STOP_MOTION_STEP := 1.0 / 8.0
const MOVEMENT_GRACE_SECONDS := 0.12
const CONTACT_ROLL_DEGREES := 6.0
## Dog is the approved battle-size reference. Every species reuses its
## current pixel scale so normalized ImageGen frames read as one cast.
const SIZE_REFERENCE_CHARACTER_ID := "dog"
const TARGET_IDLE_CELL_WIDTH := 1.0
const FOOT_CLEARANCE := 0.025
const MIN_TRAP_PIXEL_SIZE := 0.00415
const FRAME_HEIGHT_NORMALIZED_CHARACTER_ID := "penguin"
const FRAME_HEIGHT_NORMALIZED_ACTION := &"WalkUp"

var actor: GameActor
var definition: CharacterDefinition
var color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID

var _sprite_set: CharacterSpriteSet
var _visual_pivot: Node3D
var _character_sprite: Sprite3D
var _character_material: ShaderMaterial
var _trap_sprite: Sprite3D
var _team_ring: MeshInstance3D
var _contact_shadow: MeshInstance3D
var _current_action: StringName = &"Idle"
var _current_frame := 0
var _last_facing: StringName = &"down"
var _animation_accumulator := 0.0
var _stop_motion_time := 0.0
var _last_logic_position: Vector2
var _has_last_logic_position := false
var _movement_grace_remaining := 0.0
var _is_walking := false
var _battle_pixel_size := 0.00265


func bind_actor(
		logic_actor: GameActor,
		character_id: String,
		new_color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID
	) -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	actor = logic_actor
	definition = CharacterCatalog.get_definition(character_id)
	_sprite_set = definition.load_sprite_set()
	color_id = new_color_id
	name = "%sSpriteView3D" % definition.id.capitalize()
	_build_visual()
	if is_instance_valid(actor):
		_last_logic_position = actor.position
		_has_last_logic_position = true
		_last_facing = _facing_name(actor.get_facing())
		actor.tree_exiting.connect(queue_free, CONNECT_ONE_SHOT)


func get_current_action() -> StringName:
	return _current_action


func get_current_frame() -> int:
	return _current_frame


func get_current_facing() -> StringName:
	return _last_facing


func get_walk_pose_snapshot() -> Dictionary:
	var planted_foot := &""
	if _current_action in CharacterSpriteSet.WALK_ACTIONS:
		if _current_frame == 0:
			planted_foot = &"left"
		elif _current_frame == 2:
			planted_foot = &"right"
	return {
		"action": _current_action,
		"facing": _last_facing,
		"frame": _current_frame,
		"planted_foot": planted_foot,
		"body_roll_degrees": rad_to_deg(_visual_pivot.rotation.z),
		"body_height": _visual_pivot.position.y,
	}


func get_min_idle_cell_width() -> float:
	if _sprite_set == null:
		return 0.0
	var minimum_width := INF
	for facing: StringName in CharacterSpriteSet.DIRECTIONS:
		var texture := _sprite_set.texture_for(&"Idle", 0, facing)
		var visible_width := _visible_texture_width(texture)
		if visible_width > 0:
			minimum_width = minf(
				minimum_width,
				float(visible_width) * _battle_pixel_size
			)
	return 0.0 if is_inf(minimum_width) else minimum_width


func get_max_idle_cell_height() -> float:
	if _sprite_set == null:
		return 0.0
	var maximum_height := 0.0
	for facing: StringName in CharacterSpriteSet.DIRECTIONS:
		var texture := _sprite_set.texture_for(&"Idle", 0, facing)
		var visible_height := _visible_texture_height(texture)
		maximum_height = maxf(
			maximum_height,
			float(visible_height) * _battle_pixel_size
		)
	return maximum_height


func get_battle_pixel_size() -> float:
	return _battle_pixel_size


func get_visible_frame_cell_height() -> float:
	if not is_instance_valid(_character_sprite) \
			or _character_sprite.texture == null:
		return 0.0
	return float(_visible_texture_height(_character_sprite.texture)) \
		* _character_sprite.pixel_size \
		* _character_sprite.scale.y


func get_character_feet_clearance() -> float:
	if not is_instance_valid(_character_sprite) \
			or _character_sprite.texture == null:
		return -INF
	return _character_sprite.position.y - (
		float(_character_sprite.texture.get_height())
		* _character_sprite.pixel_size
		* _character_sprite.scale.y
		* 0.5
	)


func has_action(action: StringName) -> bool:
	return _sprite_set != null and _sprite_set.has_action(action)


func set_color_id(new_color_id: String) -> void:
	if not PaintPalette.is_valid_color_id(new_color_id):
		return
	color_id = new_color_id
	if is_instance_valid(_character_material):
		_character_material.set_shader_parameter(
			"team_color",
			PaintPalette.get_color(color_id)
		)
	if is_instance_valid(_team_ring):
		_team_ring.material_override = StorybookMaterialLibrary.make(
			PaintPalette.get_color(color_id),
			0.9,
			false,
			0.08
		)


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

	var can_walk := actor.velocity.length_squared() > 1.0 \
		and not actor.stats.is_dead \
		and not actor.stats.is_trapped
	if can_walk and travelled_distance > 0.001:
		_movement_grace_remaining = MOVEMENT_GRACE_SECONDS
	else:
		_movement_grace_remaining = maxf(
			0.0,
			_movement_grace_remaining - delta
		)
	_is_walking = can_walk and (
		travelled_distance > 0.001 or _movement_grace_remaining > 0.0
	)
	if _is_walking:
		_last_facing = _facing_name(actor.get_facing())

	_animation_accumulator += minf(delta, STOP_MOTION_STEP * 3.0)
	while _animation_accumulator >= STOP_MOTION_STEP:
		_animation_accumulator -= STOP_MOTION_STEP
		_stop_motion_time += STOP_MOTION_STEP
		var desired_action := _desired_action()
		if desired_action != _current_action:
			_set_action(desired_action)
		elif desired_action in CharacterSpriteSet.WALK_ACTIONS:
			_current_frame = (
				_current_frame + 1
			) % CharacterSpriteSet.WALK_FRAME_COUNT
			_apply_frame_texture()
		_apply_stop_motion_pose()

	_apply_visibility_state()


func _build_visual() -> void:
	_battle_pixel_size = _calculate_battle_pixel_size()
	_visual_pivot = Node3D.new()
	_visual_pivot.name = "DirectionalPaperPuppetPivot"
	add_child(_visual_pivot)
	_build_contact_shadow()
	_build_team_ring()

	_character_sprite = Sprite3D.new()
	_character_sprite.name = "ImageGenDirectionalCharacterSprite"
	_character_sprite.centered = true
	_character_sprite.pixel_size = _battle_pixel_size
	StorybookMaterialLibrary.configure_billboard(_character_sprite)
	_visual_pivot.add_child(_character_sprite)

	_trap_sprite = Sprite3D.new()
	_trap_sprite.name = "TrapBubbleSprite"
	var trap_texture := load(
		"res://assets/art/storybook25d/effects/trap_bubble.png"
	) as Texture2D
	_trap_sprite.texture = trap_texture
	_trap_sprite.centered = true
	_trap_sprite.pixel_size = maxf(
		MIN_TRAP_PIXEL_SIZE,
		_battle_pixel_size * 1.08
	)
	_trap_sprite.position = Vector3(
		0.0,
		_grounded_center_height(
			trap_texture,
			_trap_sprite.pixel_size
		),
		-0.015
	)
	_trap_sprite.modulate = Color(1.0, 1.0, 1.0, 0.82)
	StorybookMaterialLibrary.configure_billboard(_trap_sprite, true)
	_trap_sprite.visible = false
	_visual_pivot.add_child(_trap_sprite)
	_set_action(&"Idle", true)


func _calculate_battle_pixel_size() -> float:
	var reference_definition := CharacterCatalog.get_definition(
		SIZE_REFERENCE_CHARACTER_ID
	)
	var reference_sprite_set := reference_definition.load_sprite_set()
	if reference_sprite_set == null:
		return _sprite_set.pixel_size if _sprite_set != null else 0.00265
	var narrowest_idle_width := _narrowest_idle_visible_width(
		reference_sprite_set
	)
	if narrowest_idle_width <= 0:
		return reference_sprite_set.pixel_size
	return maxf(
		reference_sprite_set.pixel_size,
		TARGET_IDLE_CELL_WIDTH / float(narrowest_idle_width)
	)


func _narrowest_idle_visible_width(
		sprite_set: CharacterSpriteSet
	) -> int:
	if sprite_set == null:
		return 0
	var narrowest_idle_width := 0
	for facing: StringName in CharacterSpriteSet.DIRECTIONS:
		var texture := sprite_set.texture_for(&"Idle", 0, facing)
		var visible_width := _visible_texture_width(texture)
		if visible_width <= 0:
			continue
		if narrowest_idle_width == 0:
			narrowest_idle_width = visible_width
		else:
			narrowest_idle_width = mini(narrowest_idle_width, visible_width)
	return narrowest_idle_width


func _visible_texture_width(texture: Texture2D) -> int:
	if texture == null:
		return 0
	var image := texture.get_image()
	if image == null:
		return texture.get_width()
	return image.get_used_rect().size.x


func _visible_texture_height(texture: Texture2D) -> int:
	if texture == null:
		return 0
	var image := texture.get_image()
	if image == null:
		return texture.get_height()
	return image.get_used_rect().size.y


func _frame_vertical_scale(
		texture: Texture2D,
		frame_facing: StringName
	) -> float:
	if definition == null \
			or definition.id != FRAME_HEIGHT_NORMALIZED_CHARACTER_ID \
			or _current_action != FRAME_HEIGHT_NORMALIZED_ACTION:
		return 1.0
	var reference := _sprite_set.texture_for(&"Idle", 0, frame_facing)
	var reference_height := _visible_texture_height(reference)
	var frame_height := _visible_texture_height(texture)
	if reference_height <= 0 or frame_height <= 0:
		return 1.0
	return float(reference_height) / float(frame_height)


func _grounded_center_height(
		texture: Texture2D,
		pixel_size: float,
		vertical_scale: float = 1.0
	) -> float:
	if texture == null:
		return FOOT_CLEARANCE
	return float(texture.get_height()) \
		* pixel_size \
		* vertical_scale \
		* 0.5 \
		+ FOOT_CLEARANCE


func _build_contact_shadow() -> void:
	_contact_shadow = MeshInstance3D.new()
	_contact_shadow.name = "ContactShadow"
	var mesh := CylinderMesh.new()
	mesh.top_radius = 0.38
	mesh.bottom_radius = 0.38
	mesh.height = 0.008
	mesh.radial_segments = 20
	_contact_shadow.mesh = mesh
	_contact_shadow.position.y = 0.006
	_contact_shadow.material_override = StorybookMaterialLibrary.make(
		Color(0.16, 0.12, 0.1, 0.22)
	)
	add_child(_contact_shadow)


func _build_team_ring() -> void:
	_team_ring = MeshInstance3D.new()
	_team_ring.name = "TeamColorRing"
	var mesh := TorusMesh.new()
	mesh.inner_radius = 0.39
	mesh.outer_radius = 0.435
	mesh.rings = 16
	mesh.ring_segments = 28
	_team_ring.mesh = mesh
	_team_ring.position.y = 0.012
	_team_ring.material_override = StorybookMaterialLibrary.make(
		PaintPalette.get_color(color_id),
		0.9,
		false,
		0.08
	)
	_team_ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(_team_ring)


func _desired_action() -> StringName:
	if actor.stats.is_trapped:
		return &"Trapped"
	if actor.stats.is_dead:
		return &"Idle"
	if not _is_walking:
		return &"Idle"
	match _last_facing:
		&"up":
			return &"WalkUp"
		&"left":
			return &"WalkLeft"
		&"right":
			return &"WalkRight"
		_:
			return &"WalkDown"


func _set_action(action: StringName, force: bool = false) -> void:
	if not force and action == _current_action:
		return
	_current_action = action
	_current_frame = 0
	_apply_frame_texture()


func _apply_frame_texture() -> void:
	if not is_instance_valid(_character_sprite) or _sprite_set == null:
		return
	var frame_facing := &"down" if _current_action == &"Trapped" else _last_facing
	var texture := _sprite_set.texture_for(
		_current_action,
		_current_frame,
		frame_facing
	)
	var mask := _sprite_set.mask_for(
		_current_action,
		_current_frame,
		frame_facing
	)
	if texture == null or mask == null:
		return
	_character_sprite.texture = texture
	var vertical_scale := _frame_vertical_scale(texture, frame_facing)
	_character_sprite.scale = Vector3(1.0, vertical_scale, 1.0)
	_character_sprite.position.y = _grounded_center_height(
		texture,
		_character_sprite.pixel_size,
		vertical_scale
	)
	if not is_instance_valid(_character_material):
		_character_material = StorybookMaterialLibrary.make_sprite_material(
			texture,
			mask,
			PaintPalette.get_color(color_id)
		)
		_character_sprite.material_override = _character_material
	else:
		_character_material.set_shader_parameter("base_texture", texture)
		_character_material.set_shader_parameter("tint_mask", mask)


func _apply_stop_motion_pose() -> void:
	var bob := 0.0
	var roll := 0.0
	var squash := 1.0
	var screen_shift := 0.0
	if _current_action in CharacterSpriteSet.WALK_ACTIONS:
		match _current_frame:
			0:
				bob = 0.02
				roll = deg_to_rad(CONTACT_ROLL_DEGREES)
				squash = 0.98
				screen_shift = 0.045
			1:
				bob = 0.07
				roll = deg_to_rad(2.0)
				squash = 1.025
				screen_shift = 0.015
			2:
				bob = 0.02
				roll = -deg_to_rad(CONTACT_ROLL_DEGREES)
				squash = 0.98
				screen_shift = -0.045
			3:
				bob = 0.07
				roll = -deg_to_rad(2.0)
				squash = 1.025
				screen_shift = -0.015
	elif _current_action == &"Trapped":
		bob = 0.1 + sin(_stop_motion_time * 5.0) * 0.045
		roll = sin(_stop_motion_time * 4.0) * 0.045
	else:
		bob = sin(_stop_motion_time * 2.25) * 0.012
		squash = 1.0 + sin(_stop_motion_time * 2.25) * 0.008

	_visual_pivot.position = Vector3(screen_shift, bob, 0.0)
	_visual_pivot.rotation.z = roll
	_visual_pivot.scale = Vector3(2.0 - squash, squash, 1.0)
	_contact_shadow.scale = Vector3(
		1.0 + bob * 0.45,
		1.0,
		1.0 + bob * 0.45
	)
	_trap_sprite.visible = actor.stats.is_trapped
	if actor.stats.is_trapped:
		var trap_color_id: String = actor.color_id
		if is_instance_valid(actor.last_attacker):
			trap_color_id = actor.last_attacker.color_id
		var trap_color := PaintPalette.get_color(trap_color_id)
		_trap_sprite.modulate = Color(
			lerpf(1.0, trap_color.r, 0.22),
			lerpf(1.0, trap_color.g, 0.22),
			lerpf(1.0, trap_color.b, 0.22),
			0.86
		)


func _facing_name(facing: GameActor.Facing) -> StringName:
	match facing:
		GameActor.Facing.UP:
			return &"up"
		GameActor.Facing.LEFT:
			return &"left"
		GameActor.Facing.RIGHT:
			return &"right"
		_:
			return &"down"


func _apply_visibility_state() -> void:
	if actor.stats.is_invincible() and not actor.stats.is_dead:
		_visual_pivot.visible = int(Time.get_ticks_msec() / 90) % 2 == 0
	else:
		_visual_pivot.visible = true
