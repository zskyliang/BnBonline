class_name BubbleView3D
extends Node3D
## Team-colored ImageGen soap-bubble Sprite3D with stepped fuse anticipation.

const STOP_MOTION_STEP := 1.0 / 12.0
const BUBBLE_ROOT := (
	"res://assets/art/storybook25d/effects/character_bubbles/"
)
const BUBBLE_PIXEL_SIZE := 0.00225

var bubble: GameBubble
var _initial_fuse_ms := 1
var _visual_accumulator := 0.0
var _visual_elapsed := 0.0
var _bubble_sprite: Sprite3D
var _bubble_texture: Texture2D
var _bubble_mask: Texture2D
var _skin_character_id := "cat"
var _skin_color := PaintPalette.get_color(PaintPalette.DEFAULT_PLAYER_COLOR_ID)
var _last_progress := 0.0


func bind_bubble(logic_bubble: GameBubble) -> void:
	bubble = logic_bubble
	name = "BubbleSprite3D_%s_%s" % [bubble.cell.x, bubble.cell.y]
	position = GameConstants.grid_to_world_3d(bubble.cell, 0.04)
	_initial_fuse_ms = maxi(1, bubble.milliseconds_until_explosion())
	_skin_color = PaintPalette.get_color(bubble.color_id)
	if is_instance_valid(bubble.bubble_owner):
		_skin_character_id = CharacterCatalog.migrate_legacy_id(
			bubble.bubble_owner.character_id
		)
	_build_visual()
	bubble.tree_exiting.connect(queue_free, CONNECT_ONE_SHOT)


func fuse_progress() -> float:
	return _last_progress


func fuse_stage() -> int:
	return clampi(floori(_last_progress * 3.0), 0, 2)


func get_skin_character_id() -> String:
	return _skin_character_id


func get_visual_cell_width() -> float:
	if not is_instance_valid(_bubble_texture):
		return 0.0
	var image := _bubble_texture.get_image()
	if image == null:
		return 0.0
	var used_rect := image.get_used_rect()
	return float(used_rect.size.x) * BUBBLE_PIXEL_SIZE


func _process(delta: float) -> void:
	if not is_instance_valid(bubble):
		queue_free()
		return
	position = GameConstants.grid_to_world_3d(bubble.cell, 0.04)
	_visual_accumulator += minf(delta, STOP_MOTION_STEP * 3.0)
	while _visual_accumulator >= STOP_MOTION_STEP:
		_visual_accumulator -= STOP_MOTION_STEP
		_visual_elapsed += STOP_MOTION_STEP
		_last_progress = clampf(
			1.0 - float(bubble.milliseconds_until_explosion()) / float(_initial_fuse_ms),
			0.0,
			1.0
		)
		_apply_stop_motion_pose()


func _build_visual() -> void:
	_bubble_texture = load(
		BUBBLE_ROOT + _skin_character_id + ".png"
	) as Texture2D
	_bubble_mask = load(
		BUBBLE_ROOT + _skin_character_id + "_mask.png"
	) as Texture2D
	_bubble_sprite = Sprite3D.new()
	_bubble_sprite.name = (
		_skin_character_id.capitalize() + "ImageGenBubbleSprite"
	)
	_bubble_sprite.texture = _bubble_texture
	_bubble_sprite.pixel_size = BUBBLE_PIXEL_SIZE
	_bubble_sprite.position.y = 0.53
	_bubble_sprite.material_override = StorybookMaterialLibrary.make_sprite_material(
		_bubble_texture,
		_bubble_mask,
		_skin_color,
		0.74
	)
	StorybookMaterialLibrary.configure_billboard(_bubble_sprite)
	add_child(_bubble_sprite)

	var shadow := MeshInstance3D.new()
	shadow.name = "BubbleContactShadow"
	var mesh := CylinderMesh.new()
	mesh.top_radius = 0.38
	mesh.bottom_radius = 0.38
	mesh.height = 0.006
	mesh.radial_segments = 18
	shadow.mesh = mesh
	shadow.position.y = 0.005
	shadow.material_override = StorybookMaterialLibrary.make(
		Color(0.13, 0.12, 0.16, 0.18)
	)
	add_child(shadow)


func _apply_stop_motion_pose() -> void:
	var urgency := _last_progress * _last_progress
	var pulse := sin(_visual_elapsed * lerpf(7.0, 18.0, urgency))
	var squash := 1.0 + pulse * lerpf(0.025, 0.13, urgency)
	_bubble_sprite.scale = Vector3(
		squash * 1.02,
		(2.0 - squash) * 0.98,
		1.0
	)
	_bubble_sprite.rotation.z = sin(_visual_elapsed * 2.6) * lerpf(
		0.035,
		0.14,
		urgency
	)
	var stage_brightness := 1.0 + float(fuse_stage()) * 0.05
	_bubble_sprite.modulate = Color(
		stage_brightness,
		stage_brightness,
		stage_brightness,
		1.0
	)
