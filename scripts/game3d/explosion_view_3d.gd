class_name ExplosionView3D
extends Node3D
## Pooled three-stage ImageGen foam burst for one logical explosion.

signal release_requested(view: ExplosionView3D)

const STOP_MOTION_STEP := 1.0 / 12.0
const FLASH_END := 0.083
const SPLASH_END := 0.25
const EFFECT_ROOT := "res://assets/art/storybook25d/effects/"
const STAGE_STEMS: Array[String] = ["pop_core", "cross_splash", "foam_burst"]

var effect: ExplosionEffect
var _center_sprite: Sprite3D
var _splash_cells: MultiMeshInstance3D
var _visual_accumulator := 0.0
var _visual_elapsed := 0.0
var _active := false
var _release_emitted := false
var _effect_color := PaintPalette.get_color(PaintPalette.DEFAULT_PLAYER_COLOR_ID)


func activate(logic_effect: ExplosionEffect, _droplet_count: int = 4) -> void:
	_ensure_visuals()
	_disconnect_effect()
	effect = logic_effect
	name = "ImageGenFoamExplosion"
	_visual_accumulator = 0.0
	_visual_elapsed = 0.0
	_release_emitted = false
	_active = true
	visible = true
	set_process(true)
	_build_cell_multimesh()
	_apply_effect_color()
	_apply_stage_pose(0.0)
	if is_instance_valid(effect):
		effect.tree_exiting.connect(_on_effect_exiting, CONNECT_ONE_SHOT)


func bind_effect(logic_effect: ExplosionEffect) -> void:
	activate(logic_effect)


func deactivate() -> void:
	_disconnect_effect()
	effect = null
	_active = false
	visible = false
	set_process(false)
	transform = Transform3D.IDENTITY


func is_active() -> bool:
	return _active


func visual_stage() -> int:
	if _visual_elapsed < FLASH_END:
		return 0
	if _visual_elapsed < SPLASH_END:
		return 1
	return 2


func _ready() -> void:
	_ensure_visuals()
	if not _active:
		deactivate()


func _process(delta: float) -> void:
	if not _active:
		return
	if not is_instance_valid(effect):
		_request_release()
		return
	_visual_accumulator += minf(delta, STOP_MOTION_STEP * 3.0)
	if _visual_accumulator < STOP_MOTION_STEP:
		return
	while _visual_accumulator >= STOP_MOTION_STEP:
		_visual_accumulator -= STOP_MOTION_STEP
		_visual_elapsed += STOP_MOTION_STEP
	_apply_stage_pose(_visual_elapsed)


func _ensure_visuals() -> void:
	if is_instance_valid(_center_sprite):
		return
	_center_sprite = Sprite3D.new()
	_center_sprite.name = "ImageGenExplosionStage"
	_center_sprite.pixel_size = 0.0032
	_center_sprite.position.y = 0.48
	StorybookMaterialLibrary.configure_billboard(_center_sprite)
	add_child(_center_sprite)

	_splash_cells = MultiMeshInstance3D.new()
	_splash_cells.name = "WatercolorSplashCells"
	_splash_cells.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(_splash_cells)


func _build_cell_multimesh() -> void:
	if not is_instance_valid(effect):
		return
	position = GameConstants.grid_to_world_3d(effect.center_cell, 0.0)
	var mesh := QuadMesh.new()
	mesh.size = Vector2(0.96, 0.96)
	var multi_mesh := MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	multi_mesh.instance_count = effect.cells.size()
	multi_mesh.mesh = mesh
	var flat_basis := Basis(Vector3.RIGHT, -PI * 0.5)
	for index: int in range(effect.cells.size()):
		var cell: Vector2i = effect.cells[index]
		var offset := GameConstants.grid_to_world_3d(cell, 0.095) - position
		multi_mesh.set_instance_transform(index, Transform3D(flat_basis, offset))
	_splash_cells.multimesh = multi_mesh


func _apply_effect_color() -> void:
	if not is_instance_valid(effect):
		return
	_effect_color = PaintPalette.get_color(effect.color_id)
	_apply_stage_texture(visual_stage())
	var texture := load(EFFECT_ROOT + "foam_burst.png") as Texture2D
	var mask := load(EFFECT_ROOT + "foam_burst_mask.png") as Texture2D
	_splash_cells.material_override = StorybookMaterialLibrary.make_sprite_material(
		texture,
		mask,
		_effect_color,
		0.82
	)


func _apply_stage_texture(stage: int) -> void:
	var stem := STAGE_STEMS[clampi(stage, 0, STAGE_STEMS.size() - 1)]
	var texture := load(EFFECT_ROOT + stem + ".png") as Texture2D
	var mask := load(EFFECT_ROOT + stem + "_mask.png") as Texture2D
	_center_sprite.texture = texture
	_center_sprite.material_override = StorybookMaterialLibrary.make_sprite_material(
		texture,
		mask,
		_effect_color,
		0.82
	)


func _apply_stage_pose(elapsed: float) -> void:
	var stage := visual_stage()
	_apply_stage_texture(stage)
	_splash_cells.visible = stage >= 1
	if stage == 0:
		_center_sprite.modulate = Color.WHITE
		_splash_cells.transparency = 0.0
		var flash_scale := 0.68 + elapsed / FLASH_END * 0.48
		_center_sprite.scale = Vector3.ONE * flash_scale
	elif stage == 1:
		_center_sprite.modulate = Color.WHITE
		_splash_cells.transparency = 0.0
		var splash_progress := clampf(
			(elapsed - FLASH_END) / (SPLASH_END - FLASH_END),
			0.0,
			1.0
		)
		_center_sprite.scale = Vector3.ONE * lerpf(0.45, 1.02, splash_progress)
		_splash_cells.scale = Vector3.ONE * lerpf(0.42, 1.0, splash_progress)
	else:
		var foam_progress := clampf(
			(elapsed - SPLASH_END) / maxf(
				0.001,
				GameConstants.EXPLOSION_SECONDS - SPLASH_END
			),
			0.0,
			1.0
		)
		_center_sprite.scale = Vector3.ONE * lerpf(0.92, 1.18, foam_progress)
		_center_sprite.modulate = Color(1.0, 1.0, 1.0, lerpf(1.0, 0.14, foam_progress))
		_splash_cells.transparency = lerpf(0.14, 0.95, foam_progress)


func _disconnect_effect() -> void:
	if is_instance_valid(effect) and effect.tree_exiting.is_connected(_on_effect_exiting):
		effect.tree_exiting.disconnect(_on_effect_exiting)


func _on_effect_exiting() -> void:
	_request_release()


func _request_release() -> void:
	if _release_emitted:
		return
	_release_emitted = true
	release_requested.emit(self)
