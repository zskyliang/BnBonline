class_name ExplosionView3D
extends Node3D
## Pooled three-stage hand-painted foam burst for one logical explosion.

signal release_requested(view: ExplosionView3D)

const STOP_MOTION_STEP := 1.0 / 12.0
const FLASH_END := 0.083
const SPLASH_END := 0.25
const DITHER_SHADER := preload("res://assets/materials/storybook_dither_fade.gdshader")
const EFFECT_SHAPES_PATH := \
	"res://assets/models/effects/storybook/effect_shapes.glb"

static var _core_mesh: SphereMesh
static var _cell_mesh: CapsuleMesh
static var _foam_mesh: Mesh
static var _drop_mesh: Mesh
static var _spike_mesh: Mesh

var effect: ExplosionEffect
var _core: MeshInstance3D
var _splash_cells: MultiMeshInstance3D
var _splash_spikes: MultiMeshInstance3D
var _foam_cells: MultiMeshInstance3D
var _droplet_root: Node3D
var _droplets: Array[MeshInstance3D] = []
var _foam_material: ShaderMaterial
var _cell_transforms: Array[Transform3D] = []
var _foam_transforms: Array[Transform3D] = []
var _spike_transforms: Array[Transform3D] = []
var _visual_accumulator := 0.0
var _visual_elapsed := 0.0
var _active := false
var _release_emitted := false


func activate(logic_effect: ExplosionEffect, droplet_count: int = 4) -> void:
	_ensure_visuals()
	_disconnect_effect()
	effect = logic_effect
	name = "StorybookFoamExplosion"
	_visual_accumulator = 0.0
	_visual_elapsed = 0.0
	_release_emitted = false
	_active = true
	visible = true
	set_process(true)
	_build_cell_transforms()
	_apply_effect_color()
	_set_droplet_count(clampi(droplet_count, 0, 4))
	_apply_stage_pose(0.0)
	if is_instance_valid(effect):
		effect.tree_exiting.connect(_on_effect_exiting, CONNECT_ONE_SHOT)


func _apply_effect_color() -> void:
	if not is_instance_valid(effect):
		return
	var color: Color = PaintPalette.get_color(effect.color_id)
	_core.material_override = StorybookMaterialLibrary.make(
		color.lightened(0.42), 0.84, false, 0.28
	)
	_splash_cells.material_override = StorybookMaterialLibrary.make(
		color, 0.86, false, 0.16
	)
	_splash_spikes.material_override = StorybookMaterialLibrary.make(
		color.lightened(0.18), 0.86, true, 0.06
	)
	_foam_material.set_shader_parameter("albedo_color", color.lightened(0.48))
	for index: int in range(_droplets.size()):
		_droplets[index].material_override = StorybookMaterialLibrary.make(
			color.lightened(0.18 if index % 2 == 0 else 0.42),
			0.88
		)


func bind_effect(logic_effect: ExplosionEffect) -> void:
	activate(logic_effect)


func deactivate() -> void:
	_disconnect_effect()
	effect = null
	_active = false
	visible = false
	set_process(false)
	_cell_transforms.clear()
	_foam_transforms.clear()
	_spike_transforms.clear()
	for droplet: MeshInstance3D in _droplets:
		droplet.visible = false
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
	if is_instance_valid(_core):
		return
	_ensure_shared_meshes()
	_core = MeshInstance3D.new()
	_core.name = "CompressedCoreFlash"
	_core.mesh = _core_mesh
	_core.material_override = StorybookMaterialLibrary.make(
		StorybookMaterialLibrary.PAPER, 0.84, false, 0.28
	)
	add_child(_core)

	_splash_cells = MultiMeshInstance3D.new()
	_splash_cells.name = "StorybookWaterSplashCells"
	_splash_cells.material_override = StorybookMaterialLibrary.make(
		Color("#38afd8"), 0.86, false, 0.16
	)
	add_child(_splash_cells)

	_splash_spikes = MultiMeshInstance3D.new()
	_splash_spikes.name = "HandPinchedRadialSplash"
	_splash_spikes.material_override = StorybookMaterialLibrary.make(
		Color("#67c6e1"), 0.86, true, 0.06
	)
	add_child(_splash_spikes)

	_foam_cells = MultiMeshInstance3D.new()
	_foam_cells.name = "OpaqueFoamCells"
	_foam_material = ShaderMaterial.new()
	_foam_material.shader = DITHER_SHADER
	_foam_material.set_shader_parameter("albedo_color", Color("#e8f0df"))
	_foam_material.set_shader_parameter(
		"normal_texture",
		load("res://assets/materials/storybook_paper_normal.png") as Texture2D
	)
	_foam_material.set_shader_parameter("roughness_value", 0.9)
	_foam_cells.material_override = _foam_material
	add_child(_foam_cells)

	_droplet_root = Node3D.new()
	_droplet_root.name = "StorybookDroplets"
	add_child(_droplet_root)
	for index: int in range(4):
		var droplet := MeshInstance3D.new()
		droplet.mesh = _drop_mesh
		droplet.material_override = StorybookMaterialLibrary.make(
			Color("#73cbe3") if index % 2 == 0 else StorybookMaterialLibrary.PAPER,
			0.88
		)
		_droplet_root.add_child(droplet)
		_droplets.append(droplet)


func _build_cell_transforms() -> void:
	_cell_transforms.clear()
	_foam_transforms.clear()
	if not is_instance_valid(effect):
		return
	position = GameConstants.grid_to_world_3d(effect.center_cell, 0.0)
	var cell_lookup: Dictionary = {}
	for cell: Vector2i in effect.cells:
		cell_lookup[cell] = true
	for cell: Vector2i in effect.cells:
		var offset := GameConstants.grid_to_world_3d(cell, 0.31) - position
		var direction := cell - effect.center_cell
		var is_endpoint := _is_endpoint(cell, direction, cell_lookup)
		var scale := Vector3(0.72, 0.72, 0.72)
		if direction.x != 0:
			scale = Vector3(1.18 if not is_endpoint else 1.02, 0.72 if is_endpoint else 0.58, 0.56)
		elif direction.y != 0:
			scale = Vector3(0.56, 0.72 if is_endpoint else 0.58, 1.18 if not is_endpoint else 1.02)
		else:
			scale = Vector3(1.05, 0.82, 1.05)
		_cell_transforms.append(Transform3D(Basis().scaled(scale), offset))
		var seed := absi(cell.x * 31 + cell.y * 17)
		var foam_offset := offset + Vector3(
			float(seed % 5 - 2) * 0.055,
			0.18 + float(seed % 3) * 0.045,
			float((seed / 5) % 5 - 2) * 0.055
		)
		_foam_transforms.append(
			Transform3D(Basis().scaled(Vector3(0.72, 0.66, 0.72)), foam_offset)
		)
	_assign_multimesh(_splash_cells, _cell_mesh, _cell_transforms)
	for index: int in range(8):
		var angle := float(index) * TAU / 8.0
		var direction := Vector3(cos(angle), 0.0, sin(angle))
		var orientation := Basis(Quaternion(Vector3.UP, direction))
		var spike_scale := Vector3(1.0, 0.72 + float(index % 3) * 0.08, 1.0)
		_spike_transforms.append(
			Transform3D(
				orientation.scaled(spike_scale),
				direction * 0.27 + Vector3(0.0, 0.39, 0.0)
			)
		)
	_assign_multimesh(_splash_spikes, _spike_mesh, _spike_transforms)
	_assign_multimesh(_foam_cells, _foam_mesh, _foam_transforms)
	_core.position = Vector3(0.0, 0.34, 0.0)


func _apply_stage_pose(elapsed: float) -> void:
	var stage := visual_stage()
	_core.visible = stage == 0
	_splash_cells.visible = stage == 1
	_splash_spikes.visible = stage == 1
	_foam_cells.visible = stage == 2
	_droplet_root.visible = stage == 2
	if stage == 0:
		var flash_scale := 0.72 + elapsed / FLASH_END * 0.48
		_core.scale = Vector3(flash_scale * 1.12, flash_scale * 0.7, flash_scale)
	elif stage == 1:
		var splash_progress := clampf((elapsed - FLASH_END) / (SPLASH_END - FLASH_END), 0.0, 1.0)
		_splash_cells.scale = Vector3(
			lerpf(0.48, 1.0, splash_progress),
			lerpf(0.28, 1.0, splash_progress),
			lerpf(0.48, 1.0, splash_progress)
		)
		_splash_spikes.scale = Vector3.ONE * lerpf(0.24, 1.0, splash_progress)
	else:
		var foam_progress := clampf(
			(elapsed - SPLASH_END) / maxf(0.001, GameConstants.EXPLOSION_SECONDS - SPLASH_END),
			0.0,
			1.0
		)
		_foam_cells.scale = Vector3.ONE * lerpf(0.86, 1.18, foam_progress)
		_foam_cells.position.y = foam_progress * 0.18
		_foam_material.set_shader_parameter("visible_ratio", lerpf(1.0, 0.08, foam_progress))
		for index: int in range(_droplets.size()):
			var droplet := _droplets[index]
			if not droplet.visible:
				continue
			var angle := float(index) * TAU / 4.0 + 0.35
			var radius := lerpf(0.18, 0.72, foam_progress)
			droplet.position = Vector3(
				cos(angle) * radius,
				0.34 + sin(foam_progress * PI) * (0.36 + float(index % 2) * 0.12),
				sin(angle) * radius
			)
			droplet.scale = Vector3.ONE * lerpf(0.72, 0.28, foam_progress)


func _set_droplet_count(count: int) -> void:
	for index: int in range(_droplets.size()):
		_droplets[index].visible = index < count


func _assign_multimesh(instance: MultiMeshInstance3D, mesh: Mesh, transforms: Array[Transform3D]) -> void:
	var multi_mesh := MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	multi_mesh.mesh = mesh
	multi_mesh.instance_count = transforms.size()
	for index: int in range(transforms.size()):
		multi_mesh.set_instance_transform(index, transforms[index])
	instance.multimesh = multi_mesh


func _is_endpoint(cell: Vector2i, direction: Vector2i, lookup: Dictionary) -> bool:
	if direction == Vector2i.ZERO:
		return false
	var step := Vector2i(signi(direction.x), signi(direction.y))
	return not lookup.has(cell + step)


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


static func _ensure_shared_meshes() -> void:
	_load_storybook_effect_meshes()
	if _core_mesh == null:
		_core_mesh = SphereMesh.new()
		_core_mesh.radius = 0.42
		_core_mesh.height = 0.78
		_core_mesh.radial_segments = 12
		_core_mesh.rings = 6
	if _cell_mesh == null:
		_cell_mesh = CapsuleMesh.new()
		_cell_mesh.radius = 0.34
		_cell_mesh.height = 0.78
		_cell_mesh.radial_segments = 12
		_cell_mesh.rings = 4
	if _foam_mesh == null:
		var foam_fallback := SphereMesh.new()
		foam_fallback.radius = 0.34
		foam_fallback.height = 0.62
		foam_fallback.radial_segments = 10
		foam_fallback.rings = 5
		_foam_mesh = foam_fallback
	if _drop_mesh == null:
		var drop_fallback := SphereMesh.new()
		drop_fallback.radius = 0.13
		drop_fallback.height = 0.28
		drop_fallback.radial_segments = 8
		drop_fallback.rings = 4
		_drop_mesh = drop_fallback
	if _spike_mesh == null:
		var spike_fallback := CylinderMesh.new()
		spike_fallback.top_radius = 0.025
		spike_fallback.bottom_radius = 0.13
		spike_fallback.height = 0.64
		spike_fallback.radial_segments = 7
		_spike_mesh = spike_fallback


static func _load_storybook_effect_meshes() -> void:
	if _foam_mesh != null and _drop_mesh != null and _spike_mesh != null:
		return
	var packed := load(EFFECT_SHAPES_PATH) as PackedScene
	if packed == null:
		return
	var instance := packed.instantiate()
	for node: Node in instance.find_children("*", "MeshInstance3D", true, false):
		var mesh_instance := node as MeshInstance3D
		match mesh_instance.name:
			"FoamPetal":
				_foam_mesh = mesh_instance.mesh
			"Droplet":
				_drop_mesh = mesh_instance.mesh
			"BurstSpike":
				_spike_mesh = mesh_instance.mesh
	instance.free()
