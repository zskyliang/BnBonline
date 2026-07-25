class_name BubbleView3D
extends Node3D
## Hand-pressed clay water bomb with stepped fuse anticipation.

const STOP_MOTION_STEP := 1.0 / 12.0
const AQUA := Color("#59b9dd")
const CORAL := Color("#df755e")

var bubble: GameBubble
var _initial_fuse_ms := 1
var _visual_accumulator := 0.0
var _visual_elapsed := 0.0
var _inner_body: MeshInstance3D
var _outer_shell: MeshInstance3D
var _cork_root: Node3D
var _pressure_dimples: Array[MeshInstance3D] = []
var _skin_color := AQUA
var _last_progress := 0.0


func bind_bubble(logic_bubble: GameBubble) -> void:
	bubble = logic_bubble
	name = "BubbleView3D_%s_%s" % [bubble.cell.x, bubble.cell.y]
	position = GameConstants.grid_to_world_3d(bubble.cell, 0.37)
	_initial_fuse_ms = maxi(1, bubble.milliseconds_until_explosion())
	_skin_color = CORAL if bubble.skin in ["coral", "basketball"] else AQUA
	_build_visual()
	bubble.tree_exiting.connect(queue_free, CONNECT_ONE_SHOT)


func fuse_progress() -> float:
	return _last_progress


func fuse_stage() -> int:
	return clampi(floori(_last_progress * 3.0), 0, 2)


func _process(delta: float) -> void:
	if not is_instance_valid(bubble):
		queue_free()
		return
	position = GameConstants.grid_to_world_3d(bubble.cell, 0.37)
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


func _apply_stop_motion_pose() -> void:
	var urgency := _last_progress * _last_progress
	var pulse := sin(_visual_elapsed * lerpf(7.0, 18.0, urgency))
	var squash := 1.0 + pulse * lerpf(0.025, 0.14, urgency)
	scale = Vector3(squash * 1.04, (2.0 - squash) * 0.97, squash)
	rotation.y = sin(_visual_elapsed * 2.6) * 0.06
	_cork_root.rotation.z = deg_to_rad(14.0 + pulse * lerpf(2.0, 18.0, urgency))
	var active_dimples := fuse_stage() + 1
	for index: int in range(_pressure_dimples.size()):
		var dimple := _pressure_dimples[index]
		var lit := index < active_dimples
		dimple.material_override = ClayMaterialLibrary.make(
			ClayMaterialLibrary.CREAM if lit else _skin_color.darkened(0.2),
			0.84,
			0.24 if lit else 0.0
		)


func _build_visual() -> void:
	_inner_body = MeshInstance3D.new()
	_inner_body.name = "MatteClayWaterCore"
	var inner_mesh := SphereMesh.new()
	inner_mesh.radius = 0.31
	inner_mesh.height = 0.62
	inner_mesh.radial_segments = 14
	inner_mesh.rings = 7
	_inner_body.mesh = inner_mesh
	_inner_body.scale = Vector3(1.05, 0.95, 1.0)
	_inner_body.material_override = ClayMaterialLibrary.make(_skin_color, 0.86)
	add_child(_inner_body)

	_outer_shell = MeshInstance3D.new()
	_outer_shell.name = "ThinWaterShell"
	var shell_mesh := SphereMesh.new()
	shell_mesh.radius = 0.36
	shell_mesh.height = 0.72
	shell_mesh.radial_segments = 16
	shell_mesh.rings = 8
	_outer_shell.mesh = shell_mesh
	_outer_shell.scale = Vector3(1.02, 0.96, 1.0)
	var shell_material := StandardMaterial3D.new()
	shell_material.albedo_color = _skin_color.lightened(0.2)
	shell_material.albedo_color.a = 0.22
	shell_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	shell_material.metallic = 0.0
	shell_material.metallic_specular = 0.14
	shell_material.roughness = 0.34
	shell_material.normal_enabled = true
	shell_material.normal_scale = 0.1
	shell_material.normal_texture = load("res://assets/materials/clay_detail_normal.png")
	_outer_shell.material_override = shell_material
	_outer_shell.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(_outer_shell)

	_cork_root = Node3D.new()
	_cork_root.name = "CorkAndFuse"
	_cork_root.position = Vector3(0.0, 0.35, 0.0)
	add_child(_cork_root)
	var cork := MeshInstance3D.new()
	var cork_mesh := CylinderMesh.new()
	cork_mesh.top_radius = 0.075
	cork_mesh.bottom_radius = 0.09
	cork_mesh.height = 0.15
	cork_mesh.radial_segments = 9
	cork.mesh = cork_mesh
	cork.position.y = 0.06
	cork.material_override = ClayMaterialLibrary.make(Color("#9d684d"), 0.94)
	_cork_root.add_child(cork)
	var fuse := MeshInstance3D.new()
	var fuse_mesh := CylinderMesh.new()
	fuse_mesh.top_radius = 0.022
	fuse_mesh.bottom_radius = 0.03
	fuse_mesh.height = 0.24
	fuse_mesh.radial_segments = 8
	fuse.mesh = fuse_mesh
	fuse.position = Vector3(0.07, 0.19, 0.0)
	fuse.rotation.z = deg_to_rad(32.0)
	fuse.material_override = ClayMaterialLibrary.make(ClayMaterialLibrary.CHARCOAL, 0.94)
	_cork_root.add_child(fuse)

	for index: int in range(3):
		var dimple := MeshInstance3D.new()
		dimple.name = "PressureDimple%d" % (index + 1)
		var dimple_mesh := SphereMesh.new()
		dimple_mesh.radius = 0.035
		dimple_mesh.height = 0.045
		dimple_mesh.radial_segments = 8
		dimple_mesh.rings = 4
		dimple.mesh = dimple_mesh
		dimple.position = Vector3(-0.17 + float(index) * 0.17, 0.08 - float(index % 2) * 0.06, 0.31)
		dimple.material_override = ClayMaterialLibrary.make(_skin_color.darkened(0.2), 0.86)
		_pressure_dimples.append(dimple)
		add_child(dimple)
