class_name ClayBuildingView3D
extends Node3D
## Composite CC0 building wrapper with a Compatibility-safe dithered roof.

const VISUAL_OCCLUDER_LAYER := 1 << 20
const FADE_DURATION := 0.18
const RESTORE_DELAY := 0.3
const OCCLUDED_VISIBLE_RATIO := 0.25
const DITHER_SHADER := preload("res://assets/materials/clay_dither_fade.gdshader")

var placement: BuildingPlacement
var definition: BuildingDefinition

var _base_root: Node3D
var _occludable_root: Node3D
var _roof_material: ShaderMaterial
var _visible_ratio := 1.0
var _target_visible_ratio := 1.0
var _restore_delay_remaining := 0.0


func configure(new_placement: BuildingPlacement) -> void:
	placement = new_placement
	definition = BuildingCatalog.get_definition(placement.asset_id)
	name = "%s_%d_%d" % [
		definition.id.capitalize(),
		placement.origin_cell.x,
		placement.origin_cell.y,
	]
	position = _footprint_center(placement.origin_cell, placement.footprint)
	rotation.y = float(placement.rotation_quadrants) * PI * 0.5
	_base_root = Node3D.new()
	_base_root.name = "Base"
	add_child(_base_root)
	_occludable_root = Node3D.new()
	_occludable_root.name = "Occludable"
	add_child(_occludable_root)
	_roof_material = _make_dither_material(definition.roof_color)
	match definition.kind:
		&"hedge":
			_build_hedge()
		&"tower":
			_build_tower()
		_:
			_build_house()
	if placement.occludable and definition.kind != &"hedge":
		_build_visual_occluder()
	set_process(placement.occludable)


func set_occluded(occluded: bool) -> void:
	if not placement.occludable:
		return
	if occluded:
		_restore_delay_remaining = RESTORE_DELAY
		_target_visible_ratio = OCCLUDED_VISIBLE_RATIO
	else:
		_restore_delay_remaining = maxf(0.0, _restore_delay_remaining)


func is_occluded() -> bool:
	return _target_visible_ratio < 1.0 or _visible_ratio < 0.999


func _process(delta: float) -> void:
	if _restore_delay_remaining > 0.0:
		_restore_delay_remaining = maxf(0.0, _restore_delay_remaining - delta)
		if _restore_delay_remaining <= 0.0:
			_target_visible_ratio = 1.0
	var speed := 1.0 / FADE_DURATION
	_visible_ratio = move_toward(_visible_ratio, _target_visible_ratio, delta * speed)
	_roof_material.set_shader_parameter("visible_ratio", _visible_ratio)


func _build_house() -> void:
	var width := float(placement.footprint.x) * 0.86
	var depth := float(placement.footprint.y) * 0.86
	var height := 1.02 if placement.footprint.y > 1 else 0.88
	var paths := definition.module_paths
	var wall_path := paths[0] if paths.size() > 0 else ""
	var window_path := paths[1] if paths.size() > 1 else wall_path
	var door_path := paths[2] if paths.size() > 2 else wall_path
	var roof_path := paths[3] if paths.size() > 3 else ""
	# The modular kit supplies the facade details. A single hand-pressed clay
	# volume underneath keeps each multi-cell house visually solid at game scale.
	_add_box(
		_base_root,
		Vector3(width * 0.94, height, depth * 0.94),
		Vector3(0.0, height * 0.5, 0.0),
		definition.base_color.darkened(0.025)
	)
	_add_box(
		_base_root,
		Vector3(width, 0.12, depth),
		Vector3(0.0, 0.08, 0.0),
		definition.roof_color.darkened(0.22)
	)
	for x: int in range(placement.footprint.x):
		var local_x := -width * 0.5 + (float(x) + 0.5) * width / float(placement.footprint.x)
		_add_module(
			door_path if x == 0 else window_path,
			_base_root,
			Vector3(local_x, 0.0, depth * 0.5),
			Vector3(height, height, width / float(placement.footprint.x)),
			PI * 0.5,
			definition.base_color
		)
		_add_module(
			wall_path,
			_base_root,
			Vector3(local_x, 0.0, -depth * 0.5),
			Vector3(height, height, width / float(placement.footprint.x)),
			PI * 0.5,
			definition.base_color.darkened(0.04)
		)
	for z: int in range(placement.footprint.y):
		var local_z := -depth * 0.5 + (float(z) + 0.5) * depth / float(placement.footprint.y)
		for side: float in [-1.0, 1.0]:
			_add_module(
				wall_path,
				_base_root,
				Vector3(side * width * 0.5, 0.0, local_z),
				Vector3(height, height, depth / float(placement.footprint.y)),
				0.0,
				definition.base_color.darkened(0.02 if side > 0.0 else 0.06)
			)
	var roof := _add_module(
		roof_path,
		_occludable_root,
		Vector3(0.0, height - 0.02, 0.0),
		Vector3(width * 0.98, 0.72, depth * 0.98),
		0.0,
		definition.roof_color,
		true
	)
	if roof == null:
		_add_fallback_roof(width, depth, height)
	if definition.kind == &"clinic":
		_build_clinic_sign(width, depth, height)
	elif definition.kind == &"shop":
		_build_shop_sign(width, depth, height)


func _build_tower() -> void:
	var base_size := Vector3(2.55, 0.68, 2.55)
	_add_box(_base_root, base_size, Vector3(0.0, 0.34, 0.0), definition.base_color.darkened(0.05))
	_add_box(
		_base_root,
		Vector3(2.7, 0.13, 2.7),
		Vector3(0.0, 0.73, 0.0),
		definition.roof_color.darkened(0.16)
	)
	var tower_size := Vector3(1.36, 2.65, 1.36)
	_add_box(_base_root, tower_size, Vector3(0.0, 1.65, 0.0), definition.base_color)
	_add_box(
		_base_root,
		Vector3(1.52, 0.14, 1.52),
		Vector3(0.0, 2.9, 0.0),
		definition.roof_color.darkened(0.1)
	)
	var window_path := definition.module_paths[1]
	for yaw: float in [0.0, PI * 0.5, PI, PI * 1.5]:
		var direction := Vector3(sin(yaw), 0.0, cos(yaw))
		_add_module(
			window_path,
			_base_root,
			direction * 0.7 + Vector3(0.0, 1.32, 0.0),
			Vector3(0.72, 0.72, 0.72),
			yaw + PI * 0.5,
			ClayMaterialLibrary.SKY.darkened(0.12)
		)
	for yaw: float in [0.0, PI * 0.5, PI, PI * 1.5]:
		_build_clock_face(yaw)
	var roof_mesh := CylinderMesh.new()
	roof_mesh.top_radius = 0.08
	roof_mesh.bottom_radius = 0.94
	roof_mesh.height = 1.35
	roof_mesh.radial_segments = 8
	_add_mesh(_occludable_root, roof_mesh, Vector3(0.0, 3.58, 0.0), _roof_material)
	var bell := MeshInstance3D.new()
	var bell_mesh := SphereMesh.new()
	bell_mesh.radius = 0.22
	bell_mesh.height = 0.34
	bell_mesh.radial_segments = 10
	bell.mesh = bell_mesh
	bell.position = Vector3(0.0, 2.52, 0.72)
	bell.scale = Vector3(1.0, 0.82, 0.74)
	bell.material_override = ClayMaterialLibrary.make(ClayMaterialLibrary.MUSTARD, 0.88)
	_base_root.add_child(bell)


func _build_hedge() -> void:
	var path := definition.module_paths[0] if not definition.module_paths.is_empty() else ""
	var module := BuildingCatalog.instantiate_module(path)
	if module != null:
		module.name = "QuaterniusBush"
		module.scale = Vector3.ONE * 0.24
		module.position.y = 0.02
		ClayMaterialLibrary.apply_to_model(module, definition.base_color)
		_base_root.add_child(module)
		return
	var bush_mesh := SphereMesh.new()
	bush_mesh.radius = 0.43
	bush_mesh.height = 0.72
	bush_mesh.radial_segments = 10
	bush_mesh.rings = 5
	_add_mesh(
		_base_root,
		bush_mesh,
		Vector3(0.0, 0.34, 0.0),
		ClayMaterialLibrary.make(definition.base_color, 0.94)
	)


func _build_clinic_sign(width: float, depth: float, height: float) -> void:
	var sign_root := Node3D.new()
	sign_root.position = Vector3(0.0, height * 0.54, depth * 0.5 + 0.07)
	_base_root.add_child(sign_root)
	var vertical := BoxMesh.new()
	vertical.size = Vector3(0.1, 0.36, 0.07)
	_add_mesh(sign_root, vertical, Vector3.ZERO, ClayMaterialLibrary.make(Color("#d95c55"), 0.9))
	var horizontal := BoxMesh.new()
	horizontal.size = Vector3(0.34, 0.1, 0.07)
	_add_mesh(sign_root, horizontal, Vector3.ZERO, ClayMaterialLibrary.make(Color("#d95c55"), 0.9))
	var backing := BoxMesh.new()
	backing.size = Vector3(0.48, 0.48, 0.05)
	var backing_instance := _add_mesh(
		sign_root,
		backing,
		Vector3(0.0, 0.0, -0.04),
		ClayMaterialLibrary.make(ClayMaterialLibrary.CREAM.lightened(0.04), 0.92)
	)
	backing_instance.position.x = clampf(backing_instance.position.x, -width * 0.25, width * 0.25)


func _build_shop_sign(width: float, depth: float, height: float) -> void:
	var backing := BoxMesh.new()
	backing.size = Vector3(minf(0.82, width * 0.5), 0.27, 0.08)
	_add_mesh(
		_base_root,
		backing,
		Vector3(0.0, height * 0.68, depth * 0.5 + 0.08),
		ClayMaterialLibrary.make(definition.roof_color, 0.9)
	)
	for offset: float in [-0.22, 0.0, 0.22]:
		var bead := SphereMesh.new()
		bead.radius = 0.055
		bead.height = 0.08
		bead.radial_segments = 8
		bead.rings = 4
		_add_mesh(
			_base_root,
			bead,
			Vector3(offset, height * 0.68, depth * 0.5 + 0.135),
			ClayMaterialLibrary.make(ClayMaterialLibrary.CREAM, 0.9)
		)


func _build_clock_face(yaw: float) -> void:
	var face_root := Node3D.new()
	face_root.name = "ClockFace"
	face_root.position = Vector3(0.0, 2.08, 0.0)
	face_root.rotation.y = yaw
	_base_root.add_child(face_root)
	var disk_mesh := CylinderMesh.new()
	disk_mesh.top_radius = 0.31
	disk_mesh.bottom_radius = 0.31
	disk_mesh.height = 0.055
	disk_mesh.radial_segments = 16
	var disk := _add_mesh(
		face_root,
		disk_mesh,
		Vector3(0.0, 0.0, 0.705),
		ClayMaterialLibrary.make(ClayMaterialLibrary.CREAM.lightened(0.04), 0.9)
	)
	disk.rotation.x = PI * 0.5
	var minute_mesh := BoxMesh.new()
	minute_mesh.size = Vector3(0.035, 0.21, 0.035)
	var minute := _add_mesh(
		face_root,
		minute_mesh,
		Vector3(0.0, 0.075, 0.75),
		ClayMaterialLibrary.make(ClayMaterialLibrary.CHARCOAL, 0.9)
	)
	minute.rotation.z = deg_to_rad(-18.0)
	var hour_mesh := BoxMesh.new()
	hour_mesh.size = Vector3(0.16, 0.035, 0.035)
	var hour := _add_mesh(
		face_root,
		hour_mesh,
		Vector3(0.065, -0.01, 0.752),
		ClayMaterialLibrary.make(ClayMaterialLibrary.CHARCOAL, 0.9)
	)
	hour.rotation.z = deg_to_rad(12.0)


func _build_visual_occluder() -> void:
	var body := StaticBody3D.new()
	body.name = "RoofOcclusionProxy"
	body.collision_layer = VISUAL_OCCLUDER_LAYER
	body.collision_mask = 0
	body.set_meta("clay_building_view", self)
	var shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	var is_tower := definition.kind == &"tower"
	box.size = Vector3(
		float(placement.footprint.x) * 0.88,
		2.5 if is_tower else 1.2,
		float(placement.footprint.y) * 0.88
	)
	shape.shape = box
	shape.position.y = 2.35 if is_tower else 1.28
	body.add_child(shape)
	add_child(body)


func _add_fallback_roof(width: float, depth: float, height: float) -> void:
	var mesh := PrismMesh.new()
	mesh.size = Vector3(width * 1.06, 0.72, depth * 1.06)
	_add_mesh(_occludable_root, mesh, Vector3(0.0, height + 0.32, 0.0), _roof_material)


func _add_module(
		resource_path: String,
		parent: Node3D,
		local_position: Vector3,
		module_scale: Vector3,
		yaw: float,
		color: Color,
		dithered: bool = false
	) -> Node3D:
	var module := BuildingCatalog.instantiate_module(resource_path)
	if module == null:
		return null
	module.position = local_position
	module.scale = module_scale
	module.rotation.y = yaw
	_set_model_material(module, _roof_material if dithered else ClayMaterialLibrary.make(color, 0.92))
	parent.add_child(module)
	return module


func _add_box(parent: Node3D, size: Vector3, local_position: Vector3, color: Color) -> MeshInstance3D:
	var mesh := BoxMesh.new()
	mesh.size = size
	return _add_mesh(parent, mesh, local_position, ClayMaterialLibrary.make(color, 0.92))


func _add_mesh(
		parent: Node3D,
		mesh: Mesh,
		local_position: Vector3,
		material: Material
	) -> MeshInstance3D:
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.position = local_position
	instance.material_override = material
	instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	parent.add_child(instance)
	return instance


func _set_model_material(root_node: Node, material: Material) -> void:
	if root_node is MeshInstance3D:
		var mesh_instance := root_node as MeshInstance3D
		mesh_instance.material_override = material
		mesh_instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	for child: Node in root_node.get_children():
		_set_model_material(child, material)


func _make_dither_material(color: Color) -> ShaderMaterial:
	var material := ShaderMaterial.new()
	material.shader = DITHER_SHADER
	material.set_shader_parameter("albedo_color", color)
	material.set_shader_parameter(
		"normal_texture",
		load("res://assets/materials/clay_detail_normal.png") as Texture2D
	)
	material.set_shader_parameter("visible_ratio", 1.0)
	material.set_shader_parameter("roughness_value", 0.91)
	return material


func _footprint_center(origin: Vector2i, size: Vector2i) -> Vector3:
	var first := GameConstants.grid_to_world_3d(origin)
	var last := GameConstants.grid_to_world_3d(origin + size - Vector2i.ONE)
	return (first + last) * 0.5
