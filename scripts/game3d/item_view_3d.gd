class_name ItemView3D
extends Node3D
## Lightweight animated wrapper around one original storybook pickup model.

var item_id: int
var item_type: int

var _model_root: Node3D
var _base_height: float = 0.12
var _elapsed: float = 0.0


func setup(item: ArenaItemState) -> void:
	item_id = item.item_id
	item_type = item.item_type
	name = "ArenaItem%d" % item_id
	position = GameConstants.grid_to_world_3d(item.cell, _base_height)
	_build_visual()


func _process(delta: float) -> void:
	_elapsed += delta
	if is_instance_valid(_model_root):
		_model_root.rotation.y += delta * 1.45
	position.y = _base_height + sin(_elapsed * 2.8 + float(item_id)) * 0.07


func _build_visual() -> void:
	var model_path: String = ArenaItemType.model_path(item_type)
	var packed: PackedScene = load(model_path) as PackedScene
	_model_root = Node3D.new()
	_model_root.name = "ImportedModel"
	_model_root.scale = Vector3.ONE * _model_scale()
	add_child(_model_root)
	if packed != null:
		var model: Node = packed.instantiate()
		_model_root.add_child(model)
		if model is Node3D:
			StorybookMaterialLibrary.apply_character_palette(
				model as Node3D,
				Color.WHITE,
				[]
			)
	else:
		_model_root.add_child(_fallback_mesh())

	var ring := MeshInstance3D.new()
	ring.name = "PickupRing"
	var ring_mesh := TorusMesh.new()
	ring_mesh.inner_radius = 0.25
	ring_mesh.outer_radius = 0.34
	ring_mesh.rings = 12
	ring_mesh.ring_segments = 24
	ring.mesh = ring_mesh
	ring.position.y = -0.085
	ring.material_override = StorybookMaterialLibrary.make(
		ArenaItemType.accent_color(item_type),
		0.88,
		false,
		0.12
	)
	ring.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(ring)

func _model_scale() -> float:
	match item_type:
		ArenaItemType.Value.SPEED:
			return 0.72
		ArenaItemType.Value.BUBBLE:
			return 0.62
		ArenaItemType.Value.POWER:
			return 0.78
		_:
			return 1.0


func _fallback_mesh() -> MeshInstance3D:
	var mesh_instance := MeshInstance3D.new()
	var mesh := SphereMesh.new()
	mesh.radius = 0.24
	mesh.height = 0.48
	mesh_instance.mesh = mesh
	mesh_instance.material_override = StorybookMaterialLibrary.make(
		ArenaItemType.accent_color(item_type),
		0.9
	)
	return mesh_instance
