class_name CharacterPreview3D
extends SubViewport
## Static orthographic IdleDown preview using the production battle Sprite3D.

var definition: CharacterDefinition
var color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID
var _sprite_set: CharacterSpriteSet
var _character_sprite: Sprite3D
var _visual_pivot: Node3D
var _team_ring: MeshInstance3D


func setup(
		character_definition: CharacterDefinition,
		new_color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID
	) -> void:
	definition = character_definition
	color_id = new_color_id
	_sprite_set = definition.load_sprite_set()
	size = Vector2i(256, 192)
	own_world_3d = true
	transparent_bg = true
	render_target_update_mode = SubViewport.UPDATE_DISABLED
	msaa_3d = Viewport.MSAA_DISABLED

	var world := Node3D.new()
	world.name = "StorybookSpritePreviewWorld"
	add_child(world)
	_visual_pivot = Node3D.new()
	_visual_pivot.name = "PreviewPaperPuppet"
	world.add_child(_visual_pivot)

	_character_sprite = Sprite3D.new()
	_character_sprite.name = "ImageGenPreviewSprite"
	_character_sprite.texture = _sprite_set.texture_for(&"Idle", 0, &"down")
	_character_sprite.pixel_size = _sprite_set.pixel_size
	_character_sprite.position.y = _sprite_set.ground_offset
	StorybookMaterialLibrary.configure_billboard(_character_sprite)
	_visual_pivot.add_child(_character_sprite)
	_apply_palette()

	_team_ring = MeshInstance3D.new()
	_team_ring.name = "PreviewTeamRing"
	var ring_mesh := TorusMesh.new()
	ring_mesh.inner_radius = 0.38
	ring_mesh.outer_radius = 0.43
	ring_mesh.rings = 16
	ring_mesh.ring_segments = 28
	_team_ring.mesh = ring_mesh
	_team_ring.position.y = 0.01
	world.add_child(_team_ring)
	_apply_ring_color()

	var camera := Camera3D.new()
	camera.name = "PreviewCamera"
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = 1.55
	camera.look_at_from_position(Vector3(0.0, 0.68, 3.0), Vector3(0.0, 0.68, 0.0))
	camera.current = true
	world.add_child(camera)
	_request_static_render()


func set_color_id(new_color_id: String) -> void:
	if not PaintPalette.is_valid_color_id(new_color_id):
		return
	color_id = new_color_id
	_apply_palette()
	_apply_ring_color()
	_request_static_render()


func _apply_palette() -> void:
	if not is_instance_valid(_character_sprite) or _sprite_set == null:
		return
	var texture := _sprite_set.texture_for(&"Idle", 0, &"down")
	var mask := _sprite_set.mask_for(&"Idle", 0, &"down")
	_character_sprite.texture = texture
	_character_sprite.material_override = StorybookMaterialLibrary.make_sprite_material(
		texture,
		mask,
		PaintPalette.get_color(color_id)
	)


func _apply_ring_color() -> void:
	if not is_instance_valid(_team_ring):
		return
	_team_ring.material_override = StorybookMaterialLibrary.make(
		PaintPalette.get_color(color_id).lightened(0.18),
		0.9,
		false,
		0.08
	)


func _request_static_render() -> void:
	render_target_update_mode = SubViewport.UPDATE_ONCE
