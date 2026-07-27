class_name StorybookLobbyDiorama3D
extends SubViewport
## ImageGen forest clearing with the production Sprite3D animal roster.

const STOP_MOTION_STEP := 1.0 / 8.0
const LOBBY_TEXTURE := preload(
	"res://assets/art/storybook25d/lobby/storybook_lobby.png"
)
## Feet anchors measured from the eight painted stump pads in storybook_lobby.png.
## The order follows CharacterCatalog.IDS so every animal owns one unambiguous pad.
const CHARACTER_POSITIONS: Array[Vector3] = [
	Vector3(-0.50, 2.95, 0.012),
	Vector3(0.53, 2.95, 0.012),
	Vector3(-1.30, 2.61, 0.014),
	Vector3(1.32, 2.61, 0.014),
	Vector3(-1.45, 2.12, 0.016),
	Vector3(1.42, 2.12, 0.016),
	Vector3(-0.80, 1.58, 0.018),
	Vector3(0.53, 1.58, 0.018),
]

var _puppets: Array[Dictionary] = []
var _animation_accumulator: float = 0.0
var _animation_time: float = 0.0


func _ready() -> void:
	own_world_3d = true
	transparent_bg = false
	render_target_update_mode = SubViewport.UPDATE_ALWAYS
	msaa_3d = Viewport.MSAA_DISABLED
	_build_world()


func _process(delta: float) -> void:
	_animation_accumulator += minf(delta, STOP_MOTION_STEP * 3.0)
	while _animation_accumulator >= STOP_MOTION_STEP:
		_animation_accumulator -= STOP_MOTION_STEP
		_animation_time += STOP_MOTION_STEP
		for index: int in range(_puppets.size()):
			var puppet := _puppets[index]
			var pivot := puppet["pivot"] as Node3D
			if not is_instance_valid(pivot):
				continue
			var phase := _animation_time * (1.7 + float(index % 3) * 0.14) + float(index)
			var base_position := puppet["base_position"] as Vector3
			pivot.position = base_position + Vector3.UP * sin(phase) * 0.025
			pivot.rotation.z = sin(phase * 0.72) * 0.035


func _build_world() -> void:
	var world := Node3D.new()
	world.name = "ForestSpriteLobbyWorld"
	add_child(world)

	var environment_node := WorldEnvironment.new()
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = StorybookMaterialLibrary.PAPER
	environment_node.environment = environment
	world.add_child(environment_node)

	var backdrop := Sprite3D.new()
	backdrop.name = "ImageGenLobbyBackdrop"
	backdrop.texture = LOBBY_TEXTURE
	backdrop.pixel_size = 0.00542
	backdrop.position = Vector3(0.0, 2.55, -0.2)
	backdrop.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	world.add_child(backdrop)

	for index: int in range(CharacterCatalog.IDS.size()):
		_add_character(
			world,
			CharacterCatalog.IDS[index],
			CHARACTER_POSITIONS[index],
			PaintPalette.COLOR_IDS[index % PaintPalette.COLOR_IDS.size()],
			index
		)

	var camera := Camera3D.new()
	camera.name = "LobbySpriteCamera"
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = 5.25
	camera.look_at_from_position(
		Vector3(0.0, 2.55, 10.0),
		Vector3(0.0, 2.55, 0.0)
	)
	camera.current = true
	world.add_child(camera)


func _add_character(
		world: Node3D,
		character_id: String,
		location: Vector3,
		color_id: String,
		index: int
	) -> void:
	var definition := CharacterCatalog.get_definition(character_id)
	var sprite_set := definition.load_sprite_set()
	var action: StringName = &"Idle"
	var pivot := Node3D.new()
	pivot.name = "%sLobbyPuppet" % character_id.capitalize()
	pivot.position = location
	world.add_child(pivot)
	var sprite := Sprite3D.new()
	sprite.name = "%sLobbySprite" % character_id.capitalize()
	var texture := sprite_set.texture_for(action, 0, &"down")
	var mask := sprite_set.mask_for(action, 0, &"down")
	sprite.texture = texture
	sprite.pixel_size = 0.00155
	sprite.position.y = 0.39
	sprite.material_override = StorybookMaterialLibrary.make_sprite_material(
		texture,
		mask,
		PaintPalette.get_color(color_id)
	)
	StorybookMaterialLibrary.configure_billboard(sprite)
	pivot.add_child(sprite)
	_puppets.append({
		"character_id": character_id,
		"pivot": pivot,
		"sprite": sprite,
		"action": action,
		"base_position": location,
	})


func get_pad_alignment_report() -> Array[Dictionary]:
	var result: Array[Dictionary] = []
	for index: int in range(_puppets.size()):
		var puppet := _puppets[index]
		result.append({
			"character_id": str(puppet["character_id"]),
			"pad_index": index,
			"feet_anchor": puppet["base_position"] as Vector3,
			"expected_anchor": CHARACTER_POSITIONS[index],
		})
	return result
