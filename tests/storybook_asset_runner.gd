extends SceneTree
## Import and transparency contract for every ImageGen Sprite3D runtime asset.

const SHARED_TEXTURE_PATHS: Array[String] = [
	"res://assets/art/storybook25d/environment/grass.png",
	"res://assets/art/storybook25d/environment/floor_tile_atlas.png",
	"res://assets/art/storybook25d/environment/wood_rail.png",
	"res://assets/art/storybook25d/environment/wood_corner.png",
	"res://assets/art/storybook25d/environment/conifer.png",
	"res://assets/art/storybook25d/environment/bush.png",
	"res://assets/art/storybook25d/environment/mushrooms.png",
	"res://assets/art/storybook25d/environment/stump.png",
	"res://assets/art/storybook25d/environment/wood_sign.png",
	"res://assets/art/storybook25d/environment/rocks.png",
	"res://assets/art/storybook25d/environment/flowers.png",
	"res://assets/art/storybook25d/items/leaf_shoes.png",
	"res://assets/art/storybook25d/items/bubble_gourd.png",
	"res://assets/art/storybook25d/items/paw_burst.png",
	"res://assets/art/storybook25d/effects/bubble_bomb.png",
	"res://assets/art/storybook25d/effects/trap_bubble.png",
	"res://assets/art/storybook25d/effects/pop_core.png",
	"res://assets/art/storybook25d/effects/cross_splash.png",
	"res://assets/art/storybook25d/effects/foam_burst.png",
	"res://assets/art/storybook25d/lobby/storybook_lobby.png",
	"res://assets/art/storybook25d/ui/paper_panel.png",
	"res://assets/art/storybook25d/ui/button_normal.png",
	"res://assets/art/storybook25d/ui/button_normal_wide.png",
	"res://assets/art/storybook25d/ui/button_hover.png",
	"res://assets/art/storybook25d/ui/button_hover_wide.png",
	"res://assets/art/storybook25d/ui/button_pressed.png",
	"res://assets/art/storybook25d/ui/button_pressed_wide.png",
]
const WIND_PLANTS: Array[String] = [
	"conifer",
	"bush",
	"mushrooms",
	"flowers",
	"lavender",
]

var _checks: int = 0
var _failures: int = 0


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	for character_id: String in CharacterCatalog.IDS:
		_validate_character(character_id)
		_validate_character_bubble(character_id)
	for plant_id: String in WIND_PLANTS:
		_validate_wind_plant(plant_id)
	for texture_path: String in SHARED_TEXTURE_PATHS:
		_validate_shared_texture(texture_path)
	_check(
		not _directory_contains_extension("res://assets", "glb"),
		"runtime assets contain no residual GLB files"
	)
	print("Forest ImageGen sprites: %d checks, %d failures" % [_checks, _failures])
	quit(1 if _failures > 0 else 0)


func _validate_character(character_id: String) -> void:
	var definition := CharacterCatalog.get_definition(character_id)
	var sprite_set := definition.load_sprite_set()
	_check(sprite_set != null, "%s exposes a CharacterSpriteSet" % character_id)
	_check(
		definition.sprite_set_path.ends_with("/%s" % character_id),
		"%s points to its approved ImageGen identity set" % character_id
	)
	if sprite_set == null:
		return
	_check(
		CharacterSpriteSet.ACTIONS == [
			&"Idle",
			&"WalkUp",
			&"WalkDown",
			&"WalkLeft",
			&"WalkRight",
			&"Trapped",
		],
		"%s exposes exactly the approved six logical actions" % character_id
	)
	var pose_count := 0
	for direction: StringName in CharacterSpriteSet.DIRECTIONS:
		_validate_pose(
			character_id,
			"Idle/%s" % direction,
			sprite_set.texture_for(&"Idle", 0, direction),
			sprite_set.mask_for(&"Idle", 0, direction)
		)
		pose_count += 1
	for action: StringName in CharacterSpriteSet.WALK_ACTIONS:
		for frame_index: int in range(CharacterSpriteSet.WALK_FRAME_COUNT):
			_validate_pose(
				character_id,
				"%s/%d" % [action, frame_index],
				sprite_set.texture_for(action, frame_index),
				sprite_set.mask_for(action, frame_index)
			)
			pose_count += 1
	_validate_pose(
		character_id,
		"Trapped",
		sprite_set.texture_for(&"Trapped"),
		sprite_set.mask_for(&"Trapped")
	)
	pose_count += 1
	_check(
		pose_count == 21,
		"%s contains exactly 21 directional poses" % character_id
	)
	_check(
		sprite_set.texture_path_for(&"WalkLeft", 0) \
			!= sprite_set.texture_path_for(&"WalkRight", 0),
		"%s left and right use independent source frames" % character_id
	)
	for legacy_stem: String in [
		"idle",
		"waddle",
		"waddle_body",
		"waddle_left_arm",
		"waddle_right_arm",
		"waddle_left_foot",
		"waddle_right_foot",
		"place_bubble",
		"defeat",
		"victory",
	]:
		_check(
			not FileAccess.file_exists(
				"res://assets/art/storybook25d/characters/%s/%s.png"
					% [character_id, legacy_stem]
			),
			"%s has no legacy %s runtime artwork" % [character_id, legacy_stem]
		)


func _validate_character_bubble(character_id: String) -> void:
	var root_path := (
		"res://assets/art/storybook25d/effects/character_bubbles/"
		+ character_id
	)
	var texture := load(root_path + ".png") as Texture2D
	var mask := load(root_path + "_mask.png") as Texture2D
	_check(texture != null, "%s has one exclusive ImageGen bubble" % character_id)
	_check(mask != null, "%s bubble has a local team-color mask" % character_id)
	if texture == null or mask == null:
		return
	var image := texture.get_image()
	var mask_image := mask.get_image()
	_check(
		image.get_size() == Vector2i(512, 512),
		"%s bubble uses the shared 512px canvas" % character_id
	)
	_check(
		_corners_are_transparent(image),
		"%s bubble has halo-free transparent corners" % character_id
	)
	_check(
		_mask_has_tint_pixels(mask_image),
		"%s bubble exposes a broad replaceable team-color wash" % character_id
	)
	var opaque_rect := _opaque_rect(image)
	_check(
		float(opaque_rect.size.x) * BubbleView3D.BUBBLE_PIXEL_SIZE >= 0.8,
		"%s bubble occupies at least four fifths of one board cell" % character_id
	)
	_check(
		opaque_rect.end.y >= 492,
		"%s bubble shares the grounded visual baseline" % character_id
	)


func _validate_wind_plant(plant_id: String) -> void:
	var reference := load(
		"res://assets/art/storybook25d/environment/%s.png" % plant_id
	) as Texture2D
	_check(reference != null, "%s keeps its approved identity texture" % plant_id)
	if reference == null:
		return
	var first_data := PackedByteArray()
	for frame_index: int in range(4):
		var frame := load(
			"res://assets/art/storybook25d/environment/wind/%s_%d.png"
				% [plant_id, frame_index]
		) as Texture2D
		_check(
			frame != null,
			"%s wind frame %d imports" % [plant_id, frame_index]
		)
		if frame == null:
			continue
		var image := frame.get_image()
		_check(
			image.get_size() == Vector2i(
				reference.get_width(),
				reference.get_height()
			),
			"%s wind frame %d preserves runtime scale"
				% [plant_id, frame_index]
		)
		_check(
			_corners_are_transparent(image),
			"%s wind frame %d has clean cutout corners"
				% [plant_id, frame_index]
		)
		_check(
			_opaque_rect(image).end.y >= image.get_height() - 4,
			"%s wind frame %d keeps the root baseline planted"
				% [plant_id, frame_index]
		)
		if frame_index == 0:
			first_data = image.get_data()
		else:
			_check(
				image.get_data() != first_data,
				"%s wind frame %d is independently redrawn"
					% [plant_id, frame_index]
			)


func _validate_pose(
		character_id: String,
		pose_name: String,
		texture: Texture2D,
		mask: Texture2D
	) -> void:
	_check(texture != null, "%s loads %s artwork" % [character_id, pose_name])
	_check(mask != null, "%s loads %s team mask" % [character_id, pose_name])
	if texture == null or mask == null:
		return
	var image := texture.get_image()
	var mask_image := mask.get_image()
	_check(
		image.get_size() == Vector2i(512, 512),
		"%s %s is normalized to 512x512" % [character_id, pose_name]
	)
	_check(
		image.get_size() == mask_image.get_size(),
		"%s %s base and mask share one anchor" % [character_id, pose_name]
	)
	_check(
		_corners_are_transparent(image),
		"%s %s has transparent halo-free corners" % [character_id, pose_name]
	)
	_check(
		_mask_has_tint_pixels(mask_image),
		"%s %s contains only local team-color pixels" % [character_id, pose_name]
	)
	var opaque_rect := _opaque_rect(image)
	_check(
		opaque_rect.has_area() and opaque_rect.end.y >= 480,
		"%s %s feet stay on the shared baseline" % [character_id, pose_name]
	)


func _validate_shared_texture(path: String) -> void:
	var texture := load(path) as Texture2D
	_check(texture != null, "%s imports as a runtime Texture2D" % path.get_file())
	if texture == null:
		return
	_check(
		texture.get_width() <= 2048 and texture.get_height() <= 2048,
		"%s stays within the 2048px scene-art budget" % path.get_file()
	)


func _corners_are_transparent(image: Image) -> bool:
	if image.is_empty():
		return false
	var maximum_x := image.get_width() - 1
	var maximum_y := image.get_height() - 1
	for point: Vector2i in [
		Vector2i.ZERO,
		Vector2i(maximum_x, 0),
		Vector2i(0, maximum_y),
		Vector2i(maximum_x, maximum_y),
	]:
		if image.get_pixelv(point).a > 0.03:
			return false
	return true


func _mask_has_tint_pixels(image: Image) -> bool:
	for y: int in range(0, image.get_height(), 4):
		for x: int in range(0, image.get_width(), 4):
			var pixel := image.get_pixel(x, y)
			if pixel.a > 0.1 and pixel.r > 0.2:
				return true
	return false


func _opaque_rect(image: Image) -> Rect2i:
	var minimum := Vector2i(image.get_width(), image.get_height())
	var maximum := Vector2i(-1, -1)
	for y: int in range(image.get_height()):
		for x: int in range(image.get_width()):
			if image.get_pixel(x, y).a <= 0.03:
				continue
			minimum = minimum.min(Vector2i(x, y))
			maximum = maximum.max(Vector2i(x, y))
	if maximum.x < minimum.x:
		return Rect2i()
	return Rect2i(minimum, maximum - minimum + Vector2i.ONE)


func _directory_contains_extension(path: String, extension: String) -> bool:
	var directory := DirAccess.open(path)
	if directory == null:
		return false
	directory.list_dir_begin()
	var entry := directory.get_next()
	while not entry.is_empty():
		var child_path := path.path_join(entry)
		if directory.current_is_dir():
			if not entry.begins_with(".") \
					and _directory_contains_extension(child_path, extension):
				directory.list_dir_end()
				return true
		elif entry.get_extension().to_lower() == extension.to_lower():
			directory.list_dir_end()
			return true
		entry = directory.get_next()
	directory.list_dir_end()
	return false


func _check(condition: bool, description: String) -> void:
	_checks += 1
	if condition:
		return
	_failures += 1
	push_error("STORYBOOK SPRITE ASSET FAILED: %s" % description)
