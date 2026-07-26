class_name MatchSettings
extends Resource
## Persisted appearance and camera preferences; campaign progress stays in memory.

const SAVE_PATH: String = "user://settings.cfg"
const WEB_STORAGE_KEY: String = "bnb.settings.v5"
const LEGACY_WEB_STORAGE_KEYS: Array[String] = [
	"bnb.settings.v4",
	"bnb.settings.v3",
	"bnb.settings.v2",
]

const DEFAULT_CAMERA_AZIMUTH: float = -5.0
const MIN_CAMERA_AZIMUTH: float = -35.0
const MAX_CAMERA_AZIMUTH: float = 35.0
const DEFAULT_CAMERA_ELEVATION: float = 38.0
const MIN_CAMERA_ELEVATION: float = 25.0
const MAX_CAMERA_ELEVATION: float = 75.0
const DEFAULT_CAMERA_ZOOM: float = 1.1
const MIN_CAMERA_ZOOM: float = 0.85
const MAX_CAMERA_ZOOM: float = 1.5

var character_id: String = "builder"
var player_color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID
var camera_azimuth: float = DEFAULT_CAMERA_AZIMUTH
var camera_elevation: float = DEFAULT_CAMERA_ELEVATION
var camera_zoom: float = DEFAULT_CAMERA_ZOOM


func normalize() -> void:
	if not CharacterCatalog.is_valid_id(character_id):
		character_id = "builder"
	if not PaintPalette.is_valid_color_id(player_color_id):
		player_color_id = PaintPalette.DEFAULT_PLAYER_COLOR_ID
	camera_azimuth = clampf(camera_azimuth, MIN_CAMERA_AZIMUTH, MAX_CAMERA_AZIMUTH)
	camera_elevation = clampf(
		camera_elevation,
		MIN_CAMERA_ELEVATION,
		MAX_CAMERA_ELEVATION
	)
	camera_zoom = clampf(camera_zoom, MIN_CAMERA_ZOOM, MAX_CAMERA_ZOOM)


func save_to_disk() -> void:
	normalize()
	if OS.has_feature("web"):
		_save_to_web_storage()
		return
	var config := ConfigFile.new()
	config.set_value("appearance", "character_id", character_id)
	config.set_value("appearance", "player_color_id", player_color_id)
	config.set_value("camera", "azimuth", camera_azimuth)
	config.set_value("camera", "elevation", camera_elevation)
	config.set_value("camera", "zoom", camera_zoom)
	var error := config.save(SAVE_PATH)
	if error != OK:
		push_warning("Unable to save settings: %s" % error_string(error))


static func load_from_disk() -> MatchSettings:
	var settings := MatchSettings.new()
	if OS.has_feature("web"):
		settings._load_from_web_storage()
		settings.normalize()
		return settings
	var config := ConfigFile.new()
	if config.load(SAVE_PATH) == OK:
		if config.has_section_key("appearance", "character_id"):
			settings.character_id = str(config.get_value("appearance", "character_id"))
		elif config.has_section_key("match", "character_id"):
			settings.character_id = str(config.get_value("match", "character_id"))
		else:
			settings.character_id = CharacterCatalog.migrate_legacy_id(
				str(config.get_value("match", "zodiac_id", "rat"))
			)
		settings.player_color_id = str(
			config.get_value(
				"appearance",
				"player_color_id",
				PaintPalette.DEFAULT_PLAYER_COLOR_ID
			)
		)
		settings.camera_azimuth = float(config.get_value(
			"camera",
			"azimuth",
			DEFAULT_CAMERA_AZIMUTH
		))
		settings.camera_elevation = float(config.get_value(
			"camera",
			"elevation",
			DEFAULT_CAMERA_ELEVATION
		))
		settings.camera_zoom = float(config.get_value(
			"camera",
			"zoom",
			DEFAULT_CAMERA_ZOOM
		))
	settings.normalize()
	return settings


func to_dictionary() -> Dictionary:
	normalize()
	return {
		"character_id": character_id,
		"player_color_id": player_color_id,
		"camera_azimuth": camera_azimuth,
		"camera_elevation": camera_elevation,
		"camera_zoom": camera_zoom,
	}


func apply_dictionary(values: Dictionary) -> void:
	if values.has("character_id"):
		character_id = str(values["character_id"])
	elif values.has("zodiac_id"):
		character_id = CharacterCatalog.migrate_legacy_id(str(values["zodiac_id"]))
	player_color_id = str(values.get("player_color_id", player_color_id))
	camera_azimuth = float(values.get("camera_azimuth", camera_azimuth))
	camera_elevation = float(values.get("camera_elevation", camera_elevation))
	camera_zoom = float(values.get("camera_zoom", camera_zoom))
	normalize()


func _save_to_web_storage() -> void:
	var storage: JavaScriptObject = JavaScriptBridge.get_interface("localStorage")
	if storage == null:
		return
	storage.setItem(WEB_STORAGE_KEY, JSON.stringify(to_dictionary()))


func _load_from_web_storage() -> void:
	var storage: JavaScriptObject = JavaScriptBridge.get_interface("localStorage")
	if storage == null:
		return
	var raw_value: Variant = storage.getItem(WEB_STORAGE_KEY)
	if raw_value == null or str(raw_value).is_empty():
		for legacy_key: String in LEGACY_WEB_STORAGE_KEYS:
			raw_value = storage.getItem(legacy_key)
			if raw_value != null and not str(raw_value).is_empty():
				break
	if raw_value == null or str(raw_value).is_empty():
		return
	var parsed: Variant = JSON.parse_string(str(raw_value))
	if parsed is Dictionary:
		apply_dictionary(parsed as Dictionary)
