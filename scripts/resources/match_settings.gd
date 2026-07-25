class_name MatchSettings
extends Resource
## User-adjustable match configuration persisted with ConfigFile/localStorage.

const SAVE_PATH := "user://settings.cfg"
const WEB_STORAGE_KEY := "bnb.settings.v3"
const LEGACY_WEB_STORAGE_KEY := "bnb.settings.v2"

var map_id: String = MapCatalog.HARBOR_MARKET
var ai_count: int = 3
var max_speed: int = 300
var max_bubbles: int = 8
var max_power: int = 10
var bubble_skin: String = "aqua"
var character_id: String = "builder"


func normalize() -> void:
	map_id = MapCatalog.migrate_legacy_id(map_id)
	ai_count = clampi(ai_count, 0, 4)
	max_speed = clampi(max_speed, int(GameConstants.INITIAL_SPEED), 1000)
	max_bubbles = clampi(max_bubbles, GameConstants.INITIAL_BUBBLES, 20)
	max_power = clampi(max_power, GameConstants.INITIAL_POWER, 20)
	if bubble_skin in ["basketball", "coral"]:
		bubble_skin = "coral"
	else:
		bubble_skin = "aqua"
	if not CharacterCatalog.is_valid_id(character_id):
		character_id = "builder"


func save_to_disk() -> void:
	normalize()
	if OS.has_feature("web"):
		_save_to_web_storage()
		return
	var config := ConfigFile.new()
	config.set_value("match", "map_id", map_id)
	config.set_value("match", "ai_count", ai_count)
	config.set_value("match", "max_speed", max_speed)
	config.set_value("match", "max_bubbles", max_bubbles)
	config.set_value("match", "max_power", max_power)
	config.set_value("match", "bubble_skin", bubble_skin)
	config.set_value("match", "character_id", character_id)
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
		settings.map_id = str(config.get_value("match", "map_id", settings.map_id))
		settings.ai_count = int(config.get_value("match", "ai_count", settings.ai_count))
		settings.max_speed = int(config.get_value("match", "max_speed", settings.max_speed))
		settings.max_bubbles = int(config.get_value("match", "max_bubbles", settings.max_bubbles))
		settings.max_power = int(config.get_value("match", "max_power", settings.max_power))
		settings.bubble_skin = str(config.get_value("match", "bubble_skin", settings.bubble_skin))
		if config.has_section_key("match", "character_id"):
			settings.character_id = str(config.get_value("match", "character_id"))
		else:
			settings.character_id = CharacterCatalog.migrate_legacy_id(
				str(config.get_value("match", "zodiac_id", "rat"))
			)
	settings.normalize()
	return settings


func to_dictionary() -> Dictionary:
	normalize()
	return {
		"map_id": map_id,
		"ai_count": ai_count,
		"max_speed": max_speed,
		"max_bubbles": max_bubbles,
		"max_power": max_power,
		"bubble_skin": bubble_skin,
		"character_id": character_id,
	}


func apply_dictionary(values: Dictionary) -> void:
	map_id = str(values.get("map_id", map_id))
	ai_count = int(values.get("ai_count", ai_count))
	max_speed = int(values.get("max_speed", max_speed))
	max_bubbles = int(values.get("max_bubbles", max_bubbles))
	max_power = int(values.get("max_power", max_power))
	bubble_skin = str(values.get("bubble_skin", bubble_skin))
	if values.has("character_id"):
		character_id = str(values["character_id"])
	elif values.has("zodiac_id"):
		character_id = CharacterCatalog.migrate_legacy_id(str(values["zodiac_id"]))
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
		raw_value = storage.getItem(LEGACY_WEB_STORAGE_KEY)
	if raw_value == null or str(raw_value).is_empty():
		return
	var parsed: Variant = JSON.parse_string(str(raw_value))
	if parsed is Dictionary:
		apply_dictionary(parsed as Dictionary)
