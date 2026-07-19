class_name MatchSettings
extends Resource
## User-adjustable match configuration persisted with ConfigFile.

const SAVE_PATH: String = "user://settings.cfg"

var map_id: String = "classic"
var ai_count: int = 3
var max_speed: int = 300
var max_bubbles: int = 8
var max_power: int = 10
var bubble_skin: String = "football"

func normalize() -> void:
	if map_id != "classic" and map_id != "windmill-heart":
		map_id = "classic"
	ai_count = clampi(ai_count, 0, 4)
	max_speed = clampi(max_speed, int(GameConstants.INITIAL_SPEED), 1000)
	max_bubbles = clampi(max_bubbles, GameConstants.INITIAL_BUBBLES, 20)
	max_power = clampi(max_power, GameConstants.INITIAL_POWER, 20)
	if bubble_skin != "basketball":
		bubble_skin = "football"

func save_to_disk() -> void:
	normalize()
	var config := ConfigFile.new()
	config.set_value("match", "map_id", map_id)
	config.set_value("match", "ai_count", ai_count)
	config.set_value("match", "max_speed", max_speed)
	config.set_value("match", "max_bubbles", max_bubbles)
	config.set_value("match", "max_power", max_power)
	config.set_value("match", "bubble_skin", bubble_skin)
	var error: Error = config.save(SAVE_PATH)
	if error != OK:
		push_warning("Unable to save settings: %s" % error_string(error))

static func load_from_disk() -> MatchSettings:
	var settings := MatchSettings.new()
	var config := ConfigFile.new()
	if config.load(SAVE_PATH) == OK:
		settings.map_id = str(config.get_value("match", "map_id", settings.map_id))
		settings.ai_count = int(config.get_value("match", "ai_count", settings.ai_count))
		settings.max_speed = int(config.get_value("match", "max_speed", settings.max_speed))
		settings.max_bubbles = int(config.get_value("match", "max_bubbles", settings.max_bubbles))
		settings.max_power = int(config.get_value("match", "max_power", settings.max_power))
		settings.bubble_skin = str(config.get_value("match", "bubble_skin", settings.bubble_skin))
	settings.normalize()
	return settings

