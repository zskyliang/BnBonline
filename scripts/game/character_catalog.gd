class_name CharacterCatalog
extends RefCounted
## Cosmetic-only registry for the eight clay characters.

const IDS: Array[String] = [
	"builder", "chef", "cowboy", "wizard",
	"ninja", "medic", "viking", "fighter",
]
const ASSET_SOURCE := "https://quaternius.com/packs/ultimatedanimatedcharacter.html"

static var _definitions: Dictionary = {}


static func get_all() -> Array[CharacterDefinition]:
	_ensure_definitions()
	var result: Array[CharacterDefinition] = []
	for character_id in IDS:
		result.append(_definitions[character_id])
	return result


static func get_definition(character_id: String) -> CharacterDefinition:
	_ensure_definitions()
	return _definitions.get(character_id, _definitions["builder"]) as CharacterDefinition


static func is_valid_id(character_id: String) -> bool:
	return character_id in IDS


static func assign_ai_characters(
		player_id: String,
		count: int,
		rng: RandomNumberGenerator = null
	) -> Array[String]:
	var candidates: Array[String] = IDS.duplicate()
	candidates.erase(player_id)
	var random := rng
	if random == null:
		random = RandomNumberGenerator.new()
		random.randomize()
	for index in range(candidates.size() - 1, 0, -1):
		var swap_index := random.randi_range(0, index)
		var temporary := candidates[index]
		candidates[index] = candidates[swap_index]
		candidates[swap_index] = temporary
	var result: Array[String] = []
	for index in range(mini(clampi(count, 0, 4), candidates.size())):
		result.append(candidates[index])
	return result


static func migrate_legacy_id(legacy_id: String) -> String:
	const LEGACY_MAP := {
		"rat": "builder",
		"ox": "viking",
		"tiger": "ninja",
		"rabbit": "medic",
		"dragon": "wizard",
		"snake": "cowboy",
		"horse": "fighter",
		"goat": "chef",
		"monkey": "ninja",
		"rooster": "cowboy",
		"dog": "medic",
		"pig": "chef",
	}
	return str(LEGACY_MAP.get(legacy_id, "builder"))


static func _ensure_definitions() -> void:
	if not _definitions.is_empty():
		return
	var idle: Array[String] = ["Idle"]
	var movement: Array[String] = ["Run", "Walk"]
	_add("builder", "建筑工", Color("#d9664c"), Color("#f4e7d1"), idle, movement)
	_add("chef", "厨师", Color("#e4b84f"), Color("#fff8e8"), idle, movement)
	_add("cowboy", "牛仔", Color("#3978a8"), Color("#f0d29d"), idle, movement)
	_add("wizard", "法师", Color("#8055a4"), Color("#f0d7ff"), idle, movement)
	_add("ninja", "忍者", Color("#342f35"), Color("#e8685f"), idle, movement)
	_add("medic", "医护", Color("#58a99a"), Color("#f7efdc"), idle, movement)
	_add("viking", "维京人", Color("#d8783c"), Color("#f0d099"), idle, movement)
	_add("fighter", "战士", Color("#d8698b"), Color("#f8dce4"), idle, movement)


static func _add(
		character_id: String,
		chinese_name: String,
		color: Color,
		accent: Color,
		idle_aliases: Array[String],
		move_aliases: Array[String]
	) -> void:
	var definition := CharacterDefinition.new().configure(
		character_id,
		chinese_name,
		"res://assets/models/characters/%s.glb" % character_id,
		color,
		accent,
		idle_aliases.duplicate(),
		move_aliases.duplicate(),
		0.0,
		1.0,
		ASSET_SOURCE,
		["Head", "mixamorig_Head"]
	)
	_definitions[character_id] = definition
