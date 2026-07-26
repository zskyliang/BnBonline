class_name CharacterCatalog
extends RefCounted
## Cosmetic-only registry for the eight upright forest animals.

const IDS: Array[String] = [
	"cat", "dog", "rabbit", "bear",
	"fox", "raccoon", "penguin", "capybara",
]

## Role IDs used by the previous eight-character release, migrated by card order.
const LEGACY_ROLE_MAP: Dictionary = {
	"builder": "cat",
	"chef": "dog",
	"cowboy": "rabbit",
	"wizard": "bear",
	"ninja": "fox",
	"medic": "raccoon",
	"viking": "penguin",
	"fighter": "capybara",
}

## Zodiac IDs predate the role roster and need a separate map because dog and
## rabbit are also valid IDs in the new animal roster.
const LEGACY_ZODIAC_MAP: Dictionary = {
	"rat": "cat",
	"ox": "penguin",
	"tiger": "fox",
	"rabbit": "raccoon",
	"dragon": "bear",
	"snake": "rabbit",
	"horse": "capybara",
	"goat": "dog",
	"monkey": "fox",
	"rooster": "rabbit",
	"dog": "raccoon",
	"pig": "dog",
}

const DISPLAY_NAMES: Dictionary = {
	"cat": "猫",
	"dog": "狗",
	"rabbit": "兔",
	"bear": "熊",
	"fox": "狐狸",
	"raccoon": "浣熊",
	"penguin": "企鹅",
	"capybara": "水豚",
}
const BASE_COLORS: Dictionary = {
	"cat": Color("#68747b"),
	"dog": Color("#c98a4f"),
	"rabbit": Color("#d9c6a2"),
	"bear": Color("#8f5c38"),
	"fox": Color("#c96b3f"),
	"raccoon": Color("#77756f"),
	"penguin": Color("#3e4c55"),
	"capybara": Color("#9b6d4c"),
}
const ACCENT_COLORS: Dictionary = {
	"cat": Color("#e7dfd0"),
	"dog": Color("#f1dfc6"),
	"rabbit": Color("#f2e7d0"),
	"bear": Color("#d8ad73"),
	"fox": Color("#f1d9bd"),
	"raccoon": Color("#d6d1c5"),
	"penguin": Color("#f2ead8"),
	"capybara": Color("#d8b58a"),
}

static var _definitions: Dictionary = {}


static func get_all() -> Array[CharacterDefinition]:
	_ensure_definitions()
	var result: Array[CharacterDefinition] = []
	for character_id: String in IDS:
		result.append(_definitions[character_id])
	return result


static func get_definition(character_id: String) -> CharacterDefinition:
	_ensure_definitions()
	var migrated_id := migrate_legacy_id(character_id)
	return _definitions.get(migrated_id, _definitions["cat"]) as CharacterDefinition


static func is_valid_id(character_id: String) -> bool:
	return character_id in IDS


static func assign_ai_characters(
		player_id: String,
		count: int,
		rng: RandomNumberGenerator = null
	) -> Array[String]:
	var candidates: Array[String] = IDS.duplicate()
	candidates.erase(migrate_legacy_id(player_id))
	var random := rng
	if random == null:
		random = RandomNumberGenerator.new()
		random.randomize()
	for index: int in range(candidates.size() - 1, 0, -1):
		var swap_index := random.randi_range(0, index)
		var temporary := candidates[index]
		candidates[index] = candidates[swap_index]
		candidates[swap_index] = temporary
	var result: Array[String] = []
	for index: int in range(mini(clampi(count, 0, 4), candidates.size())):
		result.append(candidates[index])
	return result


static func migrate_legacy_id(legacy_id: String) -> String:
	if legacy_id in IDS:
		return legacy_id
	return str(LEGACY_ROLE_MAP.get(legacy_id, "cat"))


static func migrate_zodiac_id(legacy_id: String) -> String:
	return str(LEGACY_ZODIAC_MAP.get(legacy_id, "cat"))


static func _ensure_definitions() -> void:
	if not _definitions.is_empty():
		return
	for character_id: String in IDS:
		_add(character_id)


static func _add(character_id: String) -> void:
	var definition := CharacterDefinition.new().configure(
		character_id,
		str(DISPLAY_NAMES[character_id]),
		"res://assets/models/characters/%s.glb" % character_id,
		BASE_COLORS[character_id] as Color,
		ACCENT_COLORS[character_id] as Color,
		["Idle"],
		["Waddle"],
		0.0,
		1.0,
		"",
		["Head"]
	)
	definition.team_tint_material_names = ["TeamTint", "FootRing"]
	definition.place_bubble_animation_aliases = ["PlaceBubble"]
	definition.trapped_animation_aliases = ["Trapped"]
	definition.defeat_animation_aliases = ["Defeat"]
	definition.victory_animation_aliases = ["Victory"]
	_definitions[character_id] = definition
