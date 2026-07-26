class_name PaintPalette
extends RefCounted
## Stable paint-team and seven-color definitions shared by logic and UI.

const TEAM_NEUTRAL: int = 0
const TEAM_PLAYER: int = 1
const TEAM_AI: int = 2

const DEFAULT_PLAYER_COLOR_ID: String = "red"
const COLOR_IDS: Array[String] = [
	"red", "orange", "yellow", "green", "cyan", "blue", "purple",
]

const COLORS: Dictionary = {
	"red": Color("#d94f45"),
	"orange": Color("#ee8a3b"),
	"yellow": Color("#ddbb3f"),
	"green": Color("#58a85d"),
	"cyan": Color("#39bfc3"),
	"blue": Color("#4b72cc"),
	"purple": Color("#8b5cc7"),
}

const LABELS: Dictionary = {
	"red": "红",
	"orange": "橙",
	"yellow": "黄",
	"green": "绿",
	"cyan": "青",
	"blue": "蓝",
	"purple": "紫",
}

const NEUTRAL_COLOR: Color = Color("#cfc7b2")


static func is_valid_color_id(color_id: String) -> bool:
	return color_id in COLOR_IDS


static func get_color(color_id: String) -> Color:
	return COLORS.get(color_id, COLORS[DEFAULT_PLAYER_COLOR_ID]) as Color


static func get_label(color_id: String) -> String:
	return str(LABELS.get(color_id, LABELS[DEFAULT_PLAYER_COLOR_ID]))


static func random_opponent_color(
		player_color_id: String,
		rng: RandomNumberGenerator
	) -> String:
	var candidates: Array[String] = COLOR_IDS.duplicate()
	candidates.erase(player_color_id)
	return candidates[rng.randi_range(0, candidates.size() - 1)]
