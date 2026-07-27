class_name CharacterStatRadar
extends Control
## Compact three-axis radar for one animal's six starting attribute points.

const MAX_INITIAL_POINTS := 4.0
const AXIS_LABELS: Array[String] = ["速度", "水泡", "威力"]
const GRID_COLOR := Color(0.28, 0.25, 0.2, 0.22)
const AXIS_COLOR := Color(0.25, 0.22, 0.18, 0.38)
const TEXT_COLOR := Color("#694139")

var definition: CharacterDefinition
var _points := Vector3(2.0, 2.0, 2.0)
var _accent := Color("#d89a45")


func configure(character_definition: CharacterDefinition) -> void:
	definition = character_definition
	_points = Vector3(
		float(definition.initial_speed_points),
		float(definition.initial_bubble_points),
		float(definition.initial_power_points)
	)
	_accent = definition.theme_color
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	queue_redraw()


func attribute_snapshot() -> Dictionary:
	return {
		"speed": int(_points.x),
		"bubble": int(_points.y),
		"power": int(_points.z),
		"total": int(_points.x + _points.y + _points.z),
	}


func _draw() -> void:
	var center := Vector2(size.x * 0.5, size.y * 0.52)
	var radius := minf(size.x * 0.23, size.y * 0.32)
	var axes := _axis_vertices(center, radius)
	for ring_index: int in range(1, 5):
		var ring_scale := float(ring_index) / MAX_INITIAL_POINTS
		var ring := PackedVector2Array()
		for vertex: Vector2 in axes:
			ring.append(center.lerp(vertex, ring_scale))
		ring.append(ring[0])
		draw_polyline(ring, GRID_COLOR, 1.0, true)
	for vertex: Vector2 in axes:
		draw_line(center, vertex, AXIS_COLOR, 1.0, true)

	var values: Array[float] = [_points.x, _points.y, _points.z]
	var polygon := PackedVector2Array()
	for index: int in range(3):
		polygon.append(
			center.lerp(
				axes[index],
				clampf(values[index] / MAX_INITIAL_POINTS, 0.0, 1.0)
			)
		)
	draw_colored_polygon(polygon, Color(_accent, 0.36))
	var outline := polygon.duplicate()
	outline.append(outline[0])
	draw_polyline(outline, _accent.darkened(0.18), 2.0, true)
	for point: Vector2 in polygon:
		draw_circle(point, 2.5, _accent.darkened(0.12))

	var font := get_theme_default_font()
	var font_size := 10
	draw_string(
		font,
		Vector2(center.x - 20.0, 10.0),
		"%s %d" % [tr(AXIS_LABELS[0]), int(_points.x)],
		HORIZONTAL_ALIGNMENT_CENTER,
		40.0,
		font_size,
		TEXT_COLOR
	)
	draw_string(
		font,
		Vector2(2.0, size.y - 2.0),
		"%s %d" % [tr(AXIS_LABELS[1]), int(_points.y)],
		HORIZONTAL_ALIGNMENT_LEFT,
		45.0,
		font_size,
		TEXT_COLOR
	)
	draw_string(
		font,
		Vector2(size.x - 47.0, size.y - 2.0),
		"%s %d" % [tr(AXIS_LABELS[2]), int(_points.z)],
		HORIZONTAL_ALIGNMENT_RIGHT,
		45.0,
		font_size,
		TEXT_COLOR
	)


func _axis_vertices(center: Vector2, radius: float) -> Array[Vector2]:
	return [
		center + Vector2(0.0, -radius),
		center + Vector2(-0.866, 0.5) * radius,
		center + Vector2(0.866, 0.5) * radius,
	]
