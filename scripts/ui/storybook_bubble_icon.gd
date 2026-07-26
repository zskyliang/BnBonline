class_name StorybookBubbleIcon
extends Control
## Lightweight leafy soap-bubble silhouette shared by the match HUD.

var bubble_color: Color = Color("#59b9dd"):
	set(value):
		bubble_color = value
		queue_redraw()


func _ready() -> void:
	custom_minimum_size = Vector2(36.0, 36.0)
	mouse_filter = Control.MOUSE_FILTER_IGNORE


func _draw() -> void:
	var center := size * Vector2(0.5, 0.56)
	var radius := minf(size.x, size.y) * 0.31
	draw_circle(center + Vector2(1.5, 2.0), radius * 1.05, Color(0.16, 0.12, 0.14, 0.18))
	draw_circle(center, radius, bubble_color)
	draw_circle(center - Vector2(radius * 0.26, radius * 0.26), radius * 0.22, Color(1.0, 0.98, 0.9, 0.68))
	draw_rect(
		Rect2(center.x - radius * 0.18, center.y - radius * 1.16, radius * 0.36, radius * 0.3),
		Color("#9d684d"),
		true
	)
	var leaf_color := Color("#6f8f4b")
	draw_colored_polygon(
		PackedVector2Array([
			center + Vector2(-radius * 0.02, -radius * 1.16),
			center + Vector2(-radius * 0.52, -radius * 1.58),
			center + Vector2(-radius * 0.14, -radius * 1.72),
		]),
		leaf_color
	)
	draw_colored_polygon(
		PackedVector2Array([
			center + Vector2(radius * 0.02, -radius * 1.16),
			center + Vector2(radius * 0.52, -radius * 1.58),
			center + Vector2(radius * 0.14, -radius * 1.72),
		]),
		leaf_color
	)
	for index: int in range(3):
		draw_circle(
			center + Vector2((-0.42 + float(index) * 0.42) * radius, radius * 0.28),
			radius * 0.075,
			StorybookMaterialLibrary.PAPER
		)
