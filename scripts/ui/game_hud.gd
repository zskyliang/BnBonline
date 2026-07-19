class_name GameHud
extends Control
## Persistent match panel plus pause and round-result overlays.

signal map_selected(map_id: String)
signal ai_count_selected(count: int)
signal max_speed_changed(value: int)
signal max_bubbles_changed(value: int)
signal max_power_changed(value: int)
signal bubble_skin_selected(skin: String)
signal resume_requested
signal restart_requested
signal quit_requested

var _timer_label: Label
var _score_box: VBoxContainer
var _speed_spin: SpinBox
var _bubble_spin: SpinBox
var _power_spin: SpinBox
var _skin_option: OptionButton
var _map_option: OptionButton
var _ai_option: OptionButton
var _pause_overlay: Control
var _end_overlay: Control
var _end_title: Label
var _end_detail: Label
var _syncing: bool = false

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_PASS
	_build_panel()
	_pause_overlay = _build_overlay(false)
	_end_overlay = _build_overlay(true)

func sync_settings(settings: MatchSettings) -> void:
	_syncing = true
	_speed_spin.value = settings.max_speed
	_bubble_spin.value = settings.max_bubbles
	_power_spin.value = settings.max_power
	_skin_option.select(1 if settings.bubble_skin == "basketball" else 0)
	_map_option.select(1 if settings.map_id == "windmill-heart" else 0)
	_ai_option.select(settings.ai_count)
	_syncing = false

func update_timer(seconds_left: float) -> void:
	var total_seconds: int = maxi(0, ceili(seconds_left))
	_timer_label.text = "%02d:%02d" % [total_seconds / 60, total_seconds % 60]

func update_scores(entries: Array[Dictionary]) -> void:
	for child: Node in _score_box.get_children():
		child.queue_free()
	for entry: Dictionary in entries:
		var label := Label.new()
		label.text = "%s  击败: %d" % [entry.get("name", "?"), entry.get("kills", 0)]
		label.add_theme_font_size_override("font_size", 14)
		label.add_theme_color_override("font_color", Color("dce9ff"))
		_score_box.add_child(label)

func show_pause() -> void:
	_pause_overlay.visible = true

func hide_pause() -> void:
	_pause_overlay.visible = false

func show_result(title: String, detail: String) -> void:
	_end_title.text = title
	_end_detail.text = detail
	_end_overlay.visible = true

func hide_result() -> void:
	_end_overlay.visible = false

func _build_panel() -> void:
	var panel := PanelContainer.new()
	panel.position = Vector2(812, 10)
	panel.size = Vector2(218, 580)
	panel.mouse_filter = Control.MOUSE_FILTER_STOP
	var panel_style := StyleBoxFlat.new()
	panel_style.bg_color = Color("101722")
	panel_style.border_color = Color("31445f")
	panel_style.set_border_width_all(1)
	panel_style.corner_radius_top_left = 12
	panel_style.corner_radius_top_right = 12
	panel_style.corner_radius_bottom_left = 12
	panel_style.corner_radius_bottom_right = 12
	panel_style.content_margin_left = 12
	panel_style.content_margin_right = 12
	panel_style.content_margin_top = 10
	panel_style.content_margin_bottom = 10
	panel.add_theme_stylebox_override("panel", panel_style)
	add_child(panel)
	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	panel.add_child(scroll)
	var content := VBoxContainer.new()
	content.custom_minimum_size.x = 188
	content.add_theme_constant_override("separation", 6)
	scroll.add_child(content)
	var title := _make_label("本局信息", 18, Color.WHITE)
	title.add_theme_constant_override("outline_size", 2)
	content.add_child(title)
	_timer_label = _make_label("05:00", 30, Color("ffe48a"))
	content.add_child(_timer_label)
	_speed_spin = _add_spin_setting(content, "人物最大速度 (px/s)", 150, 1000, 25, 300)
	_speed_spin.value_changed.connect(_on_speed_changed)
	_bubble_spin = _add_spin_setting(content, "人物最大水泡数", 2, 20, 1, 8)
	_bubble_spin.value_changed.connect(_on_bubbles_changed)
	_power_spin = _add_spin_setting(content, "人物最大威力 (格)", 2, 20, 1, 10)
	_power_spin.value_changed.connect(_on_power_changed)
	_skin_option = _add_option_setting(content, "玩家水泡皮肤", ["足球", "篮球"])
	_skin_option.item_selected.connect(_on_skin_selected)
	_map_option = _add_option_setting(content, "游戏地图", ["当前地图（经典）", "风车爱心地图"])
	_map_option.item_selected.connect(_on_map_selected)
	_ai_option = _add_option_setting(content, "AI 敌人数", ["0", "1", "2", "3", "4"])
	_ai_option.item_selected.connect(_on_ai_selected)
	var score_title := _make_label("比分", 15, Color("9fc4ff"))
	content.add_child(score_title)
	_score_box = VBoxContainer.new()
	_score_box.add_theme_constant_override("separation", 4)
	content.add_child(_score_box)
	var controls := _make_label("方向键/WASD 移动\n空格 放泡｜1 自救｜Esc 暂停", 12, Color("93a9c7"))
	controls.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	content.add_child(controls)

func _build_overlay(is_result: bool) -> Control:
	var shade := ColorRect.new()
	shade.position = Vector2.ZERO
	shade.size = Vector2(800, 600)
	shade.color = Color(0.02, 0.04, 0.08, 0.72)
	shade.mouse_filter = Control.MOUSE_FILTER_STOP
	shade.process_mode = Node.PROCESS_MODE_ALWAYS
	shade.visible = false
	add_child(shade)
	var panel := PanelContainer.new()
	panel.position = Vector2(220, 160)
	panel.size = Vector2(360, 280)
	var style := StyleBoxFlat.new()
	style.bg_color = Color("101b2d")
	style.border_color = Color("4e83c6")
	style.set_border_width_all(2)
	style.set_corner_radius_all(14)
	style.set_content_margin_all(20)
	panel.add_theme_stylebox_override("panel", style)
	shade.add_child(panel)
	var box := VBoxContainer.new()
	box.alignment = BoxContainer.ALIGNMENT_CENTER
	box.add_theme_constant_override("separation", 14)
	panel.add_child(box)
	var heading := _make_label("本局结束" if is_result else "游戏暂停", 28, Color("ffe48a"))
	heading.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	box.add_child(heading)
	var detail := _make_label("" if is_result else "Esc 或点击继续返回比赛", 15, Color("dce9ff"))
	detail.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	detail.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	box.add_child(detail)
	if is_result:
		_end_title = heading
		_end_detail = detail
	else:
		var resume := Button.new()
		resume.text = "继续游戏"
		resume.pressed.connect(resume_requested.emit)
		box.add_child(resume)
	var restart := Button.new()
	restart.text = "重新开始"
	restart.pressed.connect(restart_requested.emit)
	box.add_child(restart)
	var quit := Button.new()
	quit.text = "退出游戏"
	quit.pressed.connect(quit_requested.emit)
	box.add_child(quit)
	return shade

func _add_spin_setting(
		parent: VBoxContainer,
		label_text: String,
		minimum: float,
		maximum: float,
		step: float,
		initial: float
	) -> SpinBox:
	parent.add_child(_make_label(label_text, 12, Color("b9cae4")))
	var spin := SpinBox.new()
	spin.min_value = minimum
	spin.max_value = maximum
	spin.step = step
	spin.value = initial
	spin.custom_minimum_size.y = 28
	parent.add_child(spin)
	return spin

func _add_option_setting(parent: VBoxContainer, label_text: String, options: Array[String]) -> OptionButton:
	parent.add_child(_make_label(label_text, 12, Color("b9cae4")))
	var option := OptionButton.new()
	for text: String in options:
		option.add_item(text)
	option.custom_minimum_size.y = 28
	parent.add_child(option)
	return option

func _make_label(text: String, font_size: int, color: Color) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	return label

func _on_speed_changed(value: float) -> void:
	if not _syncing:
		max_speed_changed.emit(int(value))

func _on_bubbles_changed(value: float) -> void:
	if not _syncing:
		max_bubbles_changed.emit(int(value))

func _on_power_changed(value: float) -> void:
	if not _syncing:
		max_power_changed.emit(int(value))

func _on_skin_selected(index: int) -> void:
	if not _syncing:
		bubble_skin_selected.emit("basketball" if index == 1 else "football")

func _on_map_selected(index: int) -> void:
	if not _syncing:
		map_selected.emit("windmill-heart" if index == 1 else "classic")

func _on_ai_selected(index: int) -> void:
	if not _syncing:
		ai_count_selected.emit(index)

