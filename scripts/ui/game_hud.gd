class_name GameHud
extends Control

## Responsive app UI for lobby, match setup, in-match HUD, pause, and results.

signal setup_requested
signal match_requested(configuration: Dictionary)
signal resume_requested
signal restart_requested
signal lobby_requested
signal quit_requested
signal zoom_in_requested
signal zoom_out_requested
signal zoom_reset_requested

var timer_label: Label
var fps_label: Label

var _lobby_page: Control
var _setup_page: Control
var _match_page: Control
var _pause_overlay: Control
var _result_overlay: Control
var _score_box: VBoxContainer
var _result_score_box: VBoxContainer
var _pause_content: VBoxContainer
var _result_content: VBoxContainer
var _result_title: Label
var _result_detail: Label
var _stats_label: Label
var _hud_bubble_icon: ClayBubbleIcon
var _zoom_button: Button
var _selected_label: Label
var _map_option: OptionButton
var _ai_option: OptionButton
var _speed_spin: SpinBox
var _bubble_spin: SpinBox
var _power_spin: SpinBox
var _skin_buttons: Dictionary = {}
var _character_buttons: Dictionary = {}
var _selected_character_id: String = "builder"
var _selected_bubble_skin: String = "aqua"
var _last_scores: Array[Dictionary] = []


func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_PASS
	_build_lobby()
	_build_setup()
	_build_match_hud()
	_build_pause_overlay()
	_build_result_overlay()
	show_lobby()


func show_lobby() -> void:
	_lobby_page.visible = true
	_setup_page.visible = false
	_match_page.visible = false
	_pause_overlay.visible = false
	_result_overlay.visible = false


func show_setup(settings: MatchSettings) -> void:
	sync_settings(settings)
	_lobby_page.visible = false
	_setup_page.visible = true
	_match_page.visible = false
	_pause_overlay.visible = false
	_result_overlay.visible = false


func show_match() -> void:
	_lobby_page.visible = false
	_setup_page.visible = false
	_match_page.visible = true
	_pause_overlay.visible = false
	_result_overlay.visible = false


func sync_settings(settings: MatchSettings) -> void:
	_selected_character_id = settings.character_id
	_selected_bubble_skin = settings.bubble_skin
	if is_instance_valid(_hud_bubble_icon):
		_hud_bubble_icon.bubble_color = (
			Color("#59b9dd") if _selected_bubble_skin == "aqua" else Color("#df755e")
		)
	_map_option.select(1 if settings.map_id == MapCatalog.BELL_GARDEN else 0)
	_ai_option.select(settings.ai_count)
	_speed_spin.value = settings.max_speed
	_bubble_spin.value = settings.max_bubbles
	_power_spin.value = settings.max_power
	_refresh_character_selection()
	_refresh_skin_selection()


func update_timer(seconds_left: float) -> void:
	var total_seconds := maxi(0, ceili(seconds_left))
	timer_label.text = "%02d:%02d" % [total_seconds / 60, total_seconds % 60]


func update_fps(fps: int) -> void:
	fps_label.text = "FPS: %d" % fps


func update_zoom(percent: int) -> void:
	if is_instance_valid(_zoom_button):
		_zoom_button.text = "%d%%" % percent


func update_player_stats(actor: GameActor) -> void:
	if not is_instance_valid(actor):
		return
	var character := CharacterCatalog.get_definition(actor.character_id)
	_stats_label.text = "%s  %s　速度 %d　水泡 %d/%d　威力 %d" % [
		character.display_name,
		actor.actor_name,
		int(actor.stats.move_speed),
		actor.stats.active_bubbles,
		actor.stats.bubble_capacity,
		actor.stats.power,
	]


func update_scores(entries: Array[Dictionary]) -> void:
	_last_scores = []
	for entry in entries:
		_last_scores.append(entry.duplicate())
	_fill_score_box(_score_box, _last_scores, false)


func show_pause() -> void:
	_pause_overlay.visible = true


func hide_pause() -> void:
	_pause_overlay.visible = false


func show_result(title: String, detail: String) -> void:
	_result_title.text = title
	_result_detail.text = detail
	_fill_score_box(_result_score_box, _last_scores, true)
	_result_overlay.visible = true


func hide_result() -> void:
	_result_overlay.visible = false


func _process(_delta: float) -> void:
	if _match_page.visible:
		update_fps(Engine.get_frames_per_second())


func _build_lobby() -> void:
	_lobby_page = Control.new()
	_lobby_page.name = "LobbyPage"
	_lobby_page.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(_lobby_page)
	_add_background(_lobby_page, true)

	var layout := VBoxContainer.new()
	layout.set_anchors_preset(Control.PRESET_CENTER)
	layout.position = Vector2(-270, -180)
	layout.size = Vector2(540, 360)
	layout.alignment = BoxContainer.ALIGNMENT_CENTER
	layout.add_theme_constant_override("separation", 16)
	_lobby_page.add_child(layout)

	var seal := Label.new()
	seal.text = "轻量软陶 · 定格动画游乐园"
	seal.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	seal.add_theme_font_size_override("font_size", 20)
	seal.add_theme_color_override("font_color", Color("#8b352e"))
	layout.add_child(seal)
	var title := Label.new()
	title.text = "黏土泡泡大作战 3D"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 48)
	title.add_theme_color_override("font_color", Color("#fff8e8"))
	title.add_theme_constant_override("outline_size", 8)
	title.add_theme_color_override("font_outline_color", Color("#b24d42"))
	layout.add_child(title)
	var subtitle := Label.new()
	subtitle.text = "手捏小人、温暖庭院，熟悉的规则一点没变"
	subtitle.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	subtitle.add_theme_font_size_override("font_size", 18)
	subtitle.add_theme_color_override("font_color", Color("#35585b"))
	layout.add_child(subtitle)

	var start := _make_button("开始游戏", Color("#ef6b5b"), Vector2(280, 54))
	start.pressed.connect(setup_requested.emit)
	layout.add_child(start)
	var quit := _make_button("退出", Color("#4d9d9a"), Vector2(200, 44))
	quit.pressed.connect(quit_requested.emit)
	if not OS.has_feature("web"):
		layout.add_child(quit)


func _build_setup() -> void:
	_setup_page = Control.new()
	_setup_page.name = "SetupPage"
	_setup_page.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(_setup_page)
	_add_background(_setup_page, false)

	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	margin.add_theme_constant_override("margin_left", 24)
	margin.add_theme_constant_override("margin_right", 24)
	margin.add_theme_constant_override("margin_top", 18)
	margin.add_theme_constant_override("margin_bottom", 18)
	_setup_page.add_child(margin)

	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", _panel_style(Color(0.99, 0.94, 0.82, 0.96), Color("#e36b5e"), 20, 2))
	margin.add_child(panel)
	var page := VBoxContainer.new()
	page.add_theme_constant_override("separation", 10)
	panel.add_child(page)

	var header := HBoxContainer.new()
	page.add_child(header)
	var title := _make_label("选择你的黏土伙伴", 28, ClayMaterialLibrary.CHARCOAL)
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header.add_child(title)
	_selected_label = _make_label("", 18, Color("#317c78"))
	_selected_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	header.add_child(_selected_label)

	var body := HBoxContainer.new()
	body.size_flags_vertical = Control.SIZE_EXPAND_FILL
	body.add_theme_constant_override("separation", 16)
	page.add_child(body)

	var character_scroll := ScrollContainer.new()
	character_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	character_scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	character_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	body.add_child(character_scroll)
	var character_grid := GridContainer.new()
	character_grid.columns = 4
	character_grid.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	character_grid.add_theme_constant_override("h_separation", 10)
	character_grid.add_theme_constant_override("v_separation", 10)
	character_scroll.add_child(character_grid)
	for definition in CharacterCatalog.get_all():
		character_grid.add_child(_make_character_card(definition))

	var settings_panel := PanelContainer.new()
	settings_panel.custom_minimum_size.x = 280
	settings_panel.add_theme_stylebox_override("panel", _panel_style(Color("#fffaf0"), Color("#efba68"), 16, 1))
	body.add_child(settings_panel)
	var settings_box := VBoxContainer.new()
	settings_box.add_theme_constant_override("separation", 6)
	settings_panel.add_child(settings_box)
	settings_box.add_child(_make_label("对局设置", 20, Color("#7d3e36")))
	_map_option = _add_option(settings_box, "地图", ["软陶海岛集市", "钟楼花园"])
	_ai_option = _add_option(settings_box, "AI 数量", ["0", "1", "2", "3", "4"])
	_speed_spin = _add_spin(settings_box, "速度上限", 150, 1000, 25)
	_bubble_spin = _add_spin(settings_box, "水泡上限", 2, 20, 1)
	_power_spin = _add_spin(settings_box, "威力上限", 2, 20, 1)
	settings_box.add_child(_make_label("水泡皮肤", 13, Color("#775249")))
	var skin_row := HBoxContainer.new()
	skin_row.add_theme_constant_override("separation", 8)
	settings_box.add_child(skin_row)
	skin_row.add_child(_make_skin_card("aqua", "海盐蓝", Color("#59b9dd")))
	skin_row.add_child(_make_skin_card("coral", "珊瑚橙", Color("#df755e")))
	var hint := _make_label("角色只改变外观，不影响属性与技能。动画以 12 FPS 呈现定格质感。", 12, Color("#6e756f"))
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	settings_box.add_child(hint)

	var footer := HBoxContainer.new()
	footer.alignment = BoxContainer.ALIGNMENT_END
	footer.add_theme_constant_override("separation", 12)
	page.add_child(footer)
	var back := _make_button("返回大厅", Color("#5a9b98"), Vector2(150, 44))
	back.pressed.connect(lobby_requested.emit)
	footer.add_child(back)
	var start := _make_button("进入比赛", Color("#ef6b5b"), Vector2(190, 48))
	start.pressed.connect(_emit_match_request)
	footer.add_child(start)


func _build_match_hud() -> void:
	_match_page = Control.new()
	_match_page.name = "MatchHUD"
	_match_page.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_match_page.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_match_page)

	var top := PanelContainer.new()
	top.set_anchors_preset(Control.PRESET_TOP_WIDE)
	top.offset_left = 18
	top.offset_top = 10
	top.offset_right = -18
	top.offset_bottom = 68
	top.add_theme_stylebox_override("panel", _panel_style(Color(0.957, 0.906, 0.82, 0.94), ClayMaterialLibrary.TERRACOTTA, 18, 2))
	_match_page.add_child(top)
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 12)
	top.add_child(row)
	_hud_bubble_icon = ClayBubbleIcon.new()
	_hud_bubble_icon.name = "HudBubbleIcon"
	_hud_bubble_icon.bubble_color = Color("#59b9dd")
	_hud_bubble_icon.custom_minimum_size = Vector2(34, 34)
	row.add_child(_hud_bubble_icon)
	_stats_label = _make_label("准备中", 15, ClayMaterialLibrary.CHARCOAL)
	_stats_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(_stats_label)
	timer_label = _make_label("05:00", 28, Color("#9c4436"))
	timer_label.custom_minimum_size.x = 118
	timer_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	row.add_child(timer_label)
	fps_label = _make_label("FPS: 0", 13, Color("#4c7e68"))
	fps_label.custom_minimum_size.x = 90
	fps_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	row.add_child(fps_label)

	var scores_panel := PanelContainer.new()
	scores_panel.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	scores_panel.position = Vector2(-230, 82)
	scores_panel.size = Vector2(212, 170)
	scores_panel.add_theme_stylebox_override("panel", _panel_style(Color(0.957, 0.906, 0.82, 0.92), ClayMaterialLibrary.GRASS, 16, 2))
	_match_page.add_child(scores_panel)
	var scores_content := VBoxContainer.new()
	scores_panel.add_child(scores_content)
	scores_content.add_child(_make_label("即时排名", 17, Color("#9c4436")))
	_score_box = VBoxContainer.new()
	_score_box.add_theme_constant_override("separation", 3)
	scores_content.add_child(_score_box)

	var zoom_panel := PanelContainer.new()
	zoom_panel.name = "MapZoomControls"
	zoom_panel.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	zoom_panel.position = Vector2(-226, -82)
	zoom_panel.size = Vector2(208, 48)
	zoom_panel.mouse_filter = Control.MOUSE_FILTER_STOP
	zoom_panel.add_theme_stylebox_override(
		"panel",
		_panel_style(
			Color(0.957, 0.906, 0.82, 0.94),
			ClayMaterialLibrary.SKY.darkened(0.28),
			15,
			2
		)
	)
	_match_page.add_child(zoom_panel)
	var zoom_row := HBoxContainer.new()
	zoom_row.add_theme_constant_override("separation", 6)
	zoom_panel.add_child(zoom_row)
	var zoom_out_button := _make_button("-", Color("#5a9b98"), Vector2(42, 32))
	zoom_out_button.name = "ZoomOutButton"
	zoom_out_button.tooltip_text = "缩小地图（鼠标滚轮向下 / -）"
	zoom_out_button.pressed.connect(zoom_out_requested.emit)
	zoom_row.add_child(zoom_out_button)
	_zoom_button = _make_button("110%", Color("#e4a84e"), Vector2(78, 32))
	_zoom_button.name = "ZoomResetButton"
	_zoom_button.tooltip_text = "恢复推荐缩放（0）"
	_zoom_button.pressed.connect(zoom_reset_requested.emit)
	zoom_row.add_child(_zoom_button)
	var zoom_in_button := _make_button("+", Color("#ef6b5b"), Vector2(42, 32))
	zoom_in_button.name = "ZoomInButton"
	zoom_in_button.tooltip_text = "放大地图（鼠标滚轮向上 / +）"
	zoom_in_button.pressed.connect(zoom_in_requested.emit)
	zoom_row.add_child(zoom_in_button)

	var controls := _make_label(
		"WASD/方向键 移动　空格 放泡　1 自救　滚轮 +/- 缩放　0 复位　Esc 暂停",
		12,
		Color("#fff8e7")
	)
	controls.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	controls.offset_left = 24
	controls.offset_right = -24
	controls.offset_top = -32
	controls.offset_bottom = -8
	controls.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	controls.add_theme_constant_override("outline_size", 4)
	controls.add_theme_color_override("font_outline_color", Color("#31535a"))
	_match_page.add_child(controls)


func _build_pause_overlay() -> void:
	_pause_overlay = _make_overlay("游戏暂停")
	var detail := _make_label("游乐园休息一下，Esc 也可继续", 15, Color("#4b6767"))
	detail.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_pause_content.add_child(detail)
	var resume := _make_button("继续游戏", Color("#ef6b5b"), Vector2(230, 48))
	resume.pressed.connect(resume_requested.emit)
	_pause_content.add_child(resume)
	var restart := _make_button("重新开始", Color("#e6a94c"), Vector2(230, 44))
	restart.pressed.connect(restart_requested.emit)
	_pause_content.add_child(restart)
	var lobby := _make_button("返回大厅", Color("#4d9d9a"), Vector2(230, 44))
	lobby.pressed.connect(lobby_requested.emit)
	_pause_content.add_child(lobby)
	_pause_overlay.visible = false


func _build_result_overlay() -> void:
	_result_overlay = _make_overlay("本局结束")
	var detail := _make_label("", 15, Color("#4b6767"))
	detail.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_result_detail = detail
	_result_content.add_child(detail)
	_result_score_box = VBoxContainer.new()
	_result_score_box.add_theme_constant_override("separation", 4)
	_result_content.add_child(_result_score_box)
	var buttons := HBoxContainer.new()
	buttons.alignment = BoxContainer.ALIGNMENT_CENTER
	buttons.add_theme_constant_override("separation", 10)
	_result_content.add_child(buttons)
	var again := _make_button("再来一局", Color("#ef6b5b"), Vector2(150, 46))
	again.pressed.connect(restart_requested.emit)
	buttons.add_child(again)
	var lobby := _make_button("返回大厅", Color("#4d9d9a"), Vector2(150, 46))
	lobby.pressed.connect(lobby_requested.emit)
	buttons.add_child(lobby)
	_result_overlay.visible = false


func _make_overlay(title_text: String) -> Control:
	var overlay := ColorRect.new()
	overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	overlay.color = Color(0.06, 0.12, 0.14, 0.72)
	overlay.mouse_filter = Control.MOUSE_FILTER_STOP
	overlay.process_mode = Node.PROCESS_MODE_ALWAYS
	add_child(overlay)
	var panel := PanelContainer.new()
	panel.set_anchors_preset(Control.PRESET_CENTER)
	panel.position = Vector2(-215, -205)
	panel.size = Vector2(430, 410)
	panel.add_theme_stylebox_override("panel", _panel_style(Color("#fff6df"), Color("#ef6b5b"), 24, 3))
	overlay.add_child(panel)
	var content := VBoxContainer.new()
	content.alignment = BoxContainer.ALIGNMENT_CENTER
	content.add_theme_constant_override("separation", 12)
	panel.add_child(content)
	var title := _make_label(title_text, 32, Color("#8b4037"))
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	content.add_child(title)
	if title_text == "游戏暂停":
		_pause_content = content
	else:
		_result_content = content
		_result_title = title
	return overlay


func _make_character_card(definition: CharacterDefinition) -> Button:
	var button := Button.new()
	button.name = "%sCard" % definition.id.capitalize()
	button.toggle_mode = true
	button.custom_minimum_size = Vector2(148, 216)
	button.tooltip_text = "%s · %s" % [definition.display_name, definition.id.capitalize()]
	button.pressed.connect(_select_character.bind(definition.id))
	_character_buttons[definition.id] = button

	var content := VBoxContainer.new()
	content.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT, Control.PRESET_MODE_MINSIZE, 5)
	content.mouse_filter = Control.MOUSE_FILTER_IGNORE
	button.add_child(content)
	var viewport_container := SubViewportContainer.new()
	viewport_container.custom_minimum_size = Vector2(138, 170)
	viewport_container.stretch = true
	viewport_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
	content.add_child(viewport_container)
	var preview := CharacterPreview3D.new()
	preview.setup(definition)
	viewport_container.add_child(preview)
	var label := _make_label("%s  %s" % [definition.display_name, definition.id.capitalize()], 14, Color("#694139"))
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	content.add_child(label)
	return button


func _select_character(character_id: String) -> void:
	_selected_character_id = character_id
	_refresh_character_selection()


func _refresh_character_selection() -> void:
	var definition := CharacterCatalog.get_definition(_selected_character_id)
	if is_instance_valid(_selected_label):
		_selected_label.text = "当前选择：%s · %s" % [definition.display_name, definition.id.capitalize()]
	for character_id in _character_buttons:
		var button := _character_buttons[character_id] as Button
		button.set_pressed_no_signal(character_id == _selected_character_id)
		var background := Color("#fff8e9").lerp(definition.theme_color, 0.34) \
			if character_id == _selected_character_id else Color("#fff8e9")
		button.add_theme_stylebox_override("normal", _panel_style(background, Color("#d99c66"), 14, 2))
		button.add_theme_stylebox_override(
			"pressed",
			_panel_style(Color("#fff8e9").lerp(definition.theme_color, 0.4), Color("#ef6b5b"), 14, 3)
		)


func _select_skin(skin_id: String) -> void:
	_selected_bubble_skin = skin_id
	_refresh_skin_selection()


func _refresh_skin_selection() -> void:
	for skin_id: String in _skin_buttons:
		var button := _skin_buttons[skin_id] as Button
		var selected := skin_id == _selected_bubble_skin
		button.set_pressed_no_signal(selected)
		var color := Color("#59b9dd") if skin_id == "aqua" else Color("#df755e")
		button.add_theme_stylebox_override(
			"normal",
			_panel_style(Color("#fff8e9").lerp(color, 0.18 if selected else 0.04), color, 12, 2)
		)
		button.add_theme_stylebox_override(
			"pressed",
			_panel_style(Color("#fff8e9").lerp(color, 0.28), color.darkened(0.1), 12, 3)
		)


func _emit_match_request() -> void:
	match_requested.emit({
		"character_id": _selected_character_id,
		"map_id": MapCatalog.BELL_GARDEN if _map_option.selected == 1 else MapCatalog.HARBOR_MARKET,
		"ai_count": _ai_option.selected,
		"max_speed": int(_speed_spin.value),
		"max_bubbles": int(_bubble_spin.value),
		"max_power": int(_power_spin.value),
		"bubble_skin": _selected_bubble_skin,
	})


func _fill_score_box(box: VBoxContainer, entries: Array[Dictionary], numbered: bool) -> void:
	for child in box.get_children():
		child.queue_free()
	for index in range(entries.size()):
		var entry := entries[index]
		var definition := CharacterCatalog.get_definition(str(entry.get("character_id", "builder")))
		var prefix := "%d. " % (index + 1) if numbered else ""
		var label := _make_label(
			"%s%s %s　击败 %d" % [
				prefix,
				definition.display_name,
				entry.get("name", "?"),
				entry.get("kills", 0),
			],
			14,
			ClayMaterialLibrary.CHARCOAL if not numbered else Color("#5e4540")
		)
		box.add_child(label)


func _add_background(parent: Control, use_art: bool) -> void:
	if use_art:
		var viewport_container := SubViewportContainer.new()
		viewport_container.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
		viewport_container.stretch = true
		viewport_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
		parent.add_child(viewport_container)
		var diorama := ClayLobbyDiorama3D.new()
		viewport_container.add_child(diorama)
	var tint := ColorRect.new()
	tint.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	tint.color = Color(0.96, 0.9, 0.78, 0.08) if use_art else ClayMaterialLibrary.SKY
	tint.mouse_filter = Control.MOUSE_FILTER_IGNORE
	parent.add_child(tint)


func _add_option(parent: VBoxContainer, label_text: String, options: Array[String]) -> OptionButton:
	parent.add_child(_make_label(label_text, 13, Color("#775249")))
	var option := OptionButton.new()
	option.custom_minimum_size.y = 32
	for text in options:
		option.add_item(text)
	parent.add_child(option)
	return option


func _make_skin_card(skin_id: String, label_text: String, color: Color) -> Button:
	var button := Button.new()
	button.toggle_mode = true
	button.custom_minimum_size = Vector2(112, 50)
	button.pressed.connect(_select_skin.bind(skin_id))
	_skin_buttons[skin_id] = button
	var row := HBoxContainer.new()
	row.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT, Control.PRESET_MODE_MINSIZE, 5)
	row.mouse_filter = Control.MOUSE_FILTER_IGNORE
	button.add_child(row)
	var icon := ClayBubbleIcon.new()
	icon.bubble_color = color
	icon.custom_minimum_size = Vector2(34, 34)
	row.add_child(icon)
	var label := _make_label(label_text, 13, ClayMaterialLibrary.CHARCOAL)
	label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	row.add_child(label)
	return button


func _add_spin(parent: VBoxContainer, label_text: String, minimum: float, maximum: float, step: float) -> SpinBox:
	parent.add_child(_make_label(label_text, 13, Color("#775249")))
	var spin := SpinBox.new()
	spin.min_value = minimum
	spin.max_value = maximum
	spin.step = step
	spin.custom_minimum_size.y = 32
	parent.add_child(spin)
	return spin


func _make_button(text: String, color: Color, minimum_size: Vector2) -> Button:
	var button := Button.new()
	button.text = text
	button.custom_minimum_size = minimum_size
	button.add_theme_font_size_override("font_size", 18)
	button.add_theme_color_override("font_color", Color.WHITE)
	button.add_theme_stylebox_override("normal", _panel_style(color, color.darkened(0.18), 16, 2))
	button.add_theme_stylebox_override("hover", _panel_style(color.lightened(0.12), color.darkened(0.12), 16, 2))
	button.add_theme_stylebox_override("pressed", _panel_style(color.darkened(0.1), color.darkened(0.22), 16, 2))
	return button


func _make_label(text: String, font_size: int, color: Color) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	return label


func _panel_style(background: Color, border: Color, radius: int, border_width: int) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = background
	style.border_color = border
	style.set_border_width_all(border_width)
	style.set_corner_radius_all(radius)
	style.set_content_margin_all(10)
	style.shadow_color = Color(0.2, 0.13, 0.12, 0.2)
	style.shadow_size = 5
	style.shadow_offset = Vector2(0, 4)
	return style
