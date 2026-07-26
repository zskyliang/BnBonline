class_name GameHud
extends Control
## Responsive UI for paint-campaign setup, match HUD, pause, and progression.

signal setup_requested
signal match_requested(configuration: Dictionary)
signal resume_requested
signal restart_requested
signal retry_requested
signal skill_confirmed(skill_id: String)
signal lobby_requested
signal quit_requested
signal zoom_in_requested
signal zoom_out_requested
signal zoom_reset_requested
signal camera_reset_requested
signal camera_pose_requested(azimuth: float, elevation: float, zoom: float)
signal camera_adjustment_finished
signal settings_open_requested
signal settings_close_requested

var timer_label: Label
var fps_label: Label

var _lobby_page: Control
var _setup_page: Control
var _match_page: Control
var _pause_overlay: Control
var _result_overlay: Control
var _settings_overlay: Control
var _score_box: VBoxContainer
var _result_score_box: VBoxContainer
var _pause_content: VBoxContainer
var _result_content: VBoxContainer
var _result_title: Label
var _result_detail: Label
var _stats_label: Label
var _stage_label: Label
var _hud_bubble_icon: StorybookBubbleIcon
var _zoom_button: Button
var _item_timer_label: Label
var _item_bonus_label: Label
var _pickup_toast: Label
var _selected_label: Label
var _skill_box: VBoxContainer
var _next_stage_button: Button
var _retry_button: Button
var _character_buttons: Dictionary = {}
var _character_previews: Dictionary = {}
var _color_buttons: Dictionary = {}
var _skill_buttons: Dictionary = {}
var _selected_character_id: String = "cat"
var _selected_color_id: String = PaintPalette.DEFAULT_PLAYER_COLOR_ID
var _selected_skill_id: String = ""
var _last_scores: Array[Dictionary] = []
var _camera_zoom: float = MatchSettings.DEFAULT_CAMERA_ZOOM
var _pickup_toast_until_ms: int = 0


func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_PASS
	_build_lobby()
	_build_setup()
	_build_match_hud()
	_build_settings_overlay()
	_build_pause_overlay()
	_build_result_overlay()
	show_lobby()


func show_lobby() -> void:
	_lobby_page.visible = true
	_setup_page.visible = false
	_match_page.visible = false
	_pause_overlay.visible = false
	_result_overlay.visible = false
	_settings_overlay.visible = false


func show_setup(settings: MatchSettings) -> void:
	sync_settings(settings)
	_lobby_page.visible = false
	_setup_page.visible = true
	_match_page.visible = false
	_pause_overlay.visible = false
	_result_overlay.visible = false
	_settings_overlay.visible = false


func show_match() -> void:
	_lobby_page.visible = false
	_setup_page.visible = false
	_match_page.visible = true
	_pause_overlay.visible = false
	_result_overlay.visible = false
	_settings_overlay.visible = false


func sync_settings(settings: MatchSettings) -> void:
	_selected_character_id = settings.character_id
	_selected_color_id = settings.player_color_id
	if is_instance_valid(_hud_bubble_icon):
		_hud_bubble_icon.bubble_color = PaintPalette.get_color(_selected_color_id)
	update_camera_pose(
		settings.camera_azimuth,
		settings.camera_elevation,
		settings.camera_zoom
	)
	_refresh_character_selection()
	_refresh_color_selection()


func update_campaign(progress: RunProgress) -> void:
	if is_instance_valid(_stage_label):
		_stage_label.text = "第 %d 关 · %d 名 AI" % [
			progress.stage_number,
			progress.ai_count(),
		]
	if is_instance_valid(_hud_bubble_icon):
		_hud_bubble_icon.bubble_color = PaintPalette.get_color(progress.player_color_id)


func update_timer(seconds_left: float) -> void:
	var total_seconds: int = maxi(0, ceili(seconds_left))
	timer_label.text = "%02d:%02d" % [total_seconds / 60, total_seconds % 60]


func update_fps(fps: int) -> void:
	fps_label.text = "FPS: %d" % fps


func update_zoom(percent: int) -> void:
	_camera_zoom = float(percent) / 100.0
	if is_instance_valid(_zoom_button):
		_zoom_button.text = "%d%%" % percent


func update_camera_pose(_azimuth: float, _elevation: float, zoom: float) -> void:
	_camera_zoom = zoom
	update_zoom(roundi(zoom * 100.0))


func update_player_stats(actor: GameActor) -> void:
	if not is_instance_valid(actor):
		return
	var character := CharacterCatalog.get_definition(actor.character_id)
	_stats_label.text = "%s　速度 %d　水泡 %d/%d　威力 %d" % [
		character.display_name,
		int(actor.stats.move_speed),
		actor.stats.active_bubbles,
		actor.stats.bubble_capacity,
		actor.stats.power,
	]
	update_item_bonuses(actor)


func update_item_countdown(seconds: int) -> void:
	if not is_instance_valid(_item_timer_label):
		return
	_item_timer_label.text = (
		"下个道具 %02d 秒" % seconds
		if seconds >= 0
		else "本关道具已全部刷新"
	)


func update_item_bonuses(actor: GameActor) -> void:
	if not is_instance_valid(_item_bonus_label):
		return
	if not is_instance_valid(actor):
		_item_bonus_label.text = "临时：速度 +0　水泡 +0　威力 +0"
		return
	_item_bonus_label.text = "临时：速度 +%d　水泡 +%d　威力 +%d" % [
		actor.stats.stage_speed_items * int(GameConstants.SPEED_PER_STAGE_ITEM),
		actor.stats.stage_bubble_items,
		actor.stats.stage_power_items,
	]


func show_item_pickup(item_type: int) -> void:
	if not is_instance_valid(_pickup_toast):
		return
	_pickup_toast.text = "获得 %s！" % ArenaItemType.short_bonus(item_type)
	_pickup_toast.visible = true
	_pickup_toast_until_ms = Time.get_ticks_msec() + 1400


func update_scores(entries: Array[Dictionary]) -> void:
	_last_scores.clear()
	for entry: Dictionary in entries:
		_last_scores.append(entry.duplicate())
	_fill_score_box(_score_box, _last_scores)


func show_pause() -> void:
	_pause_overlay.visible = true


func hide_pause() -> void:
	_pause_overlay.visible = false


func show_result(title: String, detail: String) -> void:
	_result_title.text = title
	_result_detail.text = detail
	_fill_score_box(_result_score_box, _last_scores)
	_skill_box.visible = false
	_next_stage_button.visible = false
	_retry_button.visible = true
	_result_overlay.visible = true


func show_stage_result(
		won: bool,
		progress: RunProgress,
		player_cells: int,
		ai_cells: int,
		player_locked: int,
		ai_locked: int
	) -> void:
	_result_title.text = "第 %d 关胜利！" % progress.stage_number \
		if won else ("平局，重试本关" if player_cells == ai_cells else "AI 队领先")
	_result_detail.text = "玩家 %d 格（锁定 %d）　AI %d 格（锁定 %d）" % [
		player_cells,
		player_locked,
		ai_cells,
		ai_locked,
	]
	_fill_score_box(_result_score_box, _last_scores)
	_selected_skill_id = ""
	for button_value: Variant in _skill_buttons.values():
		(button_value as Button).set_pressed_no_signal(false)
	_skill_box.visible = won
	_next_stage_button.visible = won
	_next_stage_button.disabled = true
	_retry_button.visible = not won
	_result_overlay.visible = true


func hide_result() -> void:
	_result_overlay.visible = false


func show_settings() -> void:
	_settings_overlay.visible = true


func hide_settings() -> void:
	_settings_overlay.visible = false


func is_settings_visible() -> bool:
	return is_instance_valid(_settings_overlay) and _settings_overlay.visible


func _process(_delta: float) -> void:
	if _match_page.visible:
		update_fps(Engine.get_frames_per_second())
	if is_instance_valid(_pickup_toast) \
			and _pickup_toast.visible \
			and Time.get_ticks_msec() >= _pickup_toast_until_ms:
		_pickup_toast.visible = false


func _build_lobby() -> void:
	_lobby_page = Control.new()
	_lobby_page.name = "LobbyPage"
	_lobby_page.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(_lobby_page)
	_add_background(_lobby_page, true)
	var header_panel := PanelContainer.new()
	header_panel.set_anchors_preset(Control.PRESET_CENTER_TOP)
	header_panel.position = Vector2(-340, 18)
	header_panel.size = Vector2(680, 154)
	header_panel.add_theme_stylebox_override(
		"panel",
		_panel_style(Color(0.98, 0.93, 0.82, 0.91), Color("#8c6045"), 22, 2)
	)
	_lobby_page.add_child(header_panel)
	var header := VBoxContainer.new()
	header.alignment = BoxContainer.ALIGNMENT_CENTER
	header.add_theme_constant_override("separation", 2)
	header_panel.add_child(header)
	var seal := _make_label("森林绘本 · 无限闯关", 20, Color("#6b4938"))
	seal.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	header.add_child(seal)
	var title := _make_label("森林泡泡染色战", 46, Color("#7d3e36"))
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_constant_override("outline_size", 3)
	title.add_theme_color_override("font_outline_color", Color("#fff8e8"))
	header.add_child(title)
	var subtitle := _make_label("三分钟抢占地板，撞破敌方困泡永久锁定九宫格", 18, Color("#35585b"))
	subtitle.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	header.add_child(subtitle)

	var actions := VBoxContainer.new()
	actions.set_anchors_preset(Control.PRESET_CENTER_BOTTOM)
	actions.position = Vector2(-300, -142)
	actions.size = Vector2(600, 124)
	actions.alignment = BoxContainer.ALIGNMENT_END
	actions.add_theme_constant_override("separation", 10)
	_lobby_page.add_child(actions)
	var start := _make_button("开始闯关", Color("#ef6b5b"), Vector2(280, 56))
	start.pressed.connect(setup_requested.emit)
	actions.add_child(start)
	if not OS.has_feature("web"):
		var quit := _make_button("退出", Color("#4d9d9a"), Vector2(200, 44))
		quit.pressed.connect(quit_requested.emit)
		actions.add_child(quit)


func _build_setup() -> void:
	_setup_page = Control.new()
	_setup_page.name = "SetupPage"
	_setup_page.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(_setup_page)
	_add_background(_setup_page, false)
	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for side: String in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_%s" % side, 20)
	_setup_page.add_child(margin)
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override(
		"panel",
		_panel_style(Color(0.99, 0.94, 0.82, 0.97), Color("#e36b5e"), 20, 2)
	)
	margin.add_child(panel)
	var page := VBoxContainer.new()
	page.add_theme_constant_override("separation", 10)
	panel.add_child(page)
	var header := HBoxContainer.new()
	page.add_child(header)
	var title := _make_label("选择角色与阵营颜色", 28, StorybookMaterialLibrary.CHARCOAL)
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header.add_child(title)
	_selected_label = _make_label("", 17, Color("#317c78"))
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
	character_grid.add_theme_constant_override("h_separation", 9)
	character_grid.add_theme_constant_override("v_separation", 9)
	character_scroll.add_child(character_grid)
	for definition: CharacterDefinition in CharacterCatalog.get_all():
		character_grid.add_child(_make_character_card(definition))
	var color_panel := PanelContainer.new()
	color_panel.custom_minimum_size.x = 260
	color_panel.add_theme_stylebox_override(
		"panel",
		_panel_style(Color("#fffaf0"), Color("#efba68"), 16, 1)
	)
	body.add_child(color_panel)
	var color_box := VBoxContainer.new()
	color_box.add_theme_constant_override("separation", 8)
	color_panel.add_child(color_box)
	color_box.add_child(_make_label("局部队色与水泡颜色", 20, Color("#7d3e36")))
	var palette_grid := GridContainer.new()
	palette_grid.columns = 2
	palette_grid.add_theme_constant_override("h_separation", 8)
	palette_grid.add_theme_constant_override("v_separation", 8)
	color_box.add_child(palette_grid)
	for color_id: String in PaintPalette.COLOR_IDS:
		palette_grid.add_child(_make_color_button(color_id))
	var rule := _make_label(
		"每关 3 分钟。爆炸覆盖地板；只有接触撞破敌方困泡才会永久锁定九宫格。AI 数量随关卡增加，胜利后可选择一次属性强化。",
		13,
		Color("#6e5b50")
	)
	rule.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	color_box.add_child(rule)
	var footer := HBoxContainer.new()
	footer.alignment = BoxContainer.ALIGNMENT_END
	footer.add_theme_constant_override("separation", 12)
	page.add_child(footer)
	var back := _make_button("返回大厅", Color("#5a9b98"), Vector2(150, 44))
	back.pressed.connect(lobby_requested.emit)
	footer.add_child(back)
	var start := _make_button("进入第 1 关", Color("#ef6b5b"), Vector2(190, 48))
	start.pressed.connect(_emit_match_request)
	footer.add_child(start)


func _build_match_hud() -> void:
	_match_page = Control.new()
	_match_page.name = "MatchHUD"
	_match_page.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_match_page.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_match_page)
	var top := PanelContainer.new()
	top.name = "LiveStatusBar"
	top.set_anchors_preset(Control.PRESET_TOP_WIDE)
	top.offset_left = 18
	top.offset_top = 10
	top.offset_right = -18
	top.offset_bottom = 68
	top.add_theme_stylebox_override(
		"panel",
		_panel_style(Color(0.957, 0.906, 0.82, 0.78), StorybookMaterialLibrary.TERRACOTTA, 18, 2)
	)
	_match_page.add_child(top)
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 12)
	top.add_child(row)
	_hud_bubble_icon = StorybookBubbleIcon.new()
	_hud_bubble_icon.custom_minimum_size = Vector2(34, 34)
	row.add_child(_hud_bubble_icon)
	_stage_label = _make_label("第 1 关", 16, Color("#9c4436"))
	row.add_child(_stage_label)
	_stats_label = _make_label("准备中", 15, StorybookMaterialLibrary.CHARCOAL)
	_stats_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(_stats_label)
	timer_label = _make_label("03:00", 28, Color("#9c4436"))
	timer_label.custom_minimum_size.x = 108
	timer_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	row.add_child(timer_label)
	fps_label = _make_label("FPS: 0", 13, Color("#4c7e68"))
	fps_label.custom_minimum_size.x = 85
	fps_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	row.add_child(fps_label)
	var settings_button := _make_button("设置", Color("#5a9b98"), Vector2(72, 34))
	settings_button.name = "SettingsButton"
	settings_button.add_theme_font_size_override("font_size", 14)
	settings_button.pressed.connect(settings_open_requested.emit)
	row.add_child(settings_button)
	var scores_panel := PanelContainer.new()
	scores_panel.name = "LiveScoresPanel"
	scores_panel.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	scores_panel.position = Vector2(-242, 82)
	scores_panel.size = Vector2(224, 152)
	scores_panel.add_theme_stylebox_override(
		"panel",
		_panel_style(Color(0.957, 0.906, 0.82, 0.68), StorybookMaterialLibrary.GRASS, 16, 2)
	)
	_match_page.add_child(scores_panel)
	var scores_content := VBoxContainer.new()
	scores_panel.add_child(scores_content)
	scores_content.add_child(_make_label("实时占格", 17, Color("#9c4436")))
	_score_box = VBoxContainer.new()
	_score_box.add_theme_constant_override("separation", 5)
	scores_content.add_child(_score_box)
	var item_panel := PanelContainer.new()
	item_panel.name = "LiveItemPanel"
	item_panel.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	item_panel.position = Vector2(-242, 244)
	item_panel.size = Vector2(224, 86)
	item_panel.add_theme_stylebox_override(
		"panel",
		_panel_style(Color(0.957, 0.906, 0.82, 0.68), Color("#e0a344"), 14, 2)
	)
	_match_page.add_child(item_panel)
	var item_box := VBoxContainer.new()
	item_box.add_theme_constant_override("separation", 3)
	item_panel.add_child(item_box)
	_item_timer_label = _make_label("下个道具 10 秒", 15, Color("#8b4037"))
	item_box.add_child(_item_timer_label)
	_item_bonus_label = _make_label("临时：速度 +0　水泡 +0　威力 +0", 12, Color("#35585b"))
	_item_bonus_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	item_box.add_child(_item_bonus_label)

	_pickup_toast = _make_label("", 20, Color("#fff8e7"))
	_pickup_toast.set_anchors_preset(Control.PRESET_TOP_WIDE)
	_pickup_toast.offset_left = 300
	_pickup_toast.offset_right = -300
	_pickup_toast.offset_top = 78
	_pickup_toast.offset_bottom = 112
	_pickup_toast.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_pickup_toast.add_theme_constant_override("outline_size", 6)
	_pickup_toast.add_theme_color_override("font_outline_color", Color("#8b4037"))
	_pickup_toast.visible = false
	_match_page.add_child(_pickup_toast)
	var controls := _make_label(
		"WASD/方向键 移动　空格 放泡　滚轮或 +/- 缩放　0 恢复 110%　Esc 暂停",
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


func _build_settings_overlay() -> void:
	_settings_overlay = ColorRect.new()
	_settings_overlay.name = "SettingsModal"
	_settings_overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_settings_overlay.color = Color(0.04, 0.09, 0.11, 0.62)
	_settings_overlay.mouse_filter = Control.MOUSE_FILTER_STOP
	_settings_overlay.process_mode = Node.PROCESS_MODE_ALWAYS
	add_child(_settings_overlay)
	var panel := PanelContainer.new()
	panel.set_anchors_preset(Control.PRESET_CENTER)
	panel.position = Vector2(-270, -150)
	panel.size = Vector2(540, 300)
	panel.add_theme_stylebox_override(
		"panel",
		_panel_style(Color("#fff6df"), Color("#5a9b98"), 22, 3)
	)
	_settings_overlay.add_child(panel)
	var content := VBoxContainer.new()
	content.alignment = BoxContainer.ALIGNMENT_CENTER
	content.add_theme_constant_override("separation", 10)
	panel.add_child(content)
	var title := _make_label("游戏设置", 30, Color("#8b4037"))
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	content.add_child(title)
	var description := _make_label(
		"固定正交镜头：方位角 -30°、俯角 42°",
		14,
		Color("#4b6767")
	)
	description.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	content.add_child(description)
	var zoom_title := _make_label("镜头缩放", 14, Color("#5b5048"))
	zoom_title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	content.add_child(zoom_title)
	var zoom_row := HBoxContainer.new()
	zoom_row.alignment = BoxContainer.ALIGNMENT_CENTER
	zoom_row.add_theme_constant_override("separation", 8)
	content.add_child(zoom_row)
	var zoom_out := _make_button("-", Color("#5a9b98"), Vector2(48, 36))
	zoom_out.name = "ZoomOutButton"
	zoom_out.pressed.connect(zoom_out_requested.emit)
	zoom_row.add_child(zoom_out)
	_zoom_button = _make_button("110%", Color("#e4a84e"), Vector2(92, 36))
	_zoom_button.name = "ZoomResetButton"
	_zoom_button.pressed.connect(zoom_reset_requested.emit)
	zoom_row.add_child(_zoom_button)
	var zoom_in := _make_button("+", Color("#ef6b5b"), Vector2(48, 36))
	zoom_in.name = "ZoomInButton"
	zoom_in.pressed.connect(zoom_in_requested.emit)
	zoom_row.add_child(zoom_in)
	var hint := _make_label(
		"缩放范围 85%～135%；数字 0 恢复 110%",
		13,
		Color("#6e5b50")
	)
	hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	content.add_child(hint)
	var close := _make_button("保存并返回对局", Color("#ef6b5b"), Vector2(220, 44))
	close.name = "SettingsCloseButton"
	close.pressed.connect(settings_close_requested.emit)
	content.add_child(close)
	_settings_overlay.visible = false


func _build_pause_overlay() -> void:
	_pause_overlay = _make_overlay("游戏暂停")
	var detail := _make_label("当前关会保留成长与 AI 配置", 15, Color("#4b6767"))
	detail.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_pause_content.add_child(detail)
	var resume := _make_button("继续游戏", Color("#ef6b5b"), Vector2(230, 48))
	resume.pressed.connect(resume_requested.emit)
	_pause_content.add_child(resume)
	var restart := _make_button("重新开始本关", Color("#e6a94c"), Vector2(230, 44))
	restart.pressed.connect(restart_requested.emit)
	_pause_content.add_child(restart)
	var lobby := _make_button("结束闯关并返回大厅", Color("#4d9d9a"), Vector2(230, 44))
	lobby.pressed.connect(lobby_requested.emit)
	_pause_content.add_child(lobby)
	_pause_overlay.visible = false


func _build_result_overlay() -> void:
	_result_overlay = _make_overlay("本关结束")
	_result_detail = _make_label("", 15, Color("#4b6767"))
	_result_detail.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_result_content.add_child(_result_detail)
	_result_score_box = VBoxContainer.new()
	_result_score_box.add_theme_constant_override("separation", 5)
	_result_content.add_child(_result_score_box)
	_skill_box = VBoxContainer.new()
	_skill_box.add_theme_constant_override("separation", 6)
	_skill_box.add_child(_make_label("获得 1 技能点，选择下一关强化", 16, Color("#8b4037")))
	var skill_row := HBoxContainer.new()
	skill_row.alignment = BoxContainer.ALIGNMENT_CENTER
	skill_row.add_theme_constant_override("separation", 7)
	_skill_box.add_child(skill_row)
	skill_row.add_child(_make_skill_button(RunProgress.SKILL_SPEED, "速度 +10"))
	skill_row.add_child(_make_skill_button(RunProgress.SKILL_BUBBLE, "水泡 +1"))
	skill_row.add_child(_make_skill_button(RunProgress.SKILL_POWER, "威力 +1"))
	_result_content.add_child(_skill_box)
	var buttons := HBoxContainer.new()
	buttons.alignment = BoxContainer.ALIGNMENT_CENTER
	buttons.add_theme_constant_override("separation", 10)
	_result_content.add_child(buttons)
	_retry_button = _make_button("重试本关", Color("#ef6b5b"), Vector2(145, 44))
	_retry_button.pressed.connect(retry_requested.emit)
	buttons.add_child(_retry_button)
	_next_stage_button = _make_button("进入下一关", Color("#ef6b5b"), Vector2(155, 44))
	_next_stage_button.pressed.connect(_confirm_skill)
	buttons.add_child(_next_stage_button)
	var lobby := _make_button("返回大厅", Color("#4d9d9a"), Vector2(135, 44))
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
	panel.position = Vector2(-250, -220)
	panel.size = Vector2(500, 440)
	panel.add_theme_stylebox_override(
		"panel",
		_panel_style(Color("#fff6df"), Color("#ef6b5b"), 24, 3)
	)
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
	button.custom_minimum_size = Vector2(145, 210)
	button.pressed.connect(_select_character.bind(definition.id))
	_character_buttons[definition.id] = button
	var content := VBoxContainer.new()
	content.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT, Control.PRESET_MODE_MINSIZE, 5)
	content.mouse_filter = Control.MOUSE_FILTER_IGNORE
	button.add_child(content)
	var viewport_container := SubViewportContainer.new()
	viewport_container.custom_minimum_size = Vector2(135, 165)
	viewport_container.stretch = true
	viewport_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
	content.add_child(viewport_container)
	var preview := CharacterPreview3D.new()
	preview.setup(definition, _selected_color_id)
	_character_previews[definition.id] = preview
	viewport_container.add_child(preview)
	var label := _make_label(definition.display_name, 14, Color("#694139"))
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	content.add_child(label)
	return button


func _make_color_button(color_id: String) -> Button:
	var button := Button.new()
	button.toggle_mode = true
	button.text = PaintPalette.get_label(color_id)
	button.custom_minimum_size = Vector2(108, 48)
	button.add_theme_font_size_override("font_size", 18)
	button.pressed.connect(_select_color.bind(color_id))
	_color_buttons[color_id] = button
	return button


func _make_skill_button(skill_id: String, text: String) -> Button:
	var button := _make_button(text, Color("#d89a45"), Vector2(125, 42))
	button.toggle_mode = true
	button.pressed.connect(_select_skill.bind(skill_id))
	_skill_buttons[skill_id] = button
	return button


func _select_character(character_id: String) -> void:
	_selected_character_id = character_id
	_refresh_character_selection()


func _select_color(color_id: String) -> void:
	_selected_color_id = color_id
	for preview_value: Variant in _character_previews.values():
		(preview_value as CharacterPreview3D).set_color_id(color_id)
	_refresh_character_selection()
	_refresh_color_selection()


func _select_skill(skill_id: String) -> void:
	_selected_skill_id = skill_id
	for key: String in _skill_buttons:
		(_skill_buttons[key] as Button).set_pressed_no_signal(key == skill_id)
	_next_stage_button.disabled = false


func _confirm_skill() -> void:
	if _selected_skill_id.is_empty():
		return
	skill_confirmed.emit(_selected_skill_id)


func _refresh_character_selection() -> void:
	if not is_instance_valid(_selected_label):
		return
	var definition := CharacterCatalog.get_definition(_selected_character_id)
	_selected_label.text = "当前：%s · %s色" % [
		definition.display_name,
		PaintPalette.get_label(_selected_color_id),
	]
	for character_id: String in _character_buttons:
		var button := _character_buttons[character_id] as Button
		var selected: bool = character_id == _selected_character_id
		button.set_pressed_no_signal(selected)
		var tint: Color = PaintPalette.get_color(_selected_color_id)
		button.add_theme_stylebox_override(
			"normal",
			_panel_style(Color("#fff8e9").lerp(tint, 0.18 if selected else 0.02), Color("#d99c66"), 14, 2)
		)
		button.add_theme_stylebox_override(
			"pressed",
			_panel_style(Color("#fff8e9").lerp(tint, 0.28), tint.darkened(0.12), 14, 3)
		)


func _refresh_color_selection() -> void:
	for color_id: String in _color_buttons:
		var button := _color_buttons[color_id] as Button
		var selected: bool = color_id == _selected_color_id
		var color: Color = PaintPalette.get_color(color_id)
		button.set_pressed_no_signal(selected)
		button.add_theme_color_override("font_color", Color.WHITE)
		button.add_theme_stylebox_override(
			"normal",
			_panel_style(color.lightened(0.1 if selected else 0.0), color.darkened(0.18), 12, 3 if selected else 1)
		)
		button.add_theme_stylebox_override(
			"pressed",
			_panel_style(color.lightened(0.14), Color.WHITE, 12, 3)
		)


func _emit_match_request() -> void:
	match_requested.emit({
		"character_id": _selected_character_id,
		"player_color_id": _selected_color_id,
	})


func _fill_score_box(box: VBoxContainer, entries: Array[Dictionary]) -> void:
	for child: Node in box.get_children():
		child.queue_free()
	for entry: Dictionary in entries:
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		box.add_child(row)
		var swatch := ColorRect.new()
		swatch.custom_minimum_size = Vector2(22, 22)
		swatch.color = PaintPalette.get_color(str(entry.get("color_id", "red")))
		swatch.mouse_filter = Control.MOUSE_FILTER_IGNORE
		row.add_child(swatch)
		var label := _make_label(
			"%s　%d 格　锁定 %d" % [
				entry.get("name", "?"),
				entry.get("cells", 0),
				entry.get("locked", 0),
			],
			14,
			StorybookMaterialLibrary.CHARCOAL
		)
		row.add_child(label)


func _add_background(parent: Control, use_art: bool) -> void:
	if use_art:
		var viewport_container := SubViewportContainer.new()
		viewport_container.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
		viewport_container.stretch = true
		viewport_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
		parent.add_child(viewport_container)
		var diorama := StorybookLobbyDiorama3D.new()
		viewport_container.add_child(diorama)
	var tint := ColorRect.new()
	tint.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	tint.color = Color(0.96, 0.9, 0.78, 0.08) if use_art else StorybookMaterialLibrary.SKY
	tint.mouse_filter = Control.MOUSE_FILTER_IGNORE
	parent.add_child(tint)


func _make_button(text: String, color: Color, minimum_size: Vector2) -> Button:
	var button := Button.new()
	button.text = text
	button.custom_minimum_size = minimum_size
	button.add_theme_font_size_override("font_size", 17)
	button.add_theme_color_override("font_color", Color.WHITE)
	button.add_theme_stylebox_override("normal", _panel_style(color, color.darkened(0.18), 14, 2))
	button.add_theme_stylebox_override("hover", _panel_style(color.lightened(0.12), color.darkened(0.12), 14, 2))
	button.add_theme_stylebox_override("pressed", _panel_style(color.darkened(0.1), color.darkened(0.22), 14, 2))
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
	style.set_content_margin_all(9)
	style.shadow_color = Color(0.2, 0.13, 0.12, 0.18)
	style.shadow_size = 4
	style.shadow_offset = Vector2(0, 3)
	return style
