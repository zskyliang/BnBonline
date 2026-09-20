class_name GameModule
extends RefCounted


func happy_time() -> void:
	CrazyGamesBridge.happy_time()


func gameplay_start() -> void:
	CrazyGamesBridge.gameplay_start()


func gameplay_stop() -> void:
	CrazyGamesBridge.gameplay_stop()


func request_invite_url(params: Dictionary) -> String:
	return CrazyGamesBridge.request_invite_url(params)


func get_invite_link_param(param_name: String) -> String:
	return CrazyGamesBridge.get_invite_link_param(param_name)


func show_invite_button(params: Dictionary) -> void:
	CrazyGamesBridge.show_invite_button(params)


func hide_invite_button() -> void:
	CrazyGamesBridge.hide_invite_button()


func get_game_settings() -> Dictionary:
	return CrazyGamesBridge.get_game_settings()
