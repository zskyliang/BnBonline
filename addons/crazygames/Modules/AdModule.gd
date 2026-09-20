class_name AdModule
extends RefCounted

var _registered_banners: Array[Control] = []


func request_ad_async(ad_type: String) -> Dictionary:
	CrazyGamesBridge.request_ad(ad_type)
	var state: Dictionary = await CrazyGamesBridge.callbacks.ad_status_change
	if state.get("state", "") == "started":
		state = await CrazyGamesBridge.callbacks.ad_status_change
	return state


func request_banners(banners: Array) -> void:
	CrazyGamesBridge.request_banners(banners)


func register_banner(banner: Control) -> void:
	if not _registered_banners.has(banner):
		_registered_banners.append(banner)


func unregister_banner(banner: Control) -> void:
	_registered_banners.erase(banner)


func refresh_banners() -> void:
	var visible_banners_data: Array = []
	for banner: Control in _registered_banners:
		if banner.is_visible_in_tree():
			banner.regenerate_id()
			visible_banners_data.append(banner.get_data_for_sdk())
	if not visible_banners_data.is_empty():
		request_banners(visible_banners_data)
