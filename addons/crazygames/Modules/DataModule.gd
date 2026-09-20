class_name DataModule
extends RefCounted


func data_clear() -> void:
	CrazyGamesBridge.data_clear()


func data_get_item(key: String) -> String:
	return CrazyGamesBridge.data_get_item(key)


func data_has_key(key: String) -> bool:
	return CrazyGamesBridge.data_has_key(key)


func data_remove_item(key: String) -> void:
	CrazyGamesBridge.data_remove_item(key)


func data_set_item(key: String, value: String) -> void:
	CrazyGamesBridge.data_set_item(key, value)
