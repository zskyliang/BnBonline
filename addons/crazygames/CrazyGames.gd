extends Node

const VERSION: String = "1.0.2"
const GameModuleScript = preload("res://addons/crazygames/Modules/GameModule.gd")
const UserModuleScript = preload("res://addons/crazygames/Modules/UserModule.gd")
const DataModuleScript = preload("res://addons/crazygames/Modules/DataModule.gd")
const AdModuleScript = preload("res://addons/crazygames/Modules/AdModule.gd")

var godot_version: String = Engine.get_version_info().string
var is_initialised: bool = false
var has_adblock: bool = false

var Game = GameModuleScript.new()
var User = UserModuleScript.new()
var Data = DataModuleScript.new()
var Ad = AdModuleScript.new()


func _ready() -> void:
	if not OS.has_feature("web"):
		is_initialised = true
		return
	call_deferred("_init_sdk")


func _init_sdk() -> void:
	CrazyGamesBridge.callbacks.sdk_initialized.connect(_on_sdk_initialized)
	CrazyGamesBridge.init_sdk(VERSION)
	has_adblock = await CrazyGamesBridge.callbacks.adblock_detection_result


func _on_sdk_initialized() -> void:
	is_initialised = true


func is_initialised_async() -> void:
	while not is_initialised:
		await get_tree().process_frame
