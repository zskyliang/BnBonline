class_name Callbacks
extends RefCounted

signal sdk_initialized
signal adblock_detection_result(has_adblock: bool)
signal ad_status_change
signal ad_started
signal ad_finished
signal ad_error(error: Dictionary)
signal auth_listener_complete(user: Dictionary)
signal show_auth_prompt_complete(user: Dictionary)
signal show_account_link_prompt_complete(response: Dictionary)
signal get_user_complete(user: Dictionary)
signal get_user_token_complete(token: String)
signal get_xsolla_user_token_complete(token: String)

var onSdkInitialized: JavaScriptObject = JavaScriptBridge.create_callback(
	_on_sdk_initialized
)
var onAdblockDetectionResult: JavaScriptObject = JavaScriptBridge.create_callback(
	_on_adblock_detection_result
)
var onAuthListener: JavaScriptObject = JavaScriptBridge.create_callback(
	_on_auth_listener
)
var adStatusChange: JavaScriptObject = JavaScriptBridge.create_callback(
	_on_ad_status_change
)
var adStarted: JavaScriptObject = JavaScriptBridge.create_callback(_on_ad_started)
var adFinished: JavaScriptObject = JavaScriptBridge.create_callback(_on_ad_finished)
var adError: JavaScriptObject = JavaScriptBridge.create_callback(_on_ad_error)
var showAuthPrompt: JavaScriptObject = JavaScriptBridge.create_callback(
	_on_show_auth_prompt_callback
)
var showAccountLinkPrompt: JavaScriptObject = JavaScriptBridge.create_callback(
	_on_show_account_link_prompt_callback
)
var getUser: JavaScriptObject = JavaScriptBridge.create_callback(
	_on_get_user_callback
)
var getUserToken: JavaScriptObject = JavaScriptBridge.create_callback(
	_on_get_user_token_callback
)
var getXsollaUserToken: JavaScriptObject = JavaScriptBridge.create_callback(
	_on_get_xsolla_user_token_callback
)


func init() -> void:
	var window: JavaScriptObject = JavaScriptBridge.get_interface("window")
	window.onSdkInitialized = onSdkInitialized
	window.onAdblockDetectionResult = onAdblockDetectionResult
	window.onAuthListener = onAuthListener
	window.onAdStatusChange = adStatusChange
	window.onAdStarted = adStarted
	window.onAdFinished = adFinished
	window.onAdError = adError
	window.onShowAuthPrompt = showAuthPrompt
	window.onShowAccountLinkPrompt = showAccountLinkPrompt
	window.onGetUser = getUser
	window.onGetUserToken = getUserToken
	window.onGetXsollaUserToken = getXsollaUserToken


func _on_sdk_initialized(_args: Array) -> void:
	sdk_initialized.emit()


func _on_adblock_detection_result(args: Array) -> void:
	adblock_detection_result.emit(bool(args[0]))


func _on_ad_status_change(args: Array) -> void:
	var data: Variant = JSON.parse_string(str(args[0]))
	if data is not Dictionary or not (data as Dictionary).has("state"):
		ad_error.emit({"error": "invalid SDK ad status"})
		return
	var state := data as Dictionary
	ad_status_change.emit(state)
	match str(state["state"]):
		"started":
			ad_started.emit()
		"finished":
			ad_finished.emit()
		"error":
			ad_error.emit(state.get("error", {}) as Dictionary)


func _on_ad_started(_args: Array) -> void:
	ad_started.emit()


func _on_ad_finished(_args: Array) -> void:
	ad_finished.emit()


func _on_ad_error(args: Array) -> void:
	var error: Variant = JSON.parse_string(str(args[0]))
	ad_error.emit(error as Dictionary if error is Dictionary else {})


func _on_auth_listener(args: Array) -> void:
	var user: Variant = JSON.parse_string(str(args[0]))
	auth_listener_complete.emit(user as Dictionary if user is Dictionary else {})


func _on_show_auth_prompt_callback(args: Array) -> void:
	_emit_dictionary_result(args, "user", show_auth_prompt_complete)


func _on_show_account_link_prompt_callback(args: Array) -> void:
	_emit_dictionary_result(args, "response", show_account_link_prompt_complete)


func _on_get_user_callback(args: Array) -> void:
	_emit_dictionary_result(args, "user", get_user_complete)


func _on_get_user_token_callback(args: Array) -> void:
	_emit_string_result(args, "token", get_user_token_complete)


func _on_get_xsolla_user_token_callback(args: Array) -> void:
	_emit_string_result(args, "token", get_xsolla_user_token_complete)


func _emit_dictionary_result(
		args: Array,
		value_key: String,
		completed_signal: Signal
	) -> void:
	var data: Variant = JSON.parse_string(str(args[0]))
	if data is not Dictionary:
		completed_signal.emit({})
		return
	var result := data as Dictionary
	var value: Variant = result.get(value_key, result.get("error", {}))
	if value is String:
		var parsed: Variant = JSON.parse_string(value)
		value = parsed if parsed is Dictionary else {}
	completed_signal.emit(value as Dictionary if value is Dictionary else {})


func _emit_string_result(
		args: Array,
		value_key: String,
		completed_signal: Signal
	) -> void:
	var data: Variant = JSON.parse_string(str(args[0]))
	if data is Dictionary:
		completed_signal.emit(str((data as Dictionary).get(value_key, "")))
	else:
		completed_signal.emit("")
