class_name WebPlatformAdapter
extends Node

signal locale_detected(language_code: String)
signal sdk_ready(environment: String)

var _visibility_callback: JavaScriptObject
var _blur_callback: JavaScriptObject
var _focus_callback: JavaScriptObject
var _settings_callback: JavaScriptObject
var _paused_by_visibility: bool = false
var _visibility_mute_requested: bool = false
var _platform_mute_requested: bool = false
var _external_mute_applied: bool = false
var _master_mute_before_external: bool = false
var _sdk_initialized: bool = false
var _sdk_available: bool = false
var _desired_gameplay_active: bool = false
var _reported_gameplay_active: bool = false
var _sdk_environment: String = "disabled"
var _crazygames_sdk: Node
var _crazygames_bridge: Node
var _crazygames_game: RefCounted


func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	if not OS.has_feature("web"):
		return
	_install_browser_input_guard()
	_configure_responsive_canvas()
	_visibility_callback = JavaScriptBridge.create_callback(_on_visibility_changed)
	_blur_callback = JavaScriptBridge.create_callback(_on_browser_blurred)
	_focus_callback = JavaScriptBridge.create_callback(_on_browser_focused)
	var document: JavaScriptObject = JavaScriptBridge.get_interface("document")
	if document != null:
		document.addEventListener("visibilitychange", _visibility_callback)
	var window: JavaScriptObject = JavaScriptBridge.get_interface("window")
	if window != null:
		window.addEventListener("blur", _blur_callback)
		window.addEventListener("focus", _focus_callback)
	call_deferred("_initialize_crazygames_sdk")


func _exit_tree() -> void:
	if not OS.has_feature("web"):
		return
	request_gameplay_stop()
	var document: JavaScriptObject = JavaScriptBridge.get_interface("document")
	if document != null and _visibility_callback != null:
		document.removeEventListener("visibilitychange", _visibility_callback)
	var window: JavaScriptObject = JavaScriptBridge.get_interface("window")
	if window != null:
		if _blur_callback != null:
			window.removeEventListener("blur", _blur_callback)
		if _focus_callback != null:
			window.removeEventListener("focus", _focus_callback)
	if _sdk_available and _settings_callback != null:
		JavaScriptBridge.eval("""
			if (window.__bnbCrazySettingsForwarder) {
				window.CrazyGames.SDK.game.removeSettingsChangeListener(
					window.__bnbCrazySettingsForwarder
				);
				delete window.__bnbCrazySettingsForwarder;
				delete window.__bnbCrazySettingsChanged;
			}
		""", true)
	_visibility_mute_requested = false
	_platform_mute_requested = false
	_refresh_external_mute()


func request_gameplay_start() -> void:
	_desired_gameplay_active = true
	_sync_gameplay_state()


func request_gameplay_stop() -> void:
	_desired_gameplay_active = false
	_sync_gameplay_state()


func update_game_state(state: Dictionary) -> void:
	if not OS.has_feature("web"):
		return
	# The controller already builds a fresh snapshot. A shallow copy is enough
	# here and avoids recursively duplicating nested state on the Web main thread.
	var enriched_state := state.duplicate(false)
	enriched_state["crazygames"] = {
		"initialized": _sdk_initialized,
		"environment": _sdk_environment,
		"gameplay_active": _reported_gameplay_active,
		"mute_audio": _platform_mute_requested,
	}
	JavaScriptBridge.eval(
		"window.__bnbGameState = %s;" % JSON.stringify(enriched_state),
		true
	)


func get_sdk_environment() -> String:
	return _sdk_environment


func _initialize_crazygames_sdk() -> void:
	_crazygames_sdk = get_node_or_null("/root/CrazyGames")
	_crazygames_bridge = get_node_or_null("/root/CrazyGamesBridge")
	if _crazygames_sdk == null or _crazygames_bridge == null:
		return
	while not bool(_crazygames_sdk.get("is_initialised")):
		await get_tree().process_frame
	_sdk_initialized = true
	_sdk_environment = str(_crazygames_bridge.call("get_environment"))
	_sdk_available = _sdk_environment in ["local", "crazygames"]
	JavaScriptBridge.eval(
		"window.__bnbCrazyGames = %s;" % JSON.stringify({
			"initialized": true,
			"environment": _sdk_environment,
		}),
		true
	)
	if not _sdk_available:
		sdk_ready.emit(_sdk_environment)
		return
	_crazygames_game = _crazygames_sdk.get("Game") as RefCounted
	_apply_platform_settings(
		_crazygames_game.call("get_game_settings") as Dictionary
	)
	_install_platform_settings_listener()
	var system_info := (
		_crazygames_bridge.call("get_system_info") as Dictionary
	)
	var locale := str(system_info.get("locale", "en"))
	locale_detected.emit("zh" if locale.to_lower().begins_with("zh") else "en")
	_sync_gameplay_state()
	sdk_ready.emit(_sdk_environment)


func _install_platform_settings_listener() -> void:
	_settings_callback = JavaScriptBridge.create_callback(
		_on_platform_settings_changed
	)
	var window: JavaScriptObject = JavaScriptBridge.get_interface("window")
	if window == null:
		return
	window.__bnbCrazySettingsChanged = _settings_callback
	JavaScriptBridge.eval("""
		window.__bnbCrazySettingsForwarder = (settings) => {
			window.__bnbCrazySettingsChanged(JSON.stringify(settings || {}));
		};
		window.CrazyGames.SDK.game.addSettingsChangeListener(
			window.__bnbCrazySettingsForwarder
		);
	""", true)


func _on_platform_settings_changed(arguments: Array) -> void:
	if arguments.is_empty():
		return
	var parsed: Variant = JSON.parse_string(str(arguments[0]))
	if parsed is Dictionary:
		_apply_platform_settings(parsed as Dictionary)


func _apply_platform_settings(game_settings: Dictionary) -> void:
	_platform_mute_requested = bool(game_settings.get("muteAudio", false))
	_refresh_external_mute()


func _sync_gameplay_state() -> void:
	if not _sdk_initialized or not _sdk_available:
		return
	if _desired_gameplay_active == _reported_gameplay_active:
		return
	_reported_gameplay_active = _desired_gameplay_active
	if _reported_gameplay_active:
		_crazygames_game.call("gameplay_start")
	else:
		_crazygames_game.call("gameplay_stop")


func _install_browser_input_guard() -> void:
	JavaScriptBridge.eval("""
		(() => {
			if (window.__bnbInputGuardInstalled) return;
			window.__bnbInputGuardInstalled = true;
			const blocked = new Set([
				"ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space",
				"Equal", "Minus", "NumpadAdd", "NumpadSubtract", "Numpad0"
			]);
			window.addEventListener("keydown", (event) => {
				if (!event.ctrlKey && !event.metaKey && blocked.has(event.code)) {
					event.preventDefault();
				}
			}, { passive: false });
			window.addEventListener("keyup", (event) => {
				if (!event.ctrlKey && !event.metaKey && blocked.has(event.code)) {
					event.preventDefault();
				}
			}, { passive: false });
			window.addEventListener("wheel", (event) => {
				if (event.target?.tagName === "CANVAS") event.preventDefault();
			}, { passive: false });
			window.addEventListener("contextmenu", (event) => {
				if (event.target?.tagName === "CANVAS") event.preventDefault();
			});
		})();
	""", true)


func _configure_responsive_canvas() -> void:
	JavaScriptBridge.eval("""
		(() => {
			const canvas = document.querySelector("canvas");
			if (!canvas) return;
			document.documentElement.style.width = "100%";
			document.documentElement.style.height = "100%";
			document.body.style.margin = "0";
			document.body.style.width = "100%";
			document.body.style.height = "100%";
			document.body.style.overflow = "hidden";
			document.body.style.overscrollBehavior = "none";
			document.body.style.webkitUserSelect = "none";
			document.body.style.userSelect = "none";
			document.body.style.webkitTouchCallout = "none";
			document.body.style.background = "#a9d8e8";
			canvas.style.width = "100vw";
			canvas.style.height = "100vh";
			canvas.style.display = "block";
			canvas.style.touchAction = "none";
			window.__bnbGameState = {
				mode: "loading",
				coordinate_system: "board cells; origin top-left; +x right; +y down"
			};
			window.render_game_to_text = () => JSON.stringify(
				window.__bnbGameState
			);
		})();
	""", true)


func _on_visibility_changed(_arguments: Array) -> void:
	var document: JavaScriptObject = JavaScriptBridge.get_interface("document")
	if document == null:
		return
	if bool(document.hidden):
		_pause_for_background()
	else:
		_restore_after_background()


func _on_browser_blurred(_arguments: Array) -> void:
	_pause_for_background()


func _on_browser_focused(_arguments: Array) -> void:
	_restore_after_background()


func _pause_for_background() -> void:
	_visibility_mute_requested = true
	_refresh_external_mute()
	var controller := get_parent() as MatchController
	if controller != null \
			and controller.app_state == MatchController.AppState.MATCH \
			and not controller.get("_is_paused"):
		_paused_by_visibility = true
		controller.call_deferred("_pause_match", false)


func _restore_after_background() -> void:
	_visibility_mute_requested = false
	_refresh_external_mute()
	if _paused_by_visibility:
		_paused_by_visibility = false


func _refresh_external_mute() -> void:
	var should_mute := _visibility_mute_requested or _platform_mute_requested
	if should_mute and not _external_mute_applied:
		_master_mute_before_external = AudioServer.is_bus_mute(0)
		_external_mute_applied = true
	if should_mute:
		AudioServer.set_bus_mute(0, true)
	elif _external_mute_applied:
		AudioServer.set_bus_mute(0, _master_mute_before_external)
		_external_mute_applied = false
