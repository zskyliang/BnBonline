class_name WebPlatformAdapter
extends Node

var _visibility_callback: JavaScriptObject
var _blur_callback: JavaScriptObject
var _focus_callback: JavaScriptObject
var _paused_by_visibility: bool = false
var _muted_by_visibility: bool = false
var _was_master_muted: bool = false


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


func _exit_tree() -> void:
	if not OS.has_feature("web") or _visibility_callback == null:
		return
	var document: JavaScriptObject = JavaScriptBridge.get_interface("document")
	if document != null:
		document.removeEventListener("visibilitychange", _visibility_callback)
	var window: JavaScriptObject = JavaScriptBridge.get_interface("window")
	if window != null:
		window.removeEventListener("blur", _blur_callback)
		window.removeEventListener("focus", _focus_callback)


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
			document.body.style.background = "#a9d8e8";
			canvas.style.width = "100vw";
			canvas.style.height = "100vh";
			canvas.style.display = "block";
			canvas.style.touchAction = "none";
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
	if not _muted_by_visibility:
		_was_master_muted = AudioServer.is_bus_mute(0)
		AudioServer.set_bus_mute(0, true)
		_muted_by_visibility = true
	var controller := get_parent() as MatchController
	if controller != null \
			and controller.app_state == MatchController.AppState.MATCH \
			and not controller.get("_is_paused"):
		_paused_by_visibility = true
		controller.call_deferred("_pause_match")


func _restore_after_background() -> void:
	if _muted_by_visibility:
		AudioServer.set_bus_mute(0, _was_master_muted)
		_muted_by_visibility = false
	if _paused_by_visibility:
		_paused_by_visibility = false
