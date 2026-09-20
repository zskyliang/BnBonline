extends Node

const CallbacksScript = preload("res://addons/crazygames/CrazyGamesCallbacks.gd")

var callbacks = CallbacksScript.new()


func init_sdk(version: String) -> void:
	callbacks.init()
	var version_json := JSON.stringify(version)
	var js := """
		(() => {
			window.GodotSDK = {
				version: %s,
				pointerLockElement: undefined,
				unlockPointer() {
					this.pointerLockElement = document.pointerLockElement || null;
					if (this.pointerLockElement && document.exitPointerLock) {
						document.exitPointerLock();
					}
				},
				lockPointer() {
					if (this.pointerLockElement?.requestPointerLock) {
						this.pointerLockElement.requestPointerLock();
					}
				}
			};
			const initOptions = {
				wrapper: { engine: "godot", sdkVersion: %s }
			};
			const finishInit = async () => {
				try {
					await window.CrazyGames.SDK.init(initOptions);
					window.onSdkInitialized();
					window.CrazyGames.SDK.ad.hasAdblock()
						.then((result) => window.onAdblockDetectionResult(result))
						.catch(() => window.onAdblockDetectionResult(false));
					window.CrazyGames.SDK.user.addAuthListener((user) => {
						window.onAuthListener(JSON.stringify(user || {}));
					});
				} catch (error) {
					console.error("CrazyGames SDK initialization failed", error);
					window.onAdblockDetectionResult(false);
				}
			};
			if (window.CrazyGames?.SDK) {
				finishInit();
				return;
			}
			const script = document.createElement("script");
			script.src = "https://sdk.crazygames.com/crazygames-sdk-v3.js";
			script.addEventListener("load", finishInit, { once: true });
			script.addEventListener("error", () => {
				console.error("CrazyGames SDK script failed to load");
				window.onAdblockDetectionResult(false);
			}, { once: true });
			document.head.appendChild(script);
		})();
	""" % [version_json, version_json]
	JavaScriptBridge.eval(js, true)


func request_ad(ad_type: String) -> void:
	var js := """
		window.CrazyGames.SDK.ad.requestAd(%s, {
			adStarted: function () {
				window.GodotSDK.unlockPointer();
				window.onAdStarted();
				window.onAdStatusChange(JSON.stringify({"state": "started"}));
			},
			adFinished: function () {
				window.GodotSDK.lockPointer();
				window.onAdFinished();
				window.onAdStatusChange(JSON.stringify({"state": "finished"}));
			},
			adError: function (error) {
				window.onAdError(JSON.stringify(error || {}));
				window.onAdStatusChange(JSON.stringify({
					"state": "error",
					"error": error || {}
				}));
			}
		});
	""" % JSON.stringify(ad_type)
	JavaScriptBridge.eval(js, true)


func request_banners(banners: Array) -> void:
	JavaScriptBridge.eval(
		"window.CrazyGames.SDK.banner.requestOverlayBanners(%s);"
			% JSON.stringify(banners),
		true
	)


func happy_time() -> void:
	JavaScriptBridge.eval("window.CrazyGames.SDK.game.happytime();", true)


func gameplay_start() -> void:
	JavaScriptBridge.eval("window.CrazyGames.SDK.game.gameplayStart();", true)


func gameplay_stop() -> void:
	JavaScriptBridge.eval("window.CrazyGames.SDK.game.gameplayStop();", true)


func request_invite_url(params: Dictionary) -> String:
	var value: Variant = JavaScriptBridge.eval(
		"window.CrazyGames.SDK.game.inviteLink(%s);"
			% JSON.stringify(params),
		true
	)
	return str(value) if value != null else ""


func get_invite_link_param(param_name: String) -> String:
	var value: Variant = JavaScriptBridge.eval(
		"window.CrazyGames.SDK.game.getInviteParam(%s);"
			% JSON.stringify(param_name),
		true
	)
	return str(value) if value != null else ""


func show_invite_button(params: Dictionary) -> String:
	var value: Variant = JavaScriptBridge.eval(
		"window.CrazyGames.SDK.game.showInviteButton(%s);"
			% JSON.stringify(params),
		true
	)
	return str(value) if value != null else ""


func hide_invite_button() -> void:
	JavaScriptBridge.eval(
		"window.CrazyGames.SDK.game.hideInviteButton();",
		true
	)


func get_game_settings() -> Dictionary:
	var settings_json: Variant = JavaScriptBridge.eval(
		"JSON.stringify(window.CrazyGames.SDK.game.settings || {});",
		true
	)
	var result: Variant = JSON.parse_string(str(settings_json))
	return result as Dictionary if result is Dictionary else {}


func is_user_account_available() -> bool:
	return bool(JavaScriptBridge.eval(
		"window.CrazyGames.SDK.user.isUserAccountAvailable;",
		true
	))


func show_auth_prompt() -> void:
	JavaScriptBridge.eval("""
		window.CrazyGames.SDK.user.showAuthPrompt()
			.then((user) => window.onShowAuthPrompt(
				JSON.stringify({user: user || {}})
			))
			.catch((error) => window.onShowAuthPrompt(
				JSON.stringify({error: error || {}})
			));
	""", true)


func show_account_link_prompt() -> void:
	JavaScriptBridge.eval("""
		window.CrazyGames.SDK.user.showAccountLinkPrompt()
			.then((response) => window.onShowAccountLinkPrompt(
				JSON.stringify({response: response || {}})
			))
			.catch((error) => window.onShowAccountLinkPrompt(
				JSON.stringify({error: error || {}})
			));
	""", true)


func get_user() -> void:
	JavaScriptBridge.eval("""
		window.CrazyGames.SDK.user.getUser()
			.then((user) => window.onGetUser(
				JSON.stringify({user: user || {}})
			))
			.catch((error) => window.onGetUser(
				JSON.stringify({error: error || {}})
			));
	""", true)


func get_user_token() -> void:
	JavaScriptBridge.eval("""
		window.CrazyGames.SDK.user.getUserToken()
			.then((token) => window.onGetUserToken(
				JSON.stringify({token: token || ""})
			))
			.catch(() => window.onGetUserToken(JSON.stringify({token: ""})));
	""", true)


func get_xsolla_user_token() -> void:
	JavaScriptBridge.eval("""
		window.CrazyGames.SDK.user.getXsollaUserToken()
			.then((token) => window.onGetXsollaUserToken(
				JSON.stringify({token: token || ""})
			))
			.catch(() => window.onGetXsollaUserToken(
				JSON.stringify({token: ""})
			));
	""", true)


func data_clear() -> void:
	JavaScriptBridge.eval("window.CrazyGames.SDK.data.clear();", true)


func data_get_item(key: String) -> String:
	var value: Variant = JavaScriptBridge.eval(
		"window.CrazyGames.SDK.data.getItem(%s);"
			% JSON.stringify(key),
		true
	)
	return str(value) if value != null else ""


func data_has_key(key: String) -> bool:
	return bool(JavaScriptBridge.eval(
		"window.CrazyGames.SDK.data.getItem(%s) !== null;"
			% JSON.stringify(key),
		true
	))


func data_remove_item(key: String) -> void:
	JavaScriptBridge.eval(
		"window.CrazyGames.SDK.data.removeItem(%s);"
			% JSON.stringify(key),
		true
	)


func data_set_item(key: String, value: String) -> void:
	JavaScriptBridge.eval(
		"window.CrazyGames.SDK.data.setItem(%s, %s);"
			% [JSON.stringify(key), JSON.stringify(value)],
		true
	)


func analytics_track_order(provider: String, order: Dictionary) -> void:
	JavaScriptBridge.eval(
		"window.CrazyGames.SDK.analytics.trackOrder(%s, %s);"
			% [JSON.stringify(provider), JSON.stringify(order)],
		true
	)


func copy_to_clipboard(text: String) -> void:
	JavaScriptBridge.eval(
		"navigator.clipboard.writeText(%s);" % JSON.stringify(text),
		true
	)


func get_environment() -> String:
	var value: Variant = JavaScriptBridge.eval(
		"window.CrazyGames.SDK.environment;",
		true
	)
	return str(value) if value != null else "disabled"


func get_system_info() -> Dictionary:
	var info_json: Variant = JavaScriptBridge.eval(
		"JSON.stringify(window.CrazyGames.SDK.user.systemInfo || {});",
		true
	)
	var result: Variant = JSON.parse_string(str(info_json))
	return result as Dictionary if result is Dictionary else {}
