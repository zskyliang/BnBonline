class_name UserModule
extends RefCounted


func is_user_account_available() -> bool:
	return CrazyGamesBridge.is_user_account_available()


func show_auth_prompt_async() -> Dictionary:
	CrazyGamesBridge.show_auth_prompt()
	return await CrazyGamesBridge.callbacks.show_auth_prompt_complete


func show_account_link_prompt_async() -> Dictionary:
	CrazyGamesBridge.show_account_link_prompt()
	return await CrazyGamesBridge.callbacks.show_account_link_prompt_complete


func get_user_async() -> Dictionary:
	CrazyGamesBridge.get_user()
	return await CrazyGamesBridge.callbacks.get_user_complete


func get_user_token_async() -> String:
	CrazyGamesBridge.get_user_token()
	return await CrazyGamesBridge.callbacks.get_user_token_complete


func get_xsolla_user_token_async() -> String:
	CrazyGamesBridge.get_xsolla_user_token()
	return await CrazyGamesBridge.callbacks.get_xsolla_user_token_complete
