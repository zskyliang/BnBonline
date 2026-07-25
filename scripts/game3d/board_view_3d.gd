class_name BoardView3D
extends Node3D
## Static clay-island world plus batched mutable obstacles and pickups.

const WATER_SHADER := preload("res://assets/materials/clay_water.gdshader")

var board: GameBoard
var _static_root: Node3D
var _dynamic_root: Node3D
var _dynamic_rebuild_scheduled := false
var _module_mesh_cache: Dictionary = {}


func bind_board(logic_board: GameBoard) -> void:
	if is_instance_valid(board):
		if board.cell_changed.is_connected(_on_cell_changed):
			board.cell_changed.disconnect(_on_cell_changed)
		if board.board_reset.is_connected(_on_board_reset):
			board.board_reset.disconnect(_on_board_reset)
	board = logic_board
	if not is_instance_valid(board):
		return
	board.cell_changed.connect(_on_cell_changed)
	board.board_reset.connect(_on_board_reset)
	rebuild()


func rebuild() -> void:
	_dynamic_rebuild_scheduled = false
	if is_instance_valid(_static_root):
		_static_root.queue_free()
	if is_instance_valid(_dynamic_root):
		_dynamic_root.queue_free()
	_static_root = Node3D.new()
	_static_root.name = "ClayIslandStatic"
	add_child(_static_root)
	_dynamic_root = Node3D.new()
	_dynamic_root.name = "ClayIslandDynamic"
	add_child(_dynamic_root)
	if not is_instance_valid(board) or board.cells.is_empty() or board.map_data == null:
		return
	_build_water_and_island()
	_build_ground()
	_build_buildings()
	_build_shoreline()
	_build_decorations()
	_build_nature_ring()
	_rebuild_dynamic()


func get_building_views() -> Array[ClayBuildingView3D]:
	var result: Array[ClayBuildingView3D] = []
	if not is_instance_valid(_static_root):
		return result
	for node: Node in _static_root.get_children():
		if node is ClayBuildingView3D:
			result.append(node as ClayBuildingView3D)
	return result


func _on_board_reset() -> void:
	rebuild()


func _on_cell_changed(_cell: Vector2i, _new_code: int) -> void:
	if _dynamic_rebuild_scheduled:
		return
	_dynamic_rebuild_scheduled = true
	call_deferred("_rebuild_dynamic")


func _rebuild_dynamic() -> void:
	_dynamic_rebuild_scheduled = false
	if not is_instance_valid(board):
		return
	if is_instance_valid(_dynamic_root):
		_dynamic_root.queue_free()
	_dynamic_root = Node3D.new()
	_dynamic_root.name = "ClayIslandDynamic"
	add_child(_dynamic_root)
	var stalls: Array[Vector2i] = []
	var carts: Array[Vector2i] = []
	var crates: Array[Vector2i] = []
	var bubble_items: Array[Vector2i] = []
	var speed_items: Array[Vector2i] = []
	var power_items: Array[Vector2i] = []
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			var cell := Vector2i(x, y)
			match board.cells[y][x]:
				3:
					match _cell_seed(cell, board.map_data.theme_id) % 3:
						0:
							stalls.append(cell)
						1:
							carts.append(cell)
						_:
							crates.append(cell)
				GameConstants.ITEM_BUBBLE:
					bubble_items.append(cell)
				GameConstants.ITEM_SPEED:
					speed_items.append(cell)
				GameConstants.ITEM_POWER:
					power_items.append(cell)
	_make_imported_batch(
		"MarketStalls",
		stalls,
		BuildingCatalog.KENNEY_ROOT + (
			"stall-red.glb" if board.map_data.theme_id == "harbor" else "stall-green.glb"
		),
		ClayMaterialLibrary.TERRACOTTA,
		Vector3.ONE * 0.7,
		0.0
	)
	_make_imported_batch(
		"FlowerCarts",
		carts,
		BuildingCatalog.KENNEY_ROOT + "cart.glb",
		ClayMaterialLibrary.MUSTARD,
		Vector3.ONE * 0.62,
		0.02
	)
	_make_crate_batch(crates)
	_make_item_batch("BubbleItems", bubble_items, ClayMaterialLibrary.SKY.darkened(0.12), 0)
	_make_item_batch("SpeedItems", speed_items, ClayMaterialLibrary.GRASS.lightened(0.12), 1)
	_make_item_batch("PowerItems", power_items, ClayMaterialLibrary.TERRACOTTA.lightened(0.08), 2)


func _build_water_and_island() -> void:
	var water := MeshInstance3D.new()
	water.name = "OpaqueClayWater"
	var water_mesh := PlaneMesh.new()
	water_mesh.size = Vector2(100.0, 90.0)
	water_mesh.subdivide_width = 24
	water_mesh.subdivide_depth = 20
	water.mesh = water_mesh
	water.position.y = -0.72
	var water_material := ShaderMaterial.new()
	water_material.shader = WATER_SHADER
	water_material.set_shader_parameter(
		"detail_normal",
		load("res://assets/materials/clay_detail_normal.png") as Texture2D
	)
	water.material_override = water_material
	water.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_static_root.add_child(water)

	var clay_body := MeshInstance3D.new()
	clay_body.name = "IslandClayBody"
	var body_mesh := BoxMesh.new()
	body_mesh.size = Vector3(17.2, 0.82, 15.2)
	clay_body.mesh = body_mesh
	clay_body.position.y = -0.43
	clay_body.material_override = ClayMaterialLibrary.make(Color("#b97453"), 0.94)
	_static_root.add_child(clay_body)

	var cream_shore := MeshInstance3D.new()
	cream_shore.name = "CreamShore"
	var shore_mesh := BoxMesh.new()
	shore_mesh.size = Vector3(16.55, 0.16, 14.55)
	cream_shore.mesh = shore_mesh
	cream_shore.position.y = -0.075
	cream_shore.material_override = ClayMaterialLibrary.make(Color("#e7cc9f"), 0.94)
	_static_root.add_child(cream_shore)

	var grass_top := MeshInstance3D.new()
	grass_top.name = "GrassIslandTop"
	var grass_mesh := BoxMesh.new()
	grass_mesh.size = Vector3(15.75, 0.12, 13.75)
	grass_top.mesh = grass_mesh
	grass_top.position.y = 0.015
	grass_top.material_override = ClayMaterialLibrary.make(
		Color("#77ad68") if board.map_data.theme_id == "harbor" else Color("#83b778"),
		0.94
	)
	_static_root.add_child(grass_top)


func _build_ground() -> void:
	var path_tiles: Array[Vector2i] = []
	var accent_tiles: Array[Vector2i] = []
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			if board.map_data.barrier_cells[y][x] == 9:
				continue
			var cell := Vector2i(x, y)
			if board.map_data.ground_cells[y][x] == 2:
				accent_tiles.append(cell)
			else:
				path_tiles.append(cell)
	var primary := Color("#cda97f") if board.map_data.theme_id == "harbor" else Color("#cfc092")
	_make_tile_batch("ClayPath", path_tiles, primary, 0.095)
	_make_tile_batch("ClayPathAccent", accent_tiles, primary.darkened(0.08), 0.105)


func _build_buildings() -> void:
	for placement: BuildingPlacement in board.map_data.building_units:
		var building := ClayBuildingView3D.new()
		_static_root.add_child(building)
		building.configure(placement)
		if placement.occludable:
			building.add_to_group("clay_occludable_buildings")


func _build_shoreline() -> void:
	var shore_cells: Array[Vector2i] = []
	for y: int in range(GameConstants.GRID_ROWS):
		for x: int in range(GameConstants.GRID_COLUMNS):
			if board.map_data.barrier_cells[y][x] == 9:
				shore_cells.append(Vector2i(x, y))
	var mesh := BoxMesh.new()
	mesh.size = Vector3(0.98, 0.2, 0.98)
	_make_multimesh(
		"HandPressedShore",
		shore_cells,
		mesh,
		ClayMaterialLibrary.make(Color("#c89267"), 0.94),
		0.12,
		Vector3.ONE,
		_static_root
	)


func _build_decorations() -> void:
	for decoration: Dictionary in board.map_data.decorations:
		var cell := decoration.get("cell", Vector2i.ZERO) as Vector2i
		var rotation_quadrants := int(decoration.get("rotation", 0))
		match str(decoration.get("type", "")):
			"dock":
				_build_dock(cell, rotation_quadrants)
			"fountain":
				_build_fountain(cell)
			"garden_gate":
				_build_gate(cell)


func _build_dock(cell: Vector2i, rotation_quadrants: int) -> void:
	var root := Node3D.new()
	root.name = "ClayDock"
	root.position = GameConstants.grid_to_world_3d(cell, 0.18)
	root.rotation.y = float(rotation_quadrants) * PI * 0.5
	_static_root.add_child(root)
	for index: int in range(3):
		var module := BuildingCatalog.instantiate_module(
			BuildingCatalog.KENNEY_ROOT + "planks.glb"
		)
		if module == null:
			continue
		module.position.z = float(index - 1) * 0.72
		module.scale = Vector3(0.8, 0.8, 0.8)
		_set_model_material(module, ClayMaterialLibrary.make(Color("#9d664c"), 0.94))
		root.add_child(module)


func _build_fountain(cell: Vector2i) -> void:
	var module := BuildingCatalog.instantiate_module(
		BuildingCatalog.KENNEY_ROOT + "fountain-round.glb"
	)
	if module == null:
		return
	module.name = "GardenFountain"
	module.position = GameConstants.grid_to_world_3d(cell, 0.16)
	module.scale = Vector3.ONE * 0.72
	_set_model_material(module, ClayMaterialLibrary.make(ClayMaterialLibrary.SKY.darkened(0.1), 0.82))
	_static_root.add_child(module)


func _build_gate(cell: Vector2i) -> void:
	var module := BuildingCatalog.instantiate_module(
		BuildingCatalog.KENNEY_ROOT + "fence-gate.glb"
	)
	if module == null:
		return
	module.name = "GardenGate"
	module.position = GameConstants.grid_to_world_3d(cell, 0.16)
	module.scale = Vector3.ONE * 0.8
	_set_model_material(module, ClayMaterialLibrary.make(Color("#9d664c"), 0.94))
	_static_root.add_child(module)


func _build_nature_ring() -> void:
	var placements: Array[Dictionary] = [
		{"asset": "common_tree", "position": Vector3(-6.75, 0.04, -5.72), "scale": 0.24},
		{"asset": "twisted_tree", "position": Vector3(6.72, 0.04, -5.45), "scale": 0.18},
		{"asset": "common_tree", "position": Vector3(6.65, 0.04, 5.55), "scale": 0.21},
		{"asset": "bush_flowers", "position": Vector3(-6.68, 0.08, 4.82), "scale": 0.18},
		{"asset": "clover", "position": Vector3(0.2, 0.1, -5.82), "scale": 0.13},
		{"asset": "flower_group", "position": Vector3(-5.55, 0.12, 5.78), "scale": 0.16},
		{"asset": "round_rock", "position": Vector3(5.65, 0.02, 5.72), "scale": 0.23},
	]
	for item: Dictionary in placements:
		var path := BuildingCatalog.nature_path(str(item["asset"]))
		var module := BuildingCatalog.instantiate_module(path)
		if module == null:
			continue
		module.name = str(item["asset"]).capitalize()
		module.position = item["position"] as Vector3
		module.scale = Vector3.ONE * float(item["scale"])
		ClayMaterialLibrary.clayify_imported_model(module, ClayMaterialLibrary.GRASS)
		_static_root.add_child(module)


func _make_tile_batch(batch_name: String, cells: Array[Vector2i], color: Color, height: float) -> void:
	var mesh := BoxMesh.new()
	mesh.size = Vector3(0.975, 0.06, 0.975)
	_make_multimesh(
		batch_name,
		cells,
		mesh,
		ClayMaterialLibrary.make(color, 0.94),
		height,
		Vector3.ONE,
		_static_root
	)


func _make_crate_batch(cells: Array[Vector2i]) -> void:
	var mesh := BoxMesh.new()
	mesh.size = Vector3(0.72, 0.66, 0.72)
	_make_multimesh(
		"HandPressedCrateStacks",
		cells,
		mesh,
		ClayMaterialLibrary.make(Color("#a86545"), 0.93),
		0.34,
		Vector3.ONE,
		_dynamic_root
	)


func _make_imported_batch(
		batch_name: String,
		cells: Array[Vector2i],
		resource_path: String,
		color: Color,
		module_scale: Vector3,
		height: float
	) -> void:
	var mesh := _get_module_mesh(resource_path)
	if mesh == null:
		_make_crate_batch(cells)
		return
	_make_multimesh(
		batch_name,
		cells,
		mesh,
		ClayMaterialLibrary.make(color, 0.92),
		height,
		module_scale,
		_dynamic_root
	)


func _make_item_batch(batch_name: String, cells: Array[Vector2i], color: Color, shape_index: int) -> void:
	var mesh: PrimitiveMesh
	if shape_index == 1:
		var cylinder := CylinderMesh.new()
		cylinder.top_radius = 0.2
		cylinder.bottom_radius = 0.26
		cylinder.height = 0.38
		mesh = cylinder
	elif shape_index == 2:
		var prism := PrismMesh.new()
		prism.size = Vector3(0.42, 0.42, 0.42)
		mesh = prism
	else:
		var sphere := SphereMesh.new()
		sphere.radius = 0.24
		sphere.height = 0.48
		sphere.radial_segments = 12
		sphere.rings = 6
		mesh = sphere
	_make_multimesh(
		batch_name,
		cells,
		mesh,
		ClayMaterialLibrary.make(color, 0.88, 0.08),
		0.32,
		Vector3.ONE,
		_dynamic_root
	)


func _make_multimesh(
		batch_name: String,
		cells: Array[Vector2i],
		mesh: Mesh,
		material: Material,
		height: float,
		base_scale: Vector3,
		parent: Node3D
	) -> void:
	if cells.is_empty():
		return
	var multi_mesh := MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	multi_mesh.use_colors = true
	multi_mesh.mesh = mesh
	multi_mesh.instance_count = cells.size()
	for index: int in range(cells.size()):
		var cell := cells[index]
		var seed := _cell_seed(cell, batch_name)
		var yaw := float(seed % 4) * PI * 0.5 + float(seed % 11 - 5) * 0.008
		var scale_variation := 0.96 + float(seed % 9) * 0.01
		var basis := Basis(Vector3.UP, yaw).scaled(base_scale * scale_variation)
		multi_mesh.set_instance_transform(
			index,
			Transform3D(basis, GameConstants.grid_to_world_3d(cell, height))
		)
		var shade := 0.94 + float(seed % 7) * 0.01
		multi_mesh.set_instance_color(index, Color(shade, shade, shade, 1.0))
	var instance := MultiMeshInstance3D.new()
	instance.name = batch_name
	instance.multimesh = multi_mesh
	if material is BaseMaterial3D:
		var batch_material := material.duplicate() as BaseMaterial3D
		batch_material.vertex_color_use_as_albedo = true
		instance.material_override = batch_material
	else:
		instance.material_override = material
	instance.cast_shadow = (
		GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		if batch_name.begins_with("ClayPath")
		else GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	)
	parent.add_child(instance)


func _get_module_mesh(resource_path: String) -> Mesh:
	if _module_mesh_cache.has(resource_path):
		return _module_mesh_cache[resource_path] as Mesh
	var module := BuildingCatalog.instantiate_module(resource_path)
	if module == null:
		return null
	var mesh_instance := _find_first_mesh(module)
	var mesh: Mesh = mesh_instance.mesh if mesh_instance != null else null
	module.free()
	if mesh != null:
		_module_mesh_cache[resource_path] = mesh
	return mesh


func _find_first_mesh(node: Node) -> MeshInstance3D:
	if node is MeshInstance3D:
		return node as MeshInstance3D
	for child: Node in node.get_children():
		var result := _find_first_mesh(child)
		if result != null:
			return result
	return null


func _set_model_material(root_node: Node, material: Material) -> void:
	if root_node is MeshInstance3D:
		(root_node as MeshInstance3D).material_override = material
	for child: Node in root_node.get_children():
		_set_model_material(child, material)


func _cell_seed(cell: Vector2i, salt: String) -> int:
	return absi((cell.x + 17) * 73856093 ^ (cell.y + 31) * 19349663 ^ salt.hash())
