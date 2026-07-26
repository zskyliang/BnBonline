"""Generate original storybook props, effects, floor, and forest environments.

The script is intentionally deterministic and uses only Blender primitives and
small authored meshes.  Every public generator resets the scene, writes a
source .blend, and exports a Web-safe GLB under assets/models/.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector


PROJECT_ROOT = Path("/Users/slzeng/Documents/work/vibe/game/BnBonline")
AUTHORING_ROOT = PROJECT_ROOT / "art" / "blender"
RUNTIME_ROOT = PROJECT_ROOT / "assets" / "models"
PREVIEW_ROOT = AUTHORING_ROOT / "previews"


def _reset_scene() -> None:
    if getattr(bpy.context, "object", None) and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def _material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float = 0.88,
    metallic: float = 0.0,
    transmission: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    material.roughness = roughness
    material.metallic = metallic
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = roughness
        principled.inputs["Metallic"].default_value = metallic
        principled.inputs["Transmission Weight"].default_value = transmission
        principled.inputs["Alpha"].default_value = color[3]
    if color[3] < 0.999:
        material.surface_render_method = "DITHERED"
    material["storybook_material"] = True
    return material


def _finish_mesh(
    obj: bpy.types.Object,
    material: bpy.types.Material,
    *,
    smooth: bool = True,
    bevel: float = 0.0,
) -> bpy.types.Object:
    obj.data.materials.append(material)
    obj["material_role"] = material.name
    if smooth:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    if bevel > 0.0:
        modifier = obj.modifiers.new("SoftStorybookEdges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 2
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    obj.select_set(False)
    return obj


def _uv(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    segments: int = 14,
    rings: int = 7,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    return _finish_mesh(obj, material)


def _ico(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    subdivisions: int = 2,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=subdivisions,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    return _finish_mesh(obj, material)


def _box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bevel: float = 0.05,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    return _finish_mesh(obj, material, smooth=False, bevel=bevel)


def _cylinder(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
    *,
    vertices: int = 12,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    return _finish_mesh(obj, material)


def _cone(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
    *,
    vertices: int = 10,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=radius,
        radius2=0.02,
        depth=depth,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    return _finish_mesh(obj, material)


def _leaf(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    vertices = [
        (0.0, -1.0, 0.0),
        (0.68, -0.28, 0.05),
        (0.58, 0.35, 0.03),
        (0.0, 1.0, 0.0),
        (-0.58, 0.35, 0.03),
        (-0.68, -0.28, 0.05),
    ]
    faces = [(0, 1, 2, 3, 4, 5)]
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.scale = scale
    obj.rotation_euler.z = rotation_z
    solidify = obj.modifiers.new("PaperThickness", "SOLIDIFY")
    solidify.thickness = 0.045
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=solidify.name)
    return _finish_mesh(obj, material, smooth=False, bevel=0.02)


def _star(
    name: str,
    location: tuple[float, float, float],
    outer_radius: float,
    inner_radius: float,
    depth: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    vertices: list[tuple[float, float, float]] = []
    point_count = 10
    for z in (-depth * 0.5, depth * 0.5):
        for index in range(point_count):
            angle = math.pi * 0.5 + index * math.pi / 5.0
            radius = outer_radius if index % 2 == 0 else inner_radius
            vertices.append((math.cos(angle) * radius, math.sin(angle) * radius, z))
    faces: list[tuple[int, ...]] = []
    faces.append(tuple(range(point_count - 1, -1, -1)))
    faces.append(tuple(range(point_count, point_count * 2)))
    for index in range(point_count):
        nxt = (index + 1) % point_count
        faces.append((index, nxt, point_count + nxt, point_count + index))
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    return _finish_mesh(obj, material, smooth=False, bevel=0.045)


def _look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_preview_scene(
    asset_name: str,
    *,
    camera_location: tuple[float, float, float] = (2.8, -5.5, 2.8),
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.5),
    ortho_scale: float = 2.4,
    resolution: tuple[int, int] = (768, 768),
) -> None:
    preview_material = _material("PreviewPaper", (0.88, 0.82, 0.69, 1.0))
    bpy.ops.mesh.primitive_plane_add(size=80.0, location=(0.0, 0.0, -0.035))
    ground = bpy.context.object
    ground.name = "PreviewGround"
    ground["preview_only"] = True
    ground.data.materials.append(preview_material)

    bpy.ops.object.light_add(type="AREA", location=(-4.0, -4.0, 7.0))
    key = bpy.context.object
    key.name = "PreviewKey"
    key["preview_only"] = True
    key.data.energy = 180.0
    key.data.shape = "DISK"
    key.data.size = 6.0
    key.data.color = (1.0, 0.84, 0.68)
    _look_at(key, camera_target)

    bpy.ops.object.light_add(type="AREA", location=(4.0, 1.0, 4.0))
    fill = bpy.context.object
    fill.name = "PreviewFill"
    fill["preview_only"] = True
    fill.data.energy = 55.0
    fill.data.size = 5.0
    fill.data.color = (0.64, 0.78, 1.0)
    _look_at(fill, camera_target)

    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.name = "PreviewCamera"
    camera["preview_only"] = True
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale
    _look_at(camera, camera_target)
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(PREVIEW_ROOT / f"{asset_name}-blender-preview.png")
    scene.world.color = (0.88, 0.82, 0.69)
    scene.view_settings.look = "Medium High Contrast"
    bpy.ops.render.render(write_still=True)


def _remove_preview_objects() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.get("preview_only", False):
            bpy.data.objects.remove(obj, do_unlink=True)


def _mesh_stats() -> dict[str, int]:
    triangles = 0
    vertices = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.get("preview_only", False):
            continue
        vertices += len(obj.data.vertices)
        triangles += sum(max(1, len(poly.vertices) - 2) for poly in obj.data.polygons)
    return {"triangles": triangles, "vertices": vertices}


def _save_and_export(
    *,
    source_path: Path,
    runtime_path: Path,
    asset_id: str,
    budget_triangles: int,
) -> dict[str, object]:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    stats = _mesh_stats()
    if stats["triangles"] > budget_triangles:
        raise RuntimeError(
            f"{asset_id} exceeds triangle budget: {stats['triangles']} > {budget_triangles}"
        )
    scene = bpy.context.scene
    scene["asset_pipeline"] = "Forest Bubble Paint / storybook-v2"
    scene["asset_id"] = asset_id
    scene["triangle_count"] = stats["triangles"]
    scene["vertex_count"] = stats["vertices"]
    bpy.ops.wm.save_as_mainfile(filepath=str(source_path))
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if not obj.get("preview_only", False):
            obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(runtime_path),
        export_format="GLB",
        use_selection=True,
        export_materials="EXPORT",
        export_animations=False,
        export_yup=True,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(source_path))
    return {
        "asset_id": asset_id,
        "blend": str(source_path),
        "glb": str(runtime_path),
        **stats,
    }


def _prepare_scene(asset_id: str) -> None:
    _reset_scene()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene["reference_path"] = str(
        PROJECT_ROOT
        / "art"
        / "reference"
        / "production"
        / (
            "items-modeling-reference-v1.png"
            if asset_id in {"leaf_shoes", "bubble_gourd", "paw_burst"}
            else "bubble-effects-reference-v1.png"
            if asset_id in {"bubble_bomb", "effect_shapes"}
            else "forest-board-concept-v1.png"
            if asset_id in {"storybook_floor_tile", "forest_board_decor"}
            else "storybook-lobby-concept-v1.png"
        )
    )


def generate_leaf_shoes() -> dict[str, object]:
    _prepare_scene("leaf_shoes")
    green = _material("LeafGreen", (0.30, 0.55, 0.12, 1.0))
    dark = _material("LeafDark", (0.10, 0.24, 0.055, 1.0))
    cream = _material("WingCream", (0.90, 0.82, 0.56, 1.0))
    for side, x in (("L", -0.22), ("R", 0.22)):
        _box(
            f"Shoe.{side}",
            (x, 0.0, 0.18),
            (0.18, 0.30, 0.16),
            green,
            bevel=0.10,
        )
        _box(
            f"Sole.{side}",
            (x, -0.02, 0.045),
            (0.20, 0.32, 0.045),
            dark,
            bevel=0.045,
        )
        _leaf(
            f"TopLeaf.{side}",
            (x, 0.03, 0.47),
            (0.13, 0.23, 0.13),
            green,
            rotation_z=-0.18 if side == "L" else 0.18,
        )
        for wing_index in range(2):
            _leaf(
                f"Wing.{side}.{wing_index}",
                (
                    x + (-0.22 if side == "L" else 0.22),
                    -0.02 + wing_index * 0.10,
                    0.25 + wing_index * 0.08,
                ),
                (0.085, 0.13, 0.08),
                cream,
                rotation_z=-0.9 if side == "L" else 0.9,
            )
    _add_preview_scene("leaf-shoes", ortho_scale=1.55)
    _remove_preview_objects()
    return _save_and_export(
        source_path=AUTHORING_ROOT / "props" / "leaf_shoes.blend",
        runtime_path=RUNTIME_ROOT / "items" / "storybook" / "leaf_shoes.glb",
        asset_id="leaf_shoes",
        budget_triangles=2000,
    )


def generate_bubble_gourd() -> dict[str, object]:
    _prepare_scene("bubble_gourd")
    glass = _material("BubbleGlass", (0.44, 0.72, 0.90, 0.78), roughness=0.28)
    highlight = _material("BubbleHighlight", (0.91, 0.96, 0.95, 1.0))
    cork = _material("Cork", (0.43, 0.24, 0.095, 1.0))
    rope = _material("Rope", (0.56, 0.35, 0.12, 1.0))
    _uv("LowerBubble", (0.0, 0.0, 0.29), (0.31, 0.31, 0.31), glass)
    _uv("UpperBubble", (0.0, 0.0, 0.70), (0.21, 0.21, 0.21), glass)
    _ico("LowerHighlight", (-0.11, -0.27, 0.38), (0.065, 0.018, 0.10), highlight)
    _ico("UpperHighlight", (-0.075, -0.18, 0.76), (0.045, 0.015, 0.07), highlight)
    _cylinder("Cork", (0.0, 0.0, 0.96), 0.10, 0.14, cork)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.225,
        minor_radius=0.026,
        major_segments=18,
        minor_segments=6,
        location=(0.0, 0.0, 0.52),
    )
    knot = bpy.context.object
    knot.name = "RopeKnot"
    _finish_mesh(knot, rope)
    _add_preview_scene("bubble-gourd", ortho_scale=1.45)
    _remove_preview_objects()
    return _save_and_export(
        source_path=AUTHORING_ROOT / "props" / "bubble_gourd.blend",
        runtime_path=RUNTIME_ROOT / "items" / "storybook" / "bubble_gourd.glb",
        asset_id="bubble_gourd",
        budget_triangles=2000,
    )


def generate_paw_burst() -> dict[str, object]:
    _prepare_scene("paw_burst")
    gold = _material("GoldenBurst", (0.96, 0.60, 0.08, 1.0), roughness=0.70)
    paw = _material("PawInset", (0.48, 0.22, 0.055, 1.0))
    _star("GoldenPawBurst", (0.0, 0.0, 0.48), 0.48, 0.31, 0.13, gold)
    _ico("PawPad", (0.0, -0.075, 0.48), (0.19, 0.075, 0.15), paw)
    for index, x in enumerate((-0.23, -0.075, 0.075, 0.23)):
        height = 0.70 + (0.05 if index in {1, 2} else 0.0)
        _ico(
            f"PawToe.{index + 1}",
            (x, -0.075, height),
            (0.075, 0.060, 0.095),
            paw,
        )
    _add_preview_scene("paw-burst", ortho_scale=1.35)
    _remove_preview_objects()
    return _save_and_export(
        source_path=AUTHORING_ROOT / "props" / "paw_burst.blend",
        runtime_path=RUNTIME_ROOT / "items" / "storybook" / "paw_burst.glb",
        asset_id="paw_burst",
        budget_triangles=2000,
    )


def generate_bubble_bomb() -> dict[str, object]:
    _prepare_scene("bubble_bomb")
    tint = _material("TeamTint", (0.30, 0.65, 0.82, 0.84), roughness=0.32)
    highlight = _material("BubbleHighlight", (0.94, 0.97, 0.93, 1.0))
    cork = _material("Cork", (0.45, 0.25, 0.10, 1.0))
    leaf = _material("Leaf", (0.25, 0.48, 0.10, 1.0))
    _uv("BubbleCore", (0.0, 0.0, 0.34), (0.34, 0.34, 0.34), tint)
    _ico("BubbleHighlight", (-0.13, -0.29, 0.47), (0.075, 0.018, 0.12), highlight)
    _cylinder("LeafCork", (0.0, 0.0, 0.72), 0.095, 0.15, cork)
    _leaf("Leaf.L", (-0.08, 0.0, 0.88), (0.095, 0.17, 0.09), leaf, rotation_z=-0.45)
    _leaf("Leaf.R", (0.08, 0.0, 0.88), (0.095, 0.17, 0.09), leaf, rotation_z=0.45)
    _add_preview_scene("bubble-bomb", ortho_scale=1.35)
    _remove_preview_objects()
    return _save_and_export(
        source_path=AUTHORING_ROOT / "effects" / "bubble_bomb.blend",
        runtime_path=RUNTIME_ROOT / "effects" / "storybook" / "bubble_bomb.glb",
        asset_id="bubble_bomb",
        budget_triangles=2000,
    )


def generate_effect_shapes() -> dict[str, object]:
    _prepare_scene("effect_shapes")
    foam = _material("Foam", (0.88, 0.94, 0.88, 1.0))
    water = _material("TeamTint", (0.30, 0.65, 0.82, 1.0))
    _uv("FoamPetal", (-0.45, 0.0, 0.18), (0.20, 0.11, 0.34), foam)
    _uv("Droplet", (0.0, 0.0, 0.24), (0.13, 0.10, 0.24), water)
    _cone("BurstSpike", (0.45, 0.0, 0.30), 0.16, 0.60, water, vertices=7)
    _add_preview_scene("effect-shapes", ortho_scale=1.55)
    _remove_preview_objects()
    return _save_and_export(
        source_path=AUTHORING_ROOT / "effects" / "effect_shapes.blend",
        runtime_path=RUNTIME_ROOT / "effects" / "storybook" / "effect_shapes.glb",
        asset_id="effect_shapes",
        budget_triangles=2000,
    )


def generate_storybook_floor_tile() -> dict[str, object]:
    _prepare_scene("storybook_floor_tile")
    cream = _material("StorybookTile", (0.83, 0.74, 0.55, 1.0))
    points = [
        (-0.47, -0.44),
        (-0.16, -0.48),
        (0.18, -0.46),
        (0.48, -0.40),
        (0.46, 0.17),
        (0.40, 0.47),
        (-0.18, 0.45),
        (-0.46, 0.39),
    ]
    vertices = [(x, y, z) for z in (-0.035, 0.035) for x, y in points]
    faces: list[tuple[int, ...]] = [
        tuple(range(7, -1, -1)),
        tuple(range(8, 16)),
    ]
    for index in range(8):
        nxt = (index + 1) % 8
        faces.append((index, nxt, 8 + nxt, 8 + index))
    mesh = bpy.data.meshes.new("StorybookFloorTileMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    tile = bpy.data.objects.new("StorybookFloorTile", mesh)
    bpy.context.collection.objects.link(tile)
    _finish_mesh(tile, cream, smooth=False, bevel=0.035)
    _add_preview_scene("storybook-floor-tile", ortho_scale=1.65)
    _remove_preview_objects()
    return _save_and_export(
        source_path=AUTHORING_ROOT / "environment" / "storybook_floor_tile.blend",
        runtime_path=RUNTIME_ROOT / "environment" / "storybook" / "storybook_floor_tile.glb",
        asset_id="storybook_floor_tile",
        budget_triangles=2000,
    )


def _build_mushroom(
    prefix: str,
    location: tuple[float, float, float],
    scale: float,
    stem_material: bpy.types.Material,
    cap_material: bpy.types.Material,
) -> None:
    x, y, z = location
    _cylinder(f"{prefix}.Stem", (x, y, z + 0.20 * scale), 0.10 * scale, 0.40 * scale, stem_material)
    _uv(
        f"{prefix}.Cap",
        (x, y, z + 0.43 * scale),
        (0.28 * scale, 0.28 * scale, 0.16 * scale),
        cap_material,
    )


def _build_paper_tree(
    prefix: str,
    location: tuple[float, float, float],
    scale: float,
    trunk_material: bpy.types.Material,
    leaf_material: bpy.types.Material,
) -> None:
    x, y, z = location
    _cylinder(f"{prefix}.Trunk", (x, y, z + 0.42 * scale), 0.12 * scale, 0.84 * scale, trunk_material)
    for layer in range(3):
        _cone(
            f"{prefix}.Crown.{layer + 1}",
            (x, y, z + (0.80 + layer * 0.28) * scale),
            (0.48 - layer * 0.08) * scale,
            0.70 * scale,
            leaf_material,
            vertices=8,
        )


def generate_forest_board_decor() -> dict[str, object]:
    _prepare_scene("forest_board_decor")
    wood = _material("ForestWood", (0.32, 0.16, 0.065, 1.0))
    wood_light = _material("WoodHighlight", (0.50, 0.28, 0.11, 1.0))
    moss = _material("Moss", (0.24, 0.40, 0.10, 1.0))
    leaf_dark = _material("LeafDark", (0.07, 0.23, 0.13, 1.0))
    leaf_light = _material("LeafLight", (0.32, 0.49, 0.15, 1.0))
    cream = _material("MushroomStem", (0.83, 0.74, 0.55, 1.0))
    red = _material("MushroomRed", (0.66, 0.12, 0.075, 1.0))
    blue = _material("MushroomBlue", (0.12, 0.34, 0.52, 1.0))
    flower = _material("Flowers", (0.92, 0.55, 0.10, 1.0))

    _box("Frame.North", (0.0, 6.68, 0.13), (7.82, 0.18, 0.13), wood, bevel=0.10)
    _box("Frame.South", (0.0, -6.68, 0.13), (7.82, 0.18, 0.13), wood, bevel=0.10)
    _box("Frame.West", (-7.68, 0.0, 0.13), (0.18, 6.52, 0.13), wood, bevel=0.10)
    _box("Frame.East", (7.68, 0.0, 0.13), (0.18, 6.52, 0.13), wood, bevel=0.10)
    for index, location in enumerate(
        [(-6.8, -7.15, 0.10), (-3.5, -7.25, 0.10), (3.8, -7.25, 0.10), (6.9, -7.10, 0.10),
         (-7.95, -3.5, 0.10), (-7.95, 3.8, 0.10), (7.95, -3.0, 0.10), (7.95, 3.6, 0.10)]
    ):
        _ico(
            f"Moss.{index + 1}",
            location,
            (0.62, 0.42, 0.16),
            moss if index % 2 == 0 else leaf_light,
            subdivisions=1,
        )
    tree_specs = [
        ("NW", (-8.45, 5.5, 0.0), 1.20, leaf_dark),
        ("NE", (8.45, 5.2, 0.0), 1.14, leaf_dark),
        ("SW", (-8.35, -5.0, 0.0), 0.92, leaf_light),
        ("SE", (8.30, -5.0, 0.0), 0.88, leaf_light),
    ]
    for prefix, location, scale, leaf_material in tree_specs:
        _build_paper_tree(prefix, location, scale, wood_light, leaf_material)
    _build_mushroom("Mushroom.NW", (-7.25, 6.95, 0.0), 0.92, cream, red)
    _build_mushroom("Mushroom.NE", (7.05, 7.05, 0.0), 0.82, cream, blue)
    _build_mushroom("Mushroom.SW", (-6.65, -7.35, 0.0), 0.75, cream, blue)
    _build_mushroom("Mushroom.SE", (6.65, -7.30, 0.0), 0.86, cream, red)
    for index, (x, y) in enumerate(
        [(-5.2, 7.15), (-2.4, -7.2), (2.6, 7.18), (5.0, -7.18), (-8.15, 0.2), (8.15, 0.6)]
    ):
        _ico(f"Flower.{index + 1}", (x, y, 0.18), (0.14, 0.14, 0.18), flower, subdivisions=1)
    _add_preview_scene(
        "forest-board-decor",
        camera_location=(-11.5, -16.5, 15.0),
        camera_target=(0.0, 0.0, 0.0),
        ortho_scale=22.0,
        resolution=(1280, 720),
    )
    _remove_preview_objects()
    return _save_and_export(
        source_path=AUTHORING_ROOT / "environment" / "forest_board_decor.blend",
        runtime_path=RUNTIME_ROOT / "environment" / "storybook" / "forest_board_decor.glb",
        asset_id="forest_board_decor",
        budget_triangles=12000,
    )


def generate_storybook_lobby() -> dict[str, object]:
    _prepare_scene("storybook_lobby")
    grass = _material("LobbyGrass", (0.31, 0.45, 0.14, 1.0))
    path = _material("LobbyPath", (0.78, 0.66, 0.43, 1.0))
    wood = _material("LobbyWood", (0.34, 0.17, 0.065, 1.0))
    paper = _material("SignPaper", (0.87, 0.77, 0.56, 1.0))
    leaf_dark = _material("LobbyLeafDark", (0.07, 0.23, 0.13, 1.0))
    leaf_light = _material("LobbyLeafLight", (0.34, 0.50, 0.15, 1.0))
    cream = _material("LobbyCream", (0.85, 0.75, 0.56, 1.0))
    red = _material("LobbyMushroom", (0.68, 0.15, 0.075, 1.0))
    accent_colors = [
        (0.36, 0.67, 0.76, 1.0),
        (0.46, 0.69, 0.28, 1.0),
        (0.83, 0.46, 0.35, 1.0),
        (0.53, 0.42, 0.72, 1.0),
        (0.84, 0.50, 0.18, 1.0),
        (0.48, 0.39, 0.65, 1.0),
        (0.34, 0.58, 0.72, 1.0),
        (0.74, 0.57, 0.22, 1.0),
    ]
    _cylinder("LobbyIsland", (0.0, 0.0, -0.15), 5.5, 0.30, grass, vertices=24)
    _box("LobbyPath", (0.0, -3.7, 0.02), (0.85, 2.2, 0.04), path, bevel=0.25)
    _box("SignPaper", (0.0, 3.6, 2.2), (2.4, 0.12, 0.78), paper, bevel=0.18)
    _cylinder("SignPost.L", (-2.35, 3.65, 1.25), 0.13, 2.50, wood)
    _cylinder("SignPost.R", (2.35, 3.65, 1.25), 0.13, 2.50, wood)
    _box("SignTop", (0.0, 3.62, 2.95), (2.65, 0.14, 0.12), wood, bevel=0.10)
    positions = [
        (-2.6, 1.4), (-0.9, 1.7), (0.9, 1.7), (2.6, 1.4),
        (-2.6, -0.6), (-0.9, -0.9), (0.9, -0.9), (2.6, -0.6),
    ]
    for index, ((x, y), color) in enumerate(zip(positions, accent_colors)):
        mat = _material(f"CharacterSpot.{index + 1}", color)
        _cylinder(f"CharacterSpot.{index + 1}", (x, y, 0.06), 0.55, 0.12, mat, vertices=20)
    _build_paper_tree("LobbyTree.L", (-4.6, 2.3, 0.0), 1.10, wood, leaf_dark)
    _build_paper_tree("LobbyTree.R", (4.6, 2.3, 0.0), 1.10, wood, leaf_dark)
    _build_paper_tree("LobbyTree.BackL", (-3.6, 4.0, 0.0), 0.78, wood, leaf_light)
    _build_paper_tree("LobbyTree.BackR", (3.6, 4.0, 0.0), 0.78, wood, leaf_light)
    _build_mushroom("LobbyMushroom.L", (-4.2, -2.7, 0.0), 0.80, cream, red)
    _build_mushroom("LobbyMushroom.R", (4.2, -2.7, 0.0), 0.72, cream, red)
    _add_preview_scene(
        "storybook-lobby",
        camera_location=(-8.5, -12.0, 9.2),
        camera_target=(0.0, 0.3, 0.7),
        ortho_scale=13.2,
        resolution=(1280, 720),
    )
    _remove_preview_objects()
    return _save_and_export(
        source_path=AUTHORING_ROOT / "environment" / "storybook_lobby.blend",
        runtime_path=RUNTIME_ROOT / "environment" / "storybook" / "storybook_lobby.glb",
        asset_id="storybook_lobby",
        budget_triangles=18000,
    )


def generate_all_world_assets() -> list[dict[str, object]]:
    generators = [
        generate_leaf_shoes,
        generate_bubble_gourd,
        generate_paw_burst,
        generate_bubble_bomb,
        generate_effect_shapes,
        generate_storybook_floor_tile,
        generate_forest_board_decor,
        generate_storybook_lobby,
    ]
    return [generator() for generator in generators]
