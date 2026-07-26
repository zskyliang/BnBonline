"""Generate all eight approved Forest Bubble Paint storybook characters with Blender.

Run from Blender's Python environment. The public entry point is:

    generate_character("cat")
    generate_all_characters()

Each call rebuilds the current scene, creates a compact shared-style rig,
authors six stepped 8 FPS actions, saves an individual .blend source, exports a
GLB for Godot, and renders a square Blender preview.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector


PROJECT_ROOT = Path("/Users/slzeng/Documents/work/vibe/game/BnBonline")
BLEND_DIR = PROJECT_ROOT / "art" / "blender" / "characters"
PREVIEW_DIR = PROJECT_ROOT / "art" / "blender" / "previews"
GLB_DIR = PROJECT_ROOT / "assets" / "models" / "characters"

FPS = 8
ACTION_NAMES = (
    "Idle",
    "Waddle",
    "PlaceBubble",
    "Trapped",
    "Defeat",
    "Victory",
)

SPECIES = {
    "cat": {
        "fur": (0.14, 0.18, 0.22, 1.0),
        "fur_dark": (0.04, 0.05, 0.06, 1.0),
        "belly": (0.80, 0.74, 0.63, 1.0),
        "muzzle": (0.85, 0.79, 0.67, 1.0),
        "nose": (0.62, 0.19, 0.18, 1.0),
        "team": (0.36, 0.67, 0.76, 1.0),
        "body_scale": (0.38, 0.30, 0.43),
        "head_scale": (0.43, 0.34, 0.38),
    },
    "bear": {
        "fur": (0.28, 0.11, 0.045, 1.0),
        "fur_dark": (0.075, 0.03, 0.012, 1.0),
        "belly": (0.68, 0.40, 0.17, 1.0),
        "muzzle": (0.72, 0.48, 0.22, 1.0),
        "nose": (0.06, 0.025, 0.012, 1.0),
        "team": (0.53, 0.42, 0.72, 1.0),
        "body_scale": (0.41, 0.32, 0.45),
        "head_scale": (0.44, 0.35, 0.39),
    },
    "dog": {
        "fur": (0.66, 0.39, 0.16, 1.0),
        "fur_dark": (0.22, 0.105, 0.045, 1.0),
        "belly": (0.89, 0.76, 0.52, 1.0),
        "muzzle": (0.91, 0.79, 0.58, 1.0),
        "nose": (0.08, 0.035, 0.016, 1.0),
        "team": (0.46, 0.69, 0.28, 1.0),
        "body_scale": (0.39, 0.31, 0.43),
        "head_scale": (0.43, 0.34, 0.38),
    },
    "rabbit": {
        "fur": (0.78, 0.64, 0.42, 1.0),
        "fur_dark": (0.27, 0.14, 0.075, 1.0),
        "belly": (0.94, 0.87, 0.70, 1.0),
        "muzzle": (0.95, 0.88, 0.72, 1.0),
        "nose": (0.72, 0.28, 0.28, 1.0),
        "team": (0.83, 0.46, 0.35, 1.0),
        "body_scale": (0.36, 0.29, 0.42),
        "head_scale": (0.40, 0.33, 0.37),
    },
    "fox": {
        "fur": (0.86, 0.28, 0.055, 1.0),
        "fur_dark": (0.16, 0.055, 0.022, 1.0),
        "belly": (0.91, 0.77, 0.54, 1.0),
        "muzzle": (0.94, 0.84, 0.66, 1.0),
        "nose": (0.045, 0.022, 0.012, 1.0),
        "team": (0.84, 0.50, 0.18, 1.0),
        "body_scale": (0.37, 0.29, 0.42),
        "head_scale": (0.43, 0.33, 0.37),
    },
    "raccoon": {
        "fur": (0.36, 0.36, 0.33, 1.0),
        "fur_dark": (0.07, 0.065, 0.055, 1.0),
        "belly": (0.73, 0.68, 0.57, 1.0),
        "muzzle": (0.77, 0.72, 0.62, 1.0),
        "nose": (0.035, 0.030, 0.026, 1.0),
        "team": (0.48, 0.39, 0.65, 1.0),
        "body_scale": (0.38, 0.30, 0.43),
        "head_scale": (0.43, 0.34, 0.38),
    },
    "penguin": {
        "fur": (0.07, 0.09, 0.12, 1.0),
        "fur_dark": (0.025, 0.03, 0.04, 1.0),
        "belly": (0.93, 0.90, 0.80, 1.0),
        "muzzle": (0.95, 0.92, 0.84, 1.0),
        "nose": (0.92, 0.48, 0.08, 1.0),
        "team": (0.34, 0.58, 0.72, 1.0),
        "body_scale": (0.39, 0.31, 0.45),
        "head_scale": (0.41, 0.33, 0.39),
    },
    "capybara": {
        "fur": (0.50, 0.29, 0.12, 1.0),
        "fur_dark": (0.13, 0.07, 0.030, 1.0),
        "belly": (0.70, 0.48, 0.24, 1.0),
        "muzzle": (0.29, 0.17, 0.085, 1.0),
        "nose": (0.075, 0.045, 0.025, 1.0),
        "team": (0.74, 0.57, 0.22, 1.0),
        "body_scale": (0.43, 0.34, 0.47),
        "head_scale": (0.42, 0.35, 0.37),
    },
}


def _reset_scene() -> None:
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    for data_collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(data_collection):
            if datablock.users == 0:
                data_collection.remove(datablock)


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    material.roughness = 0.88
    material.metallic = 0.0
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = 0.88
        principled.inputs["Metallic"].default_value = 0.0
        principled.inputs["Specular IOR Level"].default_value = 0.22
    material["storybook_material"] = True
    return material


def _make_materials(species: str) -> dict[str, bpy.types.Material]:
    palette = SPECIES[species]
    return {
        "fur": _material("Fur", palette["fur"]),
        "fur_dark": _material("SpeciesMarking", palette["fur_dark"]),
        "belly": _material("Belly", palette["belly"]),
        "muzzle": _material("Muzzle", palette["muzzle"]),
        "eye": _material("Eyes", (0.96, 0.94, 0.88, 1.0)),
        "pupil": _material("Pupils", (0.06, 0.055, 0.05, 1.0)),
        "nose": _material("Nose", palette["nose"]),
        "team": _material("TeamTint", palette["team"]),
        "ring": _material("FootRing", palette["team"]),
    }


def _new_rig(species: str) -> bpy.types.Object:
    armature_data = bpy.data.armatures.new("ForestAnimalRig")
    rig = bpy.data.objects.new("ForestAnimalRig", armature_data)
    bpy.context.collection.objects.link(rig)
    rig.show_in_front = True
    rig["character_id"] = species
    rig["unit_height"] = 1.3
    rig["animation_fps"] = FPS
    rig["team_tint_material_names"] = "TeamTint,FootRing"

    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    bones = {
        "Root": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.18), None),
        "Body": ((0.0, 0.0, 0.36), (0.0, 0.0, 0.86), "Root"),
        "Head": ((0.0, 0.0, 0.82), (0.0, 0.0, 1.25), "Body"),
        "Arm.L": ((-0.25, 0.0, 0.72), (-0.43, 0.0, 0.48), "Body"),
        "Arm.R": ((0.25, 0.0, 0.72), (0.43, 0.0, 0.48), "Body"),
        "Leg.L": ((-0.16, 0.0, 0.34), (-0.16, 0.0, 0.07), "Root"),
        "Leg.R": ((0.16, 0.0, 0.34), (0.16, 0.0, 0.07), "Root"),
        "Ear.L": ((-0.20, 0.0, 1.13), (-0.22, 0.0, 1.32), "Head"),
        "Ear.R": ((0.20, 0.0, 1.13), (0.22, 0.0, 1.32), "Head"),
        "Tail": ((0.0, 0.16, 0.49), (0.27, 0.22, 0.58), "Body"),
    }
    edit_bones: dict[str, bpy.types.EditBone] = {}
    for bone_name, (head, tail, parent_name) in bones.items():
        bone = armature_data.edit_bones.new(bone_name)
        bone.head = head
        bone.tail = tail
        if parent_name:
            bone.parent = edit_bones[parent_name]
        edit_bones[bone_name] = bone

    bpy.ops.object.mode_set(mode="POSE")
    for pose_bone in rig.pose.bones:
        pose_bone.rotation_mode = "XYZ"
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def _parent_to_bone(obj: bpy.types.Object, rig: bpy.types.Object, bone_name: str) -> None:
    world_matrix = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_world = world_matrix
    obj["attached_bone"] = bone_name


def _smooth_mesh(obj: bpy.types.Object) -> None:
    if obj.type != "MESH":
        return
    for polygon in obj.data.polygons:
        polygon.use_smooth = True


def _assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    obj.data.materials.append(material)
    obj["material_role"] = material.name


def _freeze_mesh_transform(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.context.view_layer.update()


def _ico(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    rig: bpy.types.Object,
    bone: str,
    subdivisions: int = 2,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=subdivisions,
        radius=1.0,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    _smooth_mesh(obj)
    _assign_material(obj, material)
    _freeze_mesh_transform(obj)
    _parent_to_bone(obj, rig, bone)
    return obj


def _uv(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    rig: bpy.types.Object,
    bone: str,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16,
        ring_count=8,
        radius=1.0,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    _smooth_mesh(obj)
    _assign_material(obj, material)
    _freeze_mesh_transform(obj)
    _parent_to_bone(obj, rig, bone)
    return obj


def _cone(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
    rig: bpy.types.Object,
    bone: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(
        vertices=3,
        radius1=radius,
        radius2=0.025,
        depth=depth,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    _assign_material(obj, material)
    _freeze_mesh_transform(obj)
    _parent_to_bone(obj, rig, bone)
    return obj


def _torus(
    name: str,
    material: bpy.types.Material,
    rig: bpy.types.Object,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.34,
        minor_radius=0.014,
        major_segments=32,
        minor_segments=6,
        location=(0.0, 0.0, 0.018),
    )
    obj = bpy.context.object
    obj.name = name
    _smooth_mesh(obj)
    _assign_material(obj, material)
    _parent_to_bone(obj, rig, "Root")
    return obj


def _build_common_face(
    species: str,
    rig: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    palette = SPECIES[species]
    _uv(
        "Body",
        (0.0, 0.0, 0.50),
        palette["body_scale"],
        materials["fur"],
        rig,
        "Body",
    )
    _uv(
        "Head",
        (0.0, -0.01, 0.99),
        palette["head_scale"],
        materials["fur"],
        rig,
        "Head",
    )
    _ico(
        "Belly",
        (0.0, -0.292, 0.50),
        (0.255, 0.046, 0.31),
        materials["belly"],
        rig,
        "Body",
    )

    for side, x in (("L", -0.14), ("R", 0.14)):
        _uv(
            f"Eye.{side}",
            (x, -0.326, 1.07),
            (0.137, 0.052, 0.151),
            materials["eye"],
            rig,
            "Head",
        )
    _ico(
        "Pupil.L",
        (-0.127, -0.378, 1.065),
        (0.030, 0.018, 0.041),
        materials["pupil"],
        rig,
        "Head",
        2,
    )
    _ico(
        "Pupil.R",
        (0.151, -0.378, 1.083),
        (0.028, 0.018, 0.039),
        materials["pupil"],
        rig,
        "Head",
        2,
    )

    for side, x in (("L", -0.068), ("R", 0.068)):
        _ico(
            f"Muzzle.{side}",
            (x, -0.350, 0.91),
            (0.105, 0.060, 0.082),
            materials["muzzle"],
            rig,
            "Head",
        )
    _ico(
        "Nose",
        (0.0, -0.414, 0.935),
        (0.052, 0.031, 0.041),
        materials["nose"],
        rig,
        "Head",
    )
    _ico(
        "MouthDot",
        (0.0, -0.405, 0.870),
        (0.018, 0.012, 0.020),
        materials["fur_dark"],
        rig,
        "Head",
        1,
    )


def _build_limbs(
    rig: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    for side, x, angle in (("L", -0.35, -0.20), ("R", 0.35, 0.20)):
        _uv(
            f"Arm.{side}",
            (x, -0.01, 0.62),
            (0.115, 0.115, 0.27),
            materials["fur"],
            rig,
            f"Arm.{side}",
            (0.0, angle, 0.0),
        )
        _ico(
            f"ForepawTint.{side}",
            (x + (-0.045 if side == "L" else 0.045), -0.028, 0.405),
            (0.09, 0.10, 0.075),
            materials["team"],
            rig,
            f"Arm.{side}",
        )
    for side, x in (("L", -0.17), ("R", 0.17)):
        _uv(
            f"Leg.{side}",
            (x, 0.0, 0.19),
            (0.155, 0.145, 0.20),
            materials["fur"],
            rig,
            f"Leg.{side}",
        )
        _ico(
            f"HindpawTint.{side}",
            (x, -0.105, 0.078),
            (0.14, 0.11, 0.060),
            materials["team"],
            rig,
            f"Leg.{side}",
        )


def _build_cat(
    rig: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    _build_common_face("cat", rig, materials)
    _build_limbs(rig, materials)
    for side, x in (("L", -0.29), ("R", 0.29)):
        _cone(
            f"Ear.{side}",
            (x, -0.005, 1.285),
            0.17,
            0.32,
            materials["fur"],
            rig,
            f"Ear.{side}",
            (0.78, 0.64, 1.0),
        )
        _cone(
            f"EarTint.{side}",
            (x, -0.075, 1.282),
            0.105,
            0.22,
            materials["team"],
            rig,
            f"Ear.{side}",
            (0.72, 0.34, 0.88),
        )
    for index, x in enumerate((-0.10, 0.0, 0.10)):
        _ico(
            f"ForeheadStripe.{index + 1}",
            (x, -0.347, 1.245 - abs(x) * 0.25),
            (0.025, 0.018, 0.085),
            materials["fur_dark"],
            rig,
            "Head",
            1,
            (0.0, (-0.22 + index * 0.22), 0.0),
        )
    _uv(
        "Tail.Base",
        (0.24, 0.20, 0.51),
        (0.10, 0.10, 0.25),
        materials["fur"],
        rig,
        "Tail",
        (0.0, 0.78, 0.0),
    )
    _uv(
        "Tail.Middle",
        (0.39, 0.20, 0.68),
        (0.09, 0.09, 0.22),
        materials["fur"],
        rig,
        "Tail",
        (0.0, 0.70, 0.0),
    )
    _ico(
        "TailTint",
        (0.48, 0.19, 0.83),
        (0.10, 0.10, 0.16),
        materials["team"],
        rig,
        "Tail",
    )
    _ico(
        "BackTint",
        (0.0, 0.294, 0.64),
        (0.16, 0.038, 0.13),
        materials["team"],
        rig,
        "Body",
    )


def _build_bear(
    rig: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    _build_common_face("bear", rig, materials)
    _build_limbs(rig, materials)
    for side, x in (("L", -0.26), ("R", 0.26)):
        _ico(
            f"Ear.{side}",
            (x, -0.01, 1.215),
            (0.135, 0.105, 0.135),
            materials["fur"],
            rig,
            f"Ear.{side}",
        )
        _ico(
            f"EarTint.{side}",
            (x, -0.105, 1.215),
            (0.068, 0.026, 0.068),
            materials["team"],
            rig,
            f"Ear.{side}",
            2,
        )
    _ico(
        "Tail",
        (0.0, 0.31, 0.44),
        (0.105, 0.075, 0.105),
        materials["fur"],
        rig,
        "Tail",
    )
    _ico(
        "BackTint",
        (0.0, 0.315, 0.65),
        (0.18, 0.035, 0.14),
        materials["team"],
        rig,
        "Body",
    )


def _build_dog(
    rig: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    _build_common_face("dog", rig, materials)
    _build_limbs(rig, materials)
    for side, x, tilt in (("L", -0.33, -0.16), ("R", 0.33, 0.16)):
        _uv(
            f"Ear.{side}",
            (x, -0.015, 1.075),
            (0.145, 0.105, 0.245),
            materials["fur_dark"],
            rig,
            f"Ear.{side}",
            (0.0, tilt, 0.10 if side == "L" else -0.10),
        )
        _ico(
            f"EarTint.{side}",
            (x, -0.112, 1.025),
            (0.073, 0.026, 0.115),
            materials["team"],
            rig,
            f"Ear.{side}",
        )
    _uv(
        "Tail",
        (0.25, 0.21, 0.54),
        (0.095, 0.09, 0.27),
        materials["fur_dark"],
        rig,
        "Tail",
        (0.0, 0.72, 0.0),
    )
    _ico(
        "TailTint",
        (0.43, 0.20, 0.70),
        (0.095, 0.085, 0.12),
        materials["team"],
        rig,
        "Tail",
    )
    _ico(
        "BackTint",
        (0.0, 0.300, 0.64),
        (0.16, 0.035, 0.13),
        materials["team"],
        rig,
        "Body",
    )


def _build_rabbit(
    rig: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    _build_common_face("rabbit", rig, materials)
    _build_limbs(rig, materials)
    for side, x, angle in (("L", -0.19, -0.055), ("R", 0.19, 0.055)):
        _uv(
            f"Ear.{side}",
            (x, 0.0, 1.42),
            (0.115, 0.075, 0.40),
            materials["fur"],
            rig,
            f"Ear.{side}",
            (0.0, angle, -0.06 if side == "L" else 0.06),
        )
        _uv(
            f"EarTint.{side}",
            (x, -0.070, 1.42),
            (0.056, 0.020, 0.29),
            materials["team"],
            rig,
            f"Ear.{side}",
        )
    _ico(
        "Tail",
        (0.0, 0.34, 0.45),
        (0.135, 0.11, 0.135),
        materials["muzzle"],
        rig,
        "Tail",
    )
    _ico(
        "BackTint",
        (0.0, 0.286, 0.66),
        (0.145, 0.032, 0.12),
        materials["team"],
        rig,
        "Body",
    )


def _build_fox(
    rig: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    _build_common_face("fox", rig, materials)
    _build_limbs(rig, materials)
    for side, x in (("L", -0.28), ("R", 0.28)):
        _cone(
            f"Ear.{side}",
            (x, -0.005, 1.285),
            0.17,
            0.33,
            materials["fur"],
            rig,
            f"Ear.{side}",
            (0.72, 0.58, 1.0),
        )
        _cone(
            f"EarTint.{side}",
            (x, -0.070, 1.285),
            0.095,
            0.21,
            materials["team"],
            rig,
            f"Ear.{side}",
            (0.68, 0.30, 0.85),
        )
    for side, x, yaw in (("L", -0.31, -0.55), ("R", 0.31, 0.55)):
        _cone(
            f"CheekTuft.{side}",
            (x, -0.29, 0.95),
            0.13,
            0.24,
            materials["muzzle"],
            rig,
            "Head",
            (0.72, 0.44, 0.88),
        ).rotation_euler.y = yaw
    _uv(
        "Tail.Base",
        (0.24, 0.20, 0.52),
        (0.13, 0.12, 0.32),
        materials["fur"],
        rig,
        "Tail",
        (0.0, 0.78, 0.0),
    )
    _uv(
        "Tail.Middle",
        (0.48, 0.19, 0.70),
        (0.17, 0.14, 0.34),
        materials["fur"],
        rig,
        "Tail",
        (0.0, 0.72, 0.0),
    )
    _ico(
        "TailTint",
        (0.70, 0.18, 0.91),
        (0.19, 0.14, 0.20),
        materials["team"],
        rig,
        "Tail",
    )
    _ico(
        "BackTint",
        (0.0, 0.288, 0.65),
        (0.16, 0.034, 0.13),
        materials["team"],
        rig,
        "Body",
    )


def _build_raccoon(
    rig: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    _build_common_face("raccoon", rig, materials)
    _build_limbs(rig, materials)
    for side, x, tilt in (("L", -0.15, -0.22), ("R", 0.15, 0.22)):
        _uv(
            f"EyeMask.{side}",
            (x, -0.315, 1.075),
            (0.20, 0.028, 0.125),
            materials["fur_dark"],
            rig,
            "Head",
            (0.0, tilt, 0.0),
        )
    for side, x in (("L", -0.26), ("R", 0.26)):
        _ico(
            f"Ear.{side}",
            (x, -0.01, 1.23),
            (0.13, 0.10, 0.13),
            materials["fur"],
            rig,
            f"Ear.{side}",
        )
        _ico(
            f"EarTint.{side}",
            (x, -0.104, 1.23),
            (0.062, 0.022, 0.062),
            materials["team"],
            rig,
            f"Ear.{side}",
        )
    tail_points = (
        ((0.24, 0.20, 0.50), (0.13, 0.12, 0.27), materials["fur_dark"]),
        ((0.42, 0.20, 0.63), (0.14, 0.12, 0.25), materials["team"]),
        ((0.59, 0.19, 0.76), (0.14, 0.115, 0.22), materials["fur_dark"]),
        ((0.73, 0.18, 0.88), (0.13, 0.105, 0.18), materials["team"]),
    )
    for index, (location, scale, material) in enumerate(tail_points):
        _ico(
            f"TailRing.{index + 1}",
            location,
            scale,
            material,
            rig,
            "Tail",
        )
    _ico(
        "BackTint",
        (0.0, 0.295, 0.64),
        (0.16, 0.035, 0.13),
        materials["team"],
        rig,
        "Body",
    )


def _build_penguin(
    rig: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    palette = SPECIES["penguin"]
    _uv(
        "Body",
        (0.0, 0.0, 0.50),
        palette["body_scale"],
        materials["fur"],
        rig,
        "Body",
    )
    _uv(
        "Head",
        (0.0, -0.01, 0.99),
        palette["head_scale"],
        materials["fur"],
        rig,
        "Head",
    )
    _ico(
        "FacePatch",
        (0.0, -0.321, 1.035),
        (0.285, 0.048, 0.245),
        materials["muzzle"],
        rig,
        "Head",
    )
    _ico(
        "Belly",
        (0.0, -0.300, 0.51),
        (0.275, 0.048, 0.34),
        materials["belly"],
        rig,
        "Body",
    )
    for side, x in (("L", -0.14), ("R", 0.14)):
        _uv(
            f"Eye.{side}",
            (x, -0.352, 1.08),
            (0.137, 0.045, 0.151),
            materials["eye"],
            rig,
            "Head",
        )
    _ico(
        "Pupil.L",
        (-0.125, -0.397, 1.075),
        (0.030, 0.015, 0.041),
        materials["pupil"],
        rig,
        "Head",
    )
    _ico(
        "Pupil.R",
        (0.154, -0.397, 1.09),
        (0.028, 0.015, 0.039),
        materials["pupil"],
        rig,
        "Head",
    )
    _ico(
        "Beak",
        (0.0, -0.425, 0.96),
        (0.095, 0.035, 0.055),
        materials["nose"],
        rig,
        "Head",
    )
    for side, x, tilt in (("L", -0.35, -0.26), ("R", 0.35, 0.26)):
        _uv(
            f"Flipper.{side}",
            (x, -0.005, 0.62),
            (0.105, 0.085, 0.31),
            materials["fur"],
            rig,
            f"Arm.{side}",
            (0.0, tilt, 0.0),
        )
        _ico(
            f"FlipperTint.{side}",
            (x + (-0.04 if side == "L" else 0.04), -0.020, 0.39),
            (0.085, 0.075, 0.085),
            materials["team"],
            rig,
            f"Arm.{side}",
        )
    for side, x in (("L", -0.17), ("R", 0.17)):
        _uv(
            f"Foot.{side}",
            (x, -0.09, 0.075),
            (0.16, 0.16, 0.06),
            materials["nose"],
            rig,
            f"Leg.{side}",
        )
    _ico(
        "BackTint",
        (0.0, 0.307, 0.65),
        (0.17, 0.035, 0.14),
        materials["team"],
        rig,
        "Body",
    )


def _build_capybara(
    rig: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    _build_common_face("capybara", rig, materials)
    _build_limbs(rig, materials)
    _ico(
        "CapyMuzzle",
        (0.0, -0.377, 0.94),
        (0.205, 0.072, 0.145),
        materials["muzzle"],
        rig,
        "Head",
    )
    _ico(
        "CapyNose",
        (0.0, -0.452, 0.975),
        (0.060, 0.024, 0.038),
        materials["nose"],
        rig,
        "Head",
    )
    for side, x in (("L", -0.25), ("R", 0.25)):
        _ico(
            f"Ear.{side}",
            (x, 0.0, 1.21),
            (0.095, 0.075, 0.095),
            materials["fur"],
            rig,
            f"Ear.{side}",
        )
        _ico(
            f"EarTint.{side}",
            (x, -0.070, 1.21),
            (0.044, 0.018, 0.044),
            materials["team"],
            rig,
            f"Ear.{side}",
        )
    _ico(
        "BackTint",
        (0.0, 0.332, 0.66),
        (0.19, 0.038, 0.15),
        materials["team"],
        rig,
        "Body",
    )


def _reset_pose(rig: bpy.types.Object) -> None:
    for pose_bone in rig.pose.bones:
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_euler = (0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)


def _key_pose(
    rig: bpy.types.Object,
    frame: int,
    transforms: dict[str, dict[str, tuple[float, float, float]]],
) -> None:
    # Evaluate the destination frame before assigning the authored pose.
    # Setting the frame afterwards makes Blender re-evaluate the active action
    # and overwrites the values that are about to be keyed; that silently
    # collapsed every stepped animation to its first pose.
    bpy.context.scene.frame_set(frame)
    _reset_pose(rig)
    for bone_name, values in transforms.items():
        pose_bone = rig.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        if "location" in values:
            pose_bone.location = values["location"]
        if "rotation" in values:
            pose_bone.rotation_euler = values["rotation"]
        if "scale" in values:
            pose_bone.scale = values["scale"]
    for pose_bone in rig.pose.bones:
        pose_bone.keyframe_insert("location", frame=frame, group=pose_bone.name)
        pose_bone.keyframe_insert("rotation_euler", frame=frame, group=pose_bone.name)
        pose_bone.keyframe_insert("scale", frame=frame, group=pose_bone.name)


def _action(
    rig: bpy.types.Object,
    name: str,
    poses: list[tuple[int, dict[str, dict[str, tuple[float, float, float]]]]],
) -> bpy.types.Action:
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    rig.animation_data_create()
    rig.animation_data.action = action
    for frame, pose in poses:
        _key_pose(rig, frame, pose)
    for layer in action.layers:
        for strip in layer.strips:
            for channel_bag in strip.channelbags:
                for fcurve in channel_bag.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        keyframe.interpolation = "CONSTANT"
    action["fps"] = FPS
    action["stepped"] = True
    action["loop"] = name in {"Idle", "Waddle", "Trapped"}
    return action


def _make_actions(rig: bpy.types.Object, species: str) -> list[bpy.types.Action]:
    tail = {
        "cat": 0.20,
        "dog": 0.15,
        "rabbit": 0.06,
        "bear": 0.08,
        "fox": 0.24,
        "raccoon": 0.22,
        "penguin": 0.04,
        "capybara": 0.025,
    }[species]
    ear = {
        "cat": 0.12,
        "dog": 0.18,
        "rabbit": 0.24,
        "bear": 0.05,
        "fox": 0.13,
        "raccoon": 0.09,
        "penguin": 0.0,
        "capybara": 0.045,
    }[species]
    actions = [
        _action(
            rig,
            "Idle",
            [
                (1, {}),
                (
                    5,
                    {
                        "Body": {"rotation": (0.0, 0.0, 0.025)},
                        "Head": {"rotation": (0.0, 0.018, -0.018)},
                        "Tail": {"rotation": (0.0, tail, 0.05)},
                    },
                ),
                (9, {}),
            ],
        ),
        _action(
            rig,
            "Waddle",
            [
                (
                    1,
                    {
                        # A planted left step: the whole body sits over the
                        # supporting foot while the opposite paw visibly lifts.
                        # The arms remain open like a toddler balancing instead
                        # of hanging beside the torso.
                        "Root": {"location": (-0.050, 0.0, 0.030)},
                        "Body": {"rotation": (0.0, 0.0, -0.14)},
                        "Head": {"rotation": (0.0, 0.0, 0.075)},
                        "Arm.L": {"rotation": (0.16, -0.08, -0.92)},
                        "Arm.R": {"rotation": (-0.10, 0.08, 0.82)},
                        "Leg.L": {
                            "location": (0.0, 0.018, -0.010),
                            "rotation": (-0.18, 0.0, -0.08),
                        },
                        "Leg.R": {
                            "location": (0.0, -0.055, 0.085),
                            "rotation": (0.42, 0.0, 0.10),
                        },
                        "Tail": {"rotation": (0.0, -tail, -0.08)},
                        "Ear.L": {"rotation": (ear, 0.0, 0.0)},
                    },
                ),
                (
                    3,
                    {
                        # Both feet briefly pass under the body.  A higher root
                        # pose stops world translation from reading as a slide.
                        "Root": {"location": (-0.018, 0.0, 0.075)},
                        "Body": {"rotation": (0.035, 0.0, -0.055)},
                        "Head": {"rotation": (-0.025, 0.0, 0.030)},
                        "Arm.L": {"rotation": (0.06, -0.05, -0.82)},
                        "Arm.R": {"rotation": (-0.02, 0.05, 0.90)},
                        "Leg.L": {
                            "location": (0.0, 0.0, 0.035),
                            "rotation": (-0.04, 0.0, -0.03),
                        },
                        "Leg.R": {
                            "location": (0.0, -0.025, 0.055),
                            "rotation": (0.18, 0.0, 0.04),
                        },
                    },
                ),
                (
                    5,
                    {
                        "Root": {"location": (0.050, 0.0, 0.030)},
                        "Body": {"rotation": (0.0, 0.0, 0.14)},
                        "Head": {"rotation": (0.0, 0.0, -0.075)},
                        "Arm.L": {"rotation": (-0.10, -0.08, -0.82)},
                        "Arm.R": {"rotation": (0.16, 0.08, 0.92)},
                        "Leg.L": {
                            "location": (0.0, -0.055, 0.085),
                            "rotation": (0.42, 0.0, -0.10),
                        },
                        "Leg.R": {
                            "location": (0.0, 0.018, -0.010),
                            "rotation": (-0.18, 0.0, 0.08),
                        },
                        "Tail": {"rotation": (0.0, tail, 0.08)},
                        "Ear.R": {"rotation": (ear, 0.0, 0.0)},
                    },
                ),
                (
                    7,
                    {
                        "Root": {"location": (0.018, 0.0, 0.075)},
                        "Body": {"rotation": (0.035, 0.0, 0.055)},
                        "Head": {"rotation": (-0.025, 0.0, -0.030)},
                        "Arm.L": {"rotation": (-0.02, -0.05, -0.90)},
                        "Arm.R": {"rotation": (0.06, 0.05, 0.82)},
                        "Leg.L": {
                            "location": (0.0, -0.025, 0.055),
                            "rotation": (0.18, 0.0, -0.04),
                        },
                        "Leg.R": {
                            "location": (0.0, 0.0, 0.035),
                            "rotation": (-0.04, 0.0, 0.03),
                        },
                    },
                ),
                (
                    9,
                    {
                        "Root": {"location": (-0.050, 0.0, 0.030)},
                        "Body": {"rotation": (0.0, 0.0, -0.14)},
                        "Head": {"rotation": (0.0, 0.0, 0.075)},
                        "Arm.L": {"rotation": (0.16, -0.08, -0.92)},
                        "Arm.R": {"rotation": (-0.10, 0.08, 0.82)},
                        "Leg.L": {
                            "location": (0.0, 0.018, -0.010),
                            "rotation": (-0.18, 0.0, -0.08),
                        },
                        "Leg.R": {
                            "location": (0.0, -0.055, 0.085),
                            "rotation": (0.42, 0.0, 0.10),
                        },
                        "Tail": {"rotation": (0.0, -tail, -0.08)},
                    },
                ),
            ],
        ),
        _action(
            rig,
            "PlaceBubble",
            [
                (1, {}),
                (
                    3,
                    {
                        "Root": {"location": (0.0, 0.0, -0.055)},
                        "Body": {"rotation": (0.16, 0.0, 0.0)},
                        "Head": {"rotation": (-0.10, 0.0, 0.0)},
                        "Arm.L": {"rotation": (-0.72, 0.0, -0.20)},
                        "Arm.R": {"rotation": (-0.72, 0.0, 0.20)},
                        "Leg.L": {"rotation": (0.18, 0.0, 0.0)},
                        "Leg.R": {"rotation": (0.18, 0.0, 0.0)},
                    },
                ),
                (
                    5,
                    {
                        "Root": {"location": (0.0, -0.035, -0.09)},
                        "Body": {"rotation": (0.24, 0.0, 0.0)},
                        "Head": {"rotation": (-0.14, 0.0, 0.0)},
                        "Arm.L": {"rotation": (-1.0, 0.0, -0.28)},
                        "Arm.R": {"rotation": (-1.0, 0.0, 0.28)},
                    },
                ),
                (8, {}),
            ],
        ),
        _action(
            rig,
            "Trapped",
            [
                (
                    1,
                    {
                        "Root": {"location": (0.0, 0.0, 0.05)},
                        "Arm.L": {"rotation": (-0.42, 0.0, -0.62)},
                        "Arm.R": {"rotation": (-0.42, 0.0, 0.62)},
                        "Leg.L": {"rotation": (0.28, 0.0, -0.22)},
                        "Leg.R": {"rotation": (-0.28, 0.0, 0.22)},
                    },
                ),
                (
                    5,
                    {
                        "Root": {"location": (0.0, 0.0, 0.10)},
                        "Body": {"rotation": (0.0, 0.0, 0.07)},
                        "Head": {"rotation": (0.0, 0.0, -0.05)},
                        "Arm.L": {"rotation": (-0.50, 0.0, -0.72)},
                        "Arm.R": {"rotation": (-0.35, 0.0, 0.54)},
                        "Leg.L": {"rotation": (0.35, 0.0, -0.25)},
                        "Leg.R": {"rotation": (-0.22, 0.0, 0.18)},
                        "Tail": {"rotation": (0.0, tail, 0.1)},
                    },
                ),
                (
                    9,
                    {
                        "Root": {"location": (0.0, 0.0, 0.05)},
                        "Arm.L": {"rotation": (-0.42, 0.0, -0.62)},
                        "Arm.R": {"rotation": (-0.42, 0.0, 0.62)},
                        "Leg.L": {"rotation": (0.28, 0.0, -0.22)},
                        "Leg.R": {"rotation": (-0.28, 0.0, 0.22)},
                    },
                ),
            ],
        ),
        _action(
            rig,
            "Defeat",
            [
                (1, {}),
                (
                    3,
                    {
                        "Root": {"location": (0.0, 0.0, 0.05)},
                        "Body": {"rotation": (0.0, 0.22, -0.35)},
                        "Arm.L": {"rotation": (0.0, 0.0, -0.35)},
                        "Arm.R": {"rotation": (0.0, 0.0, 0.62)},
                    },
                ),
                (
                    6,
                    {
                        "Root": {
                            "location": (0.08, 0.0, -0.20),
                            "rotation": (0.0, 1.30, -0.12),
                        },
                        "Body": {"rotation": (0.0, 0.18, -0.20)},
                        "Head": {"rotation": (0.0, -0.15, 0.16)},
                        "Arm.L": {"rotation": (0.0, 0.0, -0.55)},
                        "Arm.R": {"rotation": (0.0, 0.0, 0.48)},
                    },
                ),
                (
                    9,
                    {
                        "Root": {
                            "location": (0.10, 0.0, -0.23),
                            "rotation": (0.0, 1.48, -0.10),
                        },
                        "Head": {"rotation": (0.0, -0.12, 0.12)},
                        "Arm.L": {"rotation": (0.0, 0.0, -0.48)},
                        "Arm.R": {"rotation": (0.0, 0.0, 0.42)},
                    },
                ),
            ],
        ),
        _action(
            rig,
            "Victory",
            [
                (1, {}),
                (
                    3,
                    {
                        "Root": {"location": (0.0, 0.0, 0.13)},
                        "Body": {"rotation": (-0.08, 0.0, 0.0)},
                        "Arm.L": {"rotation": (0.0, 0.0, -1.0)},
                        "Arm.R": {"rotation": (0.0, 0.0, 1.0)},
                        "Leg.L": {"rotation": (-0.12, 0.0, 0.0)},
                        "Leg.R": {"rotation": (0.12, 0.0, 0.0)},
                        "Ear.L": {"rotation": (-ear, 0.0, 0.0)},
                        "Ear.R": {"rotation": (-ear, 0.0, 0.0)},
                    },
                ),
                (
                    5,
                    {
                        "Root": {"location": (0.0, 0.0, 0.20)},
                        "Body": {"rotation": (-0.12, 0.0, 0.0)},
                        "Head": {"rotation": (0.08, 0.0, 0.0)},
                        "Arm.L": {"rotation": (0.0, 0.0, -1.15)},
                        "Arm.R": {"rotation": (0.0, 0.0, 1.15)},
                        "Leg.L": {"rotation": (-0.18, 0.0, -0.06)},
                        "Leg.R": {"rotation": (0.18, 0.0, 0.06)},
                        "Tail": {"rotation": (0.0, tail, 0.18)},
                    },
                ),
                (
                    7,
                    {
                        "Root": {"location": (0.0, 0.0, 0.04)},
                        "Arm.L": {"rotation": (0.0, 0.0, -0.82)},
                        "Arm.R": {"rotation": (0.0, 0.0, 0.82)},
                    },
                ),
                (10, {}),
            ],
        ),
    ]
    rig.animation_data.action = actions[0]
    return actions


def _mesh_stats() -> tuple[int, int]:
    triangles = 0
    vertices = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.get("preview_only", False):
            continue
        vertices += len(obj.data.vertices)
        triangles += sum(max(1, len(poly.vertices) - 2) for poly in obj.data.polygons)
    return triangles, vertices


def _world_mesh_bounds() -> tuple[Vector, Vector]:
    minimum = Vector((math.inf, math.inf, math.inf))
    maximum = Vector((-math.inf, -math.inf, -math.inf))
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.get("preview_only", False):
            continue
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            minimum.x = min(minimum.x, world_corner.x)
            minimum.y = min(minimum.y, world_corner.y)
            minimum.z = min(minimum.z, world_corner.z)
            maximum.x = max(maximum.x, world_corner.x)
            maximum.y = max(maximum.y, world_corner.y)
            maximum.z = max(maximum.z, world_corner.z)
    return minimum, maximum


def _normalize_rig_height(rig: bpy.types.Object, target_height: float = 1.3) -> float:
    bpy.context.view_layer.update()
    minimum, maximum = _world_mesh_bounds()
    height = maximum.z - minimum.z
    if height <= 0.0001:
        return height
    uniform_scale = target_height / height
    rig.scale = (uniform_scale, uniform_scale, uniform_scale)
    bpy.context.view_layer.update()
    scaled_minimum, _scaled_maximum = _world_mesh_bounds()
    rig.location.z -= scaled_minimum.z
    bpy.context.view_layer.update()
    final_minimum, final_maximum = _world_mesh_bounds()
    return final_maximum.z - final_minimum.z


def _look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_preview_scene(species: str) -> None:
    bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.0, 0.0, -0.025))
    ground = bpy.context.object
    ground.name = "PreviewGround"
    ground["preview_only"] = True
    ground_material = _material("PreviewPaper", (0.80, 0.73, 0.61, 1.0))
    ground.data.materials.append(ground_material)

    bpy.ops.object.light_add(type="AREA", location=(-3.5, -4.5, 6.0))
    key = bpy.context.object
    key.name = "PreviewKey"
    key["preview_only"] = True
    key.data.energy = 120.0
    key.data.shape = "DISK"
    key.data.size = 5.0
    key.data.color = (1.0, 0.82, 0.62)
    _look_at(key, (0.0, 0.0, 0.65))

    bpy.ops.object.light_add(type="AREA", location=(3.0, -1.0, 3.0))
    fill = bpy.context.object
    fill.name = "PreviewFill"
    fill["preview_only"] = True
    fill.data.energy = 45.0
    fill.data.size = 4.0
    fill.data.color = (0.64, 0.78, 1.0)
    _look_at(fill, (0.0, 0.0, 0.7))

    bpy.ops.object.camera_add(location=(2.4, -5.6, 2.15))
    camera = bpy.context.object
    camera.name = "PreviewCamera"
    camera["preview_only"] = True
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 1.72
    _look_at(camera, (0.0, 0.0, 0.67))
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = str(PREVIEW_DIR / f"{species}-blender-preview.png")
    scene.world.color = (0.80, 0.73, 0.61)
    scene.view_settings.look = "Medium High Contrast"
    scene.render.image_settings.color_mode = "RGBA"
    bpy.context.scene.frame_set(1)
    bpy.ops.render.render(write_still=True)


def _remove_preview_objects() -> None:
    for obj in list(bpy.context.scene.objects):
        if obj.get("preview_only", False):
            bpy.data.objects.remove(obj, do_unlink=True)


def _export_glb(species: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if not obj.get("preview_only", False):
            obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(GLB_DIR / f"{species}.glb"),
        export_format="GLB",
        use_selection=True,
        export_materials="EXPORT",
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_merge_animation="ACTION",
        export_anim_slide_to_zero=True,
        export_optimize_animation_size=True,
        export_optimize_animation_keep_anim_armature=True,
        export_yup=True,
    )


def generate_character(species: str) -> dict[str, object]:
    if species not in SPECIES:
        raise ValueError(f"Unsupported species: {species}")
    BLEND_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    GLB_DIR.mkdir(parents=True, exist_ok=True)

    _reset_scene()
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = 10
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene["asset_pipeline"] = "Forest Bubble Paint / storybook-v2"
    scene["character_id"] = species
    if species in {"cat", "bear"}:
        reference_path = (
            PROJECT_ROOT / "art" / "reference" / "pilot" / f"{species}-turnaround.png"
        )
    else:
        reference_path = (
            PROJECT_ROOT
            / "art"
            / "reference"
            / "production"
            / f"{species}-modeling-reference-v1.png"
        )
    scene["reference_path"] = str(reference_path)

    materials = _make_materials(species)
    rig = _new_rig(species)
    builders = {
        "cat": _build_cat,
        "dog": _build_dog,
        "rabbit": _build_rabbit,
        "bear": _build_bear,
        "fox": _build_fox,
        "raccoon": _build_raccoon,
        "penguin": _build_penguin,
        "capybara": _build_capybara,
    }
    builders[species](rig, materials)
    _torus("TeamFootRing", materials["ring"], rig)
    generated_height = _normalize_rig_height(rig)
    actions = _make_actions(rig, species)
    triangles, vertices = _mesh_stats()
    rig["triangle_count"] = triangles
    rig["vertex_count"] = vertices
    rig["bone_count"] = len(rig.data.bones)
    rig["action_names"] = ",".join(action.name for action in actions)
    rig["generated_height"] = generated_height

    _add_preview_scene(species)
    _remove_preview_objects()
    blend_path = BLEND_DIR / f"{species}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    _export_glb(species)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    return {
        "species": species,
        "blend": str(blend_path),
        "glb": str(GLB_DIR / f"{species}.glb"),
        "preview": str(PREVIEW_DIR / f"{species}-blender-preview.png"),
        "triangles": triangles,
        "vertices": vertices,
        "bones": len(rig.data.bones),
        "actions": [action.name for action in actions],
        "height": generated_height,
        "height_target": 1.3,
    }


def generate_all_characters() -> list[dict[str, object]]:
    return [generate_character(species) for species in SPECIES]
