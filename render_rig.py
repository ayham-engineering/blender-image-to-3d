"""Cross-platform headless Blender render harness.

Renders a 6-view grid (front, back, left, right, top, three-quarter) of the
geometry produced by a user-supplied script, using a fixed EEVEE rig so
output is byte-comparable across machines/GPUs/OSes.

The rig (this file, run OUTSIDE Blender) never touches bpy directly for the
render itself. It writes a temp wrapper script that:
  1. clears the scene to an empty factory state
  2. exec()s the geometry script's source in a fresh namespace (so the
     geometry script cannot see or move the rig's camera/lights)
  3. builds camera + lights + engine + resolution, renders 6 views

Usage:
    from render_rig import render
    grid_path, error = render(Path("my_geometry_script.py"), Path("out_dir"))
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover - only needed on the host side
    Image = None


TILE_SIZE = 256
GRID_COLS = 3
GRID_ROWS = 2
GRID_SIZE = (TILE_SIZE * GRID_COLS, TILE_SIZE * GRID_ROWS)  # 768x512

VIEW_ORDER = ["front", "back", "left", "right", "top", "three_quarter"]

TRACEBACK_MARKER = "Traceback (most recent call last)"

# Lines containing any of these substrings are known-harmless noise and must
# never be mistaken for the start/body of a real traceback.
BENIGN_NOISE_SUBSTRINGS = ("OpenColorIO", "color_management")


# --------------------------------------------------------------------------
# Blender binary resolution
# --------------------------------------------------------------------------

def find_blender_binary() -> Path:
    """Resolve the Blender executable across Linux/Windows.

    Resolution order:
      1. BLENDER_BIN env var
      2. shutil.which("blender")
      3. (Windows) glob Program Files for Blender */blender.exe or
         blender-launcher.exe
    """
    env_bin = os.environ.get("BLENDER_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"BLENDER_BIN is set to '{env_bin}' but that path does not exist. "
            "Fix the BLENDER_BIN environment variable."
        )

    which_bin = shutil.which("blender")
    if which_bin:
        return Path(which_bin)

    if os.name == "nt" or sys.platform == "win32":
        patterns = [
            "C:/Program Files/Blender Foundation/Blender */blender.exe",
            "C:/Program Files/Blender Foundation/Blender */blender-launcher.exe",
        ]
        candidates = []
        for pattern in patterns:
            candidates.extend(glob.glob(pattern))
        if candidates:
            # Prefer the highest version directory (lexicographic works for
            # "Blender 4.2", "Blender 5.1" style directory names).
            candidates.sort()
            return Path(candidates[-1])

    raise FileNotFoundError(
        "Could not find the Blender executable. Set the BLENDER_BIN "
        "environment variable to the full path of the Blender binary "
        "(e.g. BLENDER_BIN=/usr/bin/blender or "
        "BLENDER_BIN=\"C:/Program Files/Blender Foundation/Blender 5.1/blender.exe\")."
    )


# --------------------------------------------------------------------------
# The wrapper script that runs INSIDE Blender
# --------------------------------------------------------------------------

_WRAPPER_TEMPLATE = '''
import bpy
import json
import math
import mathutils
import random
import traceback
import sys

GEOMETRY_SCRIPT_PATH = {geometry_script_path!r}
RESULT_PATH = {result_path!r}
OUT_DIR = {out_dir!r}

RESOLUTION = 256
SAMPLES = 64
SEED = 0

VIEW_ORDER = {view_order!r}


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def exec_geometry_script():
    with open(GEOMETRY_SCRIPT_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    namespace = {{"__name__": "__geometry__", "bpy": bpy}}
    exec(compile(source, GEOMETRY_SCRIPT_PATH, "exec"), namespace)


def mesh_world_bounds():
    """World-space bounding box (min, max) of all mesh objects."""
    minimum = mathutils.Vector((float("inf"),) * 3)
    maximum = mathutils.Vector((float("-inf"),) * 3)
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        found = True
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ mathutils.Vector(corner)
            minimum.x = min(minimum.x, world_corner.x)
            minimum.y = min(minimum.y, world_corner.y)
            minimum.z = min(minimum.z, world_corner.z)
            maximum.x = max(maximum.x, world_corner.x)
            maximum.y = max(maximum.y, world_corner.y)
            maximum.z = max(maximum.z, world_corner.z)
    if not found:
        # No mesh geometry: fall back to a unit cube around the origin so
        # the rig still produces a sane render instead of crashing.
        minimum = mathutils.Vector((-1, -1, -1))
        maximum = mathutils.Vector((1, 1, 1))
    return minimum, maximum


def setup_engine():
    scene = bpy.context.scene
    engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in engines else "BLENDER_EEVEE_NEXT"
    return scene.render.engine


def setup_render_settings(scene):
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # Samples: try EEVEE Next taa_render_samples, fall back to legacy EEVEE.
    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = SAMPLES
        if hasattr(eevee, "taa_samples"):
            eevee.taa_samples = SAMPLES
        if hasattr(eevee, "use_taa_reprojection"):
            eevee.use_taa_reprojection = False

    # Flat mid-gray world background, no HDRI.
    world = bpy.data.worlds.new("RigWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        bg.inputs[1].default_value = 1.0
    scene.world = world


def setup_lighting(target, radius):
    """Fixed 3-point lighting relative to the subject's bounding sphere."""
    key = bpy.data.lights.new("KeyLight", type="SUN")
    key.energy = 3.0
    key_obj = bpy.data.objects.new("KeyLight", key)
    key_obj.location = target + mathutils.Vector((radius * 2, -radius * 2, radius * 2.5))
    bpy.context.scene.collection.objects.link(key_obj)
    aim_at(key_obj, target)

    fill = bpy.data.lights.new("FillLight", type="SUN")
    fill.energy = 1.2
    fill_obj = bpy.data.objects.new("FillLight", fill)
    fill_obj.location = target + mathutils.Vector((-radius * 2.5, -radius * 1.5, radius * 1.2))
    bpy.context.scene.collection.objects.link(fill_obj)
    aim_at(fill_obj, target)

    rim = bpy.data.lights.new("RimLight", type="SUN")
    rim.energy = 1.8
    rim_obj = bpy.data.objects.new("RimLight", rim)
    rim_obj.location = target + mathutils.Vector((0, radius * 3, radius * 1.5))
    bpy.context.scene.collection.objects.link(rim_obj)
    aim_at(rim_obj, target)


def aim_at(obj, target_point):
    direction = target_point - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_camera():
    cam_data = bpy.data.cameras.new("RigCamera")
    cam_obj = bpy.data.objects.new("RigCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def position_camera(cam_obj, center, distance, azimuth_deg, elevation_deg):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = distance * math.cos(el) * math.sin(az)
    y = -distance * math.cos(el) * math.cos(az)
    z = distance * math.sin(el)
    cam_obj.location = center + mathutils.Vector((x, y, z))
    aim_at(cam_obj, center)


def render_view(scene, out_path):
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def setup_silhouette(scene):
    """Reconfigure the scene for a pure black-on-white silhouette pass.

    Every mesh gets a black Emission shader, so it renders flat black (0,0,0)
    regardless of lighting; the world background is set to pure white. This
    isolates SHAPE only — no shading, no color, no material — which is the
    signal the geometry coder can actually control.
    """
    # Pure white world background.
    world = bpy.data.worlds.new("SilhouetteWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = 1.0
    scene.world = world

    # Flat black emission material assigned to every mesh (overriding all slots).
    black_mat = bpy.data.materials.new("SilhouetteBlack")
    black_mat.use_nodes = True
    nt = black_mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    emission = nt.nodes.new("ShaderNodeEmission")
    emission.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    emission.inputs[1].default_value = 1.0
    output = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(emission.outputs[0], output.inputs[0])

    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(black_mat)


def main():
    random.seed(SEED)
    clear_scene()
    exec_geometry_script()

    engine_used = setup_engine()
    scene = bpy.context.scene
    setup_render_settings(scene)

    bbox_min, bbox_max = mesh_world_bounds()
    center = (bbox_min + bbox_max) / 2.0
    diagonal = (bbox_max - bbox_min).length
    radius = max(diagonal, 0.01)

    # Auto-frame: distance chosen so the subject fills a consistent
    # fraction of the frame regardless of absolute scale.
    fill_fraction = 0.6
    fov = math.radians(50.0)
    distance = (radius / 2.0) / math.tan(fov / 2.0) / fill_fraction

    setup_lighting(center, radius)

    cam_obj = make_camera()
    cam_obj.data.lens_unit = "FOV"
    cam_obj.data.angle = fov

    views = {{
        "front": (0, 0),
        "back": (180, 0),
        "left": (270, 0),
        "right": (90, 0),
        "top": (0, 89.9),
        "three_quarter": (45, 30),
    }}

    # Precompute camera placements once so both passes use identical framing.
    placements = {{}}
    for name in VIEW_ORDER:
        placements[name] = views[name]

    # Pass 1: shaded RGB render.
    rendered_paths = {{}}
    for name in VIEW_ORDER:
        azimuth, elevation = placements[name]
        position_camera(cam_obj, center, distance, azimuth, elevation)
        out_path = OUT_DIR + "/" + name + ".png"
        render_view(scene, out_path)
        rendered_paths[name] = out_path

    # Pass 2: black-on-white silhouette render, identical camera framing.
    setup_silhouette(scene)
    silhouette_paths = {{}}
    for name in VIEW_ORDER:
        azimuth, elevation = placements[name]
        position_camera(cam_obj, center, distance, azimuth, elevation)
        out_path = OUT_DIR + "/" + name + "_silhouette.png"
        render_view(scene, out_path)
        silhouette_paths[name] = out_path

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {{"engine": engine_used, "views": rendered_paths, "silhouettes": silhouette_paths}},
            f,
        )


try:
    main()
except Exception:
    traceback.print_exc(file=sys.stderr)
    raise
'''


def _write_wrapper(geometry_script_path: Path, out_dir: Path, result_path: Path) -> Path:
    wrapper_source = _WRAPPER_TEMPLATE.format(
        geometry_script_path=str(geometry_script_path.resolve()),
        out_dir=str(out_dir.resolve()),
        result_path=str(result_path.resolve()),
        view_order=VIEW_ORDER,
    )
    fd, wrapper_path_str = tempfile.mkstemp(suffix="_render_wrapper.py", text=True)
    wrapper_path = Path(wrapper_path_str)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(wrapper_source)
    return wrapper_path


# --------------------------------------------------------------------------
# stderr traceback scanning
# --------------------------------------------------------------------------

def _extract_traceback(stderr_text: str) -> Optional[str]:
    """Scan stderr for a Python traceback, ignoring known-benign noise lines."""
    lines = stderr_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if TRACEBACK_MARKER in line:
            start_idx = i
            break
    if start_idx is None:
        return None

    traceback_lines = [lines[start_idx]]
    for line in lines[start_idx + 1:]:
        if any(noise in line for noise in BENIGN_NOISE_SUBSTRINGS):
            continue
        traceback_lines.append(line)
        stripped = line.strip()
        # A traceback ends at the exception message line, which is not
        # indented and not one of the "File ..."/"  ..." body lines.
        if stripped and not stripped.startswith("File ") and not line.startswith((" ", "\t")):
            if not stripped.startswith("Traceback"):
                break
    return "\n".join(traceback_lines)


# --------------------------------------------------------------------------
# Grid assembly
# --------------------------------------------------------------------------

def _build_grid(view_paths: dict, grid_out_path: Path, bg_color=(128, 128, 128)) -> Path:
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required to build the render grid: pip install Pillow")

    grid = Image.new("RGB", GRID_SIZE, color=bg_color)
    for idx, name in enumerate(VIEW_ORDER):
        tile_path = Path(view_paths[name])
        tile = Image.open(tile_path).convert("RGB")
        if tile.size != (TILE_SIZE, TILE_SIZE):
            tile = tile.resize((TILE_SIZE, TILE_SIZE))
        col = idx % GRID_COLS
        row = idx // GRID_COLS
        grid.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))

    grid.save(grid_out_path)
    return grid_out_path


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def render(script_path: Path, out_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """Render 6 fixed views of the geometry in script_path, tiled into grid PNGs.

    Produces two grids in the identical 6-view 768x512 layout:
      - grid.png: shaded RGB render (mid-gray world, 3-point lighting)
      - silhouette_grid.png: pure black object on pure white background,
        isolating SHAPE only (no color/texture/lighting)

    Returns (grid_path, silhouette_path, None) on success, or
    (None, None, error_str) on failure. Never raises: all Blender/subprocess
    errors are captured and returned.
    """
    script_path = Path(script_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        blender_bin = find_blender_binary()
    except FileNotFoundError as e:
        return None, None, str(e)

    result_path = out_dir / "_render_result.json"
    if result_path.exists():
        result_path.unlink()

    try:
        wrapper_path = _write_wrapper(script_path, out_dir, result_path)
    except OSError as e:
        return None, None, f"Failed to write render wrapper script: {e}"

    try:
        proc = subprocess.run(
            [str(blender_bin), "-b", "-P", str(wrapper_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as e:
        return None, None, f"Blender render timed out after 600s.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
    except OSError as e:
        return None, None, f"Failed to launch Blender binary '{blender_bin}': {e}"
    finally:
        try:
            wrapper_path.unlink(missing_ok=True)
        except OSError:
            pass

    traceback_text = _extract_traceback(proc.stderr or "")
    if traceback_text is not None:
        return None, None, traceback_text

    if not result_path.exists():
        return None, None, (
            "Blender exited without producing a result file and no traceback "
            "was found in stderr. Raw stderr:\n" + (proc.stderr or "(empty)")
        )

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, None, f"Failed to read render result file: {e}"

    grid_path = out_dir / "grid.png"
    silhouette_path = out_dir / "silhouette_grid.png"
    try:
        _build_grid(result["views"], grid_path)
        # Silhouette grid uses a white gutter to match its black-on-white tiles.
        _build_grid(result["silhouettes"], silhouette_path, bg_color=(255, 255, 255))
    except Exception as e:
        return None, None, f"Failed to assemble render grid: {e}"

    return grid_path, silhouette_path, None


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

_TEST_CUBE_SCRIPT = '''
import bpy
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
'''


def _get_blender_version(blender_bin: Path) -> str:
    try:
        proc = subprocess.run(
            [str(blender_bin), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        first_line = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
        return first_line.strip() or "(unknown version)"
    except (subprocess.TimeoutExpired, OSError, IndexError):
        return "(unknown version)"


if __name__ == "__main__":
    scratch_dir = Path(tempfile.gettempdir()) / "render_rig_smoke_test"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    test_script_path = scratch_dir / "test_cube.py"
    test_script_path.write_text(_TEST_CUBE_SCRIPT, encoding="utf-8")

    try:
        blender_bin = find_blender_binary()
        print(f"Resolved Blender binary: {blender_bin}")
        print(f"Blender version: {_get_blender_version(blender_bin)}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    grid_path, silhouette_path, error = render(test_script_path, scratch_dir / "output")
    if error is not None:
        print("RENDER FAILED:", file=sys.stderr)
        print(error, file=sys.stderr)
        sys.exit(1)

    print(f"RGB grid rendered successfully:        {grid_path}")
    print(f"Silhouette grid rendered successfully: {silhouette_path}")
